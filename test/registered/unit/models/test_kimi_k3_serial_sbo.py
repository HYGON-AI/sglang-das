"""CPU contracts for K3 serial SBO, using real method bodies and CPU operators.

AST loading avoids importing vendor modules that initialize a GPU at import.
These tests validate control flow and math composition, not native DeepEP or
the Triton kernel. Run the separate HCU test for kernel/graph correctness.
"""

import __future__

import ast
import contextlib
import os
import sys
import unittest
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


REPO = next(
    p for p in Path(__file__).resolve().parents if (p / "python/sglang").is_dir()
)
SRT = REPO / "python/sglang/srt"


def load_nodes(path, names, scope):
    nodes = []
    for n in ast.parse(path.read_text()).body:
        if isinstance(n, ast.ClassDef) and n.name in names:
            allowed = names[n.name]
            if allowed is not None:
                n.body = [
                    m
                    for m in n.body
                    if isinstance(m, ast.FunctionDef) and m.name in allowed
                ]
                n.bases = [
                    ast.Attribute(
                        value=ast.Name(id="nn", ctx=ast.Load()),
                        attr="Module",
                        ctx=ast.Load(),
                    )
                ]
            elif n.name == "DeepEPDispatcher":
                n.body = [m for m in n.body if getattr(m, "name", None) != "__init__"]
            for m in n.body:
                if isinstance(m, ast.FunctionDef):
                    m.decorator_list = []  # Test method logic without Dynamo wrappers.
            nodes.append(n)
    tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(
        compile(tree, str(path), "exec", flags=__future__.annotations.compiler_flag),
        scope,
    )


def load_sources():
    dispatch_module = ModuleType("serial_sbo_cpu_dispatch")
    sys.modules[dispatch_module.__name__] = dispatch_module
    d = dispatch_module.__dict__
    d.update(nn=nn, dataclass=dataclass)
    base = SRT / "layers/moe/token_dispatcher/base.py"
    exec(compile(base.read_text(), str(base), "exec"), d)
    d.update(
        auto=auto, DeepEPMode=Enum("DeepEPMode", ["AUTO", "NORMAL", "LOW_LATENCY"])
    )
    load_nodes(
        SRT / "layers/moe/token_dispatcher/deepep.py",
        {
            "DeepEPPDispatchHooks": None,
            "_DeepEPCombineHooks": None,
            "_Stage": None,
            "DeepEPDispatcher": None,
        },
        d,
    )
    load_nodes(
        SRT / "batch_overlap/two_batch_overlap.py",
        {"MaybeTboDeepEPDispatcher": None},
        d,
    )
    k = dict(
        nn=nn,
        torch=torch,
        os=os,
        TYPE_CHECKING=False,
        _is_npu=False,
        _add3=lambda a, b, c: a + b if c is None else (a + b) + c,
        ExpertLocationDispatchInfo=SimpleNamespace(init_new=lambda **kw: None),
    )
    load_nodes(
        SRT / "models/kimi_k3.py",
        {
            "KimiK3MoE": {
                "_init_serial_sbo",
                "_serial_sbo_experts",
                "_forward_unfused",
                "_reduce_latent",
                "_forward_shared_experts",
            },
            "KimiK3MLP": {"forward", "_serial_sbo_gate_up", "_serial_sbo_down"},
        },
        k,
    )
    return d, k


D, K = load_sources()


class FakeImpl:
    return_recv_hook = True

    def __init__(self, trace):
        self.trace = trace
        self.fail = None

    def dispatch_a(self, hidden_states, topk_output):
        self.trace.append("dispatch_send")
        return (hidden_states,)

    def dispatch_b(self, x):
        self.trace.append("dispatch_recv")
        return x

    def combine_a(self, hidden_states, topk_ids, topk_weights):
        self.trace.append("combine_send")
        if self.fail == "combine":
            raise RuntimeError("injected combine")
        return (hidden_states,)

    def combine_b(self, x):
        self.trace.append("combine_recv")
        return x


def dispatcher(trace):
    inner = D["DeepEPDispatcher"].__new__(D["DeepEPDispatcher"])
    D["BaseDispatcher"].__init__(inner)
    inner.deepep_mode = D["DeepEPMode"].LOW_LATENCY
    inner._stage = D["_Stage"].INITIAL
    inner._deepep_dispatch_hooks = D["DeepEPPDispatchHooks"]()
    inner._deepep_combine_hooks = None
    inner._low_latency_dispatcher = FakeImpl(trace)
    inner._get_impl = lambda: inner._low_latency_dispatcher
    wrapper = D["MaybeTboDeepEPDispatcher"].__new__(D["MaybeTboDeepEPDispatcher"])
    D["BaseDispatcher"].__init__(wrapper)
    wrapper._inners = [inner]
    return wrapper, inner


class Experts(nn.Module):
    def __init__(self, trace):
        super().__init__()
        self.trace = trace
        self.dispatcher, self.inner = dispatcher(trace)
        self.fail = None

    def forward(self, x, topk):
        if self.fail == "before":
            raise RuntimeError("injected before")
        x = self.dispatcher.dispatch(hidden_states=x, topk_output=topk)
        self.trace.append("routed")
        if self.fail == "after_dispatch":
            raise RuntimeError("injected after dispatch")
        x = torch.sin(x)
        return self.dispatcher.combine(
            combine_input=SimpleNamespace(
                hidden_states=x, topk_ids=None, topk_weights=None
            )
        )


class Linear(nn.Module):
    def __init__(self, weight, trace, name):
        super().__init__()
        self.weight, self.trace, self.name = weight, trace, name

    def forward(self, x, **kwargs):
        self.trace.append(self.name)
        return x @ self.weight.to(x.dtype), None


class Shared(nn.Module):
    def __init__(self, trace):
        super().__init__()
        self.trace = trace
        self.fail = None

    def _serial_sbo_gate_up(self, x, quant=None):
        self.trace.append("shared_up")
        if self.fail == "up":
            raise RuntimeError("injected shared up")
        return x * 2, None

    def _serial_sbo_down(self, gate_up):
        self.trace.append("shared_down")
        if self.fail == "down":
            raise RuntimeError("injected shared down")
        return torch.tanh(gate_up) * 0.25

    def forward(self, x, **kwargs):
        return self._serial_sbo_down(self._serial_sbo_gate_up(x)[0])


def model(enabled):
    trace = []
    m = K["KimiK3MoE"]()
    m._serial_sbo_enabled = enabled
    m._sbo_shared_overlap = False
    m._shared_experts_tp1 = True
    m._shared_experts_tp_comm = False
    m.shared_experts = Shared(trace)
    m.alt_stream = None
    m._can_overlap_shared_experts_npu = lambda x: False
    m._ep_front = lambda x: None
    m._ep_front_overlap = lambda x: None
    m.gate = lambda x: x
    m.topk = lambda *a, **kw: None
    m.layer_idx = 1
    m._reuse_input_int8_enabled = False
    m.use_latent_moe = True
    m.moe_hidden_size = 3
    m._use_mega_moe = False
    m._routed_needs_reduce = False
    m.tp_size = 32
    m.routed_expert_down_proj = Linear(
        torch.arange(24).float().reshape(8, 3) / 64, trace, "latent_down"
    )
    m.experts = Experts(trace)
    up = torch.arange(24).float().reshape(3, 8) / 64

    def latent_up(x):
        trace.append("latent_up")
        return x @ up.to(x.dtype)

    m._forward_latent_up = latent_up
    return m, trace


def assert_clean(test, m):
    inner = m.experts.inner
    test.assertFalse(inner._deepep_dispatch_hooks.hook_dict)
    test.assertTrue(
        inner._deepep_combine_hooks is None or not inner._deepep_combine_hooks.hook_dict
    )


class TestSerialSbo(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for api in ["current_stream", "stream", "Event", "Stream", "_lazy_init"]:
            self.stack.enter_context(
                patch.object(
                    torch.cuda,
                    api,
                    side_effect=AssertionError("Unexpected GPU API: " + api),
                )
            )

    def test_outputs_and_order_with_changing_batches(self):
        torch.manual_seed(23)
        for dtype in [torch.float32, torch.bfloat16]:
            for prefix in [False, True]:
                off, off_trace = model(False)
                on, trace = model(True)
                for n in [0, 1, 4, 8, 16, 17, 32, 80, 128, 1, 0, 16]:
                    with self.subTest(dtype=dtype, prefix=prefix, tokens=n):
                        x = torch.randn(n, 8).to(dtype)
                        residual = torch.randn_like(x) if prefix else None
                        trace.clear()
                        a = off._forward_unfused(x, prefix_sum=residual)
                        b = on._forward_unfused(x, prefix_sum=residual)
                        self.assertTrue(torch.equal(a, b))
                        assert_clean(self, on)
                        expected = (["latent_down"] if n else []) + ["dispatch_send"]
                        if n:
                            expected += ["shared_up"]
                        expected += ["dispatch_recv", "routed", "combine_send"]
                        if n:
                            expected += ["shared_down"]
                        expected += ["combine_recv"]
                        if n:
                            expected += ["latent_up"]
                        self.assertEqual(trace, expected)

    def test_existing_outer_hooks_are_preserved(self):
        m, trace = model(True)
        outer = m.experts.dispatcher
        handles = [
            outer.register_pre_dispatch_hook(lambda *a: trace.append("pre_dispatch")),
            outer.register_post_dispatch_hook(lambda *a: trace.append("post_dispatch")),
            outer.register_pre_combine_hook(lambda *a: trace.append("pre_combine")),
            outer.register_post_combine_hook(lambda *a: trace.append("post_combine")),
        ]
        for _ in range(2):
            m._forward_unfused(torch.ones(4, 8), prefix_sum=None)
            assert_clean(self, m)
        for name in ["pre_dispatch", "post_dispatch", "pre_combine", "post_combine"]:
            self.assertEqual(trace.count(name), 2)
        for h in handles:
            h.remove()

    def test_hook_cleanup_on_exceptions(self):
        for location in [
            "before",
            "after_dispatch",
            "up",
            "down",
            "combine",
            "register_dispatch",
            "register_combine",
        ]:
            with self.subTest(location=location):
                m, trace = model(True)

                def fail(*args, **kwargs):
                    raise RuntimeError("injected registration")

                if location.startswith("register_"):
                    setattr(
                        m.experts.dispatcher,
                        "register_deepep_" + location[9:] + "_hook",
                        fail,
                    )
                elif location in ["up", "down"]:
                    m.shared_experts.fail = location
                elif location == "combine":
                    m.experts.inner._low_latency_dispatcher.fail = location
                else:
                    m.experts.fail = location
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    m._forward_unfused(torch.ones(4, 8), prefix_sum=None)
                assert_clean(self, m)

    def test_hook_registration_rolls_back(self):
        for kind in ["dispatch", "combine"]:
            with self.subTest(kind=kind):
                wrapper, inner = dispatcher([])
                method = "register_deepep_" + kind + "_hook"
                bad = SimpleNamespace(
                    **{
                        method: lambda h: (_ for _ in ()).throw(
                            ValueError("unsupported")
                        )
                    }
                )
                wrapper._inners.append(bad)
                with self.assertRaises(ValueError):
                    getattr(wrapper, method)(lambda d: None)
                self.assertFalse(getattr(inner, "_deepep_" + kind + "_hooks").hook_dict)

    def test_combine_window_rejects_wrong_modes(self):
        for mode in [D["DeepEPMode"].NORMAL, D["DeepEPMode"].AUTO]:
            _, inner = dispatcher([])
            inner.deepep_mode = mode
            with self.assertRaises(ValueError):
                inner.register_deepep_combine_hook(lambda d: None)
        _, inner = dispatcher([])
        inner._low_latency_dispatcher.return_recv_hook = False
        with self.assertRaises(ValueError):
            inner.register_deepep_combine_hook(lambda d: None)

    def test_missing_or_duplicate_callbacks_fail_and_cleanup(self):
        for failure in ["missing", "duplicate_up", "duplicate_down", "wrong_order"]:
            m, _ = model(True)

            def run(x, topk):
                inner = m.experts.inner
                if failure == "missing":
                    return x
                if failure == "wrong_order":
                    inner._deepep_combine_hooks(inner)
                inner._deepep_dispatch_hooks(inner)
                if failure == "duplicate_up":
                    inner._deepep_dispatch_hooks(inner)
                inner._deepep_combine_hooks(inner)
                if failure == "duplicate_down":
                    inner._deepep_combine_hooks(inner)
                return x

            m.experts.forward = run
            with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                m._forward_unfused(torch.ones(4, 8), prefix_sum=None)
            assert_clean(self, m)

    def test_selector_and_incompatible_configurations(self):
        utils = ModuleType("sglang.srt.layers.moe.utils")
        utils.get_deepep_mode = lambda: SimpleNamespace(value="low_latency")
        utils.is_sbo_enabled = lambda: False
        utils.is_tbo_enabled = lambda: False
        with (
            patch.dict(sys.modules, {utils.__name__: utils}),
            patch.dict(
                K,
                {
                    "_is_hcu": True,
                    "get_moe_a2a_backend": lambda: SimpleNamespace(
                        is_deepep=lambda: True
                    ),
                },
            ),
        ):
            for value in [None, "off", "serial", "overlap", "", "typo"]:
                with patch.dict(os.environ, {}, clear=True):
                    if value is not None:
                        os.environ["SGLANG_K3_SBO_METHOD"] = value
                    m, _ = model(False)
                    if value in [None, "off", "serial"]:
                        self.assertEqual(m._init_serial_sbo(), value == "serial")
                    else:
                        with self.assertRaises(ValueError):
                            m._init_serial_sbo()
            for name, value in [
                ("use_latent_moe", False),
                ("_shared_experts_tp1", False),
                ("_shared_experts_tp_comm", True),
                ("shared_experts", None),
                ("alt_stream", object()),
            ]:
                m, _ = model(False)
                setattr(m, name, value)
                with patch.dict(os.environ, {"SGLANG_K3_SBO_METHOD": "serial"}):
                    with self.assertRaises(ValueError):
                        m._init_serial_sbo()
                with patch.dict(os.environ, {"SGLANG_K3_SBO_METHOD": "off"}):
                    self.assertFalse(m._init_serial_sbo())
            for name in ["is_sbo_enabled", "is_tbo_enabled"]:
                with (
                    patch.object(utils, name, return_value=True),
                    patch.dict(os.environ, {"SGLANG_K3_SBO_METHOD": "serial"}),
                ):
                    with self.assertRaises(ValueError):
                        model(False)[0]._init_serial_sbo()
            for value in ["normal", "auto"]:
                with (
                    patch.object(
                        utils,
                        "get_deepep_mode",
                        return_value=SimpleNamespace(value=value),
                    ),
                    patch.dict(os.environ, {"SGLANG_K3_SBO_METHOD": "serial"}),
                ):
                    with self.assertRaises(ValueError):
                        model(False)[0]._init_serial_sbo()
            for name, func in [
                ("_is_hcu", False),
                (
                    "get_moe_a2a_backend",
                    lambda: SimpleNamespace(is_deepep=lambda: False),
                ),
            ]:
                with (
                    patch.dict(K, {name: func}),
                    patch.dict(os.environ, {"SGLANG_K3_SBO_METHOD": "serial"}),
                ):
                    with self.assertRaises(ValueError):
                        model(False)[0]._init_serial_sbo()

    def test_shared_tail_preserves_fused_and_unfused_quantization_boundary(self):
        helper = ModuleType("sglang.srt.layers.k3_sbo_gemm")
        helper.project = lambda linear, x, input_quant_args=None: linear(
            x, input_quant_args=input_quant_args
        )
        with (
            patch.dict(sys.modules, {helper.__name__: helper}),
            patch.dict(K, {"_lightop_int8_input_supported": lambda x: True}),
        ):
            for fused in [False, True]:
                m = K["KimiK3MLP"]()
                m._dp_attention = False
                m._dense_attn_tp = False
                m.gate_up_proj = lambda x, **kw: (torch.cat([x, x * 2], dim=-1), None)

                class Act:
                    beta, linear_beta = 4.0, 25.0

                    def __call__(self, x):
                        a, b = x.chunk(2, dim=-1)
                        return torch.tanh(a) * b

                m.act_fn = Act()

                def quant(x, beta, linear_beta):
                    y = m.act_fn(x)
                    return y.round().to(torch.int8), torch.ones((len(y), 1))

                m._fused_situ_int8_op = quant if fused else None

                def down(x, input_quant_args=None):
                    if input_quant_args is not None:
                        q, scale = input_quant_args
                        x = q.float() * scale
                    return x * 0.125, None

                m.down_proj = down
                for n in [1, 16, 17]:
                    x = torch.randn(n, 8)
                    expected = m(x)
                    gate_up = m._serial_sbo_gate_up(x)[0]
                    self.assertTrue(torch.equal(expected, m._serial_sbo_down(gate_up)))


class TestProjectionSelection(unittest.TestCase):
    """Metadata and fallback contracts; no Triton/GPU execution here."""

    def setUp(self):
        p = SRT / "layers/k3_sbo_gemm.py"
        nodes = [
            n
            for n in ast.parse(p.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name in ["_eligible", "project"]
        ]
        self.scope = dict(torch=torch, _supported_device=lambda index: True)
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), str(p), "exec"), self.scope
        )
        self.method = type("SlimQuantW4A8Int8LinearMethod", (), {"w8a8_strategy": 3})
        module = ModuleType("sglang.srt.layers.quantization.slimquant_w4a8")
        module.SlimQuantW4A8Int8LinearMethod = self.method
        patcher = patch.dict(sys.modules, {module.__name__: module})
        patcher.start()
        self.addCleanup(patcher.stop)

    def metadata(self):
        device = torch.device("cuda", 0)
        x = SimpleNamespace(
            is_cuda=True,
            dtype=torch.bfloat16,
            ndim=2,
            shape=(8, 7168),
            requires_grad=False,
            device=device,
            stride=lambda i: 7168 if i == 0 else 1,
        )
        weight = SimpleNamespace(
            dtype=torch.int8,
            shape=(12288, 7168),
            device=device,
            is_contiguous=lambda: True,
        )
        scale = SimpleNamespace(
            dtype=torch.float32,
            device=device,
            requires_grad=False,
            numel=lambda: 12288,
            is_contiguous=lambda: True,
        )
        linear = SimpleNamespace(
            tp_size=1,
            bias=None,
            gather_output=False,
            quant_method=self.method(),
            weight=weight,
            weight_scale=scale,
        )
        return linear, x

    def test_supported_metadata_and_rejections(self):
        eligible = self.scope["_eligible"]
        self.assertTrue(eligible(*self.metadata()))
        for field, value in [
            ("tp_size", 2),
            ("bias", object()),
            ("gather_output", True),
            ("quant_method", SimpleNamespace(w8a8_strategy=3)),
        ]:
            linear, x = self.metadata()
            setattr(linear, field, value)
            self.assertFalse(eligible(linear, x), field)
        for field, value in [
            ("is_cuda", False),
            ("dtype", torch.float16),
            ("ndim", 3),
            ("shape", (0, 7168)),
            ("shape", (17, 7168)),
            ("shape", (8, 8192)),
            ("requires_grad", True),
            ("stride", lambda i: 2),
        ]:
            linear, x = self.metadata()
            setattr(x, field, value)
            self.assertFalse(eligible(linear, x), field)
        for owner, field, value in [
            ("weight", "device", torch.device("cuda", 1)),
            ("weight", "shape", (8192, 7168)),
            ("weight", "dtype", torch.bfloat16),
            ("weight", "is_contiguous", lambda: False),
            ("weight_scale", "device", torch.device("cuda", 1)),
            ("weight_scale", "dtype", torch.bfloat16),
            ("weight_scale", "is_contiguous", lambda: False),
            ("weight_scale", "requires_grad", True),
            ("weight_scale", "numel", lambda: 1),
        ]:
            linear, x = self.metadata()
            setattr(getattr(linear, owner), field, value)
            self.assertFalse(eligible(linear, x), (owner, field))
        linear, x = self.metadata()
        linear.quant_method.w8a8_strategy = 1
        self.assertFalse(eligible(linear, x))
        with patch.dict(self.scope, {"_supported_device": lambda index: False}):
            self.assertFalse(eligible(*self.metadata()))

    def test_native_fallback_preserves_arguments(self):
        calls = []
        marker = object()

        class Native:
            def __call__(self, *args, **kwargs):
                calls.append((args, kwargs))
                return marker

        for n in [0, 1, 16, 17, 128]:
            x = torch.zeros(n, 8, dtype=torch.bfloat16)
            for quant in [
                None,
                (torch.zeros(n, 8, dtype=torch.int8), torch.ones(n, 1)),
                (),
            ]:
                self.assertIs(self.scope["project"](Native(), x, quant), marker)
                args, kwargs = calls[-1]
                self.assertIs(args[0], x)
                self.assertEqual(
                    set(kwargs), set() if quant is None else {"input_quant_args"}
                )
                if quant is not None:
                    self.assertIs(kwargs["input_quant_args"], quant)


if __name__ == "__main__":
    unittest.main()
