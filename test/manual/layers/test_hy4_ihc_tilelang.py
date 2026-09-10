"""Offline wiring tests and opt-in HCU numerical verification for TileLang iHC."""

import ast
import os
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers import hy4_ihc_tilelang as ihc

with patch.dict("sys.modules", {
    "sglang.benchmark.serving": SimpleNamespace(run_benchmark=Mock()),
    "sglang.test.run_eval": SimpleNamespace(run_eval=Mock()),
}):
    from sglang.test.test_utils import CustomTestCase


def model_forward(tile):
    # Execute the actual method without loading unrelated model/attention backends.
    path = Path(ihc.__file__).parents[1] / "models/hunyuan_v4.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HYV4HCPreLayer")
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    scope = dict(torch=torch, Optional=Optional, envs=envs,
                 try_tilelang_ihc_pre=tile)
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(path), "exec"), scope)
    return scope["forward"]


def reference(x, w, scale, base, eps=1e-6, hc_eps=1e-5, magnitude=2.0):
    flat = x.flatten(1).float()
    gates = (flat @ w.T) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
    pre = torch.sigmoid(gates[:, :4] * scale[0] + base[:4]) + hc_eps
    post = magnitude * torch.sigmoid(gates[:, 4:] * scale[1] + base[4:]) + hc_eps
    return (pre.unsqueeze(-1) * x).sum(1).to(x.dtype), post


def model_method(class_name, method, **overrides):
    path = Path(ihc.__file__).parents[1] / "models/hunyuan_v4.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    scope = dict(torch=torch, Optional=Optional, RMSNorm=object, envs=envs,
                 try_tilelang_ihc_pre=ihc.try_tilelang_ihc_pre,
                 try_tilelang_ihc_post=ihc.try_tilelang_ihc_post,
                 try_tilelang_ihc_head=ihc.try_tilelang_ihc_head)
    scope.update(overrides)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[method]


def head_reference(x, w, scale, base, eps=1e-6, hc_eps=1e-5):
    flat = x.flatten(1).float()
    gates = (flat @ w.T) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
    pre = torch.sigmoid(gates * scale.reshape(()) + base) + hc_eps
    return (pre.unsqueeze(-1) * x.float()).sum(1).to(x.dtype)


class TestTileLangIHC(CustomTestCase):
    def setUp(self):
        super().setUp()
        ctx = patch.dict(os.environ, {"SGLANG_OPT_HY4_IHC_TILELANG": "1"})
        ctx.start()
        self.addCleanup(ctx.stop)
        ihc._get_kernels.cache_clear()
        self.addCleanup(ihc._get_kernels.cache_clear)
        ihc._warn.cache_clear()
        self.x = torch.zeros(2, 4, 64, dtype=torch.bfloat16)
        self.w = torch.zeros(8, 256)
        self.scale, self.base = torch.ones(2), torch.zeros(8)

    def call(self, x=None):
        return ihc.try_tilelang_ihc_pre(self.x if x is None else x, self.w, self.scale, self.base, 1e-6, 1e-5, 2.)

    def test_default_off(self):
        os.environ.pop("SGLANG_OPT_HY4_IHC_TILELANG")
        with patch.object(ihc, "_get_kernel") as get:
            self.assertIsNone(self.call())
            get.assert_not_called()

    def test_cpu_fallback(self):
        with patch.object(ihc, "_get_kernel") as get:
            self.assertIsNone(self.call())
            get.assert_not_called()

    def test_invalid_inputs(self):
        cases = [
            (self.x[:, :3], self.w, self.scale, self.base, "residual"),
            (self.x[:, :, :16], self.w, self.scale, self.base, "multiple"),
            (self.x, self.w[:4], self.scale, self.base, "weight"),
            (self.x, self.w, self.scale[:1], self.base, "scale"),
            (self.x, self.w, self.scale, self.base[:4], "base"),
            (self.x.float(), self.w, self.scale, self.base, "BF16"),
            (self.x, self.w.bfloat16(), self.scale, self.base, "FP32"),
            (self.x, self.w.to("meta"), self.scale, self.base, "device"),
            (self.x, self.w.requires_grad_(), self.scale, self.base, "inference"),
        ]
        for x, w, s, b, message in cases:
            with self.subTest(message=message):
                self.assertIn(message, ihc._unsupported(x, w, s, b))

    def test_missing_package(self):
        with patch.object(ihc.importlib, "import_module", side_effect=ImportError("missing")) as imp:
            for _ in range(2):
                for name in ("ihc_pre", "ihc_post", "ihc_head"):
                    self.assertIsNone(ihc._get_kernel(name))
            imp.assert_called_once_with("boltops.ihc")

    def test_missing_symbol(self):
        names = ("ihc_pre", "ihc_post", "ihc_head")
        for missing in names:
            for noncallable in (False, True):
                with self.subTest(missing=missing, noncallable=noncallable):
                    ihc._get_kernels.cache_clear()
                    funcs = {name: Mock() for name in names if name != missing}
                    module = SimpleNamespace(**funcs)
                    if noncallable:
                        setattr(module, missing, 123)
                    with patch.object(ihc.importlib, "import_module", return_value=module) as imp:
                        for _ in range(2):
                            for name in names:
                                self.assertIsNone(ihc._get_kernel(name))
                        imp.assert_called_once_with("boltops.ihc")
                    for fn in funcs.values():
                        fn.assert_not_called()

    def test_partial_backend_model_fallback(self):
        torch.manual_seed(43)
        x = torch.randn(2, 4, 64).bfloat16()
        w = torch.randn(8, 256) * .02
        linear = Mock(side_effect=lambda flat: (flat @ w.T, None))
        linear.weight = w
        owner = SimpleNamespace(hc_fn=linear, hc_scale=self.scale, hc_base=self.base,
                                hc_mult=4, rms_norm_eps=1e-6, hc_eps=1e-5, magnitude=2.)
        partial = SimpleNamespace(ihc_pre=Mock(), ihc_post=Mock())
        with patch.object(ihc.importlib, "import_module", return_value=partial), patch.object(ihc, "_unsupported", return_value=None):
            actual = model_forward(ihc.try_tilelang_ihc_pre)(owner, x)
        for got, expected in zip(actual, reference(x, w, self.scale, self.base)):
            torch.testing.assert_close(got, expected)
        partial.ihc_pre.assert_not_called()
        partial.ihc_post.assert_not_called()

    def test_empty_does_not_resolve_or_launch(self):
        with patch.object(ihc, "_unsupported", return_value=None), patch.object(ihc, "_get_kernel") as get:
            r, p = self.call(self.x[:0])
            self.assertEqual(r.shape, (0, 64))
            self.assertEqual(p.shape, (0, 4))
            get.assert_not_called()

    def test_forward_arguments_and_contiguity(self):
        kernel = Mock(return_value=(self.x[:, 0], torch.zeros(2, 4)))
        with patch.object(ihc, "_unsupported", return_value=None), patch.object(ihc, "_get_kernel", return_value=kernel), patch.object(torch.cuda, "device", return_value=nullcontext()):
            self.call(self.x.transpose(0, 1).contiguous().transpose(0, 1))
        args = kernel.call_args.args
        self.assertTrue(all(t.is_contiguous() for t in args[:4]))
        self.assertEqual(args[4:], (1e-6, 1e-5, 2.))

    def test_compile_error_propagates(self):
        with patch.object(ihc, "_unsupported", return_value=None), patch.object(ihc, "_get_kernel", return_value=Mock(side_effect=RuntimeError("compile"))), patch.object(torch.cuda, "device", return_value=nullcontext()):
            with self.assertRaisesRegex(RuntimeError, "compile"):
                self.call()

    def test_model_routing_and_optional_norm(self):
        torch.manual_seed(42)
        x = torch.randn(2, 4, 64).bfloat16()
        w = torch.randn(8, 256) * 0.02
        linear = Mock(side_effect=lambda flat: (flat @ w.T, None))
        linear.weight = w
        owner = SimpleNamespace(hc_fn=linear, hc_scale=self.scale, hc_base=self.base,
                                hc_mult=4, rms_norm_eps=1e-6, hc_eps=1e-5, magnitude=2.)
        expected = reference(x, w, self.scale, self.base)
        for fused in (None, expected):
            tile = Mock(return_value=fused)
            forward = model_forward(tile)
            got = forward(owner, x)
            for a, b in zip(got, expected):
                torch.testing.assert_close(a, b)
            norm = torch.linspace(.5, 1.5, 64)
            got_norm, post = forward(owner, x, norm, 1e-5)
            f = expected[0].float()
            ref_norm = (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-5) * norm).bfloat16()
            torch.testing.assert_close(got_norm, ref_norm)
            torch.testing.assert_close(post, expected[1])
        os.environ["SGLANG_OPT_HY4_IHC_TILELANG"] = "0"
        tile = Mock()
        got = model_forward(tile)(owner, x)
        tile.assert_not_called()
        torch.testing.assert_close(got[0], expected[0])

    def test_resolver_keeps_all_three_symbols(self):
        funcs = {name: Mock() for name in ("ihc_pre", "ihc_post", "ihc_head")}
        with patch.object(ihc.importlib, "import_module", return_value=SimpleNamespace(**funcs)) as imp:
            for _ in range(2):
                for name, fn in funcs.items():
                    self.assertIs(ihc._get_kernel(name), fn)
            imp.assert_called_once_with("boltops.ihc")

    def test_post_head_off_and_cpu_fallback(self):
        for enabled in ("0", "1"):
            os.environ["SGLANG_OPT_HY4_IHC_TILELANG"] = enabled
            with patch.object(ihc, "_get_kernel") as get:
                self.assertIsNone(ihc.try_tilelang_ihc_post(self.x[:, 0], self.x, torch.ones(2, 4)))
                self.assertIsNone(ihc.try_tilelang_ihc_head(self.x, self.w[:4], self.scale[:1], self.base[:4], 1e-6, 1e-5))
                get.assert_not_called()

    def test_post_model_route_fallback_and_no_mutation(self):
        x = torch.randn(2, 64).bfloat16()
        residual = torch.randn(2, 4, 64).bfloat16()
        post = torch.randn(2, 4)
        old = residual.clone()
        expected = (post.unsqueeze(-1) * x.float().unsqueeze(1) + residual.float()).bfloat16()
        for result in (None, expected):
            tile = Mock(return_value=result)
            fn = model_method("HYV4HCLayer", "post", try_tilelang_ihc_post=tile)
            torch.testing.assert_close(fn(None, x, residual, post), expected)
            tile.assert_called_once_with(x, residual, post)
            torch.testing.assert_close(residual, old)
        fn = model_method("HYV4HCLayer", "post", try_tilelang_ihc_post=Mock(side_effect=RuntimeError("post launch")))
        with self.assertRaisesRegex(RuntimeError, "post launch"):
            fn(None, x, residual, post)

    def test_head_model_route_norm_and_fallback(self):
        x = torch.randn(2, 4, 64).bfloat16()
        w = torch.randn(4, 256) / 16
        scale, base = torch.tensor([0.7]), torch.linspace(-.3, .3, 4)
        linear = Mock(side_effect=lambda flat: (flat @ w.T, None))
        linear.weight = w
        owner = SimpleNamespace(hc_head_fn=linear, hc_head_scale=scale, hc_head_base=base,
                                config=SimpleNamespace(rms_norm_eps=1e-6, hc_eps=1e-5))
        expected = head_reference(x, w, scale, base)
        for result in (None, expected):
            fn = model_method("HYV4HCHeadLayer", "forward", try_tilelang_ihc_head=Mock(return_value=result))
            torch.testing.assert_close(fn(owner, x), expected)
            norm = Mock(side_effect=lambda t: t * 2)
            torch.testing.assert_close(fn(owner, x, norm), expected * 2)
            norm.assert_called_once()
        fn = model_method("HYV4HCHeadLayer", "forward", try_tilelang_ihc_head=Mock(side_effect=RuntimeError("head launch")))
        with self.assertRaisesRegex(RuntimeError, "head launch"):
            fn(owner, x)

    @unittest.skipUnless(os.environ.get("SGLANG_TEST_HY4_IHC_TILELANG") == "1", "requires idle HCU opt-in")
    def test_real_hcu_full_pipeline(self):
        self.assertIsNotNone(torch.version.hip)
        gen = torch.Generator().manual_seed(20260909)
        cases = [(d, t, False) for d in (64, 6144) for t in (1, 63, 64, 65, 129)]
        # Match the supplied BoltOps diff's small-weight fixtures as well.
        cases += [(d, t, True) for d in (4096, 7168) for t in (1, 32)]
        with torch.inference_mode():
            for d, tokens, diff_fixture in cases:
                with self.subTest(d=d, tokens=tokens, diff_fixture=diff_fixture):
                    factor = 1e-4 if diff_fixture else (4*d)**-0.5
                    residual = torch.randn(tokens, 4, d, generator=gen).bfloat16()
                    w = torch.randn(8, 4*d, generator=gen) * factor
                    hw = torch.randn(4, 4*d, generator=gen) * factor
                    scale = torch.tensor([.01, .01] if diff_fixture else [.7, 1.2])
                    base = torch.zeros(8) if diff_fixture else torch.linspace(-.3, .3, 8)
                    hs = torch.tensor([.01 if diff_fixture else .7])
                    hb = torch.zeros(4) if diff_fixture else torch.linspace(-.3, .3, 4)
                    eps, hc_eps = (1e-5, 1e-6) if diff_fixture else (1e-6, 1e-5)
                    r = residual.cuda()
                    rc = r.clone()
                    reduced, gates = ihc.try_tilelang_ihc_pre(r, w.cuda(), scale.cuda(), base.cuda(), eps, hc_eps, 2.)
                    reduced_ref, gates_ref = reference(residual, w, scale, base, eps, hc_eps)
                    # Independent post oracle uses the same exact inputs as the kernel.
                    y = torch.randn(tokens, d, generator=gen).bfloat16().cuda()
                    out = ihc.try_tilelang_ihc_post(y, r, gates)
                    self.assertIsNotNone(out)
                    out_ref = (gates.cpu().unsqueeze(-1) * y.cpu().float().unsqueeze(1) + residual.float()).bfloat16()
                    head = ihc.try_tilelang_ihc_head(out, hw.cuda(), hs.cuda(), hb.cuda(), eps, hc_eps)
                    self.assertIsNotNone(head)
                    head_ref = head_reference(out.cpu(), hw, hs, hb, eps, hc_eps)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(r, rc, rtol=0, atol=0)
                    checks = [
                        ("pre_reduced", reduced, reduced_ref, 1e-2, 2e-3 if diff_fixture else 1e-2),
                        ("pre_post_gate", gates, gates_ref, 1e-5 if diff_fixture else 2e-3, 1e-6 if diff_fixture else 2e-3),
                        ("post_residual", out, out_ref, 1e-2, 1e-2),
                        ("head", head, head_ref, 1e-2, 2e-3 if diff_fixture else 1e-2),
                    ]
                    for name, actual, expected, rtol, atol in checks:
                        delta = (actual.cpu().float() - expected.float()).abs()
                        print(f"FULL T={tokens} D={d} diff_fixture={diff_fixture} {name} max_abs={delta.max().item():.9g} mean_abs={delta.mean().item():.9g} rtol={rtol} atol={atol}", flush=True)
                        torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)
            # Zero-token DP ranks must not compile or launch either new API.
            x = torch.empty(0, 4, 64, device="cuda", dtype=torch.bfloat16)
            with patch.object(ihc, "_get_kernel", side_effect=AssertionError("empty launch")):
                self.assertEqual(ihc.try_tilelang_ihc_post(x[:, 0], x, torch.empty(0, 4, device="cuda")).shape, x.shape)
                self.assertEqual(ihc.try_tilelang_ihc_head(x, torch.zeros(4, 256, device="cuda"), torch.ones(1, device="cuda"), torch.zeros(4, device="cuda"), 1e-6, 1e-5).shape, (0, 64))

    @unittest.skipUnless(os.environ.get("SGLANG_TEST_HY4_IHC_TILELANG") == "1", "requires idle HCU opt-in")
    def test_real_hcu(self):
        self.assertIsNotNone(torch.version.hip)
        self.assertTrue(torch.cuda.is_available())
        # A fixed CPU RNG provides identical inputs across container versions.
        gen = torch.Generator().manual_seed(20260908)
        with torch.inference_mode():
            for d in (64, 6144):
                for tokens in (1, 63, 64, 65, 129):
                    with self.subTest(d=d, tokens=tokens):
                        x = torch.randn(tokens, 4, d, generator=gen).bfloat16()
                        w = torch.randn(8, 4*d, generator=gen) / (4*d)**0.5
                        scale = torch.tensor([0.7, 1.2])
                        base = torch.linspace(-0.3, 0.3, 8)
                        ref = reference(x, w, scale, base)
                        got = ihc.try_tilelang_ihc_pre(x.cuda(), w.cuda(), scale.cuda(), base.cuda(), 1e-6, 1e-5, 2.)
                        self.assertIsNotNone(got, "unexpected eager fallback")
                        torch.cuda.synchronize()
                        for name, actual, expected, tol in zip(("reduced", "post"), got, ref, (1e-2, 2e-3)):
                            actual = actual.cpu()
                            delta = (actual.float() - expected.float()).abs()
                            print(f"T={tokens} D={d} {name} max_abs={delta.max().item():.9g} mean_abs={delta.mean().item():.9g} rtol=atol={tol}", flush=True)
                            # BF16 reduced has final storage rounding; post stays FP32.
                            torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)


if __name__ == "__main__":
    unittest.main()
