# SPDX-License-Identifier: Apache-2.0
"""CPU scope/loader tests and opt-in GPU parity tests for MiniMax-H3 INT8."""

import ast
import importlib.util
import os
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

REPO = Path(__file__).resolve().parents[5]
RUNTIME = REPO / "python/sglang/multimodal_gen/runtime"


@contextmanager
def cpu_runtime():
    """Isolate server/Triton imports; GPU kernel execution is not mocked as valid."""
    envs = SimpleNamespace(
        SGLANG_MINIMAX_H3_INT8_ENABLE=False, SGLANG_MINIMAX_H3_INT8_FUSION=True
    )
    linear = ModuleType("linear")
    linear.LinearBase = type("LinearBase", (torch.nn.Module,), {})
    linear.LinearMethodBase = type("LinearMethodBase", (), {})
    linear.UnquantizedLinearMethod = type("UnquantizedLinearMethod", (), {})
    base = ModuleType("base_config")
    base.QuantizationConfig = type("QuantizationConfig", (), {})
    attrs = ModuleType("weight_attrs")
    attrs.set_weight_attrs = lambda tensor, values: [
        setattr(tensor, key, value) for key, value in values.items()
    ]
    mm = ModuleType("sglang.multimodal_gen")
    mm.envs = envs
    modules = {
        "sglang.multimodal_gen": mm,
        "sglang.multimodal_gen.runtime.layers.linear": linear,
        "sglang.multimodal_gen.runtime.layers.quantization.configs.base_config": base,
        "sglang.multimodal_gen.runtime.utils.weight_attrs": attrs,
    }

    triton = ModuleType("triton")
    tl = ModuleType("triton.language")
    extra = ModuleType("triton.language.extra")
    extra.libdevice = ModuleType("triton.language.extra.libdevice")
    tl.constexpr = object()
    tl.extra = extra
    triton.language = tl
    triton.jit = lambda fn: fn
    modules.update(
        {
            "triton": triton,
            "triton.language": tl,
            "triton.language.extra": extra,
            "triton.language.extra.libdevice": extra.libdevice,
        }
    )
    name = "sglang.multimodal_gen.runtime.layers.quantization.minimax_h3_int8"
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            name, RUNTIME / "layers/quantization/minimax_h3_int8.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        yield module, envs, linear


class TestMiniMaxH3Int8Scope(unittest.TestCase):
    def setUp(self):
        context = cpu_runtime()
        self.module, self.envs, self.linear = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def config(self, compressed=False):
        self.envs.SGLANG_MINIMAX_H3_INT8_ENABLE = True
        return self.module.MiniMaxH3Int8Config(
            targets=["blocks.0.attn.qkv_proj"], compressed_adaln=compressed
        )

    def test_disabled_by_default(self):
        with self.assertRaisesRegex(ValueError, "ENABLE=1"):
            self.module.MiniMaxH3Int8Config(targets=["blocks.0.attn.qkv_proj"])

    def test_generic_config_cannot_enable_adapter(self):
        self.envs.SGLANG_MINIMAX_H3_INT8_ENABLE = True
        with self.assertRaisesRegex(ValueError, "validated checkpoint"):
            self.module.MiniMaxH3Int8Config.from_config(
                {"quant_method": "minimax_h3_int8"}
            )

    def test_only_manifest_layers_are_quantized(self):
        config = self.config()
        layer = self.linear.LinearBase()
        self.assertIsInstance(
            config.get_quant_method(layer, "blocks.0.attn.qkv_proj"),
            self.module.MiniMaxH3Int8LinearMethod,
        )
        for prefix in (
            "condition_proj",
            "token_refiner.blocks.0.attn.qkv_proj",
            "blocks.0.adaln_proj.linear",
            "unrelated.blocks.0.attn.qkv_proj",
        ):
            self.assertIsInstance(
                config.get_quant_method(layer, prefix),
                self.linear.UnquantizedLinearMethod,
            )
        self.assertIsNone(
            config.get_quant_method(torch.nn.LayerNorm(4), "blocks.0.attn.qkv_proj")
        )

    def test_parameter_types_and_tp_scale_axes(self):
        method = self.module.MiniMaxH3Int8LinearMethod()
        layer = torch.nn.Module()
        method.create_weights(layer, 64, [32, 32, 32], 128, 192, torch.bfloat16)
        self.assertEqual(layer.weight.shape, (96, 64))
        self.assertEqual(layer.weight.dtype, torch.int8)
        self.assertEqual(layer.weight_scale_inv.shape, (96, 1))
        self.assertEqual(layer.weight_scale_inv.output_dim, 0)
        self.assertFalse(hasattr(layer.weight_scale_inv, "input_dim"))
        layer.weight_scale_inv.data.fill_(0.1)
        method.process_weights_after_loading(layer)
        layer.weight_scale_inv.data[0] = float("nan")
        with self.assertRaises(ValueError):
            method.process_weights_after_loading(layer)

    def test_fusion_can_be_disabled_independently(self):
        self.envs.SGLANG_MINIMAX_H3_INT8_FUSION = False
        config = self.config(compressed=True)
        method = config.get_quant_method(
            self.linear.LinearBase(), "blocks.0.attn.qkv_proj"
        )
        self.assertFalse(method.fused)

    def test_modulation_gate_is_instance_scoped(self):
        tree = ast.parse((RUNTIME / "models/dits/minimax_h3.py").read_text())
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_use_int8_fp32_modulation"
        )
        namespace = {}
        exec(
            compile(ast.Module(body=[fn], type_ignores=[]), "modulation_gate", "exec"),
            namespace,
        )
        gate = namespace[fn.name]
        arch = SimpleNamespace(adaln_curve_grid=1025)
        self.assertFalse(gate(arch, None))
        self.assertFalse(gate(arch, object()))
        self.assertFalse(gate(arch, self.config()))
        config = self.config(compressed=True)
        self.assertTrue(gate(arch, config))
        self.assertFalse(gate(SimpleNamespace(adaln_curve_grid=None), config))
        config.fused = False
        self.assertFalse(gate(arch, config))

    def test_default_modulation_calls_remain_unfused(self):
        tree = ast.parse((RUNTIME / "models/dits/minimax_h3.py").read_text())
        for name in ("_modulate_scale_shift", "_modulate_gate"):
            fn = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == name
            )
            defaults = dict(
                zip((a.arg for a in fn.args.kwonlyargs), fn.args.kw_defaults)
            )
            self.assertIs(ast.literal_eval(defaults["fused_fp32"]), False)


@unittest.skipUnless(
    os.getenv("SGLANG_TEST_H3_INT8_GPU") == "1",
    "requires a reserved GPU and SGLANG_TEST_H3_INT8_GPU=1",
)
class TestMiniMaxH3Int8Kernels(unittest.TestCase):
    def test_fused_linear_matches_reference(self):
        from sglang.multimodal_gen.runtime.layers.quantization.minimax_h3_int8 import (
            linear,
        )

        for m, k, n in ((1, 64, 64), (31, 128, 64), (4097, 64, 128), (17, 14336, 64)):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            w = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
            scale = torch.rand(n, 1, device="cuda") * 0.01
            for bias in (None, torch.randn(n, device="cuda", dtype=torch.bfloat16)):
                reference = linear(x, w, scale, bias, mode="reference")
                result = linear(x, w, scale, bias, mode="both")
                self.assertTrue(torch.equal(result, reference), (m, k, n))
                self.assertTrue(bool(torch.isfinite(result).all()))

    def test_fp32_modulation_matches_reference_without_aliasing(self):
        from sglang.multimodal_gen.runtime.layers.quantization.minimax_h3_int8 import (
            affine,
            gate,
        )

        x = torch.randn(37, 128, device="cuda", dtype=torch.bfloat16)
        other = torch.randn_like(x)
        packed = torch.randn(3, 128 * 3, device="cuda")
        shift, scale, gates = packed.chunk(3, -1)
        ids = torch.arange(37, device="cuda") % 3
        before = x.clone()
        expected = (x * (1 + scale[ids]) + shift[ids]).bfloat16()
        self.assertTrue(torch.equal(affine(x, shift, scale, ids), expected))
        expected_gate = (x + gates[ids] * other).bfloat16()
        result = gate(x, gates, other, ids)
        self.assertTrue(torch.equal(result, expected_gate))
        self.assertNotEqual(result.data_ptr(), x.data_ptr())
        self.assertTrue(torch.equal(x, before))


if __name__ == "__main__":
    unittest.main()
