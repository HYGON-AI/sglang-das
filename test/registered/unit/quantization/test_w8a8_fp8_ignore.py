import unittest

import torch

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.w8a8_fp8 import (
    W8A8Fp8Config,
    W8A8Fp8LinearMethod,
)


class TestW8A8Fp8IgnoredLayers(unittest.TestCase):
    def config(self, ignore=None):
        return W8A8Fp8Config.from_config(
            {
                "quant_method": "compressed-tensors",
                "ignore": (
                    ignore
                    if ignore is not None
                    else [
                        r"re:.*\.linear_attn\.in_proj_a",
                        r"re:.*\.linear_attn\.in_proj_b",
                    ]
                ),
                "packed_modules_mapping": {
                    "in_proj_ba": ["in_proj_b", "in_proj_a"],
                    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
                },
            }
        )

    def linear(self, config, prefix):
        return ReplicatedLinear(
            8,
            4,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=config,
            prefix=prefix,
        )

    def test_bf16_gdn_gates_keep_checkpoint_values_without_fp8_scales(self):
        # Qwen3.5 Channel-FP8 stores both GDN gate projections as BF16,
        # without weight_scale tensors. They are fused into in_proj_ba.
        layer = self.linear(self.config(), "model.layers.0.linear_attn.in_proj_ba")
        self.assertIsInstance(layer.quant_method, UnquantizedLinearMethod)
        self.assertEqual(layer.weight.dtype, torch.bfloat16)
        self.assertFalse(hasattr(layer, "weight_scale"))
        checkpoint = torch.linspace(-0.75, 0.75, 32).reshape(4, 8).bfloat16()
        layer.weight_loader(layer.weight, checkpoint)
        self.assertTrue(torch.equal(layer.weight, checkpoint))

    def test_quantized_gdn_projections_still_allocate_fp8_scales(self):
        layer = self.linear(self.config(), "model.layers.0.linear_attn.in_proj_qkvz")
        self.assertIsInstance(layer.quant_method, W8A8Fp8LinearMethod)
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(layer.weight_scale.shape), (4, 1))

    def test_unfused_exact_checkpoint_ignore(self):
        prefix = "model.layers.0.self_attn.q_proj"
        layer = self.linear(self.config([prefix]), prefix)
        self.assertIsInstance(layer.quant_method, UnquantizedLinearMethod)

    def test_mixed_precision_fused_shards_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "different quantization schemes"):
            self.linear(
                self.config([r"re:.*\.linear_attn\.in_proj_a"]),
                "model.layers.0.linear_attn.in_proj_ba",
            )

    def test_default_online_quantization_is_unchanged(self):
        layer = self.linear(W8A8Fp8Config(), "model.layers.0.linear_attn.in_proj_ba")
        self.assertIsInstance(layer.quant_method, W8A8Fp8LinearMethod)
        self.assertEqual(layer.weight.dtype, torch.bfloat16)
        self.assertIsNone(layer.weight_scale)


class TestW8A8Fp8AiterLayoutOwnership(unittest.TestCase):
    def assert_runner(self, *, env_value, packed_path, is_hcu, expected):
        import os
        from unittest.mock import patch

        from sglang.srt.layers.moe import MoeRunnerBackend, MoeRunnerConfig
        from sglang.srt.layers.quantization import w8a8_fp8

        method = object.__new__(w8a8_fp8.W8A8FP8MoEMethod)
        config = MoeRunnerConfig()
        with (
            patch.dict(os.environ),
            patch.object(w8a8_fp8, "_is_hcu", is_hcu),
            patch.object(w8a8_fp8, "_use_fp8_w8a8_moe", packed_path),
            patch.object(
                w8a8_fp8, "get_moe_runner_backend", return_value=MoeRunnerBackend.TRITON
            ),
            patch.object(w8a8_fp8, "MoeRunner") as runner,
        ):
            if env_value is None:
                os.environ.pop("SGLANG_ROCM_USE_AITER_MOE", None)
            else:
                os.environ["SGLANG_ROCM_USE_AITER_MOE"] = env_value
            method.create_moe_runner(torch.nn.Module(), config)
            runner.assert_called_once_with(getattr(MoeRunnerBackend, expected), config)

    def test_default_hcu_fallback_uses_runner_that_owns_shuffle(self):
        self.assert_runner(
            env_value=None, packed_path=False, is_hcu=True, expected="AITER"
        )

    def test_explicit_aiter_fallback_opt_out_keeps_triton(self):
        self.assert_runner(
            env_value="0", packed_path=False, is_hcu=True, expected="TRITON"
        )

    def test_lightop_packed_weights_do_not_enter_aiter_shuffle(self):
        self.assert_runner(
            env_value="1", packed_path=True, is_hcu=True, expected="TRITON"
        )

    def test_other_hardware_keeps_triton(self):
        self.assert_runner(
            env_value="1", packed_path=False, is_hcu=False, expected="TRITON"
        )


if __name__ == "__main__":
    unittest.main()
