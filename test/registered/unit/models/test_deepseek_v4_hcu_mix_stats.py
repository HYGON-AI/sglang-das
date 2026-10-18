"""HCU mHC dispatch survives the extraction into deepseek_v4_mhc."""

import sys
import unittest
from contextlib import ExitStack
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestHcuMixStats(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.cfg = mhc.HcConfig(2, 20, 1e-6, 1e-6, 4, True, False)
        self.hc = mhc.HcSubLayer(
            self.cfg,
            torch.randn(8, 8),
            torch.ones(3),
            torch.zeros(8),
            None,
            object(),
            object(),
        )
        self.lightop = ModuleType("lightop")
        self.result = (torch.ones(1), torch.ones(1), torch.ones(1))
        self.lightop.hc_mix_stats_sinkhorn_deepgemm = Mock(return_value=self.result)
        self.lightop.hc_mix_stats_sinkhorn_bf16x3 = Mock(return_value=self.result)
        self.kernels = ModuleType("sglang.kernels.ops.layernorm.mhc")
        self.kernels.hc_mix_stats = Mock(side_effect=AssertionError("CUDA path"))
        self.kernels.hc_mix_stats_sinkhorn = Mock(
            side_effect=AssertionError("CUDA path")
        )
        self.kernels.hc_split_sinkhorn = Mock(
            side_effect=lambda mixes, *args: (mixes,) * 3
        )
        self.stack.enter_context(
            patch.dict(
                sys.modules,
                {
                    "lightop": self.lightop,
                    "sglang.kernels.ops.layernorm.mhc": self.kernels,
                },
            )
        )
        self.stack.enter_context(patch.object(mhc, "_use_hcu_hc_mix_stats", True))
        self.stack.enter_context(patch.object(mhc, "_is_gfx95_supported", False))
        self.stack.enter_context(patch.object(mhc, "_is_xpu", False))
        self.stack.enter_context(
            patch.object(
                mhc,
                "get_platform",
                return_value=SimpleNamespace(
                    is_sm90=False,
                    is_sm100=False,
                    is_blackwell=False,
                ),
            )
        )
        self.invariant = self.stack.enter_context(
            patch(
                "sglang.srt.batch_invariant_ops.is_batch_invariant_mode_enabled",
                return_value=False,
            )
        )

    def test_projection_boundaries(self):
        for rows in (1, 4095, 4096, 65536, 65537):
            with self.subTest(rows=rows):
                x = torch.zeros(rows, 2, 4, dtype=torch.bfloat16)
                self.lightop.hc_mix_stats_sinkhorn_deepgemm.reset_mock()
                self.lightop.hc_mix_stats_sinkhorn_bf16x3.reset_mock()
                self.assertIs(mhc.mix_stats(self.hc, x), self.result)
                bf16 = 4096 <= rows <= 65536
                chosen = (
                    self.lightop.hc_mix_stats_sinkhorn_bf16x3
                    if bf16
                    else self.lightop.hc_mix_stats_sinkhorn_deepgemm
                )
                unused = (
                    self.lightop.hc_mix_stats_sinkhorn_deepgemm
                    if bf16
                    else self.lightop.hc_mix_stats_sinkhorn_bf16x3
                )
                chosen.assert_called_once()
                unused.assert_not_called()
                args = chosen.call_args.args
                self.assertEqual(args[0].shape, (rows, 8))
                self.assertIs(
                    args[1], self.hc.bf16_parts if bf16 else self.hc.tf32_parts
                )
                self.assertIs(args[2], self.hc.scale)
                self.assertIs(args[3], self.hc.base)
                self.assertEqual(args[4:], (20, 1e-6, 1e-6))
        self.kernels.hc_split_sinkhorn.assert_not_called()

    def test_torch_fallbacks(self):
        for scenario in (
            "disabled",
            "invariant",
            "missing_tf32",
            "missing_bf16",
            "dtype",
            "layout",
        ):
            with self.subTest(scenario=scenario):
                rows = 4096 if scenario == "missing_bf16" else 2
                x = torch.randn(rows, 2, 4, dtype=torch.bfloat16)
                hc = self.hc
                if scenario == "missing_tf32":
                    hc = hc._replace(tf32_parts=None)
                if scenario == "missing_bf16":
                    hc = hc._replace(bf16_parts=None)
                if scenario == "dtype":
                    x = x.float()
                if scenario == "layout":
                    x = torch.randn(rows, 2, 8, dtype=torch.bfloat16)[:, :, ::2]
                self.invariant.return_value = scenario == "invariant"
                with patch.object(mhc, "_use_hcu_hc_mix_stats", scenario != "disabled"):
                    result = mhc.mix_stats(hc, x)
                xf = x.flatten(1).float()
                expected = F.linear(xf, hc.fn) * torch.rsqrt(
                    xf.square().mean(-1, keepdim=True) + self.cfg.rms_eps
                )
                for output in result:
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.lightop.hc_mix_stats_sinkhorn_deepgemm.assert_not_called()
        self.lightop.hc_mix_stats_sinkhorn_bf16x3.assert_not_called()

    def test_weight_cache_binds_attention_and_ffn(self):
        from sglang.srt.models import deepseek_v4 as model

        layer = SimpleNamespace(
            input_layernorm=SimpleNamespace(weight=torch.ones(4)),
            post_attention_layernorm=SimpleNamespace(weight=torch.ones(4)),
            hc_pre_from_prev_sublayer=True,
            config=SimpleNamespace(model_type="deepseek_v41"),
            hc_cfg=self.cfg,
            hc_attn_fn=torch.zeros(24, 20480),
            hc_ffn_fn=torch.ones(24, 20480),
            hc_attn_scale=self.hc.scale,
            hc_ffn_scale=self.hc.scale,
            hc_attn_base=self.hc.base,
            hc_ffn_base=self.hc.base,
        )
        layer._init_hyper_connections = lambda: (
            model.DeepseekV4DecoderLayer._init_hyper_connections(layer)
        )
        tf32_parts, bf16_parts = (object(), object()), (object(), object())
        self.lightop.split_tf32_hc_weight = Mock(side_effect=tf32_parts)
        self.lightop.split_bf16_hc_weight = Mock(side_effect=bf16_parts)
        model.DeepseekV4DecoderLayer.refresh_mhc_norm_weight_cache(layer)
        self.assertIs(layer.attn_hc.tf32_parts, tf32_parts[0])
        self.assertIs(layer.ffn_hc.tf32_parts, tf32_parts[1])
        self.assertIs(layer.attn_hc.bf16_parts, bf16_parts[0])
        self.assertIs(layer.ffn_hc.bf16_parts, bf16_parts[1])
        self.assertIs(layer.attn_hc.fn, layer.hc_attn_fn)
        self.assertIs(layer.ffn_hc.fn, layer.hc_ffn_fn)
        self.invariant.return_value = True
        with patch.object(
            model,
            "get_platform",
            return_value=SimpleNamespace(is_sm90=False, is_sm100=False),
        ):
            model.DeepseekV4DecoderLayer.refresh_mhc_norm_weight_cache(layer)
        for hc in (layer.attn_hc, layer.ffn_hc):
            self.assertIsNone(hc.tf32_parts)
            self.assertIsNone(hc.bf16_parts)

    def test_execution_errors_are_visible(self):
        self.lightop.hc_mix_stats_sinkhorn_deepgemm.side_effect = RuntimeError(
            "device error"
        )
        with self.assertRaisesRegex(RuntimeError, "device error"):
            mhc.mix_stats(self.hc, torch.zeros(1, 2, 4, dtype=torch.bfloat16))
        self.kernels.hc_split_sinkhorn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
