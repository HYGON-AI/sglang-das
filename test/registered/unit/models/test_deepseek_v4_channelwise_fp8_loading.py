"""CPU regressions for serialized channelwise FP8 wo_a and LM-head weights."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_loader.weight_utils import RUNAI_STREAMER_TENSOR_ATTR
from sglang.srt.models import deepseek_v4
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeepseekV4ChannelwiseFp8Loading(CustomTestCase):
    def setUp(self):
        self.weight = torch.tensor(
            [[2.0, 4.0, 6.0, 8.0], [4.0, 8.0, 12.0, 16.0]],
            dtype=torch.float32,
        ).to(torch.float8_e4m3fn)
        self.scale = torch.tensor([[0.25], [0.5]], dtype=torch.float32)
        self.expected = (self.weight.float() * self.scale).to(torch.bfloat16)

    def test_load_weights_dequantizes_only_a_bf16_head(self):
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                weight = torch.nn.Parameter(
                    torch.empty_like(self.weight, dtype=dtype), requires_grad=False
                )
                weight.weight_loader = lambda param, value: param.data.copy_(value)
                params = {"lm_head.weight": weight}
                if dtype == torch.float8_e4m3fn:
                    scale = torch.nn.Parameter(
                        torch.empty_like(self.scale), requires_grad=False
                    )
                    scale.weight_loader = lambda param, value: param.data.copy_(value)
                    params["lm_head.weight_scale_inv"] = scale
                model = SimpleNamespace(
                    named_parameters=lambda: params.items(),
                    config=SimpleNamespace(n_routed_experts=0, num_hidden_layers=1),
                    num_fused_shared_experts=0,
                    wo_a_fp8=False,
                    quant_config=None,
                    pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
                    remap_weight_name_to_dpsk_hf_format=deepseek_v4.DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format,
                    post_load_weights=lambda **kwargs: None,
                    model=SimpleNamespace(layers=[]),
                    _prewarm_mhc_kernels=lambda: None,
                )
                with (
                    patch.object(
                        deepseek_v4.FusedMoE,
                        "make_expert_params_mapping",
                        return_value=[],
                    ),
                    patch.object(deepseek_v4, "should_async_load", return_value=False),
                    patch("sglang.kernels.ops.gemm.bf16_fp32.prewarm_auto_bf16_fp32"),
                ):
                    deepseek_v4.DeepseekV4ForCausalLM.load_weights(
                        model,
                        [
                            ("head.weight", self.weight),
                            ("head.weight_scale_inv", self.scale),
                        ],
                    )
                expected = self.expected if dtype == torch.bfloat16 else self.weight
                torch.testing.assert_close(
                    weight.float(), expected.float(), rtol=0, atol=0
                )
                if dtype == torch.float8_e4m3fn:
                    torch.testing.assert_close(scale, self.scale, rtol=0, atol=0)

    def test_wo_a_channelwise_scale_both_orders_and_prefixes(self):
        for prefix in ("layers.0.attn.wo_a", "mtp.2.attn.wo_a"):
            for scale_first in (False, True):
                with self.subTest(prefix=prefix, scale_first=scale_first):
                    pairs = [
                        (prefix + ".weight", self.weight),
                        (prefix + ".weight_scale_inv", self.scale),
                    ]
                    if scale_first:
                        pairs.reverse()
                    converted = dict(deepseek_v4._dequant_fp8_wo_a_streaming(pairs))
                    self.assertEqual(set(converted), {prefix + ".weight"})
                    torch.testing.assert_close(
                        converted[prefix + ".weight"], self.expected, rtol=0, atol=0
                    )

    def test_wo_a_original_block_scale_both_orders(self):
        scale = torch.tensor([[0.25, 0.5]])
        expected = self.weight.float() * scale.repeat_interleave(
            2, 0
        ).repeat_interleave(2, 1)
        for scale_first in (False, True):
            with self.subTest(scale_first=scale_first):
                pairs = [
                    ("layers.0.attn.wo_a.weight", self.weight),
                    ("layers.0.attn.wo_a.scale", scale),
                ]
                if scale_first:
                    pairs.reverse()
                converted = dict(deepseek_v4._dequant_fp8_wo_a_streaming(pairs))
                torch.testing.assert_close(
                    converted["layers.0.attn.wo_a.weight"],
                    expected.bfloat16(),
                    rtol=0,
                    atol=0,
                )

    def test_lm_head_channelwise_scale_both_orders_and_shapes(self):
        for scale_first in (False, True):
            for flat in (False, True):
                with self.subTest(scale_first=scale_first, flat=flat):
                    pairs = [
                        ("head.weight", self.weight),
                        (
                            "head.weight_scale_inv",
                            self.scale.flatten() if flat else self.scale,
                        ),
                    ]
                    if scale_first:
                        pairs.reverse()
                    converted = dict(deepseek_v4._dequant_fp8_lm_head_streaming(pairs))
                    self.assertEqual(set(converted), {"head.weight"})
                    torch.testing.assert_close(
                        converted["head.weight"], self.expected, rtol=0, atol=0
                    )

    def test_lm_head_non_fp8_weight_is_unchanged(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.int8):
            with self.subTest(dtype=dtype):
                weight = self.weight.float().to(dtype)
                converted = dict(
                    deepseek_v4._dequant_fp8_lm_head_streaming(
                        [("head.weight", weight)]
                    )
                )
                self.assertIs(converted["head.weight"], weight)

    def test_lm_head_non_fp8_scale_is_not_consumed(self):
        weight = self.weight.bfloat16()
        for scale_first in (False, True):
            with self.subTest(scale_first=scale_first):
                pairs = [("head.weight", weight), ("head.weight_scale_inv", self.scale)]
                if scale_first:
                    pairs.reverse()
                converted = dict(deepseek_v4._dequant_fp8_lm_head_streaming(pairs))
                self.assertIs(converted["head.weight"], weight)
                self.assertIs(converted["head.weight_scale_inv"], self.scale)

    def test_lm_head_missing_partner_fails(self):
        for pair in (
            ("head.weight", self.weight),
            ("head.weight_scale_inv", self.scale),
        ):
            with self.subTest(name=pair[0]):
                with self.assertRaisesRegex(ValueError, "must both be present"):
                    list(deepseek_v4._dequant_fp8_lm_head_streaming([pair]))

    def test_lm_head_invalid_scale_shape_fails(self):
        for shape in ((1, 1), (2, 2), (2, 1, 1)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "expected channelwise scale"):
                    list(
                        deepseek_v4._dequant_fp8_lm_head_streaming(
                            [
                                ("head.weight", self.weight),
                                ("head.weight_scale_inv", torch.ones(shape)),
                            ]
                        )
                    )

    def test_unrelated_weights_and_draft_heads_are_unchanged(self):
        pairs = [
            ("mtp.2.markov_head.head.weight", self.weight),
            ("mtp.2.markov_head.head.weight_scale_inv", self.scale),
            ("layers.0.attn.wq_a.weight", self.weight),
        ]
        for convert in (
            deepseek_v4._dequant_fp8_lm_head_streaming,
            deepseek_v4._dequant_fp8_wo_a_streaming,
        ):
            with self.subTest(convert=convert.__name__):
                converted = list(convert(pairs))
                self.assertEqual(
                    [name for name, _ in converted], [name for name, _ in pairs]
                )
                for (_, actual), (_, expected) in zip(converted, pairs):
                    self.assertIs(actual, expected)

    def test_pending_runai_buffers_survive_reuse(self):
        for head in (False, True):
            for scale_first in (False, True):
                with self.subTest(head=head, scale_first=scale_first):
                    prefix = "head" if head else "layers.0.attn.wo_a"
                    convert = (
                        deepseek_v4._dequant_fp8_lm_head_streaming
                        if head
                        else deepseek_v4._dequant_fp8_wo_a_streaming
                    )
                    first = self.scale.clone() if scale_first else self.weight.clone()
                    setattr(first, RUNAI_STREAMER_TENSOR_ATTR, True)

                    def stream():
                        yield (
                            prefix
                            + (".weight_scale_inv" if scale_first else ".weight"),
                            first,
                        )
                        first.view(torch.uint8).zero_()
                        yield (
                            prefix
                            + (".weight" if scale_first else ".weight_scale_inv"),
                            self.weight if scale_first else self.scale,
                        )

                    converted = dict(convert(stream()))
                    torch.testing.assert_close(
                        converted[prefix + ".weight"], self.expected, rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
