# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""HCU INT8 quantization and W8A8 GEMM reference checks."""

import unittest

import torch
from lightop.quant import per_token_quant_int8

from sglang.srt.layers.quantization.compressed_tensors.quant_ops import (
    blaslt_scaled_mm,
)
from sglang.test.ci.ci_register import register_hcu_ci

register_hcu_ci(
    est_time=120,
    suite="nightly-hcu-core-functional",
    nightly=True,
)


def _quantize_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scales = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-10) / 127.0
    quantized = torch.round(weight / scales).clamp(-128, 127).to(torch.int8)
    return quantized.contiguous(), scales


class TestBW1100Int8W8A8ReferenceHCU(unittest.TestCase):
    def test_per_token_int8_quantization_matches_reference(self):
        torch.manual_seed(2026)
        source = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
        quantized, scales = per_token_quant_int8(source)

        expected_scales = (
            source.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-10) / 127.0
        )
        expected_quantized = (
            torch.round(source.float() / expected_scales)
            .clamp(-128, 127)
            .to(torch.int8)
        )

        torch.testing.assert_close(
            scales.float(), expected_scales, rtol=1e-4, atol=1e-6
        )
        quant_diff = (quantized.to(torch.int16) - expected_quantized.to(torch.int16)).abs()
        self.assertLessEqual(quant_diff.max().item(), 1)
        self.assertLessEqual(
            quant_diff.count_nonzero().item() / quant_diff.numel(),
            0.001,
        )

    def test_w8a8_blaslt_gemm_matches_dequantized_reference(self):
        torch.manual_seed(2026)
        for m, n, k in ((1, 128, 128), (16, 256, 256), (32, 128, 512)):
            with self.subTest(m=m, n=n, k=k):
                source = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                weight = torch.randn(n, k, device="cuda", dtype=torch.float32)
                q_input, input_scales = per_token_quant_int8(source)
                q_weight, weight_scales = _quantize_per_channel(weight)

                actual = blaslt_scaled_mm(
                    q_input,
                    # HCU KME consumes a column-major [K, N] weight view.
                    q_weight.t(),
                    input_scales,
                    weight_scales,
                    out_dtype=torch.bfloat16,
                )
                reference = (
                    (q_input.float() * input_scales.float())
                    @ (q_weight.float() * weight_scales.float()).t()
                ).to(torch.bfloat16)

                torch.testing.assert_close(actual, reference, rtol=2e-2, atol=0.25)


if __name__ == "__main__":
    unittest.main()
