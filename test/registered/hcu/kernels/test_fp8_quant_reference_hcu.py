# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""HCU FP8 per-token-group quantization numeric reference tests."""

import math
import unittest

import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
)
from sglang.test.ci.ci_register import register_hcu_ci

register_hcu_ci(
    est_time=120,
    suite="nightly-hcu-core-functional",
    nightly=True,
)
register_hcu_ci(est_time=60, suite="stage-b-test-1-hcu-small")


class TestBW1100FP8QuantReferenceHCU(unittest.TestCase):
    def test_per_token_group_quant_fp8_reference(self):
        torch.manual_seed(2026)
        for tokens, hidden_size, group_size in (
            (1, 128, 16),
            (4, 256, 32),
            (16, 512, 64),
            (32, 1024, 128),
        ):
            with self.subTest(
                tokens=tokens,
                hidden_size=hidden_size,
                group_size=group_size,
            ):
                source = torch.randn(
                    tokens,
                    hidden_size,
                    device="cuda",
                    dtype=torch.bfloat16,
                ).contiguous()
                quantized, scales = per_token_group_quant_fp8(
                    source,
                    group_size=group_size,
                )
                # E4M3FN and E4M3FNUZ have different finite ranges.
                quant_info = torch.finfo(quantized.dtype)
                quant_max = quant_info.max
                # Preserve a one-ULP rounding allowance at the largest exponent:
                # 16 for E4M3FNUZ, 32 for E4M3FN.
                max_quant_step = math.ldexp(
                    quant_info.eps, math.frexp(quant_max)[1] - 1
                )

                grouped = source.float().reshape(tokens, -1, group_size)
                reference_scales = (
                    grouped.abs().amax(dim=-1).clamp_min(1e-10) / quant_max
                )
                reference_quantized = (
                    (grouped / reference_scales.unsqueeze(-1))
                    .clamp(-quant_max, quant_max)
                    .to(quantized.dtype)
                    .reshape_as(quantized)
                )

                torch.testing.assert_close(
                    scales,
                    reference_scales,
                    rtol=1e-4,
                    atol=1e-6,
                )
                quant_diff = (quantized.float() - reference_quantized.float()).abs()
                self.assertLess(
                    quant_diff.count_nonzero().item() / quant_diff.numel(),
                    0.005,
                )
                self.assertLessEqual(
                    quant_diff.max().item(),
                    max_quant_step,
                )


if __name__ == "__main__":
    unittest.main()
