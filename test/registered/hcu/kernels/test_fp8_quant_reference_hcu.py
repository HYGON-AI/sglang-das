# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""HCU FP8 per-token-group quantization numeric reference tests."""

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

HCU_FP8_QUANT_MAX = 224.0


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

                grouped = source.float().reshape(tokens, -1, group_size)
                reference_scales = (
                    grouped.abs().amax(dim=-1).clamp_min(1e-10)
                    / HCU_FP8_QUANT_MAX
                )
                reference_quantized = (
                    (grouped / reference_scales.unsqueeze(-1))
                    .clamp(-HCU_FP8_QUANT_MAX, HCU_FP8_QUANT_MAX)
                    .to(quantized.dtype)
                    .reshape_as(quantized)
                )

                torch.testing.assert_close(
                    scales,
                    reference_scales,
                    rtol=1e-4,
                    atol=1e-6,
                )
                quant_diff = (
                    quantized.float() - reference_quantized.float()
                ).abs()
                self.assertLess(
                    quant_diff.count_nonzero().item() / quant_diff.numel(),
                    0.005,
                )
                self.assertLessEqual(
                    quant_diff.max().item(),
                    16.0,
                )


if __name__ == "__main__":
    unittest.main()
