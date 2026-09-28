"""Correctness tests for the HCU BF16-to-FP32 GEMM kernel."""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4.gemm import _auto_dispatch_bf16_fp32
from sglang.kernels.ops.attention.dsv4.hcu_linear_bf16_fp32 import (
    hcu_linear_bf16_fp32,
    hcu_linear_bf16_fp32_supported,
)
from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.test_utils import CustomTestCase

register_hcu_ci(est_time=120, suite="stage-b-test-1-hcu-small")


class TestLinearBf16Fp32Hcu(CustomTestCase):
    def _assert_matches_torch(self, m: int, n: int, k: int) -> None:
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

        self.assertTrue(hcu_linear_bf16_fp32_supported(x, weight))
        actual = hcu_linear_bf16_fp32(x, weight)
        expected = torch.nn.functional.linear(x.float(), weight.float())

        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_mfma_inner_tile_shapes(self):
        self._assert_matches_torch(m=1, n=256, k=6144)
        self._assert_matches_torch(m=64, n=512, k=6144)

    def test_mfma_split_k_shapes(self):
        self._assert_matches_torch(m=32, n=1024, k=4096)
        self._assert_matches_torch(m=64, n=2048, k=4096)

    def test_auto_dispatch_matches_fp32_reference(self):
        x = torch.randn((32, 6144), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn((256, 6144), device="cuda", dtype=torch.bfloat16)

        actual = _auto_dispatch_bf16_fp32(x, weight)
        expected = torch.nn.functional.linear(x.float(), weight.float())

        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    @patch(
        "sglang.kernels.ops.attention.dsv4.gemm._get_aiter_tgemm",
        return_value=None,
    )
    def test_auto_dispatch_uses_sgl_without_aiter(self, _mock_get_aiter):
        x = torch.randn((64, 4096), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn((1024, 4096), device="cuda", dtype=torch.bfloat16)

        with patch(
            "sglang.kernels.ops.attention.dsv4.hcu_linear_bf16_fp32."
            "hcu_linear_bf16_fp32",
            wraps=hcu_linear_bf16_fp32,
        ) as mock_sgl:
            actual = _auto_dispatch_bf16_fp32(x, weight)
        expected = torch.nn.functional.linear(x.float(), weight.float())

        mock_sgl.assert_called_once()
        call_x, call_weight = mock_sgl.call_args.args
        self.assertIs(call_x, x)
        self.assertIs(call_weight, weight)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_unsupported_shapes_are_rejected(self):
        x = torch.randn((65, 128), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16)

        self.assertFalse(hcu_linear_bf16_fp32_supported(x, weight))
        with self.assertRaises(ValueError):
            hcu_linear_bf16_fp32(x, weight)


if __name__ == "__main__":
    unittest.main(verbosity=3)
