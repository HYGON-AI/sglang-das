"""Unit tests for linear_bf16_fp32 backend selection."""

import importlib
import unittest
from unittest import mock

import torch

import sglang.kernels.ops.attention.dsv4.gemm as gemm_module
from sglang.kernels.ops.attention.dsv4.gemm import _select_auto_backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestLinearBf16Fp32Dispatch(CustomTestCase):
    @staticmethod
    def _select(
        *,
        m: int,
        n: int,
        k: int = 4096,
        is_hcu_device: bool = False,
        profile_enabled: bool = False,
        sgl_supported: bool = False,
        aiter_available: bool = False,
    ) -> str:
        return _select_auto_backend(
            m=m,
            n=n,
            k=k,
            x_dtype=torch.bfloat16,
            y_dtype=torch.bfloat16,
            x_contiguous=True,
            y_contiguous=True,
            is_hcu_device=is_hcu_device,
            hcu_profile_enabled=profile_enabled,
            sgl_supported=sgl_supported,
            aiter_available=aiter_available,
        )

    def test_non_hcu_always_falls_back_to_cublas(self):
        for m, n in ((32, 256), (1024, 512), (769, 1024)):
            self.assertEqual(self._select(m=m, n=n), "cublas")

    def test_unsupported_input_contract_falls_back_to_cublas(self):
        common = dict(
            m=32,
            n=1024,
            k=4096,
            is_hcu_device=True,
            hcu_profile_enabled=True,
            sgl_supported=True,
            aiter_available=True,
        )
        self.assertEqual(
            _select_auto_backend(
                x_dtype=torch.float16,
                y_dtype=torch.bfloat16,
                x_contiguous=True,
                y_contiguous=True,
                **common,
            ),
            "cublas",
        )
        self.assertEqual(
            _select_auto_backend(
                x_dtype=torch.bfloat16,
                y_dtype=torch.bfloat16,
                x_contiguous=False,
                y_contiguous=True,
                **common,
            ),
            "cublas",
        )
        self.assertEqual(
            self._select(
                m=32,
                n=128,
                is_hcu_device=True,
                profile_enabled=True,
                sgl_supported=True,
                aiter_available=True,
            ),
            "cublas",
        )

    def test_hcu_shape_boundaries(self):
        tuned = dict(is_hcu_device=True, profile_enabled=True)
        self.assertEqual(
            self._select(m=64, n=256, sgl_supported=True, **tuned),
            "sgl",
        )
        self.assertEqual(
            self._select(m=65, n=256, sgl_supported=True, **tuned),
            "torch",
        )
        self.assertEqual(self._select(m=769, n=256, **tuned), "cublas")
        self.assertEqual(
            self._select(m=64, n=512, sgl_supported=True, **tuned),
            "sgl",
        )
        self.assertEqual(self._select(m=1024, n=512, **tuned), "torch")
        self.assertEqual(self._select(m=1025, n=512, **tuned), "cublas")
        self.assertEqual(
            self._select(m=64, n=1024, sgl_supported=True, **tuned),
            "sgl",
        )
        self.assertEqual(
            self._select(m=64, n=2048, sgl_supported=True, **tuned),
            "sgl",
        )
        self.assertEqual(self._select(m=768, n=1024, **tuned), "torch")
        self.assertEqual(self._select(m=769, n=1024, **tuned), "cublas")

    def test_hcu_profile_boundaries(self):
        tuned = dict(is_hcu_device=True, profile_enabled=True)
        self.assertEqual(
            self._select(
                m=32,
                n=1024,
                sgl_supported=True,
                aiter_available=True,
                **tuned,
            ),
            "aiter",
        )
        self.assertEqual(
            self._select(m=1, n=256, k=4096, sgl_supported=True, **tuned),
            "sgl",
        )
        self.assertEqual(
            self._select(
                m=1,
                n=256,
                k=6144,
                sgl_supported=True,
                aiter_available=True,
                **tuned,
            ),
            "aiter",
        )
        self.assertEqual(
            self._select(m=1024, n=256, aiter_available=True, **tuned),
            "aiter",
        )
        self.assertEqual(
            self._select(m=4096, n=256, aiter_available=True, **tuned),
            "aiter",
        )
        self.assertEqual(
            self._select(m=1280, n=512, aiter_available=True, **tuned),
            "aiter",
        )
        for m in (1408, 2048, 2816):
            self.assertEqual(
                self._select(m=m, n=512, k=6144, aiter_available=True, **tuned),
                "cublas",
            )
        for m in (1408, 2048, 2816):
            self.assertEqual(
                self._select(m=m, n=512, k=4096, aiter_available=True, **tuned),
                "aiter",
            )
        self.assertEqual(
            self._select(m=3072, n=512, aiter_available=True, **tuned),
            "aiter",
        )
        self.assertEqual(
            self._select(m=1024, n=1024, aiter_available=False, **tuned),
            "cublas",
        )

    def test_sgl_requires_kernel_support_and_aligned_k(self):
        tuned = dict(is_hcu_device=True, profile_enabled=True)
        self.assertEqual(
            self._select(m=32, n=1024, sgl_supported=False, **tuned),
            "torch",
        )
        self.assertEqual(
            self._select(m=32, n=1024, k=130, sgl_supported=True, **tuned),
            "torch",
        )
        self.assertEqual(
            self._select(
                m=32,
                n=1024,
                is_hcu_device=True,
                profile_enabled=False,
                sgl_supported=True,
            ),
            "cublas",
        )
    def test_unknown_n_does_not_select_aiter(self):
        self.assertEqual(
            self._select(
                m=32,
                n=1536,
                k=7168,
                sgl_supported=True,
                aiter_available=True,
                is_hcu_device=True,
                profile_enabled=True,
            ),
            "cublas",
        )

    def test_public_auto_dispatch_precedes_legacy_aiter(self):
        sentinel = object()
        x = torch.empty((1, 4), dtype=torch.bfloat16)
        y = torch.empty((2, 4), dtype=torch.bfloat16)
        with mock.patch.object(
            gemm_module, "_linear_bf16_fp32_algo", "auto"
        ), mock.patch.object(
            gemm_module, "_auto_dispatch_bf16_fp32", return_value=sentinel
        ):
            result = gemm_module.linear_bf16_fp32(x, y)
        self.assertIs(result, sentinel)

        importlib.import_module(
            "sglang.kernels.ops.attention.dsv4.hcu_linear_bf16_fp32"
        )
        import sglang.kernels.ops.attention.dsv4 as dsv4

        self.assertTrue(callable(dsv4.linear_bf16_fp32))


if __name__ == "__main__":
    unittest.main()
