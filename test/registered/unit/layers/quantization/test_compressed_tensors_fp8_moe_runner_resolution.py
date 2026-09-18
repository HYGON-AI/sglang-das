"""Keep FP8 MoE weight preparation and runtime on the same resolved backend."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from compressed_tensors.quantization import QuantizationStrategy

import sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe as fp8_moe
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCompressedTensorsFp8MoeRunnerResolution(CustomTestCase):
    def setUp(self):
        self.method = fp8_moe.CompressedTensorsW8A8Fp8MoE.__new__(
            fp8_moe.CompressedTensorsW8A8Fp8MoE
        )
        self.method.weight_quant = SimpleNamespace(
            strategy=QuantizationStrategy.CHANNEL
        )

    def resolve(self, backend, *, use_aiter):
        with (
            patch.object(fp8_moe, "get_moe_runner_backend", return_value=backend),
            patch.object(
                fp8_moe,
                "get_moe_a2a_backend",
                return_value=MoeA2ABackend.NONE,
            ),
            patch.object(fp8_moe, "_use_aiter", use_aiter),
        ):
            return self.method._resolve_moe_runner_backend()

    def test_auto_without_aiter_resolves_to_triton(self):
        self.assertIs(
            self.resolve(MoeRunnerBackend.AUTO, use_aiter=False),
            MoeRunnerBackend.TRITON,
        )

    def test_auto_with_supported_aiter_resolves_to_aiter(self):
        self.assertIs(
            self.resolve(MoeRunnerBackend.AUTO, use_aiter=True),
            MoeRunnerBackend.AITER,
        )

    def test_explicit_triton_is_not_overridden_by_aiter_availability(self):
        self.assertIs(
            self.resolve(MoeRunnerBackend.TRITON, use_aiter=True),
            MoeRunnerBackend.TRITON,
        )

    def test_explicit_deep_gemm_is_preserved(self):
        self.assertIs(
            self.resolve(MoeRunnerBackend.DEEP_GEMM, use_aiter=False),
            MoeRunnerBackend.DEEP_GEMM,
        )


if __name__ == "__main__":
    unittest.main()
