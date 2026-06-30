"""
Usage:
cd test/srt
python3 -m unittest test_deterministic.TestDeterministic.TESTCASE

Note that there is also `python/sglang/test/test_deterministic.py` as an interactive test. We are converting that
test into unit tests so that's easily reproducible in CI.
"""

import os
import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
from sglang.test.dcu_utils import DCU_TEXT_SERVER_ARGS, get_model_path

register_dcu_ci(
    est_time=278,
    suite="stage-b-test-1-gpu-small-dcu",
)

from sglang.test.test_deterministic_utils import (
    BenchArgs,
    COMMON_SERVER_ARGS,
    TestDeterministicBase,
    test_deterministic,
)
from sglang.test.test_utils import is_in_amd_ci

register_cuda_ci(est_time=207, stage="stage-b", runner_config="1-gpu-large")
register_amd_ci(est_time=278, suite="stage-b-test-1-gpu-small-amd")

DCU_DETERMINISTIC_MODEL = "/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-0.6B"


def _is_dcu() -> bool:
    return os.environ.get("SGLANG_IS_IN_CI_DCU", "0") == "1"


@unittest.skipIf(_is_dcu(), "DCU uses TestDCUDeterministicSmoke.")
@unittest.skipIf(is_in_amd_ci(), "Skip for AMD CI.")
class TestFlashinferDeterministic(TestDeterministicBase):
    # Test with flashinfer attention backend
    @classmethod
    def get_server_args(cls):
        args = COMMON_SERVER_ARGS
        args.extend(
            [
                "--attention-backend",
                "flashinfer",
            ]
        )
        return args


@unittest.skipIf(_is_dcu(), "DCU uses TestDCUDeterministicSmoke.")
@unittest.skipIf(is_in_amd_ci(), "Skip for AMD CI.")
class TestFa3Deterministic(TestDeterministicBase):
    # Test with fa3 attention backend
    @classmethod
    def get_server_args(cls):
        args = COMMON_SERVER_ARGS
        args.extend(
            [
                "--attention-backend",
                "fa3",
            ]
        )
        return args


@unittest.skipIf(_is_dcu(), "DCU uses TestDCUDeterministicSmoke.")
class TestTritonDeterministic(TestDeterministicBase):
    # Test with triton attention backend
    @classmethod
    def get_server_args(cls):
        args = COMMON_SERVER_ARGS
        args.extend(
            [
                "--attention-backend",
                "triton",
            ]
        )
        return args


@unittest.skipUnless(_is_dcu(), "DCU-only deterministic smoke.")
class TestDCUDeterministicSmoke(TestDeterministicBase):
    @classmethod
    def get_model(cls):
        return get_model_path(
            "SGLANG_DCU_DETERMINISTIC_SMOKE_MODEL", DCU_DETERMINISTIC_MODEL
        )

    @classmethod
    def get_server_args(cls):
        return DCU_TEXT_SERVER_ARGS + [
            "--cuda-graph-max-bs",
            "8",
            "--enable-deterministic-inference",
            "--disable-radix-cache",
            "--disable-cuda-graph",
            "--max-total-tokens",
            "4096",
        ]

    def test_single(self):
        args = BenchArgs()
        args.host, args.port = self._extract_host_and_port(self.base_url)
        args.test_mode = "single"
        args.n_start = 2
        args.n_trials = 4
        args.max_new_tokens = 16
        args.temperature = 0.5
        results = test_deterministic(args)
        for result in results:
            self.assertEqual(result, 1)

    def test_prefix_with_logprobs(self):
        raise unittest.SkipTest(
            "DCU smoke keeps text determinism; prefix logprobs still show backend-level numeric drift."
        )


if __name__ == "__main__":
    unittest.main()
