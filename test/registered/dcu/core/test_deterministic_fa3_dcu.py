"""
Usage:
cd test/srt
python3 -m unittest test_deterministic.TestDeterministic.TESTCASE

Note that there is also `python/sglang/test/test_deterministic.py` as an interactive test. We are converting that
test into unit tests so that's easily reproducible in CI.
"""

import unittest

from sglang.test.ci.ci_register import register_dcu_ci

register_dcu_ci(
    est_time=900,
    suite="stage-b-test-1-gpu-small-dcu",
    disabled="DCU disabled retest: fa3 small-model deterministic path still fails in batch_invariant_ops persistent matmul with Triton shared-memory OOR and scheduler exit -9.",
)

from sglang.test.test_deterministic_utils import (
    COMMON_SERVER_ARGS,
    TestDeterministicBase,
)
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST



@unittest.skip("DCU smoke keeps fa3 only.")
class TestFlashinferDeterministic(TestDeterministicBase):
    # Test with flashinfer attention backend
    @classmethod
    def get_server_args(cls):
        args = list(COMMON_SERVER_ARGS)
        args.extend(
            [
                "--attention-backend",
                "flashinfer",
            ]
        )
        return args


class TestFa3Deterministic(TestDeterministicBase):
    # Test with fa3 attention backend
    @classmethod
    def get_model(cls):
        return DEFAULT_SMALL_MODEL_NAME_FOR_TEST

    @classmethod
    def get_server_args(cls):
        args = list(COMMON_SERVER_ARGS)
        args.extend(
            [
                "--attention-backend",
                "fa3",
                "--page-size",
                "64",
                "--disable-cuda-graph",
            ]
        )
        return args


@unittest.skip("DCU smoke keeps fa3 only.")
class TestTritonDeterministic(TestDeterministicBase):
    # Test with triton attention backend
    @classmethod
    def get_server_args(cls):
        args = list(COMMON_SERVER_ARGS)
        args.extend(
            [
                "--attention-backend",
                "triton",
            ]
        )
        return args


if __name__ == "__main__":
    unittest.main()
