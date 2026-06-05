"""
Usage:
python3 -m unittest test_overlap_schedule.TestOverlapSchedule.test_radix_attention_chunked_prefill
python3 test_overlap_schedule.py
"""

import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
from sglang.test.test_utils import CustomTestCase, run_mmlu_test

register_cuda_ci(est_time=245, suite="stage-b-test-1-gpu-large")
register_amd_ci(est_time=275, suite="stage-b-test-1-gpu-small-amd")


# DCU_CSV_COVERED_UNVERIFIED: Enabled from sglang.csv historical DCU coverage; not re-tested in this framework pass.
register_dcu_ci(
    est_time=120,
    suite="stage-b-test-1-gpu-small-dcu",
    disabled="DCU PR baseline deferred: scheduler path needs BW1100 repeat validation before required CI.",
)

class TestOverlapSchedule(CustomTestCase):
    def test_no_radix_attention_chunked_prefill(self):
        run_mmlu_test(
            disable_radix_cache=True, chunked_prefill_size=32, disable_overlap=True
        )

    def test_no_radix_attention_no_chunked_prefill(self):
        run_mmlu_test(
            disable_radix_cache=True, chunked_prefill_size=-1, disable_overlap=True
        )

    def test_radix_attention_chunked_prefill(self):
        run_mmlu_test(
            disable_radix_cache=False, chunked_prefill_size=32, disable_overlap=True
        )

    def test_radix_attention_no_chunked_prefill(self):
        run_mmlu_test(
            disable_radix_cache=False, chunked_prefill_size=-1, disable_overlap=True
        )


if __name__ == "__main__":
    unittest.main()
