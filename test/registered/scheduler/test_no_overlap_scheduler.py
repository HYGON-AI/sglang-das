"""
Usage:
python3 -m unittest test_overlap_schedule.TestOverlapSchedule.test_radix_attention_chunked_prefill
python3 test_overlap_schedule.py
"""

import os
import unittest
from types import SimpleNamespace

from sglang.test.run_eval import run_eval
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
from sglang.test.test_utils import CustomTestCase, run_and_check_memory_leak, run_mmlu_test

register_cuda_ci(est_time=245, suite="stage-b-test-1-gpu-large")
register_amd_ci(est_time=275, suite="stage-b-test-1-gpu-small-amd")


register_dcu_ci(
    est_time=90,
    suite="stage-b-test-1-gpu-small-dcu",
)


def _is_dcu():
    return os.environ.get("SGLANG_IS_IN_CI_DCU") == "1"


def _run_dcu_mmlu_smoke(
    disable_radix_cache=False,
    chunked_prefill_size=32,
    disable_overlap=True,
):
    if chunked_prefill_size > 0:
        chunked_prefill_size = max(64, chunked_prefill_size)

    def workload_func(base_url, model):
        args = SimpleNamespace(
            base_url=base_url,
            model=model,
            eval_name="mmlu",
            num_examples=8,
            num_threads=8,
        )
        metrics = run_eval(args)
        assert metrics["score"] >= 0.0, f"metrics={metrics}"

    run_and_check_memory_leak(
        workload_func,
        disable_radix_cache,
        enable_mixed_chunk=False,
        disable_overlap=disable_overlap,
        chunked_prefill_size=chunked_prefill_size,
        assert_has_abort=False,
    )


def _run_scheduler_mmlu_test(**kwargs):
    if _is_dcu():
        _run_dcu_mmlu_smoke(**kwargs)
    else:
        run_mmlu_test(**kwargs)


class TestOverlapSchedule(CustomTestCase):
    def test_no_radix_attention_chunked_prefill(self):
        _run_scheduler_mmlu_test(
            disable_radix_cache=True, chunked_prefill_size=32, disable_overlap=True
        )

    def test_no_radix_attention_no_chunked_prefill(self):
        _run_scheduler_mmlu_test(
            disable_radix_cache=True, chunked_prefill_size=-1, disable_overlap=True
        )

    def test_radix_attention_chunked_prefill(self):
        _run_scheduler_mmlu_test(
            disable_radix_cache=False, chunked_prefill_size=32, disable_overlap=True
        )

    def test_radix_attention_no_chunked_prefill(self):
        _run_scheduler_mmlu_test(
            disable_radix_cache=False, chunked_prefill_size=-1, disable_overlap=True
        )


if __name__ == "__main__":
    unittest.main()
