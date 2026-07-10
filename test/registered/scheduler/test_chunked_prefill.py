# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
python3 -m unittest test_chunked_prefill.TestChunkedPrefill.test_mixed_chunked_prefill_without_radix_cache
"""

import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import requests

from sglang.test.run_eval import run_eval
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    run_and_check_memory_leak,
    run_mmlu_test,
    run_mulit_request_test,
)

register_cuda_ci(est_time=312, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=312, suite="stage-b-test-1-gpu-small-amd")
register_dcu_ci(est_time=120, suite="stage-b-test-1-gpu-small-dcu")


def _is_dcu():
    return os.environ.get("SGLANG_IS_IN_CI_DCU") == "1"


def _run_dcu_mmlu_smoke(disable_radix_cache=False, enable_mixed_chunk=False):
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
        enable_mixed_chunk,
        disable_overlap=False,
        chunked_prefill_size=32,
        assert_has_abort=False,
    )


def _run_dcu_multi_request_smoke(enable_mixed_chunk=False, chunked_prefill_size=2048):
    def workload_func(base_url, model):
        def run_one(_):
            response = requests.post(
                f"{base_url}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 4,
                    },
                },
            )
            assert response.status_code == 200, response.text
            response.json()

        with ThreadPoolExecutor(2) as executor:
            list(executor.map(run_one, list(range(2))))

    run_and_check_memory_leak(
        workload_func,
        disable_radix_cache=False,
        enable_mixed_chunk=enable_mixed_chunk,
        disable_overlap=False,
        chunked_prefill_size=chunked_prefill_size,
        assert_has_abort=False,
    )


def _run_chunked_mmlu_test(**kwargs):
    if _is_dcu():
        _run_dcu_mmlu_smoke(**kwargs)
    else:
        run_mmlu_test(**kwargs)


def _run_chunked_multi_request_test(**kwargs):
    if _is_dcu():
        _run_dcu_multi_request_smoke(**kwargs)
    else:
        run_mulit_request_test(**kwargs)


class TestChunkedPrefill(CustomTestCase):
    def test_chunked_prefill(self):
        _run_chunked_mmlu_test(disable_radix_cache=False, enable_mixed_chunk=False)

    def test_mixed_chunked_prefill(self):
        _run_chunked_mmlu_test(disable_radix_cache=False, enable_mixed_chunk=True)

    def test_chunked_prefill_without_radix_cache(self):
        _run_chunked_mmlu_test(disable_radix_cache=True, enable_mixed_chunk=False)

    def test_mixed_chunked_prefill_without_radix_cache(self):
        _run_chunked_mmlu_test(disable_radix_cache=True, enable_mixed_chunk=True)

    def test_mixed_chunked_prefill_multi_requests(self):
        _run_chunked_multi_request_test(
            enable_mixed_chunk=True,
            chunked_prefill_size=2048,
        )


if __name__ == "__main__":
    unittest.main()
