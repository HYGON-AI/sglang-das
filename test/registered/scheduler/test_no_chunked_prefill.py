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

import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_hcu_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    run_bench_serving,
    run_mmlu_test,
)

register_cuda_ci(est_time=108, suite="stage-b-test-1-gpu-large")
register_amd_ci(est_time=108, suite="stage-b-test-1-gpu-small-amd")
register_hcu_ci(est_time=108, suite="stage-b-test-1-gpu-small-hcu")
register_hcu_ci(est_time=108, suite="nightly-hcu-functional-long", nightly=True)


class TestNoChunkedPrefill(CustomTestCase):

    def test_no_chunked_prefill(self):
        run_mmlu_test(
            disable_radix_cache=False, enable_mixed_chunk=False, chunked_prefill_size=-1
        )

    def test_no_chunked_prefill_without_radix_cache(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=10,
            request_rate=float("inf"),
            other_server_args=["--disable-radix-cache", "--chunked-prefill-size", "-1"],
        )

        assert res["completed"] == 10


if __name__ == "__main__":
    unittest.main()
