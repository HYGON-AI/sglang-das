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

import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_hcu_ci
from sglang.test.test_utils import CustomTestCase, run_mmlu_test, run_mulit_request_test

register_cuda_ci(est_time=312, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=312, suite="stage-b-test-1-gpu-small-amd")
register_hcu_ci(est_time=312, suite="stage-b-test-1-gpu-small-hcu", disabled='HCU Full Enabled run 26941698027 failed; keep disabled until BW1100 failure is fixed or revalidated.')


class TestChunkedPrefill(CustomTestCase):
    def test_chunked_prefill(self):
        run_mmlu_test(disable_radix_cache=False, enable_mixed_chunk=False)

    def test_mixed_chunked_prefill(self):
        run_mmlu_test(disable_radix_cache=False, enable_mixed_chunk=True)

    def test_chunked_prefill_without_radix_cache(self):
        run_mmlu_test(disable_radix_cache=True, enable_mixed_chunk=False)

    def test_mixed_chunked_prefill_without_radix_cache(self):
        run_mmlu_test(disable_radix_cache=True, enable_mixed_chunk=True)

    def test_mixed_chunked_prefill_multi_requests(self):
        run_mulit_request_test(
            enable_mixed_chunk=True,
            chunked_prefill_size=2048,
        )


if __name__ == "__main__":
    unittest.main()
