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
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    run_bench_offline_throughput,
    run_bench_one_batch,
    write_github_step_summary,
)

register_cuda_ci(est_time=120, suite="stage-b-test-1-gpu-large")
register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-large-amd")
register_hcu_ci(est_time=120, suite="nightly-hcu-perf", nightly=True, disabled='HCU Full Enabled run 26941698027 failed; keep disabled until BW1100 failure is fixed or revalidated.')


class TestBenchOneBatch1GPU(CustomTestCase):

    def test_bs1_small(self):
        _, output_throughput, _ = run_bench_one_batch(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST, ["--cuda-graph-max-bs", "2"]
        )
        self.assertGreater(output_throughput, 50)

    def test_bs1_default(self):
        output_throughput = run_bench_offline_throughput(
            DEFAULT_MODEL_NAME_FOR_TEST, ["--cuda-graph-max-bs", "2"]
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs1_default (llama-3.1-8b)\n"
                f"output_throughput: {output_throughput:.2f} token/s\n"
            )
            self.assertGreater(output_throughput, 135)


if __name__ == "__main__":
    unittest.main()
