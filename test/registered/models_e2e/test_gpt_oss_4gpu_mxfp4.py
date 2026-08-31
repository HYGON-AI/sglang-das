# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Hygon modifications to this file are licensed under the Apache License,
# Version 2.0 (the "License"); you may not use these modifications except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from sglang.test.ci.ci_register import register_cuda_ci, register_hcu_ci

# HCU_CSV_CI_UNVERIFIED: Registered from sglang.csv CI coverage; not re-tested in this framework pass.
register_hcu_ci(
    est_time=300,
    suite="nightly-hcu-4",
    nightly=True,
    disabled="HCU CSV CI placeholder: 4-GPU GPT-OSS path needs BW1100 large-model validation before enabling.",
)

from sglang.test.gpt_oss_common import BaseTestGptOss

register_cuda_ci(est_time=388, stage="base-c", runner_config="4-gpu-h100")
register_cuda_ci(est_time=121, stage="base-c", runner_config="4-gpu-b200")


class TestGptOss4GpuMxfp4(BaseTestGptOss):
    def test_mxfp4_120b(self):
        self.run_test(
            model_variant="120b",
            quantization="mxfp4",
            expected_score_of_reasoning_effort={
                "low": 0.58,
            },
            other_args=[
                "--tp",
                "4",
                "--cuda-graph-max-bs-decode",
                "200",
            ],
        )


if __name__ == "__main__":
    unittest.main()
