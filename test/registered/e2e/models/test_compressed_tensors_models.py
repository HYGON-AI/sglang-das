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

# Model tests for compressed tensors (FP8)

import unittest
from types import SimpleNamespace

from sglang.srt.utils import is_hip, kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_hcu_ci

# HCU_CSV_COVERED_UNVERIFIED: Enabled from sglang.csv historical HCU coverage; not re-tested in this framework pass.
register_hcu_ci(
    est_time=42,
    suite="stage-b-test-1-hcu-small",
    disabled="HCU PR baseline deferred: model matrix path needs local model mapping and BW1100 repeat validation.",
)

from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=64, stage="extra-a", runner_config="1-gpu-large")
register_amd_ci(est_time=42, suite="stage-b-test-1-gpu-small-amd")


class TestCompressedTensorsLlama3FP8(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "RedHatAI/Meta-Llama-3.1-8B-FP8"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        if is_hip():
            # Lower threshold for AMD because FP8 dtype differs (fp8_fnuz)
            self.assertGreaterEqual(metrics["score"], 0.40)
        else:
            self.assertGreaterEqual(metrics["score"], 0.45)


if __name__ == "__main__":
    unittest.main()
