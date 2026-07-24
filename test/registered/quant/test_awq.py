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
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_hcu_ci

# HCU_CSV_COVERED_UNVERIFIED: Enabled from sglang.csv historical HCU coverage; not re-tested in this framework pass.
register_hcu_ci(
    est_time=120,
    suite="stage-b-test-1-gpu-small-hcu",
    disabled="HCU PR baseline deferred: quantization path needs BW1100 numeric/backend validation before required CI.",
)

from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_AWQ_MOE_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=160, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=200, suite="stage-b-test-1-gpu-large-amd")


class TestAWQ(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_AWQ_MOE_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--trust-remote-code"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.64)


@unittest.skipIf(is_in_amd_ci(), "AWQ Marlin is not supported on AMD GPUs")
class TestAWQMarlinBfloat16(CustomTestCase):
    """
    Verify that the model can be loaded with bfloat16 dtype and awq_marlin quantization
    """

    @classmethod
    def setUpClass(cls):
        cls.model = "QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--dtype", "bfloat16", "--quantization", "awq_marlin"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.83)


if __name__ == "__main__":
    unittest.main()
