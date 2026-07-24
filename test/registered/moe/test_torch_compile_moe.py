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

import time
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import is_cuda, kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_hcu_ci

# HCU_CSV_COVERED_UNVERIFIED: Enabled from sglang.csv historical HCU coverage; not re-tested in this framework pass.
register_hcu_ci(
    est_time=1400,
    suite="stage-b-test-1-gpu-small-hcu",
    disabled="HCU PR baseline deferred: MoE path needs local model/backend validation before required CI.",
)

from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_BASE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=130, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=1400, suite="stage-b-test-1-gpu-small-amd")


class TestTorchCompileMoe(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_BASE
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--enable-torch-compile", "--torch-compile-max-bs", "4"],
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
        self.assertGreaterEqual(metrics["score"], 0.50)

    def run_decode(self, max_new_tokens):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
        )
        return response.json()

    def test_throughput(self):
        # Warmup
        res = self.run_decode(16)

        max_tokens = 256
        tic = time.perf_counter()
        res = self.run_decode(max_tokens)
        tok = time.perf_counter()
        print(f"{res=}")
        throughput = max_tokens / (tok - tic)
        if is_cuda():
            self.assertGreaterEqual(throughput, 285)
        elif is_in_amd_ci():
            # relax for mi300x
            self.assertGreaterEqual(throughput, 240)
        else:
            self.assertGreaterEqual(throughput, 270)


if __name__ == "__main__":
    unittest.main()
