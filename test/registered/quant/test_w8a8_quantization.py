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

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci, register_hcu_ci

# HCU_CSV_CI_UNVERIFIED: Registered from sglang.csv CI coverage; not re-tested in this framework pass.
register_hcu_ci(
    est_time=120,
    suite="stage-b-test-1-hcu-small",
    nightly=False,
    disabled="HCU CSV CI placeholder: W8A8 quantization path needs BW1100 validation before enabling.",
)

from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=232, stage="extra-a", runner_config="1-gpu-large")


class BaseW8A8Test(CustomTestCase):
    model: str = None
    quantization: str = None
    gsm8k_accuracy_threshold: float = None
    throughput_threshold: float = None

    @classmethod
    def setUpClass(cls):
        if cls is BaseW8A8Test:
            raise unittest.SkipTest("Skip base test class")

        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = []
        if cls.quantization:
            other_args.extend(["--quantization", cls.quantization])

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        if cls is BaseW8A8Test:
            return
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        if self.gsm8k_accuracy_threshold is None:
            self.skipTest("gsm8k_accuracy_threshold not set for this test")

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
        print(metrics)
        self.assertGreater(metrics["score"], self.gsm8k_accuracy_threshold)

    def run_decode(self, max_new_tokens):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
                "ignore_eos": True,
            },
        )
        return response.json()

    def test_throughput(self):

        max_tokens = 256
        tic = time.perf_counter()
        res = self.run_decode(max_tokens)
        tok = time.perf_counter()
        print(res["text"])
        throughput = max_tokens / (tok - tic)
        print(f"Throughput: {throughput} tokens/s")
        self.assertGreaterEqual(throughput, self.throughput_threshold)


class TestW8A8Int8(BaseW8A8Test):
    model = "neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a8"
    quantization = "w8a8_int8"
    gsm8k_accuracy_threshold = 0.69
    throughput_threshold = 200


class TestW8A8Fp8(BaseW8A8Test):
    model = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic"
    quantization = "w8a8_fp8"
    gsm8k_accuracy_threshold = 0.69
    throughput_threshold = 200


class TestW8A8Fp8MoE(BaseW8A8Test):
    model = "RedHatAI/Qwen3-30B-A3B-FP8-dynamic"
    quantization = "w8a8_fp8"
    gsm8k_accuracy_threshold = 0.88
    throughput_threshold = 180


if __name__ == "__main__":
    unittest.main()
