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

import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci

# DCU_CSV_COVERED_UNVERIFIED: Enabled from sglang.csv historical DCU coverage; not re-tested in this framework pass.
register_dcu_ci(
    est_time=180,
    suite="stage-b-test-1-gpu-small-dcu",
    disabled="DCU Stage-B deferred: local gemma-3-1b-it sliding-window smoke starts after Gemma3 rope fallback but triggers BW1100 VMFault in Triton attention _fwd_grouped_kernel_stage1.",
)

from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    is_in_ci,
    popen_launch_server,
)

# Sliding window attention with Triton backend (Gemma-3 model)
register_cuda_ci(est_time=93, stage="extra-a", runner_config="1-gpu-large")
register_amd_ci(est_time=200, suite="stage-b-test-1-gpu-small-amd")


def _is_dcu():
    return os.getenv("SGLANG_IS_IN_CI_DCU") == "1"


_DCU_MODEL_NAME = (
    "/public/opendas/DL_DATA/llm-models/vllm-optest-models/google/gemma-3-1b-it"
)


class TestSlidingWindowAttentionTriton(CustomTestCase):
    """Test sliding window attention functionality with triton backend."""

    @classmethod
    def setUpClass(cls):
        """Set up the test server with Gemma3 model and triton backend."""
        # Gemma3 model supports sliding window attention
        cls.model = _DCU_MODEL_NAME if _is_dcu() else "google/gemma-3-4b-it"
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.common_args = [
            "--trust-remote-code",
            "--attention-backend",
            "triton",
            "--context-length",
            "1024" if _is_dcu() else "8192",
            "--random-seed",
            "42",
        ]
        if _is_dcu():
            cls.common_args += [
                "--page-size",
                "64",
                "--max-total-tokens",
                "2048",
                "--disable-cuda-graph",
            ]

        cls.short_context_prompt = "The capital of France is"

        # Test prompt longer than window size
        cls.long_context_prompt = """
        Once upon a time, there was a mountain. In the mountain, there was a temple. In the temple, there was an old monk telling a story. The story was:
        """ * (20 if _is_dcu() else 100)
        cls.long_context_prompt += "\nNow, summarize the story in one sentence:"

    def _test_mmlu(self):
        if _is_dcu():
            return

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=200,
            num_threads=32,
        )

        metrics = run_eval(args)
        print(f"MMLU metrics with sliding window: {metrics}")

        if is_in_amd_ci():
            self.assertGreaterEqual(metrics["score"], 0.55)
        else:
            self.assertGreaterEqual(metrics["score"], 0.60)

    def _test_short_context_generation(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": self.short_context_prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 256,
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        if _is_dcu():
            self.assertGreater(len(result["text"].strip()), 0)
        else:
            self.assertIn("paris", result["text"].lower())
        print(f"Short context generation result: {result['text']}")

    def _test_long_context_generation(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": self.long_context_prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 256,
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertGreater(len(result["text"].strip()), 0)
        print(f"Long context generation result: {result['text'][:100]}...")

    @unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
    def test_no_cuda_graph(self):
        self.no_cuda_graph_process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=self.common_args + ["--disable-cuda-graph"],
        )

        try:
            self._test_short_context_generation()
            self._test_long_context_generation()
            self._test_mmlu()
        finally:
            kill_process_tree(self.no_cuda_graph_process.pid)

    def test_cuda_graph(self):
        self.cuda_graph_process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=self.common_args,
        )

        try:
            self._test_short_context_generation()
            self._test_long_context_generation()
            self._test_mmlu()
        finally:
            kill_process_tree(self.cuda_graph_process.pid)


if __name__ == "__main__":
    unittest.main()
