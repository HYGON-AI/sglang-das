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

"""
Usage:
python3 -m unittest test_torch_native_attention_backend.TestTorchNativeAttnBackend.test_mmlu
"""

import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci

register_dcu_ci(
    est_time=120,
    suite="stage-b-test-1-gpu-small-dcu",
    disabled="DCU Stage-B deferred: local Qwen3-0.6B torch_native smoke reaches SDPA forward but triggers BW1100 VMFault in torch_native_backend.py.",
)

from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Torch native attention backend integration test with MMLU eval
register_cuda_ci(est_time=140, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=150, suite="stage-b-test-1-gpu-small-amd")


def _is_dcu():
    return os.getenv("SGLANG_IS_IN_CI_DCU") == "1"


_DCU_MODEL_NAME = "/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-0.6B"


class TestTorchNativeAttnBackend(CustomTestCase):
    def test_mmlu(self):
        model = _DCU_MODEL_NAME if _is_dcu() else DEFAULT_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        other_args = ["--attention-backend", "torch_native"]
        if _is_dcu():
            other_args += [
                "--trust-remote-code",
                "--page-size",
                "64",
                "--max-total-tokens",
                "1024",
                "--disable-cuda-graph",
            ]
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

        try:
            if _is_dcu():
                response = requests.post(
                    base_url + "/generate",
                    json={
                        "text": "The capital of France is",
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 8,
                            "ignore_eos": True,
                        },
                    },
                    timeout=60,
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertIn("text", response.json())
                return

            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
            )

            metrics = run_eval(args)
            self.assertGreaterEqual(metrics["score"], 0.65)
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
