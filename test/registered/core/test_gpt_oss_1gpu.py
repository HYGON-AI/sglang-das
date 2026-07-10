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

import json
import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
from sglang.test.gpt_oss_common import BaseTestGptOss
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
)

register_cuda_ci(est_time=519, suite="stage-b-test-1-gpu-large")
register_amd_ci(est_time=750, suite="stage-b-test-1-gpu-small-amd-mi35x")


register_dcu_ci(
    est_time=120,
    suite="stage-b-test-1-gpu-small-dcu",
    disabled="DCU unresolved: local GPT-OSS mxfp4 config fails at startup with Unknown quantization method: mxfp4.",
)


DCU_GPT_OSS_20B = "/public/opendas/DL_DATA/llm-models/gpt-oss/gpt-oss-20b"


def _is_dcu() -> bool:
    return os.environ.get("SGLANG_IS_IN_CI_DCU", "0") == "1"


def _dcu_url() -> str:
    return f"http://127.0.0.1:{find_available_port(11001)}"


class TestGptOss1GpuDcuSmoke(CustomTestCase):
    @unittest.skip("DCU GPT-OSS mxfp4 is not supported by current quant registry.")
    def test_mxfp4_20b_responses_stream(self):
        base_url = _dcu_url()
        process = popen_launch_server(
            DCU_GPT_OSS_20B,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "fa3",
                "--page-size",
                "64",
                "--max-total-tokens",
                "512",
                "--disable-cuda-graph",
                "--disable-radix-cache",
                "--trust-remote-code",
            ],
        )
        try:
            response = requests.post(
                f"{base_url}/v1/responses",
                json={
                    "model": DCU_GPT_OSS_20B,
                    "input": "What is 1 + 1?",
                    "stream": True,
                    "temperature": 0,
                    "max_output_tokens": 32,
                },
                stream=True,
                timeout=120,
            )
            if response.status_code != 200:
                print(f"Response API failed: {response.text}")
            response.raise_for_status()

            content = ""
            for line in response.iter_lines():
                if not line:
                    continue
                decoded_line = line.decode("utf-8")
                if not decoded_line.startswith("data: "):
                    continue
                data_str = decoded_line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "response.output_text.delta":
                    content += data.get("delta", "")

            print(f"Streaming check response: {content}")
            self.assertTrue(content.strip())
        finally:
            kill_process_tree(process.pid)


@unittest.skipIf(_is_dcu(), "DCU uses TestGptOss1GpuDcuSmoke lightweight coverage.")
class TestGptOss1Gpu(BaseTestGptOss):
    def test_mxfp4_20b(self):
        self.run_test(
            model_variant="20b",
            quantization="mxfp4",
            expected_score_of_reasoning_effort={
                "low": 0.34,
                "medium": 0.34,
                "high": 0.27,  # TODO investigate
            },
        )

    def test_bf16_20b(self):
        self.run_test(
            model_variant="20b",
            quantization="bf16",
            expected_score_of_reasoning_effort={
                "low": 0.34,
                "medium": 0.34,
                "high": 0.27,  # TODO investigate
            },
        )


if __name__ == "__main__":
    unittest.main()
