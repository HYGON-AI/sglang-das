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

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.hcu_utils import (
    HCU_VLM_SERVER_ARGS,
    RED_DOT_IMAGE_DATA_URL,
    assert_chat_completion,
    get_int_env,
    get_model_path,
    get_server_args,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

register_hcu_ci(est_time=1800, suite="stage-b-test-1-gpu-small-hcu")

DEFAULT_QWEN25_VL_3B_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def _default_vlm_args() -> list[str]:
    return HCU_VLM_SERVER_ARGS + ["--disable-cuda-graph"]


class TestBW1100Qwen25VLThreeBServerHCU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = get_model_path(
            "SGLANG_HCU_QWEN25_VL_3B_MODEL", DEFAULT_QWEN25_VL_3B_MODEL
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=get_int_env("SGLANG_HCU_QWEN25_VL_3B_TIMEOUT", 1800),
            api_key=cls.api_key,
            other_args=get_server_args(
                "SGLANG_HCU_VLM_SERVER_ARGS", _default_vlm_args()
            ),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_image_chat_completion(self):
        content = assert_chat_completion(
            self.base_url,
            self.api_key,
            self.model,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is the dominant color?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": RED_DOT_IMAGE_DATA_URL},
                        },
                    ],
                }
            ],
            max_tokens=16,
        )
        self.assertGreater(len(content.strip()), 0)


if __name__ == "__main__":
    unittest.main()
