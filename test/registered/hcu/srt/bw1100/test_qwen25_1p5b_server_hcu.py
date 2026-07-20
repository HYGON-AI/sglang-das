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
    HCU_TEXT_SERVER_ARGS,
    assert_chat_completion,
    assert_generate_non_empty,
    get_model_path,
    get_server_args,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

register_hcu_ci(est_time=900, suite="stage-b-test-1-gpu-small-hcu")

DEFAULT_QWEN25_1P5B_MODEL = "/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-1.5B-Instruct"


class TestBW1100Qwen25OnePointFiveBServerHCU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = get_model_path(
            "SGLANG_HCU_QWEN25_1P5B_MODEL", DEFAULT_QWEN25_1P5B_MODEL
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=get_server_args("SGLANG_HCU_SERVER_ARGS", HCU_TEXT_SERVER_ARGS),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_chat_completion(self):
        content = assert_chat_completion(
            self.base_url,
            self.api_key,
            self.model,
            [{"role": "user", "content": "Say hello in one short sentence."}],
        )
        self.assertGreater(len(content.strip()), 0)

    def test_generate(self):
        content = assert_generate_non_empty(self.base_url, api_key=self.api_key)
        self.assertGreater(len(content.strip()), 0)


if __name__ == "__main__":
    unittest.main()
