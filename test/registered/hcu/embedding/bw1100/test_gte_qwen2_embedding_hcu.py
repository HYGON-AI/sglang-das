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

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.hcu_utils import (
    HCU_EMBEDDING_SERVER_ARGS,
    get_model_path,
    get_server_args,
    openai_base_url,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

register_hcu_ci(est_time=1200, suite="stage-b-test-1-hcu-small")
register_hcu_ci(est_time=1200, suite="nightly-hcu-api-models", nightly=True)

DEFAULT_HCU_EMBEDDING_MODEL = "/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-Embedding-0.6B"


class TestBW1100QwenEmbeddingHCU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = get_model_path(
            "SGLANG_HCU_EMBEDDING_MODEL", DEFAULT_HCU_EMBEDDING_MODEL
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=get_server_args(
                "SGLANG_HCU_EMBEDDING_SERVER_ARGS", HCU_EMBEDDING_SERVER_ARGS
            ),
        )
        cls.client = openai.Client(
            api_key=cls.api_key, base_url=openai_base_url(cls.base_url)
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_embedding_single(self):
        response = self.client.embeddings.create(
            model=self.model, input="HCU embedding smoke test"
        )
        self.assertEqual(len(response.data), 1)
        self.assertGreater(len(response.data[0].embedding), 0)

    def test_embedding_batch(self):
        response = self.client.embeddings.create(
            model=self.model, input=["HCU", "SGLang"]
        )
        self.assertEqual(len(response.data), 2)
        self.assertGreater(len(response.data[0].embedding), 0)
        self.assertGreater(len(response.data[1].embedding), 0)


if __name__ == "__main__":
    unittest.main()
