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
    assert_rerank_scores,
    get_model_path,
    get_server_args,
    repo_root_from_test_file,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

register_hcu_ci(est_time=1200, suite="stage-b-test-1-gpu-small-hcu")
register_hcu_ci(est_time=1200, suite="nightly-hcu-api-models", nightly=True)

DEFAULT_HCU_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"


def _default_reranker_args() -> list[str]:
    repo_root = repo_root_from_test_file(__file__)
    template = repo_root / "examples" / "chat_template" / "qwen3_reranker.jinja"
    if not template.exists():
        raise unittest.SkipTest(f"Qwen3 reranker chat template is missing: {template}")
    return HCU_TEXT_SERVER_ARGS + ["--disable-cuda-graph", "--chat-template", str(template)]


class TestBW1100Qwen3RerankerHCU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = get_model_path(
            "SGLANG_HCU_RERANKER_MODEL", DEFAULT_HCU_RERANKER_MODEL
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=get_server_args(
                "SGLANG_HCU_RERANKER_SERVER_ARGS", _default_reranker_args()
            ),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_rerank_scores(self):
        response = assert_rerank_scores(
            self.base_url,
            self.api_key,
            "What is the capital of France?",
            ["Paris is the capital of France.", "The sun is a star."],
        )
        self.assertGreaterEqual(len(response), 1)


if __name__ == "__main__":
    unittest.main()
