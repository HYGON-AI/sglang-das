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
    assert_generate_non_empty,
    get_int_env,
    get_model_path,
    get_server_args,
)
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

register_hcu_ci(est_time=1800, suite="nightly-hcu-quant-opt", nightly=True)

DEFAULT_QWEN3_MOE_MODEL = "/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-30B-A3B"
DEFAULT_QWEN3_MOE_EAGLE_DRAFT = "/public/opendas/DL_DATA/llm-models/eagle/qwen3_30b_moe_eagle3"


def _default_eagle_args() -> list[str]:
    draft_model = get_model_path(
        "SGLANG_HCU_QWEN3_MOE_EAGLE_DRAFT",
        DEFAULT_QWEN3_MOE_EAGLE_DRAFT,
    )
    return [
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-num-steps",
        "2",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "3",
        "--speculative-draft-model-path",
        draft_model,
        "--tp-size",
        "2",
        "--dp-size",
        "2",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--moe-dense-tp-size",
        "1",
        "--attention-backend",
        "fa3",
        "--page-size",
        "64",
        "--trust-remote-code",
        "--disable-cuda-graph",
        "--disable-custom-all-reduce",
        "--cuda-graph-max-bs",
        "16",
        "--mem-fraction-static",
        "0.55",
        "--log-level",
        "warning",
        "--log-level-http",
        "warning",
    ]


class TestBW1100Qwen3ThirtyBEagleDPSmokeHCU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = get_model_path("SGLANG_HCU_QWEN3_MOE_MODEL", DEFAULT_QWEN3_MOE_MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=get_int_env("SGLANG_HCU_QWEN3_EAGLE_DP_TIMEOUT", 1800),
            api_key=cls.api_key,
            other_args=get_server_args(
                "SGLANG_HCU_QWEN3_EAGLE_DP_ARGS",
                _default_eagle_args(),
            ),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_short_generate(self):
        content = assert_generate_non_empty(
            self.base_url, "The capital of China is", api_key=self.api_key
        )
        self.assertGreater(len(content.strip()), 0)


if __name__ == "__main__":
    unittest.main()
