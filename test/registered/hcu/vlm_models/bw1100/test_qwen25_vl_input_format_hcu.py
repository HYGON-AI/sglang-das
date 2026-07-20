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

from sglang import Engine
from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.hcu_utils import RED_DOT_IMAGE_DATA_URL, get_model_path

register_hcu_ci(est_time=1800, suite="stage-b-test-1-gpu-small-hcu")

DEFAULT_QWEN25_VL_3B_MODEL = (
    "/public/opendas/DL_DATA/llm-models/vllm-gptq-models/qwen2.5/"
    "Qwen2.5-VL-3B-Instruct"
)


class TestBW1100Qwen25VLInputFormatHCU(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = get_model_path(
            "SGLANG_HCU_QWEN25_VL_3B_MODEL", DEFAULT_QWEN25_VL_3B_MODEL
        )
        cls.engine = Engine(
            model_path=cls.model,
            attention_backend="fa3",
            mm_attention_backend="fa3",
            page_size=64,
            trust_remote_code=True,
            enable_multimodal=True,
            disable_cuda_graph=True,
            log_level="warning",
        )

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    async def test_accepts_image_data_url(self):
        output = await self.engine.async_generate(
            prompt="What is the dominant color in this image?",
            image_data=RED_DOT_IMAGE_DATA_URL,
            sampling_params={"temperature": 0, "max_new_tokens": 16},
        )
        text = output["text"].strip()
        self.assertGreater(len(text), 0)


if __name__ == "__main__":
    unittest.main()
