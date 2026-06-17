import unittest

from sglang import Engine
from sglang.test.ci.ci_register import register_dcu_ci
from sglang.test.dcu_utils import RED_DOT_IMAGE_DATA_URL, get_model_path

register_dcu_ci(est_time=1800, suite="stage-b-test-1-gpu-small-dcu")

DEFAULT_QWEN25_VL_3B_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


class TestBW1100Qwen25VLInputFormatDCU(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = get_model_path(
            "SGLANG_DCU_QWEN25_VL_3B_MODEL", DEFAULT_QWEN25_VL_3B_MODEL
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
