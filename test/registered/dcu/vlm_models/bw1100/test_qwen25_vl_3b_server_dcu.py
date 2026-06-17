import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_dcu_ci
from sglang.test.dcu_utils import (
    DCU_VLM_SERVER_ARGS,
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

register_dcu_ci(est_time=1800, suite="stage-b-test-1-gpu-small-dcu")

DEFAULT_QWEN25_VL_3B_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


class TestBW1100Qwen25VLThreeBServerDCU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = get_model_path(
            "SGLANG_DCU_QWEN25_VL_3B_MODEL", DEFAULT_QWEN25_VL_3B_MODEL
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=get_int_env("SGLANG_DCU_QWEN25_VL_3B_TIMEOUT", 1800),
            api_key=cls.api_key,
            other_args=get_server_args(
                "SGLANG_DCU_VLM_SERVER_ARGS", DCU_VLM_SERVER_ARGS
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
