import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_dcu_ci
from sglang.test.dcu_utils import (
    DCU_TEXT_SERVER_ARGS,
    assert_chat_completion,
    assert_generate_non_empty,
    get_int_env,
    get_model_path,
    get_server_args,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

register_dcu_ci(est_time=1200, suite="stage-b-test-1-gpu-small-dcu")


class TestBW1100Qwen25SevenBServerDCU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = get_model_path("SGLANG_DCU_QWEN25_7B_MODEL", DEFAULT_MODEL_NAME_FOR_TEST)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=get_int_env("SGLANG_DCU_QWEN25_7B_TIMEOUT", 1800),
            api_key=cls.api_key,
            other_args=get_server_args("SGLANG_DCU_SERVER_ARGS", DCU_TEXT_SERVER_ARGS),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_chat_completion(self):
        content = assert_chat_completion(
            self.base_url,
            self.api_key,
            self.model,
            [{"role": "user", "content": "Write one sentence about DCU CI."}],
        )
        self.assertGreater(len(content.strip()), 0)

    def test_generate(self):
        content = assert_generate_non_empty(
            self.base_url, "SGLang is", api_key=self.api_key
        )
        self.assertGreater(len(content.strip()), 0)


if __name__ == "__main__":
    unittest.main()
