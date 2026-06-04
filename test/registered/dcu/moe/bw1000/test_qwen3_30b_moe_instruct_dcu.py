import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_dcu_ci
from sglang.test.dcu_utils import (
    DCU_MOE_SERVER_ARGS,
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

register_dcu_ci(est_time=2400, suite="nightly-dcu", nightly=True)

DEFAULT_QWEN3_MOE_INSTRUCT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


class TestBW1000Qwen3ThirtyBMoEInstructDCU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = get_model_path(
            "SGLANG_DCU_QWEN3_MOE_INSTRUCT_MODEL",
            DEFAULT_QWEN3_MOE_INSTRUCT_MODEL,
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=get_int_env("SGLANG_DCU_QWEN3_MOE_INSTRUCT_TIMEOUT", 3600),
            api_key=cls.api_key,
            other_args=get_server_args(
                "SGLANG_DCU_MOE_SERVER_ARGS", DCU_MOE_SERVER_ARGS
            ),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_short_chat_completion(self):
        content = assert_chat_completion(
            self.base_url,
            self.api_key,
            self.model,
            [{"role": "user", "content": "Answer in five words: what is MoE?"}],
        )
        self.assertGreater(len(content.strip()), 0)


if __name__ == "__main__":
    unittest.main()
