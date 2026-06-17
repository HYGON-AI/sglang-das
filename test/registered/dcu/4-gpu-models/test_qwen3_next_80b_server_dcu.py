import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_dcu_ci
from sglang.test.dcu_utils import (
    DCU_TEXT_SERVER_ARGS,
    assert_generate_non_empty,
    get_int_env,
    get_model_path,
    get_server_args,
)
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, popen_launch_server

register_dcu_ci(est_time=2400, suite="nightly-dcu-4-gpu", nightly=True)

DEFAULT_QWEN3_NEXT_MODEL = (
    "/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-Next-80B-A3B-Instruct"
)
DEFAULT_QWEN3_NEXT_ARGS = DCU_TEXT_SERVER_ARGS + [
    "--tp-size",
    "4",
    "--chunked-prefill-size",
    "2048",
    "--mamba-scheduler-strategy",
    "extra_buffer",
    "--mamba-track-interval",
    "128",
    "--max-running-requests",
    "4",
    "--disable-custom-all-reduce",
]


class TestQwen3Next80BServerDCU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = get_model_path(
            "SGLANG_DCU_QWEN3_NEXT_MODEL", DEFAULT_QWEN3_NEXT_MODEL
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=get_int_env("SGLANG_DCU_QWEN3_NEXT_TIMEOUT", 3600),
            api_key=cls.api_key,
            other_args=get_server_args(
                "SGLANG_DCU_QWEN3_NEXT_SERVER_ARGS", DEFAULT_QWEN3_NEXT_ARGS
            ),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_short_generate(self):
        content = assert_generate_non_empty(
            self.base_url,
            "The capital of China is",
            max_new_tokens=16,
            api_key=self.api_key,
        )
        self.assertGreater(len(content.strip()), 0)


if __name__ == "__main__":
    unittest.main()
