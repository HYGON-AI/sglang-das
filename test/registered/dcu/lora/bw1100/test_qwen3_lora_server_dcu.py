import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_dcu_ci
from sglang.test.dcu_utils import (
    assert_generate_non_empty,
    get_int_env,
    get_model_path,
    get_server_args,
)
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

register_dcu_ci(est_time=900, suite="nightly-dcu", nightly=True)

DEFAULT_QWEN3_4B_MODEL = "Qwen/Qwen3-4B"
DEFAULT_QWEN3_LORA_1 = "nissenj/Qwen3-4B-lora-v2"
DEFAULT_QWEN3_LORA_2 = "TanXS/Qwen3-4B-LoRA-ZH-WebNovelty-v0.0"


def _default_lora_args() -> list[str]:
    lora_1 = get_model_path("SGLANG_DCU_QWEN3_LORA_1", DEFAULT_QWEN3_LORA_1)
    lora_2 = get_model_path("SGLANG_DCU_QWEN3_LORA_2", DEFAULT_QWEN3_LORA_2)
    return [
        "--lora-paths",
        lora_1,
        lora_2,
        "--max-loras-per-batch",
        "3",
        "--max-loaded-loras",
        "3",
        "--attention-backend",
        "fa3",
        "--page-size",
        "64",
        "--trust-remote-code",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        "0.60",
        "--log-level",
        "warning",
        "--log-level-http",
        "warning",
    ]


class TestBW1100Qwen3LoRAServerDCU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = get_model_path("SGLANG_DCU_QWEN3_4B_MODEL", DEFAULT_QWEN3_4B_MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=get_int_env("SGLANG_DCU_QWEN3_LORA_TIMEOUT", 900),
            api_key=cls.api_key,
            other_args=get_server_args(
                "SGLANG_DCU_QWEN3_LORA_SERVER_ARGS",
                _default_lora_args(),
            ),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_short_generate_with_lora_loaded(self):
        content = assert_generate_non_empty(
            self.base_url, "The capital of China is", api_key=self.api_key
        )
        self.assertGreater(len(content.strip()), 0)


if __name__ == "__main__":
    unittest.main()
