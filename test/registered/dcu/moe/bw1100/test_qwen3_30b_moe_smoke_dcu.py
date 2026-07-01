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

register_dcu_ci(est_time=1200, suite="nightly-dcu-large-model-4gpu", nightly=True)

DEFAULT_QWEN3_MOE_MODEL = "/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-30B-A3B"
DEFAULT_QWEN3_MOE_SMOKE_ARGS = [
    "--tp-size",
    "2",
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


class TestBW1100Qwen3ThirtyBMoESmokeDCU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = get_model_path("SGLANG_DCU_QWEN3_MOE_MODEL", DEFAULT_QWEN3_MOE_MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=get_int_env("SGLANG_DCU_QWEN3_MOE_TIMEOUT", 1200),
            api_key=cls.api_key,
            other_args=get_server_args(
                "SGLANG_DCU_QWEN3_MOE_SMOKE_ARGS",
                DEFAULT_QWEN3_MOE_SMOKE_ARGS,
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
