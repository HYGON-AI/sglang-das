import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_dcu_ci
from sglang.test.dcu_utils import (
    DCU_MOE_SERVER_ARGS,
    assert_generate_non_empty,
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

DEFAULT_QWEN3_MOE_MODEL = "/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-30B-A3B"


class TestBW1000Qwen3ThirtyBMoEServerDCU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = get_model_path("SGLANG_DCU_QWEN3_MOE_MODEL", DEFAULT_QWEN3_MOE_MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=get_int_env("SGLANG_DCU_QWEN3_MOE_TIMEOUT", 3600),
            other_args=get_server_args(
                "SGLANG_DCU_MOE_SERVER_ARGS", DCU_MOE_SERVER_ARGS
            ),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_short_generate(self):
        content = assert_generate_non_empty(self.base_url, "The capital of China is")
        self.assertGreater(len(content.strip()), 0)


if __name__ == "__main__":
    unittest.main()
