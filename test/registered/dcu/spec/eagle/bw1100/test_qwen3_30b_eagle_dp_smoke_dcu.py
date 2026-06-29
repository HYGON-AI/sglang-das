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

register_dcu_ci(est_time=1800, suite="nightly-dcu-quant-opt", nightly=True)

DEFAULT_QWEN3_MOE_MODEL = "Qwen/Qwen3-30B-A3B"
DEFAULT_QWEN3_MOE_EAGLE_DRAFT = "Tengyunw/qwen3_30b_moe_eagle3"


def _default_eagle_args() -> list[str]:
    draft_model = get_model_path(
        "SGLANG_DCU_QWEN3_MOE_EAGLE_DRAFT",
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


class TestBW1100Qwen3ThirtyBEagleDPSmokeDCU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = get_model_path("SGLANG_DCU_QWEN3_MOE_MODEL", DEFAULT_QWEN3_MOE_MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=get_int_env("SGLANG_DCU_QWEN3_EAGLE_DP_TIMEOUT", 1800),
            api_key=cls.api_key,
            other_args=get_server_args(
                "SGLANG_DCU_QWEN3_EAGLE_DP_ARGS",
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
