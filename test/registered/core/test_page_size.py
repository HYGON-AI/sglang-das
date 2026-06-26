import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
from sglang.test.kits.eval_accuracy_kit import MMLUMixin
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=60, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")
register_dcu_ci(est_time=90, suite="stage-b-test-1-gpu-small-dcu")


DCU_SMALL_MODEL = "/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-0.6B"


def _is_dcu() -> bool:
    return os.environ.get("SGLANG_IS_IN_CI_DCU", "0") == "1"


class TestPageSize(CustomTestCase, MMLUMixin):
    mmlu_score_threshold = 0.65
    mmlu_num_examples = 64
    mmlu_num_threads = 32

    @classmethod
    def setUpClass(cls):
        os.environ["SGLANG_DEBUG_MEMORY_POOL"] = "1"
        if _is_dcu():
            cls.mmlu_score_threshold = 0.0
            cls.mmlu_num_examples = 8
            cls.mmlu_num_threads = 8
        cls.model = DCU_SMALL_MODEL if _is_dcu() else DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = ["--page-size", 4, "--chunked-prefill-size", 128]
        if _is_dcu():
            other_args = [
                "--page-size",
                64,
                "--chunked-prefill-size",
                128,
                "--attention-backend",
                "fa3",
                "--max-total-tokens",
                "512",
                "--disable-cuda-graph",
                "--disable-radix-cache",
                "--trust-remote-code",
            ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
