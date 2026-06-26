import os
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci


def _is_dcu():
    return os.getenv("SGLANG_IS_IN_CI_DCU") == "1"


register_dcu_ci(
    est_time=300,
    suite="stage-b-test-1-gpu-small-dcu",
)

from sglang.test.kits.eval_accuracy_kit import MMLUMixin
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    find_available_port,
    is_in_amd_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=126, stage="extra-a", runner_config="1-gpu-large")
register_amd_ci(est_time=1100, suite="stage-b-test-1-gpu-small-amd")

_DCU_MODEL_NAME = "/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-0.6B"


class TestTorchCompile(CustomTestCase, MMLUMixin):
    mmlu_score_threshold = 0.65
    mmlu_num_examples = 64
    mmlu_num_threads = 32

    @classmethod
    def setUpClass(cls):
        cls.model = _DCU_MODEL_NAME if _is_dcu() else DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = (
            f"http://127.0.0.1:{find_available_port(11001)}"
            if _is_dcu()
            else DEFAULT_URL_FOR_TEST
        )
        dcu_args = []
        if _is_dcu():
            dcu_args = [
                "--attention-backend",
                "fa3",
                "--page-size",
                "64",
                "--trust-remote-code",
                "--max-total-tokens",
                "1024",
                "--disable-cuda-graph",
            ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-torch-compile",
                "--cuda-graph-max-bs",
                "4",
                *dcu_args,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_decode(self, max_new_tokens):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
        )
        return response.json()

    def test_mmlu(self):
        if _is_dcu():
            self.skipTest("DCU CI validates torch-compile serving with a short decode smoke; MMLU is too long for this lane.")
        super().test_mmlu()

    def test_throughput(self):
        # Warmup
        res = self.run_decode(16)

        max_tokens = 16 if _is_dcu() else 256
        tic = time.perf_counter()
        res = self.run_decode(max_tokens)
        tok = time.perf_counter()
        print(f"{res=}")
        throughput = max_tokens / (tok - tic)
        print(f"Throughput: {throughput} tokens/s")

        if _is_dcu():
            self.assertIn("text", res)
            self.assertGreater(throughput, 0)
            return

        if is_in_amd_ci():
            self.assertGreaterEqual(throughput, 145)
        else:
            self.assertGreaterEqual(throughput, 152)


if __name__ == "__main__":
    unittest.main()
