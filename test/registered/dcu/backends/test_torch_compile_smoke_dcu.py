import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_dcu_ci

register_dcu_ci(
    est_time=1800,
    suite="stage-b-test-1-gpu-small-dcu",
    disabled="DCU disabled retest: torch-compile small-model smoke starts compilation but exceeds the 600s server launch timeout; keep as nightly/专项 candidate before enabling.",
)

from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)



class TestTorchCompileDCU(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--enable-torch-compile", "--cuda-graph-max-bs", "4", "--attention-backend", "fa3", "--page-size", "64", "--trust-remote-code"],
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

    def test_torch_compile_generate(self):
        res = self.run_decode(16)
        self.assertIn("text", res)
        self.assertTrue(res["text"])


if __name__ == "__main__":
    unittest.main()
