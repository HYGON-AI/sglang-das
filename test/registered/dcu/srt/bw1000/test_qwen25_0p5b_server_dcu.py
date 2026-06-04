import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_dcu_ci
from sglang.test.dcu_utils import (
    DCU_TEXT_SERVER_ARGS,
    assert_chat_completion,
    get_server_args,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    find_available_port,
    popen_launch_server,
)

register_dcu_ci(est_time=600, suite="stage-b-test-1-gpu-small-dcu")

DEFAULT_QWEN25_0P5B_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _get_model_path() -> str:
    model_path = os.environ.get(
        "SGLANG_DCU_QWEN25_0P5B_MODEL",
        os.environ.get("SGLANG_DCU_SERVER_SMOKE_MODEL", DEFAULT_QWEN25_0P5B_MODEL),
    )
    if model_path.startswith(("/", ".")) and not os.path.isdir(model_path):
        if (
            "SGLANG_DCU_QWEN25_0P5B_MODEL" in os.environ
            or "SGLANG_DCU_SERVER_SMOKE_MODEL" in os.environ
        ):
            raise FileNotFoundError(f"DCU Qwen2.5-0.5B model path missing: {model_path}")
        raise unittest.SkipTest(f"Default DCU model path does not exist: {model_path}")
    return model_path


class TestBW1000Qwen25HalfBServerDCU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _get_model_path()
        cls.api_key = "sk-123456"
        port = find_available_port(11001)
        cls.base_url = f"http://127.0.0.1:{port}"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=get_server_args("SGLANG_DCU_SERVER_ARGS", DCU_TEXT_SERVER_ARGS),
            env={
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                "SGLANG_USE_MODELSCOPE": os.environ.get("SGLANG_USE_MODELSCOPE", "1"),
                "SGLANG_USE_LIGHTOP": os.environ.get("SGLANG_USE_LIGHTOP", "1"),
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_health(self):
        response = requests.get(f"{self.base_url}/health", timeout=10)
        self.assertEqual(response.status_code, 200)

    def test_models(self):
        response = requests.get(
            f"{self.base_url}/v1/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=10,
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertTrue(any(model["id"] == self.model for model in data))

    def test_chat_completion(self):
        content = assert_chat_completion(
            self.base_url,
            self.api_key,
            self.model,
            [{"role": "user", "content": "Hello, reply with one short sentence."}],
            max_tokens=32,
        )
        self.assertGreater(len(content.strip()), 0)


if __name__ == "__main__":
    unittest.main()
