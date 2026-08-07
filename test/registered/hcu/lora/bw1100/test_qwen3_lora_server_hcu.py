# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.hcu_server_guard import HcuServerGuard
from sglang.test.hcu_utils import get_int_env, get_model_path, get_server_args
from sglang.test.test_utils import find_available_port

register_hcu_ci(est_time=900, suite="nightly-hcu-api-models", nightly=True)
register_hcu_ci(est_time=180, suite="stage-b-test-1-hcu-small")

DEFAULT_QWEN3_4B_MODEL = "/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-4B"
DEFAULT_QWEN3_LORA_1 = (
    "/public/opendas/DL_DATA/llm-models/lora/nissenj/Qwen3-4B-lora-v2"
)
DEFAULT_QWEN3_LORA_2 = (
    "/public/opendas/DL_DATA/llm-models/lora/TanXS/"
    "Qwen3-4B-LoRA-ZH-WebNovelty-v0.0"
)
LORA_NAME_1 = "qwen3-lora-v2"
LORA_NAME_2 = "qwen3-webnovel"


def _default_lora_args() -> list[str]:
    lora_1 = get_model_path("SGLANG_HCU_QWEN3_LORA_1", DEFAULT_QWEN3_LORA_1)
    lora_2 = get_model_path("SGLANG_HCU_QWEN3_LORA_2", DEFAULT_QWEN3_LORA_2)
    return [
        "--lora-paths",
        f"{LORA_NAME_1}={lora_1}",
        f"{LORA_NAME_2}={lora_2}",
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


class TestBW1100Qwen3LoRAServerHCU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = get_model_path("SGLANG_HCU_QWEN3_4B_MODEL", DEFAULT_QWEN3_4B_MODEL)
        cls.lora_1 = get_model_path("SGLANG_HCU_QWEN3_LORA_1", DEFAULT_QWEN3_LORA_1)
        cls.lora_2 = get_model_path("SGLANG_HCU_QWEN3_LORA_2", DEFAULT_QWEN3_LORA_2)
        cls.base_url = f"http://127.0.0.1:{find_available_port(11400)}"
        cls.api_key = "sk-123456"
        cls.server = HcuServerGuard(
            cls.model,
            cls.base_url,
            timeout=get_int_env("SGLANG_HCU_QWEN3_LORA_TIMEOUT", 900),
            api_key=cls.api_key,
            other_args=get_server_args(
                "SGLANG_HCU_QWEN3_LORA_SERVER_ARGS",
                _default_lora_args(),
            ),
        )
        cls.server.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.server.__exit__(None, None, None)

    @classmethod
    def _headers(cls) -> dict[str, str]:
        return {"Authorization": f"Bearer {cls.api_key}"}

    @classmethod
    def _generate(cls, lora_path: str | None) -> dict:
        response = requests.post(
            cls.base_url + "/generate",
            headers=cls._headers(),
            json={
                "text": "The capital of China is",
                "lora_path": lora_path,
                "sampling_params": {"temperature": 0, "max_new_tokens": 24},
            },
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("output_ids"):
            raise AssertionError(f"empty LoRA generation: {payload}")
        return payload

    def _assert_adapters_listed(self, expected: set[str]) -> None:
        response = requests.get(
            self.base_url + "/v1/models", headers=self._headers(), timeout=60
        )
        response.raise_for_status()
        adapters = {
            item["id"] for item in response.json()["data"] if item.get("parent")
        }
        self.assertEqual(adapters, expected)

    def test_lora_lifecycle_and_concurrent_requests(self):
        self._assert_adapters_listed({LORA_NAME_1, LORA_NAME_2})

        request_adapters = [None, LORA_NAME_1, LORA_NAME_2, LORA_NAME_1]
        with ThreadPoolExecutor(max_workers=len(request_adapters)) as executor:
            outputs = list(executor.map(self._generate, request_adapters))
        self.assertEqual(len(outputs), len(request_adapters))

        unload = requests.post(
            self.base_url + "/unload_lora_adapter",
            headers=self._headers(),
            json={"lora_name": LORA_NAME_1},
            timeout=120,
        )
        unload.raise_for_status()
        self._assert_adapters_listed({LORA_NAME_2})

        load = requests.post(
            self.base_url + "/load_lora_adapter",
            headers=self._headers(),
            json={"lora_name": LORA_NAME_1, "lora_path": self.lora_1},
            timeout=300,
        )
        load.raise_for_status()
        self._assert_adapters_listed({LORA_NAME_1, LORA_NAME_2})
        self._generate(LORA_NAME_1)
        self.assertIsNone(self.server.process.poll())


if __name__ == "__main__":
    unittest.main()
