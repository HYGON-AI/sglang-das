# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.hcu_pd_utils import (
    DECODE_PORT,
    PREFILL_PORT,
    ROUTER_PORT,
    resolve_minimax_m27_model_path,
)

register_hcu_ci(
    est_time=10800,
    suite="nightly-hcu-disaggregation-16",
    nightly=True,
)


def _normalized_model_path(value: str) -> str:
    return value.strip().rstrip("/")


def _write_result(payload: dict) -> None:
    result_path = os.environ.get("HCU_PD_SMOKE_RESULT_PATH")
    if not result_path:
        return
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


class TestMiniMaxM27PDHCU(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prefill_ip = os.environ["HCU_PD_PREFILL_IP"]
        cls.decode_ip = os.environ["HCU_PD_DECODE_IP"]
        cls.model_path = resolve_minimax_m27_model_path()
        cls.prefill_url = f"http://{cls.prefill_ip}:{PREFILL_PORT}"
        cls.decode_url = f"http://{cls.decode_ip}:{DECODE_PORT}"
        cls.router_url = f"http://{cls.prefill_ip}:{ROUTER_PORT}"

    def _assert_health_and_model(self, base_url: str) -> None:
        response = requests.get(f"{base_url}/health", timeout=10)
        self.assertEqual(response.status_code, 200, response.text)

        response = requests.get(f"{base_url}/model_info", timeout=10)
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(
            _normalized_model_path(payload.get("model_path", "")),
            _normalized_model_path(self.model_path),
            payload,
        )

    def _chat(self) -> dict:
        response = requests.post(
            f"{self.router_url}/v1/chat/completions",
            json={
                "model": self.model_path,
                "messages": [
                    {"role": "user", "content": "请只输出“北京”两个字，不要解释。"}
                ],
                "temperature": 0,
                "max_tokens": 256,
            },
            timeout=180,
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        choice = payload["choices"][0]
        self.assertNotEqual(choice.get("finish_reason"), "length", payload)
        content = choice["message"]["content"]
        self.assertIsInstance(content, str)
        self.assertTrue(content.strip(), payload)
        self.assertNotIn("\ufffd", content)
        final_content = content.rsplit("</think>", 1)[-1].strip()
        self.assertIn("北京", final_content, payload)
        return {
            "content": content,
            "finish_reason": choice.get("finish_reason"),
        }

    def test_minimax_m27_pd_smoke(self) -> None:
        started_at = time.monotonic()
        self._assert_health_and_model(self.prefill_url)
        self._assert_health_and_model(self.decode_url)

        single_response = self._chat()
        with ThreadPoolExecutor(max_workers=4) as pool:
            concurrent_responses = list(pool.map(lambda _: self._chat(), range(4)))

        _write_result(
            {
                "model": self.model_path,
                "single_request": single_response,
                "concurrent_request_count": len(concurrent_responses),
                "duration_seconds": round(time.monotonic() - started_at, 3),
                "passed": True,
            }
        )


if __name__ == "__main__":
    unittest.main()
