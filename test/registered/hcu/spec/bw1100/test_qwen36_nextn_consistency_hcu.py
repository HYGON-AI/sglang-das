# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.6 NEXTN output and accepted-token checks on two HCU devices."""

import unittest
from dataclasses import replace

import requests

from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.hcu_cookbook_utils import (
    HCU_COOKBOOK_API_KEY,
    CookbookServer,
    QWEN36_27B_2GPU,
)
from sglang.test.test_utils import find_available_port

register_hcu_ci(
    est_time=2400,
    suite="nightly-hcu-2",
    nightly=True,
)


def _generate(base_url: str, model_path: str, prompt: str) -> str:
    response = requests.post(
        base_url.rstrip("/") + "/generate",
        headers={"Authorization": f"Bearer {HCU_COOKBOOK_API_KEY}"},
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 16,
            },
        },
        timeout=240,
    )
    response.raise_for_status()
    payload = response.json()
    text = payload.get("text", "") if isinstance(payload, dict) else ""
    if not text.strip():
        raise AssertionError(
            f"NEXTN generation returned empty output for {model_path}: {payload}"
        )
    return text


class TestBW1100Qwen36NextNConsistencyHCU(unittest.TestCase):
    def test_deterministic_output_and_accept_length(self):
        port = find_available_port(11600)
        base_url = f"http://127.0.0.1:{port}"
        config = replace(
            QWEN36_27B_2GPU,
            server_args=[
                *QWEN36_27B_2GPU.server_args,
                "--watchdog-timeout",
                "1200",
            ],
        )
        with CookbookServer(config, base_url) as server:
            prompts = (
                "The capital of China is",
                "Write the first eight positive even numbers:",
                "Complete this sequence: 1, 1, 2, 3, 5, 8,",
            )
            for prompt in prompts:
                first = _generate(base_url, server.model_path, prompt)
                second = _generate(base_url, server.model_path, prompt)
                self.assertEqual(first, second)

            response = requests.get(
                base_url + "/server_info",
                headers={"Authorization": f"Bearer {HCU_COOKBOOK_API_KEY}"},
                timeout=60,
            )
            response.raise_for_status()
            internal_states = response.json().get("internal_states", [])
            self.assertTrue(internal_states, "server_info has no internal states")
            accept_length = internal_states[0].get("avg_spec_accept_length")
            self.assertIsNotNone(accept_length)
            print(f"avg_spec_accept_length={float(accept_length):.4f}")
            self.assertGreater(float(accept_length), 1.0)


if __name__ == "__main__":
    unittest.main()
