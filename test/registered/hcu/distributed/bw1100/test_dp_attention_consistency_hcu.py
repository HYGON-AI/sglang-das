# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""HCU DP Attention request consistency with a local MoE model."""

from __future__ import annotations

import os
import unittest

import requests

from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.hcu_server_guard import HcuServerGuard
from sglang.test.hcu_utils import get_model_path
from sglang.test.test_utils import find_available_port

register_hcu_ci(est_time=1200, suite="nightly-hcu-2", nightly=True)

DEFAULT_MODEL = "/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-30B-A3B"
PROMPTS = (
    "The capital of China is",
    "Complete this sequence: 1, 1, 2, 3, 5, 8,",
    "Write the first six positive even numbers:",
)


def _generate(base_url: str, prompt: str) -> tuple[int, ...]:
    response = requests.post(
        base_url.rstrip("/") + "/generate",
        json={
            "text": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": 32},
        },
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()
    output_ids = payload.get("output_ids")
    if not output_ids:
        raise AssertionError(f"empty DP Attention output: {payload}")
    return tuple(output_ids)


class TestBW1100DPAttentionConsistencyHCU(unittest.TestCase):
    def test_dp_attention_replicas_return_identical_outputs(self):
        model = get_model_path("SGLANG_HCU_DP_ATTENTION_MODEL", DEFAULT_MODEL)
        base_url = f"http://127.0.0.1:{find_available_port(11700)}"
        env = os.environ.copy()
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        server_args = [
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
            "--mem-fraction-static",
            "0.55",
            "--watchdog-timeout",
            "1200",
            "--log-level",
            "warning",
            "--log-level-http",
            "warning",
        ]

        with HcuServerGuard(
            model,
            base_url,
            timeout=1800,
            other_args=server_args,
            env=env,
        ) as server:
            for prompt in PROMPTS:
                outputs = [_generate(base_url, prompt) for _ in range(4)]
                self.assertTrue(all(output == outputs[0] for output in outputs))
            self.assertIsNone(server.process.poll())


if __name__ == "__main__":
    unittest.main()
