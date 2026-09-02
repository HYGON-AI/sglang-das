# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""HCU PP and TP+PP deterministic request consistency smoke test."""

from __future__ import annotations

import os
import unittest

import requests

from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.hcu_server_guard import HcuServerGuard
from sglang.test.hcu_utils import get_model_path
from sglang.test.test_utils import find_available_port

register_hcu_ci(
    est_time=1500,
    suite="nightly-hcu-4-functional",
    nightly=True,
)

DEFAULT_MODEL = (
    "/public/opendas/DL_DATA/llm-models/vllm-gptq-models/qwen2.5/Qwen2.5-7B"
)
PROMPTS = (
    ("The capital of France is", "paris"),
    ("2 + 3 =", "5"),
    ("Complete this sequence: 1, 1, 2, 3, 5,", "8"),
)


def _offline_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    return env


def _server_args(mode: str) -> list[str]:
    parallel_args = {
        "tp4": ["--tp-size", "4"],
        "pp4": ["--pp-size", "4"],
        "tp2-pp2": ["--tp-size", "2", "--pp-size", "2"],
    }[mode]
    return [
        *parallel_args,
        "--attention-backend",
        "fa3",
        "--page-size",
        "64",
        "--trust-remote-code",
        "--disable-cuda-graph",
        "--chunked-prefill-size",
        "256",
        "--mem-fraction-static",
        "0.55",
        "--log-level",
        "warning",
        "--log-level-http",
        "warning",
    ]


def _generate(base_url: str, prompt: str, expected: str) -> tuple[int, ...]:
    response = requests.post(
        base_url.rstrip("/") + "/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 24,
            },
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    output_ids = payload.get("output_ids")
    if not output_ids:
        raise AssertionError(f"empty generation for prompt {prompt!r}: {payload}")
    output_text = payload.get("text", "").lower()
    if expected not in output_text:
        raise AssertionError(
            f"missing expected answer {expected!r} for {prompt!r}: {payload}"
        )
    return tuple(output_ids)


class TestBW1100PipelineParallelConsistencyHCU(unittest.TestCase):
    def _run_mode(self, mode: str) -> None:
        model = get_model_path("SGLANG_HCU_PARALLEL_MODEL", DEFAULT_MODEL)
        base_url = f"http://127.0.0.1:{find_available_port(11200)}"
        with HcuServerGuard(
            model,
            base_url,
            timeout=1200,
            other_args=_server_args(mode),
            env=_offline_env(),
        ):
            for prompt, expected in PROMPTS:
                first = _generate(base_url, prompt, expected)
                second = _generate(base_url, prompt, expected)
                self.assertEqual(first, second, f"{mode} output is not repeatable")

    def test_tp4_pp4_and_tp2_pp2_request_consistency(self):
        self._run_mode("tp4")
        self._run_mode("pp4")
        self._run_mode("tp2-pp2")


if __name__ == "__main__":
    unittest.main()
