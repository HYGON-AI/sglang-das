# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""HCU Torch Compile and CUDA Graph startup/replay smoke test."""

from __future__ import annotations

import os
import unittest

import requests

from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.hcu_server_guard import HcuServerGuard
from sglang.test.hcu_utils import get_model_path
from sglang.test.test_utils import find_available_port

register_hcu_ci(est_time=1200, suite="nightly-hcu-1", nightly=True)

DEFAULT_MODEL = (
    "/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-7B-Instruct"
)


class TestBW1100TorchCompileCudaGraphHCU(unittest.TestCase):
    def test_compile_and_graph_replay(self):
        model = get_model_path("SGLANG_HCU_COMPILE_MODEL", DEFAULT_MODEL)
        base_url = f"http://127.0.0.1:{find_available_port(11300)}"
        env = os.environ.copy()
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        args = [
            "--enable-torch-compile",
            "--torch-compile-max-bs",
            "4",
            "--cuda-graph-max-bs",
            "4",
            "--watchdog-timeout",
            "1800",
            "--attention-backend",
            "fa3",
            "--page-size",
            "64",
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.65",
            "--log-level",
            "warning",
            "--log-level-http",
            "warning",
        ]

        with HcuServerGuard(
            model,
            base_url,
            timeout=1800,
            other_args=args,
            env=env,
        ) as server:
            payload = {
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            }
            first = requests.post(
                base_url + "/generate", json=payload, timeout=600
            )
            first.raise_for_status()
            replay = requests.post(
                base_url + "/generate", json=payload, timeout=600
            )
            replay.raise_for_status()
            self.assertEqual(
                first.json().get("output_ids"), replay.json().get("output_ids")
            )

            batch = requests.post(
                base_url + "/generate",
                json={
                    "text": [
                        "The capital of China is",
                        "2 + 3 =",
                        "Water freezes at",
                        "Complete: 1, 1, 2, 3, 5,",
                    ],
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 16,
                    },
                },
                timeout=600,
            )
            batch.raise_for_status()
            outputs = batch.json()
            self.assertEqual(len(outputs), 4)
            self.assertTrue(all(item.get("output_ids") for item in outputs))
            self.assertIsNone(server.process.poll())


if __name__ == "__main__":
    unittest.main()
