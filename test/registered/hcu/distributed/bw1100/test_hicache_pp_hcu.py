# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""HCU HiCache with TP2/PP2 and Mooncake storage smoke test."""

from __future__ import annotations

import os
import signal
import subprocess
import time
import unittest

import requests

from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.hcu_server_guard import HcuServerGuard
from sglang.test.hcu_utils import get_model_path
from sglang.test.test_utils import find_available_port

register_hcu_ci(
    est_time=900,
    suite="nightly-hcu-4-functional",
    nightly=True,
)

DEFAULT_MODEL = (
    "/public/opendas/DL_DATA/llm-models/vllm-gptq-models/qwen2.5/Qwen2.5-7B"
)


class _MooncakeServices:
    def __init__(self):
        self.master_port = find_available_port(50051)
        self.metadata_port = find_available_port(8080)
        self.processes: list[subprocess.Popen] = []

    def __enter__(self):
        commands = (
            [
                "python3",
                "-m",
                "mooncake.http_metadata_server",
                "--port",
                str(self.metadata_port),
            ],
            [
                "mooncake_master",
                "--port",
                str(self.master_port),
                "--default_kv_lease_ttl=60s",
            ],
        )
        try:
            for command in commands:
                self.processes.append(
                    subprocess.Popen(
                        command,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )
            self._wait_ready()
            return self
        except Exception:
            self.stop()
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def _wait_ready(self):
        ready_after = time.monotonic() + 3
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if any(process.poll() is not None for process in self.processes):
                raise RuntimeError("Mooncake service exited during startup")
            try:
                requests.get(
                    f"http://127.0.0.1:{self.metadata_port}/metadata",
                    timeout=2,
                )
                if time.monotonic() >= ready_after:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise RuntimeError("Mooncake services did not become ready")

    def stop(self):
        for process in reversed(self.processes):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process in reversed(self.processes):
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        self.processes.clear()

    def env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "MOONCAKE_MASTER": f"127.0.0.1:{self.master_port}",
                "MOONCAKE_PROTOCOL": "tcp",
                "MC_MS_AUTO_DISC": "0",
                "MOONCAKE_DEVICE": "",
                "MOONCAKE_TE_META_DATA_SERVER": (
                    f"http://127.0.0.1:{self.metadata_port}/metadata"
                ),
                "MOONCAKE_GLOBAL_SEGMENT_SIZE": "4294967296",
                "SGLANG_ENABLE_DETERMINISTIC_INFERENCE": "1",
            }
        )
        return env


class TestBW1100HiCachePipelineParallelHCU(unittest.TestCase):
    def test_hicache_tp2_pp2_cache_lifecycle(self):
        model = get_model_path("SGLANG_HCU_HICACHE_PP_MODEL", DEFAULT_MODEL)
        base_url = f"http://127.0.0.1:{find_available_port(11400)}"
        server_args = [
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            "1.2",
            "--hicache-mem-layout",
            "layout_hcu",
            "--hicache-storage-backend",
            "mooncake",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            "--enable-cache-report",
            "--tp-size",
            "2",
            "--pp-size",
            "2",
            "--chunked-prefill-size",
            "256",
            "--attention-backend",
            "fa3",
            "--page-size",
            "64",
            "--disable-cuda-graph",
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.6",
            "--log-level",
            "warning",
            "--log-level-http",
            "warning",
        ]

        with _MooncakeServices() as mooncake:
            with HcuServerGuard(
                model,
                base_url,
                timeout=1200,
                other_args=server_args,
                env=mooncake.env(),
            ) as server:
                payload = {
                    "text": "HiCache pipeline parallel validation. " * 128,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 16,
                    },
                }
                first = requests.post(
                    base_url + "/generate", json=payload, timeout=300
                )
                first.raise_for_status()
                first_payload = first.json()
                self.assertTrue(first_payload.get("output_ids"))

                flush = requests.post(
                    base_url + "/flush_cache",
                    params={"timeout": 30},
                    timeout=40,
                )
                flush.raise_for_status()

                second = requests.post(
                    base_url + "/generate", json=payload, timeout=300
                )
                second.raise_for_status()
                second_payload = second.json()
                self.assertEqual(
                    first_payload.get("output_ids"),
                    second_payload.get("output_ids"),
                )
                cache_details = second_payload.get("meta_info", {}).get(
                    "cached_tokens_details"
                ) or {}
                self.assertGreater(cache_details.get("storage", 0), 0)
                self.assertIn(
                    "Mooncake", cache_details.get("storage_backend", "")
                )
                self.assertIsNone(server.process.poll())


if __name__ == "__main__":
    unittest.main()
