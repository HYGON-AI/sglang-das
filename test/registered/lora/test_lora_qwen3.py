# Copyright 2023-2025 SGLang Team
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
# ==============================================================================

import multiprocessing as mp
import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_dcu_ci
from sglang.test.dcu_utils import DCU_TEXT_SERVER_ARGS, get_server_args

register_dcu_ci(
    est_time=180,
    suite="stage-b-test-1-gpu-small-dcu",
)

from sglang.test.lora_utils import (
    LORA_MODELS_QWEN3,
    run_lora_multiple_batch_on_model_cases,
)
from sglang.test.test_utils import CustomTestCase
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    find_available_port,
    popen_launch_server,
)

register_amd_ci(
    est_time=30,
    suite="stage-b-test-1-gpu-small-amd",
    disabled="see https://github.com/sgl-project/sglang/issues/13107",
)


DCU_QWEN3_4B_MODEL = "/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-4B"
DCU_QWEN3_LORA_A = "/public/opendas/DL_DATA/llm-models/lora/nissenj/Qwen3-4B-lora-v2"
DCU_QWEN3_LORA_B = (
    "/public/opendas/DL_DATA/llm-models/lora/TanXS/Qwen3-4B-LoRA-ZH-WebNovelty-v0.0"
)


def _is_dcu():
    return os.environ.get("SGLANG_IS_IN_CI_DCU") == "1"


def _dcu_lora_generate(base_url, lora_name, prompt):
    response = requests.post(
        base_url.rstrip("/") + "/generate",
        json={
            "text": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            "lora_path": lora_name,
        },
        timeout=120,
    )
    response.raise_for_status()
    text = response.json().get("text", "")
    if not text or not text.strip():
        raise AssertionError(f"LoRA {lora_name} returned empty text: {response.text}")
    return text


class TestLoRAQwen3(CustomTestCase):
    def test_ci_lora_models(self):
        if _is_dcu():
            self._run_dcu_qwen3_lora_smoke()
            return

        run_lora_multiple_batch_on_model_cases(LORA_MODELS_QWEN3)

    def _run_dcu_qwen3_lora_smoke(self):
        base_model = os.environ.get("SGLANG_DCU_QWEN3_LORA_BASE", DCU_QWEN3_4B_MODEL)
        lora_a = os.environ.get("SGLANG_DCU_QWEN3_LORA_A", DCU_QWEN3_LORA_A)
        lora_b = os.environ.get("SGLANG_DCU_QWEN3_LORA_B", DCU_QWEN3_LORA_B)
        for path in [base_model, lora_a, lora_b]:
            if not os.path.exists(path):
                self.skipTest(f"DCU Qwen3 LoRA path does not exist: {path}")

        port = find_available_port(11001)
        base_url = f"http://127.0.0.1:{port}"
        server_args = get_server_args(
            "SGLANG_DCU_QWEN3_LORA_SERVER_ARGS",
            DCU_TEXT_SERVER_ARGS
            + [
                "--disable-cuda-graph",
                "--disable-radix-cache",
                "--enable-lora",
                "--lora-paths",
                f"a={lora_a}",
                f"b={lora_b}",
                "--max-loras-per-batch",
                "3",
                "--max-running-requests",
                "4",
            ],
        )
        process = popen_launch_server(
            base_model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args,
            env={
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                "SGLANG_USE_MODELSCOPE": os.environ.get("SGLANG_USE_MODELSCOPE", "1"),
                "SGLANG_USE_LIGHTOP": os.environ.get("SGLANG_USE_LIGHTOP", "1"),
            },
        )
        try:
            _dcu_lora_generate(base_url, "a", "Write one sentence about Paris.")
            _dcu_lora_generate(base_url, "b", "写一句关于人工智能的话。")
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
