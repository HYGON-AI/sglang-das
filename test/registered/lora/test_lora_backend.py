# Copyright 2023-2024 SGLang Team
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
from typing import List

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
from sglang.test.lora_utils import (
    ALL_OTHER_LORA_MODELS,
    BACKENDS,
    CI_LORA_MODELS,
    DEFAULT_PROMPTS,
    TORCH_DTYPES,
    LoRAAdaptor,
    LoRAModelCase,
    run_lora_test_one_by_one,
)
from sglang.test.test_utils import CustomTestCase, is_in_ci

register_cuda_ci(est_time=200, suite="stage-b-test-1-gpu-small")
register_amd_ci(
    est_time=200,
    suite="stage-b-test-1-gpu-small-amd",
    disabled="see https://github.com/sgl-project/sglang/issues/13107",
)
register_dcu_ci(
    est_time=200,
    suite="stage-b-test-1-gpu-small-dcu",
)

DCU_QWEN3_LORA_MODEL = LoRAModelCase(
    base="/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-4B",
    adaptors=[
        LoRAAdaptor(
            name="/public/opendas/DL_DATA/llm-models/lora/nissenj/Qwen3-4B-lora-v2",
            prefill_tolerance=3e-1,
            decode_tolerance=3e-1,
        ),
    ],
    max_loras_per_batch=1,
)
DCU_PROMPTS = ["SGL is a"]
DCU_BACKENDS = ["csgmv", "triton"]
DCU_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _is_dcu() -> bool:
    return os.environ.get("SGLANG_IS_IN_CI_DCU", "0") == "1"


class TestLoRABackend(CustomTestCase):

    def _run_backend_on_model_cases(self, model_cases: List[LoRAModelCase]):
        if _is_dcu():
            self._run_dcu_backend_on_model_cases(model_cases)
            return

        for model_case in model_cases:
            # If skip_long_prompt is True, filter out prompts longer than 1000 characters
            prompts = (
                DEFAULT_PROMPTS
                if not model_case.skip_long_prompt
                else [p for p in DEFAULT_PROMPTS if len(p) < 1000]
            )
            for torch_dtype in TORCH_DTYPES:
                for backend in BACKENDS:
                    run_lora_test_one_by_one(
                        prompts,
                        model_case,
                        torch_dtype,
                        max_new_tokens=32,
                        backend=backend,
                    )

    def _run_dcu_backend_on_model_cases(self, model_cases: List[LoRAModelCase]):
        for model_case in model_cases:
            for path in [model_case.base, *[adaptor.name for adaptor in model_case.adaptors]]:
                if not os.path.exists(path):
                    self.skipTest(f"DCU LoRA backend path does not exist: {path}")

            for torch_dtype in TORCH_DTYPES:
                for backend in DCU_BACKENDS:
                    run_lora_test_one_by_one(
                        DCU_PROMPTS,
                        model_case,
                        torch_dtype,
                        max_new_tokens=8,
                        backend=backend,
                        disable_cuda_graph=True,
                        disable_radix_cache=True,
                        mem_fraction_static=0.65,
                        attention_backend="fa3",
                        page_size=64,
                        max_lora_rank=32,
                        lora_target_modules=DCU_LORA_TARGET_MODULES,
                    )

    def test_ci_lora_models(self):
        if _is_dcu():
            self._run_backend_on_model_cases([DCU_QWEN3_LORA_MODEL])
            return

        self._run_backend_on_model_cases(CI_LORA_MODELS)

    def test_all_lora_models(self):
        if is_in_ci():
            return

        # Retain ONLY_RUN check here
        filtered_models = []
        for model_case in ALL_OTHER_LORA_MODELS:
            if "ONLY_RUN" in os.environ and os.environ["ONLY_RUN"] != model_case.base:
                continue
            filtered_models.append(model_case)

        self._run_backend_on_model_cases(filtered_models)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
