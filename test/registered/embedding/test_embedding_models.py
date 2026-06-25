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
import random
import unittest
from typing import Optional

import torch
from transformers import AutoConfig, AutoTokenizer

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
from sglang.test.runners import DEFAULT_PROMPTS, HFRunner, SRTRunner
from sglang.test.test_utils import (
    CustomTestCase,
    get_similarities,
    is_in_amd_ci,
    is_in_ci,
)

# Embedding model tests
register_amd_ci(
    est_time=73,
    suite="stage-b-test-1-gpu-small-amd",
    disabled="see https://github.com/sgl-project/sglang/issues/11127",
)
register_cuda_ci(est_time=73, suite="stage-b-test-1-gpu-small")

register_dcu_ci(
    est_time=120,
    suite="stage-b-test-1-gpu-small-dcu",
)

if os.environ.get("SGLANG_IS_IN_CI_DCU"):
    _dcu_embedding_model = os.environ.get(
        "SGLANG_TEST_DEFAULT_SMALL_EMBEDDING_MODEL_NAME",
        "/public/opendas/DL_DATA/llm-models/vllm-optest-models/Qwen/Qwen3-Embedding-0.6B",
    )
    MODEL_TO_CONFIG = {_dcu_embedding_model: (1, 1e-5)}
else:
    MODEL_TO_CONFIG = {
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct": (1, 1e-5),
        "intfloat/e5-mistral-7b-instruct": (1, 1e-5),
        "marco/mcdse-2b-v1": (1, 1e-5),
        "Qwen/Qwen3-Embedding-8B": (1, 1e-5),
        # Temporarily disable before this model is fixed
        # "jason9693/Qwen2.5-1.5B-apeach": (1, 1e-5),
    }
MODELS = [(key, *MODEL_TO_CONFIG[key]) for key in MODEL_TO_CONFIG]

TORCH_DTYPES = [torch.float16]
DCU_PROMPTS = ["DCU embedding smoke test", "SGLang embedding check"]


def _is_dcu() -> bool:
    return os.environ.get("SGLANG_IS_IN_CI_DCU", "0") == "1"


class TestEmbeddingModels(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def _truncate_prompts(self, prompts, model_path):
        config = AutoConfig.from_pretrained(model_path)
        max_length = getattr(config, "max_position_embeddings", 2048)

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        truncated_prompts = []
        for prompt in prompts:
            tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
            if len(tokens.input_ids[0]) > max_length:
                truncated_text = tokenizer.decode(
                    tokens.input_ids[0][: max_length - 1], skip_special_tokens=True
                )
                truncated_prompts.append(truncated_text)
            else:
                truncated_prompts.append(prompt)
        return truncated_prompts

    def assert_close_prefill_logits(
        self,
        prompts,
        model_path,
        tp_size,
        torch_dtype,
        prefill_tolerance,
        matryoshka_dim: Optional[int] = None,
    ) -> None:
        test_prompts = DCU_PROMPTS if _is_dcu() else prompts
        truncated_prompts = self._truncate_prompts(test_prompts, model_path)

        with HFRunner(
            model_path,
            torch_dtype=torch_dtype,
            model_type="embedding",
            matryoshka_dim=matryoshka_dim,
        ) as hf_runner:
            hf_outputs = hf_runner.forward(truncated_prompts)

        attention_backend = (
            "fa3" if _is_dcu() else ("triton" if is_in_amd_ci() else None)
        )
        with SRTRunner(
            model_path,
            tp_size=tp_size,
            torch_dtype=torch_dtype,
            model_type="embedding",
            attention_backend=attention_backend,
            page_size=64 if _is_dcu() else None,
            disable_cuda_graph=_is_dcu(),
            disable_radix_cache=_is_dcu(),
            json_model_override_args=(
                {"matryoshka_dimensions": [matryoshka_dim]} if matryoshka_dim else None
            ),
        ) as srt_runner:
            srt_outputs = srt_runner.forward(
                truncated_prompts, dimensions=matryoshka_dim
            )

        for i in range(len(test_prompts)):
            hf_logits = torch.Tensor(hf_outputs.embed_logits[i])
            srt_logits = torch.Tensor(srt_outputs.embed_logits[i])

            similarity = torch.tensor(get_similarities(hf_logits, srt_logits))
            print("similarity diff", abs(similarity - 1))

            if len(test_prompts[i]) <= 1000:
                assert torch.all(
                    abs(similarity - 1) < prefill_tolerance
                ), "embeddings are not all close"

    def test_prefill_logits(self):
        if _is_dcu():
            self._run_dcu_embedding_smoke(matryoshka_dim=None)
            return

        models_to_test = MODELS

        if is_in_ci():
            models_to_test = [random.choice(MODELS)]

        for model, tp_size, prefill_tolerance in models_to_test:
            for torch_dtype in TORCH_DTYPES:
                self.assert_close_prefill_logits(
                    DEFAULT_PROMPTS, model, tp_size, torch_dtype, prefill_tolerance
                )

    def test_matryoshka_embedding(self):
        if _is_dcu():
            self._run_dcu_embedding_smoke(matryoshka_dim=128)
            return

        matryoshka_model = (
            _dcu_embedding_model
            if os.environ.get("SGLANG_IS_IN_CI_DCU")
            else "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
        )
        models_to_test = [(matryoshka_model, *MODEL_TO_CONFIG[matryoshka_model])]

        for model, tp_size, prefill_tolerance in models_to_test:
            for torch_dtype in TORCH_DTYPES:
                self.assert_close_prefill_logits(
                    DEFAULT_PROMPTS,
                    model,
                    tp_size,
                    torch_dtype,
                    prefill_tolerance,
                    matryoshka_dim=128,
                )

    def _run_dcu_embedding_smoke(self, matryoshka_dim: Optional[int] = None):
        model_path = _dcu_embedding_model
        if not os.path.exists(model_path):
            self.skipTest(f"DCU embedding model path does not exist: {model_path}")

        json_model_override_args = (
            {
                "is_matryoshka": True,
                "matryoshka_dimensions": [matryoshka_dim],
            }
            if matryoshka_dim
            else None
        )
        with SRTRunner(
            model_path,
            tp_size=1,
            torch_dtype=torch.float16,
            model_type="embedding",
            attention_backend="fa3",
            page_size=64,
            disable_cuda_graph=True,
            disable_radix_cache=True,
            json_model_override_args=json_model_override_args,
        ) as srt_runner:
            srt_outputs = srt_runner.forward(
                DCU_PROMPTS, dimensions=matryoshka_dim
            )

        self.assertEqual(len(srt_outputs.embed_logits), len(DCU_PROMPTS))
        for embedding in srt_outputs.embed_logits:
            self.assertGreater(len(embedding), 0)
            if matryoshka_dim:
                self.assertEqual(len(embedding), matryoshka_dim)


if __name__ == "__main__":
    unittest.main()
