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

import os
import shlex
import unittest
from pathlib import Path

import openai
import requests


# HCU CI-owned helpers can auto-run the HCU PR gate.
HCU_TEXT_SERVER_ARGS = [
    "--attention-backend",
    "fa3",
    "--page-size",
    "64",
    "--trust-remote-code",
    "--log-level",
    "warning",
    "--log-level-http",
    "warning",
]

HCU_VLM_SERVER_ARGS = [
    "--attention-backend",
    "fa3",
    "--mm-attention-backend",
    "fa3",
    "--page-size",
    "64",
    "--enable-multimodal",
    "--trust-remote-code",
    "--log-level",
    "warning",
    "--log-level-http",
    "warning",
]

HCU_MOE_SERVER_ARGS = HCU_TEXT_SERVER_ARGS + [
    "--tp-size",
    "2",
]

HCU_EMBEDDING_SERVER_ARGS = HCU_TEXT_SERVER_ARGS + [
    "--is-embedding",
    "--enable-metrics",
]


def get_model_path(env_name: str, default_path: str) -> str:
    model_path = os.environ.get(env_name, default_path)
    if model_path.startswith(("/", ".")) and not os.path.exists(model_path):
        if env_name in os.environ:
            raise AssertionError(
                f"{env_name} points to a missing local model path: {model_path}"
            )
        raise unittest.SkipTest(f"Default HCU model path does not exist: {model_path}")
    return model_path


def get_server_args(env_name: str, default_args: list[str]) -> list[str]:
    value = os.environ.get(env_name)
    if value:
        return shlex.split(value)
    return list(default_args)


def get_int_env(env_name: str, default: int) -> int:
    value = os.environ.get(env_name)
    if value in (None, ""):
        return default
    return int(value)


def repo_root_from_test_file(test_file: str) -> Path:
    path = Path(test_file).resolve()
    for parent in path.parents:
        if (parent / "python" / "sglang").exists() and (parent / "test").exists():
            return parent
    raise RuntimeError(f"Cannot locate repo root from {test_file}")


def openai_base_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/v1"


def assert_chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 16,
) -> str:
    client = openai.Client(api_key=api_key, base_url=openai_base_url(base_url))
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise AssertionError("chat completion returned empty content")
    return content


def assert_generate_non_empty(
    base_url: str,
    text: str = "The capital of France is",
    max_new_tokens: int = 16,
    api_key: str | None = None,
) -> str:
    headers = None
    if api_key:
        headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.post(
        base_url.rstrip("/") + "/generate",
        headers=headers,
        json={
            "text": text,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        content = payload.get("text") or payload.get("output") or str(payload)
    else:
        content = str(payload)
    if not content.strip():
        raise AssertionError(f"generate returned empty response: {payload}")
    return content


def assert_rerank_scores(base_url: str, api_key: str, query: str, documents: list[str]):
    response = requests.post(
        openai_base_url(base_url) + "/rerank",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"query": query, "documents": documents},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise AssertionError(f"rerank returned invalid response: {payload}")
    if "score" not in payload[0] or not isinstance(payload[0]["score"], (float, int)):
        raise AssertionError(f"rerank response does not contain numeric score: {payload}")
    return payload


RED_DOT_IMAGE_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/"
    "pLvAAAAAElFTkSuQmCC"
)
