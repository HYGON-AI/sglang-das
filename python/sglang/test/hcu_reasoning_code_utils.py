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

"""Shared MATH-500 and HumanEval evaluation helpers for HCU nightly CI."""

import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import time
import unittest
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from sglang.test.hcu_cookbook_utils import (
    HCU_COOKBOOK_API_KEY,
    CookbookServer,
    HcuCookbookModelConfig,
)
from sglang.test.hcu_utils import openai_base_url

DEFAULT_MATH500_DATA_PATH = (
    "/public/opendas/DL_DATA/opencompass_data/math/test_prm800k_500.jsonl"
)
DEFAULT_MATH500_SHA256 = (
    "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"
)
DEFAULT_HUMANEVAL_DATA_PATH = (
    "/public/opendas/DL_DATA/opencompass_data/humaneval/" "human-eval-v2-20210705.jsonl"
)
DEFAULT_HUMANEVAL_SHA256 = (
    "1d49078ba3e2b196b9344535bef34a43021f038fad9561d6ee7c53450609a6a2"
)
DEFAULT_OUTPUT_DIR = "/sglang-checkout/test-results/hcu-reasoning-code"

MATH500_PROMPT = """Solve the following math problem. Show your reasoning, use an exact form whenever possible, and put only the final answer inside \\boxed{{}} at the end.

{problem}
"""
HUMANEVAL_PROMPT = """Complete the Python function below. Return only valid Python code containing the complete function definition. Do not use Markdown fences and do not add an explanation.

{prompt}
"""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def _model_threshold(model_key: str, metric: str, default: float) -> float:
    prefix = "SGLANG_HCU_REASONING_CODE"
    model_env = re.sub(r"[^A-Za-z0-9]+", "_", model_key).strip("_").upper()
    metric_env = metric.upper()
    for name in (
        f"{prefix}_{model_env}_{metric_env}_THRESHOLD",
        f"{prefix}_{metric_env}_THRESHOLD",
    ):
        if os.environ.get(name) not in (None, ""):
            return float(os.environ[name])
    return default


def _resolve_dataset(
    path_env: str,
    hash_env: str,
    default_path: str,
    default_hash: str,
    expected_rows: int,
) -> tuple[Path, list[dict[str, Any]], str]:
    configured_path = os.environ.get(path_env, default_path)
    path = Path(configured_path)
    if not path.is_file():
        raise AssertionError(f"{path_env} points to a missing file: {path}")

    rows = _read_jsonl(path)
    if len(rows) != expected_rows:
        raise AssertionError(
            f"{path_env} expected {expected_rows} rows, found {len(rows)}: {path}"
        )

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected_hash = os.environ.get(hash_env)
    if expected_hash in (None, "") and configured_path == default_path:
        expected_hash = default_hash
    if expected_hash and digest != expected_hash:
        raise AssertionError(
            f"{path_env} SHA256 mismatch: expected {expected_hash}, got {digest}"
        )
    return path, rows, digest


def _limited_rows(
    rows: list[dict[str, Any]], env_name: str, default: int
) -> list[dict[str, Any]]:
    requested = _int_env(env_name, default)
    if not 1 <= requested <= len(rows):
        raise AssertionError(
            f"{env_name} must be between 1 and {len(rows)}, got {requested}"
        )
    return rows[:requested]


def _request_chat(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    extra_body: dict[str, Any],
    disable_thinking: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_tokens": max_tokens,
        **extra_body,
    }
    if disable_thinking:
        chat_template_kwargs = dict(payload.get("chat_template_kwargs") or {})
        chat_template_kwargs["enable_thinking"] = False
        payload["chat_template_kwargs"] = chat_template_kwargs

    timeout = _int_env("SGLANG_HCU_REASONING_CODE_REQUEST_TIMEOUT", 3600)
    retries = _int_env("SGLANG_HCU_REASONING_CODE_REQUEST_RETRIES", 3)
    last_error: Exception | None = None
    started = time.monotonic()
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                openai_base_url(base_url) + "/chat/completions",
                headers={"Authorization": f"Bearer {HCU_COOKBOOK_API_KEY}"},
                json=payload,
                timeout=timeout,
            )
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                raise requests.RequestException(
                    f"HTTP {response.status_code}: {response.text[:4000]}"
                ) from exc
            result = response.json()
            choice = result["choices"][0]
            message = choice["message"]
            return {
                "content": message.get("content") or "",
                "reasoning_content": message.get("reasoning_content") or "",
                "usage": result.get("usage") or {},
                "finish_reason": choice.get("finish_reason"),
                "latency_s": round(time.monotonic() - started, 3),
            }
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"chat request failed after {retries} attempts: {last_error}")


def _model_context_length(base_url: str, model: str) -> int:
    response = requests.get(
        openai_base_url(base_url) + "/models",
        headers={"Authorization": f"Bearer {HCU_COOKBOOK_API_KEY}"},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    cards = payload.get("data") or []
    card = next((item for item in cards if item.get("id") == model), None)
    if card is None and cards:
        card = cards[0]
    context_length = int((card or {}).get("max_model_len") or 0)
    if context_length <= 0:
        raise AssertionError(f"invalid /v1/models context length: {payload}")
    return context_length


def _math_score(
    answer: str, content: str, reasoning_content: str
) -> tuple[bool, str, str]:
    try:
        from math_verify import (
            ExprExtractionConfig,
            LatexExtractionConfig,
            parse,
            verify,
        )
    except ImportError as exc:
        raise AssertionError(
            "MATH-500 requires math-verify==0.8.0 in the HCU CI image"
        ) from exc

    prediction_text = "\n".join(part for part in (reasoning_content, content) if part)
    gold = parse(f"${answer}$", extraction_config=[LatexExtractionConfig()])
    prediction = parse(
        prediction_text,
        extraction_config=[
            LatexExtractionConfig(boxed_match_priority=0),
            ExprExtractionConfig(),
        ],
    )
    return (
        bool(gold and prediction and verify(gold, prediction)),
        repr(gold),
        repr(prediction),
    )


def _extract_code(text: str, entry_point: str) -> str:
    fence_matches = re.findall(
        r"```(?:python)?\s*(.*?)```", text or "", flags=re.IGNORECASE | re.DOTALL
    )
    if fence_matches:
        text = next(
            (block for block in fence_matches if f"def {entry_point}(" in block),
            fence_matches[0],
        )
    text = re.sub(
        r"<think>.*?</think>", "", text or "", flags=re.IGNORECASE | re.DOTALL
    ).strip()
    lines = text.splitlines()
    code_start = next(
        (
            index
            for index, line in enumerate(lines)
            if re.match(
                r"\s*(?:from\s+\S+\s+import|import\s+\S+|def\s+|async\s+def\s+|@)",
                line,
            )
        ),
        None,
    )
    if code_start is not None:
        lines = lines[code_start:]

    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end]).rstrip() + "\n"
        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            continue
        if any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == entry_point
            for node in tree.body
        ):
            return "\n" + candidate

    definition_position = text.find(f"def {entry_point}(")
    if definition_position >= 0:
        text = text[definition_position:]
    else:
        lines = text.splitlines()
        if lines and lines[0] and not lines[0][0].isspace():
            text = "\n".join(("    " + line) if line else line for line in lines)
    return "\n" + text.rstrip() + "\n"


def _completion_tokens(response: dict[str, Any]) -> int:
    return int((response.get("usage") or {}).get("completion_tokens", 0))


def _response_hit_limit(response: dict[str, Any], max_tokens: int) -> bool:
    return (
        _completion_tokens(response) >= max_tokens
        or response.get("finish_reason") == "length"
    )


def _numeric_usage(response: dict[str, Any]) -> dict[str, int]:
    return {
        key: int(value)
        for key, value in (response.get("usage") or {}).items()
        if isinstance(value, (int, float))
    }


def _generate_one(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    retry_max_tokens: int | None,
    context_length: int,
    context_safety_margin: int,
    loop_breaker_repetition_penalty: float,
    extra_body: dict[str, Any],
    disable_thinking: bool,
) -> dict[str, Any]:
    initial_response = _request_chat(
        base_url,
        model,
        prompt,
        max_tokens,
        extra_body,
        disable_thinking,
    )
    attempts = [{"max_tokens": max_tokens, "response": initial_response}]
    selected_response = initial_response
    retry_error = None
    effective_retry_max_tokens = None
    loop_breaker_used = False
    if (
        retry_max_tokens is not None
        and retry_max_tokens > max_tokens
        and _response_hit_limit(initial_response, max_tokens)
    ):
        prompt_tokens = int(
            (initial_response.get("usage") or {}).get("prompt_tokens", 0)
        )
        effective_retry_max_tokens = min(
            retry_max_tokens,
            context_length - prompt_tokens - context_safety_margin,
        )
        if effective_retry_max_tokens <= max_tokens:
            return {
                "response": None,
                "attempts": attempts,
                "retry_error": (
                    "no larger retry fits the model context: "
                    f"context_length={context_length}, "
                    f"prompt_tokens={prompt_tokens}, "
                    f"safety_margin={context_safety_margin}"
                ),
                "effective_retry_max_tokens": effective_retry_max_tokens,
                "loop_breaker_used": loop_breaker_used,
            }
        try:
            selected_response = _request_chat(
                base_url,
                model,
                prompt,
                effective_retry_max_tokens,
                extra_body,
                disable_thinking,
            )
            attempts.append(
                {
                    "max_tokens": effective_retry_max_tokens,
                    "response": selected_response,
                }
            )
            if loop_breaker_repetition_penalty > 1.0 and _response_hit_limit(
                selected_response, effective_retry_max_tokens
            ):
                loop_breaker_body = {
                    **extra_body,
                    "repetition_penalty": loop_breaker_repetition_penalty,
                }
                loop_breaker_used = True
                selected_response = _request_chat(
                    base_url,
                    model,
                    prompt,
                    effective_retry_max_tokens,
                    loop_breaker_body,
                    disable_thinking,
                )
                attempts.append(
                    {
                        "max_tokens": effective_retry_max_tokens,
                        "response": selected_response,
                        "extra_body": {
                            "repetition_penalty": loop_breaker_repetition_penalty
                        },
                    }
                )
        except Exception as exc:
            retry_error = f"{type(exc).__name__}: {exc}"
            selected_response = None
    return {
        "response": selected_response,
        "attempts": attempts,
        "retry_error": retry_error,
        "effective_retry_max_tokens": effective_retry_max_tokens,
        "loop_breaker_used": loop_breaker_used,
    }


def _run_parallel(
    items: list[dict[str, Any]],
    prompt_builder,
    base_url: str,
    model: str,
    max_tokens: int,
    retry_max_tokens: int | None,
    context_length: int,
    context_safety_margin: int,
    loop_breaker_repetition_penalty: float,
    workers: int,
    extra_body: dict[str, Any],
    disable_thinking: bool,
    label: str,
):
    started = time.monotonic()
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _generate_one,
                base_url,
                model,
                prompt_builder(item),
                max_tokens,
                retry_max_tokens,
                context_length,
                context_safety_margin,
                loop_breaker_repetition_penalty,
                extra_body,
                disable_thinking,
            ): (index, item)
            for index, item in enumerate(items)
        }
        for future in as_completed(futures):
            index, item = futures[future]
            completed += 1
            try:
                yield index, item, future.result(), None
            except Exception as exc:
                yield index, item, None, f"{type(exc).__name__}: {exc}"
            if completed == 1 or completed % 10 == 0 or completed == len(items):
                print(
                    f"HCU {label}: progress={completed}/{len(items)}, "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )


class HcuReasoningCodeTestBase(unittest.TestCase):
    """Base class; concrete model files provide config and thresholds."""

    config: HcuCookbookModelConfig
    model_key: str
    math500_threshold: float
    humaneval_threshold: float
    math_loop_breaker_repetition_penalty = 1.0
    human_max_tokens = 2048

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not getattr(cls, "config", None):
            raise AssertionError("HCU reasoning/code test requires a model config")
        for name, value in {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }.items():
            os.environ.setdefault(name, value)

        cls.math_path, math_rows, cls.math_sha256 = _resolve_dataset(
            "SGLANG_HCU_MATH500_DATA_PATH",
            "SGLANG_HCU_MATH500_SHA256",
            DEFAULT_MATH500_DATA_PATH,
            DEFAULT_MATH500_SHA256,
            500,
        )
        cls.humaneval_path, human_rows, cls.humaneval_sha256 = _resolve_dataset(
            "SGLANG_HCU_HUMANEVAL_DATA_PATH",
            "SGLANG_HCU_HUMANEVAL_SHA256",
            DEFAULT_HUMANEVAL_DATA_PATH,
            DEFAULT_HUMANEVAL_SHA256,
            164,
        )
        cls.math_rows = _limited_rows(math_rows, "SGLANG_HCU_MATH500_NUM_EXAMPLES", 500)
        cls.human_rows = _limited_rows(
            human_rows, "SGLANG_HCU_HUMANEVAL_NUM_EXAMPLES", 164
        )
        if importlib.util.find_spec("math_verify") is None:
            raise AssertionError(
                "MATH-500 requires math-verify==0.8.0 in the HCU CI image"
            )

        cls.num_threads = _int_env("SGLANG_HCU_REASONING_CODE_NUM_THREADS", 32)
        legacy_math_max_tokens = _int_env("SGLANG_HCU_MATH500_MAX_TOKENS", 8192)
        cls.math_max_tokens = _int_env(
            "SGLANG_HCU_MATH500_INITIAL_MAX_TOKENS",
            legacy_math_max_tokens,
        )
        cls.math_retry_max_tokens = _int_env(
            "SGLANG_HCU_MATH500_RETRY_MAX_TOKENS", 131072
        )
        cls.context_safety_margin = _int_env(
            "SGLANG_HCU_MATH500_CONTEXT_SAFETY_MARGIN", 64
        )
        cls.math_loop_breaker_repetition_penalty = float(
            os.environ.get(
                "SGLANG_HCU_MATH500_LOOP_BREAKER_REPETITION_PENALTY",
                cls.math_loop_breaker_repetition_penalty,
            )
        )
        if cls.math_retry_max_tokens <= cls.math_max_tokens:
            raise AssertionError(
                "SGLANG_HCU_MATH500_RETRY_MAX_TOKENS must exceed the initial limit"
            )
        cls.human_max_tokens = _int_env(
            "SGLANG_HCU_HUMANEVAL_MAX_TOKENS", cls.human_max_tokens
        )
        cls.request_timeout = _int_env(
            "SGLANG_HCU_REASONING_CODE_REQUEST_TIMEOUT", 3600
        )
        cls.math500_threshold = _model_threshold(
            cls.model_key, "math500", cls.math500_threshold
        )
        cls.humaneval_threshold = _model_threshold(
            cls.model_key, "humaneval", cls.humaneval_threshold
        )
        cls.output_dir = (
            Path(
                os.environ.get(
                    "SGLANG_HCU_REASONING_CODE_OUTPUT_DIR", DEFAULT_OUTPUT_DIR
                )
            )
            / cls.model_key
        )
        if cls.output_dir.exists():
            shutil.rmtree(cls.output_dir)
        cls.output_dir.mkdir(parents=True, mode=0o777, exist_ok=True)
        cls.output_dir.chmod(0o777)
        cls.server = CookbookServer(cls.config, "http://127.0.0.1:30000")
        try:
            cls.server.__enter__()
            cls.context_length = _model_context_length(
                cls.server.base_url, cls.server.model_path
            )
            _write_json(
                cls.output_dir / "manifest.json",
                {
                    "model_key": cls.model_key,
                    "model_name": cls.config.name,
                    "model_path": cls.server.model_path,
                    "math500": {
                        "data_path": str(cls.math_path),
                        "sha256": cls.math_sha256,
                        "num_examples": len(cls.math_rows),
                        "initial_max_tokens": cls.math_max_tokens,
                        "retry_max_tokens": cls.math_retry_max_tokens,
                        "server_context_length": cls.context_length,
                        "context_safety_margin": cls.context_safety_margin,
                        "retry_condition": (
                            "completion_tokens reaches limit or finish_reason is length"
                        ),
                        "request_timeout_seconds": cls.request_timeout,
                        "loop_breaker_repetition_penalty": (
                            cls.math_loop_breaker_repetition_penalty
                            if cls.math_loop_breaker_repetition_penalty > 1.0
                            else None
                        ),
                        "threshold": cls.math500_threshold,
                    },
                    "humaneval": {
                        "data_path": str(cls.humaneval_path),
                        "sha256": cls.humaneval_sha256,
                        "num_examples": len(cls.human_rows),
                        "max_tokens": cls.human_max_tokens,
                        "server_context_length": cls.context_length,
                        "threshold": cls.humaneval_threshold,
                        "timeout_seconds": 5,
                        "request_timeout_seconds": cls.request_timeout,
                    },
                    "temperature": 0,
                    "num_threads": cls.num_threads,
                },
            )
        except Exception:
            cls.server.__exit__(None, None, None)
            cls.server = None
            raise

    @classmethod
    def tearDownClass(cls):
        try:
            if getattr(cls, "server", None) is not None:
                cls.server.__exit__(None, None, None)
        finally:
            super().tearDownClass()

    def test_01_math500(self):
        started = time.monotonic()
        records: list[dict[str, Any]] = []
        failures = 0
        parse_failures = 0
        initial_capped = 0
        retried_capped = 0
        retry_failures = 0
        unresolved_capped = 0
        loop_breaker_attempted = 0
        max_initial_completion_tokens = 0
        max_final_completion_tokens = 0
        effective_retry_limits: list[int] = []
        correct = 0
        usage_totals: Counter[str] = Counter()
        selected_usage_totals: Counter[str] = Counter()
        by_subject: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
        by_level: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])

        for index, item, generation, error in _run_parallel(
            self.math_rows,
            lambda row: MATH500_PROMPT.format(problem=row["problem"]),
            self.server.base_url,
            self.server.model_path,
            self.math_max_tokens,
            self.math_retry_max_tokens,
            self.context_length,
            self.context_safety_margin,
            self.math_loop_breaker_repetition_penalty,
            self.num_threads,
            self.config.eval_kwargs,
            False,
            f"MATH-500 {self.config.name}",
        ):
            if error:
                failures += 1
                records.append(
                    {
                        "dataset_index": index,
                        "unique_id": item.get("unique_id"),
                        "error": error,
                    }
                )
                continue

            attempts = generation["attempts"]
            initial_response = attempts[0]["response"]
            initial_tokens = _completion_tokens(initial_response)
            max_initial_completion_tokens = max(
                max_initial_completion_tokens, initial_tokens
            )
            initial_hit_limit = _response_hit_limit(
                initial_response, attempts[0]["max_tokens"]
            )
            initial_capped += int(initial_hit_limit)
            retried = len(attempts) > 1 or generation["retry_error"] is not None
            retried_capped += int(retried)
            loop_breaker_used = bool(generation["loop_breaker_used"])
            loop_breaker_attempted += int(loop_breaker_used)
            if retried and generation["effective_retry_max_tokens"] is not None:
                effective_retry_limits.append(generation["effective_retry_max_tokens"])
            for attempt in attempts:
                usage_totals.update(_numeric_usage(attempt["response"]))

            if generation["retry_error"] is not None:
                failures += 1
                retry_failures += 1
                unresolved_capped += 1
                records.append(
                    {
                        "dataset_index": index,
                        "unique_id": item.get("unique_id"),
                        "error": (
                            "truncation retry failed: " f"{generation['retry_error']}"
                        ),
                        "truncation_retry": {
                            "attempted": True,
                            "initial_max_tokens": attempts[0]["max_tokens"],
                            "initial_response": initial_response,
                            "retry_max_tokens_ceiling": self.math_retry_max_tokens,
                            "effective_retry_max_tokens": generation[
                                "effective_retry_max_tokens"
                            ],
                            "context_length": self.context_length,
                            "context_safety_margin": self.context_safety_margin,
                            "error": generation["retry_error"],
                            "loop_breaker_repetition_penalty": (
                                self.math_loop_breaker_repetition_penalty
                                if loop_breaker_used
                                else None
                            ),
                        },
                    }
                )
                continue

            response = generation["response"]
            selected_max_tokens = attempts[-1]["max_tokens"]
            final_tokens = _completion_tokens(response)
            max_final_completion_tokens = max(max_final_completion_tokens, final_tokens)
            final_hit_limit = _response_hit_limit(response, selected_max_tokens)
            unresolved_capped += int(final_hit_limit)
            selected_usage_totals.update(_numeric_usage(response))
            retry_metadata = None
            if retried:
                retry_metadata = {
                    "attempted": True,
                    "initial_max_tokens": attempts[0]["max_tokens"],
                    "initial_response": initial_response,
                    "retry_max_tokens_ceiling": self.math_retry_max_tokens,
                    "effective_retry_max_tokens": selected_max_tokens,
                    "context_length": self.context_length,
                    "context_safety_margin": self.context_safety_margin,
                    "retry_completion_tokens": final_tokens,
                    "retry_finish_reason": response.get("finish_reason"),
                    "retry_hit_limit": final_hit_limit,
                    "loop_breaker_repetition_penalty": (
                        self.math_loop_breaker_repetition_penalty
                        if loop_breaker_used
                        else None
                    ),
                    "pre_loop_breaker_response": (
                        attempts[-2]["response"] if loop_breaker_used else None
                    ),
                }
            try:
                passed, gold_parsed, prediction_parsed = _math_score(
                    item["answer"],
                    response["content"],
                    response["reasoning_content"],
                )
            except Exception as exc:
                passed = False
                gold_parsed = ""
                prediction_parsed = ""
                response["score_error"] = f"{type(exc).__name__}: {exc}"
            parse_failed = not prediction_parsed or prediction_parsed == "[]"
            parse_failures += int(parse_failed)
            correct += int(passed)
            by_subject[str(item["subject"])][0] += int(passed)
            by_subject[str(item["subject"])][1] += 1
            by_level[str(item["level"])][0] += int(passed)
            by_level[str(item["level"])][1] += 1
            records.append(
                {
                    "dataset_index": index,
                    **item,
                    **response,
                    "passed": passed,
                    "gold_parsed": gold_parsed,
                    "prediction_parsed": prediction_parsed,
                    "truncation_retry": retry_metadata,
                }
            )

        records.sort(key=lambda row: row["dataset_index"])
        invalid_outputs = sum(
            1
            for row in records
            if row.get("error")
            or row.get("score_error")
            or not row.get("prediction_parsed")
            or row.get("prediction_parsed") == "[]"
            or (row.get("truncation_retry") or {}).get("retry_hit_limit")
        )
        accuracy = correct / len(self.math_rows)
        summary = {
            "model": self.config.name,
            "total": len(self.math_rows),
            "correct": correct,
            "accuracy": accuracy,
            "generation_failures": failures,
            "parse_failures": parse_failures,
            "invalid_outputs": invalid_outputs,
            "initial_max_tokens": self.math_max_tokens,
            "retry_max_tokens": self.math_retry_max_tokens,
            "context_length": self.context_length,
            "context_safety_margin": self.context_safety_margin,
            "min_effective_retry_max_tokens": (
                min(effective_retry_limits) if effective_retry_limits else None
            ),
            "max_effective_retry_max_tokens": (
                max(effective_retry_limits) if effective_retry_limits else None
            ),
            "max_initial_completion_tokens": max_initial_completion_tokens,
            "max_final_completion_tokens": max_final_completion_tokens,
            "max_completion_tokens": max_final_completion_tokens,
            "initial_capped_samples": initial_capped,
            "retried_capped_samples": retried_capped,
            "retry_failures": retry_failures,
            "loop_breaker_repetition_penalty": (
                self.math_loop_breaker_repetition_penalty
                if self.math_loop_breaker_repetition_penalty > 1.0
                else None
            ),
            "loop_breaker_attempted_samples": loop_breaker_attempted,
            "unresolved_capped_samples": unresolved_capped,
            "capped_samples": unresolved_capped,
            "threshold": self.math500_threshold,
            "elapsed_s": round(time.monotonic() - started, 3),
            "usage": dict(usage_totals),
            "selected_usage": dict(selected_usage_totals),
            "by_subject": {
                key: {
                    "correct": value[0],
                    "total": value[1],
                    "accuracy": value[0] / value[1],
                }
                for key, value in sorted(by_subject.items())
            },
            "by_level": {
                key: {
                    "correct": value[0],
                    "total": value[1],
                    "accuracy": value[0] / value[1],
                }
                for key, value in sorted(by_level.items())
            },
        }
        _write_jsonl(self.output_dir / "math500_results.jsonl", records)
        _write_json(self.output_dir / "math500_summary.json", summary)
        print(f"HCU MATH-500 summary: {json.dumps(summary, ensure_ascii=False)}")

        self.assertEqual(failures, 0, f"MATH-500 request failures: {failures}")
        self.assertLessEqual(
            invalid_outputs / len(self.math_rows),
            0.01,
            "MATH-500 invalid output rate exceeds 1%: "
            f"{invalid_outputs}/{len(self.math_rows)}",
        )
        self.assertGreaterEqual(accuracy, self.math500_threshold)

    def test_02_humaneval_generation(self):
        started = time.monotonic()
        records: list[dict[str, Any]] = []
        failures = 0
        capped = 0
        usage_totals: Counter[str] = Counter()

        for index, item, generation, error in _run_parallel(
            self.human_rows,
            lambda row: HUMANEVAL_PROMPT.format(prompt=row["prompt"]),
            self.server.base_url,
            self.server.model_path,
            self.human_max_tokens,
            None,
            self.context_length,
            self.context_safety_margin,
            1.0,
            self.num_threads,
            self.config.eval_kwargs,
            True,
            f"HumanEval generation {self.config.name}",
        ):
            if error:
                failures += 1
                records.append(
                    {
                        "dataset_index": index,
                        "task_id": item.get("task_id"),
                        "error": error,
                    }
                )
                continue

            response = generation["response"]
            usage = response["usage"]
            capped += int(_response_hit_limit(response, self.human_max_tokens))
            usage_totals.update(_numeric_usage(response))
            records.append(
                {
                    "dataset_index": index,
                    "task_id": item["task_id"],
                    "completion": _extract_code(
                        response["content"], item["entry_point"]
                    ),
                    "raw_response": response["content"],
                    "reasoning_content": response["reasoning_content"],
                    "usage": usage,
                    "latency_s": response["latency_s"],
                }
            )

        records.sort(key=lambda row: row["dataset_index"])
        summary = {
            "model": self.config.name,
            "total": len(self.human_rows),
            "generation_failures": failures,
            "capped_samples": capped,
            "threshold": self.humaneval_threshold,
            "elapsed_s": round(time.monotonic() - started, 3),
            "usage": dict(usage_totals),
            "execution": "deferred to the restricted host-side judge container",
        }
        _write_jsonl(self.output_dir / "humaneval_samples.jsonl", records)
        _write_json(self.output_dir / "humaneval_generation_summary.json", summary)
        print(
            "HCU HumanEval generation summary: "
            + json.dumps(summary, ensure_ascii=False)
        )
        self.assertEqual(failures, 0, f"HumanEval request failures: {failures}")
        self.assertEqual(len(records), len(self.human_rows))
        self.assertEqual(capped, 0, f"HumanEval truncated outputs: {capped}")
