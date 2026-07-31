# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Model registry and runner for the standalone HCU EvalScope suite."""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from sglang.test.hcu_cookbook_utils import (
    DEEPSEEK_V32_CHANNEL_FP8_8GPU,
    GLM51_CHANNEL_INT8_8GPU,
    HCU_COOKBOOK_API_KEY,
    KIMI_K26_8GPU,
    MINIMAX_M25_FP8_8GPU,
    QWEN35_397B_A17B_CHANNEL_FP8_4GPU,
    QWEN36_35B_A3B_2GPU,
    QWEN3_30B_A3B_4GPU,
    QWEN3_32B_4GPU,
    CookbookServer,
    HcuCookbookModelConfig,
)
from sglang.test.hcu_utils import openai_base_url
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST


EVALSCOPE_VERSION = "1.9.1"
REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "scripts" / "ci" / "hcu" / "run_hcu_evalscope_accuracy.sh"
DEFAULT_OUTPUT_ROOT = Path("/sglang-checkout/test-results/hcu-evalscope")


QWEN25_7B_INSTRUCT = HcuCookbookModelConfig(
    name="Qwen2.5-7B-Instruct",
    env_name="SGLANG_HCU_QWEN25_7B_MODEL",
    default_path="/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-7B-Instruct",
    tp_size=1,
    timeout=3600,
    dtype_or_quant="bf16",
    server_args=[
        "--attention-backend",
        "fa3",
        "--page-size",
        "64",
        "--trust-remote-code",
        "--log-level",
        "warning",
        "--log-level-http",
        "warning",
    ],
)


@dataclass(frozen=True)
class HcuEvalScopeCase:
    key: str
    config: HcuCookbookModelConfig
    thresholds: dict[str, float | None]


EVALSCOPE_CASES = {
    case.key: case
    for case in (
        HcuEvalScopeCase(
            "qwen25_7b_instruct",
            QWEN25_7B_INSTRUCT,
            {"gsm8k": 0.85, "math_500": 0.73, "humaneval": 0.75},
        ),
        HcuEvalScopeCase(
            "qwen3_32b",
            QWEN3_32B_4GPU,
            {"gsm8k": 0.94, "math_500": 0.87, "humaneval": 0.90},
        ),
        HcuEvalScopeCase(
            "qwen3_30b_a3b",
            QWEN3_30B_A3B_4GPU,
            {"gsm8k": 0.94, "math_500": 0.90, "humaneval": 0.92},
        ),
        HcuEvalScopeCase(
            "qwen36_35b_a3b",
            QWEN36_35B_A3B_2GPU,
            {"gsm8k": 0.93, "math_500": 0.76, "humaneval": 0.84},
        ),
        HcuEvalScopeCase(
            "deepseek_v32_channel_fp8",
            DEEPSEEK_V32_CHANNEL_FP8_8GPU,
            {"gsm8k": 0.86, "math_500": 0.72, "humaneval": None},
        ),
        HcuEvalScopeCase(
            "glm51_channel_int8",
            GLM51_CHANNEL_INT8_8GPU,
            {"gsm8k": 0.94, "math_500": 0.92, "humaneval": 0.92},
        ),
        HcuEvalScopeCase(
            "kimi_k26",
            KIMI_K26_8GPU,
            {"gsm8k": 0.95, "math_500": 0.94, "humaneval": 0.96},
        ),
        HcuEvalScopeCase(
            "minimax_m25",
            MINIMAX_M25_FP8_8GPU,
            {"gsm8k": 0.95, "math_500": 0.92, "humaneval": 0.92},
        ),
        HcuEvalScopeCase(
            "qwen35_397b_a17b_channel_fp8",
            QWEN35_397B_A17B_CHANNEL_FP8_4GPU,
            {"gsm8k": 0.95, "math_500": 0.94, "humaneval": 0.89},
        ),
    )
}


def _effective_thresholds(case: HcuEvalScopeCase) -> dict[str, float | None]:
    thresholds = dict(case.thresholds)
    raw_override = os.environ.get("SGLANG_HCU_EVALSCOPE_THRESHOLDS_JSON")
    if raw_override:
        override = json.loads(raw_override)
        if not isinstance(override, dict):
            raise ValueError("SGLANG_HCU_EVALSCOPE_THRESHOLDS_JSON must be an object")
        unknown = sorted(set(override) - set(thresholds))
        if unknown:
            raise ValueError(f"unknown EvalScope threshold datasets: {unknown}")
        thresholds.update(override)
    return thresholds


def _require_evalscope() -> None:
    try:
        installed_version = importlib.metadata.version("evalscope")
    except importlib.metadata.PackageNotFoundError as error:
        raise AssertionError(
            "EvalScope is not installed; install "
            "scripts/ci/hcu/requirements_evalscope.txt"
        ) from error
    if installed_version != EVALSCOPE_VERSION:
        raise AssertionError(
            f"EvalScope {EVALSCOPE_VERSION} is required; found {installed_version}"
        )


def run_hcu_evalscope_case(case_key: str) -> dict:
    """Start one model once and evaluate all three local datasets."""

    if case_key not in EVALSCOPE_CASES:
        raise KeyError(f"unknown HCU EvalScope case: {case_key}")
    if not RUNNER.is_file():
        raise AssertionError(f"EvalScope runner does not exist: {RUNNER}")
    _require_evalscope()

    case = EVALSCOPE_CASES[case_key]
    model_path = case.config.resolve_model_path()
    limits = {
        "gsm8k": os.environ.get("SGLANG_HCU_EVALSCOPE_GSM8K_LIMIT", "200"),
        "math_500": os.environ.get("SGLANG_HCU_EVALSCOPE_MATH_LIMIT", "200"),
        "humaneval": os.environ.get("SGLANG_HCU_EVALSCOPE_HUMANEVAL_LIMIT", "164"),
    }
    thresholds = _effective_thresholds(case)
    if os.environ.get("SGLANG_HCU_EVALSCOPE_ENFORCE_PARTIAL_THRESHOLDS") != "1":
        full_counts = {"gsm8k": "200", "math_500": "200", "humaneval": "164"}
        for dataset, count in limits.items():
            if count != full_counts[dataset]:
                thresholds[dataset] = None

    output_parent = Path(
        os.environ.get("SGLANG_HCU_EVALSCOPE_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    )
    output_parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix=f"{case.key}-", dir=output_parent))
    summary_path = output_dir / "summary.json"

    env = os.environ.copy()
    env.update(
        {
            "API_URL": openai_base_url(DEFAULT_URL_FOR_TEST),
            "API_KEY": HCU_COOKBOOK_API_KEY,
            "MODEL_ID": model_path,
            "MODEL_KEY": case.key,
            "MODEL_NAME": case.config.name,
            "OUTPUT_DIR": str(output_dir),
            "SUMMARY_PATH": str(summary_path),
            "BATCH_SIZE": env.get("SGLANG_HCU_EVALSCOPE_BATCH_SIZE", "16"),
            "MAX_TOKENS": env.get("SGLANG_HCU_EVALSCOPE_MAX_TOKENS", "16384"),
            "GSM8K_LIMIT": limits["gsm8k"],
            "MATH_LIMIT": limits["math_500"],
            "HUMANEVAL_LIMIT": limits["humaneval"],
            "EXTRA_BODY_JSON": json.dumps(case.config.eval_kwargs or None),
            "THRESHOLDS_JSON": json.dumps(thresholds),
        }
    )

    with CookbookServer(case.config, DEFAULT_URL_FOR_TEST):
        completed = subprocess.run(
            ["bash", str(RUNNER)],
            cwd=REPO_ROOT,
            env=env,
            check=False,
        )

    if not summary_path.is_file():
        raise AssertionError(
            f"{case.config.name} EvalScope did not write {summary_path}; "
            f"runner exit code={completed.returncode}"
        )
    with summary_path.open(encoding="utf-8") as file:
        summary = json.load(file)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if completed.returncode != 0 or summary.get("status") != "passed":
        raise AssertionError(
            f"{case.config.name} EvalScope failed; "
            f"summary={summary_path}, runner exit code={completed.returncode}"
        )
    return summary
