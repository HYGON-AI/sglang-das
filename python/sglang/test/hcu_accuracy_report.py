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

"""Structured result output for the HCU accuracy notification pipeline."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

RESULT_DIR_ENV = "SGLANG_HCU_ACCURACY_RESULT_DIR"
SCHEMA_VERSION = 1
_MODEL_KEY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _optional_float(name: str, value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _required_rate(name: str, value: object) -> float:
    number = _optional_float(name, value)
    assert number is not None
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {number}")
    return number


def write_hcu_accuracy_result(
    *,
    model_key: str,
    model: str,
    score: object,
    threshold: object,
    num_examples: Optional[int],
    invalid_rate: Optional[object],
    latency_seconds: Optional[object],
    source_test: str,
) -> Optional[Path]:
    """Print one normalized result and persist it when CI configured a directory."""

    if not _MODEL_KEY_PATTERN.fullmatch(model_key):
        raise ValueError(f"invalid model_key={model_key!r}")
    normalized_score = _required_rate("score", score)
    normalized_threshold = _required_rate("threshold", threshold)
    normalized_invalid_rate = _optional_float("invalid_rate", invalid_rate)
    if (
        normalized_invalid_rate is not None
        and not 0.0 <= normalized_invalid_rate <= 1.0
    ):
        raise ValueError(
            f"invalid_rate must be in [0, 1], got {normalized_invalid_rate}"
        )
    normalized_latency = _optional_float("latency_seconds", latency_seconds)
    if normalized_latency is not None and normalized_latency < 0.0:
        raise ValueError(
            f"latency_seconds must be non-negative, got {normalized_latency}"
        )
    if num_examples is not None and num_examples <= 0:
        raise ValueError(f"num_examples must be positive, got {num_examples}")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_key": model_key,
        "model": model,
        "dataset": "gsm8k",
        "score": normalized_score,
        "threshold": normalized_threshold,
        "passed": normalized_score >= normalized_threshold,
        "num_examples": num_examples,
        "invalid_rate": normalized_invalid_rate,
        "latency_seconds": normalized_latency,
        "source_test": Path(source_test).name,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)

    result_dir_value = os.environ.get(RESULT_DIR_ENV, "").strip()
    if not result_dir_value:
        print(f"HCU_ACCURACY_RESULT={serialized}", flush=True)
        return None

    result_dir = Path(result_dir_value)
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / f"{model_key}.json"
    if output_path.exists():
        raise FileExistsError(f"duplicate HCU accuracy result: {output_path}")

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=result_dir,
        prefix=f".{model_key}.",
        suffix=".tmp",
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        tmp_file.write(serialized)
        tmp_file.write("\n")
    os.replace(tmp_path, output_path)
    print(f"HCU_ACCURACY_RESULT={serialized}", flush=True)
    return output_path
