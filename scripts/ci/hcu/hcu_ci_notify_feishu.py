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

"""Collect shared HCU GSM8K results and send one signed Feishu card."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

SCHEMA_VERSION = 1
EXPECTED_MODELS = (
    ("qwen25_7b_instruct", "Qwen2.5-7B-Instruct"),
    ("qwen3_32b", "Qwen3-32B"),
    ("qwen3_30b_a3b", "Qwen3-30B-A3B"),
    ("qwen36_35b_a3b", "Qwen3.6-35B-A3B"),
    ("deepseek_v32_channel_fp8", "DeepSeek-V3.2-Channel-FP8"),
    ("glm51_channel_int8", "GLM-5.1-Channel-INT8"),
    ("kimi_k26", "Kimi-K2.6"),
    ("minimax_m25", "MiniMax-M2.5"),
    ("qwen35_397b_channel_fp8", "Qwen3.5-397B-A17B-Channel-FP8"),
)
PREVIEW_METRICS = {
    "qwen25_7b_instruct": (0.91, 0.88, 100),
    "qwen3_32b": (0.98, 0.90, 100),
    "qwen3_30b_a3b": (0.96, 0.93, 100),
    "qwen36_35b_a3b": (0.97, 0.93, 100),
    "deepseek_v32_channel_fp8": (0.76, 0.75, 200),
    "glm51_channel_int8": (0.94, 0.93, 200),
    "kimi_k26": (0.9363, 0.92, 1319),
    "minimax_m25": (0.93, 0.93, 200),
    "qwen35_397b_channel_fp8": (0.98, 0.92, 200),
}
EXPECTED_MODEL_KEYS = {key for key, _ in EXPECTED_MODELS}
EXPECTED_PARTITIONS = {"accuracy-text-0", "accuracy-text-1"}
MAX_CARD_BYTES = 20 * 1024


@dataclass(frozen=True)
class AccuracyResult:
    model_key: str
    model: str
    score: float
    threshold: float
    num_examples: Optional[int]
    invalid_rate: Optional[float]
    latency_seconds: Optional[float]
    source_test: str

    @property
    def passed(self) -> bool:
        return self.score >= self.threshold


@dataclass(frozen=True)
class PartitionStatus:
    partition: str
    outcome: str
    run_id: int
    run_attempt: int
    target_ref: str
    commit_sha: str
    image_ref: str
    image_id: str
    runner_name: str


@dataclass
class CollectedResults:
    results: Dict[str, AccuracyResult] = field(default_factory=dict)
    partition_statuses: Dict[str, PartitionStatus] = field(default_factory=dict)
    duplicate_models: Set[str] = field(default_factory=set)
    duplicate_partitions: Set[str] = field(default_factory=set)
    diagnostics: List[str] = field(default_factory=list)

    @property
    def partition_outcomes(self) -> Dict[str, str]:
        return {
            partition: status.outcome
            for partition, status in self.partition_statuses.items()
        }

    @property
    def missing_models(self) -> Set[str]:
        return EXPECTED_MODEL_KEYS - set(self.results)

    @property
    def regressions(self) -> Set[str]:
        return {key for key, result in self.results.items() if not result.passed}

    @property
    def missing_partitions(self) -> Set[str]:
        return EXPECTED_PARTITIONS - set(self.partition_statuses)

    @property
    def failed_partitions(self) -> Dict[str, str]:
        return {
            name: outcome
            for name, outcome in self.partition_outcomes.items()
            if outcome != "success"
        }


def _finite_float(payload: dict, key: str, path: Path) -> float:
    value = float(payload[key])
    if not math.isfinite(value):
        raise ValueError(f"{path}: {key} must be finite")
    return value


def _optional_float(payload: dict, key: str, path: Path) -> Optional[float]:
    value = payload.get(key)
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path}: {key} must be finite")
    return number


def _load_accuracy_result(payload: dict, path: Path) -> AccuracyResult:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema_version")
    if payload.get("dataset") != "gsm8k":
        raise ValueError(f"{path}: dataset must be gsm8k")

    model_key = str(payload["model_key"])
    if model_key not in EXPECTED_MODEL_KEYS:
        raise ValueError(f"{path}: unexpected model_key={model_key!r}")
    score = _finite_float(payload, "score", path)
    threshold = _finite_float(payload, "threshold", path)
    if not 0.0 <= score <= 1.0 or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{path}: score and threshold must be in [0, 1]")

    num_examples = payload.get("num_examples")
    if num_examples is not None:
        num_examples = int(num_examples)
        if num_examples <= 0:
            raise ValueError(f"{path}: num_examples must be positive")
    invalid_rate = _optional_float(payload, "invalid_rate", path)
    if invalid_rate is not None and not 0.0 <= invalid_rate <= 1.0:
        raise ValueError(f"{path}: invalid_rate must be in [0, 1]")
    latency_seconds = _optional_float(payload, "latency_seconds", path)
    if latency_seconds is not None and latency_seconds < 0.0:
        raise ValueError(f"{path}: latency_seconds must be non-negative")

    return AccuracyResult(
        model_key=model_key,
        model=str(payload["model"]),
        score=score,
        threshold=threshold,
        num_examples=num_examples,
        invalid_rate=invalid_rate,
        latency_seconds=latency_seconds,
        source_test=str(payload["source_test"]),
    )


def _load_partition_status(
    payload: dict,
    path: Path,
    *,
    expected_run_id: Optional[int],
    expected_run_attempt: Optional[int],
) -> PartitionStatus:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported partition schema_version")
    partition = str(payload["partition"])
    outcome = str(payload["outcome"])
    if partition not in EXPECTED_PARTITIONS:
        raise ValueError(f"{path}: unexpected partition={partition!r}")
    if outcome not in {"success", "failure", "cancelled", "skipped"}:
        raise ValueError(f"{path}: unexpected outcome={outcome!r}")
    run_id = int(payload["run_id"])
    run_attempt = int(payload["run_attempt"])
    if expected_run_id is not None and run_id != expected_run_id:
        raise ValueError(
            f"{path}: run_id={run_id} does not match expected {expected_run_id}"
        )
    if expected_run_attempt is not None and run_attempt != expected_run_attempt:
        raise ValueError(
            f"{path}: run_attempt={run_attempt} does not match expected "
            f"{expected_run_attempt}"
        )

    required_strings = {}
    for key in (
        "target_ref",
        "commit_sha",
        "image_ref",
        "image_id",
        "runner_name",
    ):
        value = payload[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path}: {key} must be a non-empty string")
        required_strings[key] = value

    return PartitionStatus(
        partition=partition,
        outcome=outcome,
        run_id=run_id,
        run_attempt=run_attempt,
        target_ref=required_strings["target_ref"],
        commit_sha=required_strings["commit_sha"],
        image_ref=required_strings["image_ref"],
        image_id=required_strings["image_id"],
        runner_name=required_strings["runner_name"],
    )


def collect_results(
    results_dir: Path,
    *,
    expected_run_id: Optional[int] = None,
    expected_run_attempt: Optional[int] = None,
) -> CollectedResults:
    collected = CollectedResults()
    if not results_dir.is_dir():
        collected.diagnostics.append(
            f"shared result directory is missing: {results_dir}"
        )
        return collected

    for path in sorted(results_dir.rglob("*.json")):
        relative_path = path.relative_to(results_dir)
        if any(part.startswith(".") for part in relative_path.parts):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"{path}: top-level JSON value must be an object")

            if path.name == "partition-status.json":
                status = _load_partition_status(
                    payload,
                    path,
                    expected_run_id=expected_run_id,
                    expected_run_attempt=expected_run_attempt,
                )
                partition = status.partition
                if (
                    partition in collected.partition_statuses
                    or partition in collected.duplicate_partitions
                ):
                    collected.duplicate_partitions.add(partition)
                    collected.diagnostics.append(
                        f"duplicate partition status for {partition}"
                    )
                    collected.partition_statuses.pop(partition, None)
                else:
                    collected.partition_statuses[partition] = status
                continue

            result = _load_accuracy_result(payload, path)
            key = result.model_key
            if key in collected.results or key in collected.duplicate_models:
                collected.duplicate_models.add(key)
                collected.results.pop(key, None)
                collected.diagnostics.append(f"duplicate model result for {key}")
                continue
            collected.results[key] = result
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            collected.diagnostics.append(str(exc))

    for partition in sorted(collected.missing_partitions):
        collected.diagnostics.append(f"missing partition status for {partition}")
    for field_name in ("target_ref", "commit_sha", "image_ref", "image_id"):
        values = {
            getattr(status, field_name)
            for status in collected.partition_statuses.values()
        }
        if len(values) > 1:
            collected.diagnostics.append(
                f"partition metadata mismatch for {field_name}: {sorted(values)}"
            )
    return collected


def build_preview_results(
    *,
    branch: str,
    target_ref: str,
    commit_sha: str,
    image: str,
    run_id: int,
    run_attempt: int,
) -> CollectedResults:
    collected = CollectedResults()
    for model_key, display_name in EXPECTED_MODELS:
        score, threshold, num_examples = PREVIEW_METRICS[model_key]
        collected.results[model_key] = AccuracyResult(
            model_key=model_key,
            model=display_name,
            score=score,
            threshold=threshold,
            num_examples=num_examples,
            invalid_rate=0.0,
            latency_seconds=None,
            source_test="notification-preview",
        )

    for partition in EXPECTED_PARTITIONS:
        collected.partition_statuses[partition] = PartitionStatus(
            partition=partition,
            outcome="success",
            run_id=run_id,
            run_attempt=run_attempt,
            target_ref=target_ref,
            commit_sha=commit_sha,
            image_ref=image,
            image_id="sha256:notification-preview",
            runner_name="notification-preview",
        )
    return collected


def feishu_signature(timestamp: int, secret: str) -> str:
    string_to_sign = f"{timestamp}\n{secret}".encode("utf-8")
    digest = hmac.new(string_to_sign, digestmod=hashlib.sha256).digest()
    return base64.b64encode(digest).decode("ascii")


def _single_line(value: str, limit: int = 160) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _card_column(content: str, weight: int) -> dict:
    return {
        "tag": "column",
        "width": "weighted",
        "weight": weight,
        "vertical_align": "top",
        "elements": [
            {
                "tag": "div",
                "text": {"tag": "lark_md", "content": content},
            }
        ],
    }


def _column_set(columns: List[Tuple[str, int]], *, grey: bool = False) -> dict:
    return {
        "tag": "column_set",
        "flex_mode": "none",
        "background_style": "grey" if grey else "default",
        "columns": [_card_column(content, weight) for content, weight in columns],
    }


def _short_image_id(image_id: str) -> str:
    compact = _single_line(image_id)
    if compact.startswith("sha256:"):
        compact = compact[len("sha256:") :]
    return compact[:12] or "unknown"


def _overall_status(
    collected: CollectedResults, _workflow_result: str
) -> Tuple[str, str, str]:
    if collected.regressions:
        return "red", "存在未达标", "存在模型精度低于阈值"
    # The shared accuracy partition also runs MiMo and MMLU. If either fails
    # after all nine expected GSM8K results were written, the nine-model card
    # is still complete and should retain its result-derived status.
    has_infra_issue = bool(collected.missing_models or collected.diagnostics)
    if has_infra_issue:
        return "orange", "结果不完整", "存在精度分片或结果异常"
    return (
        "green",
        "全部通过",
        f"{len(EXPECTED_MODELS)} 个 GSM8K 基线全部达到阈值",
    )


def build_card(
    collected: CollectedResults,
    *,
    branch: str,
    target_ref: str,
    commit_sha: str,
    image: str,
    workflow_result: str,
    run_url: str,
    preview: bool = False,
) -> dict:
    color, _, _ = _overall_status(collected, workflow_result)
    passed_count = sum(result.passed for result in collected.results.values())
    regression_count = len(collected.regressions)
    missing_count = len(collected.missing_models)
    completed_count = len(collected.results)
    image_ids = sorted(
        {
            status.image_id
            for status in collected.partition_statuses.values()
            if status.image_id
        }
    )
    resolved_image = (
        ", ".join(_short_image_id(image_id) for image_id in image_ids)
        if image_ids
        else "unknown"
    )

    if preview:
        conclusion = "通知样式预览，以下为合成数据，不代表实际测试结果。"
    elif regression_count:
        conclusion_parts = [f"{regression_count} 个模型未达到阈值"]
        if missing_count:
            conclusion_parts.append(f"{missing_count} 个模型未完成")
        conclusion = "本次" + "，".join(conclusion_parts) + "。"
    elif missing_count:
        conclusion = (
            f"已完成的 {completed_count} 个模型全部达到阈值，"
            f"{missing_count} 个模型未完成。"
        )
    elif color == "orange":
        conclusion = (
            f"已完成的 {completed_count} 个模型全部达到阈值，" "本次结果状态不完整。"
        )
    else:
        conclusion = f"本次 {completed_count} 个模型全部达到阈值。"

    summary = _column_set(
        [
            (f"<font color='green'>**{passed_count}**</font>\n通过", 1),
            (
                f"<font color='{'red' if regression_count else 'grey'}'>"
                f"**{regression_count}**</font>\n未达标",
                1,
            ),
            (
                f"<font color='{'orange' if missing_count else 'grey'}'>"
                f"**{missing_count}**</font>\n未完成",
                1,
            ),
        ],
        grey=True,
    )

    table_rows = []
    for model_key, display_name in EXPECTED_MODELS:
        result = collected.results.get(model_key)
        if result is None:
            row = {
                "model": display_name,
                "score": "--",
                "threshold": "--",
                "samples": "--",
                "status": [{"text": "未完成", "color": "orange"}],
            }
        else:
            row_status = (
                {"text": "通过", "color": "green"}
                if result.passed
                else {"text": "未达标", "color": "red"}
            )
            row = {
                "model": display_name,
                "score": _percent(result.score),
                "threshold": _percent(result.threshold),
                "samples": (
                    str(result.num_examples)
                    if result.num_examples is not None
                    else "--"
                ),
                "status": [row_status],
            }
        table_rows.append(row)

    accuracy_table = {
        "tag": "table",
        "element_id": "accuracy_table",
        "page_size": len(EXPECTED_MODELS),
        "row_height": "low",
        "freeze_first_column": True,
        "header_style": {
            "text_align": "left",
            "text_size": "normal",
            "background_style": "grey",
            "text_color": "default",
            "bold": True,
            "lines": 1,
        },
        "columns": [
            {
                "name": "model",
                "display_name": "模型",
                "data_type": "text",
                "width": "250px",
                "horizontal_align": "left",
                "vertical_align": "center",
            },
            {
                "name": "score",
                "display_name": "精度",
                "data_type": "text",
                "width": "80px",
                "horizontal_align": "right",
                "vertical_align": "center",
            },
            {
                "name": "threshold",
                "display_name": "阈值",
                "data_type": "text",
                "width": "80px",
                "horizontal_align": "right",
                "vertical_align": "center",
            },
            {
                "name": "samples",
                "display_name": "样本数",
                "data_type": "text",
                "width": "80px",
                "horizontal_align": "right",
                "vertical_align": "center",
            },
            {
                "name": "status",
                "display_name": "结论",
                "data_type": "options",
                "width": "80px",
                "horizontal_align": "center",
                "vertical_align": "center",
            },
        ],
        "rows": table_rows,
    }

    image_name = _single_line(image.rsplit("/", 1)[-1] or image, 64)
    metadata = (
        f"{_single_line(branch, 64)} @ "
        f"{_single_line(commit_sha[:8] or 'unknown', 12)} · "
        f"{image_name} @ {resolved_image}"
    )
    if target_ref and target_ref not in {branch, commit_sha}:
        metadata += f" · ref {_single_line(target_ref, 32)}"

    elements = [
        {
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": f"**结论**：{conclusion}",
            },
        },
        summary,
        {"tag": "hr"},
        {
            "tag": "div",
            "text": {"tag": "lark_md", "content": "**模型精度**"},
        },
        accuracy_table,
        {"tag": "hr"},
        {
            "tag": "div",
            "text": {
                "tag": "plain_text",
                "content": _single_line(metadata, 240),
                "text_size": "notation",
                "text_color": "grey",
                "lines": 2,
            },
        },
        {
            "tag": "button",
            "text": {"tag": "plain_text", "content": "查看 GitHub Actions"},
            "type": "primary",
            "width": "default",
            "behaviors": [
                {
                    "type": "open_url",
                    "default_url": run_url,
                }
            ],
        },
    ]
    preview_label = "预览 - " if preview else " - "
    return {
        "schema": "2.0",
        "header": {
            "template": color,
            "title": {
                "tag": "plain_text",
                "content": (
                    f"[HCU CI] GSM8K 精度{preview_label}"
                    f"{passed_count}/{len(EXPECTED_MODELS)} 通过"
                ),
            },
        },
        "body": {"elements": elements},
    }


def _validate_webhook(webhook: str) -> None:
    parsed = urllib.parse.urlparse(webhook)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("FEISHU_HCU_CI_WEBHOOK must be an HTTPS URL")
    if "/open-apis/bot/v2/hook/" not in parsed.path:
        raise ValueError("FEISHU_HCU_CI_WEBHOOK must be a Feishu v2 bot webhook")


def _response_succeeded(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("code") == 0:
        return True
    return payload.get("StatusCode") == 0


def send_card(
    webhook: str,
    secret: str,
    card: dict,
    *,
    max_attempts: int = 3,
    timeout_seconds: float = 15.0,
    urlopen_func: Callable = urllib.request.urlopen,
    sleep_func: Callable[[float], None] = time.sleep,
    time_func: Callable[[], float] = time.time,
) -> None:
    _validate_webhook(webhook)
    if not secret:
        raise ValueError("FEISHU_HCU_CI_SIGNING_SECRET is required")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    last_error = "unknown error"
    for attempt in range(1, max_attempts + 1):
        timestamp = int(time_func())
        payload = {
            "timestamp": str(timestamp),
            "sign": feishu_signature(timestamp, secret),
            "msg_type": "interactive",
            "card": card,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if len(body) > MAX_CARD_BYTES:
            raise ValueError(f"Feishu payload exceeds {MAX_CARD_BYTES} bytes")
        request = urllib.request.Request(
            webhook,
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urlopen_func(request, timeout=timeout_seconds) as response:
                response_body = response.read().decode("utf-8", errors="replace")
            response_payload = json.loads(response_body)
            if _response_succeeded(response_payload):
                return
            last_error = f"Feishu returned {response_body[:300]}"
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {_single_line(str(exc), 240)}"

        if attempt < max_attempts:
            sleep_func(2**attempt)

    raise RuntimeError(
        f"failed to send Feishu notification after {max_attempts} attempts: {last_error}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--target-ref", required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--workflow-result", required=True)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--run-attempt", required=True, type=int)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Send a clearly labeled card with synthetic model results",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.preview:
        collected = build_preview_results(
            branch=args.branch,
            target_ref=args.target_ref,
            commit_sha=args.commit_sha,
            image=args.image,
            run_id=args.run_id,
            run_attempt=args.run_attempt,
        )
    else:
        collected = collect_results(
            args.results_dir,
            expected_run_id=args.run_id,
            expected_run_attempt=args.run_attempt,
        )
    card = build_card(
        collected,
        branch=args.branch,
        target_ref=args.target_ref,
        commit_sha=args.commit_sha,
        image=args.image,
        workflow_result=args.workflow_result,
        run_url=args.run_url,
        preview=args.preview,
    )
    if args.dry_run:
        print(json.dumps(card, ensure_ascii=False, indent=2))
        return 0

    webhook = os.environ.get("FEISHU_HCU_CI_WEBHOOK", "")
    secret = os.environ.get("FEISHU_HCU_CI_SIGNING_SECRET", "")
    send_card(webhook, secret, card)
    print(
        f"Sent one Feishu HCU accuracy {'preview ' if args.preview else ''}card: "
        f"passed={sum(result.passed for result in collected.results.values())} "
        f"regressions={len(collected.regressions)} "
        f"missing={len(collected.missing_models)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
