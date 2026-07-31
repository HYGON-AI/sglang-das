# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""EvalScope helpers for deterministic HCU accuracy evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Any


DATASET_NAMES = ("gsm8k", "math_500", "humaneval")
FENCE_PATTERN = re.compile(
    r"```(?:[A-Za-z0-9_+.-]+)?[ \t]*\r?\n(.*?)```", re.DOTALL
)


def extract_final_code(text: str) -> str:
    """Extract the final generated code after any reasoning section."""

    final_text = text.rsplit("</think>", 1)[-1]
    blocks = FENCE_PATTERN.findall(final_text)
    if not blocks:
        blocks = FENCE_PATTERN.findall(text)
    code = blocks[-1] if blocks else final_text
    return textwrap.dedent(code).strip() + "\n"


def _patched_humaneval_postprocess(_cls: type, text: str) -> str:
    return extract_final_code(text)


def run_evalscope_cli(argv: list[str]) -> None:
    """Run EvalScope after applying the validated HumanEval extraction fix."""

    from evalscope.benchmarks.humaneval.humaneval_adapter import (
        HumanevalAdapter,
    )
    from evalscope.cli.cli import run_cmd

    HumanevalAdapter._postprocess = classmethod(  # type: ignore[method-assign]
        _patched_humaneval_postprocess
    )
    sys.argv = ["evalscope", *argv]
    run_cmd()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_jsonl(
    path: Path, expected_count: int, expected_sha256: str
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"dataset file does not exist: {path}")

    with path.open("rb") as file:
        count = sum(1 for line in file if line.strip())
    if count != expected_count:
        raise ValueError(
            f"{path} contains {count} records; expected {expected_count}"
        )

    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{path} SHA256 is {actual_sha256}; expected {expected_sha256}"
        )
    return {
        "path": str(path),
        "records": count,
        "sha256": actual_sha256,
    }


def _report_sample_count(report: dict[str, Any]) -> int:
    value = report.get("num")
    if isinstance(value, int):
        return value

    for metric in report.get("metrics", []):
        if metric.get("name") == "mean_acc" and isinstance(metric.get("num"), int):
            return int(metric["num"])

    value = report.get("perf_metrics", {}).get("summary", {}).get("n_samples")
    if isinstance(value, int):
        return value
    raise ValueError("report does not contain a valid sample count")


def load_report(
    output_root: Path,
    dataset: str,
    expected_count: int,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    if dataset not in DATASET_NAMES:
        raise ValueError(f"unsupported dataset: {dataset}")

    report_root = output_root / dataset / "reports"
    candidates = sorted(report_root.glob(f"*/{dataset}.json"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected one {dataset} report under {report_root}, "
            f"found {len(candidates)}"
        )

    report_path = candidates[0]
    with report_path.open(encoding="utf-8") as file:
        report = json.load(file)
    if report.get("dataset_name") != dataset:
        raise ValueError(
            f"{report_path} is for {report.get('dataset_name')!r}, not {dataset!r}"
        )

    score = report.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        raise ValueError(f"{report_path} does not contain a numeric score")
    score = float(score)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{report_path} score is outside [0, 1]: {score}")

    sample_count = _report_sample_count(report)
    if sample_count != expected_count:
        raise ValueError(
            f"{report_path} contains {sample_count} evaluated samples; "
            f"expected {expected_count}"
        )

    perf_summary = report.get("perf_metrics", {}).get("summary", {})
    truncated_outputs = None
    review_files = sorted(
        (output_root / dataset / "reviews").glob("*/*.jsonl")
    )
    if review_files and max_tokens is not None:
        truncated_outputs = 0
        for review_file in review_files:
            with review_file.open(encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    assistant_messages = [
                        message
                        for message in row.get("messages", [])
                        if message.get("role") == "assistant"
                    ]
                    if not assistant_messages:
                        continue
                    output_tokens = (
                        assistant_messages[-1]
                        .get("perf_metrics", {})
                        .get("output_tokens")
                    )
                    if isinstance(output_tokens, int) and output_tokens >= max_tokens:
                        truncated_outputs += 1

    return {
        "dataset": dataset,
        "score": score,
        "samples": sample_count,
        "report": str(report_path),
        "latency_mean_seconds": perf_summary.get("latency", {}).get("mean"),
        "average_output_tps": perf_summary.get("throughput", {}).get(
            "avg_output_tps"
        ),
        "truncated_outputs": truncated_outputs,
    }


def build_summary(
    output_root: Path,
    model_key: str,
    model_name: str,
    expected_counts: dict[str, int],
    thresholds: dict[str, float | None],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    failed = False
    max_tokens = (metadata or {}).get("max_tokens")
    if not isinstance(max_tokens, int):
        max_tokens = None

    for dataset, expected_count in expected_counts.items():
        threshold = thresholds.get(dataset)
        try:
            result = load_report(
                output_root, dataset, expected_count, max_tokens=max_tokens
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            results[dataset] = {
                "dataset": dataset,
                "score": None,
                "threshold": threshold,
                "samples": None,
                "status": "missing",
                "error": str(error),
            }
            failed = True
            continue

        result["threshold"] = threshold
        if threshold is None:
            result["status"] = "report_only"
        elif result["score"] >= threshold:
            result["status"] = "passed"
        else:
            result["status"] = "failed"
            failed = True
        results[dataset] = result

    return {
        "model_key": model_key,
        "model": model_name,
        "status": "failed" if failed else "passed",
        "metadata": metadata or {},
        "datasets": results,
    }


def _parse_json_object(value: str, argument_name: str) -> dict[str, Any]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError(f"{argument_name} must be a JSON object")
    return payload


def _validate_data_command(args: argparse.Namespace) -> int:
    files = {
        "gsm8k_train": validate_jsonl(
            Path(args.gsm8k_train),
            7473,
            "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
        ),
        "gsm8k_test": validate_jsonl(
            Path(args.gsm8k_test),
            1319,
            "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
        ),
        "math_500": validate_jsonl(
            Path(args.math_500),
            500,
            "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132",
        ),
        "humaneval": validate_jsonl(
            Path(args.humaneval),
            164,
            "1d49078ba3e2b196b9344535bef34a43021f038fad9561d6ee7c53450609a6a2",
        ),
    }
    print(json.dumps(files, indent=2))
    return 0


def _summarize_command(args: argparse.Namespace) -> int:
    expected_counts = {
        name: count
        for name, count in (
            ("gsm8k", args.gsm8k_count),
            ("math_500", args.math_count),
            ("humaneval", args.humaneval_count),
        )
        if count is not None
    }
    thresholds = _parse_json_object(args.thresholds_json, "--thresholds-json")
    for dataset, value in thresholds.items():
        if dataset not in DATASET_NAMES:
            raise ValueError(f"unsupported threshold dataset: {dataset}")
        if value is not None and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"invalid threshold for {dataset}: {value!r}")
        thresholds[dataset] = None if value is None else float(value)

    metadata = _parse_json_object(args.metadata_json, "--metadata-json")
    summary = build_summary(
        Path(args.output_root),
        args.model_key,
        args.model_name,
        expected_counts,
        thresholds,
        metadata,
    )
    output_path = Path(args.summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-data")
    validate.add_argument("--gsm8k-train", required=True)
    validate.add_argument("--gsm8k-test", required=True)
    validate.add_argument("--math-500", required=True)
    validate.add_argument("--humaneval", required=True)
    validate.set_defaults(handler=_validate_data_command)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--output-root", required=True)
    summarize.add_argument("--summary", required=True)
    summarize.add_argument("--model-key", required=True)
    summarize.add_argument("--model-name", required=True)
    summarize.add_argument("--gsm8k-count", type=int)
    summarize.add_argument("--math-count", type=int)
    summarize.add_argument("--humaneval-count", type=int)
    summarize.add_argument("--thresholds-json", default="{}")
    summarize.add_argument("--metadata-json", default="{}")
    summarize.set_defaults(handler=_summarize_command)
    return parser


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "evalscope":
        run_evalscope_cli(sys.argv[2:])
        return 0

    args = _build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
