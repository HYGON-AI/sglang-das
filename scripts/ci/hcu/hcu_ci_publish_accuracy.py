# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Publish one HCU accuracy partition to shared CI storage."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = 1
RUN_DIR_PATTERN = re.compile(r"run-[0-9]+$")
PARTITIONS = {"accuracy-text-0", "accuracy-text-1"}
OUTCOMES = {"success", "failure", "cancelled", "skipped"}


def _chmod_directory(path: Path) -> None:
    path.chmod(0o2775)


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _chmod_directory(path)


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, sort_keys=True)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    path.chmod(0o664)


def prune_expired_runs(
    shared_root: Path,
    *,
    retention_days: int,
    current_run_id: int,
    now: Optional[float] = None,
) -> list[Path]:
    if retention_days < 1 or not shared_root.is_dir():
        return []
    cutoff = (time.time() if now is None else now) - retention_days * 86400
    current_name = f"run-{current_run_id}"
    removed = []
    for candidate in shared_root.iterdir():
        if (
            candidate.name == current_name
            or not RUN_DIR_PATTERN.fullmatch(candidate.name)
            or candidate.is_symlink()
            or not candidate.is_dir()
        ):
            continue
        try:
            if candidate.stat().st_mtime >= cutoff:
                continue
            shutil.rmtree(candidate)
            removed.append(candidate)
        except FileNotFoundError:
            continue
        except OSError as exc:
            print(
                f"Warning: failed to prune shared result directory {candidate}: {exc}"
            )
    return removed


def publish_partition(
    *,
    local_results_dir: Path,
    shared_root: Path,
    run_id: int,
    run_attempt: int,
    partition: str,
    outcome: str,
    target_ref: str,
    commit_sha: str,
    image_ref: str,
    image_id: str,
    runner_name: str,
    retention_days: int = 14,
) -> Path:
    if run_id < 1 or run_attempt < 1:
        raise ValueError("run_id and run_attempt must be positive")
    if partition not in PARTITIONS:
        raise ValueError(f"unexpected partition: {partition!r}")
    if outcome not in OUTCOMES:
        raise ValueError(f"unexpected partition outcome: {outcome!r}")

    previous_umask = os.umask(0o002)
    try:
        _ensure_directory(shared_root)
        prune_expired_runs(
            shared_root,
            retention_days=retention_days,
            current_run_id=run_id,
        )

        run_dir = shared_root / f"run-{run_id}"
        attempt_dir = run_dir / f"attempt-{run_attempt}"
        _ensure_directory(run_dir)
        _ensure_directory(attempt_dir)

        final_dir = attempt_dir / partition
        if final_dir.exists():
            raise FileExistsError(
                f"shared result partition already exists: {final_dir}"
            )

        temporary_dir = (
            attempt_dir / f".{partition}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        )
        temporary_dir.mkdir(mode=0o2775)
        _chmod_directory(temporary_dir)
        result_files = []
        try:
            if local_results_dir.is_dir():
                for source in sorted(local_results_dir.glob("*.json")):
                    if source.name == "partition-status.json":
                        continue
                    if source.is_symlink() or not source.is_file():
                        continue
                    destination = temporary_dir / source.name
                    shutil.copyfile(source, destination)
                    destination.chmod(0o664)
                    result_files.append(source.name)

            _write_json(
                temporary_dir / "partition-status.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "partition": partition,
                    "outcome": outcome,
                    "run_id": run_id,
                    "run_attempt": run_attempt,
                    "target_ref": target_ref,
                    "commit_sha": commit_sha,
                    "image_ref": image_ref,
                    "image_id": image_id,
                    "runner_name": runner_name,
                    "result_files": result_files,
                },
            )
            temporary_dir.rename(final_dir)
        except Exception:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise
    finally:
        os.umask(previous_umask)

    print(
        f"Published HCU accuracy partition: path={final_dir} "
        f"outcome={outcome} files={len(result_files)}"
    )
    return final_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-results-dir", required=True, type=Path)
    parser.add_argument("--shared-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--run-attempt", required=True, type=int)
    parser.add_argument("--partition", required=True, choices=sorted(PARTITIONS))
    parser.add_argument("--outcome", required=True, choices=sorted(OUTCOMES))
    parser.add_argument("--target-ref", required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--image-ref", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--runner-name", required=True)
    parser.add_argument("--retention-days", type=int, default=14)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    publish_partition(
        local_results_dir=args.local_results_dir,
        shared_root=args.shared_root,
        run_id=args.run_id,
        run_attempt=args.run_attempt,
        partition=args.partition,
        outcome=args.outcome,
        target_ref=args.target_ref,
        commit_sha=args.commit_sha,
        image_ref=args.image_ref,
        image_id=args.image_id,
        runner_name=args.runner_name,
        retention_days=args.retention_days,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
