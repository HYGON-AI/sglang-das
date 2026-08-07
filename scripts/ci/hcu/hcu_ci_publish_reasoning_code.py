# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Publish one HCU reasoning/code result to shared CI storage."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Iterable, Optional

SCHEMA_VERSION = 1
SHARED_GID = 1002
RUN_DIR_PATTERN = re.compile(r"run-[0-9]+$")
MODEL_KEY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
OUTCOMES = {"success", "failure", "cancelled", "skipped"}


def _apply_new_path_permissions(path: Path, mode: int) -> None:
    os.chown(path, -1, SHARED_GID)
    path.chmod(mode)


def _ensure_directory(path: Path) -> None:
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    if not cursor.is_dir():
        raise NotADirectoryError(f"shared result path is not a directory: {cursor}")

    for directory in reversed(missing):
        try:
            directory.mkdir()
            _apply_new_path_permissions(directory, 0o2775)
        except FileExistsError:
            if not directory.is_dir():
                raise NotADirectoryError(
                    f"shared result path is not a directory: {directory}"
                )

    if not os.access(path, os.W_OK | os.X_OK):
        raise PermissionError(f"shared result directory is not writable: {path}")


def _write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as output_file:
        json.dump(payload, output_file, sort_keys=True)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    _apply_new_path_permissions(path, 0o664)


def _copy_file(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"reasoning/code input must be a regular file: {source}")
    shutil.copyfile(source, destination)
    _apply_new_path_permissions(destination, 0o664)


def _copy_tree(source: Path, destination: Path) -> list[str]:
    copied = []
    if not source.is_dir():
        return copied
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"reasoning/code input cannot contain symlinks: {path}")
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir()
            _apply_new_path_permissions(target, 0o2775)
            continue
        _copy_file(path, target)
        copied.append(str(relative))
    return copied


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
            print(f"Warning: failed to prune shared result directory {candidate}: {exc}")
    return removed


def publish_model(
    *,
    local_model_dir: Path,
    log_files: Iterable[Path],
    shared_root: Path,
    run_id: int,
    run_attempt: int,
    model_key: str,
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
    if not MODEL_KEY_PATTERN.fullmatch(model_key):
        raise ValueError(f"invalid model_key={model_key!r}")
    if outcome not in OUTCOMES:
        raise ValueError(f"unexpected outcome={outcome!r}")

    previous_umask = os.umask(0o002)
    try:
        _ensure_directory(shared_root)
        prune_expired_runs(
            shared_root,
            retention_days=retention_days,
            current_run_id=run_id,
        )
        attempt_dir = (
            shared_root / f"run-{run_id}" / f"attempt-{run_attempt}"
        )
        _ensure_directory(attempt_dir)

        final_dir = attempt_dir / model_key
        if final_dir.exists():
            raise FileExistsError(f"shared model result already exists: {final_dir}")

        temporary_dir = (
            attempt_dir / f".{model_key}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        )
        temporary_dir.mkdir()
        _apply_new_path_permissions(temporary_dir, 0o2775)
        copied_files = []
        try:
            copied_files.extend(_copy_tree(local_model_dir, temporary_dir))
            existing_logs = [path for path in log_files if path.is_file()]
            if existing_logs:
                logs_dir = temporary_dir / "logs"
                logs_dir.mkdir()
                _apply_new_path_permissions(logs_dir, 0o2775)
                for source in existing_logs:
                    destination = logs_dir / source.name
                    _copy_file(source, destination)
                    copied_files.append(str(destination.relative_to(temporary_dir)))

            _write_json(
                temporary_dir / "status.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "model_key": model_key,
                    "outcome": outcome,
                    "run_id": run_id,
                    "run_attempt": run_attempt,
                    "target_ref": target_ref,
                    "commit_sha": commit_sha,
                    "image_ref": image_ref,
                    "image_id": image_id,
                    "runner_name": runner_name,
                    "files": sorted(copied_files),
                },
            )
            temporary_dir.rename(final_dir)
        except Exception:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise
    finally:
        os.umask(previous_umask)

    print(
        f"Published HCU reasoning/code result: path={final_dir} "
        f"outcome={outcome} files={len(copied_files)}"
    )
    return final_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-model-dir", required=True, type=Path)
    parser.add_argument("--log-file", action="append", default=[], type=Path)
    parser.add_argument("--shared-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--run-attempt", required=True, type=int)
    parser.add_argument("--model-key", required=True)
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
    publish_model(
        local_model_dir=args.local_model_dir,
        log_files=args.log_file,
        shared_root=args.shared_root,
        run_id=args.run_id,
        run_attempt=args.run_attempt,
        model_key=args.model_key,
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
