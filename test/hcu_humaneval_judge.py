#!/usr/bin/env python3

# Copyright (c) OpenAI
# SPDX-License-Identifier: MIT
#
# The execution guard and correctness-check structure below are adapted from
# openai/human-eval 1.0.3. Generated code must still be run inside an external
# restricted container; reliability_guard is not a security sandbox.

import argparse
import contextlib
import faulthandler
import io
import json
import os
import platform
import signal
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Manager, Process
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    def read(self, *args, **kwargs):
        raise OSError("captured output is write-only")

    def readline(self, *args, **kwargs):
        raise OSError("captured output is write-only")

    def readlines(self, *args, **kwargs):
        raise OSError("captured output is write-only")

    def readable(self):
        return False


class RedirectStdin(contextlib._RedirectStream):
    _stream = "stdin"


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("timed out")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with RedirectStdin(stream):
                yield


@contextlib.contextmanager
def temporary_working_directory():
    with tempfile.TemporaryDirectory() as directory:
        current = os.getcwd()
        os.chdir(directory)
        try:
            yield
        finally:
            os.chdir(current)


def reliability_guard(maximum_memory_bytes: int | None = None) -> None:
    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )
        if platform.uname().system != "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK,
                (maximum_memory_bytes, maximum_memory_bytes),
            )

    faulthandler.disable()
    import builtins
    import shutil
    import subprocess
    import sys

    builtins.exit = None
    builtins.quit = None
    builtins.help = None
    os.environ["OMP_NUM_THREADS"] = "1"
    for name in (
        "kill",
        "system",
        "putenv",
        "remove",
        "removedirs",
        "rmdir",
        "fchdir",
        "setuid",
        "fork",
        "forkpty",
        "killpg",
        "rename",
        "renames",
        "truncate",
        "replace",
        "unlink",
        "fchmod",
        "fchown",
        "chmod",
        "chown",
        "chroot",
        "lchflags",
        "lchmod",
        "lchown",
        "getcwd",
        "chdir",
    ):
        if hasattr(os, name):
            setattr(os, name, None)
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None
    subprocess.Popen = None
    for name in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
        sys.modules[name] = None


def unsafe_execute(
    problem: dict[str, Any], completion: str, timeout: float, result
) -> None:
    with temporary_working_directory():
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        reliability_guard(512 * 1024 * 1024)
        check_program = (
            problem["prompt"]
            + completion
            + "\n"
            + problem["test"]
            + "\n"
            + f"check({problem['entry_point']})"
        )
        try:
            with swallow_io(), time_limit(timeout):
                exec(check_program, {})
            result.append("passed")
        except TimeoutException:
            result.append("timed out")
        except BaseException as exc:
            result.append(f"failed: {exc}")
        finally:
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir


def check_correctness(
    problem: dict[str, Any], completion: str, timeout: float, index: int
) -> tuple[int, dict[str, Any]]:
    with Manager() as manager:
        result = manager.list()
        process = Process(
            target=unsafe_execute,
            args=(problem, completion, timeout, result),
        )
        process.start()
        process.join(timeout=timeout + 1)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
        if not result:
            result.append("timed out")
        return index, {
            "task_id": problem["task_id"],
            "passed": result[0] == "passed",
            "result": result[0],
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    samples = read_jsonl(args.samples)
    problems_list = read_jsonl(args.problems)
    problems = {row["task_id"]: row for row in problems_list}
    expected_ids = [row["task_id"] for row in problems_list[: args.expected]]
    sample_ids = [row.get("task_id") for row in samples]
    if len(samples) != args.expected:
        raise AssertionError(f"expected {args.expected} samples, found {len(samples)}")
    if len(set(sample_ids)) != len(sample_ids):
        raise AssertionError("HumanEval samples contain duplicate task IDs")
    if set(sample_ids) != set(expected_ids):
        raise AssertionError("HumanEval samples do not match the expected task IDs")
    if any("completion" not in row for row in samples):
        raise AssertionError("HumanEval sample is missing completion")

    started = time.monotonic()
    ordered_results: list[dict[str, Any] | None] = [None] * len(samples)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                check_correctness,
                problems[sample["task_id"]],
                sample["completion"],
                args.timeout,
                index,
            )
            for index, sample in enumerate(samples)
        ]
        completed = 0
        for future in as_completed(futures):
            index, result = future.result()
            ordered_results[index] = {**samples[index], **result}
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == len(samples):
                print(
                    f"HumanEval judge progress={completed}/{len(samples)}", flush=True
                )

    results = [row for row in ordered_results if row is not None]
    passed = sum(bool(row["passed"]) for row in results)
    timeouts = sum(row["result"] == "timed out" for row in results)
    pass_at_1 = passed / len(results)
    summary = {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_at_1": pass_at_1,
        "timeouts": timeouts,
        "threshold": args.threshold,
        "elapsed_s": round(time.monotonic() - started, 3),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "humaneval_results.jsonl", results)
    (args.output_dir / "humaneval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if pass_at_1 >= args.threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
