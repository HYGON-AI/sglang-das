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
"""CUDA core dump and py-spy dump utilities."""

from __future__ import annotations

import logging
import os
import platform
import selectors
import signal
import subprocess
import time
from errno import ENXIO
from pathlib import Path
from typing import List

import psutil

logger = logging.getLogger(__name__)


def _resolve_cuda_coredump_pipe_path(proc: psutil.Process) -> Path:
    pipe_template = os.environ.get("CUDA_COREDUMP_PIPE")
    if pipe_template is None:
        pipe_path = f"corepipe.cuda.{platform.node()}.{proc.pid}"
    else:
        pipe_path = (
            pipe_template.replace("%h", platform.node())
            .replace("%p", str(proc.pid))
            .replace("%t", str(int(time.time())))
        )

    path = Path(pipe_path)
    if path.is_absolute():
        return path

    try:
        return Path(proc.cwd()) / path
    except (psutil.Error, OSError):
        return Path.cwd() / path


def _is_sglang_scheduler_process(proc: psutil.Process) -> bool:
    try:
        proc_title = " ".join(proc.cmdline())
    except (psutil.Error, OSError):
        return False
    return proc_title.startswith("sglang::scheduler")


def collect_scheduler_processes() -> List[psutil.Process]:
    current = psutil.Process()
    return [
        proc
        for proc in current.children(recursive=True)
        if _is_sglang_scheduler_process(proc)
    ]


PYSPY_ATTEMPT_SECONDS = 5.0
PYSPY_REAP_SECONDS = 1.0
PYSPY_OUTPUT_LIMIT = 1024 * 1024
CRASH_DIAGNOSTICS_SECONDS = 20.0
_PYSPY_PROCESSES = {}


def _kill_dump_process(proc, deadline):
    # Each dumper owns a separate session. Never signal the target scheduler.
    # Kill the group even if its leader exited: descendants can retain the pipe.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(
            timeout=max(0.0, min(PYSPY_REAP_SECONDS, deadline - time.monotonic()))
        )
    except subprocess.TimeoutExpired:
        logger.error("py-spy PID %s did not reap before cleanup deadline", proc.pid)


def stop_active_pyspy(deadline):
    for proc in list(_PYSPY_PROCESSES.values()):
        _kill_dump_process(proc, deadline)


def _run_pyspy(cmd, deadline, cancel_event=None):
    attempt_deadline = min(deadline, time.monotonic() + PYSPY_ATTEMPT_SECONDS)
    if time.monotonic() >= attempt_deadline or (
        cancel_event is not None and cancel_event.is_set()
    ):
        return dict(returncode=None, timed_out=True, output="", truncated=False)

    # Reserve part of the attempt's budget for killing/reaping the dumper.
    execution_deadline = attempt_deadline - min(
        PYSPY_REAP_SECONDS, max(0.0, attempt_deadline - time.monotonic()) / 2
    )
    proc = subprocess.Popen(
        cmd,
        shell=False,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    _PYSPY_PROCESSES[proc.pid] = proc
    captured = bytearray()
    truncated = False
    timed_out = False
    selector = None
    try:
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        while selector.get_map():
            remaining = execution_deadline - time.monotonic()
            if remaining <= 0 or (cancel_event is not None and cancel_event.is_set()):
                timed_out = True
                break
            for key, _ in selector.select(min(remaining, 0.1)):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                room = PYSPY_OUTPUT_LIMIT - len(captured)
                captured.extend(chunk[:room])
                truncated |= len(chunk) > room
        if not timed_out:
            try:
                proc.wait(timeout=max(0.0, execution_deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out = True
        return dict(
            returncode=proc.returncode,
            timed_out=timed_out,
            output=captured.decode(errors="replace"),
            truncated=truncated,
        )
    finally:
        try:
            _kill_dump_process(proc, attempt_deadline)
        finally:
            if selector is not None:
                selector.close()
            proc.stdout.close()
            _PYSPY_PROCESSES.pop(proc.pid, None)


def pyspy_dump_schedulers(scheduler_only=False, *, deadline=None, cancel_event=None):
    """Dump stacks with a shared deadline and bounded output per attempt.

    A non-native retry is allowed only after an ordinary command failure and
    when a full attempt still fits in the remaining diagnostics budget.
    """
    deadline = (
        deadline
        if deadline is not None
        else time.monotonic() + CRASH_DIAGNOSTICS_SECONDS
    )
    procs = collect_scheduler_processes() if scheduler_only else [psutil.Process()]
    if not procs:
        logger.error("No sglang scheduler processes found for py-spy dump.")
    targets = []
    for proc in procs:
        try:
            if proc.status() != psutil.STATUS_ZOMBIE:
                targets.append((proc.pid, proc.create_time()))
        except psutil.Error:
            pass
    outcomes = []
    for pid, birth in targets:
        for native in (True, False):
            if not native and deadline - time.monotonic() < PYSPY_ATTEMPT_SECONDS:
                return outcomes
            if time.monotonic() >= deadline or (
                cancel_event is not None and cancel_event.is_set()
            ):
                return outcomes
            try:
                target = psutil.Process(pid)
                if (
                    target.create_time() != birth
                    or target.status() == psutil.STATUS_ZOMBIE
                ):
                    break
                cmd = (
                    ["py-spy", "dump"]
                    + (["--native"] if native else [])
                    + ["--pid", str(pid)]
                )
                result = _run_pyspy(cmd, deadline, cancel_event)
                outcomes.append(dict(pid=pid, native=native, **result))
                logger.error(
                    "Pyspy dump PID %s: rc=%s timeout=%s truncated=%s\n%s",
                    pid,
                    result["returncode"],
                    result["timed_out"],
                    result["truncated"],
                    result["output"],
                )
                if result["timed_out"]:
                    return outcomes
                if result["returncode"] is not None and result["returncode"] <= 0:
                    break
            except (OSError, psutil.Error):
                logger.exception("Unable to run py-spy for PID %s", pid)
                break
    return outcomes


def trigger_cuda_user_coredump(scheduler_only=False):
    """Trigger CUDA user-induced GPU core dumps by writing to coredump pipes."""
    if os.environ.get("CUDA_ENABLE_USER_TRIGGERED_COREDUMP") != "1":
        logger.error(
            "CUDA user-triggered coredump is not enabled. Set "
            "CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1 before CUDA initialization."
        )

    if scheduler_only:
        procs = collect_scheduler_processes()
        if not procs:
            logger.error("No sglang scheduler processes found for CUDA coredump.")
            return
    else:
        procs = [psutil.Process()]

    for proc in procs:
        pipe_path = _resolve_cuda_coredump_pipe_path(proc)
        try:
            fd = os.open(pipe_path, os.O_WRONLY | os.O_NONBLOCK)
            try:
                os.write(fd, b"1")
            finally:
                os.close(fd)
            logger.error(
                "Triggered CUDA user coredump for PID %s via %s",
                proc.pid,
                pipe_path,
            )
        except FileNotFoundError:
            logger.error(
                "CUDA coredump pipe not found for PID %s: %s. Ensure "
                "CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1 was set before this "
                "process initialized CUDA.",
                proc.pid,
                pipe_path,
            )
        except OSError as e:
            if e.errno == ENXIO:
                logger.error(
                    "CUDA coredump pipe has no reader for PID %s: %s",
                    proc.pid,
                    pipe_path,
                )
            else:
                logger.exception(
                    "Failed to trigger CUDA user coredump for PID %s via %s",
                    proc.pid,
                    pipe_path,
                )
