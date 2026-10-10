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
import math
import os
import platform
import selectors
import shutil
import signal
import subprocess
import threading
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


PYSPY_NATIVE_SECONDS = 8.0
PYSPY_PYTHON_SECONDS = 4.0
PYSPY_REAP_SECONDS = 1.0
PYSPY_OUTPUT_LIMIT = 1024 * 1024
CRASH_DIAGNOSTICS_SECONDS = 20.0
CRASH_SETTLE_SECONDS = 5.0
FATAL_EXIT_SECONDS = 30.0
CHILD_CLEANUP_SECONDS = 5.0
_PYSPY_PROCESSES = {}
_PYSPY_STOPPING = threading.Event()


def _group_has_live_members(pgid):
    for member in psutil.process_iter():
        try:
            if (
                os.getpgid(member.pid) == pgid
                and member.status() != psutil.STATUS_ZOMBIE
            ):
                return True
        except (ProcessLookupError, psutil.NoSuchProcess):
            continue
        except (PermissionError, psutil.AccessDenied):
            return True  # Cannot prove that the group is empty.
    return False


def _kill_dump_process(proc, deadline):
    # The leader must remain unreaped until its group has been signalled. The
    # per-process lock also prevents a concurrent cleanup from using a stale PID.
    with proc._pyspy_lock:
        if _PYSPY_PROCESSES.get(proc.pid) is proc:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                # Darwin can return EPERM for a group containing only zombies.
                # Do not suppress a genuine failure to signal a live descendant.
                if platform.system() != "Darwin" or _group_has_live_members(proc.pid):
                    raise
            _PYSPY_PROCESSES.pop(proc.pid, None)
    try:
        proc.wait(
            timeout=max(0.0, min(PYSPY_REAP_SECONDS, deadline - time.monotonic()))
        )
    except subprocess.TimeoutExpired:
        logger.error("py-spy PID %s did not reap before cleanup deadline", proc.pid)


def stop_active_pyspy(deadline):
    _PYSPY_STOPPING.set()
    for proc in list(_PYSPY_PROCESSES.values()):
        _kill_dump_process(proc, deadline)


def _emergency_exit():
    # Best effort only: do not wait for locks, subprocesses, or log sinks here.
    try:
        _PYSPY_STOPPING.set()
        for proc in list(_PYSPY_PROCESSES.values()):
            if not proc._pyspy_lock.acquire(blocking=False):
                continue
            try:
                if _PYSPY_PROCESSES.get(proc.pid) is proc:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except OSError:
                        pass
            finally:
                proc._pyspy_lock.release()
    finally:
        os._exit(1)


def _exited_without_reaping(proc):
    if hasattr(os, "waitid") and hasattr(os, "WNOWAIT"):
        # Neither Popen.poll() nor wait() is safe before killpg: both reap.
        return (
            os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            is not None
        )
    # macOS Python does not expose waitid. An exited, unreaped child is a zombie.
    return psutil.Process(proc.pid).status() == psutil.STATUS_ZOMBIE


def _run_pyspy(cmd, deadline, cancel_event=None, *, processes=None):
    execution_seconds = (
        PYSPY_NATIVE_SECONDS if "--native" in cmd else PYSPY_PYTHON_SECONDS
    )
    now = time.monotonic()
    attempt_deadline = min(deadline, now + execution_seconds + PYSPY_REAP_SECONDS)

    def cancelled():
        return _PYSPY_STOPPING.is_set() or (
            cancel_event is not None and cancel_event.is_set()
        )

    if now >= attempt_deadline or cancelled():
        return dict(returncode=None, timed_out=True, output="", truncated=False)
    execution_deadline = min(
        now + execution_seconds,
        attempt_deadline - min(PYSPY_REAP_SECONDS, (attempt_deadline - now) / 2),
    )
    proc = subprocess.Popen(
        cmd,
        shell=False,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    proc._pyspy_lock = threading.Lock()
    _PYSPY_PROCESSES[proc.pid] = proc
    if processes is not None:
        processes.append(proc)
    captured = bytearray()
    truncated = False
    timed_out = False
    selector = None
    try:
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        while True:
            remaining = execution_deadline - time.monotonic()
            if remaining <= 0 or cancelled():
                timed_out = True
                break
            if not selector.get_map():
                # Serialize observation with all paths that can reap this child.
                with proc._pyspy_lock:
                    if _PYSPY_PROCESSES.get(
                        proc.pid
                    ) is not proc or _exited_without_reaping(proc):
                        break
                time.sleep(min(remaining, 0.01))
                continue
            for key, _ in selector.select(min(remaining, 0.1)):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                room = PYSPY_OUTPUT_LIMIT - len(captured)
                captured.extend(chunk[:room])
                truncated |= len(chunk) > room
    finally:
        try:
            _kill_dump_process(proc, attempt_deadline)
        finally:
            if selector is not None:
                selector.close()
            proc.stdout.close()
    return dict(
        returncode=proc.returncode,
        timed_out=timed_out,
        output=captured.decode(errors="replace"),
        truncated=truncated,
    )


def pyspy_dump_schedulers(scheduler_only=False, *, deadline=None, cancel_event=None):
    """Dump each PID concurrently under one deadline, retrying without native.

    A local timeout or ordinary failure allows a non-native attempt. Cancellation
    and the shared deadline stop all work; partial remaining budgets are usable.
    """
    deadline = (
        deadline
        if deadline is not None
        else time.monotonic() + CRASH_DIAGNOSTICS_SECONDS
    )
    executable = shutil.which("py-spy")
    if executable is None:
        logger.warning("py-spy is not installed; skipping stack dumps")
        return []
    procs = collect_scheduler_processes() if scheduler_only else [psutil.Process()]
    targets = {}
    for proc in procs:
        try:
            if proc.status() != psutil.STATUS_ZOMBIE:
                targets[proc.pid] = proc.create_time()
        except psutil.Error:
            pass
    if not targets:
        logger.error("No live processes found for py-spy dump.")
    stopped = threading.Event()

    class Cancellation:
        def is_set(self):
            return (
                stopped.is_set()
                or _PYSPY_STOPPING.is_set()
                or (cancel_event is not None and cancel_event.is_set())
            )

    cancellation = Cancellation()
    outcomes = {pid: [] for pid in targets}
    processes = []

    def dump(pid, birth):
        for native in (True, False):
            if time.monotonic() >= deadline or cancellation.is_set():
                break
            try:
                target = psutil.Process(pid)
                if (
                    target.create_time() != birth
                    or target.status() == psutil.STATUS_ZOMBIE
                ):
                    break
                cmd = (
                    [executable, "dump"]
                    + (["--native"] if native else [])
                    + ["--pid", str(pid)]
                )
                result = _run_pyspy(cmd, deadline, cancellation, processes=processes)
                outcomes[pid].append(dict(pid=pid, native=native, **result))
                logger.error(
                    "Pyspy dump PID %s: rc=%s timeout=%s truncated=%s\n%s",
                    pid,
                    result["returncode"],
                    result["timed_out"],
                    result["truncated"],
                    result["output"],
                )
                if (
                    not result["timed_out"]
                    and result["returncode"] is not None
                    and result["returncode"] <= 0
                ):
                    break
            except (OSError, psutil.Error):
                logger.exception("Unable to run py-spy for PID %s", pid)
                break

    # Daemon threads avoid executor shutdown joining a blocked log sink forever.
    workers = []
    join_deadline = deadline - min(
        PYSPY_REAP_SECONDS, max(0.0, deadline - time.monotonic()) / 2
    )
    try:
        for pid, birth in targets.items():
            if cancellation.is_set() or time.monotonic() >= deadline:
                break
            worker = threading.Thread(target=dump, args=(pid, birth), daemon=True)
            worker.start()
            workers.append(worker)
        for worker in workers:
            while worker.is_alive() and not cancellation.is_set():
                remaining = join_deadline - time.monotonic()
                if remaining <= 0:
                    break
                worker.join(timeout=min(remaining, 0.1))
    finally:
        stopped.set()
        # Reap this invocation's dumpers before returning, including cancellation.
        # Keep other concurrent diagnostic invocations independent.
        try:
            for proc in list(processes):
                _kill_dump_process(proc, deadline)
        finally:
            for worker in workers:
                worker.join(timeout=max(0.0, deadline - time.monotonic()))
    # Snapshot each list: late completion must not mutate the returned result.
    return [result for results in outcomes.values() for result in list(results)]


class FatalExitBudget:
    """Coordinate one CUDA wait extension with both fatal-exit deadlines."""

    def __init__(self):
        now = time.monotonic()
        self.pyspy_deadline = now + CRASH_DIAGNOSTICS_SECONDS
        self.diagnostics_deadline = self.pyspy_deadline
        self.exit_deadline = now + FATAL_EXIT_SECONDS
        self.cuda_wait_deadline = 0.0
        self._extended = False
        self._closed = False
        self._done = False
        self._condition = threading.Condition()

    def cuda_triggered(self, wait_seconds):
        # Called immediately after a successful pipe write, before any logging.
        wait_seconds = max(0.0, wait_seconds) if math.isfinite(wait_seconds) else 0.0
        with self._condition:
            now = time.monotonic()
            if self._closed or now >= self.diagnostics_deadline:
                return
            if not self._extended:
                self.diagnostics_deadline += wait_seconds
                self.exit_deadline += wait_seconds
                self._extended = True
            self.cuda_wait_deadline = max(self.cuda_wait_deadline, now + wait_seconds)
            self._condition.notify_all()

    def finish(self):
        with self._condition:
            self._done = True
            self._condition.notify_all()

    def wait_for_diagnostics(self):
        with self._condition:
            while True:
                now = time.monotonic()
                remaining = self.diagnostics_deadline - now
                if self._done:
                    # A later diagnostic exception must not cut short a CUDA
                    # dump that was already triggered successfully.
                    remaining = min(remaining, self.cuda_wait_deadline - now)
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            self._closed = True

    def enforce_exit(self):
        # The timer is never cancelled/replaced when CUDA extends its deadline.
        with self._condition:
            while True:
                remaining = self.exit_deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
        _emergency_exit()


def run_fatal_exit(diagnostics, cleanup, *, budget=None):
    """Bound diagnostics/cleanup; requires a runnable Python interpreter."""
    budget = budget if budget is not None else FatalExitBudget()
    timer = threading.Thread(target=budget.enforce_exit, name="fatal-exit", daemon=True)
    timer.start()  # Before logging or diagnostics, either of which may block.
    logger.error("SGLANG_FATAL_EXIT_BEGIN pid=%s", os.getpid())
    cancelled = threading.Event()

    def work():
        try:
            diagnostics(budget.pyspy_deadline, cancelled)
        except BaseException:
            logger.exception("Crash diagnostics failed; fatal exit will continue")
        finally:
            budget.finish()

    worker = threading.Thread(target=work, name="fatal-diagnostics", daemon=True)
    try:
        worker.start()
        budget.wait_for_diagnostics()
    finally:
        cancelled.set()
        try:
            try:
                stop_active_pyspy(time.monotonic() + PYSPY_REAP_SECONDS)
            finally:
                cleanup()
        except BaseException:
            logger.exception("Fatal child cleanup failed")
        finally:
            _emergency_exit()


def trigger_cuda_user_coredump(
    scheduler_only=False, *, deadline=None, cancel_event=None, on_trigger=None
):
    """Trigger CUDA user-induced GPU core dumps by writing to coredump pipes."""
    if os.environ.get("CUDA_ENABLE_USER_TRIGGERED_COREDUMP") != "1":
        logger.error(
            "CUDA user-triggered coredump is not enabled. Set "
            "CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1 before CUDA initialization."
        )
        return False

    if scheduler_only:
        procs = collect_scheduler_processes()
        if not procs:
            logger.error("No sglang scheduler processes found for CUDA coredump.")
            return False
    else:
        procs = [psutil.Process()]

    triggered = False
    for proc in procs:
        if (deadline is not None and time.monotonic() >= deadline) or (
            cancel_event is not None and cancel_event.is_set()
        ):
            break
        pipe_path = _resolve_cuda_coredump_pipe_path(proc)
        try:
            fd = os.open(pipe_path, os.O_WRONLY | os.O_NONBLOCK)
            try:
                os.write(fd, b"1")
                triggered = True
                if on_trigger is not None:
                    on_trigger()
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
    return triggered
