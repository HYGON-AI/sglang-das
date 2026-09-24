"""Lightweight ``__main__`` bootstrap for spawned HTTP workers.

This module must remain safe to execute as ``__mp_main__`` before a worker's
health-check thread starts. Keep imports here restricted to the standard
library and do not add package-level SGLang imports.
"""

import importlib
import logging
import sys
import time
from contextlib import contextmanager

_MISSING = object()
_BOOTSTRAP_FILE = __file__


def _restore_attribute(module, name, original_value):
    if original_value is _MISSING:
        try:
            delattr(module, name)
        except AttributeError:
            pass
    else:
        setattr(module, name, original_value)


@contextmanager
def use_multiprocessing_spawn_bootstrap():
    """Make multiprocessing spawn re-execute only this lightweight module.

    ``multiprocessing.spawn`` prefers ``__main__.__spec__.name`` and otherwise
    falls back to ``__main__.__file__``. Overriding both fields covers module
    entrypoints (``python -m sglang.launch_server``), console scripts
    (``sglang serve``), and embedding applications uniformly.
    """
    main_module = sys.modules.get("__main__")
    if main_module is None:
        yield
        return

    original_spec = getattr(main_module, "__spec__", _MISSING)
    original_file = getattr(main_module, "__file__", _MISSING)
    main_module.__spec__ = None
    main_module.__file__ = _BOOTSTRAP_FILE
    try:
        yield
    finally:
        _restore_attribute(main_module, "__spec__", original_spec)
        _restore_attribute(main_module, "__file__", original_file)


def _multiprocess_with_startup_wait(base_class, startup_timeout, logger):
    class MultiprocessWithStartupWait(base_class):
        def _wait_for_worker_startup(self, process, deadline):
            while True:
                self.handle_signals()
                if self.should_exit.is_set():
                    return False
                if not process.process.is_alive():
                    return False

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                if process.is_ready(timeout=min(1, remaining)):
                    return True

                time.sleep(0.1)

        def init_processes(self):
            super().init_processes()

            deadline = time.monotonic() + startup_timeout
            for process in self.processes:
                if self._wait_for_worker_startup(process, deadline):
                    continue

                if not self.should_exit.is_set():
                    logger.error(
                        "Child process [%s] did not finish application startup "
                        "within %.1f seconds; stopping the parent process.",
                        process.pid,
                        startup_timeout,
                    )
                self.should_exit.set()
                return

    return MultiprocessWithStartupWait


@contextmanager
def use_uvicorn_worker_startup_wait(startup_timeout):
    """Give initial workers a startup phase before runtime health checks.

    Uvicorn normally applies its runtime ping timeout immediately after spawning
    workers. A worker can therefore be killed while importing an application if
    a native extension holds the GIL long enough to starve Uvicorn's pong
    thread. Wait for every initial worker to report application readiness before
    entering Uvicorn's normal liveness loop. The runtime health-check timeout is
    unchanged after startup.
    """
    if startup_timeout <= 0:
        raise ValueError("startup_timeout must be positive")

    uvicorn_main = importlib.import_module("uvicorn.main")
    original_multiprocess = uvicorn_main.Multiprocess
    uvicorn_main.Multiprocess = _multiprocess_with_startup_wait(
        original_multiprocess,
        startup_timeout,
        logging.getLogger("uvicorn.error"),
    )
    try:
        yield
    finally:
        uvicorn_main.Multiprocess = original_multiprocess
