"""Fatal-exit subprocess tests and CPU-only execution of tokenizer entrypoints."""

import ast
import asyncio
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sglang.srt.utils import cudacore_pyspy_dump_utils as crash
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=35, suite="base-a-test-cpu")


class TestFatalExit(unittest.TestCase):
    def run_exit(self, code, *, defaults=False):
        source = repr(crash.__file__)
        setup = (
            "import importlib.util, time\n"
            f"spec = importlib.util.spec_from_file_location('crash', {source})\n"
            "c = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(c)\n"
        )
        if not defaults:
            setup += "c.CRASH_DIAGNOSTICS_SECONDS = 0.3\nc.FATAL_EXIT_SECONDS = 1.0\n"
        start = time.monotonic()
        proc = subprocess.run(
            [sys.executable, "-c", setup + code],
            timeout=26 if defaults else 4,
            capture_output=True,
            text=True,
        )
        elapsed = time.monotonic() - start
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertLess(elapsed, 25 if defaults else 3)
        return proc, elapsed

    def test_blocked_diagnostics_use_real_default_budget(self):
        proc, elapsed = self.run_exit(
            "c.run_fatal_exit(lambda deadline, event: time.sleep(60), lambda: None)",
            defaults=True,
        )
        self.assertGreaterEqual(elapsed, crash.CRASH_DIAGNOSTICS_SECONDS)
        self.assertIn("SGLANG_FATAL_EXIT_BEGIN", proc.stderr)

    def test_blocked_cleanup_uses_fallback(self):
        _, elapsed = self.run_exit(
            "c.run_fatal_exit(lambda deadline, event: None, lambda: time.sleep(60))"
        )
        self.assertGreaterEqual(elapsed, 1.0)

    def test_blocked_logging_uses_fallback(self):
        _, elapsed = self.run_exit(
            "c.logger.error = lambda *args: time.sleep(60)\n"
            "c.run_fatal_exit(lambda deadline, event: None, lambda: None)"
        )
        self.assertGreaterEqual(elapsed, 1.0)

    def test_diagnostic_exception_does_not_skip_cleanup(self):
        proc, _ = self.run_exit(
            "def diagnostics(deadline, event):\n    raise ValueError('diagnostic failed')\n"
            "c.run_fatal_exit(diagnostics, lambda: print('cleanup reached', flush=True))"
        )
        self.assertIn("cleanup reached", proc.stdout)

    def test_dumper_cleanup_exception_does_not_skip_child_cleanup(self):
        proc, _ = self.run_exit(
            "def stop(deadline):\n    raise OSError('dumper cleanup failed')\n"
            "c.stop_active_pyspy = stop\n"
            "c.run_fatal_exit(lambda deadline, event: None, lambda: print('cleanup reached', flush=True))"
        )
        self.assertIn("cleanup reached", proc.stdout)

    def test_child_cleanup_exception_still_exits_nonzero(self):
        self.run_exit(
            "def cleanup():\n    raise RuntimeError('cleanup failed')\n"
            "c.run_fatal_exit(lambda deadline, event: None, cleanup)"
        )

    def test_diagnostics_are_cancelled_before_cleanup(self):
        proc, _ = self.run_exit(
            "events = []\n"
            "def diagnostics(deadline, event):\n    events.append(event)\n    event.wait(60)\n"
            "def cleanup():\n    print(events[0].is_set(), flush=True)\n"
            "c.run_fatal_exit(diagnostics, cleanup)"
        )
        self.assertIn("True", proc.stdout)


class TestFatalRoutes(unittest.TestCase):
    def setUp(self):
        # Execute the real method bodies with dependency stand-ins. Importing
        # TokenizerManager itself requires the model/GPU dependency stack.
        source = Path(crash.__file__).parents[1] / "managers" / "tokenizer_manager.py"
        tree = ast.parse(source.read_text())
        manager = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "TokenizerManager"
        )
        methods = [
            n
            for n in manager.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name
            in ("_fatal_shutdown", "sigterm_watchdog", "dump_requests_before_crash")
        ]
        handler = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "SignalHandler"
        )
        wrapper = next(
            n
            for n in tree.body
            if isinstance(n, ast.AsyncFunctionDef)
            and n.name == "print_exception_wrapper"
        )
        self.exits = []
        self.kill = Mock()
        self.ns = dict(
            os=os,
            sys=sys,
            socket=socket,
            asyncio=asyncio,
            threading=threading,
            time=time,
            logger=logging.getLogger("fatal-route-test"),
            CHILD_CLEANUP_SECONDS=5,
            CRASH_DIAGNOSTICS_SECONDS=20,
            CRASH_SETTLE_SECONDS=0,
            ServerStatus=types.SimpleNamespace(UnHealthy="unhealthy"),
            get_exception_traceback=lambda: "fixture",
            get_bool_env_var=lambda name: False,
            collect_scheduler_processes=lambda: [],
            ShutdownReq=object,
            run_fatal_exit=lambda diagnostics, cleanup: self.exits.append(
                (diagnostics, cleanup)
            ),
            kill_process_tree=self.kill,
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                ast.ClassDef(
                    name="TokenizerManager",
                    bases=[],
                    keywords=[],
                    body=methods,
                    decorator_list=[],
                ),
                handler,
                wrapper,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), self.ns)
        self.manager = self.ns["TokenizerManager"]()
        self.manager._fatal_exit_started = False
        self.manager.server_status = "healthy"
        self.manager._subprocess_watchdog = Mock()
        self.manager.dump_requests_before_crash = Mock()
        self.manager.gracefully_exit = False
        self.manager.rid_to_state = {}
        self.manager._dispatch_to_scheduler = Mock()

    def test_repeated_sigquit_is_idempotent(self):
        handler = self.ns["SignalHandler"](self.manager)
        handler.running_phase_sigquit_handler()
        handler.running_phase_sigquit_handler()
        self.assertEqual(self.manager.server_status, "unhealthy")
        self.assertEqual(len(self.exits), 1)
        diagnostics, cleanup = self.exits[0]
        event = threading.Event()
        diagnostics(42, event)
        self.manager._subprocess_watchdog.stop.assert_called_once()
        self.manager.dump_requests_before_crash.assert_called_once_with(
            deadline=42, cancel_event=event
        )
        cleanup()
        self.kill.assert_called_once_with(
            os.getpid(), include_parent=False, wait_timeout=5
        )

    def test_unhealthy_sigterm_uses_fatal_exit(self):
        self.manager.gracefully_exit = True
        self.manager.server_status = "unhealthy"
        asyncio.run(self.manager.sigterm_watchdog())
        self.assertEqual(len(self.exits), 1)
        self.manager._dispatch_to_scheduler.assert_not_called()

    def test_bound_exception_uses_fatal_exit(self):
        async def failed(manager):
            raise ValueError("fixture")

        asyncio.run(
            self.ns["print_exception_wrapper"](types.MethodType(failed, self.manager))
        )
        self.assertEqual(len(self.exits), 1)
        self.assertEqual(self.manager.server_status, "unhealthy")

    def test_unbound_exception_uses_fatal_exit(self):
        async def failed():
            raise ValueError("fixture")

        asyncio.run(self.ns["print_exception_wrapper"](failed))
        self.assertEqual(len(self.exits), 1)
        self.exits[0][1]()
        self.kill.assert_called_once_with(
            os.getpid(), include_parent=False, wait_timeout=5
        )

    def test_healthy_sigterm_keeps_graceful_path(self):
        handler = self.ns["SignalHandler"](self.manager)
        handler.sigterm_handler()
        with self.assertRaises(SystemExit) as exited:
            asyncio.run(self.manager.sigterm_watchdog())
        self.assertEqual(exited.exception.code, 0)
        self.assertFalse(self.exits)
        self.manager._dispatch_to_scheduler.assert_called_once()
        self.kill.assert_called_once_with(os.getpid(), include_parent=True)

    def test_crash_dump_shares_deadline_and_skips_unsent_cuda_wait(self):
        self.manager.crash_dump_folder = ""
        self.manager.crash_dump_performed = False
        self.ns["envs"] = types.SimpleNamespace(
            SGLANG_PYSPY_DUMP_BEFORE_CRASH=Mock(get=Mock(return_value=True)),
            SGLANG_CUDA_COREDUMP_BEFORE_CRASH=Mock(get=Mock(return_value=True)),
            SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS=Mock(get=Mock(return_value=60)),
        )
        self.ns["collect_scheduler_processes"] = lambda: [object()]
        self.ns["pyspy_dump_schedulers"] = Mock()
        self.ns["trigger_cuda_user_coredump"] = Mock(return_value=False)
        event = Mock(is_set=Mock(return_value=False))
        deadline = time.monotonic() + 2
        self.ns["TokenizerManager"].dump_requests_before_crash(
            self.manager, deadline=deadline, cancel_event=event
        )
        for name in ("pyspy_dump_schedulers", "trigger_cuda_user_coredump"):
            self.ns[name].assert_called_once_with(
                scheduler_only=True, deadline=deadline, cancel_event=event
            )
        event.wait.assert_called_once_with(timeout=0)

    def test_cancelled_dump_does_not_start_diagnostics(self):
        self.manager.crash_dump_folder = ""
        self.manager.crash_dump_performed = False
        self.ns["envs"] = types.SimpleNamespace(
            SGLANG_PYSPY_DUMP_BEFORE_CRASH=Mock(get=Mock(return_value=True)),
            SGLANG_CUDA_COREDUMP_BEFORE_CRASH=Mock(get=Mock(return_value=False)),
        )
        collect = self.ns["collect_scheduler_processes"] = Mock()
        event = threading.Event()
        event.set()
        self.ns["TokenizerManager"].dump_requests_before_crash(
            self.manager, cancel_event=event
        )
        collect.assert_not_called()


class TestCudaDumpBudget(unittest.TestCase):
    def test_disabled_cuda_does_not_open_pipe(self):
        with patch.dict(
            os.environ, {"CUDA_ENABLE_USER_TRIGGERED_COREDUMP": "0"}
        ), patch.object(crash.os, "open") as open_pipe:
            self.assertFalse(crash.trigger_cuda_user_coredump())
            open_pipe.assert_not_called()

    def test_cancelled_and_expired_cuda_dump_does_not_open_pipe(self):
        event = threading.Event()
        event.set()
        with patch.dict(
            os.environ, {"CUDA_ENABLE_USER_TRIGGERED_COREDUMP": "1"}
        ), patch.object(crash.os, "open") as open_pipe:
            self.assertFalse(crash.trigger_cuda_user_coredump(deadline=0))
            self.assertFalse(crash.trigger_cuda_user_coredump(cancel_event=event))
            open_pipe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
