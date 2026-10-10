"""Fatal-exit subprocess tests and CPU-only execution of tokenizer entrypoints."""

import ast
import asyncio
import logging
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import psutil

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

    def test_fallback_kills_active_dumper_group_without_logging(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "pids"
            dumper = (
                "import os, subprocess, sys, time; from pathlib import Path; "
                "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
                f"Path({str(pid_file)!r}).write_text(str(os.getpid()) + ' ' + str(child.pid)); "
                "time.sleep(60)"
            )
            self.run_exit(
                "import threading, sys\n"
                "from pathlib import Path\n"
                "c.PYSPY_PYTHON_SECONDS = 60\n"
                f"threading.Thread(target=c._run_pyspy, args=([sys.executable, '-c', {dumper!r}], time.monotonic()+60), daemon=True).start()\n"
                f"while not Path({str(pid_file)!r}).exists(): time.sleep(0.01)\n"
                "c.logger.error = lambda *args: time.sleep(60)\n"
                "c.run_fatal_exit(lambda deadline,event: None, lambda: None)"
            )
            for pid in map(int, pid_file.read_text().split()):
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    try:
                        if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                            break
                    except psutil.NoSuchProcess:
                        break
                    time.sleep(0.01)
                else:
                    os.kill(pid, 9)
                    self.fail(f"Dumper {pid} survived fallback")

    def test_cuda_extension_preserves_wait_and_extended_fallback(self):
        for block_cleanup in (False, True):
            with self.subTest(block_cleanup=block_cleanup):
                proc, elapsed = self.run_exit(
                    "budget = c.FatalExitBudget()\n"
                    "def diagnostics(deadline,event):\n"
                    "    time.sleep(0.1)\n"
                    "    budget.cuda_triggered(0.6)\n"
                    "    event.wait(max(0, budget.cuda_wait_deadline-time.monotonic()))\n"
                    "    print('wait completed', flush=True)\n"
                    f"c.run_fatal_exit(diagnostics, lambda: time.sleep({60 if block_cleanup else 0}), budget=budget)"
                )
                self.assertIn("wait completed", proc.stdout)
                self.assertGreaterEqual(elapsed, 1.6 if block_cleanup else 0.7)

    def test_diagnostic_failure_after_cuda_trigger_preserves_wait(self):
        proc, elapsed = self.run_exit(
            "budget = c.FatalExitBudget()\n"
            "def diagnostics(deadline,event):\n"
            "    budget.cuda_triggered(0.6)\n"
            "    raise ValueError('failure after trigger')\n"
            "c.run_fatal_exit(diagnostics, lambda: print('cleanup', flush=True), budget=budget)"
        )
        self.assertGreaterEqual(elapsed, 0.6)
        self.assertIn("cleanup", proc.stdout)

    def test_force_hook_failure_and_blocking_still_exit(self):
        for hook in ("raise RuntimeError('hook')", "time.sleep(60)"):
            self.run_exit(
                "def hook():\n    " + hook + "\n"
                "def diagnostics(deadline,event):\n"
                "    try: pass\n"
                "    finally: hook()\n"
                "c.run_fatal_exit(diagnostics, lambda: None)"
            )


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
            FatalExitBudget=crash.FatalExitBudget,
            ServerStatus=types.SimpleNamespace(UnHealthy="unhealthy"),
            get_exception_traceback=lambda: "fixture",
            get_bool_env_var=lambda name: False,
            collect_scheduler_processes=lambda: [],
            ShutdownReq=type("ShutdownReq", (), {}),
            run_fatal_exit=lambda diagnostics, cleanup, **kwargs: self.exits.append(
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
        self.manager._server_stop_hook = None
        self.manager.asyncio_tasks = []
        self.manager.force_exit_handler = Mock()

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
        self.manager.dump_requests_before_crash.assert_called_once()
        self.assertEqual(
            self.manager.dump_requests_before_crash.call_args.kwargs["deadline"], 42
        )
        self.assertIs(
            self.manager.dump_requests_before_crash.call_args.kwargs["cancel_event"],
            event,
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
        self.assertIsInstance(
            self.manager._dispatch_to_scheduler.call_args.args[0],
            self.ns["ShutdownReq"],
        )

    def test_healthy_sigterm_with_server_stop_hook(self):
        self.manager._server_stop_hook = Mock()
        self.ns["SignalHandler"](self.manager).sigterm_handler()
        asyncio.run(self.manager.sigterm_watchdog())
        self.assertFalse(self.exits)
        self.manager._dispatch_to_scheduler.assert_called_once()
        self.assertIsInstance(
            self.manager._dispatch_to_scheduler.call_args.args[0],
            self.ns["ShutdownReq"],
        )
        self.manager._server_stop_hook.assert_called_once()

    def test_unhealthy_sigterm_keeps_force_exit_hook_after_dump_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                self.manager._fatal_exit_started = False
                self.manager.force_exit_handler.reset_mock()
                self.manager.gracefully_exit = True
                self.manager.server_status = "unhealthy"
                self.manager.dump_requests_before_crash.side_effect = (
                    ValueError("dump") if fail else None
                )
                asyncio.run(self.manager.sigterm_watchdog())
                diagnostics, _ = self.exits[-1]
                if fail:
                    with self.assertRaises(ValueError):
                        diagnostics(time.monotonic() + 1, threading.Event())
                else:
                    diagnostics(time.monotonic() + 1, threading.Event())
                self.manager.force_exit_handler.assert_called_once()

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
            self.ns[name].assert_called_once()
            kwargs = self.ns[name].call_args.kwargs
            self.assertEqual(kwargs["deadline"], deadline)
            self.assertIs(kwargs["cancel_event"], event)
        event.wait.assert_called_once_with(timeout=0)

    def test_successful_cuda_dump_uses_full_wait_beyond_original_deadline(self):
        self.manager.crash_dump_folder = ""
        self.manager.crash_dump_performed = False
        self.ns["envs"] = types.SimpleNamespace(
            SGLANG_PYSPY_DUMP_BEFORE_CRASH=Mock(get=Mock(return_value=False)),
            SGLANG_CUDA_COREDUMP_BEFORE_CRASH=Mock(get=Mock(return_value=True)),
            SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS=Mock(get=Mock(return_value=60)),
        )
        self.ns["collect_scheduler_processes"] = lambda: [object()]

        def trigger(**kwargs):
            kwargs["on_trigger"]()
            return True

        self.ns["trigger_cuda_user_coredump"] = trigger
        event = Mock(is_set=Mock(return_value=False))
        budget = crash.FatalExitBudget()
        initial_exit = budget.exit_deadline
        self.ns["TokenizerManager"].dump_requests_before_crash(
            self.manager,
            deadline=time.monotonic() + 1,
            cancel_event=event,
            budget=budget,
        )
        self.assertEqual(budget.exit_deadline, initial_exit + 60)
        self.assertGreater(event.wait.call_args.kwargs["timeout"], 59)

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
        with (
            patch.dict(os.environ, {"CUDA_ENABLE_USER_TRIGGERED_COREDUMP": "0"}),
            patch.object(crash.os, "open") as open_pipe,
        ):
            self.assertFalse(crash.trigger_cuda_user_coredump())
            open_pipe.assert_not_called()

    def test_cancelled_and_expired_cuda_dump_does_not_open_pipe(self):
        event = threading.Event()
        event.set()
        with (
            patch.dict(os.environ, {"CUDA_ENABLE_USER_TRIGGERED_COREDUMP": "1"}),
            patch.object(crash.os, "open") as open_pipe,
        ):
            self.assertFalse(crash.trigger_cuda_user_coredump(deadline=0))
            self.assertFalse(crash.trigger_cuda_user_coredump(cancel_event=event))
            open_pipe.assert_not_called()


class TestBudgetCoordination(unittest.TestCase):
    def test_only_success_extends_once_and_last_trigger_sets_wait(self):
        with patch.object(crash.time, "monotonic", return_value=100):
            budget = crash.FatalExitBudget()
        self.assertEqual(
            (budget.diagnostics_deadline, budget.exit_deadline), (120, 130)
        )
        with patch.object(crash.time, "monotonic", return_value=110):
            budget.cuda_triggered(60)
        with patch.object(crash.time, "monotonic", return_value=112):
            budget.cuda_triggered(60)
        self.assertEqual(
            (budget.diagnostics_deadline, budget.exit_deadline), (180, 190)
        )
        self.assertEqual(budget.cuda_wait_deadline, 172)

    def test_expired_or_closed_budget_cannot_be_reopened(self):
        with patch.object(crash.time, "monotonic", return_value=100):
            budget = crash.FatalExitBudget()
        with patch.object(crash.time, "monotonic", return_value=121):
            budget.cuda_triggered(60)
        self.assertEqual(budget.exit_deadline, 130)
        budget.finish()
        budget.wait_for_diagnostics()
        with patch.object(crash.time, "monotonic", return_value=110):
            budget.cuda_triggered(60)
        self.assertEqual(budget.exit_deadline, 130)

    def test_pipe_success_extends_before_log_and_failures_do_not(self):
        for failure in (None, FileNotFoundError(), OSError(crash.ENXIO, "no reader")):
            with self.subTest(failure=failure):
                budget = crash.FatalExitBudget()
                initial = budget.exit_deadline
                callback = Mock(side_effect=lambda: budget.cuda_triggered(60))

                def logged(*args):
                    self.assertEqual(
                        budget.exit_deadline, initial + (60 if failure is None else 0)
                    )

                with (
                    patch.dict(
                        os.environ, {"CUDA_ENABLE_USER_TRIGGERED_COREDUMP": "1"}
                    ),
                    patch.object(
                        crash.os, "open", return_value=123, side_effect=failure
                    ),
                    patch.object(crash.os, "write", return_value=1),
                    patch.object(crash.os, "close"),
                    patch.object(crash.logger, "error", side_effect=logged),
                ):
                    triggered = crash.trigger_cuda_user_coredump(on_trigger=callback)
                self.assertEqual(triggered, failure is None)
                self.assertEqual(callback.call_count, int(failure is None))

    def test_extension_wakes_waiting_coordinator(self):
        with patch.object(crash, "CRASH_DIAGNOSTICS_SECONDS", 0.15):
            budget = crash.FatalExitBudget()
        waiter = threading.Thread(target=budget.wait_for_diagnostics)
        waiter.start()
        try:
            budget.cuda_triggered(0.5)
            time.sleep(0.2)
            self.assertTrue(waiter.is_alive())
        finally:
            budget.finish()
            waiter.join(1)
        self.assertFalse(waiter.is_alive())


class TestRouterFatalRoutes(unittest.TestCase):
    def setUp(self):
        source = (
            Path(crash.__file__).parents[1] / "managers" / "multi_tokenizer_mixin.py"
        )
        tree = ast.parse(source.read_text())
        router = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "MultiTokenizerRouter"
        )
        method = next(
            n
            for n in router.body
            if isinstance(n, ast.FunctionDef) and n.name == "_fatal_shutdown"
        )
        wrapper = next(
            n
            for n in tree.body
            if isinstance(n, ast.AsyncFunctionDef)
            and n.name == "print_exception_wrapper"
        )
        self.exits = []
        self.logger = Mock()
        self.kill = Mock()
        self.ns = dict(
            os=os,
            logger=self.logger,
            get_exception_traceback=lambda: "fixture",
            CHILD_CLEANUP_SECONDS=5,
            kill_process_tree=self.kill,
            run_fatal_exit=lambda diagnostics, cleanup: self.exits.append(
                (diagnostics, cleanup)
            ),
        )
        module = ast.Module(
            body=[
                ast.ClassDef(
                    name="MultiTokenizerRouter",
                    bases=[],
                    keywords=[],
                    body=[method],
                    decorator_list=[],
                ),
                wrapper,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), self.ns)
        self.router = self.ns["MultiTokenizerRouter"]()
        self.router._fatal_exit_started = False

    def test_bound_router_failure_is_idempotent_without_dump_method(self):
        async def failed(router):
            raise ValueError("fixture")

        self.assertFalse(hasattr(self.router, "dump_requests_before_crash"))
        for _ in range(2):
            asyncio.run(
                self.ns["print_exception_wrapper"](
                    types.MethodType(failed, self.router)
                )
            )
        self.assertEqual(len(self.exits), 1)
        self.logger.error.assert_not_called()
        diagnostics, cleanup = self.exits[0]
        diagnostics(0, threading.Event())
        cleanup()
        self.logger.error.assert_called_once()
        self.kill.assert_called_once()

    def test_unbound_router_failure_also_uses_fatal_entry(self):
        async def failed():
            raise ValueError("fixture")

        asyncio.run(self.ns["print_exception_wrapper"](failed))
        self.assertEqual(len(self.exits), 1)
        self.logger.error.assert_not_called()


if __name__ == "__main__":
    unittest.main()
