"""Exercise py-spy process handling without tracing or starting a model."""

import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import psutil

from sglang.srt.utils import cudacore_pyspy_dump_utils as crash
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


@unittest.skipUnless(os.name == "posix", "process groups require POSIX")
class TestBoundedPyspy(unittest.TestCase):
    def setUp(self):
        crash._PYSPY_STOPPING.clear()
        self.addCleanup(crash._PYSPY_STOPPING.clear)
        which = patch.object(crash.shutil, "which", return_value="py-spy")
        which.start()
        self.addCleanup(which.stop)
        for name, value in [
            ("PYSPY_NATIVE_SECONDS", 0.8),
            ("PYSPY_PYTHON_SECONDS", 0.8),
            ("PYSPY_REAP_SECONDS", 0.2),
        ]:
            patcher = patch.object(crash, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_dump(self, code, budget=2, cancel_event=None):
        start = time.monotonic()
        result = crash._run_pyspy(
            [sys.executable, "-c", code], start + budget, cancel_event
        )
        self.assertLess(time.monotonic() - start, min(budget, 1.0) + 1.0)
        self.assertFalse(crash._PYSPY_PROCESSES)
        return result

    def assert_dead(self, pid):
        # Container PID 1 may leave a zombie, which can no longer trace anything.
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                    return
            except psutil.NoSuchProcess:
                return
            time.sleep(0.01)
        os.kill(pid, signal.SIGKILL)
        self.fail(f"Dumper descendant {pid} survived cleanup")

    def test_hung_dumper_and_descendant_are_killed(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "child.pid"
            result = self.run_dump(
                "import subprocess,sys,time; from pathlib import Path; "
                "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                f"Path({str(pid_file)!r}).write_text(str(p.pid)); time.sleep(60)"
            )
            self.assertTrue(result["timed_out"])
            self.assert_dead(int(pid_file.read_text()))

    def test_exited_dumper_with_descendant_holding_pipe(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "child.pid"
            result = self.run_dump(
                "import subprocess,sys; from pathlib import Path; "
                "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                f"Path({str(pid_file)!r}).write_text(str(p.pid))"
            )
            self.assertTrue(result["timed_out"])
            self.assert_dead(int(pid_file.read_text()))

    def test_descendant_is_killed_even_after_closing_pipe(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "child.pid"
            result = self.run_dump(
                "import subprocess,sys; from pathlib import Path; "
                "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],"
                "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
                f"Path({str(pid_file)!r}).write_text(str(p.pid))"
            )
            self.assertEqual(result["returncode"], 0)
            self.assert_dead(int(pid_file.read_text()))

    def test_output_is_capped_but_pipe_is_drained(self):
        result = self.run_dump("import sys; sys.stdout.write('x' * (4 * 1024 * 1024))")
        self.assertEqual(result["returncode"], 0)
        self.assertFalse(result["timed_out"])
        self.assertTrue(result["truncated"])
        self.assertEqual(len(result["output"]), crash.PYSPY_OUTPUT_LIMIT)

    def test_short_remaining_budget(self):
        result = self.run_dump("import time; time.sleep(60)", budget=0.3)
        self.assertTrue(result["timed_out"])

    def test_expired_or_cancelled_attempt_does_not_spawn(self):
        event = threading.Event()
        event.set()
        for deadline, cancel_event in [(0, None), (time.monotonic() + 10, event)]:
            with (
                self.subTest(deadline=deadline),
                patch.object(crash.subprocess, "Popen") as spawn,
            ):
                result = crash._run_pyspy(["py-spy"], deadline, cancel_event)
                self.assertTrue(result["timed_out"])
                spawn.assert_not_called()

    def test_cancellation_stops_running_attempt(self):
        event = threading.Event()
        timer = threading.Timer(0.15, event.set)
        timer.start()
        self.addCleanup(timer.cancel)
        result = self.run_dump("import time; time.sleep(60)", cancel_event=event)
        self.assertTrue(result["timed_out"])

    def test_local_failure_and_timeout_retry_without_native(self):
        ordinary = dict(returncode=1, timed_out=False, output="", truncated=False)
        success = dict(ordinary, returncode=0)
        with patch.object(crash, "_run_pyspy", side_effect=[ordinary, success]) as run:
            self.assertEqual(len(crash.pyspy_dump_schedulers()), 2)
            self.assertIn("--native", run.call_args_list[0].args[0])
            self.assertNotIn("--native", run.call_args_list[1].args[0])
        for result in [ordinary, dict(ordinary, timed_out=True)]:
            with (
                self.subTest(result=result),
                patch.object(crash, "_run_pyspy", return_value=result) as run,
            ):
                self.assertEqual(
                    len(crash.pyspy_dump_schedulers(deadline=time.monotonic() + 0.2)), 2
                )
                self.assertEqual(run.call_count, 2)
        with patch.object(
            crash, "_run_pyspy", return_value=dict(ordinary, returncode=-signal.SIGKILL)
        ) as run:
            self.assertEqual(len(crash.pyspy_dump_schedulers()), 1)

    def test_all_targets_share_one_deadline(self):
        target = psutil.Process()
        with (
            patch.object(crash, "collect_scheduler_processes", return_value=[target]),
            patch.object(
                crash,
                "_run_pyspy",
                return_value=dict(
                    returncode=0, timed_out=False, output="", truncated=False
                ),
            ) as run,
        ):
            deadline = time.monotonic() + 10
            self.assertEqual(
                len(crash.pyspy_dump_schedulers(True, deadline=deadline)), 1
            )
            self.assertEqual([call.args[1] for call in run.call_args_list], [deadline])

    def test_gone_zombie_and_reused_pid_are_skipped(self):
        original = Mock(
            pid=123,
            create_time=Mock(return_value=10),
            status=Mock(return_value=psutil.STATUS_RUNNING),
        )
        reused = Mock(
            create_time=Mock(return_value=11),
            status=Mock(return_value=psutil.STATUS_RUNNING),
        )
        with (
            patch.object(crash, "collect_scheduler_processes", return_value=[original]),
            patch.object(crash, "_run_pyspy") as run,
        ):
            with patch.object(
                crash.psutil, "Process", side_effect=psutil.NoSuchProcess(123)
            ):
                self.assertEqual(crash.pyspy_dump_schedulers(True), [])
            with patch.object(crash.psutil, "Process", return_value=reused):
                self.assertEqual(crash.pyspy_dump_schedulers(True), [])
            original.status.return_value = psutil.STATUS_ZOMBIE
            self.assertEqual(crash.pyspy_dump_schedulers(True), [])
            run.assert_not_called()

    def test_missing_pyspy_is_best_effort(self):
        with (
            patch.object(crash.shutil, "which", return_value=None),
            patch.object(crash.logger, "warning") as warn,
            patch.object(crash.subprocess, "Popen") as spawn,
        ):
            self.assertEqual(crash.pyspy_dump_schedulers(), [])
            warn.assert_called_once()
            spawn.assert_not_called()

    def test_selector_failure_still_reaps_dumper(self):
        spawn = crash.subprocess.Popen
        children = []

        def record(*args, **kwargs):
            proc = spawn(*args, **kwargs)
            children.append(proc)
            return proc

        with (
            patch.object(crash.subprocess, "Popen", side_effect=record),
            patch.object(
                crash.selectors,
                "DefaultSelector",
                side_effect=OSError("selector unavailable"),
            ),
        ):
            with self.assertRaises(OSError):
                crash._run_pyspy(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    time.monotonic() + 2,
                )
        self.assertIsNotNone(children[0].poll())
        self.assertFalse(crash._PYSPY_PROCESSES)

    def test_eight_targets_timeout_then_retry_concurrently(self):
        targets = [
            Mock(
                pid=i,
                create_time=Mock(return_value=i),
                status=Mock(return_value=psutil.STATUS_RUNNING),
            )
            for i in range(1, 9)
        ]
        barrier = threading.Barrier(8)
        deadline = time.monotonic() + 2

        def run(cmd, shared_deadline, event, **kwargs):
            self.assertEqual(shared_deadline, deadline)
            if "--native" in cmd:
                barrier.wait(timeout=1)
                return dict(returncode=-9, timed_out=True, output="", truncated=False)
            return dict(returncode=0, timed_out=False, output="stack", truncated=False)

        with (
            patch.object(crash, "collect_scheduler_processes", return_value=targets),
            patch.object(
                crash.psutil, "Process", side_effect=lambda pid: targets[pid - 1]
            ),
            patch.object(crash, "_run_pyspy", side_effect=run),
        ):
            results = crash.pyspy_dump_schedulers(True, deadline=deadline)
        self.assertEqual(len(results), 16)
        self.assertEqual(
            {r["pid"] for r in results if not r["native"]}, set(range(1, 9))
        )

    def test_total_deadline_and_cancellation_prevent_retry(self):
        event = threading.Event()

        def run(*args, **kwargs):
            event.set()
            return dict(returncode=-9, timed_out=True, output="", truncated=False)

        with patch.object(crash, "_run_pyspy", side_effect=run) as mocked:
            self.assertEqual(len(crash.pyspy_dump_schedulers(cancel_event=event)), 1)
            self.assertEqual(mocked.call_count, 1)
        with patch.object(crash, "_run_pyspy") as mocked:
            self.assertEqual(crash.pyspy_dump_schedulers(deadline=0), [])
            mocked.assert_not_called()

    def test_kill_precedes_reap_and_stale_cleanup_cannot_signal(self):
        events = []
        proc = Mock(pid=987654)
        proc._pyspy_lock = threading.Lock()
        proc.wait.side_effect = lambda **kwargs: events.append("wait")
        crash._PYSPY_PROCESSES[proc.pid] = proc
        with patch.object(
            crash.os, "killpg", side_effect=lambda *args: events.append("kill")
        ):
            crash._kill_dump_process(proc, time.monotonic() + 1)
            crash._kill_dump_process(proc, time.monotonic() + 1)
        self.assertEqual(events, ["kill", "wait", "wait"])

    def test_concurrent_stop_and_reader_reap_safely(self):
        timer = threading.Timer(
            0.1, lambda: crash.stop_active_pyspy(time.monotonic() + 1)
        )
        timer.start()
        try:
            self.run_dump("import time; time.sleep(60)")
        finally:
            timer.join(2)
        self.assertFalse(timer.is_alive())

    def test_public_cancellation_reaps_owned_dumpers_before_return(self):
        event = threading.Event()
        original = crash._run_pyspy

        def run(cmd, deadline, cancellation, **kwargs):
            return original(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                deadline,
                cancellation,
                **kwargs,
            )

        timer = threading.Timer(0.15, event.set)
        timer.start()
        try:
            with patch.object(crash, "_run_pyspy", side_effect=run):
                crash.pyspy_dump_schedulers(
                    deadline=time.monotonic() + 2, cancel_event=event
                )
            self.assertFalse(crash._PYSPY_PROCESSES)
        finally:
            timer.join(2)

    def test_non_waitid_observer_does_not_reap(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        try:
            with patch.object(crash.os, "WNOWAIT", create=True):
                # Force the portability path without altering the real OS APIs.
                del crash.os.WNOWAIT
                deadline = time.monotonic() + 2
                while (
                    not crash._exited_without_reaping(proc)
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
                self.assertIsNone(proc.returncode)
                self.assertEqual(
                    psutil.Process(proc.pid).status(), psutil.STATUS_ZOMBIE
                )
        finally:
            proc.wait(timeout=2)


if __name__ == "__main__":
    unittest.main()
