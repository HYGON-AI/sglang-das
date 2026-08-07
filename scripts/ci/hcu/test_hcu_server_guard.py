# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types
import unittest
from unittest import mock

import psutil  # Keep psutil registered while loading the isolated guard module.


def _load_guard_module():
    """Load the guard without importing the full accelerator runtime."""

    def kill_process_tree(pid, wait_timeout=None):
        del wait_timeout
        os.kill(pid, signal.SIGTERM)

    def popen_launch_server(*args, **kwargs):
        raise AssertionError("popen_launch_server must be mocked by this unit test")

    sglang = types.ModuleType("sglang")
    sglang.__path__ = []
    srt = types.ModuleType("sglang.srt")
    srt.__path__ = []
    srt_utils = types.ModuleType("sglang.srt.utils")
    srt_utils.kill_process_tree = kill_process_tree
    test_package = types.ModuleType("sglang.test")
    test_package.__path__ = []
    test_utils = types.ModuleType("sglang.test.test_utils")
    test_utils.popen_launch_server = popen_launch_server

    module_path = (
        Path(__file__).resolve().parents[3]
        / "python"
        / "sglang"
        / "test"
        / "hcu_server_guard.py"
    )
    spec = importlib.util.spec_from_file_location(
        "hcu_server_guard_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    modules = {
        "sglang": sglang,
        "sglang.srt": srt,
        "sglang.srt.utils": srt_utils,
        "sglang.test": test_package,
        "sglang.test.test_utils": test_utils,
    }
    with mock.patch.dict(sys.modules, modules):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


guard = _load_guard_module()


class _FakeProcess:
    def __init__(self, pid=321):
        self.pid = pid
        self.return_code = None
        self.wait_calls = []
        self.killed = False

    def poll(self):
        return self.return_code

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        self.return_code = -9
        return self.return_code

    def kill(self):
        self.killed = True
        self.return_code = -9


class HcuServerIdentityTest(unittest.TestCase):
    def test_preexisting_listener_blocks_launch(self):
        listeners = [
            {"pid": 111, "address": "127.0.0.1:11000", "command": "sglang serve"}
        ]
        with mock.patch.object(
            guard, "_tcp_listeners", return_value=listeners
        ), mock.patch.object(
            guard, "_try_read_model_info", return_value="/models/old"
        ), mock.patch.object(
            guard, "popen_launch_server"
        ) as launch:
            with self.assertRaisesRegex(RuntimeError, "already occupied"):
                with guard.HcuServerGuard(
                    "/models/new", "http://127.0.0.1:11000", timeout=10
                ):
                    pass
        launch.assert_not_called()

    def test_wrong_model_is_rejected(self):
        process = _FakeProcess()
        with mock.patch.object(
            guard, "_read_model_info", return_value={"model_path": "/models/old"}
        ):
            with self.assertRaisesRegex(RuntimeError, "model identity mismatch"):
                guard._assert_server_identity(
                    process,
                    "http://127.0.0.1:11000",
                    "/models/new",
                    None,
                )

    def test_foreign_listener_pid_is_rejected(self):
        process = _FakeProcess(pid=321)
        listeners = [
            {"pid": 999, "address": "127.0.0.1:11000", "command": "sglang serve"}
        ]
        with mock.patch.object(
            guard, "_read_model_info", return_value={"model_path": "/models/new"}
        ), mock.patch.object(
            guard, "_tcp_listeners", return_value=listeners
        ), mock.patch.object(
            guard, "_process_tree_pids", return_value={321, 322}
        ):
            with self.assertRaisesRegex(RuntimeError, "does not belong"):
                guard._assert_server_identity(
                    process,
                    "http://127.0.0.1:11000",
                    "/models/new",
                    None,
                )

    def test_successful_context_stops_server(self):
        process = _FakeProcess()
        with mock.patch.object(guard, "_assert_port_free"), mock.patch.object(
            guard, "popen_launch_server", return_value=process
        ), mock.patch.object(guard, "_assert_server_identity"), mock.patch.object(
            guard.HcuServerGuard, "stop"
        ) as stop:
            with guard.HcuServerGuard(
                "/models/new", "http://127.0.0.1:11000", timeout=10
            ) as server:
                self.assertIs(server.process, process)
        stop.assert_called_once_with()


class HcuServerShutdownTest(unittest.TestCase):
    def test_stop_reaps_real_listener(self):
        process = subprocess.Popen(
            [sys.executable, "-m", "http.server", "11000"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 5
            while not guard._tcp_listeners("http://127.0.0.1:11000"):
                if time.monotonic() >= deadline:
                    self.fail("test HTTP server did not start")
                time.sleep(0.05)

            server = guard.HcuServerGuard(
                "/models/new",
                "http://127.0.0.1:11000",
                timeout=10,
                shutdown_timeout=5,
            )
            server.process = process
            server.stop()
            self.assertIsNotNone(process.poll())
            self.assertFalse(guard._tcp_listeners("http://127.0.0.1:11000"))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)

    def test_stop_waits_for_process_tree_and_port(self):
        process = _FakeProcess()
        server = guard.HcuServerGuard(
            "/models/new", "http://127.0.0.1:11000", timeout=10
        )
        server.process = process
        with mock.patch.object(
            guard, "kill_process_tree"
        ) as kill_tree, mock.patch.object(guard, "_wait_for_port_release") as wait_port:
            server.stop()
        kill_tree.assert_called_once()
        self.assertEqual(kill_tree.call_args.args, (process.pid,))
        self.assertGreater(kill_tree.call_args.kwargs["wait_timeout"], 0)
        self.assertEqual(len(process.wait_calls), 1)
        wait_port.assert_called_once()
        self.assertIsNone(server.process)

    def test_port_release_timeout_is_an_error(self):
        listeners = [
            {"pid": 111, "address": "127.0.0.1:11000", "command": "sglang serve"}
        ]
        with mock.patch.object(guard, "_tcp_listeners", return_value=listeners):
            with self.assertRaisesRegex(RuntimeError, "remained occupied"):
                guard._wait_for_port_release("http://127.0.0.1:11000", timeout=0)

    def test_cleanup_error_does_not_replace_existing_test_error(self):
        server = guard.HcuServerGuard(
            "/models/new", "http://127.0.0.1:11000", timeout=10
        )
        with mock.patch.object(
            server, "stop", side_effect=RuntimeError("cleanup failed")
        ):
            self.assertFalse(
                server.__exit__(ValueError, ValueError("test failed"), None)
            )


if __name__ == "__main__":
    unittest.main()
