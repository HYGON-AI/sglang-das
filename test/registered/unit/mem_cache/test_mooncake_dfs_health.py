import importlib.util
import os
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

_MODULE_PATH = (
    Path(__file__).parents[4]
    / "python/sglang/srt/mem_cache/storage/mooncake_store/dfs_health.py"
)
_SPEC = importlib.util.spec_from_file_location("mooncake_dfs_health", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
AVAILABLE = _MODULE.AVAILABLE
UNAVAILABLE = _MODULE.UNAVAILABLE
MooncakeDfsHealthManager = _MODULE.MooncakeDfsHealthManager
inject_dfs_health_config = _MODULE.inject_dfs_health_config
resolve_dfs_health_config = _MODULE.resolve_dfs_health_config

_CI_MODULE_PATH = (
    Path(__file__).parents[4] / "python/sglang/test/ci/ci_register.py"
)
_CI_SPEC = importlib.util.spec_from_file_location("ci_register", _CI_MODULE_PATH)
assert _CI_SPEC is not None and _CI_SPEC.loader is not None
_CI_MODULE = importlib.util.module_from_spec(_CI_SPEC)
_CI_SPEC.loader.exec_module(_CI_MODULE)
register_cpu_ci = _CI_MODULE.register_cpu_ci
register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeStore:
    def __init__(self, read_probes, write_probes):
        self.read_probes = iter(read_probes)
        self.write_probes = iter(write_probes)
        self.policies = []

    def probe_dfs_read(self):
        return next(self.read_probes)

    def probe_dfs_write(self):
        return next(self.write_probes)

    def set_dfs_access_policy(self, allow_read, allow_write):
        self.policies.append((allow_read, allow_write))
        return 0


class MooncakeDfsHealthManagerTest(unittest.TestCase):
    def test_resolver_uses_cli_then_environment_then_defaults(self):
        env = {
            "SGLANG_MOONCAKE_DFS_HEALTH_PROBE_INTERVAL_S": "4",
            "SGLANG_MOONCAKE_DFS_HEALTH_FAILURE_THRESHOLD": "5",
            "SGLANG_MOONCAKE_DFS_HEALTH_SUCCESS_THRESHOLD": "6",
            "SGLANG_MOONCAKE_DFS_HEALTH_PROBE_TIMEOUT_S": "7",
        }
        server_args = type(
            "ServerArgs",
            (),
            {
                "mooncake_dfs_health_probe_interval_s": 8,
                "mooncake_dfs_health_failure_threshold": None,
                "mooncake_dfs_health_success_threshold": None,
                "mooncake_dfs_health_probe_timeout_s": None,
            },
        )()
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(
                resolve_dfs_health_config(server_args=server_args),
                {
                    "probe_interval_s": 8,
                    "failure_threshold": 5,
                    "success_threshold": 6,
                    "probe_timeout_s": 7,
                },
            )

    def test_resolver_uses_defaults_when_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                resolve_dfs_health_config(),
                {
                    "probe_interval_s": 5,
                    "failure_threshold": 2,
                    "success_threshold": 2,
                    "probe_timeout_s": 5,
                },
            )

    def test_resolver_rejects_non_positive_value(self):
        with patch.dict(
            os.environ,
            {"SGLANG_MOONCAKE_DFS_HEALTH_PROBE_TIMEOUT_S": "0"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "probe_timeout_s must be positive"):
                resolve_dfs_health_config()

    def test_injector_preserves_extra_config_and_uses_cli_values(self):
        server_args = type(
            "ServerArgs",
            (),
            {
                "mooncake_dfs_health_probe_interval_s": 11,
                "mooncake_dfs_health_failure_threshold": 3,
                "mooncake_dfs_health_success_threshold": 4,
                "mooncake_dfs_health_probe_timeout_s": 9,
            },
        )()
        original = {"master_server_address": "127.0.0.1:50051"}

        resolved = inject_dfs_health_config(original, server_args=server_args)

        self.assertEqual(original, {"master_server_address": "127.0.0.1:50051"})
        self.assertEqual(resolved["master_server_address"], "127.0.0.1:50051")
        self.assertEqual(
            resolved["_mooncake_dfs_health_config"],
            {
                "probe_interval_s": 11,
                "failure_threshold": 3,
                "success_threshold": 4,
                "probe_timeout_s": 9,
            },
        )

    def test_injector_keeps_explicit_health_values_over_cli(self):
        server_args = type(
            "ServerArgs",
            (),
            {
                "mooncake_dfs_health_probe_interval_s": 11,
                "mooncake_dfs_health_failure_threshold": 3,
                "mooncake_dfs_health_success_threshold": 4,
                "mooncake_dfs_health_probe_timeout_s": 9,
            },
        )()
        resolved = inject_dfs_health_config(
            {"_mooncake_dfs_health_config": {"probe_interval_s": 17}},
            server_args=server_args,
        )

        health_config = resolved["_mooncake_dfs_health_config"]
        self.assertEqual(health_config["probe_interval_s"], 17)
        self.assertEqual(health_config["failure_threshold"], 3)

    def _manager(self, read_probes, write_probes):
        env = {
            "SGLANG_MOONCAKE_DFS_HEALTH_FAILURE_THRESHOLD": "2",
            "SGLANG_MOONCAKE_DFS_HEALTH_SUCCESS_THRESHOLD": "2",
            "SGLANG_MOONCAKE_DFS_HEALTH_PROBE_INTERVAL_S": "10",
        }
        patcher = patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        store = _FakeStore(read_probes, write_probes)
        return store, MooncakeDfsHealthManager(store)

    def test_write_degrades_without_disabling_dfs_read(self):
        store, manager = self._manager(
            [0, 0, 0, 0],
            [0, 0, -1, -1],
        )
        manager.probe_once()
        manager.probe_once()
        self.assertEqual(store.policies[-1], (True, True))

        manager.probe_once()
        snapshot = manager.probe_once()
        self.assertEqual(snapshot.read_state, AVAILABLE)
        self.assertEqual(snapshot.write_state, UNAVAILABLE)
        self.assertEqual(store.policies[-1], (True, False))

    def test_real_io_timeout_trips_only_affected_capability_immediately(self):
        store, manager = self._manager([0, 0], [0, 0])
        manager.probe_once()
        manager.probe_once()
        self.assertEqual(store.policies[-1], (True, True))

        manager.report_operation_timeout("read", -1607)
        snapshot = manager.snapshot()
        self.assertEqual(snapshot.read_state, UNAVAILABLE)
        self.assertEqual(snapshot.write_state, AVAILABLE)
        self.assertEqual(snapshot.read_error, -1607)
        self.assertEqual(store.policies[-1], (False, True))

        manager.report_operation_timeout("write", -1606)
        snapshot = manager.snapshot()
        self.assertEqual(snapshot.write_state, UNAVAILABLE)
        self.assertEqual(snapshot.write_error, -1606)
        self.assertEqual(store.policies[-1], (False, False))

    def test_real_io_timeout_never_reopens_consensus_disabled_capability(self):
        store, manager = self._manager([0, 0], [0, 0])
        manager.probe_once()
        manager.probe_once()
        manager.apply_effective_policy(True, False)

        manager.report_operation_timeout("read", -1607)
        self.assertEqual(store.policies[-1], (False, False))

    def test_read_and_write_recover_independently(self):
        store, manager = self._manager(
            [-1, -1, 0, 0, 0, 0],
            [-1, -1, -1, -1, 0, 0],
        )
        manager.probe_once()
        manager.probe_once()
        self.assertEqual(store.policies[-1], (False, False))

        manager.probe_once()
        snapshot = manager.probe_once()
        self.assertEqual(snapshot.read_state, AVAILABLE)
        self.assertEqual(snapshot.write_state, UNAVAILABLE)
        self.assertEqual(store.policies[-1], (True, False))

        manager.probe_once()
        snapshot = manager.probe_once()
        self.assertEqual(snapshot.write_state, AVAILABLE)
        self.assertEqual(store.policies[-1], (True, True))

    def test_deferred_policy_waits_for_coordinated_result(self):
        store = _FakeStore([0], [0])
        manager = MooncakeDfsHealthManager(
            store,
            {
                "probe_interval_s": 10,
                "failure_threshold": 1,
                "success_threshold": 1,
                "probe_timeout_s": 1,
            },
            defer_policy_application=True,
        )

        snapshot = manager.probe_once()
        self.assertEqual(snapshot.read_state, AVAILABLE)
        self.assertEqual(snapshot.write_state, AVAILABLE)
        self.assertEqual(manager.local_policy(), (True, True))
        self.assertEqual(store.policies, [(False, False)])

        manager.apply_effective_policy(True, False)
        manager.apply_effective_policy(True, False)
        self.assertEqual(store.policies, [(False, False), (True, False)])

    def test_failed_policy_application_preserves_previous_effective_policy(self):
        class Store(_FakeStore):
            result = 0

            def set_dfs_access_policy(self, allow_read, allow_write):
                self.policies.append((allow_read, allow_write))
                return self.result

        store = Store([0], [0])
        manager = MooncakeDfsHealthManager(
            store,
            {
                "probe_interval_s": 10,
                "failure_threshold": 1,
                "success_threshold": 1,
                "probe_timeout_s": 1,
            },
            defer_policy_application=True,
        )
        manager.apply_effective_policy(True, False)
        self.assertEqual(manager._effective_policy, (True, False))

        store.result = -900
        with self.assertRaisesRegex(RuntimeError, "error=-900"):
            manager.apply_effective_policy(False, False)

        self.assertEqual(manager._effective_policy, (True, False))

    def test_stuck_read_does_not_block_or_fail_write_probe(self):
        release_read = threading.Event()

        class Store(_FakeStore):
            def __init__(self):
                super().__init__([], [0, 0])
                self.read_calls = 0

            def probe_dfs_read(self):
                self.read_calls += 1
                release_read.wait()
                return 0

        store = Store()
        self.addCleanup(release_read.set)
        manager = MooncakeDfsHealthManager(
            store,
            {
                "probe_interval_s": 10,
                "failure_threshold": 1,
                "success_threshold": 1,
                "probe_timeout_s": 1,
            },
        )

        started_at = time.monotonic()
        snapshot = manager.probe_once()

        self.assertLess(time.monotonic() - started_at, 1.5)
        self.assertEqual(snapshot.read_state, UNAVAILABLE)
        self.assertEqual(snapshot.write_state, AVAILABLE)
        self.assertEqual(store.policies[-1], (False, True))
        snapshot = manager.probe_once()
        self.assertEqual(store.read_calls, 1)
        self.assertEqual(snapshot.write_state, AVAILABLE)

    def test_stuck_write_does_not_block_or_fail_read_probe(self):
        release_write = threading.Event()

        class Store(_FakeStore):
            def __init__(self):
                super().__init__([0, 0], [])
                self.write_calls = 0

            def probe_dfs_write(self):
                self.write_calls += 1
                release_write.wait()
                return 0

        store = Store()
        self.addCleanup(release_write.set)
        manager = MooncakeDfsHealthManager(
            store,
            {
                "probe_interval_s": 10,
                "failure_threshold": 1,
                "success_threshold": 1,
                "probe_timeout_s": 1,
            },
        )

        started_at = time.monotonic()
        snapshot = manager.probe_once()

        self.assertLess(time.monotonic() - started_at, 1.5)
        self.assertEqual(snapshot.read_state, AVAILABLE)
        self.assertEqual(snapshot.write_state, UNAVAILABLE)
        self.assertEqual(store.policies[-1], (True, False))
        snapshot = manager.probe_once()
        self.assertEqual(store.write_calls, 1)
        self.assertEqual(snapshot.read_state, AVAILABLE)

    def test_late_probe_success_does_not_reset_timeout_failures(self):
        release_probes = [threading.Event(), threading.Event()]
        first_read_finished = threading.Event()
        first_write_finished = threading.Event()

        class Store(_FakeStore):
            def __init__(self):
                super().__init__([], [])
                self.read_calls = 0
                self.write_calls = 0

            def _probe(self, capability):
                if capability == "read":
                    call_index = self.read_calls
                    self.read_calls += 1
                else:
                    call_index = self.write_calls
                    self.write_calls += 1
                release_probes[call_index].wait()
                if call_index == 0:
                    if capability == "read":
                        first_read_finished.set()
                    else:
                        first_write_finished.set()
                return 0

            def probe_dfs_read(self):
                return self._probe("read")

            def probe_dfs_write(self):
                return self._probe("write")

        store = Store()
        self.addCleanup(release_probes[0].set)
        self.addCleanup(release_probes[1].set)
        manager = MooncakeDfsHealthManager(
            store,
            {
                "probe_interval_s": 10,
                "failure_threshold": 2,
                "success_threshold": 1,
                "probe_timeout_s": 1,
            },
        )

        first = manager.probe_once()
        self.assertEqual(first.read_failures, 1)
        self.assertEqual(first.write_failures, 1)

        release_probes[0].set()
        self.assertTrue(first_read_finished.wait(1))
        self.assertTrue(first_write_finished.wait(1))
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            with manager._probe_lock:
                if all(call is None for call in manager._probe_calls.values()):
                    break
            time.sleep(0.01)
        else:
            self.fail("late probe calls did not release their in-flight slots")

        second = manager.probe_once()
        self.assertEqual(store.read_calls, 2)
        self.assertEqual(store.write_calls, 2)
        self.assertEqual(second.read_state, UNAVAILABLE)
        self.assertEqual(second.write_state, UNAVAILABLE)
        self.assertEqual(second.read_failures, 2)
        self.assertEqual(second.write_failures, 2)


if __name__ == "__main__":
    unittest.main()
