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


if __name__ == "__main__":
    unittest.main()
