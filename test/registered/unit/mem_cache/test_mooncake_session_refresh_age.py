"""Admission may skip the lease refresh of a freshly leased prefetch session.

``SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S`` lets ``revalidate_host_prefetch``
trust a private session whose oldest key was leased less than that many
seconds ago. The age is measured from a monotonic timestamp taken before the
RPC that granted the lease, so it never understates the lease Mooncake sees.
"""

import os
import threading
import types
import unittest
from unittest import mock
from unittest.mock import MagicMock

from sglang.srt.mem_cache.storage.mooncake_store import mooncake_direct_linker
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _ContendedLock:
    """A lock that signals when a thread starts waiting for it while held."""

    def __init__(self):
        self._lock = threading.Lock()
        self.contended = threading.Event()

    def __enter__(self):
        if self._lock.locked():
            self.contended.set()
        self._lock.acquire()
        return self

    def __exit__(self, *exc):
        self._lock.release()


class _Clock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


def _linker(age_s=5.0, refresh_results=None, refresh_error=None):
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.session_refresh_age_s = age_s
    linker.host_prefetch_enabled = True
    linker.host_prefetch_lock = threading.Lock()
    linker.host_prefetch_entries = {}
    linker.session_lock = threading.Lock()
    linker.prepared_load_sessions = {}
    linker.session_refcounts = {}
    linker.session_sources = {}

    def start(keys):
        return [0] * len(keys), ["dfs"] * len(keys)

    def refresh(keys):
        if refresh_error is not None:
            raise refresh_error
        if refresh_results is not None:
            return list(refresh_results)
        return [0] * len(keys)

    store = types.SimpleNamespace(
        batch_get_session_start_with_sources=MagicMock(side_effect=start),
        batch_get_session_refresh=MagicMock(side_effect=refresh),
        batch_get_session_end=MagicMock(return_value=0),
    )
    linker.storage = types.SimpleNamespace(
        store=store,
        _get_hybrid_page_component_keys=lambda keys, transfer: (keys, None),
        _tag_keys=lambda keys: list(keys),
    )
    return linker


def _prefetched(linker, clock, rid="rid", keys=("k0", "k1"), session="s"):
    transfer = types.SimpleNamespace(keys=list(keys))
    with mock.patch.object(mooncake_direct_linker.time, "monotonic", clock):
        assert linker._prepare_expanded_load(session, [transfer])
    linker.host_prefetch_entries[rid] = {
        "state": "dfs_prefetched",
        "session_rid": session,
    }


def _revalidate(linker, clock, rid="rid"):
    with mock.patch.object(mooncake_direct_linker.time, "monotonic", clock):
        return linker.revalidate_host_prefetch(rid)


class TestSessionRefreshAge(CustomTestCase):
    def test_young_private_session_skips_refresh(self):
        clock = _Clock()
        linker = _linker()
        _prefetched(linker, clock)
        clock.now += 4.9
        self.assertTrue(_revalidate(linker, clock))
        linker.storage.store.batch_get_session_refresh.assert_not_called()

    def test_old_session_refreshes_and_resets_its_base(self):
        clock = _Clock()
        linker = _linker()
        _prefetched(linker, clock)
        clock.now += 5.0
        self.assertTrue(_revalidate(linker, clock))
        linker.storage.store.batch_get_session_refresh.assert_called_once_with(
            ["k0", "k1"]
        )
        self.assertEqual(linker.session_lease_base, {"k0": 105.0, "k1": 105.0})
        # Freshly refreshed, so the next admission can skip.
        clock.now += 1.0
        self.assertTrue(_revalidate(linker, clock))
        linker.storage.store.batch_get_session_refresh.assert_called_once()

    def test_disabled_always_refreshes(self):
        clock = _Clock()
        linker = _linker(age_s=0.0)
        _prefetched(linker, clock)
        self.assertTrue(_revalidate(linker, clock))
        linker.storage.store.batch_get_session_refresh.assert_called_once()

    def test_oldest_key_decides(self):
        clock = _Clock()
        linker = _linker()
        _prefetched(linker, clock)
        linker.session_lease_base["k1"] = clock.now - 10.0
        self.assertTrue(_revalidate(linker, clock))
        linker.storage.store.batch_get_session_refresh.assert_called_once()

    def test_key_without_base_counts_as_old(self):
        clock = _Clock()
        linker = _linker()
        _prefetched(linker, clock)
        del linker.session_lease_base["k0"]
        self.assertTrue(_revalidate(linker, clock))
        linker.storage.store.batch_get_session_refresh.assert_called_once()

    def test_shared_session_always_refreshes(self):
        clock = _Clock()
        linker = _linker()
        _prefetched(linker, clock)
        # A second session reuses k1: its Mooncake session is no longer private.
        _prefetched(linker, clock, rid="other", keys=("k1",), session="s2")
        self.assertEqual(linker.session_refcounts["k1"], 2)
        self.assertTrue(_revalidate(linker, clock))
        linker.storage.store.batch_get_session_refresh.assert_called_once()

    def test_reused_key_loses_its_base(self):
        clock = _Clock()
        linker = _linker()
        _prefetched(linker, clock)
        clock.now += 3.0
        _prefetched(linker, clock, rid="other", keys=("k1", "k2"), session="s2")
        self.assertEqual(linker.session_lease_base, {"k0": 100.0, "k2": 103.0})

    def test_formerly_shared_key_stays_refresh_only(self):
        """Review P1-1: A shares k with B; A's refresh retires the native
        session and A is released, leaving B the sole holder of a young base."""
        clock = _Clock()
        linker = _linker()
        _prefetched(linker, clock, rid="b", keys=("k",), session="sb")
        _prefetched(linker, clock, rid="a", keys=("k",), session="sa")
        clock.now += 1.0
        linker.storage.store.batch_get_session_refresh.side_effect = lambda keys: [
            7
        ] * len(keys)
        self.assertFalse(_revalidate(linker, clock, rid="a"))
        linker._abort_prepared_load_now("sa")
        self.assertEqual(linker.session_refcounts, {"k": 1})
        # B is private again and young, but must still ask Mooncake.
        self.assertFalse(_revalidate(linker, clock, rid="b"))
        self.assertEqual(linker.storage.store.batch_get_session_refresh.call_count, 2)
        # Even a successful refresh does not restore the skip for this generation.
        linker.storage.store.batch_get_session_refresh.side_effect = lambda keys: [
            0
        ] * len(keys)
        self.assertTrue(_revalidate(linker, clock, rid="b"))
        self.assertTrue(_revalidate(linker, clock, rid="b"))
        self.assertEqual(linker.storage.store.batch_get_session_refresh.call_count, 4)
        self.assertEqual(linker.session_lease_base, {})

    def test_released_key_restarts_with_a_fresh_base(self):
        clock = _Clock()
        linker = _linker()
        _prefetched(linker, clock, rid="b", keys=("k",), session="sb")
        _prefetched(linker, clock, rid="a", keys=("k",), session="sa")
        linker._abort_prepared_load_now("sa")
        linker._abort_prepared_load_now("sb")
        clock.now += 2.0
        _prefetched(linker, clock, rid="c", keys=("k",), session="sc")
        self.assertEqual(linker.session_lease_base, {"k": 102.0})
        self.assertTrue(_revalidate(linker, clock, rid="c"))
        linker.storage.store.batch_get_session_refresh.assert_not_called()

    def test_failed_refresh_is_invalid_and_drops_bases(self):
        clock = _Clock()
        linker = _linker(refresh_results=[0, 7])
        _prefetched(linker, clock)
        clock.now += 6.0
        self.assertFalse(_revalidate(linker, clock))
        self.assertEqual(linker.session_lease_base, {})

    def test_refresh_exception_is_invalid(self):
        clock = _Clock()
        linker = _linker(refresh_error=RuntimeError("master unreachable"))
        _prefetched(linker, clock)
        clock.now += 6.0
        with self.assertLogs(level="WARNING"):
            self.assertFalse(_revalidate(linker, clock))
        self.assertEqual(linker.session_lease_base, {})

    def test_not_prefetched_is_invalid_without_age(self):
        clock = _Clock()
        linker = _linker()
        _prefetched(linker, clock)
        linker.host_prefetch_entries["rid"]["cancelled"] = True
        self.assertFalse(_revalidate(linker, clock))
        linker.storage.store.batch_get_session_refresh.assert_not_called()

    def test_release_drops_base(self):
        clock = _Clock()
        linker = _linker()
        _prefetched(linker, clock)
        _prefetched(linker, clock, rid="other", keys=("k1",), session="s2")
        linker._abort_prepared_load_now("s")
        # k1 was shared, so it lost its base already.
        self.assertEqual(linker.session_lease_base, {})
        self.assertEqual(linker.session_refcounts, {"k1": 1})
        linker._abort_prepared_load_now("s2")
        self.assertEqual(linker.session_lease_base, {})

    def test_age_includes_session_lock_wait(self):
        """Review P1-R1: another request's session start holds session_lock
        across its RPC; the admission check must age past that wait."""
        clock = _Clock()
        linker = _linker()
        linker.session_lock = _ContendedLock()
        _prefetched(linker, clock)
        clock.now += 1.0
        in_rpc = threading.Event()
        release = threading.Event()

        def slow_start(keys):
            in_rpc.set()
            if not release.wait(5):
                raise TimeoutError("the test never released the start RPC")
            return [0] * len(keys), ["dfs"] * len(keys)

        linker.storage.store.batch_get_session_start_with_sources.side_effect = (
            slow_start
        )
        transfer = types.SimpleNamespace(keys=["unrelated"])
        with mock.patch.object(mooncake_direct_linker.time, "monotonic", clock):
            worker = threading.Thread(
                target=linker._prepare_expanded_load, args=("other", [transfer])
            )
            worker.start()
            self.assertTrue(in_rpc.wait(5))
            result = {}
            admission = threading.Thread(
                target=lambda: result.update(ok=linker.revalidate_host_prefetch("rid"))
            )
            admission.start()
            # Admission blocks on the lock while the unrelated start RPC runs
            # past the skip threshold.
            self.assertTrue(linker.session_lock.contended.wait(5))
            clock.now += 10.0
            release.set()
            worker.join(5)
            admission.join(5)
        self.assertFalse(worker.is_alive())
        self.assertFalse(admission.is_alive())
        self.assertTrue(result["ok"])
        # Aged 11 s once it got the lock, so it refreshed instead of skipping.
        linker.storage.store.batch_get_session_refresh.assert_called_once()
        self.assertEqual(
            linker.session_lease_base, {"k0": 111.0, "k1": 111.0, "unrelated": 101.0}
        )

    def test_refresh_skips_base_of_key_ended_meanwhile(self):
        clock = _Clock()
        linker = _linker()
        _prefetched(linker, clock)
        clock.now += 6.0

        def refresh(keys):
            # The session is released while the refresh RPC is in flight.
            linker._abort_prepared_load_now("s")
            return [0] * len(keys)

        linker.storage.store.batch_get_session_refresh.side_effect = refresh
        self.assertTrue(_revalidate(linker, clock))
        self.assertEqual(linker.session_lease_base, {})


class TestSessionRefreshAgeConfig(CustomTestCase):
    def _config(self, **env):
        with mock.patch.dict(os.environ, env, clear=False):
            for name in (
                "SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S",
                "SGLANG_MOONCAKE_LEASE_TTL_S",
            ):
                if name not in env:
                    os.environ.pop(name, None)
            return mooncake_direct_linker._session_refresh_age_config()

    def test_default_is_disabled(self):
        self.assertEqual(self._config(), 0.0)

    def test_age_below_ttl_is_kept(self):
        self.assertEqual(self._config(SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S="5"), 5.0)

    def test_age_not_below_ttl_disables(self):
        with self.assertLogs(level="WARNING"):
            self.assertEqual(
                self._config(
                    SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S="8",
                    SGLANG_MOONCAKE_LEASE_TTL_S="8",
                ),
                0.0,
            )

    def test_negative_age_is_rejected(self):
        with self.assertRaises(ValueError):
            self._config(SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S="-1")

    def test_non_finite_values_are_rejected(self):
        for env in (
            {"SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S": "nan"},
            {"SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S": "inf"},
            {
                "SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S": "20",
                "SGLANG_MOONCAKE_LEASE_TTL_S": "nan",
            },
            {
                "SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S": "20",
                "SGLANG_MOONCAKE_LEASE_TTL_S": "inf",
            },
            {"SGLANG_MOONCAKE_LEASE_TTL_S": "0"},
            {"SGLANG_MOONCAKE_LEASE_TTL_S": "-3"},
        ):
            with self.subTest(env=env), self.assertRaises(ValueError):
                self._config(**env)

    def test_group_semantics_disable_skipping(self):
        for grouped, expected in ((True, 0.0), (False, 5.0)):
            linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
            linker.session_refresh_age_s = 5.0
            linker.storage = types.SimpleNamespace(
                _can_use_group_semantics=lambda grouped=grouped: grouped
            )
            with self.subTest(grouped=grouped):
                if grouped:
                    with self.assertLogs(level="WARNING"):
                        linker._disable_refresh_skip_for_groups()
                else:
                    linker._disable_refresh_skip_for_groups()
                self.assertEqual(linker.session_refresh_age_s, expected)

    def test_class_defaults_disable_skipping(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        self.assertEqual(linker.session_refresh_age_s, 0.0)


if __name__ == "__main__":
    unittest.main()
