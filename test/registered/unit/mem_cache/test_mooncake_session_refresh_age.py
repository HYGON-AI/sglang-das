"""SGLang owns shared references; Mooncake prepares leases on actual reads."""

import threading
import types
import unittest
from unittest.mock import MagicMock

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _linker():
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.host_prefetch_enabled = True
    linker.host_prefetch_lock = threading.Lock()
    linker.host_prefetch_entries = {}
    linker.session_lock = threading.Lock()
    linker.prepared_load_sessions = {}
    linker.session_refcounts = {}
    linker.session_sources = {}
    # Deliberately no refresh API: neither preparation nor admission needs it.
    store = types.SimpleNamespace(
        batch_get_session_start_with_sources=MagicMock(
            side_effect=lambda keys: ([0] * len(keys), ["dfs"] * len(keys))
        ),
        batch_get_session_end=MagicMock(return_value=0),
    )
    linker.storage = types.SimpleNamespace(
        store=store,
        _get_hybrid_page_component_keys=lambda keys, transfer: (keys, None),
        _tag_keys=lambda keys: list(keys),
    )
    return linker


def _prefetched(linker, rid, keys=("k0", "k1")):
    session = "prefetch-" + rid
    transfer = types.SimpleNamespace(keys=list(keys))
    assert linker._prepare_expanded_load(session, [transfer])
    linker.host_prefetch_entries[rid] = {
        "state": "dfs_prefetched",
        "session_rid": session,
    }


class TestSharedSessionOwnership(CustomTestCase):
    def test_admission_only_checks_ownership(self):
        linker = _linker()
        _prefetched(linker, "a")
        linker.storage.store.batch_get_session_start_with_sources.reset_mock()
        self.assertTrue(linker.revalidate_host_prefetch("a"))
        linker.storage.store.batch_get_session_start_with_sources.assert_not_called()
        linker.storage.store.batch_get_session_end.assert_not_called()

    def test_shared_claim_release_keeps_other_waiter(self):
        linker = _linker()
        _prefetched(linker, "a")
        _prefetched(linker, "b")
        self.assertEqual(linker.session_refcounts, {"k0": 2, "k1": 2})
        self.assertTrue(linker.claim_ready_host_prefetch("b"))
        linker.abort_prepared_load("b")
        linker.storage.store.batch_get_session_end.assert_not_called()
        self.assertTrue(linker.revalidate_host_prefetch("a"))
        self.assertEqual(linker.session_refcounts, {"k0": 1, "k1": 1})
        self.assertTrue(linker.claim_ready_host_prefetch("a"))
        linker.abort_prepared_load("a")
        linker.storage.store.batch_get_session_end.assert_called_once_with(["k0", "k1"])
        self.assertEqual(linker.session_refcounts, {})
        self.assertEqual(linker.prepared_load_sessions, {})
        self.assertEqual(linker.session_sources, {})

    def test_cancelled_or_released_prefetch_is_not_ready(self):
        linker = _linker()
        _prefetched(linker, "a")
        linker.host_prefetch_entries["a"]["cancelled"] = True
        self.assertFalse(linker.revalidate_host_prefetch("a"))
        linker.host_prefetch_entries["a"]["cancelled"] = False
        linker._abort_prepared_load_now("prefetch-a")
        self.assertFalse(linker.revalidate_host_prefetch("a"))
        self.assertFalse(linker.revalidate_host_prefetch("missing"))


if __name__ == "__main__":
    unittest.main()
