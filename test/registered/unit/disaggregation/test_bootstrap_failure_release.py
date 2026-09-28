"""A PD prefill request that fails bootstrap releases its cache-side state.

Before this fix only HiCache storage released it; with the unified-cache
external linker a waiting-queue prefetch taken at arrival kept its budget
slot, pinned session and hit markers after the request failed.
"""

import threading
import time
import types
import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.prefill import (  # noqa: E402
    SchedulerDisaggregationPrefillMixin,
)
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (  # noqa: E402
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (  # noqa: E402
    ExternalCacheHitMarker,
    UnifiedCacheLinkerWrapper,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _req(rid="r"):
    req = MagicMock()
    req.rid = rid
    req.req_pool_idx = None
    req.kv = None
    req.mamba_pool_idx = None
    req.disagg_kv_sender.failure_exception.side_effect = RuntimeError("boom")
    return req


def _scheduler(*, hicache, external_linker, tree_cache):
    s = types.SimpleNamespace(
        enable_hicache_storage=hicache,
        enable_unified_cache_external_linker=external_linker,
        tree_cache=tree_cache,
        ps=types.SimpleNamespace(tp_rank=0),
        clear_pending_chunk_send=MagicMock(),
        req_to_metadata_buffer_idx_allocator=MagicMock(),
        disagg_metadata_buffers=None,
        output_streamer=MagicMock(),
        metrics_reporter=types.SimpleNamespace(enable_metrics=False),
    )
    s._release_aborted_request = lambda rid: Scheduler._release_aborted_request(s, rid)
    return s


def _fail(s, req):
    with patch(
        "sglang.srt.disaggregation.prefill.maybe_release_metadata_buffer"
    ), patch("sglang.srt.disaggregation.prefill.prepare_abort"):
        SchedulerDisaggregationPrefillMixin.handle_bootstrap_failure(s, req)


class TestBootstrapFailureRelease(CustomTestCase):
    def test_external_linker_releases_the_request(self):
        tree_cache = MagicMock()
        s = _scheduler(hicache=False, external_linker=True, tree_cache=tree_cache)
        _fail(s, _req("r"))
        tree_cache.release_aborted_request.assert_called_once_with("r")

    def test_hicache_still_releases_once(self):
        tree_cache = MagicMock()
        s = _scheduler(hicache=True, external_linker=False, tree_cache=tree_cache)
        _fail(s, _req("r"))
        tree_cache.release_aborted_request.assert_called_once_with("r")

    def test_plain_cache_is_untouched(self):
        tree_cache = MagicMock()
        s = _scheduler(hicache=False, external_linker=False, tree_cache=tree_cache)
        _fail(s, _req("r"))
        tree_cache.release_aborted_request.assert_not_called()


def _mooncake_linker(state):
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.host_prefetch_lock = threading.Lock()
    linker.session_lock = threading.Lock()
    linker.prepared_load_sessions = {}
    linker.pending_loads = {}
    linker.host_prefetch_entries = {
        "r": {
            "state": state,
            "cancelled": False,
            "session_rid": "s-r",
            "reserved_bytes": 1 << 20,
        }
    }
    linker._abort_prepared_load_now = MagicMock()
    return linker


class TestReleaseFreesThePrefetch(CustomTestCase):
    """What release_aborted_request does to a queued prefetch."""

    def _wrapper(self, cache_linker):
        wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
        wrapper.cache = MagicMock()
        wrapper.cache_linker = cache_linker
        hit = ExternalCacheHitMarker(
            prefix_key=None, tail_hashes=["h"], device_hit_len=0
        )
        wrapper.hit_markers = {"r": hit}
        wrapper.host_prefetch_hits = {"r": hit}
        wrapper.pending_host_prefetch_submissions = {}
        wrapper.pending_loads = {}
        return wrapper

    def test_completed_prefetch_frees_its_slot_and_session(self):
        linker = _mooncake_linker("dfs_prefetched")
        wrapper = self._wrapper(linker)
        wrapper.release_request("r")
        self.assertEqual(linker.host_prefetch_entries, {})
        linker._abort_prepared_load_now.assert_called_once_with("s-r")
        self.assertEqual(wrapper.hit_markers, {})
        self.assertEqual(wrapper.host_prefetch_hits, {})

    def test_in_flight_read_is_only_marked_cancelled(self):
        linker = _mooncake_linker("reading")
        wrapper = self._wrapper(linker)
        wrapper.release_request("r")
        # The worker owns the session until its read ends, then drops it.
        self.assertTrue(linker.host_prefetch_entries["r"]["cancelled"])
        linker._abort_prepared_load_now.assert_not_called()


class _NativeStore:
    """Stubs only the Mooncake native calls; the DFS read pauses until released."""

    def __init__(self):
        self.reading = threading.Event()
        self.release_read = threading.Event()
        self.ended = []

    def batch_get_session_start_with_sources(self, keys):
        return [0] * len(keys), ["dfs"] * len(keys)

    def batch_get_session_start(self, keys):
        return [0] * len(keys)

    def batch_get_session_prefetch(self, keys):
        self.reading.set()
        assert self.release_read.wait(timeout=30)
        return [0] * len(keys)

    def batch_get_session_end(self, keys):
        self.ended.append(list(keys))


def _live_linker():
    """A real Mooncake prefetch front end: real prepare, worker, finish and
    session reference rollback over a stubbed native store."""
    from queue import Queue

    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.host_prefetch_enabled = True
    linker.host_prefetch_limit = 8
    linker.host_prefetch_max_bytes = 1 << 30
    linker.host_prefetch_lock = threading.Lock()
    linker.host_prefetch_entries = {}
    linker.host_prefetch_queue = Queue()
    linker.host_prefetch_generation = 0
    linker.session_lock = threading.Lock()
    linker.prepared_load_sessions = {}
    linker.session_refcounts = {}
    linker.session_sources = {}
    linker.pool_group = types.SimpleNamespace(
        resolve_transfers=lambda transfers, **_: list(transfers)
    )
    native = _NativeStore()
    linker.storage = types.SimpleNamespace(
        store=native,
        _get_hybrid_page_component_keys=lambda keys, transfer: (keys, 1),
        _tag_keys=lambda keys: list(keys),
    )
    linker._get_host_prefetch_object_sizes = lambda transfers: {
        key: 4 << 10 for transfer in transfers for key in transfer.keys
    }
    linker.cancel_queued_load = MagicMock(return_value=False)
    return linker, native


class TestReleaseAcrossBackgroundSubmitStages(CustomTestCase):
    """Bootstrap failure at every stage of the background prefetch submission.

    The worker really prepares a session and pauses inside the DFS read. Each
    case ends by letting the read, the worker and the ack finish, then checks
    that the session references are gone and the native session ended once.
    """

    KEY = "h-r"

    def setUp(self):
        from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
        from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
            PreparedHostPrefetch,
        )

        self.linker, self.native = _live_linker()
        self.worker = threading.Thread(
            target=self.linker.host_prefetch_thread_func, daemon=True
        )
        self.worker.start()
        wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
        wrapper.cache = MagicMock()
        wrapper.cache_linker = self.linker
        hit = ExternalCacheHitMarker(
            prefix_key=None, tail_hashes=[self.KEY], device_hit_len=0
        )
        wrapper.hit_markers = {"r": hit}
        wrapper.host_prefetch_hits = {}
        wrapper.pending_loads = {}
        self.transfers = (PoolTransfer(name=PoolName.KV, keys=[self.KEY]),)
        self.job = PreparedHostPrefetch(
            rid="r",
            locally_eligible=True,
            transfers=self.transfers,
            cache_linker=self.linker,
            cancelled=threading.Event(),
        )
        wrapper.pending_host_prefetch_submissions = {"r": (self.job, hit)}
        self.wrapper = wrapper

    def tearDown(self):
        self.native.release_read.set()
        self.linker.host_prefetch_queue.put(None)
        self.worker.join(timeout=30)
        self.assertFalse(self.worker.is_alive())

    def _background_submit(self):
        """What one background round does for this job on every rank."""
        submitted = self.linker.submit_host_prefetch("r", list(self.transfers))
        submitted = submitted and not self.job.cancelled.is_set()
        if not submitted:
            self.linker.cancel_host_prefetch("r")
        return submitted

    def _read_in_flight(self):
        self.assertTrue(self._background_submit())
        self.assertTrue(self.native.reading.wait(timeout=30))
        self.assertEqual(self.linker.host_prefetch_entries["r"]["state"], "reading")
        self.assertEqual(self.linker.session_refcounts, {self.KEY: 1})

    def _finish(self):
        self.native.release_read.set()
        deadline = time.monotonic() + 30
        while self.linker.host_prefetch_queue.unfinished_tasks:
            self.assertLess(time.monotonic(), deadline, "worker did not finish")
            time.sleep(0.01)

    def _assert_clean(self, *, other_holders=0):
        self.assertEqual(self.linker.host_prefetch_entries, {})
        self.assertEqual(self.wrapper.pending_host_prefetch_submissions, {})
        self.assertEqual(self.wrapper.host_prefetch_hits, {})
        self.assertNotIn("r", self.wrapper.hit_markers)
        self.assertFalse(
            any(
                rid.startswith("__sglang") for rid in self.linker.prepared_load_sessions
            )
        )
        if other_holders:
            self.assertEqual(self.linker.session_refcounts, {self.KEY: other_holders})
            self.assertEqual(self.native.ended, [])
        else:
            self.assertEqual(self.linker.session_refcounts, {})
            self.assertEqual(self.linker.session_sources, {})
            self.assertEqual(self.native.ended, [[self.KEY]])

    def test_job_queued_not_yet_submitted(self):
        self.wrapper.release_request("r")
        self.assertTrue(self.job.cancelled.is_set())
        self.assertFalse(self._background_submit())
        self.assertFalse(
            self.wrapper.complete_host_prefetch_submission(self.job, False)
        )
        self._finish()
        self.assertEqual(self.linker.host_prefetch_entries, {})
        self.assertEqual(self.linker.session_refcounts, {})

    def test_ack_applied_then_release_during_read(self):
        self._read_in_flight()
        self.assertTrue(self.wrapper.complete_host_prefetch_submission(self.job, True))
        self.wrapper.release_request("r")
        # Only marked while the read runs: the session is not ended early.
        self.assertTrue(self.linker.host_prefetch_entries["r"]["cancelled"])
        self.assertEqual(self.native.ended, [])
        self._finish()
        self._assert_clean()

    def test_late_successful_ack_after_release(self):
        self._read_in_flight()
        self.wrapper.release_request("r")
        self.assertFalse(self.wrapper.complete_host_prefetch_submission(self.job, True))
        self._finish()
        self._assert_clean()

    def test_repeated_release_ends_the_session_once(self):
        self._read_in_flight()
        self.assertTrue(self.wrapper.complete_host_prefetch_submission(self.job, True))
        self.wrapper.release_request("r")
        self.wrapper.release_request("r")
        self._finish()
        self.wrapper.release_request("r")
        self._assert_clean()

    def test_shared_key_holder_keeps_the_session(self):
        # Another request already holds the same key.
        with self.linker.session_lock:
            self.linker.session_refcounts[self.KEY] = 1
            self.linker.session_sources[self.KEY] = "dfs"
            self.linker.prepared_load_sessions["other"] = [self.KEY]
        self.assertTrue(self._background_submit())
        self.assertTrue(self.wrapper.complete_host_prefetch_submission(self.job, True))
        deadline = time.monotonic() + 30
        while self.linker.host_prefetch_queue.unfinished_tasks:
            self.assertLess(time.monotonic(), deadline)
            self.native.release_read.set()
            time.sleep(0.01)
        self.wrapper.release_request("r")
        self._assert_clean(other_holders=1)
        self.linker._abort_prepared_load_now("other")
        self.assertEqual(self.native.ended, [[self.KEY]])

    def test_bootstrap_failure_reaches_the_cleanup(self):
        self._read_in_flight()
        self.assertTrue(self.wrapper.complete_host_prefetch_submission(self.job, True))
        tree_cache = types.SimpleNamespace(
            release_aborted_request=lambda rid: self.wrapper.release_request(rid)
        )
        s = _scheduler(hicache=False, external_linker=True, tree_cache=tree_cache)
        _fail(s, _req("r"))
        self.assertTrue(self.linker.host_prefetch_entries["r"]["cancelled"])
        self._finish()
        self._assert_clean()


if __name__ == "__main__":
    unittest.main()
