"""A waiting-queue prefetch that needed no DFS read must still be re-checked.

The worker releases the session of a ``no_prefetch_needed`` request right
away, so its objects may be evicted while it waits for admission. Admission
re-checks them and, if any is gone, sends every rank back to the normal path.
"""

import threading
import types
import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.unified_cache.components import (
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    ExternalCacheHitMarker,
    UnifiedCacheLinkerWrapper,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _linker(exist=None, error=None):
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.host_prefetch_enabled = True
    linker.host_prefetch_lock = threading.Lock()
    linker.host_prefetch_entries = {}
    linker._abort_prepared_load_now = MagicMock()

    def batch_exist(keys):
        if error is not None:
            raise error
        return [exist.get(key, 0) for key in keys]

    linker.storage = types.SimpleNamespace(
        _batch_exist=MagicMock(side_effect=batch_exist)
    )
    return linker


def _finish_no_prefetch(linker, rid="rid", keys=("k0", "k1")):
    entry = {"state": "preparing", "session_rid": "session"}
    linker.host_prefetch_entries[rid] = entry
    linker._finish_host_prefetch(
        rid, entry, "session", "no_prefetch_needed", keys=list(keys)
    )
    return entry


class TestNoPrefetchNeededRevalidation(CustomTestCase):
    def test_finish_keeps_keys_and_releases_session(self):
        linker = _linker(exist={})
        entry = _finish_no_prefetch(linker)
        self.assertEqual(entry["state"], "no_prefetch_needed")
        self.assertEqual(entry["no_prefetch_keys"], ["k0", "k1"])
        linker._abort_prepared_load_now.assert_called_once_with("session")

    def test_all_keys_present_is_valid(self):
        linker = _linker(exist={"k0": 1, "k1": 1})
        _finish_no_prefetch(linker)
        self.assertTrue(linker.revalidate_no_prefetch_needed("rid"))
        linker.storage._batch_exist.assert_called_once_with(["k0", "k1"])

    def test_evicted_key_is_invalid(self):
        linker = _linker(exist={"k0": 1})
        _finish_no_prefetch(linker)
        with self.assertLogs(level="WARNING"):
            self.assertFalse(linker.revalidate_no_prefetch_needed("rid"))

    def test_existence_query_error_is_invalid(self):
        linker = _linker(error=RuntimeError("master unreachable"))
        _finish_no_prefetch(linker)
        with self.assertLogs(level="WARNING"):
            self.assertFalse(linker.revalidate_no_prefetch_needed("rid"))

    def test_other_states_are_invalid(self):
        linker = _linker(exist={"k0": 1, "k1": 1})
        self.assertFalse(linker.revalidate_no_prefetch_needed("missing"))
        entry = _finish_no_prefetch(linker)
        entry["cancelled"] = True
        self.assertFalse(linker.revalidate_no_prefetch_needed("rid"))
        linker.storage._batch_exist.assert_not_called()


class _Cache:
    """Records the admission reductions.

    ``peer_mins[i]`` is the other ranks' MIN for the i-th reduction (ready,
    valid, claimed); reductions beyond the list see no failing peer.
    """

    def __init__(self, peer_mins=()):
        self.peer_mins = list(peer_mins)
        self.reductions = []
        self.page_size = 1
        self.tree_core = types.SimpleNamespace(
            empty_match_result=types.SimpleNamespace(device_indices="empty")
        )
        # One component that cannot build a transfer stops load_back right
        # after the prefetch decision, before any device-side work. On the
        # prefetched path construction joins the claim reduction as
        # [claimed, constructed]; on the enabled normal fallback it is its own
        # reduction. Disabled prefetch keeps rank-local construction.
        component = MagicMock()
        component.build_external_linker_transfer.return_value = None
        self._components_tuple = (component,)

    def _all_reduce_attn_groups(self, tensor, op):
        assert op == torch.distributed.ReduceOp.MIN
        # A scalar or a vector verdict; peers are injected element-wise.
        local = tensor.tolist() if tensor.numel() > 1 else int(tensor.item())
        stage = len(self.reductions)
        peer = self.peer_mins[stage] if stage < len(self.peer_mins) else None
        if peer is not None:
            tensor.copy_(torch.minimum(tensor, torch.tensor(peer, dtype=tensor.dtype)))
        self.reductions.append(local)


def _wrapper(cache, cache_linker):
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = cache
    wrapper.cache_linker = cache_linker
    wrapper._waiting_queue_prefetch_enabled = True
    wrapper.hit_markers = {}
    wrapper.host_prefetch_hits = {}
    wrapper._update_load = MagicMock()
    hit = ExternalCacheHitMarker(prefix_key=None, tail_hashes=["h"], device_hit_len=0)
    wrapper.hit_markers["rid"] = hit
    wrapper.host_prefetch_hits["rid"] = hit
    return wrapper


def _no_prefetch_linker(valid):
    cache_linker = MagicMock()
    cache_linker.get_host_prefetch_status.return_value = "no_prefetch_needed"
    if isinstance(valid, BaseException):
        cache_linker.revalidate_no_prefetch_needed.side_effect = valid
    else:
        cache_linker.revalidate_no_prefetch_needed.return_value = valid
    return cache_linker


class TestCancelWaitingQueuePrefetch(CustomTestCase):
    def _wrapper(self):
        wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
        wrapper.cache_linker = MagicMock()
        wrapper._waiting_queue_prefetch_enabled = True
        wrapper.hit_markers = {"rid": object()}
        wrapper.host_prefetch_hits = {}
        wrapper.pending_host_prefetch_submissions = {}
        return wrapper

    def test_untracked_request_does_not_cancel_backend_prefetch(self):
        wrapper = self._wrapper()

        wrapper.cancel_waiting_queue_prefetch("rid")

        self.assertNotIn("rid", wrapper.hit_markers)
        wrapper.cache_linker.cancel_host_prefetch.assert_not_called()

    def test_tracked_host_prefetch_is_cancelled(self):
        wrapper = self._wrapper()
        wrapper.host_prefetch_hits["rid"] = object()

        wrapper.cancel_waiting_queue_prefetch("rid")

        self.assertNotIn("rid", wrapper.host_prefetch_hits)
        wrapper.cache_linker.cancel_host_prefetch.assert_called_once_with("rid")

    def test_pending_submission_is_cancelled(self):
        wrapper = self._wrapper()
        job = types.SimpleNamespace(cancelled=threading.Event())
        wrapper.pending_host_prefetch_submissions["rid"] = (job, object())

        wrapper.cancel_waiting_queue_prefetch("rid")

        self.assertTrue(job.cancelled.is_set())
        self.assertNotIn("rid", wrapper.pending_host_prefetch_submissions)
        wrapper.cache_linker.cancel_host_prefetch.assert_called_once_with("rid")


class _NoPrefetchMapAccess(dict):
    def get(self, key, default=None):
        raise AssertionError("disabled prefetch map must not be queried")

    def pop(self, key, default=None):
        raise AssertionError("disabled prefetch map must not be popped")


class TestDisabledPrefetchMapFastPath(CustomTestCase):
    def _wrapper(self, cache):
        wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
        wrapper.cache = cache
        wrapper.cache_linker = MagicMock()
        wrapper._waiting_queue_prefetch_enabled = False
        wrapper.hit_markers = {}
        wrapper.host_prefetch_hits = _NoPrefetchMapAccess()
        wrapper.pending_host_prefetch_submissions = _NoPrefetchMapAccess()
        return wrapper

    def test_match_skips_host_prefetch_map_query(self):
        cache = types.SimpleNamespace(page_size=1, pp_size=1)
        wrapper = self._wrapper(cache)
        req = types.SimpleNamespace(rid="rid")
        result = types.SimpleNamespace(
            device_indices=types.SimpleNamespace(numel=lambda: 0)
        )

        self.assertIs(wrapper.match([], req, result), result)

    def test_normal_load_skips_host_prefetch_map_pop(self):
        cache = _Cache()
        first_component = MagicMock()
        transfer = object()
        first_component.build_external_linker_transfer.return_value = transfer
        cache._components_tuple = (first_component, *cache._components_tuple)
        wrapper = self._wrapper(cache)
        wrapper.hit_markers["rid"] = ExternalCacheHitMarker(
            prefix_key=None, tail_hashes=["h0"], device_hit_len=0
        )
        wrapper._build_load_transfers = MagicMock(return_value=([], False, None))
        wrapper._abort_disagreed_load = MagicMock()
        wrapper._update_load = MagicMock()
        req = types.SimpleNamespace(rid="rid", last_node="node")

        indices, node = wrapper.load_back(req)

        self.assertEqual(indices, "empty")
        self.assertEqual(node, "node")
        first_component.build_external_linker_transfer.assert_called_once_with(
            LinkerTransferPhase.LOAD, None, ["h0"]
        )
        wrapper._build_load_transfers.assert_not_called()
        wrapper._abort_disagreed_load.assert_not_called()
        wrapper._update_load.assert_called_once_with(
            ExternalLinkerLoadPhase.ABORT, req, [(first_component, transfer)], 1
        )
        self.assertEqual(cache.reductions, [])

    def test_normal_load_success_has_no_construction_reduction(self):
        cache = _Cache()
        cache.pp_size = 2
        transfer = types.SimpleNamespace(name=PoolName.KV, device_indices="loaded")
        component = cache._components_tuple[0]
        component.build_external_linker_transfer.return_value = transfer
        wrapper = self._wrapper(cache)
        wrapper.hit_markers["rid"] = ExternalCacheHitMarker(
            prefix_key=None, tail_hashes=["h0"], device_hit_len=0
        )
        wrapper.cache_linker.revalidate_load.return_value = True
        wrapper._build_load_transfers = MagicMock()
        wrapper._update_load = MagicMock(return_value=[transfer])
        wrapper._queue_load = MagicMock()
        req = types.SimpleNamespace(rid="rid", last_node="node")

        self.assertEqual(wrapper.load_back(req), ("loaded", "node"))

        wrapper._build_load_transfers.assert_not_called()
        self.assertEqual(cache.reductions, [])
        wrapper.cache_linker.revalidate_load.assert_called_once_with([transfer])
        wrapper._update_load.assert_called_once_with(
            ExternalLinkerLoadPhase.PREPARE, req, [(component, transfer)], 1
        )
        wrapper._queue_load.assert_called_once_with(
            "rid", "node", [transfer], anchor="node"
        )

    def test_normal_load_build_error_propagates_without_consensus_or_abort(self):
        cache = _Cache()
        first_component = MagicMock()
        first_component.build_external_linker_transfer.return_value = object()
        error = RuntimeError("construction failed")
        cache._components_tuple[0].build_external_linker_transfer.side_effect = error
        cache._components_tuple = (first_component, *cache._components_tuple)
        wrapper = self._wrapper(cache)
        wrapper.hit_markers["rid"] = ExternalCacheHitMarker(
            prefix_key=None, tail_hashes=["h0"], device_hit_len=0
        )
        wrapper._update_load = MagicMock()
        req = types.SimpleNamespace(rid="rid", last_node="node")

        with self.assertRaises(RuntimeError) as raised:
            wrapper.load_back(req)

        self.assertIs(raised.exception, error)
        self.assertEqual(cache.reductions, [])
        wrapper._update_load.assert_not_called()
        wrapper.cache_linker.revalidate_load.assert_not_called()

    def test_release_skips_prefetch_cancel_helper_but_keeps_normal_cleanup(self):
        wrapper = self._wrapper(None)
        wrapper.hit_markers["rid"] = object()
        wrapper.pending_loads = {}
        wrapper.cache_linker.cancel_queued_load.return_value = False
        wrapper.cancel_waiting_queue_prefetch = MagicMock(
            side_effect=AssertionError("disabled prefetch helper called")
        )

        wrapper.release_request("rid")

        self.assertNotIn("rid", wrapper.hit_markers)
        wrapper.cancel_waiting_queue_prefetch.assert_not_called()
        wrapper.cache_linker.cancel_queued_load.assert_called_once_with("rid")

    def test_disabled_wrapper_allocates_no_prefetch_maps_and_resets(self):
        cache = types.SimpleNamespace(
            tree_core=types.SimpleNamespace(),
            write_through_threshold=0,
            _all_reduce_attn_groups=MagicMock(),
        )
        cache_linker = MagicMock()
        cache_linker.waiting_queue_prefetch_enabled.return_value = False
        wrapper = UnifiedCacheLinkerWrapper(cache, cache_linker)

        self.assertIsNone(wrapper.host_prefetch_hits)
        self.assertIsNone(wrapper.pending_host_prefetch_submissions)
        self.assertEqual(
            wrapper.get_host_prefetch_admission_state("rid"), "not_tracked"
        )
        self.assertFalse(wrapper.prefetch_to_host(object()))
        cache_linker.get_host_prefetch_status.assert_not_called()
        cache._all_reduce_attn_groups.assert_not_called()

        wrapper.reset()

        cache_linker.reset.assert_called_once()


class TestLoadBackNoPrefetchNeeded(CustomTestCase):
    def test_enabled_normal_fallback_keeps_construction_consensus(self):
        cache = _Cache(peer_mins=[0])
        component = cache._components_tuple[0]
        transfer = object()
        component.build_external_linker_transfer.return_value = transfer
        wrapper = _wrapper(cache, MagicMock())
        wrapper.host_prefetch_hits.clear()
        req = types.SimpleNamespace(rid="rid", last_node="node")

        self.assertEqual(wrapper.load_back(req), ("empty", "node"))

        self.assertEqual(cache.reductions, [1])
        wrapper._update_load.assert_called_once_with(
            ExternalLinkerLoadPhase.ABORT, req, [(component, transfer)], 1
        )
        wrapper.cache_linker.revalidate_load.assert_not_called()

    def _load_back(self, cache_linker, peer_mins=()):
        cache = _Cache(peer_mins=peer_mins)
        wrapper = _wrapper(cache, cache_linker)
        req = types.SimpleNamespace(rid="rid", last_node="node")
        wrapper.load_back(req)
        return cache

    def test_surviving_keys_keep_the_prepared_path(self):
        cache_linker = _no_prefetch_linker(True)
        cache = self._load_back(cache_linker)
        cache_linker.revalidate_no_prefetch_needed.assert_called_once_with("rid")
        # ready, valid, then [claimed, constructed] in one reduction; the
        # failed construction ends in a miss and retires the prefetch.
        self.assertEqual(cache.reductions, [1, 1, [1, 0]])
        cache_linker.cancel_host_prefetch.assert_called_with("rid")
        cache_linker.abort_prepared_load.assert_not_called()

    def test_evicted_keys_fall_back_to_the_normal_path(self):
        cache_linker = _no_prefetch_linker(False)
        cache = self._load_back(cache_linker)
        # ready, valid, then the normal path's construction.
        self.assertEqual(cache.reductions, [1, 0, 0])
        cache_linker.cancel_host_prefetch.assert_called_once_with("rid")
        cache_linker.abort_prepared_load.assert_not_called()

    def test_check_error_still_joins_the_reduction(self):
        cache_linker = _no_prefetch_linker(RuntimeError("boom"))
        cache = self._load_back(cache_linker)
        self.assertEqual(cache.reductions, [1, 0, 0])
        cache_linker.abort_prepared_load.assert_not_called()

    def test_peer_failure_overrides_local_success(self):
        cache_linker = _no_prefetch_linker(True)
        cache = self._load_back(cache_linker, peer_mins=[0])
        # ready is already 0 from the peer, so no rank reaches validation;
        # the normal path then agrees on construction.
        self.assertEqual(cache.reductions, [1, 0])
        cache_linker.revalidate_no_prefetch_needed.assert_not_called()
        cache_linker.abort_prepared_load.assert_not_called()

    def test_peer_validation_failure_sends_every_status_to_the_normal_path(self):
        # Ready passes on every rank, then a peer fails validation: both a
        # local no_prefetch_needed and a local dfs_prefetched rank must drop
        # their prefetch and take the normal path, claiming nothing.
        dfs_linker = MagicMock()
        dfs_linker.get_host_prefetch_status.return_value = "dfs_prefetched"
        dfs_linker.revalidate_host_prefetch.return_value = True
        for cache_linker in (_no_prefetch_linker(True), dfs_linker):
            cache = self._load_back(cache_linker, peer_mins=[1, 0])
            self.assertEqual(cache.reductions, [1, 1, 0])
            cache_linker.cancel_host_prefetch.assert_called_once_with("rid")
            cache_linker.claim_ready_host_prefetch.assert_not_called()
            cache_linker.abort_prepared_load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
