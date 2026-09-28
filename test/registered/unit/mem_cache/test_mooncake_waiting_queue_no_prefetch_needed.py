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

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
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
    entry = {"state": "preparing", "session_rid": "session", "reserved_bytes": 7}
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
        self.assertEqual(entry["reserved_bytes"], 0)
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
        # [claimed, constructed]; on the normal path it is its own reduction.
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


class TestLoadBackNoPrefetchNeeded(CustomTestCase):
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
