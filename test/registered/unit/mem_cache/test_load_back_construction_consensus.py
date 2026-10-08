"""load_back agrees on LOAD transfer construction before any later collective.

Building a LOAD transfer allocates device slots and can fail on one rank
alone. Every rank must then abort together, each freeing what it built and
any claimed prefetch session, instead of one rank returning while the others
enter revalidate_load or the load itself. A component returning None is an
expected miss; an unexpected build error still joins the same agreement and
cleanup, then only the rank that hit it re-raises.
"""

import datetime
import socket
import threading
import time
import types
import unittest
from unittest.mock import MagicMock

import torch
import torch.distributed as dist

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer  # noqa: E402
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (  # noqa: E402
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.unified_cache.components.full_component import (  # noqa: E402
    FullComponent,
)
from sglang.srt.mem_cache.unified_cache.components.swa_component import (  # noqa: E402
    SWAComponent,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (  # noqa: E402
    ExternalCacheHitMarker,
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

REPEATS = 20
SESSION = "__sglang_waiting_queue_prefetch__:r:1"


class BuildStop(BaseException):
    """A non-Exception interruption raised while building."""


class ReachedPrepare(Exception):
    """Raised by the fake components at PREPARE: load_back got past every
    agreement and would start loading."""


class FakeComponent:
    """A persistent allocator: one unit per built transfer, ABORT frees it."""

    def __init__(self, name):
        self.name = name
        self.failure = None
        self.allocated = 0
        self.abort_error = None

    def build_external_linker_transfer(self, phase, node, keys):
        assert phase == LinkerTransferPhase.LOAD
        if self.failure == "none":
            return None
        if self.failure == "raise":
            raise RuntimeError("build failed")
        if self.failure == "assert":
            raise AssertionError("tree invariant broken")
        if self.failure == "stop":
            raise BuildStop()
        self.allocated += 1
        return PoolTransfer(
            name=self.name, device_indices=torch.arange(4), keys=list(keys)
        )

    def update_external_linker_load(self, phase, req, full, transfer, prefix_len, **_):
        if phase == ExternalLinkerLoadPhase.PREPARE:
            raise ReachedPrepare()
        assert phase == ExternalLinkerLoadPhase.ABORT
        if self.abort_error is not None:
            raise self.abort_error
        self.allocated -= 1


def _prefetch_backend(other_holder, claim="ok"):
    """A real Mooncake linker holding one ready prefetch session for "r";
    only the native store calls are stubbed. claim="fail" makes the real
    claim return False (the RID already holds another prepared session);
    claim="raise" makes it raise."""
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.host_prefetch_lock = threading.Lock()
    linker.session_lock = threading.Lock()
    linker.host_prefetch_entries = {
        "r": {
            "state": "dfs_prefetched",
            "cancelled": False,
            "session_rid": SESSION,
            "reserved_bytes": 8 << 10,
        }
    }
    linker.prepared_load_sessions = {SESSION: ["k0"]}
    linker.session_refcounts = {"k0": 2 if other_holder else 1}
    linker.session_sources = {"k0": "dfs"}
    if other_holder:
        linker.prepared_load_sessions["other"] = ["k0"]
    if claim == "fail":
        linker.prepared_load_sessions["r"] = ["k9"]
        linker.session_refcounts["k9"] = 1
    if claim == "raise":

        def raising_claim(rid):
            raise RuntimeError("claim failed")

        linker.claim_ready_host_prefetch = raising_claim
    ended = []
    linker.storage = types.SimpleNamespace(
        store=types.SimpleNamespace(
            batch_get_session_refresh=lambda keys: [0] * len(keys),
            batch_get_session_end=lambda keys: ended.append(list(keys)),
        )
    )
    return linker, ended


def _no_prefetch_needed_backend():
    """A real Mooncake linker whose "r" entry needed no DFS read: its session
    was already released, only the existence check of its keys is stubbed."""
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.host_prefetch_lock = threading.Lock()
    linker.session_lock = threading.Lock()
    linker.host_prefetch_entries = {
        "r": {
            "state": "no_prefetch_needed",
            "cancelled": False,
            "session_rid": SESSION,
            "reserved_bytes": 0,
            "no_prefetch_keys": ["k0"],
        }
    }
    linker.prepared_load_sessions = {}
    linker.session_refcounts = {}
    linker.session_sources = {}
    ended = []
    linker.storage = types.SimpleNamespace(
        _batch_exist=lambda keys: [1] * len(keys),
        store=types.SimpleNamespace(
            batch_get_session_refresh=lambda keys: [0] * len(keys),
            batch_get_session_end=lambda keys: ended.append(list(keys)),
        ),
    )
    return linker, ended


def _no_prefetch_needed_released(linker, ended):
    return (
        not linker.host_prefetch_entries
        and not linker.prepared_load_sessions
        and linker.session_refcounts == {}
        and ended == []
    )


def _session_released(linker, ended, other_holder, claim="ok"):
    if linker.host_prefetch_entries or SESSION in linker.prepared_load_sessions:
        return False
    if claim == "fail":
        # The session that made the claim fail belongs to someone else.
        if linker.prepared_load_sessions.get("r") != ["k9"]:
            return False
        return linker.session_refcounts == {"k9": 1} and ended == [["k0"]]
    if "r" in linker.prepared_load_sessions:
        return False
    if other_holder:
        return linker.session_refcounts == {"k0": 1} and ended == []
    return linker.session_refcounts == {} and ended == [["k0"]]


# scenario -> (path, {rank: (component index, failure)}, other session holder,
#              {rank: claim behaviour})
# "mixed" is the prefetched path with ranks 2 and 3 in no_prefetch_needed.
SCENARIOS = {
    "normal_second_component_none": ("normal", {3: (1, "none")}, False, {}),
    "normal_first_component_raises": ("normal", {1: (0, "raise")}, False, {}),
    "prefetch_one_rank_fails": ("prefetch", {2: (0, "none")}, False, {}),
    "prefetch_second_component_raises": ("prefetch", {0: (1, "raise")}, False, {}),
    "prefetch_fails_with_other_holder": ("prefetch", {2: (0, "none")}, True, {}),
    "prefetch_claim_fails_one_rank": ("prefetch", {}, False, {1: "fail"}),
    "prefetch_build_and_claim_fail": ("prefetch", {1: (0, "none")}, False, {3: "fail"}),
    "prefetch_claim_raises_one_rank": ("prefetch", {}, False, {2: "raise"}),
    "prefetch_all_built": ("prefetch", {}, False, {}),
    "all_built": ("normal", {}, False, {}),
    "normal_component_assert": ("normal", {3: (0, "assert")}, False, {}),
    "prefetch_component_assert": ("prefetch", {1: (1, "assert")}, False, {}),
    "mixed_all_built": ("mixed", {}, False, {}),
    "mixed_one_rank_none": ("mixed", {2: (0, "none")}, False, {}),
    "mixed_claim_fails_one_rank": ("mixed", {}, False, {0: "fail"}),
    "mixed_claim_raises_one_rank": ("mixed", {}, False, {1: "raise"}),
    "mixed_build_raises_on_no_prefetch_rank": ("mixed", {3: (1, "raise")}, False, {}),
}
NO_PREFETCH_NEEDED_RANKS = {2, 3}


def _cache(cp_group, tp_group, components, reductions):
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.attn_cp_group = cp_group
    cache.attn_tp_group = tp_group
    cache.tp_world_size = 1
    # page_size is a property backed by tree_core.
    cache.tree_core = types.SimpleNamespace(
        page_size=1,
        empty_match_result=types.SimpleNamespace(
            device_indices=torch.empty(0, dtype=torch.int64)
        ),
    )
    cache._components_tuple = tuple(components)
    all_reduce = cache._all_reduce_attn_groups

    def counted(tensor, op):
        reductions.append(tensor.tolist() if tensor.numel() > 1 else int(tensor.item()))
        all_reduce(tensor, op)

    cache._all_reduce_attn_groups = counted
    return cache


def _wrapper(cache, backend, prefetch):
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = cache
    wrapper.cache_linker = backend
    hit = ExternalCacheHitMarker(
        prefix_key=None, tail_hashes=["h0", "h1"], device_hit_len=0
    )
    wrapper.hit_markers = {"r": hit}
    wrapper.host_prefetch_hits = {"r": hit} if prefetch else {}
    wrapper.pending_host_prefetch_submissions = {}
    return wrapper


def _rank(rank, world_size, port, results):
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=10),
    )
    try:
        cp = [dist.new_group(r, backend="gloo") for r in ([0, 1], [2, 3])]
        tp = [dist.new_group(r, backend="gloo") for r in ([0, 2], [1, 3])]
        # LayerSplit-style revalidation group: its MIN hangs if a rank skips it.
        reval = dist.new_group([0, 1, 2, 3], backend="gloo")
        out = {}
        for name, (path, failures, other_holder, claims) in SCENARIOS.items():
            # One allocator pair for all repeats: a leak accumulates.
            components = [FakeComponent(PoolName.KV), FakeComponent(PoolName.SWA)]
            if rank in failures:
                index, failure = failures[rank]
                components[index].failure = failure
            claim = claims.get(rank, "ok")
            reps = []
            for _ in range(REPEATS):
                reductions = []
                cache = _cache(cp[rank // 2], tp[rank % 2], components, reductions)
                revalidated = []

                def revalidate_load(transfers, _seen=revalidated):
                    _seen.append(1)
                    verdict = torch.ones(1, dtype=torch.int)
                    dist.all_reduce(verdict, op=dist.ReduceOp.MIN, group=reval)
                    return False  # stop right after it: a plain miss

                no_prefetch = path == "mixed" and rank in NO_PREFETCH_NEEDED_RANKS
                if no_prefetch:
                    backend, ended = _no_prefetch_needed_backend()
                elif path in ("prefetch", "mixed"):
                    backend, ended = _prefetch_backend(other_holder, claim)
                else:
                    backend, ended = MagicMock(), None
                backend.revalidate_load = revalidate_load
                wrapper = _wrapper(cache, backend, path != "normal")
                try:
                    indices, node = wrapper.load_back(
                        types.SimpleNamespace(rid="r", last_node="node")
                    )
                    result = "miss" if indices.numel() == 0 and node == "node" else "?"
                except ReachedPrepare:
                    result = "prepare"
                except (RuntimeError, AssertionError) as error:
                    # The failing rank re-raises after the shared cleanup;
                    # report it rather than dying so the others finish.
                    result = f"raised:{error}"
                allocated = [c.allocated for c in components]
                if result == "prepare":
                    for c in components:  # the load owns them now
                        c.allocated = 0
                reps.append(
                    (
                        result,
                        allocated,
                        len(revalidated),
                        (
                            None
                            if path == "normal" or result == "prepare"
                            else (
                                _no_prefetch_needed_released(backend, ended)
                                if no_prefetch
                                else _session_released(
                                    backend, ended, other_holder, claim
                                )
                            )
                        ),
                        len(reductions),
                        (
                            len(reductions[2])
                            if path != "normal"
                            and len(reductions) > 2
                            and isinstance(reductions[2], list)
                            else None
                        ),
                    )
                )
            out[name] = reps
        dist.barrier()
        results[rank] = out
    finally:
        dist.destroy_process_group()


class TestConstructionConsensus(CustomTestCase):
    def test_every_rank_aborts_together_every_time(self):
        import torch.multiprocessing as mp

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with mp.Manager() as manager:
            results = manager.dict()
            context = mp.spawn(_rank, args=(4, port, results), nprocs=4, join=False)
            deadline = time.monotonic() + 180
            try:
                while not context.join(timeout=2):
                    if time.monotonic() > deadline:
                        self.fail("a rank hung: construction was not agreed")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.kill()
                    process.join(timeout=10)
            outcomes = [results[rank] for rank in range(4)]

        # (result, allocated, revalidations, session released, reductions,
        #  length of the third reduction)
        miss_prefetch = ("miss", [0, 0], 0, True, 3, 2)
        expected = {
            "normal_second_component_none": ("miss", [0, 0], 0, None, 1, None),
            # The raising rank re-raises (see overrides); the others miss.
            "normal_first_component_raises": ("miss", [0, 0], 0, None, 1, None),
            "normal_component_assert": ("miss", [0, 0], 0, None, 1, None),
            # ready, valid, then [claimed, constructed] in one reduction.
            "prefetch_one_rank_fails": miss_prefetch,
            "prefetch_second_component_raises": miss_prefetch,
            "prefetch_fails_with_other_holder": miss_prefetch,
            # Everyone built, a claim failed: free, then the normal path builds
            # again, agrees once more and reaches revalidate_load.
            "prefetch_claim_fails_one_rank": ("miss", [0, 0], 1, True, 4, 2),
            # A build failure wins: a miss without retrying the normal path.
            "prefetch_build_and_claim_fail": miss_prefetch,
            "prefetch_claim_raises_one_rank": miss_prefetch,
            "prefetch_component_assert": miss_prefetch,
            # Ranks 2 and 3 needed no DFS read; the agreement is the same.
            "mixed_all_built": ("prepare", [1, 1], 0, None, 3, 2),
            "mixed_one_rank_none": miss_prefetch,
            "mixed_claim_fails_one_rank": ("miss", [0, 0], 1, True, 4, 2),
            "mixed_claim_raises_one_rank": miss_prefetch,
            "mixed_build_raises_on_no_prefetch_rank": miss_prefetch,
            # Three reductions, built once, straight on to PREPARE.
            "prefetch_all_built": ("prepare", [1, 1], 0, None, 3, 2),
            # Agreement passes, revalidate_load runs, then its miss frees all.
            "all_built": ("miss", [0, 0], 1, None, 1, None),
        }
        # The rank whose build or claim raised re-raises after the same
        # agreement and cleanup; nobody retries the normal path.
        raised_prefetch = lambda message: (f"raised:{message}", [0, 0], 0, True, 3, 2)
        overrides = {
            ("normal_first_component_raises", 1): (
                "raised:build failed",
                [0, 0],
                0,
                None,
                1,
                None,
            ),
            ("normal_component_assert", 3): (
                "raised:tree invariant broken",
                [0, 0],
                0,
                None,
                1,
                None,
            ),
            ("prefetch_second_component_raises", 0): raised_prefetch("build failed"),
            ("prefetch_claim_raises_one_rank", 2): raised_prefetch("claim failed"),
            ("prefetch_component_assert", 1): raised_prefetch("tree invariant broken"),
            ("mixed_claim_raises_one_rank", 1): raised_prefetch("claim failed"),
            ("mixed_build_raises_on_no_prefetch_rank", 3): raised_prefetch(
                "build failed"
            ),
        }
        for rank, out in enumerate(outcomes):
            for name, want in expected.items():
                want = overrides.get((name, rank), want)
                with self.subTest(rank=rank, scenario=name):
                    self.assertEqual(len(out[name]), REPEATS)
                    for rep, got in enumerate(out[name]):
                        self.assertEqual(got, want, f"repeat {rep}")


class TestInterruptDuringConstruction(CustomTestCase):
    """A non-Exception interruption still joins the agreement and frees what
    was built before it is re-raised."""

    def _load_back(
        self,
        prefetch,
        second_failure="stop",
        claim_error=None,
        status="dfs_prefetched",
        cancel_error=None,
        expected=BuildStop,
        abort_error=None,
    ):
        reductions = []
        first, second = FakeComponent(PoolName.KV), FakeComponent(PoolName.SWA)
        second.failure = second_failure
        first.abort_error = abort_error
        cache = types.SimpleNamespace(
            page_size=1,
            tree_core=types.SimpleNamespace(
                empty_match_result=types.SimpleNamespace(
                    device_indices=torch.empty(0, dtype=torch.int64)
                )
            ),
            _components_tuple=(first, second),
            _all_reduce_attn_groups=lambda tensor, op: reductions.append(
                tensor.tolist() if tensor.numel() > 1 else int(tensor.item())
            ),
        )
        backend = MagicMock()
        backend.get_host_prefetch_status.return_value = status
        backend.revalidate_host_prefetch.return_value = True
        backend.revalidate_no_prefetch_needed.return_value = True
        backend.claim_ready_host_prefetch.return_value = True
        if claim_error is not None:
            backend.claim_ready_host_prefetch.side_effect = claim_error
        if cancel_error is not None:
            # Interrupted inside the claim step; the cleanup cancel succeeds.
            backend.cancel_host_prefetch.side_effect = [cancel_error, None]
        wrapper = _wrapper(cache, backend, prefetch)
        with self.assertRaises(expected) as raised:
            wrapper.load_back(types.SimpleNamespace(rid="r", last_node="node"))
        self.raised = raised.exception
        return reductions, first, backend

    def test_normal_path(self):
        reductions, first, backend = self._load_back(prefetch=False)
        self.assertEqual(reductions, [0])
        self.assertEqual(first.allocated, 0)
        backend.revalidate_load.assert_not_called()

    def test_prefetch_path(self):
        reductions, first, backend = self._load_back(prefetch=True)
        self.assertEqual(reductions, [1, 1, [1, 0]])
        self.assertEqual(first.allocated, 0)
        backend.cancel_host_prefetch.assert_called_with("r")

    def test_claim_interrupt_after_building(self):
        # Built everything, then the claim is interrupted: this rank still
        # joins the reduction (as not claimed and not constructed), frees what
        # it built and re-raises.
        reductions, first, backend = self._load_back(
            prefetch=True, second_failure=None, claim_error=BuildStop()
        )
        self.assertEqual(reductions, [1, 1, [0, 0]])
        self.assertEqual(first.allocated, 0)
        backend.cancel_host_prefetch.assert_called_with("r")

    def test_no_prefetch_needed_cancel_interrupt(self):
        reductions, first, backend = self._load_back(
            prefetch=True,
            second_failure=None,
            status="no_prefetch_needed",
            cancel_error=BuildStop(),
        )
        self.assertEqual(reductions, [1, 1, [0, 0]])
        self.assertEqual(first.allocated, 0)


class TestCleanupAfterDisagreement(CustomTestCase):
    """A failing ABORT does not skip the prefetch cancel, and does not replace
    the build or claim error that caused the cleanup."""

    _load_back = TestInterruptDuringConstruction._load_back

    def test_build_error_stays_primary_when_abort_fails(self):
        reductions, _, backend = self._load_back(
            prefetch=True,
            second_failure="raise",
            expected=RuntimeError,
            abort_error=ValueError("abort failed"),
        )
        self.assertEqual(str(self.raised), "build failed")
        self.assertEqual(reductions, [1, 1, [1, 0]])
        backend.cancel_host_prefetch.assert_called_with("r")

    def test_claim_error_stays_primary_when_abort_fails(self):
        reductions, _, backend = self._load_back(
            prefetch=True,
            second_failure=None,
            claim_error=RuntimeError("claim failed"),
            expected=RuntimeError,
            abort_error=ValueError("abort failed"),
        )
        self.assertEqual(str(self.raised), "claim failed")
        self.assertEqual(reductions, [1, 1, [0, 0]])
        backend.cancel_host_prefetch.assert_called_with("r")

    def test_abort_error_raised_without_pending_error(self):
        # An expected miss (None) needs no re-raise; the cleanup error does.
        reductions, _, backend = self._load_back(
            prefetch=True,
            second_failure="none",
            expected=ValueError,
            abort_error=ValueError("abort failed"),
        )
        self.assertEqual(str(self.raised), "abort failed")
        self.assertEqual(reductions, [1, 1, [1, 0]])
        backend.cancel_host_prefetch.assert_called_with("r")

    def test_cancel_error_raised_after_successful_abort(self):
        # The free succeeds, the cancel of the cleanup raises: no pending error,
        # so the cancel error is raised.
        reductions = []
        first, second = FakeComponent(PoolName.KV), FakeComponent(PoolName.SWA)
        second.failure = "none"
        cache = types.SimpleNamespace(
            page_size=1,
            tree_core=types.SimpleNamespace(
                empty_match_result=types.SimpleNamespace(
                    device_indices=torch.empty(0, dtype=torch.int64)
                )
            ),
            _components_tuple=(first, second),
            _all_reduce_attn_groups=lambda tensor, op: reductions.append(
                tensor.tolist() if tensor.numel() > 1 else int(tensor.item())
            ),
        )
        backend = MagicMock()
        backend.get_host_prefetch_status.return_value = "dfs_prefetched"
        backend.revalidate_host_prefetch.return_value = True
        backend.claim_ready_host_prefetch.return_value = True
        backend.cancel_host_prefetch.side_effect = ValueError("cancel failed")
        wrapper = _wrapper(cache, backend, True)
        with self.assertRaisesRegex(ValueError, "cancel failed"):
            wrapper.load_back(types.SimpleNamespace(rid="r", last_node="node"))
        self.assertEqual(first.allocated, 0)

    def test_normal_path_abort_error(self):
        reductions, _, backend = self._load_back(
            prefetch=False,
            second_failure="none",
            expected=ValueError,
            abort_error=ValueError("abort failed"),
        )
        self.assertEqual(reductions, [0])
        backend.cancel_host_prefetch.assert_not_called()
        backend.revalidate_load.assert_not_called()


class _TrackingAllocator:
    """Hands out fresh slot ids and fails on a double or unknown free."""

    def __init__(self, first_id, *, alloc_none=False, available=1 << 20):
        self.next_id = first_id
        self.outstanding = set()
        self.alloc_none = alloc_none
        self.available = available
        self.freed = []

    def available_size(self):
        return self.available

    def alloc(self, num_tokens):
        if self.alloc_none:
            return None
        ids = list(range(self.next_id, self.next_id + num_tokens))
        self.next_id += num_tokens
        self.outstanding.update(ids)
        return torch.tensor(ids, dtype=torch.int64)

    def free(self, indices):
        for slot in indices.tolist():
            assert slot in self.outstanding, f"double or unknown free of {slot}"
            self.outstanding.remove(slot)
            self.freed.append(slot)


class TestRealFullSwaPartialFailure(CustomTestCase):
    """Real Full and SWA components: when SWA cannot build after Full
    allocated, the agreed abort frees Full's slots exactly once."""

    def _run(self, *, swa_alloc_none=False, swa_evict_raises=False, prefetch=False):
        full_alloc = _TrackingAllocator(100)
        swa_alloc = _TrackingAllocator(
            500, alloc_none=swa_alloc_none, available=0 if swa_evict_raises else 1 << 20
        )

        def evict(params):
            if getattr(params, "swa_num_tokens", 0):
                raise RuntimeError("swa evict failed")

        cache = types.SimpleNamespace(
            page_size=1,
            is_swa_enabled=True,
            evict=evict,
            token_to_kv_pool_allocator=types.SimpleNamespace(
                full_attn_allocator=full_alloc, swa_attn_allocator=swa_alloc
            ),
            tree_core=types.SimpleNamespace(
                empty_match_result=types.SimpleNamespace(
                    device_indices=torch.empty(0, dtype=torch.int64)
                )
            ),
            # Single rank: the agreement keeps the local verdict.
            _all_reduce_attn_groups=lambda tensor, op: None,
        )
        full = FullComponent.__new__(FullComponent)
        full.cache = cache
        swa = SWAComponent.__new__(SWAComponent)
        swa.cache = cache
        swa.sliding_window_size = 4
        cache._components_tuple = (full, swa)
        backend = MagicMock()
        backend.get_host_prefetch_status.return_value = "dfs_prefetched"
        backend.revalidate_host_prefetch.return_value = True
        backend.claim_ready_host_prefetch.return_value = True
        wrapper = _wrapper(cache, backend, prefetch)
        return wrapper, full_alloc, swa_alloc, backend

    def test_swa_none_frees_full_once(self):
        for prefetch in (False, True):
            with self.subTest(prefetch=prefetch):
                wrapper, full_alloc, swa_alloc, backend = self._run(
                    swa_alloc_none=True, prefetch=prefetch
                )
                indices, node = wrapper.load_back(
                    types.SimpleNamespace(rid="r", last_node="node")
                )
                self.assertEqual((indices.numel(), node), (0, "node"))
                self.assertEqual(full_alloc.outstanding, set())
                self.assertEqual(sorted(full_alloc.freed), [100, 101])
                self.assertEqual(swa_alloc.freed, [])
                if prefetch:
                    backend.cancel_host_prefetch.assert_called_with("r")

    def test_swa_evict_error_frees_full_once_then_raises(self):
        for prefetch in (False, True):
            with self.subTest(prefetch=prefetch):
                wrapper, full_alloc, swa_alloc, backend = self._run(
                    swa_evict_raises=True, prefetch=prefetch
                )
                with self.assertRaisesRegex(RuntimeError, "swa evict failed"):
                    wrapper.load_back(types.SimpleNamespace(rid="r", last_node="node"))
                self.assertEqual(full_alloc.outstanding, set())
                self.assertEqual(sorted(full_alloc.freed), [100, 101])
                self.assertEqual(swa_alloc.outstanding, set())
                if prefetch:
                    backend.cancel_host_prefetch.assert_called_with("r")


class TestComponentRollback(CustomTestCase):
    class _Slots:
        def to(self, dtype):
            raise RuntimeError("conversion failed")

    def test_full_component_frees_slots_it_could_not_hand_over(self):
        component = FullComponent.__new__(FullComponent)
        allocator = MagicMock()
        allocator.available_size.return_value = 1 << 20
        slots = self._Slots()
        allocator.alloc.return_value = slots
        component.cache = types.SimpleNamespace(page_size=1, evict=MagicMock())
        component._full_allocator = lambda: allocator
        with self.assertRaises(RuntimeError):
            component.build_external_linker_transfer(
                LinkerTransferPhase.LOAD, None, ["k0", "k1"]
            )
        allocator.free.assert_called_once_with(slots)

    def test_swa_component_frees_slots_it_could_not_hand_over(self):
        component = SWAComponent.__new__(SWAComponent)
        allocator = MagicMock()
        allocator.available_size.return_value = 1 << 20
        slots = self._Slots()
        allocator.alloc.return_value = slots
        component.sliding_window_size = 4
        component.cache = types.SimpleNamespace(
            page_size=1,
            evict=MagicMock(),
            token_to_kv_pool_allocator=types.SimpleNamespace(
                swa_attn_allocator=allocator
            ),
        )
        with self.assertRaises(RuntimeError):
            component.build_external_linker_transfer(
                LinkerTransferPhase.LOAD, None, ["k0", "k1"]
            )
        allocator.free.assert_called_once_with(slots)


if __name__ == "__main__":
    unittest.main()
