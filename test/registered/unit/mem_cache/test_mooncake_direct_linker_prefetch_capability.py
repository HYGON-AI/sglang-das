"""Startup capability agreement for Mooncake waiting-queue DFS prefetch.

The linker decides once, before any prefetch worker starts, whether every
rank can run the prefetch; the scheduler then reads only that decision.
"""

import time
import types
import unittest
from unittest import mock
from unittest.mock import MagicMock

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=180, suite="base-a-test-cpu")

_LINKER_LOGGER = "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker"


class _Store:
    def __init__(self, *, available=True, status="ready", missing=(), error=None):
        self.available = available
        self.status = status
        self.error = error
        self.arena_queries = 0
        for name in (
            "batch_get_session_start_with_sources",
            "batch_get_session_prefetch",
            "batch_get_session_refresh",
        ):
            if name not in missing:
                setattr(self, name, lambda *args, **kwargs: None)
        if "dfs_prefetch_arena_available" not in missing:
            self.dfs_prefetch_arena_available = self._arena_available
        self.dfs_prefetch_arena_status = lambda: self.status

    def _arena_available(self):
        self.arena_queries += 1
        if self.error is not None:
            raise self.error
        return self.available


def _linker(store):
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.storage = types.SimpleNamespace(store=store)
    return linker


def _params(cp=None, tp=None, full_tp=None):
    return types.SimpleNamespace(
        attn_cp_cache_group=cp, attn_tp_cache_group=tp, tp_cache_group=full_tp
    )


class _FakeDist:
    """Records reductions; ``peer_value`` is the MIN contribution of peers."""

    def __init__(self, world_sizes, peer_value=1):
        self.world_sizes = world_sizes
        self.peer_value = peer_value
        self.calls = []

    def patch(self):
        dist = torch.distributed
        return mock.patch.multiple(
            dist,
            is_available=lambda: True,
            is_initialized=lambda: True,
            get_world_size=lambda group=None: self.world_sizes[group],
            all_reduce=self.all_reduce,
        )

    def all_reduce(self, tensor, op=None, group=None):
        self.calls.append((group, op))
        tensor.fill_(min(int(tensor.item()), self.peer_value))


class TestLocalCapability(CustomTestCase):
    def test_not_requested_does_not_query_store(self):
        store = _Store()
        ok, reason = _linker(store)._local_host_prefetch_capability(False)
        self.assertFalse(ok)
        self.assertIn("disabled", reason)
        self.assertEqual(store.arena_queries, 0)

    def test_missing_api_is_unavailable(self):
        store = _Store(missing=("dfs_prefetch_arena_available",))
        ok, reason = _linker(store)._local_host_prefetch_capability(True)
        self.assertFalse(ok)
        self.assertIn("dfs_prefetch_arena_available", reason)

    def test_unconfigured_arena_reports_status(self):
        store = _Store(available=False, status="not configured")
        ok, reason = _linker(store)._local_host_prefetch_capability(True)
        self.assertFalse(ok)
        self.assertIn("not configured", reason)
        self.assertIn("MC_STORE_DFS_PREFETCH_ARENA_SIZE_BYTES", reason)

    def test_arena_query_error_is_unavailable(self):
        store = _Store(error=RuntimeError("boom"))
        ok, reason = _linker(store)._local_host_prefetch_capability(True)
        self.assertFalse(ok)
        self.assertIn("boom", reason)

    def test_ready_arena_is_available(self):
        ok, reason = _linker(_Store())._local_host_prefetch_capability(True)
        self.assertTrue(ok)
        self.assertEqual(reason, "ready")


class TestResolveHostPrefetch(CustomTestCase):
    def test_single_process_uses_local_verdict(self):
        linker = _linker(_Store())
        self.assertTrue(
            linker._resolve_host_prefetch_enabled(_params(), requested=True)
        )

        linker = _linker(_Store(available=False, status="not configured"))
        with self.assertLogs(_LINKER_LOGGER, level="WARNING") as logs:
            enabled = linker._resolve_host_prefetch_enabled(_params(), requested=True)
        self.assertFalse(enabled)
        self.assertIn("not configured", logs.output[0])

    def test_any_unavailable_rank_disables_all(self):
        cp, tp = object(), object()
        fake = _FakeDist({cp: 2, tp: 4}, peer_value=0)
        with fake.patch():
            enabled = _linker(_Store())._resolve_host_prefetch_enabled(
                _params(cp=cp, tp=tp), requested=True
            )
        self.assertFalse(enabled)
        # Same groups and order as UnifiedRadixCache._all_reduce_attn_groups.
        self.assertEqual(
            fake.calls,
            [
                (cp, torch.distributed.ReduceOp.MIN),
                (tp, torch.distributed.ReduceOp.MIN),
            ],
        )

    def test_reduction_falls_back_to_full_tp_group(self):
        cp, tp, full_tp = object(), object(), object()
        fake = _FakeDist({cp: 1, tp: 1, full_tp: 8})
        with fake.patch():
            enabled = _linker(_Store())._resolve_host_prefetch_enabled(
                _params(cp=cp, tp=tp, full_tp=full_tp), requested=True
            )
        self.assertTrue(enabled)
        self.assertEqual(fake.calls, [(full_tp, torch.distributed.ReduceOp.MIN)])

    def test_every_rank_participates_even_when_not_requested(self):
        cp = object()
        fake = _FakeDist({cp: 2})
        with fake.patch():
            enabled = _linker(_Store())._resolve_host_prefetch_enabled(
                _params(cp=cp), requested=False
            )
        self.assertFalse(enabled)
        self.assertEqual(len(fake.calls), 1)


def _gloo_rank(rank, world_size, port, unavailable_rank, flag_off_rank, results):
    """One rank of a CP x TP = 2 x 2 Gloo mesh running the real reduction."""
    import datetime

    # A misaligned reduction must fail fast instead of hanging the test.
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        # Every rank creates every subgroup in the same order.
        cp_groups = [
            torch.distributed.new_group([0, 1]),
            torch.distributed.new_group([2, 3]),
        ]
        tp_groups = [
            torch.distributed.new_group([0, 2]),
            torch.distributed.new_group([1, 3]),
        ]
        params = _params(cp=cp_groups[rank // 2], tp=tp_groups[rank % 2])
        store = _Store(available=rank != unavailable_rank, status="not configured")
        enabled = _linker(store)._resolve_host_prefetch_enabled(
            params, requested=rank != flag_off_rank
        )
        # A follow-up collective proves no rank is left blocked or skewed.
        torch.distributed.barrier()
        results[rank] = enabled
    finally:
        torch.distributed.destroy_process_group()


class TestResolveHostPrefetchGloo(CustomTestCase):
    def _run(self, *, unavailable_rank=-1, flag_off_rank=-1):
        import socket

        import torch.multiprocessing as mp

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        world_size = 4
        with mp.Manager() as manager:
            results = manager.dict()
            context = mp.spawn(
                _gloo_rank,
                args=(world_size, port, unavailable_rank, flag_off_rank, results),
                nprocs=world_size,
                join=False,
            )
            deadline = time.monotonic() + 300
            while not context.join(timeout=5):
                if time.monotonic() > deadline:
                    for process in context.processes:
                        if process.is_alive():
                            process.kill()
                    self.fail("Gloo ranks did not finish; a reduction is misaligned")
            return [results[rank] for rank in range(world_size)]

    def test_all_ranks_ready(self):
        self.assertEqual(self._run(), [True] * 4)

    def test_one_rank_without_arena_disables_every_rank(self):
        self.assertEqual(self._run(unavailable_rank=3), [False] * 4)

    def test_one_rank_with_flag_off_disables_every_rank(self):
        self.assertEqual(self._run(flag_off_rank=1), [False] * 4)


def _scheduler(*, flag, tree_cache):
    s = Scheduler.__new__(Scheduler)
    s.server_args = types.SimpleNamespace(
        mooncake_enable_waiting_queue_dfs_prefetch=flag,
        enable_unified_cache_external_linker=True,
        unified_cache_external_linker_backend="mooncake",
    )
    s.disaggregation_mode = DisaggregationMode.PREFILL
    s.schedule_policy = "fcfs"
    s.ps = types.SimpleNamespace(pp_size=1)
    s.tree_cache = tree_cache
    return s


class TestSchedulerPrefetchSwitch(CustomTestCase):
    """The scheduler's startup decision follows the user flag, then the
    linker's resolved capability (single rank: no reduction)."""

    def test_user_flag_off_never_touches_tree_cache(self):
        tree_cache = MagicMock()
        tree_cache.waiting_queue_prefetch_enabled.side_effect = AssertionError(
            "tree cache must not be consulted without the user flag"
        )
        s = _scheduler(flag=False, tree_cache=tree_cache)
        self.assertEqual(s._resolve_prefetch_startup_switches(None), (False, False))

    def test_follows_resolved_linker_switch(self):
        tree_cache = MagicMock()
        tree_cache.waiting_queue_prefetch_enabled.return_value = False
        s = _scheduler(flag=True, tree_cache=tree_cache)
        self.assertEqual(s._resolve_prefetch_startup_switches(None), (False, False))

        tree_cache.waiting_queue_prefetch_enabled.return_value = True
        self.assertEqual(s._resolve_prefetch_startup_switches(None), (True, True))


if __name__ == "__main__":
    unittest.main()
