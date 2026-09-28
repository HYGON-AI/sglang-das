"""Collective-order regression coverage for waiting-queue DFS prefetch."""

import os
import queue
import socket
import threading
import time
import unittest
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.distributed.parallel_state import create_custom_parallel_group
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    PreparedHostPrefetch,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.managers.scheduler import Scheduler
import sglang.srt.managers.scheduler as scheduler_module

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _FakeLinker:
    def __init__(self):
        self.submissions = []
        self.cancelled = []

    def submit_host_prefetch(self, rid, transfers):
        self.submissions.append(rid)
        return True

    def cancel_host_prefetch(self, rid):
        self.cancelled.append(rid)


class _FakeAdmissionLinker:
    def __init__(self, rank):
        self.rank = rank

    def get_host_prefetch_admission_state(self, rid):
        states = {
            0: {"a": "pending", "b": "terminal"},
            1: {"a": "dfs_prefetched"},
        }
        return states[0 if self.rank == 0 else 1].get(rid, "not_tracked")


def _distributed_round_worker(rank, world_size, port):
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        ranks = list(range(world_size))
        main_group = dist.new_group(ranks=ranks, backend="gloo")
        bg_cp_group = create_custom_parallel_group(ranks, backend="gloo")
        bg_tp_group = create_custom_parallel_group(ranks, backend="gloo")

        scheduler = Scheduler.__new__(Scheduler)
        scheduler._bg_prefetch_jobs = queue.Queue()
        scheduler._bg_prefetch_acks = queue.Queue()
        scheduler._bg_pending_prefetch_jobs = {}
        scheduler._bg_prefetch_join_age = {}
        scheduler._bg_attn_cp_cpu_group = bg_cp_group
        scheduler._bg_attn_tp_cpu_group = bg_tp_group
        # Set by Scheduler.__init__, which this fixture bypasses.
        scheduler._bg_completed_epoch = 0

        linker = _FakeLinker()

        def enqueue_job(rid):
            job = PreparedHostPrefetch(
                rid=rid,
                locally_eligible=True,
                transfers=(),
                cache_linker=linker,
                cancelled=threading.Event(),
            )
            scheduler._bg_prefetch_jobs.put(job)
            return job

        jobs = {}
        local_jobs = ["a", "b", "c", "d"] if rank == 0 else ["a"]
        for rid in local_jobs:
            job = enqueue_job(rid)
            jobs[rid] = job

        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.attn_cp_group = main_group
        cache.attn_tp_group = None
        cache.tp_world_size = world_size
        cache.tp_group = main_group
        cache.linker = _FakeAdmissionLinker(rank)

        start = threading.Barrier(2)
        bg_errors = []

        def run_background_round():
            try:
                start.wait()
                if rank == 0:
                    time.sleep(0.15)
                scheduler._run_bg_prefetch_round()
            except BaseException as exc:
                bg_errors.append(exc)

        bg_thread = threading.Thread(target=run_background_round)
        bg_thread.start()
        start.wait()
        if rank != 0:
            time.sleep(0.15)

        # This scheduler-thread collective intentionally races the background
        # protocol, but uses the original PG. Rank 0 enters it first while
        # rank 1 enters the background protocol first.
        states = cache.get_waiting_queue_prefetch_admission_states(
            ["a", "b"] if rank == 0 else ["a"]
        )
        bg_thread.join(timeout=10)
        assert not bg_thread.is_alive(), "background prefetch round did not finish"
        assert not bg_errors, f"background prefetch failed: {bg_errors}"
        assert states == {"a": "pending", "b": "terminal"}

        acknowledgements = {}
        while True:
            try:
                job, submitted = scheduler._bg_prefetch_acks.get_nowait()
            except queue.Empty:
                break
            acknowledgements[job.rid] = submitted
        expected = {"a": True}
        assert acknowledgements == expected, (rank, acknowledgements)
        assert linker.submissions == ["a"]

        # Other ranks' b snapshots arrive after they drained epoch 1. Rank 0
        # must retain b and submit it with them in epoch 2. Its cancelled d
        # should retire globally even though they never had a d job.
        if rank != 0:
            jobs["b"] = enqueue_job("b")
        else:
            jobs["d"].cancelled.set()
        dist.barrier()
        scheduler._run_bg_prefetch_round()

        acknowledgements = {}
        while True:
            try:
                job, submitted = scheduler._bg_prefetch_acks.get_nowait()
            except queue.Empty:
                break
            acknowledgements[job.rid] = submitted
        expected = {"b": True, "d": False} if rank == 0 else {"b": True}
        assert acknowledgements == expected, (rank, acknowledgements)
        assert linker.submissions == ["a", "b"]

        # c never appears on the other ranks, so the four-epoch grace expires
        # it to the ordinary admission path instead of leaving it pending.
        scheduler._run_bg_prefetch_round()
        assert scheduler._bg_prefetch_acks.empty()
        scheduler._run_bg_prefetch_round()
        acknowledgements = {}
        while True:
            try:
                job, submitted = scheduler._bg_prefetch_acks.get_nowait()
            except queue.Empty:
                break
            acknowledgements[job.rid] = submitted
        expected = {"c": False} if rank == 0 else {}
        assert acknowledgements == expected, (rank, acknowledgements)

        # Empty rounds synchronize presence flags but skip object gathers.
        scheduler._run_bg_prefetch_round()
        dist.barrier()

        # Diagnostic only: measure one empty and one single-RID scheduler
        # control round. These timings exclude real Mooncake I/O and use the
        # synthetic two-rank Gloo groups above.
        benchmark_rounds = 20
        object_gather_calls = 0
        original_all_gather_object = dist.all_gather_object

        def count_object_gather(*args, **kwargs):
            nonlocal object_gather_calls
            object_gather_calls += 1
            return original_all_gather_object(*args, **kwargs)

        dist.all_gather_object = count_object_gather
        dist.barrier()
        started = time.perf_counter()
        for _ in range(benchmark_rounds):
            assert cache.get_waiting_queue_prefetch_admission_states([]) == {}
            scheduler._run_bg_prefetch_round()
        assert object_gather_calls == 0, object_gather_calls
        idle_ms = (time.perf_counter() - started) * 1000 / benchmark_rounds
        idle_ms = torch.tensor([idle_ms], dtype=torch.float64)
        dist.all_reduce(idle_ms, op=dist.ReduceOp.MAX)

        dist.barrier()
        started = time.perf_counter()
        for index in range(benchmark_rounds):
            rid = f"bench-{index}"
            enqueue_job(rid)
            assert cache.get_waiting_queue_prefetch_admission_states([rid]) == {
                rid: "not_tracked"
            }
            scheduler._run_bg_prefetch_round()
            job, submitted = scheduler._bg_prefetch_acks.get_nowait()
            assert job.rid == rid and submitted
        assert object_gather_calls == benchmark_rounds * 3, object_gather_calls
        dist.all_gather_object = original_all_gather_object
        request_ms = (time.perf_counter() - started) * 1000 / benchmark_rounds
        request_ms = torch.tensor([request_ms], dtype=torch.float64)
        dist.all_reduce(request_ms, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(
                "BG_PREFETCH_GLOO_BENCH "
                f"ranks={world_size} iterations={benchmark_rounds} "
                f"idle_control_ms={idle_ms.item():.3f} "
                f"single_rid_control_ms={request_ms.item():.3f}"
            )

        # A request that appears only after DSPARK's non-overlap forward has
        # started must reach scheduler-owned lookup before that forward returns.
        ingress_group = create_custom_parallel_group(ranks, backend="gloo")
        ingress_cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        ingress_cache.attn_cp_group = ingress_group
        ingress_cache.attn_tp_group = None
        ingress_cache.tp_group = ingress_group
        ingress_cache.tp_world_size = world_size
        forward_started = threading.Event()
        times = {}

        class FakeRequest:
            mm_inputs = None
            input_embeds = None
            session_id = None
            session_params = None
            lora_id = None
            positional_embed_overrides = None
            sampling_params = SimpleNamespace()

        class FakeReceiver:
            def __init__(self):
                self.sent = False

            def ingress_sync_groups(self):
                return [ingress_group]

            def recv_requests(self):
                ready = (
                    rank == 0
                    and not self.sent
                    and forward_started.is_set()
                    and time.monotonic() - times["start"] >= 0.05
                )
                payload = [ready if rank == 0 else None]
                dist.broadcast_object_list(payload, src=0, group=ingress_group)
                if payload[0]:
                    self.sent = True
                    return [FakeRequest(), object()]
                return []

        def fake_forward(_batch):
            times["start"] = time.monotonic()
            forward_started.set()
            # The model and ingress collectives run on different Gloo groups.
            model_sync = torch.ones(1)
            dist.all_reduce(model_sync, group=main_group)
            time.sleep(0.35)
            times["end"] = time.monotonic()
            return "forward-result"

        ingress_scheduler = Scheduler.__new__(Scheduler)
        ingress_scheduler.enable_waiting_queue_dfs_prefetch = True
        ingress_scheduler._forward_ingress_enabled = True
        ingress_scheduler._bg_error = None
        ingress_scheduler.forward_ct = 0
        ingress_scheduler.enable_overlap = False
        ingress_scheduler.device = "cpu"
        ingress_scheduler.device_module = SimpleNamespace(
            StreamContext=lambda _stream: nullcontext()
        )
        ingress_scheduler.forward_stream = SimpleNamespace(wait_stream=lambda _s: None)
        ingress_scheduler.schedule_stream = SimpleNamespace(
            wait_stream=lambda _stream: None, synchronize=lambda: None
        )
        ingress_scheduler.model_worker = SimpleNamespace(
            forward_batch_generation=fake_forward
        )
        ingress_scheduler.tree_cache = ingress_cache
        ingress_scheduler.request_receiver = FakeReceiver()
        ingress_scheduler._request_bg_prefetch_epoch = lambda: None
        ingress_scheduler._forward_deferred_reqs = []
        ingress_scheduler.enable_hicache_storage = False
        ingress_cache.prepare_external_linker_prefetch = lambda _req: None
        linker = SimpleNamespace(
            lookup=lambda: times.setdefault("lookup", time.monotonic())
        )
        prefetch_req = SimpleNamespace(
            prefill_attempt_count=0,
            init_next_round_input=lambda *_args, **_kwargs: linker.lookup(),
        )
        ingress_scheduler.handle_generate_request = lambda _req: (
            ingress_scheduler._prefetch_kvcache(prefetch_req)
        )
        ingress_scheduler._forward_launch_executor = ThreadPoolExecutor(max_workers=1)
        try:
            batch = SimpleNamespace(spec_algorithm=SimpleNamespace(is_none=lambda: False))
            with patch.object(scheduler_module, "TokenizedGenerateReqInput", FakeRequest):
                assert (
                    ingress_scheduler._forward_with_waiting_queue_ingress(batch)
                    == "forward-result"
                )
            assert times["start"] < times["lookup"] < times["end"], times
            assert len(ingress_scheduler._forward_deferred_reqs) == 1
        finally:
            ingress_scheduler._forward_launch_executor.shutdown(wait=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


class TestSchedulerBackgroundPrefetchCollectives(CustomTestCase):
    def test_distinct_gloo_group_and_variable_request_counts(self):
        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("Gloo process groups are unavailable")

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]

        world_size = int(os.environ.get("SGLANG_BG_PREFETCH_TEST_RANKS", "2"))
        mp.spawn(
            _distributed_round_worker,
            args=(world_size, port),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
