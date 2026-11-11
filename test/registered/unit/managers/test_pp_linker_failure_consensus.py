"""Exercise linker-failure propagation through the regular PP output ring."""

import datetime
import pickle
import queue
import socket
import threading
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers import scheduler_pp_mixin as pp  # noqa: E402
from sglang.srt.managers.schedule_batch import FINISH_ABORT  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.mem_cache.storage.mooncake_store import (  # noqa: E402
    mooncake_direct_linker,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (  # noqa: E402
    LayerWiseLoadCounter,
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (  # noqa: E402
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402

register_cpu_ci(est_time=180, suite="base-a-test-cpu")


class _StopLoop(Exception):
    pass


class _Req:
    def __init__(self, rid):
        self.rid = rid
        self.to_finish = None
        self.finished_reason = None
        self.last_node = None
        self.external_kv_cache_released = False
        self.time_stats = SimpleNamespace(trace_ctx=SimpleNamespace(abort=Mock()))

    def finished(self):
        return self.finished_reason is not None


class _SendWork:
    def __init__(self, work, *buffers):
        self.work = work
        self.buffers = buffers


class _GlooPPGroup:
    """Use production typed-message routing over a small real Gloo transport."""

    def __init__(self, rank, pp_size):
        self.rank = rank
        self.pp_size = pp_size
        self.is_first_rank = rank == 0
        self.is_last_rank = rank == pp_size - 1
        self.cpu_group = dist.group.WORLD

    def send_tensor_dict(self, tensor_dict, **_kwargs):
        payload = torch.tensor(list(pickle.dumps(tensor_dict)), dtype=torch.uint8)
        size = torch.tensor([payload.numel()], dtype=torch.int64)
        destination = (self.rank + 1) % self.pp_size
        return [
            _SendWork(dist.isend(size, dst=destination), size),
            _SendWork(dist.isend(payload, dst=destination), payload),
        ]

    def recv_tensor_dict(self, **_kwargs):
        source = (self.rank - 1) % self.pp_size
        size = torch.empty(1, dtype=torch.int64)
        dist.recv(size, src=source)
        payload = torch.empty(int(size.item()), dtype=torch.uint8)
        dist.recv(payload, src=source)
        return pickle.loads(payload.numpy().tobytes())


class _Event:
    def __init__(self, stage, is_launch_event):
        self.stage = stage
        self.is_launch_event = is_launch_event

    def record(self, *_args):
        pass

    def synchronize(self):
        if self.is_launch_event:
            self.stage.completed_launch_tick = self.stage.tick


class _Stage(pp.SchedulerPPMixin):
    def __init__(self, rank, pp_size, failing_rank):
        self.rank = rank
        self.pp_size = pp_size
        self.failing_rank = failing_rank
        self.pure_chunk = False
        self.tick = -1
        self.completed_launch_tick = None
        self.failure_tick = None
        self._mark_next_event_as_launch = False
        self.ps = SimpleNamespace(
            pp_size=pp_size,
            pp_rank=rank,
            tp_size=1,
            attn_tp_rank=0,
            attn_cp_rank=0,
            attn_dp_rank=0,
            attn_cp_size=1,
            attn_tp_size=1,
        )
        self.pp_group = _GlooPPGroup(rank, pp_size)
        self.attn_tp_group = None
        self.disaggregation_mode = DisaggregationMode.NULL
        self.server_args = SimpleNamespace(enable_unified_cache_external_linker=True)
        self.responses = []
        self.failures = []
        self.failure_drain_requested = threading.Event()
        self.failure_jobs = queue.Queue()
        self.failure_workers = []
        self.requests = {
            rid: _Req(rid) for rid in ("warmup", "failed-load", "next-request")
        }
        self.tree_cache = SimpleNamespace(
            drain_linker_loads=self.drain_loads,
            has_outstanding_failed_linker_chains=lambda: False,
        )
        self.request_receiver = SimpleNamespace(recv_requests=self.recv_requests)
        self._deferred_linker_rids = set()
        self._external_kv_cleanup_rids = set()
        self._external_kv_chunked_abort_reqs = {}
        self.chunked_req = None
        self.waiting_queue = []
        self.result_queue = []
        self.device = torch.device("cpu")
        self.device_module = SimpleNamespace(
            Event=self.make_event,
            current_stream=lambda: SimpleNamespace(wait_event=lambda _event: None),
        )
        self.forward_stream_ctx = nullcontext()
        self.forward_stream = SimpleNamespace(wait_stream=lambda _stream: None)
        self.schedule_stream = object()
        self.copy_stream_ctx = nullcontext()
        self.copy_stream = SimpleNamespace(wait_stream=lambda _stream: None)

    def make_event(self):
        is_launch_event = self._mark_next_event_as_launch
        self._mark_next_event_as_launch = False
        return _Event(self, is_launch_event)

    def recv_requests(self):
        self.tick += 1
        return []

    def process_input_requests(self, _reqs):
        pass

    def _pp_send_pyobj_to_next_stage(self, *_args, **_kwargs):
        return []

    def get_next_batch_to_run(self, **_kwargs):
        request_ids = ("warmup", "failed-load", "next-request")
        batch = None
        if self.tick < len(request_ids):
            request = self.requests[request_ids[self.tick]]
            batch = SimpleNamespace(
                reqs=[request],
                forward_mode=ForwardMode.EXTEND,
                spec_algorithm=SimpleNamespace(is_dspark=lambda: False),
                return_logprob=False,
                contains_last_prefill_chunk=True,
                req_pool_indices=torch.tensor([0]),
                input_ids=torch.tensor([1]),
            )
        return SimpleNamespace(running_batch=None, batch_to_run=batch)

    def run_batch(self, batch, _proxy_tensors):
        if _proxy_tensors is not None:
            assert "__linker_failed_rids__" not in _proxy_tensors.tensors
        request = batch.reqs[0]
        if (
            request.rid == "failed-load"
            and self.failing_rank is not None
            and self.rank == self.failing_rank
        ):
            self.failure_tick = self.tick
            self.failure_drain_requested.clear()
            self.failure_jobs.put(request.rid)

            def publish_after_drain_request():
                rid = self.failure_jobs.get()
                try:
                    # The load has signalled the model-layer failure, but its
                    # completion record is delayed until the PP collector asks
                    # to wait for worker publication.
                    self.failure_drain_requested.wait()
                    self.failures.append(rid)
                finally:
                    self.failure_jobs.task_done()

            worker = threading.Thread(target=publish_after_drain_request, daemon=True)
            self.failure_workers.append(worker)
            worker.start()
        self._mark_next_event_as_launch = True
        return SimpleNamespace(
            can_run_cuda_graph=False,
            pp_hidden_states_proxy_tensors=pp.PPProxyTensors(
                {"hidden_states": torch.tensor([self.tick], dtype=torch.int64)}
            ),
            logits_output=None,
            next_token_ids=torch.tensor([self.tick], dtype=torch.int64),
        )

    def drain_loads(self, *, wait_for_completion=False):
        if wait_for_completion:
            if self.failure_tick is not None and (
                self.completed_launch_tick != self.failure_tick
            ):
                raise AssertionError("linker failure drained before launch event sync")
            self.failure_drain_requested.set()
            self.failure_jobs.join()
            self.failure_tick = None
        failed, self.failures = self.failures, []
        return failed

    def release_failure_workers(self):
        self.failure_drain_requested.set()
        self.failure_jobs.join()
        for worker in self.failure_workers:
            worker.join(timeout=2)
            if worker.is_alive():
                raise AssertionError("delayed linker failure worker did not exit")

    def _iter_external_linker_candidates(self, batch):
        return Scheduler._iter_external_linker_candidates(self, batch)

    def _pp_prep_batch_result(self, _batch, _metadata, output_tensors):
        assert "next_token_ids" in output_tensors.tensors
        return SimpleNamespace()

    def abort_external_kv_request(self, request, **_kwargs):
        request.to_finish = FINISH_ABORT("external KV load failed", status_code=500)
        self._external_kv_cleanup_rids.add(request.rid)
        return True

    def _release_aborted_request(self, _rid):
        pass

    def process_batch_result(self, batch, _result, *, linker_failed_rids=None):
        if self.pure_chunk:
            assert _result.skipped_output_comm
        Scheduler._mark_failed_linker_loads(self, batch, failed_rids=linker_failed_rids)
        for request in batch.reqs:
            self.responses.append((request.rid, 500 if request.to_finish else 200))

    def on_idle(self):
        if (
            len(self.responses) >= 3
            and self.pp_outputs is None
            and not self.last_rank_comm_queue
        ):
            raise _StopLoop


def _pp_failure_worker(rank, port, pp_size):
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=pp_size,
        timeout=datetime.timedelta(seconds=20),
    )
    try:
        for pure_chunk in (False, True):
            for failing_rank in (None, 0, pp_size - 1):
                for depth in (0, 1):
                    stage = _Stage(rank, pp_size, failing_rank)
                    stage.pure_chunk = pure_chunk
                    with (
                        patch.object(
                            pp,
                            "get_parallel",
                            return_value=SimpleNamespace(pp_async_batch_depth=depth),
                        ),
                        patch.object(pp, "set_time_batch", lambda *_a, **_k: None),
                        patch.object(
                            pp,
                            "get_pd_hidden_capture_layer_ids",
                            return_value=[],
                        ),
                        patch.object(
                            pp.envs,
                            "SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM",
                            SimpleNamespace(get=lambda: pure_chunk),
                        ),
                    ):
                        if pure_chunk:
                            # Exercise the control-only output-ring path while
                            # the model output remains skipped.
                            stage.get_next_batch_to_run = (
                                lambda **_kwargs: _pure_chunk_plan(stage)
                            )
                        try:
                            try:
                                stage.event_loop_pp()
                            except _StopLoop:
                                pass
                        finally:
                            stage.release_failure_workers()
                    stage._pp_commit_comm_work(stage.send_proxy_work)
                    stage._pp_commit_comm_work(stage.send_output_work)
                    reports = [None] * pp_size
                    dist.all_gather_object(reports, stage.responses)
                    expected = [("warmup", 200)]
                    expected.append(
                        ("failed-load", 500 if failing_rank is not None else 200)
                    )
                    expected.append(("next-request", 200))
                    assert reports == [expected] * pp_size, reports
                    dist.barrier()
    finally:
        dist.destroy_process_group()


def _pure_chunk_plan(stage):
    request_ids = ("warmup", "failed-load", "next-request")
    batch = None
    if stage.tick < len(request_ids):
        request = stage.requests[request_ids[stage.tick]]
        batch = SimpleNamespace(
            reqs=[request],
            forward_mode=ForwardMode.EXTEND,
            spec_algorithm=SimpleNamespace(is_dspark=lambda: False),
            return_logprob=False,
            contains_last_prefill_chunk=False,
            req_pool_indices=torch.tensor([0]),
            input_ids=torch.tensor([1]),
        )
    return SimpleNamespace(running_batch=None, batch_to_run=batch)


class _ObservedJoinQueue(queue.Queue):
    """Expose the moment a real backend worker queue is joined by the test."""

    def __init__(self, on_join=None):
        super().__init__()
        self.join_entered = threading.Event()
        self.on_join = on_join

    def join(self):
        self.join_entered.set()
        if self.on_join is not None:
            self.on_join()
        super().join()


class _CacheDrainHarness:
    """Run the production UnifiedRadixCache drain against one local rank."""

    def __init__(self, linker):
        self.linker = linker
        self._failed_linker_rids = []

    def _all_reduce_attn_groups(self, _tensor, _op):
        pass

    def _collect_failed_linker_loads(self, finish_count):
        UnifiedRadixCache._collect_failed_linker_loads(self, finish_count)

    def drain_linker_loads(self, *, wait_for_completion=False):
        if wait_for_completion:
            return UnifiedRadixCache.drain_linker_loads(self, wait_for_completion=True)
        return UnifiedRadixCache.drain_linker_loads(self)

    def has_outstanding_failed_linker_chains(self):
        return False


class TestPPLinkerFailureConsensus(unittest.TestCase):
    def test_pp_collect_waits_for_backend_failure_publication(self):
        cleanup_release = threading.Event()
        failure_signalled = threading.Event()

        backend = object.__new__(MooncakeDirectLinker)
        backend.pending_loads = {"failed-load": []}
        backend.pending_load_tokens = {}
        backend.host_prefetch_enabled = False
        backend.gc_frozen = True
        backend.stats = {"load": 0}
        backend.load_queue = _ObservedJoinQueue(on_join=cleanup_release.set)
        backend.completed_loads = queue.Queue()
        backend.layer_done_counter = LayerWiseLoadCounter(1)

        def fail_then_wait_for_cleanup(counter_index, _request_transfers):
            backend.layer_done_counter.fail(
                counter_index, RuntimeError("injected linker load failure")
            )
            failure_signalled.set()
            if not cleanup_release.wait(timeout=5):
                raise TimeoutError("test did not release the delayed load cleanup")
            return False

        backend.load_layer_wise = fail_then_wait_for_cleanup
        ready_event = SimpleNamespace(record=lambda: None, synchronize=lambda: None)
        with patch.object(
            mooncake_direct_linker,
            "device_module",
            SimpleNamespace(Event=lambda: ready_event),
        ):
            backend.start_layer_wise_loading()

        worker = threading.Thread(target=backend.load_thread_func, daemon=True)
        worker.start()
        try:
            self.assertTrue(
                failure_signalled.wait(timeout=5),
                "backend worker did not signal the layer failure",
            )
            self.assertEqual(backend.num_completed_loads(), 0)

            wrapper_cache = SimpleNamespace(
                tree_core=SimpleNamespace(
                    enable_external_cache_linker=False,
                    detach_external_load_chain=lambda _node, _anchor: [],
                ),
                write_through_threshold=0,
                dec_lock_ref=Mock(),
            )
            linker = UnifiedCacheLinkerWrapper(wrapper_cache, backend)
            linker.pending_loads["failed-load"] = (1, object(), 0)
            cache = _CacheDrainHarness(linker)

            # The public drain remains nonblocking unless PP explicitly opts in.
            self.assertEqual(cache.drain_linker_loads(), [])
            self.assertFalse(backend.load_queue.join_entered.is_set())

            stage = _Stage(rank=0, pp_size=2, failing_rank=None)
            stage.tree_cache = cache
            batch = SimpleNamespace(reqs=[stage.requests["failed-load"]])
            failed_rids = stage._pp_collect_linker_load_failures(batch)
            self.assertEqual(failed_rids, {"failed-load"})
            self.assertTrue(backend.load_queue.join_entered.is_set())
        finally:
            # A broken PP wait must not leave the simulated backend thread held.
            cleanup_release.set()
            backend.load_queue.join()
            backend.load_queue.put(None)
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive(), "backend worker did not exit")

    def test_failure_consensus_uses_proxy_and_output_rings(self):
        for pp_size in (2, 3):
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            mp.spawn(
                _pp_failure_worker,
                args=(port, pp_size),
                nprocs=pp_size,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
