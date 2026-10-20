"""Exercise linker-failure propagation through the regular PP output ring."""

import datetime
import pickle
import socket
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
            self.failures.append(request.rid)
            self.failure_tick = self.tick
        self._mark_next_event_as_launch = True
        return SimpleNamespace(
            can_run_cuda_graph=False,
            pp_hidden_states_proxy_tensors=pp.PPProxyTensors(
                {"hidden_states": torch.tensor([self.tick], dtype=torch.int64)}
            ),
            logits_output=None,
            next_token_ids=torch.tensor([self.tick], dtype=torch.int64),
        )

    def drain_loads(self):
        if self.failures and self.completed_launch_tick != self.failure_tick:
            raise AssertionError("linker failure drained before launch event sync")
        failed, self.failures = self.failures, []
        return failed

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
                            stage.event_loop_pp()
                        except _StopLoop:
                            pass
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


class TestPPLinkerFailureConsensus(unittest.TestCase):
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
