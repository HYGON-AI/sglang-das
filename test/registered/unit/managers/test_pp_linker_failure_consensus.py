"""Exercise the regular PP loop with real CPU proxy traffic and load failures."""

import datetime
import socket
import unittest
from collections import deque
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
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=90, suite="base-a-test-cpu")


class _StopLoop(Exception):
    pass


class _Req:
    def __init__(self, rid):
        self.rid = rid
        self.to_finish = None
        self.finished_reason = None
        self.last_node = None
        self.time_stats = SimpleNamespace(trace_ctx=SimpleNamespace(abort=Mock()))

    def finished(self):
        return self.finished_reason is not None


class _Stage(pp.SchedulerPPMixin):
    def __init__(self, rank, failing_rank):
        self.rank = rank
        self.tick = -1
        self.ps = SimpleNamespace(pp_size=2, pp_rank=rank)
        self.pp_group = SimpleNamespace(
            is_first_rank=rank == 0,
            is_last_rank=rank == 1,
            cpu_group=dist.group.WORLD,
        )
        self.disaggregation_mode = DisaggregationMode.NULL
        self.server_args = SimpleNamespace(enable_unified_cache_external_linker=True)
        self._deferred_linker_rids = set()
        self._external_kv_cleanup_rids = set()
        self._external_kv_chunked_abort_reqs = {}
        self.chunked_req = None
        self.waiting_queue = []
        self.result_queue = []
        self.failed_req = _Req("failed-load")
        self.healthy_req = _Req("next-request")
        self.responses = []
        self.proxy_receives = 0
        self.failures = [self.failed_req.rid] if rank == failing_rank else []
        self.tree_cache = SimpleNamespace(
            drain_linker_loads=self.drain_loads,
            has_outstanding_failed_linker_chains=lambda: False,
        )
        self.request_receiver = SimpleNamespace(recv_requests=self.recv_requests)
        self.device_module = SimpleNamespace(
            current_stream=lambda: SimpleNamespace(wait_event=lambda _event: None)
        )

    def init_pp_loop_state(self):
        self.pp_loop_size = 2
        self.running_mbs = [None, None]
        self.last_mbs = [None, None]
        self.mbs = [None, None]
        self.mb_metadata = [None, None]
        self.last_rank_comm_queue = deque()
        self.send_req_work = []
        self.send_proxy_work = []
        self.pp_outputs = None

    def recv_requests(self):
        self.tick += 1
        if self.tick == 3:
            raise _StopLoop
        return []

    def process_input_requests(self, _reqs):
        pass

    def _pp_send_pyobj_to_next_stage(self, *_args, **_kwargs):
        return []

    def get_next_batch_to_run(self, **_kwargs):
        request = [self.failed_req, self.healthy_req, None][self.tick]
        batch = None
        if request is not None:
            batch = SimpleNamespace(
                reqs=[request],
                forward_mode=SimpleNamespace(is_extend=lambda: True),
            )
        return SimpleNamespace(running_batch=None, batch_to_run=batch)

    def _pp_recv_proxy_tensors(self):
        if self.rank == 1:
            value = torch.empty(1, dtype=torch.int64)
            dist.recv(value, src=0)
            self.proxy_receives += 1
        return None

    def _pp_launch_batch(self, *_args):
        return SimpleNamespace(), SimpleNamespace()

    def _pp_prepare_proxy_tensor_dict_for_send(self, _result):
        return {"hidden_states": torch.tensor([self.tick])}

    def _pp_send_dict_to_next_stage(self, payload, **_kwargs):
        value = payload["hidden_states"]
        # Retain the send buffer until the production loop commits this work.
        return [SimpleNamespace(work=dist.isend(value, dst=1), value=value)]

    def _pp_commit_send_output_work_and_preprocess_output_tensors(self, *_args):
        return None, SimpleNamespace(), SimpleNamespace(synchronize=lambda: None)

    def drain_loads(self):
        failed, self.failures = self.failures, []
        return failed

    def _iter_external_linker_candidates(self, batch):
        return Scheduler._iter_external_linker_candidates(self, batch)

    def abort_external_kv_request(self, req, **_kwargs):
        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        req.to_finish = FINISH_ABORT("external KV load failed", status_code=500)
        self._external_kv_cleanup_rids.add(req.rid)
        return True

    def _release_aborted_request(self, _rid):
        pass

    def process_batch_result(self, batch, _result, *, linker_failed_rids=None):
        if linker_failed_rids is None:
            Scheduler._mark_failed_linker_loads(self, batch)
        else:
            Scheduler._mark_failed_linker_loads(
                self, batch, failed_rids=linker_failed_rids
            )
        for req in batch.reqs:
            self.responses.append((req.rid, 500 if req.to_finish else 200))

    def on_idle(self):
        pass


def _pp_failure_worker(rank, port):
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=20),
    )
    try:
        for failing_rank in (0, 1):
            for depth in (0, 1):
                stage = _Stage(rank, failing_rank)
                with patch.object(
                    pp,
                    "get_parallel",
                    return_value=SimpleNamespace(pp_async_batch_depth=depth),
                ):
                    try:
                        stage.event_loop_pp()
                    except _StopLoop:
                        pass
                stage._pp_commit_comm_work(stage.send_proxy_work)
                reports = [None, None]
                dist.all_gather_object(reports, stage.responses)
                expected = [("failed-load", 500), ("next-request", 200)]
                assert reports == [expected, expected], reports
                assert stage.proxy_receives == (2 if rank else 0)
                dist.barrier()
    finally:
        dist.destroy_process_group()


class TestPPLinkerFailureConsensus(unittest.TestCase):
    def test_pp2_stage_failure_aborts_consistently_and_next_request_progresses(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        mp.spawn(_pp_failure_worker, args=(port,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
