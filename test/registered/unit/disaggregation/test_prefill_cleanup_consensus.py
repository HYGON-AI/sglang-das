"""Multi-rank regressions for prefill inflight cleanup consensus."""

import datetime
import socket
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation import prefill  # noqa: E402
from sglang.srt.disaggregation.base import KVPoll  # noqa: E402
from sglang.srt.managers.schedule_batch import FINISH_ABORT  # noqa: E402

register_cpu_ci(est_time=90, suite="base-a-test-cpu")


class _Request:
    def __init__(self, **attributes):
        self.__dict__.update(attributes)


class _Sender:
    def __init__(self, state):
        self.state = state

    def poll(self):
        return self.state["poll"]

    def failure_exception(self):
        return None

    def clear(self):
        self.state["clear_count"] += 1

    def is_transfer_drained(self):
        return self.state["drained"]

    def abort(self):
        self.state["abort_count"] += 1


def _make_scheduler(rank, tp_group, cp_group, mode, state):
    rid = f"request-{mode}"
    sender = _Sender(state)
    req = _Request(
        rid=rid,
        bootstrap_room=7,
        to_finish=FINISH_ABORT("transfer failed"),
        finished_reason=None,
        skip_radix_cache_insert=False,
        return_logprob=False,
        req_pool_idx=1,
        kv=object(),
        mamba_pool_idx=None,
        inflight_middle_chunks=0,
        pending_bootstrap=False,
        disagg_kv_sender=sender,
        time_stats=SimpleNamespace(
            set_completion_time=lambda: None,
            trace_ctx=SimpleNamespace(
                abort=lambda **_kwargs: state.__setitem__(
                    "trace_count", state["trace_count"] + 1
                )
            ),
        ),
        prefill_transfer_failure_reported=False,
        external_kv_abort_response_via_chunked=False,
        external_kv_response_sent=False,
        external_kv_metadata_released=False,
        external_kv_cleanup_steps={},
    )
    for stage in (
        "abort_requested",
        "pending_chunk_cleared",
        "linker_released",
        "sender_abort_requested",
        "metadata_released",
        "cache_released",
        "finish_state_applied",
        "response_sent",
        "cleanup_done",
    ):
        setattr(req, f"external_kv_{stage}", False)

    def apply_finish():
        req.finished_reason = req.to_finish
        req.to_finish = None

    req.update_finish_state = apply_finish
    scheduler = SimpleNamespace(
        disagg_prefill_inflight_queue=[req],
        attn_cp_cpu_group=cp_group,
        attn_tp_cpu_group=tp_group,
        _external_kv_abort_rids={rid} if mode.startswith("external") else set(),
        _external_kv_cleanup_rids=set(),
        result_queue=[],
        req_to_metadata_buffer_idx_allocator=object(),
        disagg_metadata_buffers=SimpleNamespace(pd_hidden_pool=None),
        tree_cache=object(),
        ps=SimpleNamespace(tp_rank=rank),
        metrics_reporter=SimpleNamespace(enable_metrics=True),
        metrics_collector=SimpleNamespace(
            increment_transfer_failed_reqs=lambda: state.__setitem__(
                "failure_count", state["failure_count"] + 1
            )
        ),
        output_streamer=SimpleNamespace(
            stream_output=lambda *_args, **_kwargs: state.__setitem__(
                "response_count", state["response_count"] + 1
            )
        ),
        clear_pending_chunk_send=lambda _req: None,
        _release_aborted_request=lambda _rid: None,
    )

    def forward_drained(_req):
        return True

    def sender_terminal(target):
        return (
            prefill.SchedulerDisaggregationPrefillMixin.prefill_abort_sender_terminal(
                scheduler, target
            )
        )

    def abort_external(target, **kwargs):
        return prefill.SchedulerDisaggregationPrefillMixin.abort_external_kv_request(
            scheduler, target, **kwargs
        )

    def handle_failure(target, *, sender_terminal):
        return prefill.SchedulerDisaggregationPrefillMixin.handle_inflight_transfer_failure(
            scheduler,
            target,
            sender_terminal=sender_terminal,
        )

    scheduler.prefill_abort_forward_drained = forward_drained
    scheduler.prefill_abort_sender_terminal = sender_terminal
    scheduler.abort_external_kv_request = abort_external
    scheduler.handle_inflight_transfer_failure = handle_failure
    return scheduler, req


def _gloo_prefill_cleanup_worker(rank, port):
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=20),
    )
    try:
        # Every process creates the groups in the same order. TP groups are
        # [0, 1]/[2, 3] and CP groups are [0, 2]/[1, 3].
        tp_groups = [
            dist.new_group(ranks=[0, 1], backend="gloo"),
            dist.new_group(ranks=[2, 3], backend="gloo"),
        ]
        cp_groups = [
            dist.new_group(ranks=[0, 2], backend="gloo"),
            dist.new_group(ranks=[1, 3], backend="gloo"),
        ]
        tp_group = tp_groups[rank // 2]
        cp_group = cp_groups[rank % 2]

        for mode in ("failed", "external", "pp", "external_success"):
            # Rank 2 is locally undrained. The other ranks can finish local
            # cleanup on tick one, but no request may leave the shared queue
            # until rank 2 drains. This pattern exercises both TP and CP MINs.
            state = {
                "drained": rank != 2,
                "released": False,
                "release_count": 0,
                "metadata_release_count": 0,
                "response_count": 0,
                "abort_count": 0,
                "clear_count": 0,
                "failure_count": 0,
                "trace_count": 0,
                "poll": KVPoll.Success if mode == "external_success" else KVPoll.Failed,
            }
            scheduler, req = _make_scheduler(rank, tp_group, cp_group, mode, state)

            def release_cache(*_args, **_kwargs):
                state["released"] = True
                state["release_count"] += 1

            def release_metadata(*_args, **_kwargs):
                state["metadata_release_count"] += 1

            with (
                patch.object(prefill, "release_kv_cache", side_effect=release_cache),
                patch.object(
                    prefill,
                    "maybe_release_metadata_buffer",
                    side_effect=release_metadata,
                ),
            ):
                status = ([], [req.rid]) if mode == "pp" else None
                done = prefill.SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue(
                    scheduler, transfer_status=status
                )
                assert done == []
                assert scheduler.disagg_prefill_inflight_queue == [req]
                assert state["release_count"] == int(rank != 2)
                assert state["metadata_release_count"] == int(rank != 2)
                assert state["response_count"] == int(rank != 2)
                assert req.external_kv_cleanup_done is (rank != 2)

                # A second scheduler tick must keep the same request reachable
                # without repeating cleanup while one source is still active.
                done = prefill.SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue(
                    scheduler, transfer_status=status
                )
                assert done == []
                assert scheduler.disagg_prefill_inflight_queue == [req]
                assert state["release_count"] == int(rank != 2)
                assert state["metadata_release_count"] == int(rank != 2)
                assert state["response_count"] == int(rank != 2)

                # Once the remote source drains, local cleanup completes once
                # and all TP/CP peers may remove the request together.
                state["drained"] = True
                done = prefill.SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue(
                    scheduler, transfer_status=status
                )
                assert done == [req]
                assert scheduler.disagg_prefill_inflight_queue == []
                assert state["release_count"] == 1
                assert state["metadata_release_count"] == 1
                assert state["response_count"] == 1
                assert state["abort_count"] == 1
                assert state["clear_count"] == 0
                assert state["failure_count"] == int(not mode.startswith("external"))
                assert state["trace_count"] == int(not mode.startswith("external"))
                assert req.external_kv_cleanup_done
    finally:
        dist.destroy_process_group()


class TestPrefillCleanupConsensus(unittest.TestCase):
    def test_tp2_cp2_cleanup_readiness_is_reduced_before_queue_removal(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        mp.spawn(
            _gloo_prefill_cleanup_worker,
            args=(port,),
            nprocs=4,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
