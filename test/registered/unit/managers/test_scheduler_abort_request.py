"""Regression tests for scheduler request abortion."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSchedulerAbortRequest(CustomTestCase):
    def test_waiting_request_abort_has_no_beam_coordinator_dependency(self):
        req = SimpleNamespace(rid="waiting-request", mamba_pool_idx=None)
        send_output = MagicMock()

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.chunked_req = None
        scheduler.mm_receiver = None
        scheduler.waiting_queue = [req]
        scheduler._release_aborted_request = MagicMock()
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=send_output)
        )
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.tree_cache = MagicMock()
        scheduler.dllm_config = None
        scheduler.grammar_manager = MagicMock()
        scheduler.ps = SimpleNamespace(pp_size=1)
        scheduler.running_batch = SimpleNamespace(reqs=[])
        scheduler.last_batch = None

        Scheduler.abort_request(
            scheduler,
            AbortReq(rid=req.rid),
        )

        self.assertEqual(scheduler.waiting_queue, [])
        scheduler._release_aborted_request.assert_called_once_with(req.rid)
        send_output.assert_called_once()
        sent_abort, sent_req = send_output.call_args.args
        self.assertEqual(sent_abort.rid, req.rid)
        self.assertIs(sent_req, req)
        scheduler.grammar_manager.abort_requests.assert_called_once()


if __name__ == "__main__":
    import unittest

    unittest.main()
