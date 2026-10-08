"""Tests for scheduler requests deferred during a model forward."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSchedulerDeferredRequests(unittest.TestCase):
    def test_deferred_requests_are_replayed_once_in_order_and_cleared(self):
        scheduler = Scheduler.__new__(Scheduler)
        requests = [object(), object()]
        scheduler._forward_deferred_reqs = requests.copy()
        processed = []
        scheduler.process_input_requests = processed.append

        scheduler._process_deferred_reqs()
        scheduler._process_deferred_reqs()

        self.assertEqual(processed, [requests])
        self.assertEqual(scheduler._forward_deferred_reqs, [])

    def test_disagg_prefill_loop_replays_deferred_before_receiving(self):
        class StopPrefillLoop(Exception):
            pass

        scheduler = Scheduler.__new__(Scheduler)
        requests = [object(), object()]
        scheduler._forward_deferred_reqs = requests.copy()
        events = []

        def process_input_requests(deferred_reqs):
            events.append(("process", deferred_reqs.copy()))

        def recv_requests():
            events.append(("receive", None))
            raise StopPrefillLoop

        scheduler.process_input_requests = process_input_requests
        scheduler.request_receiver = SimpleNamespace(recv_requests=recv_requests)

        with self.assertRaises(StopPrefillLoop):
            SchedulerDisaggregationPrefillMixin.event_loop_normal_disagg_prefill(
                scheduler
            )

        self.assertEqual(events, [("process", requests), ("receive", None)])
        self.assertEqual(scheduler._forward_deferred_reqs, [])


if __name__ == "__main__":
    unittest.main()
