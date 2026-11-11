"""Unit tests for ReqTimeStats IPC serialization.

ReqTimeStatsBase.__setstate__ rebases perf_counter fields onto the receiving
process's clock anchor. Rebasing a field that was never stamped (0.0) turns
the sentinel into a tiny epsilon (sender_diff - receiver_diff), which defeats
== 0.0 / > 0.0 "was this stamped?" checks downstream. Concretely, a PD decode
server never stamps prefill_finished_time locally; if the sentinel arrives at
the tokenizer as an epsilon, first-token bookkeeping mistakes it for a real
stamp and the TTFT / inter-token-latency histograms record ~node-uptime-sized
garbage samples.
"""

import pickle
import unittest
from unittest import mock

import sglang.srt.observability.req_time_stats as rts
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSetstatePreservesUnsetTimeSentinels(CustomTestCase):
    def test_two_hop_round_trip(self):
        src = rts.SchedulerReqTimeStats()
        src.enable_metrics = True
        src.wait_queue_entry_time = 123.456
        src.prefill_finished_time = 0.0

        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_000.0):
            blob = pickle.dumps(src)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_005.0):
            hop1 = pickle.loads(blob)
            blob2 = pickle.dumps(hop1)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_009.0):
            hop2 = pickle.loads(blob2)

        self.assertEqual(hop2.prefill_finished_time, 0.0)
        self.assertAlmostEqual(hop2.wait_queue_entry_time, 123.456 - 9.0)


class TestWaitingQueueReasonStats(CustomTestCase):
    def test_disabled_keeps_request_timestamps_without_queue_state(self):
        stats = rts.SchedulerReqTimeStats()
        self.assertFalse(stats.track_queue_wait_reasons)
        stats.set_wait_queue_entry_time(ts=10.0)
        stats.set_forward_entry_time(ts=12.0)

        self.assertEqual(stats.wait_queue_entry_time, 10.0)
        self.assertEqual(stats.forward_entry_time, 12.0)
        self.assertIsNone(stats.queue_reason_durations)
        self.assertIsNone(stats.queue_reason_checks)

        stats.has_timing_data = True
        state = stats.__getstate__()
        self.assertNotIn("track_queue_wait_reasons", state)
        self.assertNotIn("queue_wait_reason", state)
        self.assertNotIn("queue_wait_reason_start_time", state)
        self.assertNotIn("queue_reason_durations", state)
        self.assertNotIn("queue_reason_checks", state)

    def test_disabled_direct_reason_methods_do_not_read_clock(self):
        stats = rts.SchedulerReqTimeStats()
        with mock.patch.object(
            rts.time,
            "perf_counter",
            side_effect=AssertionError("unexpected clock"),
        ):
            stats.set_queue_wait_reason("pending")
            stats.finish_queue_wait_reason()

        self.assertIsNone(stats.queue_reason_durations)
        self.assertIsNone(stats.queue_reason_checks)

    def test_enabled_accumulates_serializes_and_resets(self):
        stats = rts.SchedulerReqTimeStats(track_queue_wait_reasons=True)
        stats.set_wait_queue_entry_time(ts=10.0)
        self.assertEqual(stats.queue_reason_checks, {"awaiting_scheduler_check": 1})
        self.assertIsNone(stats.queue_reason_durations)

        stats.set_queue_wait_reason("waiting_for_prefetch", ts=12.0)
        stats.set_forward_entry_time(ts=17.0)
        self.assertEqual(
            stats.queue_reason_durations,
            {"awaiting_scheduler_check": 2.0, "waiting_for_prefetch": 5.0},
        )
        self.assertEqual(
            stats.queue_reason_checks,
            {"awaiting_scheduler_check": 1, "waiting_for_prefetch": 1},
        )
        self.assertEqual(stats.queue_wait_reason, "")
        self.assertEqual(stats.queue_wait_reason_start_time, 0.0)

        stats.has_timing_data = True
        state = stats.__getstate__()
        self.assertTrue(state["track_queue_wait_reasons"])
        self.assertEqual(state["queue_reason_durations"], stats.queue_reason_durations)
        self.assertEqual(state["queue_reason_checks"], stats.queue_reason_checks)
        hop1 = pickle.loads(pickle.dumps(stats))
        hop2 = pickle.loads(pickle.dumps(hop1))
        self.assertTrue(hop1.track_queue_wait_reasons)
        self.assertTrue(hop2.track_queue_wait_reasons)
        self.assertEqual(hop2.queue_reason_durations, stats.queue_reason_durations)
        self.assertEqual(hop2.queue_reason_checks, stats.queue_reason_checks)

        stats.reset_prefill_retry_time()
        self.assertEqual(stats.queue_wait_reason, "")
        self.assertEqual(stats.queue_wait_reason_start_time, 0.0)
        self.assertEqual(stats.queue_reason_durations, {})
        self.assertEqual(stats.queue_reason_checks, {})

        untouched = rts.SchedulerReqTimeStats(track_queue_wait_reasons=True)
        self.assertEqual(untouched.__getstate__(), {})
        untouched.reset_prefill_retry_time()
        self.assertIsNone(untouched.queue_reason_durations)
        self.assertIsNone(untouched.queue_reason_checks)


if __name__ == "__main__":
    unittest.main()
