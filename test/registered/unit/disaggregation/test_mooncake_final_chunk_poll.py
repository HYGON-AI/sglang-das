import threading
import time
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.base import KVPoll  # noqa: E402
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager  # noqa: E402
from sglang.srt.disaggregation.prefill import (  # noqa: E402
    SchedulerDisaggregationPrefillMixin,
)
from sglang.srt.environ import envs  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_manager(request_status, outstanding=None):
    manager = object.__new__(MooncakeKVManager)
    manager.request_status = request_status
    manager._staging_outstanding = defaultdict(int, outstanding or {})
    manager._transfer_completion_condition = threading.Condition()
    return manager


class TestMooncakeWaitForTransferRooms(CustomTestCase):
    def test_wakes_on_terminal_status(self):
        manager = make_manager({7: KVPoll.WaitingForInput, 8: KVPoll.WaitingForInput})

        def complete_rooms():
            time.sleep(0.01)
            manager.update_status(7, KVPoll.Success)
            manager.update_status(8, KVPoll.Failed)

        thread = threading.Thread(target=complete_rooms)
        thread.start()
        try:
            self.assertTrue(manager.wait_for_transfer_rooms({7, 8}, 1.0))
        finally:
            thread.join()

    def test_is_bounded(self):
        manager = make_manager({7: KVPoll.WaitingForInput})
        start_time = time.perf_counter()
        self.assertFalse(manager.wait_for_transfer_rooms({7}, 0.01))
        self.assertLess(time.perf_counter() - start_time, 0.1)

    def test_treats_cleared_room_as_terminal(self):
        manager = make_manager({})
        self.assertTrue(manager.wait_for_transfer_rooms({7}, 0.01))

    def test_disabled_without_rooms_or_timeout(self):
        manager = make_manager({7: KVPoll.WaitingForInput})
        self.assertFalse(manager.wait_for_transfer_rooms(set(), 1.0))
        self.assertFalse(manager.wait_for_transfer_rooms({7}, 0))

    def test_success_waits_for_outstanding_chunk(self):
        # The worker sets Success (and notifies) before it finishes the chunk
        # and decrements the outstanding count; sender.poll() still reports
        # Transferring in that window, so the wait must not return yet.
        manager = make_manager({7: KVPoll.WaitingForInput}, {7: 1})
        delay_s = 0.05

        def finish_chunk():
            time.sleep(0.01)
            manager.update_status(7, KVPoll.Success)
            time.sleep(delay_s)
            manager._staging_outstanding[7] -= 1
            manager._staging_outstanding.pop(7, None)
            manager._notify_transfer_waiters()

        thread = threading.Thread(target=finish_chunk)
        start_time = time.perf_counter()
        thread.start()
        try:
            self.assertTrue(manager.wait_for_transfer_rooms({7}, 2.0))
            # Checked before join(), which would hide an early return.
            elapsed = time.perf_counter() - start_time
            self.assertNotIn(7, manager._staging_outstanding)
        finally:
            thread.join()
        self.assertGreaterEqual(elapsed, delay_s)
        self.assertLess(elapsed, 1.0)

    def test_success_with_outstanding_chunk_times_out(self):
        manager = make_manager({7: KVPoll.Success}, {7: 1})
        self.assertFalse(manager.wait_for_transfer_rooms({7}, 0.01))

    def test_failed_does_not_wait_for_outstanding_chunk(self):
        manager = make_manager({7: KVPoll.Failed}, {7: 1})
        self.assertTrue(manager.wait_for_transfer_rooms({7}, 0.01))


class TestSupplementalPollFinalChunks(CustomTestCase):
    @staticmethod
    def make_scheduler(inflight_rooms, final_rooms, wait_result=True):
        kv_manager = SimpleNamespace(
            wait_for_transfer_rooms=Mock(return_value=wait_result)
        )
        scheduler = SimpleNamespace(
            _disagg_final_chunk_rooms=set(final_rooms),
            disagg_prefill_inflight_queue=[
                SimpleNamespace(bootstrap_room=room) for room in inflight_rooms
            ],
            disagg_prefill_bootstrap_queue=SimpleNamespace(kv_manager=kv_manager),
            process_disagg_prefill_inflight_queue=Mock(return_value=["done"]),
        )
        scheduler._record_final_poll_wait = (
            lambda *args: SchedulerDisaggregationPrefillMixin._record_final_poll_wait(
                scheduler, *args
            )
        )
        return scheduler, kv_manager

    def call(self, scheduler):
        return SchedulerDisaggregationPrefillMixin.maybe_supplemental_poll_final_chunks(
            scheduler
        )

    def test_disabled_by_default(self):
        scheduler, kv_manager = self.make_scheduler([1], [1])
        self.assertEqual(self.call(scheduler), [])
        kv_manager.wait_for_transfer_rooms.assert_not_called()
        scheduler.process_disagg_prefill_inflight_queue.assert_not_called()
        self.assertEqual(scheduler._disagg_final_chunk_rooms, set())

    def test_waits_only_for_final_rooms_still_inflight(self):
        scheduler, kv_manager = self.make_scheduler([1, 2, 3], [2, 3, 9])
        with envs.SGLANG_MOONCAKE_FINAL_POLL_TIMEOUT_MS.override(400):
            self.assertEqual(self.call(scheduler), ["done"])
        kv_manager.wait_for_transfer_rooms.assert_called_once_with({2, 3}, 0.4)
        scheduler.process_disagg_prefill_inflight_queue.assert_called_once_with()
        self.assertEqual(scheduler._disagg_final_chunk_rooms, set())
        self.assertEqual(scheduler._final_poll_wait_stats["rooms"], 2)
        self.assertEqual(scheduler._final_poll_wait_stats["done"], 1)
        self.assertEqual(scheduler._final_poll_wait_stats["timeouts"], 0)

    def test_extra_poll_does_not_depend_on_local_wait_result(self):
        # The extra poll is a collective: ranks whose local wait finished and
        # ranks that timed out must both run it exactly once.
        for wait_result in (True, False):
            with self.subTest(wait_result=wait_result):
                scheduler, _ = self.make_scheduler([1], [1], wait_result)
                with envs.SGLANG_MOONCAKE_FINAL_POLL_TIMEOUT_MS.override(400):
                    self.call(scheduler)
                scheduler.process_disagg_prefill_inflight_queue.assert_called_once_with()

    def test_skips_when_final_rooms_already_done(self):
        scheduler, kv_manager = self.make_scheduler([1], [2])
        with envs.SGLANG_MOONCAKE_FINAL_POLL_TIMEOUT_MS.override(400):
            self.assertEqual(self.call(scheduler), [])
        kv_manager.wait_for_transfer_rooms.assert_not_called()
        scheduler.process_disagg_prefill_inflight_queue.assert_not_called()

    def test_timeout_is_counted_and_summary_logged(self):
        scheduler, _ = self.make_scheduler([1], [1], wait_result=False)
        scheduler._final_poll_wait_stats = {"start": time.monotonic() - 61}
        with envs.SGLANG_MOONCAKE_FINAL_POLL_TIMEOUT_MS.override(400):
            with self.assertLogs("sglang.srt.disaggregation.prefill", "INFO") as logs:
                self.call(scheduler)
        self.assertIn("rounds=1 rooms=1 done=1", logs.output[-1])
        self.assertIn("local_wait_ms", logs.output[-1])
        self.assertIn("extra_poll_ms", logs.output[-1])
        self.assertIn("timeouts=1", logs.output[-1])
        self.assertEqual(set(scheduler._final_poll_wait_stats), {"start"})

    def test_send_kv_chunk_records_rooms_only_when_enabled_without_pp(self):
        req = SimpleNamespace(bootstrap_room=5)
        send = SchedulerDisaggregationPrefillMixin.send_kv_chunk
        cases = [
            (0, 1, True, None),
            (400, 2, True, None),
            (400, 1, False, None),
            (400, 1, True, {5}),
        ]
        for timeout_ms, pp_size, last_chunk, expected in cases:
            with self.subTest(timeout=timeout_ms, pp=pp_size, last=last_chunk):
                # The empty allocator stops send_kv_chunk right after recording.
                scheduler = SimpleNamespace(
                    ps=SimpleNamespace(pp_size=pp_size),
                    token_to_kv_pool_allocator=SimpleNamespace(),
                )
                with envs.SGLANG_MOONCAKE_FINAL_POLL_TIMEOUT_MS.override(timeout_ms):
                    with self.assertRaises(AttributeError):
                        send(scheduler, req, last_chunk=last_chunk)
                self.assertEqual(
                    getattr(scheduler, "_disagg_final_chunk_rooms", None), expected
                )


if __name__ == "__main__":
    unittest.main()
