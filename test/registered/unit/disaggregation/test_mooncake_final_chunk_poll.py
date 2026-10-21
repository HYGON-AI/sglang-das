import threading
import time
import unittest
from collections import OrderedDict, defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.base import KVPoll  # noqa: E402
from sglang.srt.disaggregation.mooncake.conn import (  # noqa: E402
    MooncakeKVManager,
    MooncakeKVSender,
)
from sglang.srt.disaggregation.prefill import (  # noqa: E402
    SchedulerDisaggregationPrefillMixin,
)
from sglang.srt.environ import envs  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_manager(request_status, outstanding=None):
    manager = object.__new__(MooncakeKVManager)
    manager.request_status = request_status
    manager._staging_outstanding = defaultdict(int, outstanding or {})
    manager._final_poll_enabled = True
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


class TestMooncakeDrainedAbortStatus(CustomTestCase):
    @staticmethod
    def make_manager(status=KVPoll.WaitingForInput, expected=3):
        manager = object.__new__(MooncakeKVManager)
        manager.request_status = {7: status}
        manager.prefill_response_tracker = defaultdict(set)
        manager.required_prefill_response_num_table = {7: expected}
        manager.prefill_drained_abort_rooms = set()
        manager._drained_abort_rooms = OrderedDict()
        manager._drained_abort_lock = threading.Lock()
        manager.req_to_decode_prefix_len = {}
        manager.req_to_pd_hidden_meta = {}
        manager.bootstrap_timeout = manager.waiting_timeout = 300
        manager.source_transfers_drained = Mock(return_value=True)
        manager.failure_records = {}
        manager.failure_lock = threading.Lock()
        manager._final_poll_enabled = False
        manager.enable_staging = False
        manager.record_failure = (
            lambda room, reason: manager.failure_records.__setitem__(room, reason)
        )
        return manager

    @staticmethod
    def status_msg(room, status, rank, drained_abort=False):
        msg = [str(room).encode(), str(status).encode(), str(rank).encode()]
        if drained_abort:
            msg.append(b"DRAINED_ABORT")
        return msg

    def test_drained_abort_waits_for_every_unique_prefill_rank(self):
        manager = self.make_manager()
        duplicate_abort = self.status_msg(7, KVPoll.Failed, 0, drained_abort=True)
        manager._handle_prefill_status_message(duplicate_abort)
        manager._handle_prefill_status_message(duplicate_abort)
        self.assertEqual(manager.request_status[7], KVPoll.WaitingForInput)
        self.assertEqual(len(manager.prefill_response_tracker[7]), 1)
        manager._handle_prefill_status_message(self.status_msg(7, KVPoll.Success, 1))
        manager._handle_prefill_status_message(self.status_msg(7, KVPoll.Success, 2))
        self.assertEqual(manager.request_status[7], KVPoll.Failed)

        for order in (
            ("abort", "success", "success"),
            ("success", "success", "abort"),
        ):
            with self.subTest(order=order):
                manager = self.make_manager()
                rank_for_kind = iter((0, 1, 2))
                for kind in order:
                    rank = next(rank_for_kind)
                    is_abort = kind == "abort"
                    status = KVPoll.Failed if is_abort else KVPoll.Success
                    manager._handle_prefill_status_message(
                        self.status_msg(7, status, rank, is_abort)
                    )
                    if len(manager.prefill_response_tracker[7]) < 3:
                        self.assertEqual(
                            manager.request_status[7], KVPoll.WaitingForInput
                        )
                self.assertEqual(manager.request_status[7], KVPoll.Failed)
                self.assertIn(7, manager.prefill_drained_abort_rooms)
                self.assertIn(7, manager.failure_records)
                response_count = len(manager.prefill_response_tracker[7])
                manager._handle_prefill_status_message(
                    self.status_msg(7, KVPoll.Success, 99)
                )
                self.assertEqual(manager.request_status[7], KVPoll.Failed)
                self.assertEqual(
                    len(manager.prefill_response_tracker[7]), response_count
                )

                manager.request_status.clear()
                manager._handle_prefill_status_message(
                    self.status_msg(7, KVPoll.Failed, 99, drained_abort=True)
                )
                self.assertNotIn(7, manager.request_status)

    def test_success_and_legacy_failure_keep_existing_terminal_behavior(self):
        manager = self.make_manager()
        for rank in range(3):
            manager._handle_prefill_status_message(
                self.status_msg(7, KVPoll.Success, rank)
            )
        self.assertEqual(manager.request_status[7], KVPoll.Success)
        # A duplicate terminal response neither changes state nor re-runs the
        # final scatter path.
        manager._handle_prefill_status_message(self.status_msg(7, KVPoll.Success, 2))
        self.assertEqual(len(manager.prefill_response_tracker[7]), 3)
        manager._handle_prefill_status_message(
            self.status_msg(7, KVPoll.Failed, 0, drained_abort=True)
        )
        self.assertEqual(manager.request_status[7], KVPoll.Success)

        manager = self.make_manager(expected=4)
        manager._handle_prefill_status_message(self.status_msg(7, KVPoll.Success, 0))
        manager._handle_prefill_status_message(self.status_msg(7, KVPoll.Failed, 1))
        self.assertEqual(manager.request_status[7], KVPoll.Failed)
        self.assertNotIn(7, manager.prefill_drained_abort_rooms)

        manager = self.make_manager(expected=1)
        manager._handle_prefill_status_message(self.status_msg(7, KVPoll.Success, 0))
        self.assertEqual(manager.request_status[7], KVPoll.Success)
        manager._handle_prefill_status_message(self.status_msg(7, KVPoll.Failed, 1))
        self.assertEqual(manager.request_status[7], KVPoll.Failed)

    def test_late_drained_abort_target_and_saved_session_targets_are_covered(self):
        manager = self.make_manager(status=KVPoll.Failed)
        manager.transfer_infos = {
            7: {
                "session-1": SimpleNamespace(
                    endpoint="decode-a", dst_port=100, is_dummy=False
                ),
                "session-2": SimpleNamespace(
                    endpoint="decode-a", dst_port=100, is_dummy=False
                ),
                "session-3": SimpleNamespace(
                    endpoint="decode-b", dst_port=200, is_dummy=False
                ),
                "dummy": SimpleNamespace(endpoint="", dst_port=0, is_dummy=True),
            }
        }
        manager._source_drain = SimpleNamespace(lock=threading.Lock())
        manager.record_failure = Mock()
        manager.update_status = Mock()
        manager._wake_pd_hidden_ack_waiters = Mock()
        manager.sync_status_to_decode_endpoint = Mock()
        manager._prefill_unique_rank = lambda: 4
        sender = object.__new__(MooncakeKVSender)
        sender.kv_mgr = manager
        sender.bootstrap_room = 7
        sender.conclude_state = None
        sender.trace_ctx = Mock()
        sender._drained_abort_targets = set()
        sender._drained_abort_notified_targets = set()

        sender.abort()
        self.assertEqual(
            sender._drained_abort_targets,
            {("decode-a", 100), ("decode-b", 200)},
        )
        sender.notify_drained_abort()
        sender.notify_drained_abort()
        self.assertEqual(
            {
                (call.args[0], call.args[1])
                for call in manager.sync_status_to_decode_endpoint.call_args_list
            },
            {("decode-a", 100), ("decode-b", 200)},
        )
        self.assertEqual(manager.sync_status_to_decode_endpoint.call_count, 2)
        for call in manager.sync_status_to_decode_endpoint.call_args_list:
            self.assertEqual(call.args[2:5], (7, KVPoll.Failed, 4))
            self.assertTrue(call.kwargs["is_drained_abort"])

        # After live state was cleared, late bootstrap metadata is handled
        # by the terminal marker rather than resurrecting transfer state.
        self.assertNotIn(7, manager.request_status)
        self.assertNotIn(7, manager.transfer_infos)
        target = SimpleNamespace(endpoint="decode-c", dst_port=300, is_dummy=False)
        self.assertTrue(manager._notify_drained_abort_target_if_needed(7, target))
        self.assertEqual(manager.sync_status_to_decode_endpoint.call_count, 3)
        self.assertEqual(
            manager.sync_status_to_decode_endpoint.call_args.args[:2],
            ("decode-c", 300),
        )

    def test_bootstrap_target_is_not_notified_until_source_drain(self):
        manager = self.make_manager(status=KVPoll.Failed)
        manager.update_status(7, KVPoll.WaitingForInput)
        self.assertEqual(manager.request_status[7], KVPoll.Failed)
        manager.source_transfers_drained = Mock(return_value=False)
        manager.sync_status_to_decode_endpoint = Mock()
        manager._prefill_unique_rank = lambda: 2
        target = SimpleNamespace(endpoint="decode-late", dst_port=456, is_dummy=False)
        manager._notify_drained_abort_target_if_needed(7, target)
        manager.sync_status_to_decode_endpoint.assert_not_called()

        manager.source_transfers_drained.return_value = True
        manager._notify_drained_abort_target_if_needed(7, target)
        manager.sync_status_to_decode_endpoint.assert_called_once_with(
            "decode-late",
            456,
            7,
            KVPoll.Failed,
            2,
            is_drained_abort=True,
        )

    def test_late_bootstrap_notification_error_does_not_escape_thread_hook(self):
        manager = self.make_manager(status=KVPoll.Failed)
        manager.source_transfers_drained = Mock(return_value=True)
        manager.sync_status_to_decode_endpoint = Mock(
            side_effect=RuntimeError("temporary send failure")
        )
        manager._prefill_unique_rank = lambda: 2
        target = SimpleNamespace(endpoint="decode-late", dst_port=456, is_dummy=False)

        with self.assertLogs(
            "sglang.srt.disaggregation.mooncake.conn", level="ERROR"
        ) as logs:
            manager._notify_drained_abort_target_if_needed(7, target)

        self.assertIn("temporary send failure", logs.output[0])
        manager.sync_status_to_decode_endpoint.assert_called_once()

    def test_wire_marker_is_only_added_for_drained_abort(self):
        manager = object.__new__(MooncakeKVManager)
        manager._send_multipart_locked = Mock()
        manager.sync_status_to_decode_endpoint("127.0.0.1", 12345, 7, KVPoll.Failed, 2)
        manager.sync_status_to_decode_endpoint(
            "127.0.0.1", 12345, 7, KVPoll.Failed, 2, is_drained_abort=True
        )
        self.assertEqual(
            manager._send_multipart_locked.call_args_list[0].args[1],
            [b"7", str(KVPoll.Failed).encode(), b"2"],
        )
        self.assertEqual(
            manager._send_multipart_locked.call_args_list[1].args[1],
            [b"7", str(KVPoll.Failed).encode(), b"2", b"DRAINED_ABORT"],
        )


class TestSupplementalPollFinalChunks(CustomTestCase):
    @staticmethod
    def make_scheduler(inflight_rooms, final_rooms, wait_result=True, timeout_ms=0):
        kv_manager = SimpleNamespace(
            wait_for_transfer_rooms=Mock(return_value=wait_result)
        )
        scheduler = SimpleNamespace(
            _mooncake_final_poll_timeout_ms=timeout_ms,
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
        scheduler, kv_manager = self.make_scheduler(
            [1, 2, 3], [2, 3, 9], timeout_ms=400
        )
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
                scheduler, _ = self.make_scheduler(
                    [1], [1], wait_result, timeout_ms=400
                )
                with envs.SGLANG_MOONCAKE_FINAL_POLL_TIMEOUT_MS.override(400):
                    self.call(scheduler)
                scheduler.process_disagg_prefill_inflight_queue.assert_called_once_with()

    def test_skips_when_final_rooms_already_done(self):
        scheduler, kv_manager = self.make_scheduler([1], [2], timeout_ms=400)
        with envs.SGLANG_MOONCAKE_FINAL_POLL_TIMEOUT_MS.override(400):
            self.assertEqual(self.call(scheduler), [])
        kv_manager.wait_for_transfer_rooms.assert_not_called()
        scheduler.process_disagg_prefill_inflight_queue.assert_not_called()

    def test_timeout_is_counted_and_summary_logged(self):
        scheduler, _ = self.make_scheduler(
            [1], [1], wait_result=False, timeout_ms=400
        )
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
                    _mooncake_final_poll_timeout_ms=timeout_ms,
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
