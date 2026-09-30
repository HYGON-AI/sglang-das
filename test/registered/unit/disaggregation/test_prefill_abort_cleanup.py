"""Regression tests for request-owned prefill failure cleanup progress."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation import prefill  # noqa: E402
from sglang.srt.disaggregation.base import KVPoll  # noqa: E402
from sglang.srt.disaggregation.utils import EXTERNAL_KV_LOAD_ERR_TYPE  # noqa: E402
from sglang.srt.managers.schedule_batch import FINISH_ABORT  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPrefillAbortCleanup(unittest.TestCase):
    def setUp(self):
        self.scheduler = prefill.SchedulerDisaggregationPrefillMixin()
        self.req = SimpleNamespace(
            rid="failed-request",
            to_finish=None,
            finished_reason=None,
            return_logprob=False,
            skip_radix_cache_insert=False,
            pending_bootstrap=False,
            inflight_middle_chunks=0,
            req_pool_idx=1,
            kv=object(),
            mamba_pool_idx=None,
            bootstrap_room=7,
            bootstrap_host="peer",
            time_stats=Mock(),
            disagg_kv_sender=Mock(),
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
            setattr(self.req, "external_kv_" + stage, False)

        def apply_finish():
            self.req.finished_reason = self.req.to_finish
            self.req.to_finish = None

        self.req.update_finish_state = Mock(side_effect=apply_finish)
        self.req.disagg_kv_sender.poll.return_value = KVPoll.Failed
        self.req.disagg_kv_sender.failure_exception.side_effect = RuntimeError(
            "original Mooncake failure"
        )
        s = self.scheduler
        s._external_kv_abort_rids = set()
        s._external_kv_cleanup_rids = set()
        s._pending_chunked_abort_req = None
        s.disagg_prefill_pending_chunk_rids = {self.req.rid}
        s.disagg_prefill_inflight_queue = []
        s.result_queue = []
        s._release_aborted_request = Mock()
        s.clear_pending_chunk_send = Mock(wraps=s.clear_pending_chunk_send)
        s.output_streamer = Mock()
        s.req_to_metadata_buffer_idx_allocator = object()
        s.disagg_metadata_buffers = SimpleNamespace(pd_hidden_pool=object())
        s.tree_cache = object()
        s.ps = SimpleNamespace(tp_rank=0)
        s.metrics_reporter = SimpleNamespace(enable_metrics=True)
        s.metrics_collector = Mock()
        s.attn_cp_cpu_group = object()
        s.attn_tp_cpu_group = object()

        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.metadata = self.stack.enter_context(
            patch.object(prefill, "maybe_release_metadata_buffer")
        )
        self.cache = self.stack.enter_context(patch.object(prefill, "release_kv_cache"))
        self.stack.enter_context(
            patch.object(prefill, "maybe_release_pd_hidden_rows_on_hidden_done")
        )
        self.stack.enter_context(patch.object(prefill, "logger"))

    def finalize(self, **kwargs):
        facts = dict(forward_drained=True, sender_terminal=True, emit_response=True)
        facts.update(kwargs)
        return self.scheduler.abort_external_kv_request(self.req, **facts)

    def test_both_drain_conditions_are_required(self):
        for forward_drained, sender_terminal in (
            (False, False),
            (False, True),
            (True, False),
        ):
            with self.subTest(forward=forward_drained, sender=sender_terminal):
                self.assertFalse(
                    self.finalize(
                        forward_drained=forward_drained, sender_terminal=sender_terminal
                    )
                )
                self.metadata.assert_not_called()
                self.cache.assert_not_called()
                self.req.update_finish_state.assert_not_called()
                self.scheduler.output_streamer.stream_output.assert_not_called()
                self.assertIn(self.req.rid, self.scheduler._external_kv_abort_rids)
        self.assertTrue(self.finalize())
        self.req.disagg_kv_sender.abort.assert_called_once()
        self.scheduler._release_aborted_request.assert_called_once_with(self.req.rid)

    def test_stage_failures_retry_without_repeating_completed_actions(self):
        # Each injected failure happens before the action succeeds. The contract
        # cannot roll back arbitrary partially completed allocator side effects.
        stage_names = (
            "pending",
            "linker",
            "sender",
            "finish",
            "metadata",
            "cache",
            "output",
        )
        for failing_stage in stage_names:
            with self.subTest(stage=failing_stage):
                case = TestPrefillAbortCleanup()
                case.setUp()
                try:
                    actions = [
                        case.scheduler.clear_pending_chunk_send,
                        case.scheduler._release_aborted_request,
                        case.req.disagg_kv_sender.abort,
                        case.req.update_finish_state,
                        case.metadata,
                        case.cache,
                        case.scheduler.output_streamer.stream_output,
                    ]
                    index = stage_names.index(failing_stage)
                    action = actions[index]
                    original_effect = action.side_effect
                    attempts = 0

                    def fail_once(*args, **kwargs):
                        nonlocal attempts
                        attempts += 1
                        if attempts == 1:
                            raise RuntimeError("injected cleanup failure")
                        if callable(original_effect):
                            return original_effect(*args, **kwargs)

                    action.side_effect = fail_once
                    self.assertFalse(case.finalize())
                    self.assertFalse(case.req.external_kv_cleanup_done)
                    self.assertIn(case.req.rid, case.scheduler._external_kv_abort_rids)
                    for later in actions[index + 1 :]:
                        later.assert_not_called()
                    self.assertTrue(case.finalize())
                    self.assertTrue(case.finalize())
                    for i, completed in enumerate(actions):
                        self.assertEqual(completed.call_count, 2 if i == index else 1)
                    self.assertTrue(case.req.external_kv_response_sent)
                    self.assertNotIn(
                        case.req.rid, case.scheduler._external_kv_abort_rids
                    )
                finally:
                    case.doCleanups()

    def test_release_order_and_no_duplicate_response(self):
        order = Mock()
        for name, action in (
            ("pending", self.scheduler.clear_pending_chunk_send),
            ("linker", self.scheduler._release_aborted_request),
            ("sender", self.req.disagg_kv_sender.abort),
            ("finish", self.req.update_finish_state),
            ("metadata", self.metadata),
            ("cache", self.cache),
            ("response", self.scheduler.output_streamer.stream_output),
        ):
            order.attach_mock(action, name)
        self.assertTrue(self.finalize())
        self.assertTrue(self.finalize())
        self.assertEqual(
            [call[0] for call in order.mock_calls],
            ["pending", "linker", "sender", "finish", "metadata", "cache", "response"],
        )
        self.cache.assert_called_once_with(
            self.req, self.scheduler.tree_cache, is_insert=False
        )
        self.assertEqual(self.req.finished_reason.err_type, EXTERNAL_KV_LOAD_ERR_TYPE)

    def test_logprob_failure_keeps_empty_response_lists_and_original_error(self):
        self.req.return_logprob = True
        self.req.logprob = SimpleNamespace()
        self.assertTrue(self.finalize())
        self.assertEqual(self.req.logprob.input_token_logprobs_val, [])
        self.assertEqual(self.req.logprob.input_token_logprobs_idx, [])
        self.assertEqual(self.req.logprob.input_top_logprobs_val, [])
        self.assertEqual(self.req.logprob.input_token_ids_logprobs_val, [])
        self.assertEqual(self.req.finished_reason.err_type, EXTERNAL_KV_LOAD_ERR_TYPE)

    def test_missing_sender_can_finish(self):
        self.req.disagg_kv_sender = None
        self.assertTrue(self.scheduler.prefill_abort_sender_terminal(self.req))
        self.assertTrue(self.finalize())

    def test_chunk_counter_defers_bootstrap_cleanup_without_result_queue(self):
        self.req.inflight_middle_chunks = 1
        self.scheduler.handle_bootstrap_failure(self.req)
        self.metadata.assert_not_called()
        self.cache.assert_not_called()
        self.req.inflight_middle_chunks = 0
        with patch.object(
            prefill,
            "poll_and_all_reduce_attn_cp_tp_group",
            return_value=[KVPoll.Failed],
        ):
            self.assertEqual(
                self.scheduler.process_disagg_prefill_inflight_queue(), [self.req]
            )
        self.cache.assert_called_once()

    def test_existing_abort_reason_is_preserved(self):
        reason = FINISH_ABORT("user cancellation", status_code=499)
        self.req.to_finish = reason
        self.assertTrue(self.finalize())
        self.assertIs(self.req.finished_reason, reason)
        self.assertIsNone(reason.err_type)

    def test_resource_only_completion_can_deliver_response_later(self):
        self.assertTrue(self.finalize(emit_response=False))
        self.scheduler.output_streamer.stream_output.assert_not_called()
        self.assertTrue(self.finalize())
        self.assertTrue(self.finalize())
        self.metadata.assert_called_once()
        self.cache.assert_called_once()
        self.scheduler.output_streamer.stream_output.assert_called_once()

    def test_bootstrap_before_kv_allocation_does_not_free_unallocated_cache(self):
        self.req.req_pool_idx = self.req.kv = None
        self.scheduler.handle_bootstrap_failure(self.req)
        self.assertTrue(self.req.external_kv_cleanup_done)
        self.cache.assert_not_called()
        self.assertIn("original Mooncake failure", self.req.finished_reason.message)
        self.assertIsNone(self.req.finished_reason.err_type)

    def test_bootstrap_retains_request_until_forward_result_drains(self):
        self.scheduler.result_queue = [(SimpleNamespace(reqs=[self.req]), object())]
        self.scheduler.handle_bootstrap_failure(self.req)
        self.assertEqual(self.scheduler.disagg_prefill_inflight_queue, [self.req])
        self.cache.assert_not_called()
        self.scheduler.result_queue.clear()
        with patch.object(
            prefill,
            "poll_and_all_reduce_attn_cp_tp_group",
            return_value=[KVPoll.Failed],
        ):
            self.assertEqual(
                self.scheduler.process_disagg_prefill_inflight_queue(), [self.req]
            )
        self.cache.assert_called_once()
        self.assertEqual(self.scheduler.disagg_prefill_inflight_queue, [])

    def test_inflight_handler_delegates_cleanup_and_preserves_exception(self):
        failure = self.req.disagg_kv_sender.failure_exception.side_effect
        with patch.object(
            self.scheduler, "abort_external_kv_request", return_value=False
        ) as finalizer:
            exc, done = self.scheduler.handle_inflight_transfer_failure(
                self.req, sender_terminal=False
            )
        self.assertIs(exc, failure)
        self.assertFalse(done)
        self.cache.assert_not_called()
        self.metadata.assert_not_called()
        finalizer.assert_called_once_with(
            self.req,
            forward_drained=True,
            sender_terminal=False,
            abort_message=self.req.to_finish.message,
            is_insert=False,
            emit_response=True,
        )
        self.assertIn("original Mooncake failure", self.req.to_finish.message)

    def test_inflight_queue_retains_failed_cleanup_then_finishes_once(self):
        self.scheduler.disagg_prefill_inflight_queue = [self.req]
        self.cache.side_effect = [RuntimeError("retry cache release"), None]
        with patch.object(
            prefill,
            "poll_and_all_reduce_attn_cp_tp_group",
            return_value=[KVPoll.Failed],
        ):
            self.assertEqual(self.scheduler.process_disagg_prefill_inflight_queue(), [])
            self.assertEqual(self.scheduler.disagg_prefill_inflight_queue, [self.req])
            self.assertEqual(
                self.scheduler.process_disagg_prefill_inflight_queue(), [self.req]
            )
            self.assertEqual(self.scheduler.process_disagg_prefill_inflight_queue(), [])
        self.metadata.assert_called_once()
        self.assertEqual(self.cache.call_count, 2)
        self.req.disagg_kv_sender.abort.assert_called_once()
        self.scheduler.output_streamer.stream_output.assert_called_once()
        self.scheduler.metrics_collector.increment_transfer_failed_reqs.assert_called_once()

    def test_remote_failure_and_pending_bootstrap_do_not_prove_local_terminal(self):
        self.scheduler.disagg_prefill_inflight_queue = [self.req]
        self.req.pending_bootstrap = True
        self.req.disagg_kv_sender.poll.return_value = KVPoll.Transferring
        with patch.object(
            prefill,
            "poll_and_all_reduce_attn_cp_tp_group",
            return_value=[KVPoll.Failed],
        ):
            self.assertEqual(
                self.scheduler.process_disagg_prefill_inflight_queue(
                    ([], [self.req.rid])
                ),
                [],
            )
            self.cache.assert_not_called()
            self.metadata.assert_not_called()
            self.req.disagg_kv_sender.abort.assert_called_once()
            self.req.disagg_kv_sender.poll.return_value = KVPoll.Failed
            self.assertEqual(
                self.scheduler.process_disagg_prefill_inflight_queue(([], [])),
                [self.req],
            )
        self.assertIsNone(self.req.finished_reason.err_type)

    def test_poll_exception_keeps_cleanup_pending(self):
        self.req.disagg_kv_sender.poll.side_effect = RuntimeError("poll unavailable")
        self.scheduler.handle_bootstrap_failure(self.req)
        self.cache.assert_not_called()
        self.assertEqual(self.scheduler.disagg_prefill_inflight_queue, [self.req])


if __name__ == "__main__":
    unittest.main()
