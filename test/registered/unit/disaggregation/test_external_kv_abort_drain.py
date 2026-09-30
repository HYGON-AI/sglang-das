"""CPU-only regression tests using production method bodies extracted by AST.

These avoid importing CUDA/Mooncake on Windows. They supplement, not replace,
the ordinary prefill/PP tests and real-device transport stress tests on Linux.
"""

import ast
import concurrent.futures
import importlib.util
import pathlib
import sys
import threading
import unittest
from collections import defaultdict
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = pathlib.Path(__file__).resolve().parents[4]
# CI registration is parsed statically. Load its stdlib-only implementation
# directly to keep these CPU tests runnable without importing sglang/GPU modules.
ci_spec = importlib.util.spec_from_file_location(
    "abort_cleanup_ci_register", ROOT / "python/sglang/test/ci/ci_register.py"
)
ci_module = importlib.util.module_from_spec(ci_spec)
sys.modules[ci_spec.name] = ci_module
ci_spec.loader.exec_module(ci_module)
register_cpu_ci = ci_module.register_cpu_ci
register_cpu_ci(est_time=3, suite="base-a-test-cpu")

PREFILL = ROOT / "python/sglang/srt/disaggregation/prefill.py"
SCHED = ROOT / "python/sglang/srt/managers/scheduler.py"
CONN = ROOT / "python/sglang/srt/disaggregation/mooncake/conn.py"
TRACKER = ROOT / "python/sglang/srt/disaggregation/mooncake/source_drain.py"
spec = importlib.util.spec_from_file_location("source_drain_under_test", TRACKER)
tracker_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracker_module)
SourceDrainTracker = tracker_module.SourceDrainTracker
KVPoll = SimpleNamespace(Success=1, Failed=2, Transferring=3)


class Reason:
    def __init__(self, message="failure", status_code=500, err_type="external"):
        self.message = message
        self.status_code = status_code
        self.err_type = err_type

    def to_json(self):
        return vars(self).copy()


def load_methods(path, names, env):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    methods = []
    for name in names:
        matches = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == name
        ]
        if len(matches) != 1:
            raise AssertionError((name, len(matches)))
        node = matches[0]
        node.decorator_list = []
        methods.append(node)
    cls = ast.ClassDef(
        name="UnderTest", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), env)
    return env["UnderTest"]


class TestAbortDrain(unittest.TestCase):
    def setUp(self):
        self.env = dict(
            KVPoll=KVPoll,
            HTTPStatus=HTTPStatus,
            FINISH_ABORT=Reason,
            EXTERNAL_KV_LOAD_ERR_TYPE="external",
            logger=Mock(),
            CleanupOutcomeUnknown=type("CleanupOutcomeUnknown", (RuntimeError,), {}),
            maybe_release_metadata_buffer=Mock(),
            release_kv_cache=Mock(),
            prepare_abort=Mock(),
            AbortReq=SimpleNamespace,
        )
        prefill_cls = load_methods(
            PREFILL,
            [
                "prefill_abort_sender_terminal",
                "prefill_abort_forward_drained",
                "abort_external_kv_request",
            ],
            self.env,
        )
        sched_cls = load_methods(
            SCHED,
            ["process_pending_chunked_abort", "_process_external_kv_chunked_abort"],
            self.env,
        )
        cls = type("Combined", (prefill_cls, sched_cls), {})
        self.s = cls()
        self.req = SimpleNamespace(
            rid="r",
            bootstrap_room=7,
            to_finish=Reason("original Mooncake failure"),
            finished_reason=None,
            skip_radix_cache_insert=False,
            return_logprob=False,
            req_pool_idx=1,
            kv=object(),
            mamba_pool_idx=None,
            inflight_middle_chunks=0,
            time_stats=Mock(),
            disagg_kv_sender=Mock(),
            external_kv_abort_response_via_chunked=False,
        )
        for flag in (
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
            setattr(self.req, "external_kv_" + flag, False)

        def apply_finish():
            self.req.finished_reason = self.req.to_finish
            self.req.to_finish = None

        self.req.update_finish_state = Mock(side_effect=apply_finish)
        self.req.disagg_kv_sender.poll.return_value = KVPoll.Failed
        self.req.disagg_kv_sender.is_transfer_drained.return_value = True
        s = self.s
        s._external_kv_abort_rids = {"r"}
        s._external_kv_cleanup_rids = {"r"}
        s._pending_chunked_abort_req = self.req
        s.chunked_req = self.req
        s.result_queue = []
        s.clear_pending_chunk_send = Mock()
        s._release_aborted_request = Mock()
        s.req_to_metadata_buffer_idx_allocator = object()
        s.disagg_metadata_buffers = SimpleNamespace(pd_hidden_pool=object())
        s.tree_cache = object()
        s.output_streamer = Mock()
        s.ipc_channels = SimpleNamespace(send_to_tokenizer=Mock())

    def finalize(self, **kwargs):
        facts = dict(forward_drained=True, sender_terminal=True, emit_response=True)
        facts.update(kwargs)
        return self.s.abort_external_kv_request(self.req, **facts)

    def test_chunked_result_queue_blocks_release_even_with_zero_counter(self):
        self.s.result_queue = [(SimpleNamespace(reqs=[self.req]), object())]
        self.s.process_pending_chunked_abort()
        self.env["release_kv_cache"].assert_not_called()
        self.env["maybe_release_metadata_buffer"].assert_not_called()
        self.assertIs(self.s._pending_chunked_abort_req, self.req)
        self.s.result_queue.clear()
        self.s.process_pending_chunked_abort()
        self.env["release_kv_cache"].assert_called_once()
        self.assertTrue(self.req.external_kv_response_sent)

    def test_chunked_send_failure_retains_retry_owner(self):
        send = self.s.ipc_channels.send_to_tokenizer.send_output
        send.side_effect = [RuntimeError("before delivery"), None]
        self.s.process_pending_chunked_abort()
        self.assertTrue(self.req.external_kv_cleanup_done)
        self.assertFalse(self.req.external_kv_response_sent)
        self.assertIs(self.s._pending_chunked_abort_req, self.req)
        # A resource-only completion observed from another queue cannot change
        # the response transport or send through output_streamer.
        self.assertTrue(self.finalize())
        self.s.output_streamer.stream_output.assert_not_called()
        self.s.process_pending_chunked_abort()
        self.s.process_pending_chunked_abort()
        self.assertEqual(send.call_count, 2)
        self.env["release_kv_cache"].assert_called_once()
        self.env["maybe_release_metadata_buffer"].assert_called_once()
        self.assertTrue(self.req.external_kv_response_sent)
        self.assertIsNone(self.s._pending_chunked_abort_req)

    def test_later_chunked_abort_does_not_overwrite_earlier_response_retry(self):
        send = self.s.ipc_channels.send_to_tokenizer.send_output
        send.side_effect = [RuntimeError("before delivery"), None, None]
        self.s.process_pending_chunked_abort()
        first = self.req
        second = SimpleNamespace(**vars(first))
        second.rid = "another-request"
        self.s._pending_chunked_abort_req = second
        self.s.chunked_req = second
        self.s.process_pending_chunked_abort()
        self.assertTrue(first.external_kv_response_sent)
        self.assertTrue(second.external_kv_response_sent)
        self.assertEqual(self.s._external_kv_chunked_abort_reqs, {})
        self.assertIsNone(self.s._pending_chunked_abort_req)
        self.assertEqual(
            [call.args[0].rid for call in send.call_args_list],
            [first.rid, first.rid, second.rid],
        )

    def test_unknown_kv_outcome_is_not_skipped_when_owner_fields_are_cleared(self):
        self.req.external_kv_cleanup_steps = {}

        def unknown(*args, **kwargs):
            self.req.req_pool_idx = self.req.kv = None
            self.req.external_kv_cleanup_steps["kv.cache_finished"] = ("unknown", None)
            raise RuntimeError("cache cleared owner fields then failed")

        self.env["release_kv_cache"].side_effect = unknown
        self.assertFalse(self.finalize())
        self.assertFalse(self.finalize())
        self.assertEqual(self.env["release_kv_cache"].call_count, 2)
        self.assertFalse(self.req.external_kv_cache_released)
        self.assertFalse(self.req.external_kv_cleanup_done)
        self.s.ipc_channels.send_to_tokenizer.send_output.assert_not_called()

    def test_logical_failed_is_not_a_source_drain(self):
        self.req.disagg_kv_sender.is_transfer_drained.return_value = False
        self.s.process_pending_chunked_abort()
        self.req.disagg_kv_sender.abort.assert_called_once()
        self.env["release_kv_cache"].assert_not_called()
        self.req.disagg_kv_sender.is_transfer_drained.return_value = True
        self.s.process_pending_chunked_abort()
        self.env["release_kv_cache"].assert_called_once()

    def test_finalizer_rechecks_sender_after_abort(self):
        self.req.disagg_kv_sender.abort.side_effect = lambda: setattr(
            self.req.disagg_kv_sender.is_transfer_drained, "return_value", False
        )
        self.assertFalse(self.finalize(sender_terminal=True))
        self.env["release_kv_cache"].assert_not_called()

    def test_poll_exception_retains_resources(self):
        self.req.disagg_kv_sender.poll.side_effect = RuntimeError("poll")
        self.s.process_pending_chunked_abort()
        self.env["release_kv_cache"].assert_not_called()
        self.assertIs(self.s._pending_chunked_abort_req, self.req)

    def test_each_stage_retries_without_repeating_completed_stages(self):
        for index in range(7):
            with self.subTest(stage=index):
                case = type(self)()
                case.setUp()
                case.s._pending_chunked_abort_req = None
                actions = [
                    case.s.clear_pending_chunk_send,
                    case.s._release_aborted_request,
                    case.req.disagg_kv_sender.abort,
                    case.req.update_finish_state,
                    case.env["maybe_release_metadata_buffer"],
                    case.env["release_kv_cache"],
                    case.s.output_streamer.stream_output,
                ]
                action = actions[index]
                prior = action.side_effect
                attempts = 0

                def fail_once(*args, **kwargs):
                    nonlocal attempts
                    attempts += 1
                    if attempts == 1:
                        raise RuntimeError("before side effect")
                    if callable(prior):
                        return prior(*args, **kwargs)

                action.side_effect = fail_once
                self.assertFalse(case.finalize())
                self.assertTrue(case.finalize())
                self.assertTrue(case.finalize())
                for i, completed in enumerate(actions):
                    self.assertEqual(completed.call_count, 2 if i == index else 1)

    def test_inflight_failure_delegates_and_never_directly_frees(self):
        tree = ast.parse(PREFILL.read_text(encoding="utf-8"))
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and n.name == "handle_inflight_transfer_failure"
        )
        calls = [n.func for n in ast.walk(node) if isinstance(n, ast.Call)]
        self.assertFalse(
            any(isinstance(n, ast.Name) and n.id == "release_kv_cache" for n in calls)
        )
        self.assertTrue(
            any(
                isinstance(n, ast.Attribute) and n.attr == "abort_external_kv_request"
                for n in calls
            )
        )


class TestSourceDrain(unittest.TestCase):
    def setUp(self):
        self.tracker = SourceDrainTracker()
        self.chunk = SimpleNamespace(room=7)

    def test_registration_precedes_publication_and_tracks_all_chunks(self):
        queue = Mock()
        queue.put.side_effect = lambda chunk: self.assertFalse(self.tracker.drained(7))
        self.tracker.publish(self.chunk, queue, lambda: True)
        another = SimpleNamespace(room=7)
        self.tracker.publish(another, queue, lambda: True)
        self.tracker.complete(self.chunk)
        self.tracker.complete(self.chunk)
        self.assertFalse(self.tracker.drained(7))
        self.tracker.complete(another)
        self.assertTrue(self.tracker.drained(7))

    def test_deferred_requeue_keeps_original_ownership(self):
        queue = Mock()
        self.tracker.publish(self.chunk, queue, lambda: True)
        queue.put(self.chunk)  # Worker defer / hidden ACK wake, same object.
        self.assertFalse(self.tracker.drained(7))
        self.tracker.complete(self.chunk)
        self.assertTrue(self.tracker.drained(7))

    def test_aborted_room_rejects_late_enqueue(self):
        queue = Mock()
        self.tracker.publish(self.chunk, queue, lambda: False)
        queue.put.assert_not_called()
        self.assertTrue(self.tracker.drained(7))

    def test_ambiguous_publication_stays_quarantined(self):
        queue = Mock()
        queue.put.side_effect = RuntimeError("after publication unknown")
        with self.assertRaises(RuntimeError):
            self.tracker.publish(self.chunk, queue, lambda: True)
        self.tracker.complete(self.chunk)
        self.assertFalse(self.tracker.drained(7))

    def test_worker_exception_never_claims_drain(self):
        self.tracker.poison(7)
        self.assertFalse(self.tracker.drained(7))

    def test_sender_probe_wakes_parked_chunks_but_does_not_invent_drain(self):
        env = dict(KVPoll=KVPoll)
        cls = load_methods(CONN, ["source_transfers_drained"], env)
        mgr = cls()
        mgr._source_drain = self.tracker
        mgr.request_status = {7: KVPoll.Failed}
        mgr._wake_pd_hidden_ack_waiters = Mock()
        self.tracker.publish(self.chunk, Mock(), lambda: True)
        self.assertFalse(mgr.source_transfers_drained(7))
        mgr._wake_pd_hidden_ack_waiters.assert_called_once_with(7)
        self.tracker.complete(self.chunk)
        self.assertTrue(mgr.source_transfers_drained(7))
        mgr.request_status.clear()  # failure_exception() clears the room.
        self.assertTrue(mgr.source_transfers_drained(7))

    def test_real_worker_skip_completes_only_its_own_token(self):
        class StopWorker(BaseException):
            pass

        cls = load_methods(
            CONN,
            ["transfer_worker"],
            dict(KVPoll=KVPoll, logger=Mock(), concurrent=concurrent),
        )
        mgr = cls()
        mgr.enable_trace = False
        mgr.enable_deferred_decode_kv_release = False
        mgr._source_drain = self.tracker
        mgr._staging_outstanding = defaultdict(int)
        mgr.request_status = {7: KVPoll.Failed}
        mgr.check_status = Mock(return_value=KVPoll.Failed)
        mgr._wake_pd_hidden_ack_waiters = Mock()
        mgr._has_pd_hidden_state = Mock(return_value=False)
        chunk = SimpleNamespace(
            room=7,
            staging_counted=False,
            source_event=None,
            pd_hidden_start=None,
            pd_hidden_sent=False,
            state_indices=None,
        )
        queue = Mock()
        self.tracker.publish(chunk, queue, lambda: True)
        self.tracker.publish(self.chunk, queue, lambda: True)
        queue.get.side_effect = [chunk, StopWorker()]
        with self.assertRaises(StopWorker):
            mgr.transfer_worker(queue, Mock())
        self.assertFalse(self.tracker.drained(7))
        self.tracker.complete(self.chunk)
        self.assertTrue(self.tracker.drained(7))

    def test_all_running_futures_drain_after_status_failure_or_exception(self):
        cls = load_methods(
            CONN, ["_await_transfer_futures"], dict(concurrent=concurrent)
        )
        for error in (False, True):
            for deferred in (False, True):
                with self.subTest(exception=error, deferred=deferred):
                    mgr = cls()
                    mgr.enable_deferred_decode_kv_release = deferred
                    first = concurrent.futures.Future()
                    running = concurrent.futures.Future()
                    running.set_running_or_notify_cancel()
                    failure_observed = threading.Event()
                    real_cancel = running.cancel

                    def cancel():
                        failure_observed.set()
                        return real_cancel()

                    running.cancel = cancel
                    if error:
                        first.set_exception(RuntimeError("transfer failed"))
                    else:
                        first.set_result(-1)
                    outcome = []

                    def wait():
                        try:
                            outcome.append(
                                mgr._await_transfer_futures([first, running])
                            )
                        except Exception as exc:
                            outcome.append(exc)

                    worker = threading.Thread(target=wait, daemon=True)
                    worker.start()
                    try:
                        self.assertTrue(failure_observed.wait(2))
                        self.assertTrue(worker.is_alive())
                        self.assertEqual(outcome, [])
                    finally:
                        running.set_result(0)
                        worker.join(2)
                    self.assertFalse(worker.is_alive())
                    if error:
                        self.assertIsInstance(outcome[0], RuntimeError)
                    else:
                        self.assertEqual(outcome, [-1])


if __name__ == "__main__":
    unittest.main()
