import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.unified_cache_linker import ExternalLinkerLoadError

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _StopLoop(Exception):
    pass


class _Receiver:
    def __init__(self, iterations):
        self.iterations = iterations

    def recv_requests(self):
        if self.iterations <= 0:
            raise _StopLoop
        self.iterations -= 1
        return []


class _Batch:
    def __init__(self, name):
        self.name = name

    def copy(self):
        return self


def _make_overlap_scheduler(first_batch, failed_batch, first_result):
    plans = iter(
        [
            SimpleNamespace(batch_to_run=first_batch, running_batch=object()),
            SimpleNamespace(batch_to_run=failed_batch, running_batch=object()),
        ]
    )
    scheduler = SimpleNamespace(
        request_receiver=_Receiver(2),
        process_input_requests=MagicMock(),
        _engine_paused=False,
        waiting_queue=[],
        disagg_prefill_bootstrap_queue=SimpleNamespace(pop_bootstrapped=lambda: []),
        running_batch=object(),
        last_batch=None,
        get_next_disagg_prefill_batch_to_run=lambda **_: next(plans),
        ngram_embedding_manager=SimpleNamespace(
            prepare_for_forward=lambda batch, chunked_req: batch
        ),
        chunked_req=None,
        enable_staging=False,
        run_batch=MagicMock(
            side_effect=[
                first_result,
                ExternalLinkerLoadError("Mooncake range get returned 707"),
            ]
        ),
        _apply_war_barrier=MagicMock(),
        process_batch_result=MagicMock(),
        _abort_external_linker_failed_batch=MagicMock(),
        process_disagg_prefill_inflight_queue=MagicMock(),
        launch_batch_sample_if_needed=MagicMock(),
    )
    return scheduler


class TestPrefillExternalLinkerFailure(unittest.TestCase):
    def test_overlap_loop_drains_previous_result_and_aborts_only_failed_batch(self):
        first_batch = _Batch("first")
        failed_batch = _Batch("failed")
        first_result = object()
        scheduler = _make_overlap_scheduler(
            first_batch, failed_batch, first_result
        )

        with self.assertRaises(_StopLoop):
            SchedulerDisaggregationPrefillMixin.event_loop_overlap_disagg_prefill(
                scheduler
            )

        scheduler.process_batch_result.assert_called_once_with(
            first_batch, first_result
        )
        scheduler._abort_external_linker_failed_batch.assert_called_once()
        self.assertIs(
            scheduler._abort_external_linker_failed_batch.call_args.args[0],
            failed_batch,
        )
        self.assertIsNone(scheduler.last_batch)
        self.assertEqual(list(scheduler.result_queue), [])

    def test_pd_abort_releases_sender_and_metadata_state(self):
        sender = SimpleNamespace(abort=MagicMock())
        req = SimpleNamespace(
            rid="failed-rid",
            return_logprob=False,
            req_pool_idx=None,
            disagg_kv_sender=sender,
            pending_bootstrap=True,
            time_stats=SimpleNamespace(
                trace_ctx=SimpleNamespace(abort=MagicMock())
            ),
        )
        scheduler = SimpleNamespace(
            forward_stream=SimpleNamespace(synchronize=MagicMock()),
            tree_cache=SimpleNamespace(
                mark_external_linker_load_failed=MagicMock()
            ),
            disaggregation_mode=DisaggregationMode.PREFILL,
            clear_pending_chunk_send=MagicMock(),
            req_to_metadata_buffer_idx_allocator=object(),
            disagg_metadata_buffers=SimpleNamespace(pd_hidden_pool=object()),
            _release_aborted_request=MagicMock(),
            ipc_channels=SimpleNamespace(
                send_to_tokenizer=SimpleNamespace(send_output=MagicMock())
            ),
            chunked_req=None,
            _pending_chunked_abort_req=None,
            running_batch=SimpleNamespace(filter_batch=MagicMock()),
        )
        batch = SimpleNamespace(reqs=[req])

        with patch(
            "sglang.srt.managers.scheduler.maybe_release_metadata_buffer"
        ) as release_metadata:
            Scheduler._abort_external_linker_failed_batch(
                scheduler,
                batch,
                ExternalLinkerLoadError("Mooncake range get returned 707"),
            )

        sender.abort.assert_called_once_with()
        scheduler.clear_pending_chunk_send.assert_called_once_with(req)
        release_metadata.assert_called_once()
        self.assertFalse(req.pending_bootstrap)
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_called_once()


if __name__ == "__main__":
    unittest.main()
