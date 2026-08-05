from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.observability.scheduler_metrics_mixin import (
    PrefillStats,
    SchedulerMetricsMixin,
)
from sglang.test.test_utils import CustomTestCase


class _FakeReq:
    def __init__(
        self,
        prompt_len: int,
        output_len: int = 0,
        prefix_len: int = 0,
    ):
        self.origin_input_ids = list(range(prompt_len))
        self.output_ids = list(range(output_len))
        self.prefix_indices = list(range(prefix_len))
        self.seqlen = prompt_len + output_len


class _FakeForwardMode:
    def __init__(self, *, is_mixed: bool = False, is_extend: bool = False):
        self._is_mixed = is_mixed
        self._is_extend = is_extend

    def is_mixed(self):
        return self._is_mixed

    def is_extend(self, include_draft_extend_v2: bool = False):
        return self._is_extend

    def is_decode(self):
        return not self._is_mixed and not self._is_extend


class _CollectingPublisher:
    def __init__(self):
        self.metrics = []

    def publish(self, metrics):
        self.metrics.append(metrics)


class _DummyPublisherThread:
    def __init__(self, endpoint: str, worker_id: str, dp_rank: int, **_: object):
        self.endpoint = endpoint
        self.worker_id = worker_id
        self.dp_rank = dp_rank

    def shutdown(self):
        pass


class _DummyScheduler(SchedulerMetricsMixin):
    pass


class TestForwardPassMetrics(CustomTestCase):
    def setUp(self):
        self.scheduler = _DummyScheduler()
        self.scheduler.enable_fpm = True
        self.scheduler._fpm_worker_id = "worker-7"
        self.scheduler._fpm_dp_rank = 0
        self.scheduler._fpm_publisher = _CollectingPublisher()
        self.scheduler._fpm_uses_device_timer = False
        self.scheduler._fpm_gpu_time_acc = 0.0
        self.scheduler.waiting_queue = []
        self.scheduler.disaggregation_mode = DisaggregationMode.NULL

    def _make_batch(self, **overrides):
        defaults = dict(
            forward_mode=_FakeForwardMode(),
            reqs=[],
            decoding_reqs=[],
            prefill_stats=None,
            seq_lens_cpu=[],
            seq_lens_sum=None,
            fpm_start_time=100.0,
        )
        defaults.update(overrides)
        return types.SimpleNamespace(**defaults)

    def test_emit_mixed_batch_separates_prefill_and_decode(self):
        self.scheduler._fpm_dp_rank = 3
        self.scheduler.waiting_queue = [_FakeReq(6), _FakeReq(4, output_len=2)]

        prefill_a = _FakeReq(10, prefix_len=2)
        prefill_b = _FakeReq(14, prefix_len=3)
        decode_req = _FakeReq(8, output_len=3)
        batch = self._make_batch(
            forward_mode=_FakeForwardMode(is_mixed=True, is_extend=True),
            reqs=[prefill_a, prefill_b, decode_req],
            decoding_reqs=[decode_req],
            prefill_stats=PrefillStats(
                log_input_tokens=12,
                log_hit_tokens=5,
                new_token_ratio=1.0,
                num_running_reqs=types.SimpleNamespace(),
                num_new_seqs=2,
            ),
            seq_lens_cpu=[decode_req.seqlen],
        )

        with patch(
            "sglang.srt.observability.scheduler_metrics_mixin.time.monotonic",
            return_value=104.5,
        ):
            self.scheduler._emit_forward_pass_metrics(batch)

        self.assertEqual(len(self.scheduler._fpm_publisher.metrics), 1)
        metrics = self.scheduler._fpm_publisher.metrics[0]
        self.assertEqual(metrics.worker_id, "worker-7")
        self.assertEqual(metrics.dp_rank, 3)
        self.assertEqual(metrics.wall_time, 4.5)
        self.assertEqual(metrics.scheduled_requests.num_prefill_requests, 2)
        self.assertEqual(metrics.scheduled_requests.sum_prefill_tokens, 12)
        self.assertEqual(metrics.scheduled_requests.sum_prefill_kv_tokens, 5)
        self.assertEqual(metrics.scheduled_requests.num_decode_requests, 1)
        self.assertEqual(
            metrics.scheduled_requests.sum_decode_kv_tokens, decode_req.seqlen
        )
        self.assertEqual(metrics.queued_requests.num_prefill_requests, 1)
        self.assertEqual(metrics.queued_requests.num_decode_requests, 1)

    def test_decode_metrics_fall_back_to_requests_without_cpu_mirror(self):
        decode_reqs = [_FakeReq(8, output_len=3), _FakeReq(13, output_len=5)]
        batch = self._make_batch(
            reqs=decode_reqs,
            seq_lens_cpu=None,
            seq_lens_sum=None,
        )

        metrics = self.scheduler._build_scheduled_request_metrics(batch)

        expected_sum = sum(req.seqlen for req in decode_reqs)
        self.assertEqual(metrics.num_decode_requests, len(decode_reqs))
        self.assertEqual(metrics.sum_decode_kv_tokens, expected_sum)
        self.assertEqual(
            self.scheduler._get_batch_seq_lens_sum(batch),
            expected_sum,
        )

        batch.seq_lens_sum = expected_sum + 7
        self.assertEqual(
            self.scheduler._get_batch_seq_lens_sum(batch),
            expected_sum + 7,
        )

    def test_schedule_batch_copy_preserves_seq_lens_sum_snapshot(self):
        batch = ScheduleBatch(reqs=[], seq_lens_cpu=None, seq_lens_sum=123)

        copied_batch = batch.copy()

        self.assertEqual(copied_batch.seq_lens_sum, 123)

    def test_result_snapshot_uses_pre_forward_deferred_seq_lens(self):
        current_seq_lens_cpu = torch.tensor([11, 18], dtype=torch.int64)
        next_seq_lens_cpu = torch.tensor([16, 23], dtype=torch.int64)
        batch = ScheduleBatch(
            reqs=[],
            seq_lens_cpu=None,
            seq_lens_sum=None,
            spec_info=types.SimpleNamespace(
                new_seq_lens_cpu=current_seq_lens_cpu,
            ),
        )

        result_seq_lens_cpu, result_seq_lens_sum = (
            batch.get_seq_lens_snapshot_for_result()
        )
        batch.spec_info = types.SimpleNamespace(
            new_seq_lens_cpu=next_seq_lens_cpu,
        )
        result_batch = batch.copy()
        result_batch.seq_lens_cpu = result_seq_lens_cpu
        result_batch.seq_lens_sum = result_seq_lens_sum

        self.assertIs(result_batch.seq_lens_cpu, current_seq_lens_cpu)
        self.assertIsNot(result_batch.seq_lens_cpu, next_seq_lens_cpu)
        self.assertEqual(result_batch.seq_lens_sum, None)

    def test_result_snapshot_copies_current_lengths_when_mirror_is_missing(self):
        batch = ScheduleBatch(
            reqs=[],
            device="cpu",
            seq_lens=torch.tensor([7, 12], dtype=torch.int64),
            seq_lens_cpu=None,
            # This can be stale after a spec-v2 filter/merge in overlap mode.
            seq_lens_sum=999,
            spec_info=types.SimpleNamespace(new_seq_lens_cpu=None),
        )

        result_seq_lens_cpu, result_seq_lens_sum = (
            batch.get_seq_lens_snapshot_for_result()
        )

        self.assertTrue(torch.equal(result_seq_lens_cpu, batch.seq_lens))
        self.assertIsNone(result_seq_lens_sum)

    def test_emit_uses_device_timer_gpu_time(self):
        self.scheduler._fpm_uses_device_timer = True
        self.scheduler._fpm_gpu_time_acc = 0.042
        self.scheduler.forward_pass_device_timer = types.SimpleNamespace(
            _report=lambda: None,
        )
        batch = self._make_batch()

        self.scheduler._emit_forward_pass_metrics(batch)

        self.assertEqual(len(self.scheduler._fpm_publisher.metrics), 1)
        self.assertAlmostEqual(
            self.scheduler._fpm_publisher.metrics[0].wall_time, 0.042, places=4
        )
        self.assertAlmostEqual(self.scheduler._fpm_gpu_time_acc, 0.0)

    def test_emit_skips_when_device_timer_zero(self):
        self.scheduler._fpm_uses_device_timer = True
        self.scheduler._fpm_gpu_time_acc = 0.0
        self.scheduler.forward_pass_device_timer = types.SimpleNamespace(
            _report=lambda: None,
        )
        batch = self._make_batch()

        self.scheduler._emit_forward_pass_metrics(batch)

        self.assertEqual(len(self.scheduler._fpm_publisher.metrics), 0)

    def test_emit_uses_monotonic_without_device_timer(self):
        batch = self._make_batch()

        with patch(
            "sglang.srt.observability.scheduler_metrics_mixin.time.monotonic",
            return_value=100.035,
        ):
            self.scheduler._emit_forward_pass_metrics(batch, result=None)

        self.assertEqual(len(self.scheduler._fpm_publisher.metrics), 1)
        self.assertAlmostEqual(
            self.scheduler._fpm_publisher.metrics[0].wall_time, 0.035, places=4
        )

    def test_disagg_prefill_queued_metrics(self):
        self.scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        self.scheduler.disagg_prefill_bootstrap_queue = types.SimpleNamespace(
            queue=[_FakeReq(100), _FakeReq(200), _FakeReq(50)],
        )
        batch = self._make_batch()

        with patch(
            "sglang.srt.observability.scheduler_metrics_mixin.time.monotonic",
            return_value=101.0,
        ):
            self.scheduler._emit_forward_pass_metrics(batch)

        metrics = self.scheduler._fpm_publisher.metrics[0]
        self.assertEqual(metrics.queued_requests.num_prefill_requests, 3)
        self.assertEqual(metrics.queued_requests.sum_prefill_tokens, 350)
        self.assertEqual(metrics.queued_requests.num_decode_requests, 0)

    def test_disagg_decode_queued_metrics(self):
        self.scheduler.disaggregation_mode = DisaggregationMode.DECODE
        self.scheduler.disagg_decode_prealloc_queue = types.SimpleNamespace(
            queue=[_FakeReq(10, output_len=5), _FakeReq(20, output_len=10)],
        )
        self.scheduler.disagg_decode_transfer_queue = types.SimpleNamespace(
            queue=[_FakeReq(30, output_len=15)],
        )
        batch = self._make_batch()

        with patch(
            "sglang.srt.observability.scheduler_metrics_mixin.time.monotonic",
            return_value=101.0,
        ):
            self.scheduler._emit_forward_pass_metrics(batch)

        metrics = self.scheduler._fpm_publisher.metrics[0]
        self.assertEqual(metrics.queued_requests.num_prefill_requests, 0)
        self.assertEqual(metrics.queued_requests.num_decode_requests, 3)
        self.assertEqual(metrics.queued_requests.sum_decode_kv_tokens, 15 + 30 + 45)

    def test_init_metrics_uses_server_worker_id(self):
        scheduler = _DummyScheduler()
        scheduler.server_args = types.SimpleNamespace(
            enable_metrics=False,
            enable_metrics_for_all_schedulers=False,
            extra_metric_labels=None,
            enable_forward_pass_metrics=True,
            forward_pass_metrics_worker_id="endpoint-42",
            forward_pass_metrics_ipc_name=None,
            kv_events_config=None,
        )
        scheduler.attn_tp_rank = 0
        scheduler.dp_rank = 2
        scheduler.pp_rank = 0
        scheduler.pp_size = 1
        scheduler.enable_kv_cache_events = False

        with patch(
            "sglang.srt.observability.forward_pass_metrics._FpmPublisherThread",
            _DummyPublisherThread,
        ):
            scheduler.init_metrics(tp_rank=0, pp_rank=0, dp_rank=2)

        self.assertTrue(scheduler.enable_fpm)
        self.assertEqual(scheduler._fpm_worker_id, "endpoint-42")
        self.assertEqual(scheduler._fpm_dp_rank, 2)
        self.assertEqual(scheduler._fpm_publisher.worker_id, "endpoint-42")
        self.assertEqual(scheduler._fpm_publisher.dp_rank, 2)
        self.assertTrue(scheduler._fpm_publisher.endpoint.startswith("ipc://"))
        self.assertIsNotNone(scheduler.server_args.forward_pass_metrics_ipc_name)

    def test_init_fpm_disabled_on_non_last_pp_rank(self):
        scheduler = _DummyScheduler()
        scheduler.server_args = types.SimpleNamespace(
            enable_metrics=False,
            enable_metrics_for_all_schedulers=False,
            extra_metric_labels=None,
            enable_forward_pass_metrics=True,
            forward_pass_metrics_worker_id="endpoint-42",
            forward_pass_metrics_ipc_name=None,
            kv_events_config=None,
        )
        scheduler.attn_tp_rank = 0
        scheduler.dp_rank = 0
        scheduler.pp_rank = 0
        scheduler.pp_size = 2
        scheduler.enable_kv_cache_events = False

        with patch(
            "sglang.srt.observability.forward_pass_metrics._FpmPublisherThread",
            _DummyPublisherThread,
        ):
            scheduler.init_metrics(tp_rank=0, pp_rank=0, dp_rank=0)

        self.assertFalse(scheduler.enable_fpm)


if __name__ == "__main__":
    unittest.main()
