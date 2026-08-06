"""Unit tests for the optional PD Decode StepInfo synchronization guard."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_dp_attn_mixin import (
    DP_DECODE_STEP_BUILD_ID,
    DP_DECODE_STEP_PROTOCOL_VERSION,
    MLPSyncBatchInfo,
    SchedulerDPAttnMixin,
)

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestPDDecodeStepInfoGate(CustomTestCase):
    def _new_scheduler(self) -> SchedulerDPAttnMixin:
        scheduler = SchedulerDPAttnMixin()
        scheduler.server_args = SimpleNamespace(
            disaggregation_mode="decode",
            enable_dp_attention=True,
            dp_size=2,
            disable_cuda_graph=False,
            disable_overlap_schedule=False,
        )
        scheduler.attn_tp_size = 1
        scheduler.attn_cp_size = 1
        scheduler.tp_group = MagicMock()
        scheduler.get_idle_batch = MagicMock()
        scheduler.offload_tags = set()
        scheduler.spec_algorithm = MagicMock()
        scheduler.spec_algorithm.is_eagle.return_value = False
        scheduler.model_config = SimpleNamespace(hf_config=MagicMock())
        return scheduler

    def test_guard_is_disabled_by_default(self):
        self.assertFalse(envs.SGLANG_ENABLE_PD_DECODE_STEPINFO_SYNC.default)

    def test_disabled_guard_uses_upstream_six_field_gather(self):
        sync_info = MLPSyncBatchInfo(
            dp_size=2,
            tp_size=1,
            cp_size=1,
            num_tokens=8,
            num_tokens_for_logprob=8,
            can_cuda_graph=True,
            is_extend_in_batch=False,
            local_can_run_tbo=True,
            local_forward_mode=1,
        )

        def fake_all_gather(output, local, group):
            output.view(2, 6).copy_(local.repeat(2, 1))

        with (
            patch(
                "sglang.srt.managers.scheduler_dp_attn_mixin.get_tp_group",
                return_value=SimpleNamespace(
                    active_ranks_cpu=torch.ones(2, dtype=torch.int64)
                ),
            ),
            patch(
                "torch.distributed.all_gather_into_tensor",
                side_effect=fake_all_gather,
            ) as all_gather,
            patch("torch.distributed.get_world_size") as get_world_size,
        ):
            sync_info.all_gather(device="cpu", group=object())

        all_gather.assert_called_once()
        get_world_size.assert_not_called()
        self.assertEqual(sync_info.tp0_info.shape, (2, 6))
        self.assertEqual(sync_info.global_num_tokens, [8, 8])

    def test_disabled_guard_preserves_skip_all_gather_path(self):
        scheduler = self._new_scheduler()
        sentinel = object()

        with (
            envs.SGLANG_ENABLE_PD_DECODE_STEPINFO_SYNC.override(False),
            envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(True),
            patch(
                "sglang.srt.managers.scheduler_dp_attn_mixin.require_mlp_tp_gather",
                return_value=True,
            ),
            patch(
                "sglang.srt.managers.scheduler_dp_attn_mixin.prepare_mlp_sync_batch_raw",
                return_value=sentinel,
            ) as prepare_raw,
        ):
            result = scheduler.prepare_mlp_sync_batch(None)

        self.assertIs(result, sentinel)
        self.assertIsNone(prepare_raw.call_args.kwargs["scheduler_step_info"])
        self.assertIsNone(prepare_raw.call_args.kwargs["sync_group_override"])
        self.assertFalse(hasattr(scheduler, "_dp_scheduler_epoch"))

    def test_enabled_guard_rejects_skip_all_gather(self):
        scheduler = self._new_scheduler()

        with (
            envs.SGLANG_ENABLE_PD_DECODE_STEPINFO_SYNC.override(True),
            envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(True),
            self.assertRaisesRegex(
                RuntimeError,
                "SGLANG_ENABLE_PD_DECODE_STEPINFO_SYNC=1 is incompatible",
            ),
        ):
            scheduler.prepare_mlp_sync_batch(None)

    def test_enabled_guard_attaches_stepinfo_and_advances_epoch(self):
        scheduler = self._new_scheduler()
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[]
        )
        scheduler.running_batch = SimpleNamespace(reqs=[object(), object()])
        scheduler._engine_paused = False
        scheduler._dp_scheduler_last_pd_ms = 3.6
        scheduler._dp_scheduler_pd_over_budget = False
        scheduler._dp_scheduler_epoch = 7
        scheduler.dp_scheduler_cpu_group = object()
        sentinel = object()

        with (
            envs.SGLANG_ENABLE_PD_DECODE_STEPINFO_SYNC.override(True),
            envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False),
            envs.SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.override(False),
            patch(
                "sglang.srt.managers.scheduler_dp_attn_mixin.require_mlp_tp_gather",
                return_value=True,
            ),
            patch(
                "sglang.srt.managers.scheduler_dp_attn_mixin.prepare_mlp_sync_batch_raw",
                return_value=sentinel,
            ) as prepare_raw,
        ):
            result = scheduler.prepare_mlp_sync_batch(None)

        self.assertIs(result, sentinel)
        self.assertEqual(
            prepare_raw.call_args.kwargs["scheduler_step_info"],
            [
                DP_DECODE_STEP_PROTOCOL_VERSION,
                DP_DECODE_STEP_BUILD_ID,
                7,
                0,
                0,
                0,
                2,
                0,
                4,
                0,
            ],
        )
        self.assertIs(
            prepare_raw.call_args.kwargs["sync_group_override"],
            scheduler.dp_scheduler_cpu_group,
        )
        self.assertEqual(scheduler._dp_scheduler_epoch, 8)

    def test_enabled_guard_allows_device_sync_without_cpu_group(self):
        scheduler = self._new_scheduler()
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[]
        )
        scheduler.running_batch = SimpleNamespace(reqs=[])
        scheduler._engine_paused = False
        scheduler._dp_scheduler_epoch = 0

        with (
            envs.SGLANG_ENABLE_PD_DECODE_STEPINFO_SYNC.override(True),
            envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False),
            envs.SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.override(True),
            patch(
                "sglang.srt.managers.scheduler_dp_attn_mixin.require_mlp_tp_gather",
                return_value=True,
            ),
            patch(
                "sglang.srt.managers.scheduler_dp_attn_mixin.prepare_mlp_sync_batch_raw",
                return_value=None,
            ) as prepare_raw,
        ):
            scheduler.prepare_mlp_sync_batch(None)

        self.assertIsNone(prepare_raw.call_args.kwargs["sync_group_override"])
        self.assertEqual(scheduler._dp_scheduler_epoch, 1)

    def test_enabled_guard_does_not_advance_epoch_after_sync_failure(self):
        scheduler = self._new_scheduler()
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[]
        )
        scheduler.running_batch = None
        scheduler._engine_paused = False
        scheduler._dp_scheduler_epoch = 11
        scheduler.dp_scheduler_cpu_group = object()

        with (
            envs.SGLANG_ENABLE_PD_DECODE_STEPINFO_SYNC.override(True),
            envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False),
            envs.SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.override(False),
            patch(
                "sglang.srt.managers.scheduler_dp_attn_mixin.require_mlp_tp_gather",
                return_value=True,
            ),
            patch(
                "sglang.srt.managers.scheduler_dp_attn_mixin.prepare_mlp_sync_batch_raw",
                side_effect=RuntimeError("sync failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "sync failed"),
        ):
            scheduler.prepare_mlp_sync_batch(None)

        self.assertEqual(scheduler._dp_scheduler_epoch, 11)

    def test_offload_requires_cpu_group_even_when_device_sync_is_enabled(self):
        scheduler = self._new_scheduler()
        scheduler.offload_tags = {"weights"}

        with (
            envs.SGLANG_ENABLE_PD_DECODE_STEPINFO_SYNC.override(True),
            envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False),
            envs.SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.override(True),
            self.assertRaisesRegex(
                RuntimeError, "dedicated dp_scheduler_cpu_group is not initialized"
            ),
        ):
            scheduler.prepare_mlp_sync_batch(None)


if __name__ == "__main__":
    unittest.main()
