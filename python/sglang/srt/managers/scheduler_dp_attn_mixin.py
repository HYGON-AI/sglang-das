from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.batch_overlap.two_batch_overlap import TboDPAttentionPreparer
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.metrics_collector import DPCooperationInfo
from sglang.srt.utils.common import require_mlp_tp_gather

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.managers.scheduler import Scheduler


_ENABLE_METRICS_DP_ATTENTION = envs.SGLANG_ENABLE_METRICS_DP_ATTENTION.get()

logger = logging.getLogger(__name__)
FIX_I_STEP_PROTOCOL_VERSION = 2
FIX_I_STEP_BUILD_ID = 2026071602



@dataclass
class MLPSyncBatchInfo:
    dp_size: int
    tp_size: int
    cp_size: int

    num_tokens: int
    num_tokens_for_logprob: int
    can_cuda_graph: bool
    is_extend_in_batch: bool
    local_can_run_tbo: bool
    local_forward_mode: int

    # Fix I fields appended after the original six MLPSync values:
    # [protocol, build_id, epoch, transfer, prealloc, retracted, running, paused,
    #  pd_elapsed_ms, pd_over_budget]
    scheduler_step_info: Optional[list[int]] = None
    gathered_scheduler_step_info: Optional[torch.Tensor] = None

    # some gathered elements
    tp0_info: torch.Tensor = None
    global_num_tokens: list[int] = None
    global_num_tokens_for_logprob: list[int] = None
    tbo_split_seq_index: torch.Tensor = None
    global_forward_mode: int = None
    dp_cooperation_info: Optional[DPCooperationInfo] = None

    def _get_local_tensor(self, device, dtype=torch.int64) -> torch.Tensor:
        values = [
            self.num_tokens,
            self.num_tokens_for_logprob,
            int(self.can_cuda_graph),
            int(self.is_extend_in_batch),
            int(self.local_can_run_tbo),
            self.local_forward_mode,
        ]
        if self.scheduler_step_info is not None:
            if len(self.scheduler_step_info) != 10:
                raise RuntimeError(
                    "Fix I StepInfo must contain exactly 10 scheduler fields"
                )
            values.extend(self.scheduler_step_info)
        return torch.tensor(values, device=device, dtype=dtype)

    def _get_fallback_tensor(self, device, dtype=torch.int64) -> torch.Tensor:
        values = [
            0,  # num_tokens
            0,  # num_tokens_for_logprob
            1,  # can_cuda_graph
            0,  # is_extend_in_batch
            1,  # local_can_run_tbo
            ForwardMode.IDLE.value,  # local_forward_mode
        ]
        if self.scheduler_step_info is not None:
            values.extend(self.scheduler_step_info)
        return torch.tensor(values, device=device, dtype=dtype)

    def all_gather(self, device, group: torch.distributed.ProcessGroup):
        local_info_tensor = self._get_local_tensor(device=device)
        width = int(local_info_tensor.numel())
        expected_world = self.dp_size * self.tp_size * self.cp_size
        actual_world = torch.distributed.get_world_size(group=group)
        if actual_world != expected_world:
            raise RuntimeError(
                "Fix I scheduler group size mismatch: "
                f"actual={actual_world} expected={expected_world}"
            )

        if self.scheduler_step_info is not None and device == "cpu":
            # Fix I uses list all_gather for compatibility with customized
            # ROCm/DCU Gloo builds where _allgather_base may be unavailable.
            gathered = [
                torch.empty_like(local_info_tensor)
                for _ in range(actual_world)
            ]
            torch.distributed.all_gather(
                gathered,
                local_info_tensor,
                group=group,
            )
            global_info_flat = torch.stack(gathered, dim=0)
        else:
            global_info_flat = torch.empty(
                (actual_world, width),
                dtype=torch.int64,
                device=device,
            )
            torch.distributed.all_gather_into_tensor(
                global_info_flat.flatten(),
                local_info_tensor,
                group=group,
            )
        global_info_tensor = global_info_flat.view(
            self.dp_size,
            self.tp_size * self.cp_size,
            width,
        )

        if self.scheduler_step_info is not None:
            # Validate the raw values from every scheduler rank before masking
            # inactive model ranks.  Every participant must report one identical
            # protocol version and epoch.
            step_all = global_info_flat[:, 6:].cpu()
            versions = step_all[:, 0]
            builds = step_all[:, 1]
            epochs = step_all[:, 2]
            if (
                int(versions.min().item()) != FIX_I_STEP_PROTOCOL_VERSION
                or int(versions.max().item()) != FIX_I_STEP_PROTOCOL_VERSION
            ):
                raise RuntimeError(
                    "Fix I StepInfo protocol mismatch across ranks: "
                    f"{versions.tolist()}"
                )
            if (
                int(builds.min().item()) != FIX_I_STEP_BUILD_ID
                or int(builds.max().item()) != FIX_I_STEP_BUILD_ID
            ):
                raise RuntimeError(
                    "Fix I StepInfo build mismatch across ranks: "
                    f"{builds.tolist()}"
                )
            if int(epochs.min().item()) != int(epochs.max().item()):
                raise RuntimeError(
                    "Fix I scheduler epoch mismatch across ranks: "
                    f"{epochs.tolist()}"
                )
            paused = step_all[:, 7]
            if int(paused.min().item()) != int(paused.max().item()):
                raise RuntimeError(
                    "Fix I paused-state mismatch across ranks: "
                    f"{paused.tolist()}"
                )

        if device == "cpu":
            tp_active_ranks = get_tp_group().active_ranks_cpu
        else:
            tp_active_ranks = get_tp_group().active_ranks

        # Preserve epoch/diagnostic fields for inactive ranks; only replace the
        # six model-facing MLPSync fields with idle fallback values.
        tp_info = global_info_tensor.view(expected_world, width)
        inactive = tp_active_ranks == 0
        fallback = self._get_fallback_tensor(device=device)
        if width == 6:
            tp_info[inactive] = fallback
        else:
            tp_info[inactive, :6] = fallback[:6]

        tp0_info = global_info_tensor[:, 0, :]
        self.tp0_info = tp0_info
        cpu_data = tp0_info[:, :2].cpu()
        self.global_num_tokens = cpu_data[:, 0].tolist()
        self.global_num_tokens_for_logprob = cpu_data[:, 1].tolist()
        self.can_cuda_graph = bool(tp0_info[:, 2].min().item())
        self.is_extend_in_batch = bool(tp0_info[:, 3].max().item())

        if self.scheduler_step_info is not None:
            step_tp0 = tp0_info[:, 6:].cpu()
            self.gathered_scheduler_step_info = step_tp0
            epoch = int(step_tp0[0, 2].item())
            log_every = 1024
            over_budget = int(step_tp0[:, 9].max().item())
            if torch.distributed.get_rank(group=group) == 0 and (
                over_budget or (log_every > 0 and epoch % log_every == 0)
            ):

                def _minmax(col: int) -> tuple[int, int]:
                    return (
                        int(step_tp0[:, col].min().item()),
                        int(step_tp0[:, col].max().item()),
                    )

                logger.info(
                    "DP Decode StepInfo epoch=%s transfer=%s prealloc=%s "
                    "retracted=%s running=%s paused=%s pd_ms=%s "
                    "over_budget=%s",
                    epoch,
                    _minmax(3),
                    _minmax(4),
                    _minmax(5),
                    _minmax(6),
                    _minmax(7),
                    _minmax(8),
                    _minmax(9),
                )

        if _ENABLE_METRICS_DP_ATTENTION:
            self.dp_cooperation_info = DPCooperationInfo.create(
                tp0_info[:, 5].tolist()
            )




def _update_gather_batch(
    batch: ScheduleBatch,
    mlp_sync_info: MLPSyncBatchInfo,
    require_mlp_tp_gather: bool,
    skip_all_gather=False,
):
    # TODO: handle the case when moe_dense_tp_size != 1
    if not require_mlp_tp_gather:
        batch.global_num_tokens = [mlp_sync_info.num_tokens]
        batch.global_num_tokens_for_logprob = [mlp_sync_info.num_tokens_for_logprob]
    else:
        batch.global_num_tokens = mlp_sync_info.global_num_tokens
        batch.global_num_tokens_for_logprob = (
            mlp_sync_info.global_num_tokens_for_logprob
        )
    if not skip_all_gather:
        batch.is_extend_in_batch = mlp_sync_info.is_extend_in_batch
        batch.tbo_split_seq_index = mlp_sync_info.tbo_split_seq_index
        batch.global_forward_mode = mlp_sync_info.global_forward_mode

    # Check forward mode for cuda graph
    batch.can_run_dp_cuda_graph = mlp_sync_info.can_cuda_graph


def prepare_mlp_sync_batch_raw(
    local_batch: ScheduleBatch,
    dp_size: int,
    attn_tp_size: int,
    attn_cp_size: int,
    tp_group: GroupCoordinator,
    get_idle_batch: Callable[[], ScheduleBatch],
    disable_cuda_graph: bool,
    require_mlp_tp_gather: bool,
    disable_overlap_schedule: bool,
    offload_tags: set[str],
    scheduler_step_info: Optional[list[int]] = None,
    sync_group_override: Optional[torch.distributed.ProcessGroup] = None,
):
    # Check if other DP workers have running batches
    if local_batch is None or local_batch.forward_mode.is_prebuilt():
        num_tokens = 0
        num_tokens_for_logprob = 0
    elif local_batch.forward_mode.is_decode():
        num_tokens = local_batch.batch_size()
        num_tokens_for_logprob = num_tokens
    else:
        num_tokens = local_batch.extend_num_tokens
        num_tokens_for_logprob = sum(
            # We should have at least 1 token for sample in every case.
            max(extend_len - logprob_start_len, 1)
            for logprob_start_len, extend_len in zip(
                local_batch.extend_logprob_start_lens,
                local_batch.extend_lens,
            )
        )
        assert (
            local_batch.return_logprob
            or num_tokens_for_logprob == local_batch.batch_size()
        )

    skip_all_gather = envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.get()
    can_cuda_graph = (
        local_batch is None
        or local_batch.forward_mode.is_decode_or_idle()
        or local_batch.forward_mode.is_prebuilt()
    ) and not disable_cuda_graph

    is_extend_in_batch = local_batch.forward_mode.is_extend() if local_batch else False
    if local_batch is not None:
        local_batch.is_extend_in_batch = is_extend_in_batch

    tbo_preparer = TboDPAttentionPreparer()
    if sync_group_override is not None:
        # Fix I: scheduler metadata always uses its dedicated CPU communicator.
        group = sync_group_override
        device = "cpu"
    elif len(offload_tags) == 0 and (
        disable_overlap_schedule
        or envs.SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.get()
    ):
        group = tp_group.device_group
        device = tp_group.device
    else:
        group = tp_group.cpu_group
        device = "cpu"

    local_can_run_tbo, local_forward_mode = tbo_preparer.prepare_all_gather(local_batch)

    mlp_sync_info = MLPSyncBatchInfo(
        dp_size=dp_size,
        tp_size=attn_tp_size,
        cp_size=attn_cp_size,
        num_tokens=num_tokens,
        num_tokens_for_logprob=num_tokens_for_logprob,
        can_cuda_graph=can_cuda_graph,
        is_extend_in_batch=is_extend_in_batch,
        local_can_run_tbo=local_can_run_tbo,
        local_forward_mode=local_forward_mode,
        scheduler_step_info=scheduler_step_info,
    )

    if scheduler_step_info is not None and skip_all_gather:
        raise RuntimeError(
            "Fix I is incompatible with SGLANG_SCHEDULER_SKIP_ALL_GATHER"
        )

    if not skip_all_gather:
        mlp_sync_info.all_gather(device=device, group=group)

        mlp_sync_info.tbo_split_seq_index, mlp_sync_info.global_forward_mode = (
            tbo_preparer.compute_output(
                mlp_sync_info.tp0_info[:, 4:6],
            )
        )

    need_idle_batch = skip_all_gather or max(mlp_sync_info.global_num_tokens) > 0
    if need_idle_batch:
        batch_to_gather = local_batch
        if local_batch is None:
            batch_to_gather = local_batch = get_idle_batch()
        elif local_batch.forward_mode.is_prebuilt():
            # NOTE: for prebuilt batch, we add an inner idle batch to run MLP sync
            batch_to_gather = local_batch.inner_idle_batch = get_idle_batch()
        _update_gather_batch(
            batch_to_gather, mlp_sync_info, require_mlp_tp_gather, skip_all_gather
        )

    if _ENABLE_METRICS_DP_ATTENTION and local_batch is not None:
        local_batch.dp_cooperation_info = mlp_sync_info.dp_cooperation_info

    return local_batch


class SchedulerDPAttnMixin:
    def prepare_mlp_sync_batch(self: Scheduler, local_batch: ScheduleBatch):
        # Fix I: PD Decode folds scheduler epoch, queue observability and the
        # original six MLPSync fields into one fixed-shape all-gather.
        is_disagg_decode = (
            self.server_args.disaggregation_mode == "decode"
            and self.server_args.enable_dp_attention
        )
        scheduler_step_info = None
        sync_group_override = None
        epoch = None
        if is_disagg_decode:
            sync_group_override = getattr(
                self, "dp_scheduler_cpu_group", None
            )
            if sync_group_override is None:
                raise RuntimeError(
                    "Fix I dedicated dp_scheduler_cpu_group is not initialized"
                )
            epoch = int(getattr(self, "_dp_scheduler_epoch", 0))
            transfer_n = len(self.disagg_decode_transfer_queue.queue)
            prealloc_queue = self.disagg_decode_prealloc_queue
            prealloc_n = len(getattr(prealloc_queue, "queue", []) or [])
            retracted_n = len(prealloc_queue.retracted_queue)
            running_batch = getattr(self, "running_batch", None)
            running_n = len(getattr(running_batch, "reqs", []) or [])
            paused = int(bool(getattr(self, "_engine_paused", False)))
            pd_elapsed_ms = int(
                round(float(getattr(self, "_dp_scheduler_last_pd_ms", 0.0)))
            )
            pd_over_budget = int(
                bool(getattr(self, "_dp_scheduler_pd_over_budget", False))
            )
            scheduler_step_info = [
                FIX_I_STEP_PROTOCOL_VERSION,
                FIX_I_STEP_BUILD_ID,
                epoch,
                transfer_n,
                prealloc_n,
                retracted_n,
                running_n,
                paused,
                pd_elapsed_ms,
                pd_over_budget,
            ]

        result = prepare_mlp_sync_batch_raw(
            local_batch,
            dp_size=self.server_args.dp_size,
            attn_tp_size=self.attn_tp_size,
            attn_cp_size=self.attn_cp_size,
            tp_group=self.tp_group,
            get_idle_batch=self.get_idle_batch,
            disable_cuda_graph=self.server_args.disable_cuda_graph,
            require_mlp_tp_gather=require_mlp_tp_gather(self.server_args),
            disable_overlap_schedule=self.server_args.disable_overlap_schedule,
            offload_tags=self.offload_tags,
            scheduler_step_info=scheduler_step_info,
            sync_group_override=sync_group_override,
        )
        if is_disagg_decode:
            # Increment only after all ranks completed and validated the same
            # StepInfo collective.
            self._dp_scheduler_epoch = epoch + 1
        return result

    def maybe_prepare_mlp_sync_batch(
        self: Scheduler,
        batch: Optional[ScheduleBatch],
        need_sync: Optional[bool] = None,
    ) -> Optional[ScheduleBatch]:
        """
        Helper to prepare MLP sync batch for DP attention.
        Should be called after get_new_batch_prefill().

        Args:
            batch: The batch to process
            need_sync: If specified, overrides self.require_mlp_sync for prepare_mlp_sync_batch decision
        """
        if need_sync if need_sync is not None else self.require_mlp_sync:
            batch = self.prepare_mlp_sync_batch(batch)
        return batch

    def get_idle_batch(self: Scheduler) -> ScheduleBatch:
        idle_batch = ScheduleBatch.init_new(
            [],
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )
        idle_batch.prepare_for_idle()
        return idle_batch
