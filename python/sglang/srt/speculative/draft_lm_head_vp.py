"""Node-local LM-head vocabulary parallelism for EAGLE and DSPARK."""

from __future__ import annotations

from sglang.srt.runtime_context import get_parallel
import logging
import os
from typing import List, Sequence, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def build_node_local_vp_groups(
    tp_ranks: Sequence[int],
    vp_size: int,
    local_world_size: int,
) -> List[List[int]]:
    """Split contiguous TP ranks into node-local vocabulary-parallel groups."""
    tp_ranks = list(tp_ranks)
    if vp_size <= 0:
        raise ValueError(f"vp_size must be positive, got {vp_size}.")
    if local_world_size <= 0:
        raise ValueError(f"local_world_size must be positive, got {local_world_size}.")
    if len(tp_ranks) % vp_size != 0:
        raise ValueError(
            f"TP size {len(tp_ranks)} must be divisible by VP size {vp_size}."
        )

    groups = [
        tp_ranks[start : start + vp_size] for start in range(0, len(tp_ranks), vp_size)
    ]
    for ranks in groups:
        node_ids = {rank // local_world_size for rank in ranks}
        if len(node_ids) != 1:
            raise ValueError(
                "Draft LM-head VP groups must stay within one node, but "
                f"group {ranks} crosses node boundaries for "
                f"local_world_size={local_world_size}."
            )
    return groups


def select_global_top1_from_candidates(
    gathered_candidates: torch.Tensor,
    shard_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge per-shard ``(max_score, local_token_id)`` candidates.

    ``gathered_candidates`` has shape ``[vp_size, num_rows, 2]``.  Vocabulary
    shards are ordered by global token range. ``torch.argmax`` returns the first
    maximum, so equal scores deterministically select the lowest global token id.
    """
    if gathered_candidates.ndim != 3 or gathered_candidates.shape[-1] != 2:
        raise ValueError(
            "gathered_candidates must have shape [vp_size, num_rows, 2], got "
            f"{tuple(gathered_candidates.shape)}."
        )
    if shard_size <= 0:
        raise ValueError(f"shard_size must be positive, got {shard_size}.")

    scores = gathered_candidates[..., 0]
    local_token_ids = gathered_candidates[..., 1].to(torch.int64)
    winning_shards = torch.argmax(scores, dim=0)
    row_indices = torch.arange(scores.shape[1], device=scores.device)
    winning_scores = scores[winning_shards, row_indices]
    winning_local_token_ids = local_token_ids[winning_shards, row_indices]
    winning_global_token_ids = (
        winning_shards.to(torch.int64) * shard_size + winning_local_token_ids
    )
    return winning_scores, winning_global_token_ids


class DraftLMHeadVocabParallelTop1:
    """Compute full-vocabulary EAGLE draft top-1 with a node-local VP group.

    Every rank keeps the target LM-head unchanged.  The draft path only reads
    the rank's vocabulary-row slice, gathers a fixed-size hidden-state buffer
    within the node, and exchanges one top-1 candidate per gathered row.
    """

    def __init__(
        self,
        *,
        full_weight: torch.Tensor,
        vocab_size: int,
        vp_size: int,
        max_rows_per_rank: int,
    ) -> None:
        from sglang.srt.distributed.parallel_state import create_custom_parallel_group

        if not torch.distributed.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before draft VP.")
        if full_weight.ndim != 2:
            raise ValueError(
                f"Draft LM-head weight must be 2-D, got {full_weight.ndim}-D."
            )
        if full_weight.shape[0] != vocab_size:
            raise ValueError(
                "Draft LM-head VP requires a full, unpadded local weight. "
                f"Got weight rows={full_weight.shape[0]}, vocab_size={vocab_size}."
            )
        if vocab_size % vp_size != 0:
            raise ValueError(
                f"vocab_size={vocab_size} must be divisible by vp_size={vp_size}."
            )
        if max_rows_per_rank <= 0:
            raise ValueError(
                f"max_rows_per_rank must be positive, got {max_rows_per_rank}."
            )
        if not full_weight.dtype.is_floating_point:
            raise ValueError(
                "Draft LM-head VP currently supports floating-point weights only, "
                f"got {full_weight.dtype}."
            )

        tp_group = get_parallel().tp_group
        local_world_size = int(
            os.environ.get("LOCAL_WORLD_SIZE")
            or os.environ.get("LOCAL_SIZE")
            or torch.cuda.device_count()
        )
        groups = build_node_local_vp_groups(
            tp_group.ranks,
            vp_size,
            local_world_size,
        )
        global_rank = torch.distributed.get_rank()
        group_ranks = next((ranks for ranks in groups if global_rank in ranks), None)
        if group_ranks is None:
            raise RuntimeError(
                f"Global rank {global_rank} is not present in TP ranks {tp_group.ranks}."
            )

        backend = torch.distributed.get_backend(tp_group.device_group)
        process_group = create_custom_parallel_group(group_ranks, backend=backend)
        if process_group is None:
            raise RuntimeError(
                f"Failed to create draft LM-head VP group {group_ranks}."
            )

        self.process_group = process_group
        self.group_ranks = group_ranks
        self.vp_size = vp_size
        self.vp_rank = group_ranks.index(global_rank)
        self.vocab_size = vocab_size
        self.shard_size = vocab_size // vp_size
        self.vocab_start = self.vp_rank * self.shard_size
        self.max_rows_per_rank = max_rows_per_rank
        self.total_rows = vp_size * max_rows_per_rank
        self.hidden_size = full_weight.shape[1]
        self.device = full_weight.device
        self.dtype = full_weight.dtype
        # rocBLAS on HCU can produce incorrect results for the non-contiguous
        # transpose view at larger padded row counts. Cache a contiguous
        # vocabulary shard transpose once so decode does not copy it per step.
        self.weight_shard_t = full_weight.narrow(
            0, self.vocab_start, self.shard_size
        ).T.contiguous()

        self.local_hidden = torch.empty(
            (max_rows_per_rank, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        self.gathered_hidden = torch.empty(
            (self.total_rows, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        self.local_logits = torch.empty(
            (self.total_rows, self.shard_size),
            dtype=self.dtype,
            device=self.device,
        )
        self.local_candidates = torch.empty(
            (self.total_rows, 2),
            dtype=torch.float32,
            device=self.device,
        )
        self.gathered_candidates = torch.empty(
            (self.vp_size * self.total_rows, 2),
            dtype=torch.float32,
            device=self.device,
        )

        # Initialize the RCCL communicator before CUDA graph capture.
        warmup = torch.zeros(1, dtype=torch.float32, device=self.device)
        torch.distributed.all_reduce(warmup, group=self.process_group)
        torch.cuda.synchronize(self.device)

        if self.vp_rank == 0:
            logger.info(
                "Enabled node-local draft LM-head VP top-1: ranks=%s, "
                "vocab_range_per_rank=%d, max_rows_per_rank=%d",
                self.group_ranks,
                self.shard_size,
                self.max_rows_per_rank,
            )

    @torch.no_grad()
    def project_top1(
        self,
        hidden_states: torch.Tensor,
        full_weight: torch.Tensor,
        *,
        logit_scale: float | None = None,
        final_logit_softcapping: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return full-vocabulary top-1 scores and token ids for local rows."""
        if hidden_states.ndim != 2:
            raise RuntimeError(
                "Draft LM-head VP expects 2-D hidden states, got shape "
                f"{tuple(hidden_states.shape)}."
            )
        if hidden_states.shape[1] != self.hidden_size:
            raise RuntimeError(
                "Draft LM-head VP hidden size changed after initialization: "
                f"got {hidden_states.shape[1]}, expected {self.hidden_size}."
            )
        if hidden_states.device != self.device or hidden_states.dtype != self.dtype:
            raise RuntimeError(
                "Draft LM-head VP hidden states must stay on the initialized "
                f"device/dtype, got device={hidden_states.device}, "
                f"dtype={hidden_states.dtype}; expected device={self.device}, "
                f"dtype={self.dtype}."
            )
        if full_weight.device != self.device or full_weight.dtype != self.dtype:
            raise RuntimeError(
                "Draft LM-head VP weight must stay on the initialized "
                f"device/dtype, got device={full_weight.device}, "
                f"dtype={full_weight.dtype}; expected device={self.device}, "
                f"dtype={self.dtype}."
            )
        local_rows = hidden_states.shape[0]
        if local_rows > self.max_rows_per_rank:
            raise RuntimeError(
                "Draft LM-head VP local row count exceeds its fixed communication "
                f"buffer: rows={local_rows}, max_rows={self.max_rows_per_rank}."
            )
        if full_weight.shape != (self.vocab_size, self.hidden_size):
            raise RuntimeError(
                "Draft LM-head weight shape changed after VP initialization: "
                f"got {tuple(full_weight.shape)}, expected "
                f"{(self.vocab_size, self.hidden_size)}."
            )

        if local_rows:
            self.local_hidden[:local_rows].copy_(hidden_states)
        if local_rows < self.max_rows_per_rank:
            self.local_hidden[local_rows:].zero_()
        torch.distributed.all_gather_into_tensor(
            self.gathered_hidden,
            self.local_hidden,
            group=self.process_group,
        )

        torch.mm(
            self.gathered_hidden,
            self.weight_shard_t,
            out=self.local_logits,
        )
        if logit_scale is not None:
            self.local_logits.mul_(logit_scale)
        if final_logit_softcapping:
            self.local_logits.div_(final_logit_softcapping)
            self.local_logits.tanh_()
            self.local_logits.mul_(final_logit_softcapping)

        local_scores, local_token_ids = torch.max(self.local_logits, dim=-1)
        self.local_candidates[:, 0].copy_(local_scores)
        # Vocabulary shards are much smaller than FP32's exact integer range.
        self.local_candidates[:, 1].copy_(local_token_ids)
        torch.distributed.all_gather_into_tensor(
            self.gathered_candidates,
            self.local_candidates,
            group=self.process_group,
        )

        gathered_candidates = self.gathered_candidates.view(
            self.vp_size,
            self.total_rows,
            2,
        )
        global_scores, global_token_ids = select_global_top1_from_candidates(
            gathered_candidates,
            self.shard_size,
        )
        local_start = self.vp_rank * self.max_rows_per_rank
        local_end = local_start + local_rows
        return (
            global_scores[local_start:local_end],
            global_token_ids[local_start:local_end],
        )


class DSparkLMHeadVocabParallel:
    def __init__(
        self,
        *,
        full_weight: torch.Tensor,
        vocab_size: int,
        vp_size: int,
        max_rows_per_rank: int,
        local_world_size: int,
        process_group=None,
        weight_shard_t: torch.Tensor | None = None,
        workspace: DSparkLMHeadVocabParallel | None = None,
    ) -> None:
        if vp_size <= 1 or max_rows_per_rank <= 0:
            raise ValueError(
                "DSPARK VP requires vp_size > 1 and positive row capacity."
            )
        if full_weight.ndim != 2 or full_weight.shape[0] != vocab_size:
            raise ValueError("DSPARK VP requires a full, unpadded 2-D LM-head weight.")
        if full_weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(
                "DSPARK VP requires an unquantized floating-point LM-head."
            )
        if vocab_size % vp_size:
            raise ValueError("The vocabulary size must be divisible by DSPARK VP size.")
        self._pynccl_comm = None
        self._cpu_group = None
        if workspace is not None:
            if (
                process_group is not None
                and process_group is not workspace.process_group
            ):
                raise ValueError("Shared VP workspace requires the same process group.")
            process_group = workspace.process_group
            self._pynccl_comm = workspace._pynccl_comm
            self._cpu_group = workspace._cpu_group
        if process_group is None:
            from sglang.srt.runtime_context import get_parallel

            tp_group = get_parallel().tp_group
            groups = build_node_local_vp_groups(
                tp_group.ranks, vp_size, local_world_size
            )
            rank = dist.get_rank()
            ranks = next(ranks for ranks in groups if rank in ranks)
            if ranks == tp_group.ranks:
                # VP8/TP8 uses the existing rank order. Reuse its communicator
                # instead of initializing both WORLD (for metadata exchange)
                # and a second NCCL group with their own resident buffers.
                process_group = tp_group.device_group
                self._cpu_group = tp_group.cpu_group
                comm = getattr(tp_group, "pynccl_comm", None)
                if (
                    comm is not None
                    and comm.available
                    and comm.world_size == vp_size
                    and comm.rank == dist.get_rank(process_group)
                    and comm.device == full_weight.device
                    and "ncclAllToAll" in comm.nccl._funcs
                ):
                    self._pynccl_comm = comm
            else:
                # VP groups are disjoint and known locally. Avoid object
                # collectives on WORLD, which may otherwise stage through GPUs.
                process_group = dist.new_group(
                    ranks=ranks,
                    backend=dist.get_backend(tp_group.device_group),
                    use_local_synchronization=True,
                )
        if process_group is None or dist.get_world_size(process_group) != vp_size:
            raise ValueError("DSPARK VP process group does not match vp_size.")
        if self._cpu_group is None:
            if dist.get_backend(process_group) == "gloo":
                self._cpu_group = process_group
            else:
                self._cpu_group = dist.new_group(
                    ranks=dist.get_process_group_ranks(process_group),
                    backend="gloo",
                    use_local_synchronization=True,
                )

        self.process_group = process_group
        self.vp_size = vp_size
        self.vp_rank = dist.get_rank(process_group)
        self.max_rows_per_rank = max_rows_per_rank
        self.vocab_size = vocab_size
        self.hidden_size = full_weight.shape[1]
        self.shard_size = vocab_size // vp_size
        self.dtype = full_weight.dtype
        self.device = full_weight.device
        # Own the transposed shard storage, even when a one-row/one-column
        # slice is already contiguous, so releasing the full weight frees it.
        # Contiguous layout also avoids the HCU strided-transpose GEMM issue.
        if weight_shard_t is None:
            self.weight_shard_t = (
                full_weight.detach()
                .narrow(0, self.vp_rank * self.shard_size, self.shard_size)
                .T.clone(memory_format=torch.contiguous_format)
            )
        else:
            if (
                weight_shard_t.shape != (self.hidden_size, self.shard_size)
                or weight_shard_t.device != self.device
                or weight_shard_t.dtype != self.dtype
                or not weight_shard_t.is_contiguous()
            ):
                raise ValueError("Shared VP weight shard has an incompatible layout.")
            self.weight_shard_t = weight_shard_t
        options = dict(device=self.device, dtype=self.dtype)
        if workspace is None:
            self._workspace_state = dict(rows=0, frozen=False, references=1)
            self.reserve_workspace(max_rows_per_rank)
            self._cpu_row_count = torch.empty((), dtype=torch.int64, device="cpu")
        else:
            if (
                workspace.vp_size != self.vp_size
                or workspace.hidden_size != self.hidden_size
                or workspace.vocab_size != self.vocab_size
                or workspace.device != self.device
                or workspace.dtype != self.dtype
            ):
                raise ValueError("Shared VP workspace has an incompatible layout.")
            # Initialization only: expand before either target or draft graphs
            # capture these addresses. Replays use the same ordered forward stream.
            workspace.reserve_workspace(max_rows_per_rank)
            self._workspace_state = workspace._workspace_state
            self._workspace_state["references"] += 1
            self._cpu_row_count = workspace._cpu_row_count
            for name in (
                "local_hidden",
                "gathered_hidden",
                "shard_logits",
                "received_logits",
            ):
                setattr(self, name, getattr(workspace, name))
        # Keep outputs separate. Draft logits can outlive projection while the
        # target computes p; reusing this output would corrupt the proposal q.
        self.full_logits = torch.empty((max_rows_per_rank, vocab_size), **options)

        # Initialize both collective types before CUDA Graph capture.
        if workspace is None:
            warm_send = torch.zeros(vp_size, device=self.device, dtype=self.dtype)
            warm_recv = torch.empty_like(warm_send)
            self._all_gather(warm_recv, warm_send[:1])
            self._all_to_all(warm_recv, warm_send)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

    @staticmethod
    def guard_online_weight_updates(*models) -> None:
        reason = (
            "Online weight updates are not supported while DSPARK LM-head VP "
            "is active: captured forwards retain derived vocabulary shards. "
            "Restart the service to load new weights."
        )
        for model in models:
            model._dspark_lm_head_vp_active = True
            model._dspark_vp_weight_update_error = reason
            if getattr(model, "_derived_weight_cache_error", None) is None:
                model._derived_weight_cache_error = reason

    def _all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None:
        if self._pynccl_comm is not None:
            with self._pynccl_comm.change_state(enable=True):
                self._pynccl_comm.all_gather(output, input)
        else:
            dist.all_gather_into_tensor(output, input, group=self.process_group)

    def _all_to_all(self, output: torch.Tensor, input: torch.Tensor) -> None:
        if self._pynccl_comm is not None:
            from sglang.srt.distributed.device_communicators.pynccl_wrapper import (
                buffer_type,
                cudaStream_t,
                ncclDataTypeEnum,
            )

            with self._pynccl_comm.change_state(enable=True):
                # The native RCCL collective avoids the additional per-peer
                # buffers of PyNccl's grouped send/recv implementation.
                self._pynccl_comm.nccl.ncclAllToAll(
                    buffer_type(input.data_ptr()),
                    buffer_type(output.data_ptr()),
                    input.numel() // self.vp_size,
                    ncclDataTypeEnum.from_torch(input.dtype),
                    self._pynccl_comm.comm,
                    cudaStream_t(torch.cuda.current_stream(self.device).cuda_stream),
                )
        else:
            dist.all_to_all_single(output, input, group=self.process_group)

    def reserve_workspace(self, rows: int) -> None:
        """Grow scratch once at initialization, never after use or aliasing."""
        if rows <= self._workspace_state["rows"]:
            return
        if self._workspace_state["frozen"] or self._workspace_state["references"] > 1:
            raise RuntimeError(
                "VP scratch addresses cannot change after use or sharing."
            )
        options = dict(device=self.device, dtype=self.dtype)
        total_rows = rows * self.vp_size
        self.local_hidden = torch.empty((rows, self.hidden_size), **options)
        self.gathered_hidden = torch.empty((total_rows, self.hidden_size), **options)
        self.shard_logits = torch.empty((total_rows, self.shard_size), **options)
        self.received_logits = torch.empty_like(self.shard_logits)
        self._workspace_state["rows"] = rows

    @torch.no_grad()
    def project_logits(
        self, hidden_states: torch.Tensor, *, padded_rows: int | None = None
    ) -> torch.Tensor:
        """Return local full-vocabulary logits, including for uneven/idle DP ranks.

        Graph callers supply their common capture bucket. Eager callers agree
        on the largest local row count. Every VP rank must call, including idle
        ranks with zero rows. Returned storage is reused on the next call.
        """
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_size:
            raise ValueError("DSPARK VP hidden states have an unexpected shape.")
        self._workspace_state["frozen"] = True
        rows = hidden_states.shape[0]
        if hidden_states.device != self.device:
            raise ValueError("DSPARK VP hidden states changed device.")
        if padded_rows is None:
            if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "DSPARK VP capture requires a common padded row count."
                )
            # Row counts are host metadata. Agree on the CPU without
            # reading a GPU scalar; captured calls supply padded_rows.
            self._cpu_row_count.fill_(rows)
            dist.all_reduce(
                self._cpu_row_count, op=dist.ReduceOp.MAX, group=self._cpu_group
            )
            padded_rows = int(self._cpu_row_count.item())
        if not rows <= padded_rows <= self.max_rows_per_rank:
            raise ValueError(
                f"DSPARK VP rows={rows}, padded={padded_rows} exceed "
                f"capacity={self.max_rows_per_rank}."
            )
        if padded_rows == 0:
            return self.full_logits[:0]
        local = self.local_hidden[:padded_rows]
        local[:rows].copy_(hidden_states)
        local[rows:].zero_()
        total = padded_rows * self.vp_size
        gathered = self.gathered_hidden[:total]
        partial = self.shard_logits[:total]
        received = self.received_logits[:total]
        self._all_gather(gathered, local)
        torch.mm(gathered, self.weight_shard_t, out=partial)
        self._all_to_all(received, partial)
        self.full_logits[:padded_rows].view(
            padded_rows, self.vp_size, self.shard_size
        ).copy_(
            received.view(self.vp_size, padded_rows, self.shard_size).permute(1, 0, 2)
        )
        return self.full_logits[:rows]
