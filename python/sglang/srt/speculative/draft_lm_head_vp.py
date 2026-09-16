"""Node-local vocabulary parallelism for EAGLE draft LM-head top-1."""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import torch

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
        local_world_size: int,
    ) -> None:
        from sglang.srt.distributed.parallel_state import (
            create_custom_parallel_group,
            get_tp_group,
        )

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
        if full_weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(
                "Draft LM-head VP requires FP16, BF16, or FP32 weights, "
                f"got {full_weight.dtype}."
            )

        tp_group = get_tp_group()
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
