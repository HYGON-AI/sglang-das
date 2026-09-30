"""ROCm-friendly per-batch replica dispatch for redundant EP experts.

The existing LPLB path uses a CUDA/Hopper-only cuBLASDx solver.  This module
implements the smaller problem needed by redundant-expert placement directly
in Triton: each source rank counts the current layer's routed tokens, greedily
water-fills the replicated experts across their candidate EP ranks, then
samples physical copies from the resulting probabilities.  Balancing every
source rank independently also balances their aggregate destination load, and
avoids a per-layer collective that can deadlock on asymmetric empty-token paths.

The balancing kernel is one program because the problem is tiny (128 logical
experts, eight EP ranks and at most ``num_redundant_experts`` replicated logical
experts).  Keeping the solve on device avoids a per-layer host synchronization.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl


def _next_power_of_2(value: int) -> int:
    return 1 << max(0, (value - 1).bit_length())


@triton.jit
def _count_logical_experts_kernel(
    topk_ids_ptr,
    counts_ptr,
    num_elements,
    topk: tl.constexpr,
    num_token_non_padded_ptr,
    HAS_NUM_TOKEN_NON_PADDED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_elements
    if HAS_NUM_TOKEN_NON_PADDED:
        num_valid_rows = tl.load(num_token_non_padded_ptr)
        mask = mask & ((offsets // topk) < num_valid_rows)
    logical_ids = tl.load(topk_ids_ptr + offsets, mask=mask, other=0).to(tl.int32)
    tl.atomic_add(counts_ptr + logical_ids, 1.0, mask=mask)


@triton.jit
def _build_load_aware_probabilities_kernel(
    expert_counts_ptr,
    num_valid_copies_ptr,
    copy_ranks_ptr,
    replicated_logical_ids_ptr,
    replicated_candidate_ranks_ptr,
    replicated_candidate_multiplicity_ptr,
    probabilities_ptr,
    num_logical: tl.constexpr,
    num_ranks: tl.constexpr,
    num_replicated,
    max_copies: tl.constexpr,
    BLOCK_LOGICAL: tl.constexpr,
    BLOCK_RANKS: tl.constexpr,
    BLOCK_REPLICATED: tl.constexpr,
    BLOCK_COPIES: tl.constexpr,
    NUM_WATERFILL_ITERS: tl.constexpr,
):
    logical_offsets = tl.arange(0, BLOCK_LOGICAL)
    logical_mask = logical_offsets < num_logical
    counts = tl.load(expert_counts_ptr + logical_offsets, mask=logical_mask, other=0.0)
    num_valid = tl.load(
        num_valid_copies_ptr + logical_offsets, mask=logical_mask, other=0
    ).to(tl.int32)
    primary_rank = tl.load(
        copy_ranks_ptr + logical_offsets * max_copies,
        mask=logical_mask,
        other=-1,
    ).to(tl.int32)

    # Initialize single-copy experts. Replicated rows are filled when their
    # water-fill step runs below.
    probability_offsets = tl.arange(0, BLOCK_LOGICAL * BLOCK_COPIES)
    probability_logical = probability_offsets // BLOCK_COPIES
    probability_copy = probability_offsets % BLOCK_COPIES
    probability_mask = (probability_logical < num_logical) & (
        probability_copy < max_copies
    )
    probability_num_valid = tl.load(
        num_valid_copies_ptr + probability_logical,
        mask=probability_mask,
        other=0,
    )
    initial_probability = tl.where(
        (probability_num_valid == 1) & (probability_copy == 0), 1.0, 0.0
    )
    tl.store(
        probabilities_ptr + probability_logical * max_copies + probability_copy,
        initial_probability,
        mask=probability_mask,
    )

    rank_offsets = tl.arange(0, BLOCK_RANKS)
    rank_mask = rank_offsets < num_ranks
    fixed_mask = (
        rank_mask[:, None]
        & logical_mask[None, :]
        & (num_valid[None, :] == 1)
        & (primary_rank[None, :] == rank_offsets[:, None])
    )
    rank_loads = tl.sum(tl.where(fixed_mask, counts[None, :], 0.0), axis=1)

    replicated_offsets = tl.arange(0, BLOCK_REPLICATED)
    replicated_mask = replicated_offsets < num_replicated
    replicated_ids = tl.load(
        replicated_logical_ids_ptr + replicated_offsets,
        mask=replicated_mask,
        other=0,
    ).to(tl.int32)
    replicated_counts = tl.load(
        expert_counts_ptr + replicated_ids,
        mask=replicated_mask,
        other=-1.0,
    )
    processed = ~replicated_mask
    copy_offsets = tl.arange(0, BLOCK_COPIES)

    # Largest replicated expert first makes the sequential water-fill robust
    # when different logical experts share candidate ranks.
    for step in tl.static_range(0, BLOCK_REPLICATED):
        active = step < num_replicated
        scores = tl.where(processed, -1.0, replicated_counts)
        selected = tl.argmax(scores, axis=0)
        selected_mask = replicated_offsets == selected
        logical_id = tl.sum(tl.where(selected_mask, replicated_ids, 0), axis=0).to(
            tl.int32
        )
        logical_count = tl.sum(tl.where(selected_mask, replicated_counts, 0.0), axis=0)
        processed = processed | (selected_mask & active)

        candidate_ranks = tl.load(
            replicated_candidate_ranks_ptr + selected * BLOCK_RANKS + rank_offsets,
            mask=active & rank_mask,
            other=-1,
        ).to(tl.int32)
        candidate_multiplicity = tl.load(
            replicated_candidate_multiplicity_ptr
            + selected * BLOCK_RANKS
            + rank_offsets,
            mask=active & rank_mask,
            other=0,
        ).to(tl.float32)
        candidate_mask = active & rank_mask & (candidate_ranks >= 0)

        candidate_loads = tl.sum(
            tl.where(
                candidate_mask[:, None]
                & (candidate_ranks[:, None] == rank_offsets[None, :]),
                rank_loads[None, :],
                0.0,
            ),
            axis=1,
        )
        low = tl.min(tl.where(candidate_mask, candidate_loads, float("inf")), axis=0)
        high = tl.max(
            tl.where(candidate_mask, candidate_loads, float("-inf")), axis=0
        ) + tl.maximum(logical_count, 0.0)

        # Find the water level t where sum(max(t - load[r], 0)) == M.
        for _ in tl.static_range(0, NUM_WATERFILL_ITERS):
            middle = (low + high) * 0.5
            required = tl.sum(
                tl.where(
                    candidate_mask,
                    tl.maximum(middle - candidate_loads, 0.0),
                    0.0,
                ),
                axis=0,
            )
            low = tl.where(required < logical_count, middle, low)
            high = tl.where(required < logical_count, high, middle)

        candidate_allocations = tl.where(
            candidate_mask,
            tl.maximum(high - candidate_loads, 0.0),
            0.0,
        )
        allocation_sum = tl.sum(candidate_allocations, axis=0)
        candidate_count = tl.sum(candidate_mask.to(tl.float32), axis=0)
        # Zero-count experts still need a valid probability row for the
        # dispatcher. Uniform-by-rank is deterministic and harmless.
        candidate_allocations = tl.where(
            logical_count > 0.0,
            candidate_allocations * logical_count / tl.maximum(allocation_sum, 1.0e-8),
            candidate_mask.to(tl.float32) / tl.maximum(candidate_count, 1.0),
        )

        load_increments = tl.sum(
            tl.where(
                rank_mask[:, None]
                & candidate_mask[None, :]
                & (rank_offsets[:, None] == candidate_ranks[None, :]),
                candidate_allocations[None, :],
                0.0,
            ),
            axis=1,
        )
        rank_loads += tl.where(active, load_increments, 0.0)

        copy_mask = active & (copy_offsets < max_copies)
        physical_copy_ranks = tl.load(
            copy_ranks_ptr + logical_id * max_copies + copy_offsets,
            mask=copy_mask,
            other=-1,
        ).to(tl.int32)
        per_rank_allocation = tl.sum(
            tl.where(
                candidate_mask[:, None]
                & (candidate_ranks[:, None] == physical_copy_ranks[None, :]),
                candidate_allocations[:, None],
                0.0,
            ),
            axis=0,
        )
        per_rank_multiplicity = tl.sum(
            tl.where(
                candidate_mask[:, None]
                & (candidate_ranks[:, None] == physical_copy_ranks[None, :]),
                candidate_multiplicity[:, None],
                0.0,
            ),
            axis=0,
        )
        copy_probability = per_rank_allocation / tl.maximum(per_rank_multiplicity, 1.0)
        copy_probability = tl.where(
            logical_count > 0.0,
            copy_probability / logical_count,
            copy_probability,
        )
        tl.store(
            probabilities_ptr + logical_id * max_copies + copy_offsets,
            copy_probability,
            mask=copy_mask,
        )


@triton.jit
def _dispatch_load_aware_kernel(
    topk_ids_ptr,
    probabilities_ptr,
    logical_to_physical_ptr,
    output_ptr,
    num_elements,
    seed,
    max_copies: tl.constexpr,
    BLOCK_COPIES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_elements
    logical_ids = tl.load(topk_ids_ptr + offsets, mask=mask, other=0).to(tl.int32)
    copy_offsets = tl.arange(0, BLOCK_COPIES)
    copy_mask = copy_offsets[None, :] < max_copies
    row_offsets = logical_ids[:, None] * max_copies + copy_offsets[None, :]
    probabilities = tl.load(
        probabilities_ptr + row_offsets,
        mask=mask[:, None] & copy_mask,
        other=0.0,
    )
    physical_ids = tl.load(
        logical_to_physical_ptr + row_offsets,
        mask=mask[:, None] & copy_mask,
        other=-1,
    ).to(tl.int32)
    row_sum = tl.sum(probabilities, axis=1)
    # Mix the logical id into the counter so expert-correlated token ordering
    # cannot systematically favor the same replica.
    random_values = tl.rand(seed, offsets + logical_ids * 0x45D9F3B)
    threshold = random_values * row_sum
    cumulative = tl.cumsum(probabilities, axis=1)
    chosen_copy = tl.sum((cumulative <= threshold[:, None]).to(tl.int32), axis=1)
    chosen_copy = tl.minimum(chosen_copy, max_copies - 1)
    output = tl.sum(
        tl.where(copy_offsets[None, :] == chosen_copy[:, None], physical_ids, 0),
        axis=1,
    )
    tl.store(output_ptr + offsets, output, mask=mask)


def count_logical_experts(
    topk_ids: torch.Tensor,
    counts: torch.Tensor,
    num_token_non_padded: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Count real routed assignments into a reusable float32 buffer."""
    if topk_ids.ndim != 2:
        raise ValueError(f"topk_ids must be 2-D, got {tuple(topk_ids.shape)}")
    if counts.dtype != torch.float32 or counts.ndim != 1:
        raise ValueError("counts must be a 1-D float32 tensor")
    counts.zero_()
    num_elements = topk_ids.numel()
    if num_elements == 0:
        return counts
    block = 256
    _count_logical_experts_kernel[(triton.cdiv(num_elements, block),)](
        topk_ids,
        counts,
        num_elements,
        topk=topk_ids.shape[1],
        num_token_non_padded_ptr=(
            num_token_non_padded if num_token_non_padded is not None else counts
        ),
        HAS_NUM_TOKEN_NON_PADDED=num_token_non_padded is not None,
        BLOCK=block,
    )
    return counts


def build_load_aware_probabilities(
    expert_counts: torch.Tensor,
    num_valid_copies: torch.Tensor,
    copy_ranks: torch.Tensor,
    replicated_logical_ids: torch.Tensor,
    replicated_candidate_ranks: torch.Tensor,
    replicated_candidate_multiplicity: torch.Tensor,
    probabilities: torch.Tensor,
    num_replicated: int,
) -> torch.Tensor:
    """Build per-physical-copy dispatch probabilities on the current stream."""
    num_logical, max_copies = probabilities.shape
    num_ranks = replicated_candidate_ranks.shape[1]
    block_logical = _next_power_of_2(num_logical)
    block_ranks = _next_power_of_2(num_ranks)
    block_replicated = replicated_logical_ids.numel()
    block_copies = _next_power_of_2(max_copies)
    if replicated_candidate_ranks.shape != (block_replicated, block_ranks):
        raise ValueError("replicated candidate-rank metadata has an invalid shape")
    _build_load_aware_probabilities_kernel[(1,)](
        expert_counts,
        num_valid_copies,
        copy_ranks,
        replicated_logical_ids,
        replicated_candidate_ranks,
        replicated_candidate_multiplicity,
        probabilities,
        num_logical=num_logical,
        num_ranks=num_ranks,
        num_replicated=num_replicated,
        max_copies=max_copies,
        BLOCK_LOGICAL=block_logical,
        BLOCK_RANKS=block_ranks,
        BLOCK_REPLICATED=block_replicated,
        BLOCK_COPIES=block_copies,
        NUM_WATERFILL_ITERS=16,
        num_warps=1,
    )
    return probabilities


def dispatch_load_aware(
    topk_ids: torch.Tensor,
    probabilities: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Map logical top-k ids to physical copies with device-side sampling."""
    if probabilities.shape != logical_to_physical_map.shape:
        raise ValueError(
            "load-aware probability/map shape mismatch: "
            f"{tuple(probabilities.shape)} vs {tuple(logical_to_physical_map.shape)}"
        )
    output = torch.empty_like(topk_ids)
    num_elements = topk_ids.numel()
    if num_elements == 0:
        return output
    block = 256
    max_copies = probabilities.shape[1]
    _dispatch_load_aware_kernel[(triton.cdiv(num_elements, block),)](
        topk_ids,
        probabilities,
        logical_to_physical_map,
        output,
        num_elements,
        seed=seed & 0x7FFFFFFF,
        max_copies=max_copies,
        BLOCK_COPIES=_next_power_of_2(max_copies),
        BLOCK=block,
    )
    return output


def build_load_aware_probabilities_cpu_reference(
    global_counts: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    num_ranks: int,
) -> torch.Tensor:
    """Readable CPU reference used by unit tests and placement diagnostics."""
    counts = global_counts.float().cpu()
    mapping = logical_to_physical_map.long().cpu()
    num_logical, max_copies = mapping.shape
    num_physical = int(mapping[mapping >= 0].max().item()) + 1
    physical_per_rank = math.ceil(num_physical / num_ranks)
    valid = mapping >= 0
    copy_ranks = torch.where(valid, mapping // physical_per_rank, -1)
    num_valid = valid.sum(dim=1)
    probabilities = torch.zeros((num_logical, max_copies), dtype=torch.float32)
    rank_loads = torch.zeros(num_ranks, dtype=torch.float32)

    for logical_id in range(num_logical):
        if num_valid[logical_id] == 1:
            probabilities[logical_id, 0] = 1.0
            rank_loads[copy_ranks[logical_id, 0]] += counts[logical_id]

    replicated = [
        logical_id for logical_id in range(num_logical) if num_valid[logical_id] > 1
    ]
    replicated.sort(key=lambda logical_id: float(counts[logical_id]), reverse=True)
    for logical_id in replicated:
        count = float(counts[logical_id])
        ranks = copy_ranks[logical_id, : num_valid[logical_id]].tolist()
        unique_ranks = sorted(set(ranks))
        candidate_loads = rank_loads[unique_ranks]
        if count > 0:
            low = float(candidate_loads.min())
            high = float(candidate_loads.max()) + count
            for _ in range(32):
                middle = (low + high) * 0.5
                required = float(torch.clamp(middle - candidate_loads, min=0).sum())
                if required < count:
                    low = middle
                else:
                    high = middle
            allocations = torch.clamp(high - candidate_loads, min=0)
            allocations *= count / max(float(allocations.sum()), 1.0e-8)
        else:
            allocations = torch.full(
                (len(unique_ranks),), 1.0 / len(unique_ranks), dtype=torch.float32
            )
        for rank, allocation in zip(unique_ranks, allocations.tolist()):
            rank_loads[rank] += allocation if count > 0 else 0.0
            copy_indices = [i for i, copy_rank in enumerate(ranks) if copy_rank == rank]
            probability = allocation / max(count, 1.0) / len(copy_indices)
            for copy_index in copy_indices:
                probabilities[logical_id, copy_index] = probability
    return probabilities
