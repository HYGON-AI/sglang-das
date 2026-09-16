# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Optional

import torch


def make_ep_balanced_expert_ids(
    num_tokens: int,
    topk: int,
    num_experts: int,
    *,
    ep_size: int,
    ep_rank: int,
    device: torch.device,
    dtype: torch.dtype,
    layer_id: Optional[int] = None,
) -> torch.Tensor:
    """Build deterministic routed expert IDs for EP performance benchmarks.

    Each source EP rank starts its round-robin destination sequence at a
    different rank. When all source ranks have the same number of valid token
    rows, every destination rank receives exactly the same number of
    assignments. Exact global balance with unequal source-row counts would
    require a per-step collective; without one, each source is still spread as
    evenly as possible over all destinations. Assignments are then distributed
    over each destination's local experts. ``layer_id`` rotates only the local
    expert choice, leaving the cross-rank communication pattern unchanged
    between layers.

    A token is sent to ``min(topk, ep_size)`` destination ranks. For the common
    ``topk <= ep_size`` case, this intentionally gives every token maximum
    fanout and holds token-to-rank payload constant across EP sizes. If top-k is
    larger than EP, repeated destinations use distinct local experts. This is a
    balanced synthetic benchmark; it does not reproduce the lower fanout of a
    normal router when several selected experts share a rank. The function
    assumes standard contiguous expert placement and is not intended for
    correctness or accuracy evaluation.
    """
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if topk < 0:
        raise ValueError(f"topk must be non-negative, got {topk}")
    if ep_size <= 0:
        raise ValueError(f"ep_size must be positive, got {ep_size}")
    if not 0 <= ep_rank < ep_size:
        raise ValueError(f"ep_rank must be in [0, {ep_size}), got {ep_rank}")
    if num_experts <= 0 or num_experts % ep_size != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be positive and divisible by "
            f"ep_size ({ep_size})"
        )
    if topk > num_experts:
        raise ValueError(f"topk ({topk}) must not exceed num_experts ({num_experts})")
    if topk == 0:
        return torch.empty((num_tokens, 0), device=device, dtype=dtype)

    experts_per_rank = num_experts // ep_size
    layer_offset = 0 if layer_id is None else layer_id % experts_per_rank
    assignment = torch.arange(num_tokens * topk, device=device, dtype=dtype).view(
        num_tokens, topk
    )
    destination_rank = (assignment + ep_rank) % ep_size
    if topk <= ep_size:
        local_assignment = assignment
    else:
        slot = assignment % topk
        destination_slot = slot % ep_size
        occurrence = slot // ep_size
        full_occurrences = topk // ep_size
        extra_destinations = topk % ep_size
        slot_order = (
            destination_slot * full_occurrences
            + torch.clamp(destination_slot, max=extra_destinations)
            + occurrence
        )
        local_assignment = (assignment // topk) * topk + slot_order
    local_expert = (local_assignment + layer_offset) % experts_per_rank
    return destination_rank * experts_per_rank + local_expert
