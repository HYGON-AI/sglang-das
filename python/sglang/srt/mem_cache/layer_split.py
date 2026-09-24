# Copyright 2026 SGLang Team
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
"""Batch-scoped Main-KV page plans for NSA LayerSplit."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(eq=False, frozen=True)
class MainKVPagePlan:
    """Physical pages needed by one ordinary prefill ForwardBatch.

    ``history_page_ids`` is the unique payload copied from the layer owner.
    ``all_page_ids`` additionally contains pages receiving this step's newly
    computed KV.  Object identity deliberately serves as the batch identity.
    """

    history_page_ids: torch.Tensor
    all_page_ids: torch.Tensor


def _build_unique_physical_pages(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    sequence_lens: Sequence[int],
    page_size: int,
) -> torch.Tensor:
    """Collect one physical page ID per logical page and remove duplicates."""

    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if req_to_token.ndim != 2:
        raise ValueError(
            f"req_to_token must be 2-D, got shape={tuple(req_to_token.shape)}"
        )

    batch_size = len(sequence_lens)
    if batch_size == 0:
        return torch.empty(0, dtype=torch.long, device=req_to_token.device)
    if req_pool_indices.numel() < batch_size:
        raise ValueError(
            "req_pool_indices is shorter than sequence_lens: "
            f"{req_pool_indices.numel()} < {batch_size}"
        )

    page_counts = []
    for sequence_len in sequence_lens:
        sequence_len = int(sequence_len)
        if sequence_len < 0:
            raise ValueError(
                f"sequence length must be non-negative, got {sequence_len}"
            )
        page_counts.append((sequence_len + page_size - 1) // page_size)

    total_pages = sum(page_counts)
    if total_pages == 0:
        return torch.empty(0, dtype=torch.long, device=req_to_token.device)

    max_pages = max(page_counts)
    if (max_pages - 1) * page_size >= req_to_token.shape[1]:
        raise ValueError(
            "sequence length exceeds req_to_token capacity: "
            f"max_pages={max_pages}, page_size={page_size}, "
            f"capacity={req_to_token.shape[1]}"
        )

    device = req_to_token.device
    page_counts_tensor = torch.tensor(page_counts, dtype=torch.long, device=device)
    request_rows = torch.repeat_interleave(
        req_pool_indices[:batch_size].to(device=device, dtype=torch.long),
        page_counts_tensor,
        output_size=total_pages,
    )
    request_page_starts = torch.cumsum(page_counts_tensor, dim=0) - page_counts_tensor
    repeated_page_starts = torch.repeat_interleave(
        request_page_starts,
        page_counts_tensor,
        output_size=total_pages,
    )
    logical_pages = (
        torch.arange(total_pages, dtype=torch.long, device=device)
        - repeated_page_starts
    )
    physical_page_starts = req_to_token[request_rows, logical_pages * page_size]
    physical_pages = torch.div(
        physical_page_starts, page_size, rounding_mode="floor"
    ).to(torch.long)
    # Physical page 0 is the pool's padded/dummy page, never request history.
    return torch.unique(physical_pages[physical_pages > 0], sorted=True).contiguous()


def build_main_kv_page_plan(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    prefix_lens: Sequence[int],
    current_locs: torch.Tensor,
    page_size: int,
) -> MainKVPagePlan:
    """Build the deterministic, CP-global compact layout for one batch.

    The ForwardBatch request metadata remains global when NSA prefill CP splits
    query tokens, so every rank independently obtains the same two sorted page
    sets without a metadata collective.
    """

    history_page_ids = _build_unique_physical_pages(
        req_to_token,
        req_pool_indices,
        prefix_lens,
        page_size,
    )
    current_locs = current_locs.reshape(-1)
    valid_current_locs = current_locs[current_locs >= 0]
    current_page_ids = torch.unique(
        torch.div(valid_current_locs, page_size, rounding_mode="floor").to(torch.long),
        sorted=True,
    )
    current_page_ids = current_page_ids[current_page_ids > 0]
    all_page_ids = torch.unique(
        torch.cat((history_page_ids, current_page_ids)), sorted=True
    ).contiguous()
    return MainKVPagePlan(
        history_page_ids=history_page_ids,
        all_page_ids=all_page_ids,
    )
