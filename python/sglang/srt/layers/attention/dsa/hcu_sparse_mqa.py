# Copyright 2026 Hygon Information Technology Co., Ltd.
#
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

"""Eligibility checks for the paired LightOp sparse-MQA/mask-TopK path.

The sparse Page-MQA kernel intentionally leaves logical ``+0.0`` logits
unwritten. It is therefore safe only when its matching mask-aware paged TopK
consumer is available. A rejected route must happen before the sparse producer
runs so the dense Page-MQA path remains a correct fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence

import torch


@dataclass(frozen=True)
class LightOpSparseMQARoute:
    """Validated sparse-MQA dispatch choice.

    ``group_size == 1`` selects independent rows. Values 3, 4, and 5 select
    the MTP grouped implementation, after the caller has proved that each
    consecutive group shares one request's page table.
    """

    group_size: int


@lru_cache(maxsize=1)
def lightop_sparse_mask_api_available() -> bool:
    """Return whether both the sparse producer and consumer APIs exist."""

    try:
        from lightop.attention import fast_topk_transform_sparse_mask_fused
        from lightop.gemmopt import (
            page_mqa_logits_sparse_mask,
            page_mqa_logits_sparse_mask_grouped,
        )
    except (AttributeError, ImportError, OSError):
        return False

    return all(
        callable(operation)
        for operation in (
            page_mqa_logits_sparse_mask,
            page_mqa_logits_sparse_mask_grouped,
            fast_topk_transform_sparse_mask_fused,
        )
    )


def select_lightop_sparse_mqa_route(
    *,
    enabled: bool,
    api_available: bool,
    is_hcu: bool,
    arch_name: str,
    num_cus: int,
    page_size: int,
    topk: int,
    fuse_topk: bool,
    force_unfused_topk: bool,
    topk_transform_method_name: str,
    q: torch.Tensor,
    fused_kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_context_len: int,
    batch_size: int,
    is_target_verify: bool,
    is_draft_extend_v2: bool,
    mtp_group_size: Optional[int],
    grouping_lens_cpu: Sequence[int],
) -> Optional[LightOpSparseMQARoute]:
    """Choose the native gfx938 sparse-MQA route, or return ``None``.

    The validated target is deliberately narrow: gfx938/64CU, E4M3FN Q,
    packed FP8 KV, 32 heads, page size 64, and paged TopK=2048.
    """

    if not (
        enabled
        and api_available
        and is_hcu
        and fuse_topk
        and not force_unfused_topk
        and topk_transform_method_name == "PAGED"
        and topk == 2048
        and page_size == 64
        and arch_name.startswith("gfx938")
        and num_cus == 64
    ):
        return None

    if (
        q.dtype != torch.float8_e4m3fn
        or q.dim() != 4
        or tuple(q.shape[1:]) != (1, 32, 128)
        or q.shape[0] <= 0
        or not q.is_contiguous()
    ):
        return None

    rows = q.shape[0]
    if (
        fused_kv_cache.dtype != torch.uint8
        or fused_kv_cache.dim() != 4
        or fused_kv_cache.shape[0] <= 0
        or tuple(fused_kv_cache.shape[1:]) != (64, 1, 132)
        or tuple(fused_kv_cache.stride()) != (64 * 132, 132, 132, 1)
    ):
        return None

    if any(
        tensor.device != q.device
        for tensor in (fused_kv_cache, weights, context_lens, block_table)
    ):
        return None
    if (
        weights.dtype != torch.float32
        or tuple(weights.shape) != (rows, 32)
        or not weights.is_contiguous()
    ):
        return None
    if (
        context_lens.dtype != torch.int32
        or tuple(context_lens.shape) != (rows,)
        or not context_lens.is_contiguous()
    ):
        return None
    if (
        block_table.dtype != torch.int32
        or block_table.dim() != 2
        or block_table.shape[0] != rows
        or not block_table.is_contiguous()
        or max_context_len != block_table.shape[1] * page_size
    ):
        return None

    if is_target_verify or is_draft_extend_v2:
        if mtp_group_size not in (3, 4, 5):
            return None
        if rows != batch_size * mtp_group_size:
            return None
        has_request_lens = len(grouping_lens_cpu) == batch_size and all(
            extend_len == mtp_group_size for extend_len in grouping_lens_cpu
        )
        has_expanded_row_lens = len(grouping_lens_cpu) == rows and all(
            extend_len == 1 for extend_len in grouping_lens_cpu
        )
        if not (has_request_lens or has_expanded_row_lens):
            return None
        return LightOpSparseMQARoute(group_size=mtp_group_size)

    if rows == batch_size:
        return LightOpSparseMQARoute(group_size=1)
    return None
