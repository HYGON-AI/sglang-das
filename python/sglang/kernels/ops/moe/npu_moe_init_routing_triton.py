# Copyright 2026 SGLang Team
#
# Triton fallback for ``torch.ops.npu.npu_moe_init_routing_v2``.
#
# Implements the AscendTP path used with ``drop_pad_mode=2`` finalize:
#   - sort tokens by expert id
#   - gather hidden states into expert-grouped order
#   - build gather-type ``expanded_row_idx`` (row-major: index ``t * K + k``)
#   - emit per-expert token counts (expert_tokens_num_type=1) or cumsum (type=0)
#   - optional dynamic per-token int8 quant (quant_mode=1)

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_rows_kernel(
    x_ptr,
    token_ids_ptr,
    out_ptr,
    H,
    num_rows,
    stride_xm,
    stride_xh,
    stride_om,
    stride_oh,
    BLOCK_H: tl.constexpr,
):
    """out[row] = x[token_ids[row]] for row in [0, num_rows)."""
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_m >= num_rows:
        return

    token = tl.load(token_ids_ptr + pid_m).to(tl.int64)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs_h < H
    vals = tl.load(
        x_ptr + token * stride_xm + offs_h * stride_xh,
        mask=mask,
        other=0,
    )
    tl.store(
        out_ptr + pid_m * stride_om + offs_h * stride_oh,
        vals,
        mask=mask,
    )


@triton.jit
def _fill_expanded_row_idx_kernel(
    sorted_order_ptr,
    expanded_row_idx_ptr,
    numel,
    BLOCK: tl.constexpr,
):
    """expanded_row_idx[t * K + k] = dest (row-major, matches drop_pad_mode=2)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    # sorted_order values are flat indices t*K+k into the [N, K] layout.
    flat = tl.load(sorted_order_ptr + offs, mask=mask, other=0).to(tl.int64)
    tl.store(expanded_row_idx_ptr + flat, offs.to(tl.int32), mask=mask)


def _gather_rows(
    x: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    num_rows = token_ids.numel()
    hidden = x.shape[-1]
    out = torch.empty(
        (num_rows, hidden),
        dtype=x.dtype,
        device=x.device,
    )
    if num_rows == 0:
        return out

    block_h = min(256, triton.next_power_of_2(max(hidden, 1)))
    grid = (num_rows, triton.cdiv(hidden, block_h))
    _gather_rows_kernel[grid](
        x,
        token_ids,
        out,
        hidden,
        num_rows,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_H=block_h,
    )
    return out


def _fill_expanded_row_idx(
    sorted_order: torch.Tensor,
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    numel = sorted_order.numel()
    expanded_row_idx = torch.empty(
        (numel,),
        dtype=torch.int32,
        device=sorted_order.device,
    )
    if numel == 0:
        return expanded_row_idx

    block = 256
    grid = (triton.cdiv(numel, block),)
    _fill_expanded_row_idx_kernel[grid](
        sorted_order,
        expanded_row_idx,
        numel,
        BLOCK=block,
    )
    return expanded_row_idx


def _dynamic_quant_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token int8 dynamic quant, matching npu_dynamic_quant (scale=absmax/127)."""
    try:
        from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8

        x_q, scales = per_token_quant_int8(x, scale_dtype=torch.float32)
        if scales.dim() > 0 and scales.shape[-1] == 1:
            scales = scales.squeeze(-1)
        return x_q, scales.to(torch.float32)
    except Exception:
        x_f = x.to(torch.float32)
        absmax = x_f.abs().amax(dim=-1).clamp_min(1e-10)
        scale = absmax / 127.0
        y = torch.round(x_f / scale.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
        return y, scale


def npu_moe_init_routing_v2(
    x: torch.Tensor,
    expert_idx: torch.Tensor,
    *,
    scale: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
    active_num: int = -1,
    expert_capacity: int = -1,
    expert_num: int = -1,
    drop_pad_mode: int = 0,
    expert_tokens_num_type: int = 0,
    expert_tokens_num_flag: bool = False,
    quant_mode: int = 0,
    active_expert_range: Optional[Sequence[int]] = None,
    row_idx_type: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Triton/PyTorch implementation of ``npu_moe_init_routing_v2``.

    Only the configuration used by ``NPUMoEInitRouting_v2`` is supported:
    ``drop_pad_mode=0``, ``row_idx_type=0``, ``expert_tokens_num_flag=True``,
    ``expert_tokens_num_type`` in {0, 1}, ``quant_mode`` in {-1, 1}.
    """
    if x.dim() != 2:
        raise ValueError(f"x must be 2D [num_tokens, hidden], got {tuple(x.shape)}")
    if expert_idx.dim() != 2:
        raise ValueError(
            f"expert_idx must be 2D [num_tokens, top_k], got {tuple(expert_idx.shape)}"
        )
    if x.shape[0] != expert_idx.shape[0]:
        raise ValueError(
            f"x/expert_idx token mismatch: {x.shape[0]} vs {expert_idx.shape[0]}"
        )
    if drop_pad_mode != 0:
        raise NotImplementedError(
            f"npu_moe_init_routing_v2 triton fallback only supports "
            f"drop_pad_mode=0, got {drop_pad_mode}"
        )
    if row_idx_type != 0:
        raise NotImplementedError(
            f"npu_moe_init_routing_v2 triton fallback only supports "
            f"row_idx_type=0 (gather), got {row_idx_type}"
        )
    if expert_tokens_num_type not in (0, 1):
        raise NotImplementedError(
            f"npu_moe_init_routing_v2 triton fallback only supports "
            f"expert_tokens_num_type in {{0, 1}}, got {expert_tokens_num_type}"
        )
    if quant_mode not in (-1, 1):
        raise NotImplementedError(
            f"npu_moe_init_routing_v2 triton fallback only supports "
            f"quant_mode in {{-1, 1}}, got {quant_mode}"
        )
    if scale is not None or offset is not None:
        raise NotImplementedError(
            "npu_moe_init_routing_v2 triton fallback does not support scale/offset"
        )
    if expert_capacity not in (-1, 0):
        raise NotImplementedError(
            "npu_moe_init_routing_v2 triton fallback does not support expert_capacity"
        )

    num_tokens, _hidden = x.shape
    top_k = expert_idx.shape[1]
    total = num_tokens * top_k

    if expert_num is None or expert_num < 0:
        expert_num = int(expert_idx.max().item()) + 1 if total > 0 else 0

    if active_expert_range is None or len(active_expert_range) == 0:
        expert_start, expert_end = 0, expert_num
    else:
        if len(active_expert_range) != 2:
            raise ValueError(
                f"active_expert_range must be [start, end], got {active_expert_range}"
            )
        expert_start, expert_end = int(active_expert_range[0]), int(
            active_expert_range[1]
        )
        if expert_start != 0 or expert_end != expert_num:
            raise NotImplementedError(
                "npu_moe_init_routing_v2 triton fallback only supports "
                f"active_expert_range=[0, expert_num], got {list(active_expert_range)}"
            )

    if active_num < 0:
        active_num = total
    if active_num != total:
        raise NotImplementedError(
            "npu_moe_init_routing_v2 triton fallback only supports "
            f"active_num == num_tokens * top_k ({total}), got {active_num}"
        )

    device = x.device
    if total == 0:
        expanded_x = torch.empty((0, x.shape[1]), dtype=x.dtype, device=device)
        expanded_row_idx = torch.empty((0,), dtype=torch.int32, device=device)
        expert_tokens = torch.zeros((expert_num,), dtype=torch.int32, device=device)
        expanded_scale = (
            torch.empty((0,), dtype=torch.float32, device=device)
            if quant_mode == 1
            else None
        )
        return expanded_x, expanded_row_idx, expert_tokens, expanded_scale

    if not x.is_contiguous():
        x = x.contiguous()
    expert_idx = expert_idx.to(torch.int32).contiguous()
    expert_flat = expert_idx.reshape(-1)

    # Stable sort by expert id (matches Ascend ordering within each expert).
    sorted_order = torch.argsort(expert_flat, stable=True).to(torch.int32)
    token_ids = torch.div(sorted_order, top_k, rounding_mode="floor")

    expanded_x = _gather_rows(x, token_ids)
    expanded_row_idx = _fill_expanded_row_idx(sorted_order, num_tokens, top_k)

    counts = torch.bincount(expert_flat, minlength=expert_num).to(torch.int32)
    if expert_tokens_num_type == 1:
        expert_tokens = counts
    else:
        expert_tokens = torch.cumsum(counts, dim=0).to(torch.int32)

    expanded_scale: Optional[torch.Tensor]
    if quant_mode == 1:
        expanded_x, expanded_scale = _dynamic_quant_int8(expanded_x)
    else:
        expanded_scale = None

    # expert_tokens_num_flag is informational for Ascend; we always compute counts
    # when type is 0/1. Silence unused-arg lint.
    _ = expert_tokens_num_flag

    return expanded_x, expanded_row_idx, expert_tokens, expanded_scale
