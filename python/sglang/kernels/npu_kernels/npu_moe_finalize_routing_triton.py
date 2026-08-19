"""Triton implementation of ``torch_npu.npu_moe_finalize_routing``.

Semantics match cann-bench ``moe_finalize_routing/golden.py``::

    out = skip1 + skip2 + Σ_k scales[:,k] * (expanded_permuted_rows[idx] + bias[expert])

where ``idx`` comes from ``expanded_src_to_dst_row`` with drop_pad_mode layout:
  - mode 0/1: column-major  index_pos = k * num_rows + row
  - mode 2/3: row-major     index_pos = row * K + k
  - value == -1: drop the whole term (including bias)

Runs on CUDA / Hygon HCU via the ``cuda`` device.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import triton
import triton.language as tl


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("npu_moe_finalize_routing_triton requires a CUDA/HCU device")


def _to_device(t: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    if t is None:
        return None
    return t.detach().to(device=device).contiguous()


def _pow2_ge(x: int, lo: int = 32, hi: int = 1024) -> int:
    v = lo
    while v < x and v < hi:
        v *= 2
    return min(max(v, lo), hi)


@triton.jit
def _moe_finalize_routing_kernel(
    expanded_ptr,
    skip1_ptr,
    skip2_ptr,
    bias_ptr,
    scales_ptr,
    src_to_dst_ptr,
    expert_ptr,
    out_ptr,
    NUM_ROWS,
    H,
    K,
    NUM_EXPERTS,
    stride_exp_row,
    stride_exp_h,
    stride_skip1_row,
    stride_skip1_h,
    stride_skip2_row,
    stride_skip2_h,
    stride_bias_e,
    stride_bias_h,
    stride_scales_row,
    stride_scales_k,
    stride_expert_row,
    stride_expert_k,
    stride_out_row,
    stride_out_h,
    HAS_SKIP1: tl.constexpr,
    HAS_SKIP2: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SCALES: tl.constexpr,
    HAS_EXPERT: tl.constexpr,
    DROP_PAD_MODE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """One program per source token row."""
    row = tl.program_id(0)
    if row >= NUM_ROWS:
        return

    offs = tl.arange(0, BLOCK_H)
    # Accumulate in fp32 (same as golden's low-prec → float path).
    for h0 in range(0, H, BLOCK_H):
        h = h0 + offs
        mask = h < H

        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        if HAS_SKIP1:
            acc = acc + tl.load(
                skip1_ptr + row * stride_skip1_row + h * stride_skip1_h,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        if HAS_SKIP2:
            acc = acc + tl.load(
                skip2_ptr + row * stride_skip2_row + h * stride_skip2_h,
                mask=mask,
                other=0.0,
            ).to(tl.float32)

        for k in range(0, K):
            if DROP_PAD_MODE == 0 or DROP_PAD_MODE == 1:
                index_pos = k * NUM_ROWS + row
            else:
                index_pos = row * K + k

            value = tl.load(src_to_dst_ptr + index_pos)
            valid = value != -1
            # Gather expanded row; invalid (-1) → zero contribution.
            safe_value = tl.where(valid, value, 0)
            dst = tl.load(
                expanded_ptr + safe_value * stride_exp_row + h * stride_exp_h,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            dst = tl.where(valid, dst, 0.0)

            term = dst
            if HAS_BIAS and HAS_EXPERT:
                expert_id = tl.load(
                    expert_ptr + row * stride_expert_row + k * stride_expert_k
                )
                valid_e = (expert_id >= 0) & (expert_id < NUM_EXPERTS)
                safe_e = tl.where(valid_e, expert_id, 0)
                b = tl.load(
                    bias_ptr + safe_e * stride_bias_e + h * stride_bias_h,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                b = tl.where(valid_e, b, 0.0)
                term = dst + b

            # Dropped (-1) rows contribute nothing — including bias.
            term = tl.where(valid, term, 0.0)

            if HAS_SCALES:
                scale = tl.load(
                    scales_ptr + row * stride_scales_row + k * stride_scales_k
                ).to(tl.float32)
                acc = acc + scale * term
            else:
                acc = acc + term

        tl.store(
            out_ptr + row * stride_out_row + h * stride_out_h,
            acc,
            mask=mask,
        )


def npu_moe_finalize_routing_triton(
    expanded_permuted_rows: torch.Tensor,
    skip1: Optional[torch.Tensor] = None,
    skip2: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    expanded_src_to_dst_row: Optional[torch.Tensor] = None,
    expert_for_source_row: Optional[torch.Tensor] = None,
    drop_pad_mode: int = 0,
) -> torch.Tensor:
    """Triton reference for ``torch_npu.npu_moe_finalize_routing``.

    Argument order matches the torch_npu API (skip1 before expanded_src_to_dst_row).
    """
    if expanded_src_to_dst_row is None:
        raise ValueError("expanded_src_to_dst_row is required")
    if drop_pad_mode not in (0, 1, 2, 3):
        raise ValueError(f"drop_pad_mode must be in [0, 3], got {drop_pad_mode}")
    if skip1 is None and skip2 is not None:
        raise ValueError("skip2 requires skip1 (torch_npu constraint)")

    device = _device()
    original_dtype = expanded_permuted_rows.dtype

    expanded = _to_device(expanded_permuted_rows, device)
    skip1_d = _to_device(skip1, device)
    skip2_d = _to_device(skip2, device)
    bias_d = _to_device(bias, device)
    scales_d = _to_device(scales, device)
    esdr = _to_device(expanded_src_to_dst_row, device).to(torch.int32).reshape(-1)
    expert_d = _to_device(expert_for_source_row, device)

    H = int(expanded.shape[-1])
    expanded_2d = expanded.reshape(-1, H).contiguous()
    NK = int(esdr.numel())
    K = int(scales_d.shape[1]) if scales_d is not None else 1
    if NK % K != 0:
        raise ValueError(f"expanded_src_to_dst_row length {NK} not divisible by K={K}")
    num_rows = NK // K

    null_f = torch.empty(0, device=device, dtype=torch.float32)
    null_i = torch.empty(0, device=device, dtype=torch.int32)

    has_skip1 = skip1_d is not None
    has_skip2 = skip2_d is not None
    has_bias = bias_d is not None
    has_scales = scales_d is not None
    has_expert = expert_d is not None

    if has_bias and not has_expert:
        raise ValueError("bias requires expert_for_source_row")

    # Work in fp32 for accumulation; cast back at the end (matches golden).
    skip1_f = (
        skip1_d.to(torch.float32).contiguous() if has_skip1 else null_f
    )
    skip2_f = (
        skip2_d.to(torch.float32).contiguous() if has_skip2 else null_f
    )
    bias_f = bias_d.to(torch.float32).contiguous() if has_bias else null_f
    scales_f = (
        scales_d.to(torch.float32).contiguous() if has_scales else null_f
    )
    expert_i = (
        expert_d.to(torch.int32).contiguous() if has_expert else null_i
    )
    expanded_f = expanded_2d.to(torch.float32).contiguous()

    num_experts = int(bias_f.shape[0]) if has_bias else 0

    out = torch.empty((num_rows, H), device=device, dtype=torch.float32)

    if num_rows > 0 and H > 0:
        bh = _pow2_ge(min(H, 1024))
        _moe_finalize_routing_kernel[(num_rows,)](
            expanded_f,
            skip1_f,
            skip2_f,
            bias_f,
            scales_f,
            esdr,
            expert_i,
            out,
            num_rows,
            H,
            K,
            num_experts,
            expanded_f.stride(0),
            expanded_f.stride(1),
            skip1_f.stride(0) if has_skip1 else 0,
            skip1_f.stride(1) if has_skip1 else 0,
            skip2_f.stride(0) if has_skip2 else 0,
            skip2_f.stride(1) if has_skip2 else 0,
            bias_f.stride(0) if has_bias else 0,
            bias_f.stride(1) if has_bias else 0,
            scales_f.stride(0) if has_scales else 0,
            scales_f.stride(1) if has_scales else 0,
            expert_i.stride(0) if has_expert else 0,
            expert_i.stride(1) if has_expert else 0,
            out.stride(0),
            out.stride(1),
            HAS_SKIP1=has_skip1,
            HAS_SKIP2=has_skip2,
            HAS_BIAS=has_bias,
            HAS_SCALES=has_scales,
            HAS_EXPERT=has_expert,
            DROP_PAD_MODE=drop_pad_mode,
            BLOCK_H=bh,
        )

    if original_dtype in (torch.float16, torch.bfloat16, torch.float32):
        return out.to(original_dtype)
    return out
