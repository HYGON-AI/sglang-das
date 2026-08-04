# Copyright 2026 SGLang Team
#
# Triton fallback for ``torch.ops.npu.npu_moe_finalize_routing``.
#
# Covers the AscendTP path (``NPUFinalizeRouting(drop_pad_mode=2)``):
#   out[i] = sum_k scales[i, k] * expanded[expanded_src_to_dst_row[i * K + k]]
# Optional skip1/skip2/bias are supported when provided.

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_finalize_routing_kernel(
    expanded_ptr,
    out_ptr,
    scales_ptr,
    expanded_row_idx_ptr,
    expert_ids_ptr,
    skip1_ptr,
    skip2_ptr,
    bias_ptr,
    N,
    H,
    K,
    stride_em,
    stride_eh,
    stride_om,
    stride_oh,
    stride_sk,
    ROW_MAJOR: tl.constexpr,
    HAS_SCALES: tl.constexpr,
    HAS_SKIP1: tl.constexpr,
    HAS_SKIP2: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Combine expert outputs back to original token order."""
    token = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    if HAS_SKIP1:
        acc += tl.load(
            skip1_ptr + token * stride_om + offs_h * stride_oh,
            mask=mask_h,
            other=0.0,
        ).to(tl.float32)
    if HAS_SKIP2:
        acc += tl.load(
            skip2_ptr + token * stride_om + offs_h * stride_oh,
            mask=mask_h,
            other=0.0,
        ).to(tl.float32)

    for k in range(K):
        if ROW_MAJOR:
            src_key = token * K + k
        else:
            src_key = token + k * N
        dst = tl.load(expanded_row_idx_ptr + src_key).to(tl.int64)

        vals = tl.load(
            expanded_ptr + dst * stride_em + offs_h * stride_eh,
            mask=mask_h,
            other=0.0,
        ).to(tl.float32)

        if HAS_BIAS:
            expert_id = tl.load(expert_ids_ptr + token * K + k).to(tl.int64)
            vals = vals + tl.load(
                bias_ptr + expert_id * H + offs_h,
                mask=mask_h,
                other=0.0,
            ).to(tl.float32)

        if HAS_SCALES:
            scale = tl.load(scales_ptr + token * stride_sk + k).to(tl.float32)
            vals = vals * scale

        acc += vals

    tl.store(
        out_ptr + token * stride_om + offs_h * stride_oh,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_h,
    )


def _torch_moe_finalize_routing(
    expanded_permuted_rows: torch.Tensor,
    skip1: Optional[torch.Tensor],
    skip2: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    scales: Optional[torch.Tensor],
    expanded_src_to_dst_row: torch.Tensor,
    expert_for_source_row: Optional[torch.Tensor],
    drop_pad_mode: int,
) -> torch.Tensor:
    num_rows = scales.shape[0] if scales is not None else (
        expert_for_source_row.shape[0] if expert_for_source_row is not None else 0
    )
    top_k = scales.shape[1] if scales is not None else (
        expert_for_source_row.shape[1] if expert_for_source_row is not None else 1
    )
    hidden = expanded_permuted_rows.shape[-1]
    row_major = drop_pad_mode in (2, 3)

    out = torch.zeros(
        (num_rows, hidden),
        dtype=expanded_permuted_rows.dtype,
        device=expanded_permuted_rows.device,
    )
    if skip1 is not None:
        out = out + skip1
    if skip2 is not None:
        out = out + skip2

    for k in range(top_k):
        if row_major:
            src_key = torch.arange(num_rows, device=out.device) * top_k + k
        else:
            src_key = torch.arange(num_rows, device=out.device) + k * num_rows
        dst = expanded_src_to_dst_row[src_key].to(torch.int64)
        vals = expanded_permuted_rows[dst]
        if bias is not None:
            assert expert_for_source_row is not None
            vals = vals + bias[expert_for_source_row[:, k].to(torch.int64)]
        if scales is not None:
            vals = vals * scales[:, k].to(vals.dtype).unsqueeze(-1)
        out = out + vals
    return out


def npu_moe_finalize_routing(
    expanded_permuted_rows: torch.Tensor,
    skip1: Optional[torch.Tensor] = None,
    skip2: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    expanded_src_to_dst_row: Optional[torch.Tensor] = None,
    export_for_source_row: Optional[torch.Tensor] = None,
    drop_pad_mode: int = 0,
) -> torch.Tensor:
    """Triton fallback for ``torch.ops.npu.npu_moe_finalize_routing``.

    Supported: ``drop_pad_mode`` in {0, 2} (non-drop). Drop modes 1/3 are not
    implemented. ``export_for_source_row`` is only required when ``bias`` is set.
    """
    if expanded_src_to_dst_row is None:
        raise ValueError("expanded_src_to_dst_row is required")
    if drop_pad_mode not in (0, 2):
        raise NotImplementedError(
            f"npu_moe_finalize_routing triton fallback only supports "
            f"drop_pad_mode in {{0, 2}}, got {drop_pad_mode}"
        )
    if expanded_permuted_rows.dim() != 2:
        raise ValueError(
            f"expanded_permuted_rows must be 2D, got {tuple(expanded_permuted_rows.shape)}"
        )

    if scales is not None:
        if scales.dim() != 2:
            raise ValueError(f"scales must be 2D [N, K], got {tuple(scales.shape)}")
        num_rows, top_k = scales.shape
    elif export_for_source_row is not None:
        if export_for_source_row.dim() != 2:
            raise ValueError(
                f"export_for_source_row must be 2D [N, K], "
                f"got {tuple(export_for_source_row.shape)}"
            )
        num_rows, top_k = export_for_source_row.shape
    else:
        raise ValueError("either scales or export_for_source_row is required")

    if expanded_src_to_dst_row.numel() != num_rows * top_k:
        raise ValueError(
            f"expanded_src_to_dst_row numel ({expanded_src_to_dst_row.numel()}) "
            f"!= N*K ({num_rows * top_k})"
        )

    if bias is not None and export_for_source_row is None:
        raise ValueError("export_for_source_row is required when bias is provided")

    hidden = expanded_permuted_rows.shape[-1]
    if num_rows == 0:
        return torch.empty(
            (0, hidden),
            dtype=expanded_permuted_rows.dtype,
            device=expanded_permuted_rows.device,
        )

    if not expanded_permuted_rows.is_contiguous():
        expanded_permuted_rows = expanded_permuted_rows.contiguous()
    expanded_src_to_dst_row = expanded_src_to_dst_row.to(torch.int32).contiguous()
    if scales is not None and not scales.is_contiguous():
        scales = scales.contiguous()
    if export_for_source_row is not None:
        export_for_source_row = export_for_source_row.to(torch.int32).contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()
    if skip1 is not None and not skip1.is_contiguous():
        skip1 = skip1.contiguous()
    if skip2 is not None and not skip2.is_contiguous():
        skip2 = skip2.contiguous()

    out = torch.empty(
        (num_rows, hidden),
        dtype=expanded_permuted_rows.dtype,
        device=expanded_permuted_rows.device,
    )

    block_h = min(256, triton.next_power_of_2(max(hidden, 1)))
    grid = (num_rows, triton.cdiv(hidden, block_h))
    row_major = drop_pad_mode == 2

    try:
        _moe_finalize_routing_kernel[grid](
            expanded_permuted_rows,
            out,
            scales if scales is not None else out,
            expanded_src_to_dst_row,
            export_for_source_row if export_for_source_row is not None else out,
            skip1 if skip1 is not None else out,
            skip2 if skip2 is not None else out,
            bias if bias is not None else out,
            num_rows,
            hidden,
            top_k,
            expanded_permuted_rows.stride(0),
            expanded_permuted_rows.stride(1),
            out.stride(0),
            out.stride(1),
            scales.stride(0) if scales is not None else 0,
            ROW_MAJOR=row_major,
            HAS_SCALES=scales is not None,
            HAS_SKIP1=skip1 is not None,
            HAS_SKIP2=skip2 is not None,
            HAS_BIAS=bias is not None,
            BLOCK_H=block_h,
        )
        return out
    except Exception:
        return _torch_moe_finalize_routing(
            expanded_permuted_rows,
            skip1,
            skip2,
            bias,
            scales,
            expanded_src_to_dst_row,
            export_for_source_row,
            drop_pad_mode,
        )
