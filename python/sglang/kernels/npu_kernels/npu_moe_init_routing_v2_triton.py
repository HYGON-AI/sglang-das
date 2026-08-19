"""Triton implementation of ``torch_npu.npu_moe_init_routing_v2``.

Implements the dropless paths in ``torch_npu-npu_moe_init_routing_v2.md``:

1. Stable key/value sort of ``expert_idx`` → ``sorted_row_idx`` / ``sorted_expert_idx``
2. ``expanded_row_idx`` (gather ``row_idx_type=0`` or scatter ``=1``)
3. Expert token histogram / cumsum / key-value
4. Optional static (``quant_mode=0``) / dynamic (``quant_mode=1``) int8 quant
5. Scatter-order token expand ``expanded_x[i] = x[sorted_row_idx[i] // K]``

Triton kernels cover token gather and quant epilogues; stable argsort /
``bincount`` use device PyTorch (same semantics as the CPU / PyTorch refs).

Not implemented (same as PyTorch ref): ``drop_pad_mode=1``, MXFP8 ``quant_mode`` 2/3.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

_INT8_MAX = 127.0
_INT32_MAX = torch.iinfo(torch.int32).max


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("npu_moe_init_routing_v2_triton requires a CUDA/HCU device")


def _to_device(t: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    if t is None:
        return None
    return t.detach().to(device=device).contiguous()


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.jit
def _gather_rows_kernel(
    src_ptr,
    idx_ptr,
    dst_ptr,
    N,
    H,
    stride_src_n,
    stride_src_h,
    stride_dst_n,
    stride_dst_h,
    BLOCK_H: tl.constexpr,
):
    """dst[i, :] = src[idx[i], :] for i in [0, N)."""
    pid = tl.program_id(0)
    if pid >= N:
        return
    row = tl.load(idx_ptr + pid)
    offs = tl.arange(0, BLOCK_H)
    for h0 in range(0, H, BLOCK_H):
        h = h0 + offs
        mask = h < H
        vals = tl.load(src_ptr + row * stride_src_n + h * stride_src_h, mask=mask)
        tl.store(dst_ptr + pid * stride_dst_n + h * stride_dst_h, vals, mask=mask)


@triton.jit
def _fill_gather_row_idx_i32(
    sorted_row_ptr,
    out_ptr,
    N_VALID,
):
    """out[sorted_row[i]] = i (int32) for i in [0, N_VALID)."""
    pid = tl.program_id(0)
    if pid >= N_VALID:
        return
    row = tl.load(sorted_row_ptr + pid)
    tl.store(out_ptr + row, pid.to(tl.int32))


@triton.jit
def _round_nearest(x):
    """Round half away from zero (HCU Triton has no ``tl.math.rint``)."""
    return tl.where(x >= 0, tl.math.floor(x + 0.5), tl.math.ceil(x - 0.5))


@triton.jit
def _static_quant_kernel(
    x_ptr,
    out_ptr,
    scale,
    offset,
    N,
    H,
    stride_xn,
    stride_xh,
    stride_on,
    stride_oh,
    BLOCK_H: tl.constexpr,
):
    """out = round(x * scale + offset).clamp(-128, 127) as int8."""
    pid = tl.program_id(0)
    if pid >= N:
        return
    offs = tl.arange(0, BLOCK_H)
    for h0 in range(0, H, BLOCK_H):
        h = h0 + offs
        mask = h < H
        x = tl.load(x_ptr + pid * stride_xn + h * stride_xh, mask=mask, other=0.0).to(
            tl.float32
        )
        q = _round_nearest(x * scale + offset)
        q = tl.minimum(tl.maximum(q, -128.0), 127.0)
        tl.store(
            out_ptr + pid * stride_on + h * stride_oh,
            q.to(tl.int8),
            mask=mask,
        )


@triton.jit
def _dynamic_quant_kernel(
    x_ptr,
    out_ptr,
    scale_out_ptr,
    N,
    H,
    stride_xn,
    stride_xh,
    stride_on,
    stride_oh,
    BLOCK_H: tl.constexpr,
):
    """Per-row dynamic int8: scale = amax(|x|)/127; out = round(x/scale)."""
    pid = tl.program_id(0)
    if pid >= N:
        return
    offs = tl.arange(0, BLOCK_H)
    absmax = 0.0
    for h0 in range(0, H, BLOCK_H):
        h = h0 + offs
        mask = h < H
        x = tl.load(x_ptr + pid * stride_xn + h * stride_xh, mask=mask, other=0.0).to(
            tl.float32
        )
        absmax = tl.maximum(absmax, tl.max(tl.where(mask, tl.abs(x), 0.0)))
    absmax = tl.maximum(absmax, 1e-10)
    scale = absmax / 127.0
    tl.store(scale_out_ptr + pid, scale)

    for h0 in range(0, H, BLOCK_H):
        h = h0 + offs
        mask = h < H
        x = tl.load(x_ptr + pid * stride_xn + h * stride_xh, mask=mask, other=0.0).to(
            tl.float32
        )
        q = _round_nearest(x / scale)
        q = tl.minimum(tl.maximum(q, -128.0), 127.0)
        tl.store(
            out_ptr + pid * stride_on + h * stride_oh,
            q.to(tl.int8),
            mask=mask,
        )


@triton.jit
def _apply_smooth_scale_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    N,
    H,
    stride_xn,
    stride_xh,
    stride_sn,
    stride_sh,
    stride_on,
    stride_oh,
    expert_ids_ptr,
    BROADCAST_SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """out[i] = x[i] * scale[0] (broadcast) or scale[expert_ids[i]]."""
    pid = tl.program_id(0)
    if pid >= N:
        return
    if BROADCAST_SCALE:
        srow = 0
    else:
        srow = tl.load(expert_ids_ptr + pid)
    offs = tl.arange(0, BLOCK_H)
    for h0 in range(0, H, BLOCK_H):
        h = h0 + offs
        mask = h < H
        x = tl.load(x_ptr + pid * stride_xn + h * stride_xh, mask=mask, other=0.0).to(
            tl.float32
        )
        s = tl.load(
            scale_ptr + srow * stride_sn + h * stride_sh, mask=mask, other=1.0
        ).to(tl.float32)
        tl.store(
            out_ptr + pid * stride_on + h * stride_oh,
            x * s,
            mask=mask,
        )


def _pow2_ge(x: int, lo: int = 32, hi: int = 1024) -> int:
    v = lo
    while v < x and v < hi:
        v *= 2
    return min(v, hi)


def _triton_gather_rows(src: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    n = int(idx.numel())
    h = int(src.shape[-1])
    dst = torch.empty((n, h), device=src.device, dtype=src.dtype)
    if n == 0:
        return dst
    bh = _pow2_ge(min(h, 256))
    _gather_rows_kernel[(n,)](
        src,
        idx.to(torch.int64).contiguous(),
        dst,
        n,
        h,
        src.stride(0),
        src.stride(1),
        dst.stride(0),
        dst.stride(1),
        BLOCK_H=bh,
    )
    return dst


def _triton_fill_gather_row_idx(
    sorted_row_idx: torch.Tensor, total: int, n_valid: int, device: torch.device
) -> torch.Tensor:
    out = torch.full((total,), -1, dtype=torch.int32, device=device)
    if n_valid <= 0:
        return out
    valid = sorted_row_idx[:n_valid].to(torch.int64).contiguous()
    _fill_gather_row_idx_i32[(n_valid,)](valid, out, n_valid)
    return out


def _triton_static_quant(
    x: torch.Tensor, scale: float, offset: float
) -> torch.Tensor:
    n, h = x.shape
    out = torch.empty((n, h), device=x.device, dtype=torch.int8)
    if n == 0:
        return out
    x_f = x.to(torch.float32).contiguous()
    bh = _pow2_ge(min(h, 256))
    _static_quant_kernel[(n,)](
        x_f,
        out,
        float(scale),
        float(offset),
        n,
        h,
        x_f.stride(0),
        x_f.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_H=bh,
    )
    return out


def _triton_apply_smooth_then_dynamic(
    x: torch.Tensor,
    smooth: Optional[torch.Tensor],
    expert_ids: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    n, h = x.shape
    device = x.device
    if n == 0:
        return (
            torch.empty((0, h), dtype=torch.int8, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
        )

    x_f = x.to(torch.float32).contiguous()
    if smooth is not None:
        smooth_f = smooth.to(torch.float32).contiguous()
        tmp = torch.empty_like(x_f)
        bh = _pow2_ge(min(h, 256))
        eids = (
            expert_ids.to(torch.int64).contiguous()
            if expert_ids is not None
            else torch.zeros(n, dtype=torch.int64, device=device)
        )
        _apply_smooth_scale_kernel[(n,)](
            x_f,
            smooth_f,
            tmp,
            n,
            h,
            x_f.stride(0),
            x_f.stride(1),
            smooth_f.stride(0),
            smooth_f.stride(1),
            tmp.stride(0),
            tmp.stride(1),
            eids,
            BROADCAST_SCALE=smooth_f.shape[0] == 1,
            BLOCK_H=bh,
        )
        x_f = tmp

    out = torch.empty((n, h), device=device, dtype=torch.int8)
    scale_out = torch.empty(n, device=device, dtype=torch.float32)
    bh = _pow2_ge(min(h, 256))
    _dynamic_quant_kernel[(n,)](
        x_f,
        out,
        scale_out,
        n,
        h,
        x_f.stride(0),
        x_f.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_H=bh,
    )
    return out, scale_out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def npu_moe_init_routing_v2_triton(
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
    quant_mode: int = -1,
    active_expert_range: Optional[List[int]] = None,
    row_idx_type: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Triton reference for ``torch_npu.npu_moe_init_routing_v2``.

    Signature matches ``npu_moe_init_routing_v2_pytorch``. Runs on CUDA/HCU.
    """
    del expert_capacity  # unused in dropless path

    if x.ndim != 2:
        raise ValueError(f"x must be 2D (NUM_ROWS, H), got shape {tuple(x.shape)}")
    if expert_idx.ndim != 2:
        raise ValueError(
            f"expert_idx must be 2D (NUM_ROWS, K), got shape {tuple(expert_idx.shape)}"
        )
    if x.shape[0] != expert_idx.shape[0]:
        raise ValueError(
            f"x and expert_idx row count mismatch: {x.shape[0]} vs {expert_idx.shape[0]}"
        )
    if drop_pad_mode != 0:
        raise NotImplementedError(
            "npu_moe_init_routing_v2_triton only supports drop_pad_mode=0 "
            "(dropless); drop_pad_mode=1 is not implemented"
        )
    if quant_mode not in (-1, 0, 1):
        raise NotImplementedError(
            f"npu_moe_init_routing_v2_triton unsupported quant_mode={quant_mode} "
            "(supports -1/0/1; MXFP8 modes 2/3 are not implemented)"
        )

    device = _device()
    x = _to_device(x, device)
    expert_idx = _to_device(expert_idx, device)
    scale = _to_device(scale, device)
    offset = _to_device(offset, device)

    num_rows, h = x.shape
    k = expert_idx.shape[-1]

    if expert_num is None or expert_num < 0:
        if expert_idx.numel() == 0:
            expert_num = 0
        else:
            nonneg = expert_idx[expert_idx >= 0]
            expert_num = int(nonneg.max().item()) + 1 if nonneg.numel() else 0

    if not active_expert_range:
        expert_start, expert_end = 0, expert_num
    else:
        if len(active_expert_range) != 2:
            raise ValueError("active_expert_range must be [expert_start, expert_end]")
        expert_start, expert_end = int(active_expert_range[0]), int(
            active_expert_range[1]
        )
    if expert_start < 0 or expert_end < expert_start:
        raise ValueError(
            f"invalid active_expert_range=[{expert_start}, {expert_end})"
        )
    if expert_end > expert_num:
        raise ValueError(
            f"active_expert_range end {expert_end} > expert_num {expert_num}"
        )
    range_len = expert_end - expert_start

    if quant_mode == 0:
        if scale is None or offset is None:
            raise ValueError("static quant_mode=0 requires scale and offset")
        if scale.numel() != 1 or offset.numel() != 1:
            raise ValueError(
                f"static quant_mode=0 expects scalar scale/offset (numel=1), "
                f"got scale.shape={tuple(scale.shape)}, offset.shape={tuple(offset.shape)}"
            )
    elif quant_mode == 1 and scale is not None:
        if scale.ndim != 2 or scale.shape[-1] != h:
            raise ValueError(
                f"dynamic quant_mode=1 smooth scale must be (1, H) or "
                f"(expert_end-expert_start, H) with H={h}, got {tuple(scale.shape)}"
            )
        if scale.shape[0] not in (1, range_len):
            raise ValueError(
                f"dynamic quant_mode=1 smooth scale rows must be 1 or "
                f"range_len={range_len}, got {scale.shape[0]}"
            )
    elif quant_mode == -1 and scale is not None:
        if scale.numel() != num_rows and scale.shape != (num_rows,):
            raise ValueError(
                f"non-quant scale must be (NUM_ROWS,) with NUM_ROWS={num_rows}, "
                f"got shape {tuple(scale.shape)}"
            )

    expert_idx_flat = expert_idx.reshape(-1).to(torch.int32)
    total = num_rows * k

    # Hot path (SGLang MoE): full expert range + active_num covers every slot.
    # Avoid ``in_range.sum().item()`` which forces a GPU→CPU sync every forward
    # and can stall the scheduler watchdog when the device queue is busy.
    if (
        expert_start == 0
        and expert_end == expert_num
        and active_num == total
    ):
        actual_expert_total_num = total
    else:
        in_range = (expert_idx_flat >= expert_start) & (expert_idx_flat < expert_end)
        actual_expert_total_num = int(in_range.sum().item())

    expert_idx_sort_key = torch.where(
        expert_idx_flat < expert_start,
        torch.full_like(expert_idx_flat, _INT32_MAX),
        expert_idx_flat,
    )
    sorted_row_idx = torch.argsort(expert_idx_sort_key, dim=-1, stable=True).to(
        torch.int64
    )
    sorted_expert_idx = expert_idx_flat[sorted_row_idx]

    if row_idx_type == 1:
        expanded_row_idx = sorted_row_idx.to(torch.int32)
    else:
        expanded_row_idx = _triton_fill_gather_row_idx(
            sorted_row_idx, total, actual_expert_total_num, device
        )

    expert_tokens_count: Optional[torch.Tensor] = None
    if expert_tokens_num_flag:
        if actual_expert_total_num > 0:
            ids = sorted_expert_idx[:actual_expert_total_num] - expert_start
            counts = torch.bincount(ids, minlength=range_len).to(torch.int64)
        else:
            counts = torch.zeros(range_len, dtype=torch.int64, device=device)

        if expert_tokens_num_type == 1:
            expert_tokens_count = counts
        elif expert_tokens_num_type == 0:
            expert_tokens_count = torch.cumsum(counts, dim=0)
        elif expert_tokens_num_type == 2:
            nonzero = counts > 0
            expert_ids = (
                torch.arange(range_len, device=device, dtype=torch.int64)[nonzero]
                + expert_start
            )
            kv = torch.stack([expert_ids, counts[nonzero]], dim=-1)
            if kv.shape[0] < expert_num:
                pad = torch.zeros(
                    (expert_num - kv.shape[0], 2), dtype=torch.int64, device=device
                )
                kv = torch.cat([kv, pad], dim=0)
            expert_tokens_count = kv
        else:
            raise ValueError(f"invalid expert_tokens_num_type={expert_tokens_num_type}")

    if active_num <= 0:
        active_num_eff = actual_expert_total_num
    else:
        active_num_eff = min(int(active_num), actual_expert_total_num)

    expanded_scale: Optional[torch.Tensor] = None
    if active_num_eff <= 0:
        if quant_mode in (0, 1) and x.dtype != torch.int8:
            expanded_x = torch.empty((0, h), dtype=torch.int8, device=device)
            if quant_mode == 1:
                expanded_scale = torch.empty(0, dtype=torch.float32, device=device)
        else:
            expanded_x = x.new_empty((0, h))
        return expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale

    token_ids = (sorted_row_idx[:active_num_eff] // k).to(torch.int64).contiguous()
    expanded_x = _triton_gather_rows(x, token_ids)

    if scale is not None and quant_mode == -1:
        expanded_scale = scale.reshape(-1)[token_ids].to(torch.float32)

    if quant_mode == -1:
        pass
    elif quant_mode == 0:
        if x.dtype == torch.int8:
            expanded_scale = None
        else:
            expanded_x = _triton_static_quant(
                expanded_x,
                float(scale.reshape(-1)[0].item()),
                float(offset.reshape(-1)[0].item()),
            )
            expanded_scale = None
    elif quant_mode == 1:
        if x.dtype == torch.int8:
            expanded_scale = None
        else:
            eids = None
            if scale is not None and scale.shape[0] != 1:
                eids = (sorted_expert_idx[:active_num_eff] - expert_start).to(
                    torch.int64
                )
            expanded_x, expanded_scale = _triton_apply_smooth_then_dynamic(
                expanded_x, scale, eids
            )

    return expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale
