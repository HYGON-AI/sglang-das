"""Triton implementation of ``torch_npu.npu_dynamic_quant``.

Per-token symmetric dynamic quantization (doc)::

    scale = rowMax(abs(x [* smooth])) / DTYPE_MAX
    y     = round(x [* smooth] / scale)

Supports optional ``smooth_scales`` and MoE ``group_index``.
``dst_type`` currently supports ``int8`` only (default).
Runs on CUDA / Hygon HCU via the ``cuda`` device.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

_INT8_MAX = 127.0


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("npu_dynamic_quant_triton requires a CUDA/HCU device")


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
def _round_nearest(x):
    """Round half away from zero (HCU Triton has no ``tl.math.rint``)."""
    return tl.where(x >= 0, tl.math.floor(x + 0.5), tl.math.ceil(x - 0.5))


@triton.jit
def _dynamic_quant_int8_kernel(
    x_ptr,
    smooth_ptr,  # null or [K] broadcast / [M,K] per-token after expand
    out_ptr,
    scale_ptr,
    M,
    K,
    stride_xm,
    stride_xk,
    stride_sm,
    stride_sk,
    HAS_SMOOTH: tl.constexpr,
    SMOOTH_PER_TOKEN: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One program per token row."""
    pid = tl.program_id(0)
    if pid >= M:
        return

    offs = tl.arange(0, BLOCK_K)
    absmax = 0.0
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs
        mask = k < K
        x = tl.load(x_ptr + pid * stride_xm + k * stride_xk, mask=mask, other=0.0).to(
            tl.float32
        )
        if HAS_SMOOTH:
            if SMOOTH_PER_TOKEN:
                s = tl.load(
                    smooth_ptr + pid * stride_sm + k * stride_sk, mask=mask, other=1.0
                ).to(tl.float32)
            else:
                s = tl.load(smooth_ptr + k * stride_sk, mask=mask, other=1.0).to(
                    tl.float32
                )
            x = x * s
        absmax = tl.maximum(absmax, tl.max(tl.where(mask, tl.abs(x), 0.0)))

    absmax = tl.maximum(absmax, 1e-12)
    scale = absmax / 127.0
    tl.store(scale_ptr + pid, scale)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs
        mask = k < K
        x = tl.load(x_ptr + pid * stride_xm + k * stride_xk, mask=mask, other=0.0).to(
            tl.float32
        )
        if HAS_SMOOTH:
            if SMOOTH_PER_TOKEN:
                s = tl.load(
                    smooth_ptr + pid * stride_sm + k * stride_sk, mask=mask, other=1.0
                ).to(tl.float32)
            else:
                s = tl.load(smooth_ptr + k * stride_sk, mask=mask, other=1.0).to(
                    tl.float32
                )
            x = x * s
        q = _round_nearest(x / scale)
        q = tl.minimum(tl.maximum(q, -128.0), 127.0)
        tl.store(out_ptr + pid * K + k, q.to(tl.int8), mask=mask)


def _expand_group_smooth(
    smooth_scales: torch.Tensor, group_index: torch.Tensor, m: int, k: int
) -> torch.Tensor:
    """Expand MoE smooth_scales [G,K] by cumulative ``group_index`` → [M,K]."""
    gi = group_index.to(torch.int64).reshape(-1)
    if gi.numel() != smooth_scales.shape[0]:
        raise ValueError("group_index length must equal smooth_scales.shape[0]")
    if int(gi[-1].item()) != m:
        raise ValueError(
            f"group_index[-1] must equal token count {m}, got {int(gi[-1].item())}"
        )
    out = torch.empty((m, k), device=smooth_scales.device, dtype=torch.float32)
    prev = 0
    smooth_f = smooth_scales.to(torch.float32)
    for i in range(gi.numel()):
        end = int(gi[i].item())
        if end < prev:
            raise ValueError("group_index must be monotonically non-decreasing")
        if end > prev:
            out[prev:end] = smooth_f[i].reshape(1, -1)
        prev = end
    return out.contiguous()


def npu_dynamic_quant_triton(
    x: torch.Tensor,
    *,
    smooth_scales: Optional[torch.Tensor] = None,
    group_index: Optional[torch.Tensor] = None,
    dst_type: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton reference for ``torch_npu.npu_dynamic_quant``.

    Returns ``(y, scale)`` with ``y`` int8 (same shape as ``x``) and
    ``scale`` float32 with shape ``x.shape[:-1]``.
    """
    if x.ndim < 2:
        raise ValueError("npu_dynamic_quant_triton requires x.ndim >= 2")
    if dst_type is None:
        dst_type = torch.int8
    if dst_type != torch.int8:
        raise NotImplementedError(
            f"npu_dynamic_quant_triton only supports dst_type=int8, got {dst_type}"
        )

    device = _device()
    x = _to_device(x, device)
    smooth_scales = _to_device(smooth_scales, device)
    group_index = _to_device(group_index, device)

    leading = x.shape[:-1]
    k = x.shape[-1]
    m = x.numel() // k
    x_2d = x.to(torch.float32).reshape(m, k).contiguous()

    null_f = torch.empty(0, device=device, dtype=torch.float32)
    has_smooth = False
    smooth_per_token = False
    smooth_work = null_f
    stride_sm = 0
    stride_sk = 0

    if smooth_scales is None:
        if group_index is not None:
            raise ValueError("group_index requires smooth_scales")
    elif group_index is None:
        if smooth_scales.ndim != 1 or smooth_scales.numel() != k:
            raise ValueError(
                f"smooth_scales must be 1D with {k} elements, "
                f"got shape {tuple(smooth_scales.shape)}"
            )
        has_smooth = True
        smooth_per_token = False
        smooth_work = smooth_scales.to(torch.float32).reshape(-1).contiguous()
        stride_sk = smooth_work.stride(0)
    else:
        if smooth_scales.ndim != 2 or smooth_scales.shape[-1] != k:
            raise ValueError(
                f"with group_index, smooth_scales must be 2D (*, {k}), "
                f"got shape {tuple(smooth_scales.shape)}"
            )
        has_smooth = True
        smooth_per_token = True
        smooth_work = _expand_group_smooth(smooth_scales, group_index, m, k)
        stride_sm = smooth_work.stride(0)
        stride_sk = smooth_work.stride(1)

    y_2d = torch.empty((m, k), device=device, dtype=torch.int8)
    scale_flat = torch.empty((m,), device=device, dtype=torch.float32)

    if m > 0:
        bk = _pow2_ge(min(k, 1024))
        _dynamic_quant_int8_kernel[(m,)](
            x_2d,
            smooth_work,
            y_2d,
            scale_flat,
            m,
            k,
            x_2d.stride(0),
            x_2d.stride(1),
            stride_sm,
            stride_sk,
            HAS_SMOOTH=has_smooth,
            SMOOTH_PER_TOKEN=smooth_per_token,
            BLOCK_K=bk,
        )

    return y_2d.reshape(x.shape), scale_flat.reshape(leading)


def dynamic_quant(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token symmetric dynamic quant (cann-bench golden).

    Formula (axis=-1, dtype_max=127 → int8)::

        scale = row_max(abs(x)).clamp_min(1e-12) / 127
        y     = round(x / scale).clamp(-128, 127).to(int8)
    """
    x_compute = x.to(torch.float32)
    abs_max = torch.max(torch.abs(x_compute), dim=-1, keepdim=True)[0]
    scale_out = abs_max.clamp(min=1e-12) / 127.0
    y = torch.clamp(torch.round(x_compute / scale_out), -128, 127).to(torch.int8)
    scale = scale_out.squeeze(-1).to(torch.float32)
    return y, scale
