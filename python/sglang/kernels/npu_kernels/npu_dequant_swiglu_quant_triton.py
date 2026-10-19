"""Triton implementation of ``torch_npu.npu_dequant_swiglu_quant`` (精简版).

与 cann-bench ``golden.py`` 对齐：去掉 group_index / swiglu_mode / bias /
quant_offset，固定 quant_mode=1（动态 per-token int8）。

  npu_dequant_swiglu_quant(x, weight_scale, activation_scale, quant_scale,
                           activate_left) -> (y, scale)

x 支持 int32（需 weight_scale + activation_scale）和 bfloat16 / float16
（weight_scale / activation_scale 必须为 None）。
在 CUDA / Hygon HCU（``cuda`` device）上运行。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

_INT8_MAX = 127.0
_EPS = 1e-12


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("npu_dequant_swiglu_quant requires a CUDA/HCU device")


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
def _silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _dequant_swiglu_quant_kernel(
    x_ptr,
    weight_scale_ptr,
    activation_scale_ptr,
    quant_scale_ptr,
    out_ptr,
    scale_ptr,
    M,
    H,
    stride_xm,
    stride_xh,
    HAS_DEQUANT: tl.constexpr,
    HAS_QUANT_SCALE: tl.constexpr,
    ACTIVATE_LEFT: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """One program per token row: dequant → SwiGLU → optional smooth → dynamic quant."""
    pid = tl.program_id(0)
    if pid >= M:
        return

    row_x = x_ptr + pid * stride_xm
    act = 1.0
    if HAS_DEQUANT:
        act = tl.load(activation_scale_ptr + pid).to(tl.float32)

    offs = tl.arange(0, BLOCK_H)
    absmax = 0.0

    # Pass 1: absmax over SwiGLU(+quant_scale) output
    for h0 in range(0, H, BLOCK_H):
        h = h0 + offs
        mask = h < H

        a = tl.load(row_x + h * stride_xh, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(row_x + (H + h) * stride_xh, mask=mask, other=0.0).to(tl.float32)

        if HAS_DEQUANT:
            wa = tl.load(weight_scale_ptr + h, mask=mask, other=0.0).to(tl.float32)
            wb = tl.load(weight_scale_ptr + H + h, mask=mask, other=0.0).to(tl.float32)
            a = a * wa * act
            b = b * wb * act

        if ACTIVATE_LEFT:
            out = _silu(a) * b
        else:
            out = _silu(b) * a

        if HAS_QUANT_SCALE:
            qs = tl.load(quant_scale_ptr + h, mask=mask, other=1.0).to(tl.float32)
            out = out * qs

        absmax = tl.maximum(absmax, tl.max(tl.where(mask, tl.abs(out), 0.0)))

    absmax = tl.maximum(absmax, 1e-12)
    scale = absmax / 127.0
    tl.store(scale_ptr + pid, scale)

    # Pass 2: quantize
    row_o = out_ptr + pid * H
    for h0 in range(0, H, BLOCK_H):
        h = h0 + offs
        mask = h < H

        a = tl.load(row_x + h * stride_xh, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(row_x + (H + h) * stride_xh, mask=mask, other=0.0).to(tl.float32)

        if HAS_DEQUANT:
            wa = tl.load(weight_scale_ptr + h, mask=mask, other=0.0).to(tl.float32)
            wb = tl.load(weight_scale_ptr + H + h, mask=mask, other=0.0).to(tl.float32)
            a = a * wa * act
            b = b * wb * act

        if ACTIVATE_LEFT:
            out = _silu(a) * b
        else:
            out = _silu(b) * a

        if HAS_QUANT_SCALE:
            qs = tl.load(quant_scale_ptr + h, mask=mask, other=1.0).to(tl.float32)
            out = out * qs

        q = _round_nearest(out / scale)
        q = tl.minimum(tl.maximum(q, -128.0), 127.0)
        tl.store(row_o + h, q.to(tl.int8), mask=mask)


def npu_dequant_swiglu_quant_triton(
    x: torch.Tensor,
    weight_scale: Optional[torch.Tensor] = None,
    activation_scale: Optional[torch.Tensor] = None,
    quant_scale: Optional[torch.Tensor] = None,
    activate_left: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton fused dequant + SwiGLU + dynamic per-token int8 quant.

    Args:
        x: [TokensNum, 2H] int32 / bfloat16 / float16
        weight_scale: [1, 2H] float32；x=int32 时必须；fp16/bf16 时必须 None
        activation_scale: [TokensNum] 或 [TokensNum, 1] float32；规则同上
        quant_scale: [1, H] float32；smooth quant（可选）
        activate_left: False → SiLU(B)*A；True → SiLU(A)*B

    Returns:
        y: [TokensNum, H] int8
        scale: [TokensNum] float32
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D [TokensNum, 2H], got shape {tuple(x.shape)}")
    tokens, two_h = x.shape
    if two_h % 2 != 0:
        raise ValueError(f"x last dim must be even, got {two_h}")
    h = two_h // 2

    if x.dtype == torch.int32:
        if weight_scale is None or activation_scale is None:
            raise ValueError("x=int32 requires weight_scale and activation_scale")
    elif x.dtype in (torch.bfloat16, torch.float16):
        if weight_scale is not None or activation_scale is not None:
            raise ValueError(
                f"x={x.dtype} requires weight_scale / activation_scale to be None"
            )
    else:
        raise TypeError(f"unsupported x dtype: {x.dtype}")

    device = _device()
    x = _to_device(x, device)
    weight_scale = _to_device(weight_scale, device)
    activation_scale = _to_device(activation_scale, device)
    quant_scale = _to_device(quant_scale, device)

    has_dequant = x.dtype == torch.int32
    null_f = torch.empty(0, device=device, dtype=torch.float32)

    if has_dequant:
        ws = weight_scale.to(torch.float32).reshape(-1).contiguous()
        if ws.numel() != two_h:
            raise ValueError(
                f"weight_scale must have {two_h} elements, got {ws.numel()}"
            )
        act = activation_scale.to(torch.float32).reshape(-1).contiguous()
        if act.numel() != tokens:
            raise ValueError(
                f"activation_scale must have {tokens} elements, got {act.numel()}"
            )
    else:
        ws = null_f
        act = null_f

    has_qs = quant_scale is not None
    if has_qs:
        qs = quant_scale.to(torch.float32).reshape(-1).contiguous()
        if qs.numel() != h:
            raise ValueError(f"quant_scale must have {h} elements, got {qs.numel()}")
    else:
        qs = null_f

    y = torch.empty((tokens, h), device=device, dtype=torch.int8)
    scale = torch.empty((tokens,), device=device, dtype=torch.float32)

    if tokens > 0:
        bh = _pow2_ge(min(h, 1024))
        _dequant_swiglu_quant_kernel[(tokens,)](
            x,
            ws,
            act,
            qs,
            y,
            scale,
            tokens,
            h,
            x.stride(0),
            x.stride(1),
            HAS_DEQUANT=has_dequant,
            HAS_QUANT_SCALE=has_qs,
            ACTIVATE_LEFT=activate_left,
            BLOCK_H=bh,
        )

    return y, scale


def dequant_swiglu_quant_golden(
    x: torch.Tensor,
    weight_scale: Optional[torch.Tensor] = None,
    activation_scale: Optional[torch.Tensor] = None,
    quant_scale: Optional[torch.Tensor] = None,
    activate_left: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CPU/PyTorch golden matching cann-bench ``golden.py``."""
    if x.dtype == torch.int32:
        assert weight_scale is not None and activation_scale is not None
        dequant_out = x.float() * weight_scale.float().reshape(1, -1)
        dequant_out = dequant_out * activation_scale.float().reshape(-1).unsqueeze(-1)
    elif x.dtype in (torch.bfloat16, torch.float16):
        assert weight_scale is None and activation_scale is None
        dequant_out = x.float()
    else:
        dequant_out = x.float()

    last_dim = dequant_out.shape[-1]
    assert last_dim % 2 == 0
    half = last_dim // 2
    a = dequant_out[..., :half]
    b = dequant_out[..., half:]
    silu = torch.nn.functional.silu
    swiglu_out = silu(a) * b if activate_left else silu(b) * a

    if quant_scale is not None:
        swiglu_out = swiglu_out * quant_scale.float().reshape(1, -1)

    max_per_row = swiglu_out.abs().amax(dim=-1)
    s = (max_per_row.float() / _INT8_MAX).clamp_min(_EPS)
    y = torch.clamp((swiglu_out.float() / s.unsqueeze(-1)).round(), -128, 127).to(
        torch.int8
    )
    return y, s.to(torch.float32)

