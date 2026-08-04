# Copyright 2026 SGLang Team
#
# Triton fallback for ``torch.ops.npu.npu_dequant_swiglu_quant``.
#
# Covers the AscendTP MoE path used by ``NPUSwigluQuant``:
#   quant_mode=1 (dynamic per-token int8), activate_left=True
#   - float/bf16/fp16 x: SwiGLU then dynamic quant
#   - int32 x: dequant with weight_scale * activation_scale (+ bias), then same

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.srt.utils import is_hip

_is_hip = is_hip()


@triton.jit
def _dequant_swiglu_quant_kernel(
    x_ptr,
    y_q_ptr,
    y_scale_ptr,
    weight_scale_ptr,
    act_scale_ptr,
    bias_ptr,
    quant_scale_ptr,
    M,
    H,
    stride_xm,
    stride_xh,
    stride_ym,
    stride_yh,
    HAS_WEIGHT_SCALE: tl.constexpr,
    HAS_ACT_SCALE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_QUANT_SCALE: tl.constexpr,
    ACTIVATE_LEFT: tl.constexpr,
    IS_INT32_X: tl.constexpr,
    IS_HIP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fused (optional dequant) + SwiGLU + dynamic per-token int8 quant.

    x: [M, 2H], y_q: [M, H] int8, y_scale: [M] float32
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H

    left = tl.load(
        x_ptr + row * stride_xm + cols * stride_xh,
        mask=mask,
        other=0,
    )
    right = tl.load(
        x_ptr + row * stride_xm + (H + cols) * stride_xh,
        mask=mask,
        other=0,
    )

    if IS_INT32_X:
        left_f = left.to(tl.float32)
        right_f = right.to(tl.float32)
        if HAS_WEIGHT_SCALE:
            ws_l = tl.load(weight_scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            ws_r = tl.load(weight_scale_ptr + H + cols, mask=mask, other=0.0).to(
                tl.float32
            )
            left_f = left_f * ws_l
            right_f = right_f * ws_r
        if HAS_ACT_SCALE:
            as_ = tl.load(act_scale_ptr + row).to(tl.float32)
            left_f = left_f * as_
            right_f = right_f * as_
        if HAS_BIAS:
            b_l = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            b_r = tl.load(bias_ptr + H + cols, mask=mask, other=0.0).to(tl.float32)
            left_f = left_f + b_l
            right_f = right_f + b_r
    else:
        left_f = left.to(tl.float32)
        right_f = right.to(tl.float32)

    if ACTIVATE_LEFT:
        gate = left_f * tl.sigmoid(left_f)
        out = gate * right_f
    else:
        gate = right_f * tl.sigmoid(right_f)
        out = left_f * gate

    if HAS_QUANT_SCALE:
        qs = tl.load(quant_scale_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        out = out * qs

    absmax = tl.maximum(tl.max(tl.abs(out)), 1e-10)
    scale = absmax / 127.0
    x_q = out * (127.0 / absmax)
    if IS_HIP:
        x_q = libdevice.round(x_q).to(tl.int8)
    else:
        x_q = tl.extra.cuda.libdevice.round(x_q).to(tl.int8)

    tl.store(y_q_ptr + row * stride_ym + cols * stride_yh, x_q, mask=mask)
    tl.store(y_scale_ptr + row, scale)


def _torch_dequant_swiglu_quant(
    x: torch.Tensor,
    *,
    weight_scale: Optional[torch.Tensor],
    activation_scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    quant_scale: Optional[torch.Tensor],
    activate_left: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference for the fused op."""
    if x.dtype == torch.int32:
        x_f = x.to(torch.float32)
        if weight_scale is None or activation_scale is None:
            raise ValueError(
                "weight_scale and activation_scale are required for int32 input"
            )
        ws = weight_scale.to(torch.float32).reshape(-1)
        if ws.numel() != x.shape[-1]:
            raise ValueError(
                f"weight_scale numel ({ws.numel()}) != x.shape[-1] ({x.shape[-1]})"
            )
        x_f = x_f * ws
        as_ = activation_scale.to(torch.float32).reshape(-1, 1)
        x_f = x_f * as_
        if bias is not None:
            x_f = x_f + bias.to(torch.float32).reshape(-1)
    else:
        x_f = x.to(torch.float32)

    left, right = x_f.chunk(2, dim=-1)
    if activate_left:
        out = torch.nn.functional.silu(left) * right
    else:
        out = left * torch.nn.functional.silu(right)

    if quant_scale is not None:
        out = out * quant_scale.to(torch.float32).reshape(1, -1)

    absmax = out.abs().amax(dim=-1).clamp_min(1e-10)
    scale = absmax / 127.0
    y_q = torch.round(out / scale.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
    return y_q, scale.to(torch.float32)


def npu_dequant_swiglu_quant(
    x: torch.Tensor,
    *,
    weight_scale: Optional[torch.Tensor] = None,
    activation_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    quant_scale: Optional[torch.Tensor] = None,
    quant_offset: Optional[torch.Tensor] = None,
    group_index: Optional[torch.Tensor] = None,
    activate_left: bool = False,
    quant_mode: int = 0,
    swiglu_mode: int = 0,
    clamp_limit: float = 7.0,
    glu_alpha: float = 1.702,
    glu_bias: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton fallback for ``npu_dequant_swiglu_quant``.

    Supported configuration (AscendTP ``NPUSwigluQuant`` path):
      ``quant_mode=1``, ``swiglu_mode=0``, no ``group_index`` / ``quant_offset``.
    """
    if quant_mode != 1:
        raise NotImplementedError(
            f"npu_dequant_swiglu_quant triton fallback only supports "
            f"quant_mode=1 (dynamic), got {quant_mode}"
        )
    if swiglu_mode != 0:
        raise NotImplementedError(
            f"npu_dequant_swiglu_quant triton fallback only supports "
            f"swiglu_mode=0, got {swiglu_mode}"
        )
    if group_index is not None:
        raise NotImplementedError(
            "npu_dequant_swiglu_quant triton fallback does not support group_index"
        )
    if quant_offset is not None:
        raise NotImplementedError(
            "npu_dequant_swiglu_quant triton fallback does not support quant_offset"
        )
    # Silence unused variant-SwiGLU knobs when swiglu_mode=0.
    _ = (clamp_limit, glu_alpha, glu_bias)

    if x.dim() != 2:
        raise ValueError(f"x must be 2D [M, 2H], got {tuple(x.shape)}")
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"x last dim must be even, got {x.shape[-1]}")

    if not x.is_contiguous():
        x = x.contiguous()

    m, two_h = x.shape
    h = two_h // 2
    if m == 0:
        return (
            torch.empty((0, h), dtype=torch.int8, device=x.device),
            torch.empty((0,), dtype=torch.float32, device=x.device),
        )

    is_int32 = x.dtype == torch.int32
    if is_int32 and (weight_scale is None or activation_scale is None):
        raise ValueError(
            "weight_scale and activation_scale are required when x is int32"
        )

    if weight_scale is not None:
        weight_scale = weight_scale.to(torch.float32).reshape(-1).contiguous()
        if weight_scale.numel() != two_h:
            raise ValueError(
                f"weight_scale numel ({weight_scale.numel()}) != 2H ({two_h})"
            )
    if activation_scale is not None:
        activation_scale = (
            activation_scale.to(torch.float32).reshape(-1).contiguous()
        )
        if activation_scale.numel() != m:
            raise ValueError(
                f"activation_scale numel ({activation_scale.numel()}) != M ({m})"
            )
    if bias is not None:
        bias = bias.to(torch.float32).reshape(-1).contiguous()
        if bias.numel() != two_h:
            raise ValueError(f"bias numel ({bias.numel()}) != 2H ({two_h})")
    if quant_scale is not None:
        quant_scale = quant_scale.to(torch.float32).reshape(-1).contiguous()
        if quant_scale.numel() != h:
            # Allow [1, H] / [group, H] with a single group.
            if quant_scale.numel() == 1:
                quant_scale = quant_scale.expand(h).contiguous()
            elif quant_scale.numel() % h == 0 and quant_scale.numel() // h == 1:
                quant_scale = quant_scale.reshape(-1)[:h].contiguous()
            else:
                raise ValueError(
                    f"quant_scale numel ({quant_scale.numel()}) incompatible with H={h}"
                )

    y_q = torch.empty((m, h), dtype=torch.int8, device=x.device)
    y_scale = torch.empty((m,), dtype=torch.float32, device=x.device)

    block = triton.next_power_of_2(h)
    num_warps = min(max(block // 256, 1), 8)

    try:
        _dequant_swiglu_quant_kernel[(m,)](
            x,
            y_q,
            y_scale,
            weight_scale if weight_scale is not None else y_scale,
            activation_scale if activation_scale is not None else y_scale,
            bias if bias is not None else y_scale,
            quant_scale if quant_scale is not None else y_scale,
            m,
            h,
            x.stride(0),
            x.stride(1),
            y_q.stride(0),
            y_q.stride(1),
            HAS_WEIGHT_SCALE=weight_scale is not None,
            HAS_ACT_SCALE=activation_scale is not None,
            HAS_BIAS=bias is not None,
            HAS_QUANT_SCALE=quant_scale is not None,
            ACTIVATE_LEFT=activate_left,
            IS_INT32_X=is_int32,
            IS_HIP=_is_hip,
            BLOCK=block,
            num_warps=num_warps,
            num_stages=2,
        )
        return y_q, y_scale
    except Exception:
        return _torch_dequant_swiglu_quant(
            x,
            weight_scale=weight_scale,
            activation_scale=activation_scale,
            bias=bias,
            quant_scale=quant_scale,
            activate_left=activate_left,
        )
