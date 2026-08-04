# Copyright 2026 SGLang Team
#
# Triton / PyTorch fallbacks for Ascend NPU int8 ops when torch_npu is absent
# (e.g. Hygon HCU / ROCm):
#   torch.ops.npu.npu_dynamic_quant
#   torch.ops.npu.npu_quant_matmul  (W8A8 int8 + per-token act scale)
#
# Backend selection (env SGLANG_NPU_INT8_BACKEND):
#   triton - Triton int8 GEMM (default)
#   torch  - pure PyTorch reference
#   auto   - same as triton, with torch fallback on kernel failure

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_BACKEND: Optional[str] = None
_LOGGED_BACKEND = False


def _resolve_backend() -> str:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    requested = os.environ.get("SGLANG_NPU_INT8_BACKEND", "triton").lower().strip()
    if requested not in ("auto", "torch", "triton"):
        logger.warning(
            "Unknown SGLANG_NPU_INT8_BACKEND=%s, falling back to triton", requested
        )
        requested = "triton"
    # auto == triton with torch fallback inside _scaled_mm
    _BACKEND = "torch" if requested == "torch" else "triton"
    return _BACKEND


def _log_backend_once():
    global _LOGGED_BACKEND
    if _LOGGED_BACKEND:
        return
    _LOGGED_BACKEND = True
    logger.info(
        "npu int8 op fallback backend=%s "
        "(set SGLANG_NPU_INT8_BACKEND=triton|torch)",
        _resolve_backend(),
    )


def _torch_dynamic_quant(
    x: torch.Tensor,
    *,
    smooth_scales: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token int8: scale = absmax/127, y = round(x/scale).clamp(-128, 127)."""
    if smooth_scales is not None:
        x = x * smooth_scales.to(dtype=x.dtype, device=x.device)
    x_f = x.to(torch.float32)
    absmax = x_f.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    scale = absmax / 127.0
    y = torch.round(x_f / scale).clamp(-128, 127).to(torch.int8)
    return y, scale.squeeze(-1).contiguous()


def _torch_scaled_mm(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference: (x1 @ x2) * scale_a * scale_b (+ bias)."""
    out = torch.mm(x1.to(torch.float32), x2.to(torch.float32))
    out = out * scale_a.reshape(-1, 1).to(torch.float32) * scale_b.reshape(1, -1).to(
        torch.float32
    )
    out = out.to(out_dtype)
    if bias is not None:
        out = out + bias.to(dtype=out_dtype, device=out.device)
    return out


def _is_weak_contiguous(x: torch.Tensor) -> bool:
    strides = x.stride()
    sizes = x.shape
    is_not_transpose = strides[0] == 1 and (strides[1] >= max(1, sizes[0]))
    is_transpose = strides[1] == 1 and (strides[0] >= max(1, sizes[1]))
    return is_transpose or is_not_transpose


def _as_column_scale(scale: torch.Tensor, expected_len: int) -> torch.Tensor:
    if scale.dim() <= 1:
        return scale.reshape(-1, 1)
    if scale.dim() == 2:
        if scale.shape[1] == 1:
            return scale
        if scale.shape[0] == 1 and scale.shape[1] == expected_len:
            return scale.t()
    return scale


@triton.jit
def _scaled_mm_kernel(
    a_ptr,
    b_ptr,
    scale_a_ptr,
    scale_b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_SCALE_A: tl.constexpr,
    BLOCK_SIZE_SCALE_B: tl.constexpr,
):
    """int8 GEMM: C = (A @ B) * scale_a * scale_b^T (+ bias)."""
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    masks_am = offsets_am < M
    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    masks_bn = offsets_bn < N

    offsets_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    offsets_a = stride_am * offsets_am[:, None] + stride_ak * offsets_k[None, :]
    offsets_b = stride_bk * offsets_k[:, None] + stride_bn * offsets_bn[None, :]

    offsets_scale_am = (
        tl.arange(0, BLOCK_SIZE_SCALE_A)
        + (BLOCK_SIZE_SCALE_A > 1) * pid_m * BLOCK_SIZE_M
    )
    masks_scale_am = offsets_scale_am < M
    offsets_scale_bn = (
        tl.arange(0, BLOCK_SIZE_SCALE_B)
        + (BLOCK_SIZE_SCALE_B > 1) * pid_n * BLOCK_SIZE_N
    )
    masks_scale_bn = offsets_scale_bn < N

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b
    scale_a_ptrs = scale_a_ptr + offsets_scale_am
    scale_b_ptrs = scale_b_ptr + offsets_scale_bn

    for _k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        masks_k = offsets_k < K
        a = tl.load(
            a_ptrs, mask=masks_am[:, None] & masks_k[None, :], other=0
        ).to(tl.int8)
        b = tl.load(
            b_ptrs, mask=masks_k[:, None] & masks_bn[None, :], other=0
        ).to(tl.int8)
        accumulator += tl.dot(a, b, out_dtype=tl.int32)
        offsets_k += BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    masks_scale_a = masks_scale_am[:, None] & (tl.arange(0, 1) < 1)[:, None]
    scale_a = tl.load(scale_a_ptrs[:, None], masks_scale_a)
    scale_a = scale_a.broadcast_to((BLOCK_SIZE_M, 1))
    acc_f = scale_a * accumulator.to(tl.float32)

    masks_scale_b = masks_scale_bn[:, None] & (tl.arange(0, 1) < 1)[None, :]
    scale_b = tl.load(scale_b_ptrs[:, None], masks_scale_b)
    scale_b = scale_b.broadcast_to((BLOCK_SIZE_N, 1))
    acc_f = scale_b.T * acc_f

    c = acc_f.to(c_ptr.dtype.element_ty)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offsets_bn, mask=offsets_bn < N, other=0.0)
        c = c + bias.to(c.dtype)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def _triton_scaled_mm(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Triton int8 scaled mm. Conservative tiles for HCU/ROCm int8 MFMA."""
    if not input.is_contiguous():
        input = input.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    M, K = input.shape
    N = weight.shape[1]
    assert weight.shape[0] == K
    assert input.dtype == weight.dtype == torch.int8
    assert _is_weak_contiguous(input) and _is_weak_contiguous(weight)

    scale_a = _as_column_scale(scale_a, M).to(torch.float32).contiguous()
    scale_b = _as_column_scale(scale_b, N).to(torch.float32).contiguous()
    assert scale_a.shape[1] == 1 and (scale_a.shape[0] == 1 or scale_a.shape[0] == M)
    assert scale_b.shape[1] == 1 and (scale_b.shape[0] == 1 or scale_b.shape[0] == N)

    # Conservative tiles: large BLOCK_K has faulted on some Hygon/triton combos.
    next_power_of_2_M = max(16, triton.next_power_of_2(M))
    if next_power_of_2_M <= 32:
        tile_shape = (32, 64, 64)
    elif next_power_of_2_M <= 64:
        tile_shape = (64, 64, 64)
    else:
        tile_shape = (64, 128, 64)
    block_size_m, block_size_n, block_size_k = tile_shape

    has_scalar = lambda x: x.shape[0] == 1 and x.shape[1] == 1
    block_size_sa = 1 if has_scalar(scale_a) else block_size_m
    block_size_sb = 1 if has_scalar(scale_b) else block_size_n

    if bias is not None:
        bias = bias.contiguous()

    result = torch.empty((M, N), dtype=out_dtype, device=input.device)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _scaled_mm_kernel[grid](
        input,
        weight,
        scale_a,
        scale_b,
        result,
        bias if bias is not None else result,
        M,
        N,
        K,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        result.stride(0),
        result.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        BLOCK_SIZE_SCALE_A=block_size_sa,
        BLOCK_SIZE_SCALE_B=block_size_sb,
    )
    return result


def _scaled_mm(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    backend = _resolve_backend()
    _log_backend_once()

    if backend == "triton":
        try:
            return _triton_scaled_mm(x1, x2, scale_a, scale_b, out_dtype, bias)
        except Exception as e:
            logger.warning(
                "Triton npu_quant_matmul failed (%s); falling back to torch", e
            )
            return _torch_scaled_mm(x1, x2, scale_a, scale_b, out_dtype, bias)

    return _torch_scaled_mm(x1, x2, scale_a, scale_b, out_dtype, bias)


def npu_dynamic_quant(
    x: torch.Tensor,
    *,
    smooth_scales: Optional[torch.Tensor] = None,
    group_index: Optional[torch.Tensor] = None,
    dst_type: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fallback for ``torch.ops.npu.npu_dynamic_quant`` (per-token int8)."""
    if dst_type is not None and dst_type not in (torch.int8,):
        raise NotImplementedError(
            f"npu_dynamic_quant fallback only supports dst_type=int8, got {dst_type}"
        )
    if group_index is not None:
        raise NotImplementedError(
            "npu_dynamic_quant fallback does not support group_index (MoE smooth)"
        )

    backend = _resolve_backend()
    _log_backend_once()

    if backend == "torch":
        out = _torch_dynamic_quant(x, smooth_scales=smooth_scales)
        return out

    if smooth_scales is not None:
        x = x * smooth_scales.to(dtype=x.dtype, device=x.device)
    if not x.is_contiguous():
        x = x.contiguous()

    try:
        from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8

        x_q, scales = per_token_quant_int8(x, scale_dtype=torch.float32)
        if scales.dim() > 0 and scales.shape[-1] == 1:
            scales = scales.squeeze(-1)
        return x_q, scales.to(torch.float32)
    except Exception as e:
        logger.warning("per_token_quant_int8 failed (%s); using torch", e)
        out = _torch_dynamic_quant(x, smooth_scales=None)
        return out


def npu_quant_matmul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: torch.Tensor,
    *,
    offset: Optional[torch.Tensor] = None,
    pertoken_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = None,
    **kwargs,
) -> torch.Tensor:
    """Fallback for ``torch.ops.npu.npu_quant_matmul`` (int8 W8A8 path).

    Computes: out = (x1 @ x2) * pertoken_scale[:, None] * scale[None, :] (+ bias)
    with x1/x2 int8, weight layout [K, N] (same as Ascend after transpose).
    """
    if offset is not None:
        raise NotImplementedError("npu_quant_matmul fallback does not support offset")
    if x1.dtype != torch.int8 or x2.dtype != torch.int8:
        raise TypeError(
            f"npu_quant_matmul fallback expects int8 inputs, got {x1.dtype}, {x2.dtype}"
        )
    if kwargs:
        unsupported = {
            k
            for k in kwargs
            if k
            in (
                "x1_dtype",
                "x2_dtype",
                "group_sizes",
                "pertoken_scale_dtype",
            )
            and kwargs[k] is not None
        }
        if unsupported:
            raise NotImplementedError(
                f"npu_quant_matmul int8 fallback does not support {unsupported}"
            )

    out_dtype = output_dtype if output_dtype is not None else torch.bfloat16
    orig_shape = x1.shape
    if x1.dim() > 2:
        x1_2d = x1.reshape(-1, orig_shape[-1])
    elif x1.dim() == 1:
        x1_2d = x1.unsqueeze(0)
    else:
        x1_2d = x1

    if not x2.is_contiguous():
        x2 = x2.contiguous()

    M = x1_2d.shape[0]
    if pertoken_scale is None:
        scale_a = torch.ones((M,), dtype=torch.float32, device=x1.device)
    else:
        scale_a = pertoken_scale.flatten().to(dtype=torch.float32)
        if scale_a.numel() != M:
            raise ValueError(
                f"pertoken_scale numel ({scale_a.numel()}) != M ({M})"
            )

    scale_b = scale.flatten().to(dtype=torch.float32)
    if scale_b.numel() not in (1, x2.shape[1]):
        raise ValueError(
            f"weight scale numel ({scale_b.numel()}) incompatible with N={x2.shape[1]}"
        )

    out = _scaled_mm(x1_2d, x2, scale_a, scale_b, out_dtype, bias)

    if len(orig_shape) > 2:
        out = out.reshape(orig_shape[:-1] + (x2.shape[-1],))
    elif len(orig_shape) == 1:
        out = out.squeeze(0)
    return out
