# Copyright 2026 SGLang Team
#
# Triton fallback for ``torch.ops.npu.npu_grouped_matmul``.
#
# Covers the AscendTP MoE path used by ``GroupedMatmul``:
#   split_item=2, group_type=0, group_list_type in {0, 1}
#   - BF16 / FP16 unquant:  y = x @ W[e]
#   - W8A8:                 y = (x_i8 @ W_i8[e]) * per_token_scale * scale[e] (+ bias)
#   - W4A8 (int4 packed):   unpack nibbles then same as W8A8
#   - W4A16 antiquant:      y = x @ dequant(W[e], antiquant_scale, antiquant_offset)

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
import triton
import triton.language as tl


def _as_tensor_list(
    value: Optional[Union[torch.Tensor, Sequence[Optional[torch.Tensor]]]],
) -> Optional[List[torch.Tensor]]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return [value]
    out = [v for v in value if v is not None]
    return out or None


def _unwrap_single(tensors: Optional[List[torch.Tensor]], name: str) -> Optional[torch.Tensor]:
    if tensors is None:
        return None
    if len(tensors) != 1:
        raise NotImplementedError(
            f"npu_grouped_matmul triton fallback expects a single {name} tensor, "
            f"got {len(tensors)}"
        )
    return tensors[0]


def _counts_and_offsets(
    group_list: torch.Tensor,
    group_list_type: int,
    num_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (counts[E], offsets[E+1]) with offsets exclusive cumsum of counts."""
    group_list = group_list.to(torch.int64).reshape(-1)
    if group_list_type == 1:
        counts = group_list
    elif group_list_type == 0:
        counts = torch.empty_like(group_list)
        counts[0] = group_list[0]
        if group_list.numel() > 1:
            counts[1:] = group_list[1:] - group_list[:-1]
    else:
        raise NotImplementedError(
            f"npu_grouped_matmul triton fallback only supports "
            f"group_list_type in {{0, 1}}, got {group_list_type}"
        )

    if int(counts.sum().item()) != num_tokens:
        raise ValueError(
            f"group_list token sum ({int(counts.sum().item())}) != "
            f"x.shape[0] ({num_tokens})"
        )

    offsets = torch.zeros(
        (counts.numel() + 1,),
        dtype=torch.int64,
        device=counts.device,
    )
    offsets[1:] = torch.cumsum(counts, dim=0)
    return counts, offsets


def _decode_ascend_int64_scale(scale: torch.Tensor) -> torch.Tensor:
    """Recover float32 scales packed as zero-extended uint32 bits in int64."""
    # Ascend stores float32 bit patterns in the low 32 bits of int64.
    bits = scale.to(torch.int64) & 0xFFFFFFFF
    return bits.to(torch.int32).view(torch.float32)


def _normalize_weight_scale(
    scale: Optional[torch.Tensor],
    num_experts: int,
    n: int,
) -> Optional[torch.Tensor]:
    if scale is None:
        return None
    if scale.dtype in (torch.int64, torch.int32):
        scale = _decode_ascend_int64_scale(scale)
    scale = scale.to(torch.float32).reshape(num_experts, -1)
    if scale.shape[-1] == 1:
        scale = scale.expand(num_experts, n).contiguous()
    elif scale.shape[-1] != n:
        raise ValueError(
            f"weight scale last dim ({scale.shape[-1]}) incompatible with N={n}"
        )
    return scale.contiguous()


def _unpack_int4_weight(weight: torch.Tensor) -> torch.Tensor:
    """Unpack Ascend-style int4 packed in int8/int32 to signed int8 [E, K, N]."""
    if weight.dtype == torch.int32:
        weight = weight.view(torch.int8)
    if weight.dtype != torch.int8:
        raise TypeError(f"int4 packed weight must be int8/int32, got {weight.dtype}")

    # Pack order from _pack_int4: byte = (odd << 4) | (even & 0xF)
    w = weight.to(torch.int16)
    even = w & 0x0F
    odd = (w >> 4) & 0x0F
    even = torch.where(even >= 8, even - 16, even)
    odd = torch.where(odd >= 8, odd - 16, odd)
    return torch.stack((even, odd), dim=-1).reshape(*weight.shape[:-1], -1).to(
        torch.int8
    ).contiguous()


def _prepare_weight(
    weight: torch.Tensor,
    *,
    antiquant: bool,
) -> Tuple[torch.Tensor, bool]:
    """Return (weight[E,K,N], is_int4_unpacked)."""
    if weight.dim() != 3:
        raise ValueError(f"weight must be 3D [E, K, N], got {tuple(weight.shape)}")

    if weight.dtype == torch.int32 or (
        weight.dtype == torch.int8 and antiquant
    ):
        # W4A8 stores int4 as int8 then views as int32; W4A16 may keep int32/int8.
        return _unpack_int4_weight(weight), True

    if weight.dtype == torch.int8 and not weight.is_contiguous():
        weight = weight.contiguous()
    return weight, False


@triton.jit
def _grouped_matmul_fp_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    offsets_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Per-expert FP grouped GEMM: C[m0:m1] = A[m0:m1] @ B[e]."""
    e = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    start = tl.load(offsets_ptr + e)
    end = tl.load(offsets_ptr + e + 1)
    token_m = end - start
    if token_m <= 0:
        return

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = rm < token_m
    mask_n = rn < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        mask_k = rk < K
        a = tl.load(
            a_ptr + (start + rm)[:, None] * stride_am + rk[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            b_ptr
            + e * stride_be
            + rk[:, None] * stride_bk
            + rn[None, :] * stride_bn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(a, b)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + e * N + rn, mask=mask_n, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    out = acc.to(c_ptr.dtype.element_ty)
    tl.store(
        c_ptr + (start + rm)[:, None] * stride_cm + rn[None, :] * stride_cn,
        out,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _grouped_matmul_int8_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    bias_ptr,
    offsets_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    HAS_A_SCALE: tl.constexpr,
    HAS_B_SCALE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Per-expert int8 grouped GEMM with optional per-token / per-channel scales."""
    e = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    start = tl.load(offsets_ptr + e)
    end = tl.load(offsets_ptr + e + 1)
    token_m = end - start
    if token_m <= 0:
        return

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = rm < token_m
    mask_n = rn < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        mask_k = rk < K
        a = tl.load(
            a_ptr + (start + rm)[:, None] * stride_am + rk[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0,
        ).to(tl.int8)
        b = tl.load(
            b_ptr
            + e * stride_be
            + rk[:, None] * stride_bk
            + rn[None, :] * stride_bn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0,
        ).to(tl.int8)
        acc += tl.dot(a, b, out_dtype=tl.int32)

    out = acc.to(tl.float32)
    if HAS_A_SCALE:
        a_scale = tl.load(a_scale_ptr + start + rm, mask=mask_m, other=0.0).to(
            tl.float32
        )
        out = out * a_scale[:, None]
    if HAS_B_SCALE:
        b_scale = tl.load(b_scale_ptr + e * N + rn, mask=mask_n, other=0.0).to(
            tl.float32
        )
        out = out * b_scale[None, :]
    if HAS_BIAS:
        bias = tl.load(bias_ptr + e * N + rn, mask=mask_n, other=0.0).to(tl.float32)
        out = out + bias[None, :]

    tl.store(
        c_ptr + (start + rm)[:, None] * stride_cm + rn[None, :] * stride_cn,
        out.to(c_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _launch_fp_grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    offsets: torch.Tensor,
    output_dtype: torch.dtype,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    m, k = x.shape
    e, k_w, n = weight.shape
    assert k == k_w
    out = torch.empty((m, n), dtype=output_dtype, device=x.device)

    block_m, block_n, block_k = 64, 64, 64
    max_tokens = int((offsets[1:] - offsets[:-1]).max().item()) if e > 0 else 0
    if max_tokens == 0:
        return out

    grid = (
        e,
        triton.cdiv(max_tokens, block_m),
        triton.cdiv(n, block_n),
    )
    _grouped_matmul_fp_kernel[grid](
        x,
        weight,
        out,
        bias if bias is not None else out,
        offsets,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        out.stride(0),
        out.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return out


def _launch_int8_grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    offsets: torch.Tensor,
    output_dtype: torch.dtype,
    per_token_scale: Optional[torch.Tensor],
    weight_scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    m, k = x.shape
    e, k_w, n = weight.shape
    assert k == k_w
    out = torch.empty((m, n), dtype=output_dtype, device=x.device)

    if per_token_scale is not None:
        per_token_scale = per_token_scale.reshape(-1).to(torch.float32).contiguous()
        if per_token_scale.numel() != m:
            raise ValueError(
                f"per_token_scale numel ({per_token_scale.numel()}) != M ({m})"
            )
    weight_scale = _normalize_weight_scale(weight_scale, e, n)
    if bias is not None:
        bias = bias.to(torch.float32).reshape(e, n).contiguous()

    block_m, block_n, block_k = 64, 64, 64
    max_tokens = int((offsets[1:] - offsets[:-1]).max().item()) if e > 0 else 0
    if max_tokens == 0:
        return out

    grid = (
        e,
        triton.cdiv(max_tokens, block_m),
        triton.cdiv(n, block_n),
    )
    _grouped_matmul_int8_kernel[grid](
        x,
        weight,
        out,
        per_token_scale if per_token_scale is not None else out,
        weight_scale if weight_scale is not None else out,
        bias if bias is not None else out,
        offsets,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        out.stride(0),
        out.stride(1),
        HAS_A_SCALE=per_token_scale is not None,
        HAS_B_SCALE=weight_scale is not None,
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return out


def _dequant_weight(
    weight_i8: torch.Tensor,
    antiquant_scale: torch.Tensor,
    antiquant_offset: Optional[torch.Tensor],
) -> torch.Tensor:
    """Dequant int4/int8 weights to float for WNA16 path."""
    e, k, n = weight_i8.shape
    scale = antiquant_scale.to(torch.float32)
    if scale.dim() == 3 and scale.shape[1] == 1:
        scale = scale.squeeze(1)
    scale = scale.reshape(e, -1)
    if scale.shape[-1] == 1:
        scale = scale.expand(e, n)
    elif scale.shape[-1] != n:
        raise ValueError(
            f"antiquant_scale last dim ({scale.shape[-1]}) incompatible with N={n}"
        )

    w = weight_i8.to(torch.float32)
    if antiquant_offset is not None:
        offset = antiquant_offset.to(torch.float32)
        if offset.dim() == 3 and offset.shape[1] == 1:
            offset = offset.squeeze(1)
        offset = offset.reshape(e, -1)
        if offset.shape[-1] == 1:
            offset = offset.expand(e, n)
        w = w - offset.reshape(e, 1, n)
    return (w * scale.reshape(e, 1, n)).contiguous()


def npu_grouped_matmul(
    x: Union[torch.Tensor, Sequence[torch.Tensor]],
    weight: Union[torch.Tensor, Sequence[torch.Tensor]],
    *,
    bias: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    scale: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    offset: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    antiquant_scale: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    antiquant_offset: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    per_token_scale: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    group_list: Optional[torch.Tensor] = None,
    activation_input=None,
    activation_quant_scale=None,
    activation_quant_offset=None,
    split_item: int = 0,
    group_type: Optional[int] = None,
    group_list_type: int = 0,
    act_type: int = 0,
    output_dtype: Optional[torch.dtype] = None,
    tuning_config=None,
) -> List[torch.Tensor]:
    """Triton fallback for ``torch.ops.npu.npu_grouped_matmul`` (MoE path)."""
    if split_item not in (2, 3):
        raise NotImplementedError(
            f"npu_grouped_matmul triton fallback only supports split_item=2/3, "
            f"got {split_item}"
        )
    if group_type not in (None, 0):
        raise NotImplementedError(
            f"npu_grouped_matmul triton fallback only supports group_type=0, "
            f"got {group_type}"
        )
    if act_type not in (0, None):
        raise NotImplementedError(
            f"npu_grouped_matmul triton fallback does not support act_type={act_type}"
        )
    if offset is not None:
        raise NotImplementedError(
            "npu_grouped_matmul triton fallback does not support quant offset"
        )
    for name, val in (
        ("activation_input", activation_input),
        ("activation_quant_scale", activation_quant_scale),
        ("activation_quant_offset", activation_quant_offset),
        ("tuning_config", tuning_config),
    ):
        if val is not None:
            raise NotImplementedError(
                f"npu_grouped_matmul triton fallback does not support {name}"
            )
    if group_list is None:
        raise ValueError("group_list is required for split_item=2 grouped matmul")

    x_t = _unwrap_single(_as_tensor_list(x), "x")
    w_t = _unwrap_single(_as_tensor_list(weight), "weight")
    assert x_t is not None and w_t is not None

    if x_t.dim() != 2:
        raise ValueError(f"x must be 2D [M, K], got {tuple(x_t.shape)}")
    if not x_t.is_contiguous():
        x_t = x_t.contiguous()

    bias_t = _unwrap_single(_as_tensor_list(bias), "bias")
    scale_t = _unwrap_single(_as_tensor_list(scale), "scale")
    per_token_scale_t = _unwrap_single(
        _as_tensor_list(per_token_scale), "per_token_scale"
    )
    aq_scale_t = _unwrap_single(_as_tensor_list(antiquant_scale), "antiquant_scale")
    aq_offset_t = _unwrap_single(
        _as_tensor_list(antiquant_offset), "antiquant_offset"
    )

    _, offsets = _counts_and_offsets(group_list, group_list_type, x_t.shape[0])

    is_antiquant = aq_scale_t is not None
    w_t, _ = _prepare_weight(w_t, antiquant=is_antiquant or w_t.dtype == torch.int32)

    out_dtype = output_dtype
    if out_dtype is None:
        out_dtype = (
            torch.bfloat16
            if x_t.dtype in (torch.int8, torch.int32)
            else x_t.dtype
        )

    # WNA16 antiquant: dequant weights then FP GEMM
    if is_antiquant:
        w_fp = _dequant_weight(w_t, aq_scale_t, aq_offset_t).to(x_t.dtype)
        if bias_t is not None:
            bias_t = bias_t.reshape(w_fp.shape[0], w_fp.shape[2]).contiguous()
        out = _launch_fp_grouped_matmul(x_t, w_fp, offsets, out_dtype, bias_t)
        return [out]

    # Int8 / unpacked-int4 quantized path
    if x_t.dtype == torch.int8 or w_t.dtype == torch.int8:
        if x_t.dtype != torch.int8 or w_t.dtype != torch.int8:
            raise TypeError(
                f"quantized grouped matmul expects int8 x and weight, "
                f"got {x_t.dtype}, {w_t.dtype}"
            )
        if bias_t is not None and bias_t.dim() == 1:
            bias_t = bias_t.reshape(w_t.shape[0], -1)
        out = _launch_int8_grouped_matmul(
            x_t,
            w_t,
            offsets,
            out_dtype,
            per_token_scale_t,
            scale_t,
            bias_t,
        )
        return [out]

    # BF16 / FP16 / FP32 unquant
    if w_t.dtype != x_t.dtype:
        w_t = w_t.to(dtype=x_t.dtype)
    if bias_t is not None:
        bias_t = bias_t.to(dtype=torch.float32).reshape(w_t.shape[0], w_t.shape[2])
    out = _launch_fp_grouped_matmul(x_t, w_t, offsets, out_dtype, bias_t)
    return [out]
