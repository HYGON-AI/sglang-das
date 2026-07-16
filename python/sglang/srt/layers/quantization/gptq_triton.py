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

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.utils import ScalarType, scalar_types

SUPPORTED_GPTQ_TRITON_TYPES = (scalar_types.uint4b8,)


def _make_autotune_configs() -> list[triton.Config]:
    configs: list[triton.Config] = [
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 128,
            },
            num_warps=4,
            num_stages=2,
        )
    ]
    # for block_m in (16, 32, 64):
    #     for block_n in (32, 64):
    #         for block_k in (32, 128):
    #             for num_warps in (1, 4, 8):
    #                 for num_stages in (1, 2):
    #                     if (
    #                         block_m == 64
    #                         and block_n == 64
    #                         and block_k == 128
    #                         and num_warps == 4
    #                         and num_stages == 2
    #                     ):
    #                         continue
    #                     configs.append(
    #                         triton.Config(
    #                             {
    #                                 "BLOCK_M": block_m,
    #                                 "BLOCK_N": block_n,
    #                                 "BLOCK_K": block_k,
    #                             },
    #                             num_warps=num_warps,
    #                             num_stages=num_stages,
    #                         )
    #                     )
    return configs


@triton.autotune(
    configs=_make_autotune_configs(),
    key=["M", "N", "K", "GROUP_SIZE"],
)
@triton.heuristics(
    values={
        "DIVISIBLE_M": lambda args: args["M"] % args["BLOCK_M"] == 0,
        "DIVISIBLE_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
        "DIVISIBLE_K": lambda args: args["K"] % args["BLOCK_K"] == 0,
    }
)
@triton.jit
def _gptq_packed_gemm_kernel(
    a_ptr,
    b_ptr,
    scales_ptr,
    c_ptr,
    M,
    N,
    K,
    GROUP_SIZE,
    stride_am,
    stride_ak,
    stride_qk,
    stride_qn,
    stride_sg,
    stride_sn,
    stride_cm,
    stride_cn,
    NUM_BITS: tl.constexpr,
    PACK_FACTOR: tl.constexpr,
    BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
):
    tl.static_assert(BLOCK_K % 16 == 0)
    tl.static_assert(BLOCK_K % PACK_FACTOR == 0)
    tl.static_assert(PACK_FACTOR == 8)

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    value_mask = (1 << NUM_BITS) - 1
    k_packed = K // PACK_FACTOR
    shifts = tl.arange(0, PACK_FACTOR) * NUM_BITS
    shifts = tl.broadcast_to(
        shifts[None, :], (BLOCK_N * (BLOCK_K // PACK_FACTOR), PACK_FACTOR)
    )
    shifts = tl.reshape(shifts, (BLOCK_N, BLOCK_K))

    for k0 in range(0, K, BLOCK_K):
        cur_k = k0 + tl.arange(0, BLOCK_K)
        cur_k = tl.max_contiguous(tl.multiple_of(cur_k, BLOCK_K), BLOCK_K)
        k_mask = cur_k < K
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + cur_k[None, :] * stride_ak
        a_mask = None
        if not DIVISIBLE_M and not DIVISIBLE_K:
            a_mask = m_mask[:, None] & k_mask[None, :]
        elif not DIVISIBLE_M:
            a_mask = m_mask[:, None]
        elif not DIVISIBLE_K:
            a_mask = k_mask[None, :]

        if a_mask is None:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_rows = (k0 // PACK_FACTOR) + tl.arange(0, BLOCK_K // PACK_FACTOR)
        b_row_mask = b_rows < k_packed
        b_ptrs = b_ptr + offs_n[:, None] * stride_qn + b_rows[None, :] * stride_qk
        b_mask = None
        if not DIVISIBLE_N and not DIVISIBLE_K:
            b_mask = n_mask[:, None] & b_row_mask[None, :]
        elif not DIVISIBLE_N:
            b_mask = n_mask[:, None]
        elif not DIVISIBLE_K:
            b_mask = b_row_mask[None, :]

        if b_mask is None:
            b_packed = tl.load(b_ptrs)
        else:
            b_packed = tl.load(b_ptrs, mask=b_mask, other=0)
        b_packed = b_packed.to(tl.uint32)
        b_packed = tl.interleave(b_packed, b_packed)
        b_packed = tl.interleave(b_packed, b_packed)
        b_packed = tl.interleave(b_packed, b_packed)

        b = (b_packed >> shifts) & value_mask
        b = tl.trans(b)

        scales_ptrs = (
            scales_ptr
            + (cur_k // GROUP_SIZE)[:, None] * stride_sg
            + offs_n[None, :] * stride_sn
        )
        scales_mask = None
        if not DIVISIBLE_K and not DIVISIBLE_N:
            scales_mask = k_mask[:, None] & n_mask[None, :]
        elif not DIVISIBLE_K:
            scales_mask = k_mask[:, None]
        elif not DIVISIBLE_N:
            scales_mask = n_mask[None, :]

        if scales_mask is None:
            scales = tl.load(scales_ptrs)
        else:
            scales = tl.load(scales_ptrs, mask=scales_mask, other=0.0)

        b = (b.to(tl.float32) - BIAS) * scales.to(tl.float32)
        acc = tl.dot(a, b.to(a_ptr.type.element_ty), acc, out_dtype=tl.float32)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = None
    if not DIVISIBLE_M and not DIVISIBLE_N:
        c_mask = m_mask[:, None] & n_mask[None, :]
    elif not DIVISIBLE_M:
        c_mask = m_mask[:, None]
    elif not DIVISIBLE_N:
        c_mask = n_mask[None, :]

    if c_mask is None:
        tl.store(c_ptrs, acc)
    else:
        tl.store(c_ptrs, acc, mask=c_mask)


def _infer_group_size(size_k: int, scales: torch.Tensor) -> int:
    if scales.ndim != 2:
        raise ValueError(f"scales must be rank-2, got shape={tuple(scales.shape)}")

    num_groups = scales.shape[0]
    if num_groups == 1:
        return size_k
    if size_k % num_groups != 0:
        raise ValueError(
            f"Invalid scales.shape[0]={num_groups} for K={size_k}: K must be divisible "
            "by the number of scale groups."
        )
    return size_k // num_groups


def _normalize_raw_gptq_layout(
    size_k: int,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    pack_factor: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Accept both raw GPTQ physical layouts used in tests and in SGLang layers.

    Supported physical layouts:
    - qweight: [K / pack_factor, N], scales: [G, N]
    - qweight: [N, K / pack_factor], scales: [N, G]
    """
    packed_k = size_k // pack_factor

    if qweight.ndim != 2 or scales.ndim != 2:
        raise ValueError(
            "qweight and scales must both be rank-2 tensors, got "
            f"{tuple(qweight.shape)} and {tuple(scales.shape)}."
        )

    if qweight.shape[0] == packed_k:
        size_n = qweight.shape[1]
        if scales.shape[1] != size_n:
            raise ValueError(
                "scales shape mismatch for [K/pack_factor, N] qweight layout: expected "
                f"scales.shape[1] == {size_n}, got {scales.shape[1]}."
            )
        return qweight, scales, size_n

    if qweight.shape[1] == packed_k:
        size_n = qweight.shape[0]
        if scales.shape[0] != size_n:
            raise ValueError(
                "scales shape mismatch for [N, K/pack_factor] qweight layout: expected "
                f"scales.shape[0] == {size_n}, got {scales.shape[0]}."
            )
        return qweight.t(), scales.t(), size_n

    raise ValueError(
        "qweight shape mismatch: expected raw GPTQ layout to be either "
        f"({packed_k}, N) or (N, {packed_k}), got {tuple(qweight.shape)}."
    )


def _validate_inputs(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    quant_type: ScalarType,
):
    if quant_type not in SUPPORTED_GPTQ_TRITON_TYPES:
        raise ValueError(
            f"Unsupported quant_type={quant_type}. Supported types: "
            f"{SUPPORTED_GPTQ_TRITON_TYPES}."
        )
    if not a.is_cuda or not qweight.is_cuda or not scales.is_cuda:
        raise ValueError("a, qweight, and scales must all be CUDA tensors.")
    if qweight.dtype != torch.int32:
        raise ValueError(f"qweight must be torch.int32, got {qweight.dtype}.")
    if not a.dtype.is_floating_point or not scales.dtype.is_floating_point:
        raise ValueError("a and scales must be floating point tensors.")
    if a.shape[-1] <= 0:
        raise ValueError("a.shape[-1] must be positive.")


def _validate_optional_empty(
    tensor: Optional[torch.Tensor],
    name: str,
    why: str,
) -> None:
    if tensor is None:
        return
    if tensor.numel() == 0:
        return
    raise ValueError(
        f"gptq_gemm_triton does not support non-empty {name}. {why} "
        f"Got shape={tuple(tensor.shape)}."
    )


def gptq_gemm_triton(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    quant_type: ScalarType,
    qzeros: Optional[torch.Tensor] = None,
    g_idx: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run raw GPTQ GEMM on row-packed weights.

    Supported subset:
    - raw GPTQ row-packed `qweight`
    - symmetric 4-bit weights (`pack_factor == 8`)
    - sequential grouping, where the scale row is inferred as `k // group_size`

    Unsupported metadata must be absent:
    - `qzeros`: only symmetric GPTQ is supported
    - `g_idx`: only trivial sequential group mapping is supported
    """
    _validate_inputs(a, qweight, scales, quant_type)
    _validate_optional_empty(
        qzeros,
        "qzeros",
        "The current Triton GPTQ kernel only supports symmetric GPTQ weights and "
        "therefore expects qzeros to be empty.",
    )
    _validate_optional_empty(
        g_idx,
        "g_idx",
        "The current Triton GPTQ kernel assumes sequential grouping "
        "(scale row = k // group_size) and therefore does not support "
        "act-order / desc_act or other non-trivial group mappings.",
    )

    size_k = a.shape[-1]
    size_m = a.numel() // size_k
    num_bits = quant_type.size_bits
    pack_factor = 32 // num_bits
    if pack_factor != 8:
        raise ValueError(
            f"gptq_gemm_triton currently only supports pack_factor=8 (4-bit). Got "
            f"pack_factor={pack_factor} for {quant_type}."
        )

    if size_k % pack_factor != 0:
        raise ValueError(
            f"K={size_k} must be divisible by pack_factor={pack_factor} for {quant_type}."
        )

    qweight_kernel, scales_kernel, size_n = _normalize_raw_gptq_layout(
        size_k, qweight, scales, pack_factor
    )

    group_size = _infer_group_size(size_k, scales_kernel)
    if size_k % group_size != 0:
        raise ValueError(f"K={size_k} must be divisible by group_size={group_size}.")

    a_2d = a.reshape(size_m, size_k)
    if out is None:
        out_2d = torch.empty((size_m, size_n), dtype=a.dtype, device=a.device)
    else:
        if out.shape != (*a.shape[:-1], size_n):
            raise ValueError(
                f"out has invalid shape {tuple(out.shape)}; expected "
                f"{(*a.shape[:-1], size_n)}."
            )
        out_2d = out.reshape(size_m, size_n)

    if size_m == 0 or size_n == 0:
        return out_2d.reshape(*a.shape[:-1], size_n)

    grid = lambda META: (
        triton.cdiv(size_m, META["BLOCK_M"]),
        triton.cdiv(size_n, META["BLOCK_N"]),
    )

    _gptq_packed_gemm_kernel[grid](
        a_2d,
        qweight_kernel,
        scales_kernel,
        out_2d,
        size_m,
        size_n,
        size_k,
        group_size,
        a_2d.stride(0),
        a_2d.stride(1),
        qweight_kernel.stride(0),
        qweight_kernel.stride(1),
        scales_kernel.stride(0),
        scales_kernel.stride(1),
        out_2d.stride(0),
        out_2d.stride(1),
        NUM_BITS=num_bits,
        PACK_FACTOR=pack_factor,
        BIAS=quant_type.bias,
    )

    return out_2d.reshape(*a.shape[:-1], size_n)
