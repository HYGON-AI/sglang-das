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

"""Triton reference implementation for vLLM-style paged varlen attention.

This is intentionally written for debuggability and numerical comparison rather
than peak throughput. It supports both SGLang paged KV layouts:
  * SGLANG_KV_LAYOUT_HCU_FA=0: k/v are [num_blocks, page_size, Hkv, D]
  * vLLM/HCU FA layout:        k is [num_blocks, Hkv, page_size, D],
                               v is [num_blocks, Hkv, D, page_size]
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


def _is_float8_dtype(dtype: torch.dtype) -> bool:
    return str(dtype).startswith("torch.float8")


def _attention_output_dtype(q_dtype: torch.dtype) -> torch.dtype:
    # The native FA path dequantizes FP8 inputs and returns a normal activation
    # dtype. Keeping FP8 here breaks later PyTorch ops such as BF16 gates.
    if _is_float8_dtype(q_dtype):
        return torch.bfloat16
    return q_dtype


@triton.jit
def _paged_varlen_attn_fwd_kernel(
    Q,
    K,
    V,
    CU_Q,
    SEQ_K,
    BLOCK_TABLE,
    Q_DESCALE,
    K_DESCALE,
    V_DESCALE,
    O,
    SCALE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    D: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    Q_STRIDE_T: tl.constexpr,
    Q_STRIDE_H: tl.constexpr,
    Q_STRIDE_D: tl.constexpr,
    K_STRIDE_B: tl.constexpr,
    K_STRIDE_P: tl.constexpr,
    K_STRIDE_H: tl.constexpr,
    K_STRIDE_D: tl.constexpr,
    V_STRIDE_B: tl.constexpr,
    V_STRIDE_P: tl.constexpr,
    V_STRIDE_H: tl.constexpr,
    V_STRIDE_D: tl.constexpr,
    BT_STRIDE_B: tl.constexpr,
    BT_STRIDE_BLK: tl.constexpr,
    Q_DESCALE_STRIDE_B: tl.constexpr,
    Q_DESCALE_STRIDE_H: tl.constexpr,
    K_DESCALE_STRIDE_B: tl.constexpr,
    K_DESCALE_STRIDE_H: tl.constexpr,
    V_DESCALE_STRIDE_B: tl.constexpr,
    V_DESCALE_STRIDE_H: tl.constexpr,
    O_STRIDE_T: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    O_STRIDE_D: tl.constexpr,
    HAS_Q_DESCALE: tl.constexpr,
    HAS_K_DESCALE: tl.constexpr,
    HAS_V_DESCALE: tl.constexpr,
    Q_DESCALE_HEADS: tl.constexpr,
    K_DESCALE_HEADS: tl.constexpr,
    V_DESCALE_HEADS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    q_start = tl.load(CU_Q + pid_b)
    q_end = tl.load(CU_Q + pid_b + 1)
    q_len = q_end - q_start
    k_len = tl.load(SEQ_K + pid_b)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_vd = tl.arange(0, D_V)
    q_mask = offs_m < q_len

    kv_head = pid_h // (H_Q // H_KV)
    q_descale_head = tl.where(Q_DESCALE_HEADS == 1, 0, tl.where(Q_DESCALE_HEADS == H_Q, pid_h, kv_head))
    k_descale_head = tl.where(K_DESCALE_HEADS == 1, 0, kv_head)
    v_descale_head = tl.where(V_DESCALE_HEADS == 1, 0, kv_head)
    q_scale = 1.0
    k_scale = 1.0
    v_scale = 1.0
    if HAS_Q_DESCALE:
        q_scale = tl.load(
            Q_DESCALE + pid_b * Q_DESCALE_STRIDE_B + q_descale_head * Q_DESCALE_STRIDE_H
        ).to(tl.float32)
    if HAS_K_DESCALE:
        k_scale = tl.load(
            K_DESCALE + pid_b * K_DESCALE_STRIDE_B + k_descale_head * K_DESCALE_STRIDE_H
        ).to(tl.float32)
    if HAS_V_DESCALE:
        v_scale = tl.load(
            V_DESCALE + pid_b * V_DESCALE_STRIDE_B + v_descale_head * V_DESCALE_STRIDE_H
        ).to(tl.float32)

    q_ptrs = Q + (q_start + offs_m[:, None]) * Q_STRIDE_T + pid_h * Q_STRIDE_H + offs_d[None, :] * Q_STRIDE_D
    q = tl.load(q_ptrs, mask=q_mask[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float32)
    q = q * q_scale

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D_V), tl.float32)

    q_abs_pos = k_len - q_len + offs_m
    for start_n in range(0, k_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_mask = offs_n < k_len
        logical_block = offs_n // PAGE_SIZE
        page_offset = offs_n - logical_block * PAGE_SIZE
        physical_block = tl.load(
            BLOCK_TABLE + pid_b * BT_STRIDE_B + logical_block * BT_STRIDE_BLK,
            mask=kv_mask,
            other=0,
        )

        k_ptrs = (
            K
            + physical_block[None, :] * K_STRIDE_B
            + page_offset[None, :] * K_STRIDE_P
            + kv_head * K_STRIDE_H
            + offs_d[:, None] * K_STRIDE_D
        )
        k = tl.load(k_ptrs, mask=kv_mask[None, :] & (offs_d[:, None] < D), other=0.0).to(tl.float32)
        k = k * k_scale
        qk = tl.dot(q, k) * SCALE

        qk = tl.where(q_mask[:, None] & kv_mask[None, :], qk, -float("inf"))
        if CAUSAL:
            qk = tl.where(offs_n[None, :] <= q_abs_pos[:, None], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)

        v_ptrs = (
            V
            + physical_block[:, None] * V_STRIDE_B
            + page_offset[:, None] * V_STRIDE_P
            + kv_head * V_STRIDE_H
            + offs_vd[None, :] * V_STRIDE_D
        )
        v = tl.load(v_ptrs, mask=kv_mask[:, None] & (offs_vd[None, :] < D_V), other=0.0).to(tl.float32)
        v = v * v_scale

        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    acc = acc / l_i[:, None]
    out_ptrs = O + (q_start + offs_m[:, None]) * O_STRIDE_T + pid_h * O_STRIDE_H + offs_vd[None, :] * O_STRIDE_D
    tl.store(out_ptrs, acc, mask=q_mask[:, None] & (offs_vd[None, :] < D_V))


def triton_vllm_flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: Optional[float],
    causal: bool,
    window_size: Tuple[int, int],
    block_table: torch.Tensor,
    fa_version: int,
    q_descale: Optional[torch.Tensor],
    k_descale: Optional[torch.Tensor],
    v_descale: Optional[torch.Tensor],
) -> torch.Tensor:
    if window_size != (-1, -1):
        raise NotImplementedError("Triton reference FA only supports full-context attention.")
    if q.dim() != 3:
        raise ValueError(f"q must be [total_q, Hq, D], got {tuple(q.shape)}")
    if k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            "k/v must be paged KV cache tensors. Supported layouts are "
            "[num_blocks, page_size, Hkv, D] and k=[num_blocks, Hkv, page_size, D], "
            "v=[num_blocks, Hkv, D, page_size]. "
            f"got k={tuple(k.shape)}, v={tuple(v.shape)}"
        )

    total_q, h_q, d = q.shape
    if k.shape[1] == v.shape[1] and k.shape[2] == v.shape[3]:
        # vLLM/HCU FA layout: k=[B,H,P,D], v=[B,H,Dv,P].
        h_kv = k.shape[1]
        page_size = k.shape[2]
        d_v = v.shape[2]
        k_stride_b, k_stride_p, k_stride_h, k_stride_d = (
            k.stride(0),
            k.stride(2),
            k.stride(1),
            k.stride(3),
        )
        v_stride_b, v_stride_p, v_stride_h, v_stride_d = (
            v.stride(0),
            v.stride(3),
            v.stride(1),
            v.stride(2),
        )
    elif k.shape[1] == v.shape[1] and k.shape[2] == v.shape[2]:
        # SGLANG_KV_LAYOUT_HCU_FA=0 layout: k/v=[B,P,H,D].
        page_size = k.shape[1]
        h_kv = k.shape[2]
        d_v = v.shape[-1]
        k_stride_b, k_stride_p, k_stride_h, k_stride_d = k.stride()
        v_stride_b, v_stride_p, v_stride_h, v_stride_d = v.stride()
    else:
        raise ValueError(f"k/v layout mismatch: k={tuple(k.shape)}, v={tuple(v.shape)}")

    if h_q % h_kv != 0:
        raise ValueError(f"Hq must be divisible by Hkv, got Hq={h_q}, Hkv={h_kv}")
    if softmax_scale is None:
        softmax_scale = d ** -0.5
    out_dtype = _attention_output_dtype(q.dtype)
    if max_seqlen_k <= 0 or max_seqlen_q <= 0:
        return torch.empty((total_q, h_q, d_v), device=q.device, dtype=out_dtype)

    q = q.contiguous()
    if k.stride(-1) != 1:
        k = k.contiguous()
    if v.stride(-1) != 1:
        v = v.contiguous()
    cu_seqlens_q = cu_seqlens_q.contiguous()
    seqused_k = seqused_k.contiguous()
    block_table = block_table.contiguous()
    batch = cu_seqlens_q.numel() - 1

    def _normalize_descale(descale: Optional[torch.Tensor], name: str, valid_heads):
        if descale is None:
            dummy = torch.empty((1, 1), device=q.device, dtype=torch.float32)
            return dummy, False, 1
        descale = descale.to(device=q.device, dtype=torch.float32)
        if descale.dim() == 0:
            descale = descale.view(1, 1)
        elif descale.dim() == 1:
            descale = descale.view(1, descale.shape[0])
        elif descale.dim() != 2:
            raise ValueError(f"{name} must be scalar, [heads], or [batch, heads], got {tuple(descale.shape)}")
        if descale.shape[0] not in (1, batch):
            raise ValueError(f"{name} batch dim must be 1 or {batch}, got {tuple(descale.shape)}")
        if descale.shape[1] not in valid_heads:
            raise ValueError(
                f"{name} head dim must be one of {sorted(valid_heads)}, got {tuple(descale.shape)}"
            )
        if descale.shape[0] == 1 and batch != 1:
            descale = descale.expand(batch, descale.shape[1])
        return descale.contiguous(), True, descale.shape[1]

    q_descale, has_q_descale, q_descale_heads = _normalize_descale(
        q_descale, "q_descale", {1, h_kv, h_q}
    )
    k_descale, has_k_descale, k_descale_heads = _normalize_descale(
        k_descale, "k_descale", {1, h_kv}
    )
    v_descale, has_v_descale, v_descale_heads = _normalize_descale(
        v_descale, "v_descale", {1, h_kv}
    )

    out = torch.empty((total_q, h_q, d_v), device=q.device, dtype=out_dtype)
    block_m = 16
    block_n = 64
    grid = (triton.cdiv(max_seqlen_q, block_m), h_q, batch)
    _paged_varlen_attn_fwd_kernel[grid](
        q,
        k,
        v,
        cu_seqlens_q,
        seqused_k,
        block_table,
        q_descale,
        k_descale,
        v_descale,
        out,
        float(softmax_scale),
        page_size,
        h_q,
        h_kv,
        d,
        d_v,
        block_m,
        block_n,
        causal,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_stride_b,
        k_stride_p,
        k_stride_h,
        k_stride_d,
        v_stride_b,
        v_stride_p,
        v_stride_h,
        v_stride_d,
        block_table.stride(0),
        block_table.stride(1),
        q_descale.stride(0),
        q_descale.stride(1),
        k_descale.stride(0),
        k_descale.stride(1),
        v_descale.stride(0),
        v_descale.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        has_q_descale,
        has_k_descale,
        has_v_descale,
        q_descale_heads,
        k_descale_heads,
        v_descale_heads,
        num_warps=4,
    )
    return out


def triton_vllm_flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens,
    softmax_scale: Optional[float],
    causal: bool,
    window_size: Tuple[int, int],
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    return_softmax_lse: bool = False,
) -> torch.Tensor:
    """Decode/reference wrapper for vLLM-style paged KV cache attention.

    The flash-attn with-kvcache API takes q as [B, S_q, H, D] for decode. The
    Triton reference kernel operates on varlen [total_q, H, D], so this wrapper
    builds cu_seqlens_q and delegates to triton_vllm_flash_attn_varlen_func.
    """
    if return_softmax_lse:
        raise NotImplementedError("Triton reference decode FA does not return softmax_lse yet.")
    if softcap not in (0, 0.0, None):
        raise NotImplementedError("Triton reference decode FA does not support softcap yet.")
    if block_table is None:
        raise ValueError("block_table/page_table is required for paged decode attention.")
    if q.dim() == 3:
        original_shape = q.shape
        q_4d = q.unsqueeze(1)
        squeeze_q_dim = True
    elif q.dim() == 4:
        original_shape = q.shape
        q_4d = q
        squeeze_q_dim = False
    else:
        raise ValueError(f"q must be [B,H,D] or [B,Sq,H,D], got {tuple(q.shape)}")

    batch, q_len, h_q, d = q_4d.shape
    if k_cache.shape[1] == v_cache.shape[1] and k_cache.shape[2] == v_cache.shape[3]:
        page_size = k_cache.shape[2]
    elif k_cache.shape[1] == v_cache.shape[1] and k_cache.shape[2] == v_cache.shape[2]:
        page_size = k_cache.shape[1]
    else:
        raise ValueError(
            f"k/v cache layout mismatch: k={tuple(k_cache.shape)}, v={tuple(v_cache.shape)}"
        )
    q_flat = q_4d.contiguous().view(batch * q_len, h_q, d)
    cu_seqlens_q = torch.arange(
        0,
        (batch + 1) * q_len,
        q_len,
        device=q.device,
        dtype=torch.int32,
    )
    if isinstance(cache_seqlens, int):
        seqused_k = torch.full((batch,), cache_seqlens, device=q.device, dtype=torch.int32)
        max_seqlen_k = cache_seqlens
    else:
        seqused_k = cache_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
        # Avoid GPU-to-CPU sync during CUDA/HIP graph capture. The Triton kernel
        # reads the exact per-request lengths from seqused_k; max_seqlen_k is
        # only needed as a Python-side non-empty guard.
        max_seqlen_k = block_table.shape[1] * page_size if seqused_k.numel() else 0

    out = triton_vllm_flash_attn_varlen_func(
        q=q_flat,
        k=k_cache,
        v=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=q_len,
        seqused_k=seqused_k,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        block_table=block_table,
        fa_version=2,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    out = out.view(batch, q_len, h_q, out.shape[-1])
    if squeeze_q_dim:
        return out.view(original_shape[0], original_shape[1], out.shape[-1])
    return out