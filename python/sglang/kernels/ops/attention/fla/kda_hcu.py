# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code adapted from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
"""KDA kernels (HCU variant, operator team), vendored here.

Each swapped kernel lives in this module, enabled by
``SGLANG_KDA_USE_HCU_OP`` (default off) from ``fla/kda.py`` /
``fla/chunk_intra.py`` / ``fla/fused_recurrent.py``.  Currently contains:

extend:
  - kda_gate_chunk_cumsum_vector_kernel
  - chunk_kda_fwd_intra_token_parallel (+ sub-chunk variant)
  - chunk_kda_fwd_kernel_inter_solve_fused
  - recompute_w_u_fwd (+ beta-factored head-first variant)
  - chunk_gla_fwd_kernel_o (+ local-first variant)
decode:
  - fused_recurrent_kda_packed_decode (+ direct-3D variant)
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices
from sglang.kernels.ops.attention.fla.op import exp, exp2, log
from sglang.kernels.ops.attention.fla.utils import (
    is_gather_supported,
    is_tf32_supported,
)

RCP_LN2 = 1.4426950216293335
cdiv = triton.cdiv
next_power_of_2 = triton.next_power_of_2
FLA_CHUNK_SIZE = 64

# Offline-tuned on gfx938 (BW1100): gate cumsum vector kernel.
KDA_GATE_CHUNK_CUMSUM_HCU_CONFIGS = (
    {"BS": 32, "num_warps": 2},
    {"BS": 32, "num_warps": 4},
    {"BS": 64, "num_warps": 2},
)


def get_kda_gate_chunk_cumsum_vector_hcu_config(
    T: int, H: int, _S: int, is_varlen: bool
) -> dict:
    work = T * H
    if work <= 16_384:
        return KDA_GATE_CHUNK_CUMSUM_HCU_CONFIGS[1]
    if is_varlen or T >= 4096:
        return KDA_GATE_CHUNK_CUMSUM_HCU_CONFIGS[2]
    return KDA_GATE_CHUNK_CUMSUM_HCU_CONFIGS[0]


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["g_bias"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit
def kda_gate_chunk_cumsum_vector_kernel_hcu(
    s,
    raw_beta,
    A_log,
    g_bias,
    o,
    beta_out,
    cu_seqlens,
    chunk_indices,
    cumsum_scale,
    lower_bound,
    beta,
    threshold,
    T,
    stride_beta_batch,
    stride_beta_token,
    stride_beta_head,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos = i_b * T

    if i_s == 0:
        o_beta_t = tl.arange(0, BT)
        m_beta = i_t * BT + o_beta_t < T
        if IS_VARLEN:
            p_beta = (
                raw_beta
                + (bos + i_t * BT + o_beta_t) * stride_beta_token
                + i_h * stride_beta_head
            )
        else:
            p_beta = (
                raw_beta
                + i_b * stride_beta_batch
                + (i_t * BT + o_beta_t) * stride_beta_token
                + i_h * stride_beta_head
            )
        b_beta = tl.load(p_beta, mask=m_beta, other=0.0).to(tl.float32)
        p_beta_out = beta_out + (bos + i_t * BT + o_beta_t) * H + i_h
        tl.store(p_beta_out, tl.sigmoid(b_beta), mask=m_beta)
        return

    i_s -= 1

    p_s = tl.make_block_ptr(
        s + (bos * H + i_h) * S,
        (T, S),
        (H * S, 1),
        (i_t * BT, i_s * BS),
        (BT, BS),
        (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * S,
        (T, S),
        (H * S, 1),
        (i_t * BT, i_s * BS),
        (BT, BS),
        (1, 0),
    )

    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    if HAS_BIAS:
        p_bias = tl.make_block_ptr(
            g_bias + i_h * S,
            (S,),
            (1,),
            (i_s * BS,),
            (BS,),
            (0,),
        )
        b_bias = tl.load(p_bias, boundary_check=(0,)).to(tl.float32)
        b_s += b_bias[None, :]

    b_a = tl.exp(tl.load(A_log + i_h).to(tl.float32))
    if USE_LOWER_BOUND:
        b_gate = lower_bound * tl.sigmoid(b_a * b_s)
    else:
        b_g_scaled = b_s * beta
        b_softplus = tl.where(
            b_g_scaled > threshold,
            b_s,
            (1.0 / beta) * log(1.0 + tl.exp(b_g_scaled)),
        )
        b_gate = -b_a * b_softplus

    # Boundary loads return zero, but bias and gate activation can make padded
    # rows nonzero. Padding trails valid rows, so it only affects masked stores.
    b_o = tl.cumsum(b_gate, axis=0) * cumsum_scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def fused_kda_gate_chunk_cumsum_hcu(
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: Optional[torch.Tensor] = None,
    beta: float = 1.0,
    threshold: float = 20.0,
    lower_bound: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    output_dtype: Optional[torch.dtype] = torch.float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HCU fused gate activation + chunk-local cumsum (also computes the
    activated beta).  Returns ``(gate_cumsum, beta_out)``."""
    if cu_seqlens is not None:
        assert raw_g.shape[0] == 1, (
            "Only batch size 1 is supported when cu_seqlens are provided"
        )
    B, T, H, D = raw_g.shape
    if raw_beta.shape != (B, T, H):
        raise ValueError(f"Expected raw_beta shape {(B, T, H)}, got {raw_beta.shape}")
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = cdiv(T, chunk_size) if cu_seqlens is None else len(chunk_indices)

    A_log = A_log.reshape(-1)
    if g_bias is not None:
        g_bias = g_bias.reshape(-1)
    y = torch.empty_like(raw_g, dtype=output_dtype or raw_g.dtype)
    beta_out = torch.empty(raw_beta.shape, device=raw_beta.device, dtype=torch.float32)
    config = get_kda_gate_chunk_cumsum_vector_hcu_config(
        T, H, D, cu_seqlens is not None
    )
    # For each (chunk, head), program 0 computes beta without extending a gate
    # tile's critical path. The remaining programs cover the gate dimension.
    grid = (cdiv(D, config["BS"]) + 1, NT, B * H)

    kda_gate_chunk_cumsum_vector_kernel_hcu[grid](
        s=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        g_bias=g_bias,
        o=y,
        beta_out=beta_out,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        # RCP_LN2 folds in the natural-log -> log2 conversion so downstream
        # exp2-based kernels reproduce exp(g).
        cumsum_scale=RCP_LN2,
        lower_bound=lower_bound or 0.0,
        beta=beta,
        threshold=threshold,
        T=T,
        stride_beta_batch=raw_beta.stride(0),
        stride_beta_token=raw_beta.stride(1),
        stride_beta_head=raw_beta.stride(2),
        H=H,
        S=D,
        BT=chunk_size,
        USE_LOWER_BOUND=lower_bound is not None,
        **config,
    )
    return y, beta_out


# ---------------------------------------------------------------------------
# Intra chunk: token-parallel diagonal blocks (chunk_kda_fwd_intra_token_parallel)
# ---------------------------------------------------------------------------

# Offline-tuned on BW1100 at the KDA prefill model shape.
CHUNK_KDA_INTRA_TOKEN_PARALLEL_HCU_CONFIGS = (
    {"BK": 128, "num_warps": 1, "num_stages": 1},
    {"BK": 128, "num_warps": 1, "num_stages": 2},
    {"BK": 128, "num_warps": 1, "num_stages": 3},
)


def get_chunk_kda_intra_token_parallel_hcu_config(
    T: int, H: int, HV: int, is_varlen: bool
) -> dict[str, int]:
    del T, H, HV, is_varlen
    return CHUNK_KDA_INTRA_TOKEN_PARALLEL_HCU_CONFIGS[0]


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "N"])
def chunk_kda_fwd_kernel_intra_token_parallel_hcu(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    N,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_tg, i_hg = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        left, right = 0, N
        # Unrolled binary search (max B=2^32)
        for _ in range(20):
            if left < right:
                mid = (left + right) // 2
                if i_tg < tl.load(cu_seqlens + mid + 1).to(tl.int32):
                    right = mid
                else:
                    left = mid + 1
        i_n = left

        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        i_t = i_tg - bos
    else:
        bos = (i_tg // T) * T
        i_t = i_tg % T

    if i_t >= T:
        return

    i_c = i_t // BT
    i_s = (i_t % BT) // BC
    i_tc = i_c * BT
    i_ts = i_tc + i_s * BC

    G: tl.constexpr = HV // H

    q += bos * H * K
    k += bos * H * K
    g += bos * HV * K
    Aqk += bos * HV * BT
    Akk += bos * HV * BC
    beta += bos * HV

    BK: tl.constexpr = triton.next_power_of_2(K)
    o_hv = i_hg * BH + tl.arange(0, BH)
    o_h = o_hv // G
    o_k = tl.arange(0, BK)
    m_hv = o_hv < HV
    m_k = o_k < K
    m_hk = m_hv[:, None] & m_k[None, :]

    # q/k: [B, T, H, K], manual load via mapped qk head index
    p_qk = o_h[:, None] * K + o_k[None, :]
    b_q = tl.load(q + i_t * H * K + p_qk, mask=m_hk, other=0).to(tl.float32)
    b_k = tl.load(k + i_t * H * K + p_qk, mask=m_hk, other=0).to(tl.float32)

    # g: [B, T, HV, K], beta: [B, T, HV]
    p_g = tl.make_block_ptr(
        g + i_t * HV * K, (HV, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0)
    )
    p_beta = tl.make_block_ptr(beta + i_t * HV, (HV,), (1,), (i_hg * BH,), (BH,), (0,))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
    b_k *= b_beta[:, None]

    for j in range(i_ts, min(i_t + 1, min(T, i_ts + BC))):
        b_kj = tl.load(k + j * H * K + p_qk, mask=m_hk, other=0).to(tl.float32)
        p_gj = tl.make_block_ptr(
            g + j * HV * K, (HV, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0)
        )
        b_gj = tl.load(p_gj, boundary_check=(0, 1)).to(tl.float32)

        b_kgj = tl.where(m_k[None, :], b_kj * exp2(b_g - b_gj), 0.0)
        b_Aqk = tl.sum(b_q * b_kgj, axis=1) * scale
        b_Akk = tl.sum(b_k * b_kgj, axis=1) * tl.where(j < i_t, 1.0, 0.0)

        tl.store(
            Aqk + i_t * HV * BT + o_hv * BT + j % BT,
            b_Aqk.to(Aqk.dtype.element_ty),
            mask=m_hv,
        )
        tl.store(
            Akk + i_t * HV * BC + o_hv * BC + j - i_ts,
            b_Akk.to(Akk.dtype.element_ty),
            mask=m_hv,
        )


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def chunk_kda_fwd_kernel_intra_subchunk_raw_optimized_hcu(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Compute one complete 16-token diagonal block per program."""
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        i_hv = i_bh
    else:
        i_b, i_hv = i_bh // HV, i_bh % HV
        bos = i_b * T

    i_h = i_hv // (HV // H)
    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_c = i_ti + tl.arange(0, BC)
    m_c = o_c < T
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv
    Aqk += (bos * HV + i_hv) * BT
    Akk += (bos * HV + i_hv) * BC

    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T, K), (HV * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_beta = tl.make_block_ptr(beta, (T,), (HV,), (i_ti,), (BC,), (0,))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)

    o_k = tl.arange(0, BK)
    center = min(BC // 2, T - i_ti - 1)
    b_gn = tl.load(
        g + (i_ti + center) * HV * K + o_k,
        mask=o_k < K,
        other=0.0,
    )[None, :]
    b_gm = (b_g - b_gn).to(tl.float32)
    b_gq = tl.where(m_c[:, None], exp2(b_gm), 0.0)
    b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.0)
    b_right = tl.trans(b_k * b_gk)
    b_Aqk = tl.dot(b_q * b_gq, b_right, input_precision="ieee") * scale
    b_Akk = tl.dot(b_k * b_gq, b_right, input_precision="ieee") * b_beta[:, None]

    o_i = tl.arange(0, BC)
    b_Aqk = tl.where(o_i[:, None] >= o_i[None, :], b_Aqk, 0.0)
    b_Akk = tl.where(o_i[:, None] > o_i[None, :], b_Akk, 0.0)
    p_Aqk = tl.make_block_ptr(
        Aqk, (T, BT), (HV * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0)
    )
    p_Akk = tl.make_block_ptr(
        Akk, (T, BC), (HV * BC, 1), (i_ti, 0), (BC, BC), (1, 0)
    )
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk, b_Akk.to(Akk.dtype.element_ty), boundary_check=(0, 1))


def chunk_kda_fwd_intra_token_parallel_hcu(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    sub_chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-parallel intra-chunk diagonal blocks (HCU variant).

    Writes ``Aqk`` and ``Akk`` (fp32 diagonal buffer) in place, matching
    sglang's ``chunk_kda_fwd_intra_token_parallel`` contract.  For the K3
    shape (K=128, BT=64, BC=16) uses the raw-optimized sub-chunk kernel.
    """
    B, T, H, K, HV = *q.shape, gk.shape[2]
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BT = chunk_size
    BC = sub_chunk_size
    use_subchunk = K == 128 and BT == 64 and BC == 16
    if use_subchunk:
        config = get_chunk_kda_intra_token_parallel_hcu_config(
            T=T, H=H, HV=HV, is_varlen=cu_seqlens is not None
        )
        if cu_seqlens is None:
            chunk_indices = None
            n_chunks = triton.cdiv(T, BT)
            batch_heads = B * HV
        else:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
            n_chunks = len(chunk_indices)
            batch_heads = HV
        grid = (n_chunks, triton.cdiv(BT, BC), batch_heads)
        chunk_kda_fwd_kernel_intra_subchunk_raw_optimized_hcu[grid](
            q=q,
            k=k,
            g=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akk,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            HV=HV,
            K=K,
            BT=BT,
            BC=BC,
            **config,
        )
        return Aqk, Akk

    fallback_config = {"BH": 4, "num_warps": 1}
    grid = (B * T, triton.cdiv(HV, fallback_config["BH"]))
    chunk_kda_fwd_kernel_intra_token_parallel_hcu[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        Aqk=Aqk,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        N=N,
        T=T,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BC=BC,
        **fallback_config,
    )
    return Aqk, Akk


# ---------------------------------------------------------------------------
# Intra chunk: safe-gate sub-chunk kernel (chunk_kda_fwd_kernel_intra_sub_chunk, HCU)
# ---------------------------------------------------------------------------
@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["B", "T"])
def chunk_kda_fwd_kernel_intra_sub_chunk_hcu(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GATHER: tl.constexpr,
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_c = i_ti + tl.arange(0, BC)
    m_c = o_c < T

    q = q + (bos * H + i_h) * K
    k = k + (bos * H + i_h) * K
    g = g + (bos * HV + i_hv) * K
    beta = beta + bos * HV + i_hv
    Aqk = Aqk + (bos * HV + i_hv) * BT
    Akk = Akk + (bos * HV + i_hv) * BC

    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T, K), (HV * K, 1), (i_ti, 0), (BC, BK), (1, 0))

    p_beta = tl.make_block_ptr(beta, (T,), (HV,), (i_ti,), (BC,), (0,))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)

    if USE_GATHER:
        b_gn = gather(
            b_g, tl.full([1, BK], min(BC // 2, T - i_ti - 1), dtype=tl.int16), axis=0
        )
    else:
        # caculate offset
        p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * HV * K + tl.arange(0, BK)
        b_gn = tl.load(p_gn, mask=tl.arange(0, BK) < K, other=0.0)
        b_gn = b_gn[None, :]

    # current block, keep numerical stability by subtracting the left boundary
    # less than 85 to avoid overflow in exp2
    b_gm = (b_g - b_gn).to(tl.float32)

    b_gq = tl.where(m_c[:, None], exp2(b_gm), 0.0)
    b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.0)

    b_kgt = tl.trans(b_k * b_gk)

    b_Aqk = tl.dot(b_q * b_gq, b_kgt) * scale
    b_Akk = tl.dot(b_k * b_gq, b_kgt) * b_beta[:, None]

    o_i = tl.arange(0, BC)
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk, 0.0)

    p_Aqk = tl.make_block_ptr(
        Aqk, (T, BT), (HV * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0)
    )
    p_Akk = tl.make_block_ptr(Akk, (T, BC), (HV * BC, 1), (i_ti, 0), (BC, BC), (1, 0))
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk, b_Akk.to(Akk.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()

    ################################################################################
    # forward substitution
    ################################################################################

    b_Ai = -b_Akk
    for i in range(2, min(BC, T - i_ti)):
        b_a = -tl.load(Akk + (i_ti + i) * HV * BC + o_i)
        b_a = tl.where(o_i < i, b_a, 0.0)
        b_a += tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    b_Ai += m_I
    tl.store(p_Akk, b_Ai.to(Akk.dtype.element_ty), boundary_check=(0, 1))


# Intra chunk: polynomial-solve diagonal blocks (poly_solve_optimized, HCU)
# ---------------------------------------------------------------------------
CHUNK_KDA_INTRA_SUB_CHUNK_HCU_CONFIGS: tuple[dict[str, int | bool], ...] = (
    {"BK": 128, "EARLY_AQK_STORE": False, "num_warps": 1, "num_stages": 1},
    {"BK": 128, "EARLY_AQK_STORE": True, "num_warps": 1, "num_stages": 1},
    {"BK": 128, "EARLY_AQK_STORE": True, "num_warps": 1, "num_stages": 2},
)




def get_chunk_kda_intra_sub_chunk_hcu_config(
    T: int, H: int, HV: int, is_varlen: bool
) -> dict[str, int | bool]:
    """Select one of three offline configs for the polynomial sibling."""
    del T, H, HV, is_varlen
    return CHUNK_KDA_INTRA_SUB_CHUNK_HCU_CONFIGS[0]


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def chunk_kda_fwd_kernel_intra_sub_chunk_poly_solve_optimized_hcu(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    EARLY_AQK_STORE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Invert a strict-lower 16x16 block with its nilpotent polynomial."""
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        i_hv = i_bh
    else:
        i_b, i_hv = i_bh // HV, i_bh % HV
        bos = i_b * T
    i_h = i_hv // (HV // H)
    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_c = i_ti + tl.arange(0, BC)
    m_c = o_c < T
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv
    Aqk += (bos * HV + i_hv) * BT
    Akk += (bos * HV + i_hv) * BC
    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T, K), (HV * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_beta = tl.make_block_ptr(beta, (T,), (HV,), (i_ti,), (BC,), (0,))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
    o_k = tl.arange(0, BK)
    center = min(BC // 2, T - i_ti - 1)
    b_gn = tl.load(
        g + (i_ti + center) * HV * K + o_k, mask=o_k < K, other=0.0
    )[None, :]
    b_gm = (b_g - b_gn).to(tl.float32)
    b_gq = tl.where(m_c[:, None], exp2(b_gm), 0.0)
    b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.0)
    b_right = tl.trans(b_k * b_gk)
    b_Aqk = tl.dot(b_q * b_gq, b_right, input_precision="ieee") * scale
    b_Akk = tl.dot(b_k * b_gq, b_right, input_precision="ieee") * b_beta[:, None]
    o_i = tl.arange(0, BC)
    lower_equal = o_i[:, None] >= o_i[None, :]
    lower_strict = o_i[:, None] > o_i[None, :]
    identity = (o_i[:, None] == o_i[None, :]).to(tl.float32)
    b_Aqk = tl.where(lower_equal, b_Aqk, 0.0)
    A = tl.where(lower_strict, b_Akk, 0.0)

    p_Aqk = tl.make_block_ptr(
        Aqk, (T, BT), (HV * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0)
    )
    if EARLY_AQK_STORE:
        tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))

    # Keep every polynomial input and intermediate in FP32.  Although the
    # public inverse output is BF16, narrowing between products compounds the
    # error across the A2/A4/A8 chain.
    A2 = tl.dot(
        A,
        A,
        input_precision="ieee",
        out_dtype=tl.float32,
    )
    X = tl.dot(
        identity - A,
        identity + A2,
        input_precision="ieee",
        out_dtype=tl.float32,
    )
    A4 = tl.dot(
        A2,
        A2,
        input_precision="ieee",
        out_dtype=tl.float32,
    )
    X = tl.dot(
        X,
        identity + A4,
        input_precision="ieee",
        out_dtype=tl.float32,
    )
    A8 = tl.dot(
        A4,
        A4,
        input_precision="ieee",
        out_dtype=tl.float32,
    )
    X = tl.dot(
        X,
        identity + A8,
        input_precision="ieee",
        out_dtype=tl.float32,
    )

    p_Akk = tl.make_block_ptr(
        Akk, (T, BC), (HV * BC, 1), (i_ti, 0), (BC, BC), (1, 0)
    )
    if not EARLY_AQK_STORE:
        tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk, X.to(Akk.dtype.element_ty), boundary_check=(0, 1))


# ---------------------------------------------------------------------------
# Inter + solve_tril (chunk_kda_fwd_kernel_inter_solve_fused, HCU variant)
# ---------------------------------------------------------------------------
CHUNK_KDA_INTER_SOLVE_HCU_CONFIGS: tuple[dict[str, int], ...] = (
    {"BK": 64, "num_warps": 1, "num_stages": 2},
    {"BK": 64, "num_warps": 1, "num_stages": 1},
    {"BK": 32, "num_warps": 1, "num_stages": 1},
)


def get_chunk_kda_inter_solve_hcu_config(
    T: int,
    H: int,
    HV: int,
    is_varlen: bool,
    safe_gate: bool,
) -> dict[str, int]:
    """Select one of three offline configs for the inter/solve kernel."""
    del H, HV, is_varlen, safe_gate
    if T <= 128:
        return CHUNK_KDA_INTER_SOLVE_HCU_CONFIGS[1]
    return CHUNK_KDA_INTER_SOLVE_HCU_CONFIGS[0]




@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_kda_fwd_kernel_inter_solve_fused_hcu(
    q,
    k,
    g,
    beta,
    Aqk,
    Akkd,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_SAFE_GATE: tl.constexpr,
    SOLVE_TRIL_DOT_PRECISION: tl.constexpr,
):
    """
    Fused kernel: compute inter-subchunk Akk + solve_tril in one pass.
    Prerequisite: token_parallel has already computed diagonal Akk blocks in Akkd.

    This kernel:
    1. Computes off-diagonal Aqk blocks -> writes to global
    2. Computes off-diagonal Akk blocks -> keeps in registers
    3. Loads diagonal Akk blocks from Akkd (fp32)
    4. Does forward substitution on diagonals
    5. Computes merged Akk_inv
    6. Writes Akk_inv to Akk
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    Aqk += (bos * HV + i_hv) * BT
    Akk += (bos * HV + i_hv) * BT
    Akkd += (bos * HV + i_hv) * BC

    o_i = tl.arange(0, BC)
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk32 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)

    ################################################################################
    # off-diagonal blocks
    ################################################################################
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k0 = tl.make_block_ptr(
            k, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
        )
        p_g0 = tl.make_block_ptr(
            g, (T, K), (HV * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
        )
        b_k0 = tl.load(p_k0, boundary_check=(0, 1)).to(tl.float32)
        b_g0 = tl.load(p_g0, boundary_check=(0, 1)).to(tl.float32)

        if i_tc1 < T:
            p_q1 = tl.make_block_ptr(
                q, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            p_k1 = tl.make_block_ptr(
                k, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            p_g1 = tl.make_block_ptr(
                g, (T, K), (HV * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            # [BC, BK]
            b_q1 = tl.load(p_q1, boundary_check=(0, 1)).to(tl.float32)
            b_k1 = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)
            b_g1 = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
            # [BK]
            b_gn1 = tl.load(g + i_tc1 * HV * K + o_k, mask=m_k, other=0).to(tl.float32)
            # [BC, BK]
            b_gqn = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
            # [BK, BC]
            b_kgt = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0))
            # [BC, BC]
            b_Aqk10 += tl.dot(b_q1 * b_gqn, b_kgt)
            b_Akk10 += tl.dot(b_k1 * b_gqn, b_kgt)

            if i_tc2 < T:
                p_q2 = tl.make_block_ptr(
                    q, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                p_k2 = tl.make_block_ptr(
                    k, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                p_g2 = tl.make_block_ptr(
                    g, (T, K), (HV * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                # [BC, BK]
                b_q2 = tl.load(p_q2, boundary_check=(0, 1)).to(tl.float32)
                b_k2 = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
                b_g2 = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
                # [BK]
                b_gn2 = tl.load(g + i_tc2 * HV * K + o_k, mask=m_k, other=0).to(
                    tl.float32
                )
                # [BC, BK]
                b_gqn2 = tl.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), 0)
                b_qg2 = b_q2 * b_gqn2
                b_kg2 = b_k2 * b_gqn2
                # [BK, BC]
                b_kgt = tl.trans(b_k0 * exp2(b_gn2[None, :] - b_g0))
                b_Aqk20 += tl.dot(b_qg2, b_kgt)
                b_Akk20 += tl.dot(b_kg2, b_kgt)
                # [BC, BC]
                b_kgt = tl.trans(b_k1 * exp2(b_gn2[None, :] - b_g1))
                # [BC, BC]
                b_Aqk21 += tl.dot(b_qg2, b_kgt)
                b_Akk21 += tl.dot(b_kg2, b_kgt)

                if i_tc3 < T:
                    p_q3 = tl.make_block_ptr(
                        q, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    p_k3 = tl.make_block_ptr(
                        k, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    p_g3 = tl.make_block_ptr(
                        g, (T, K), (HV * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    # [BC, BK]
                    b_q3 = tl.load(p_q3, boundary_check=(0, 1)).to(tl.float32)
                    b_k3 = tl.load(p_k3, boundary_check=(0, 1)).to(tl.float32)
                    b_g3 = tl.load(p_g3, boundary_check=(0, 1)).to(tl.float32)
                    # [BK]
                    b_gn3 = tl.load(g + i_tc3 * HV * K + o_k, mask=m_k, other=0).to(
                        tl.float32
                    )
                    # [BC, BK]
                    b_gqn3 = tl.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), 0)
                    b_qg3 = b_q3 * b_gqn3
                    b_kg3 = b_k3 * b_gqn3
                    # [BK, BC]
                    b_kgt = tl.trans(b_k0 * exp2(b_gn3[None, :] - b_g0))
                    # [BC, BC]
                    b_Aqk30 += tl.dot(b_qg3, b_kgt)
                    b_Akk30 += tl.dot(b_kg3, b_kgt)
                    # [BK, BC]
                    b_kgt = tl.trans(b_k1 * exp2(b_gn3[None, :] - b_g1))
                    # [BC, BC]
                    b_Aqk31 += tl.dot(b_qg3, b_kgt)
                    b_Akk31 += tl.dot(b_kg3, b_kgt)
                    # [BK, BC]
                    b_kgt = tl.trans(b_k2 * exp2(b_gn3[None, :] - b_g2))
                    # [BC, BC]
                    b_Aqk32 += tl.dot(b_qg3, b_kgt)
                    b_Akk32 += tl.dot(b_kg3, b_kgt)

    ################################################################################
    # save off-diagonal Aqk blocks and prepare Akk
    ################################################################################
    if i_tc1 < T:
        p_Aqk10 = tl.make_block_ptr(
            Aqk, (T, BT), (HV * BT, 1), (i_tc1, 0), (BC, BC), (1, 0)
        )
        tl.store(
            p_Aqk10, (b_Aqk10 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )

        p_b1 = tl.make_block_ptr(
            beta + bos * HV + i_hv, (T,), (HV,), (i_tc1,), (BC,), (0,)
        )
        b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
        b_Akk10 = b_Akk10 * b_b1[:, None]
    if i_tc2 < T:
        p_Aqk20 = tl.make_block_ptr(
            Aqk, (T, BT), (HV * BT, 1), (i_tc2, 0), (BC, BC), (1, 0)
        )
        p_Aqk21 = tl.make_block_ptr(
            Aqk, (T, BT), (HV * BT, 1), (i_tc2, BC), (BC, BC), (1, 0)
        )
        tl.store(
            p_Aqk20, (b_Aqk20 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )
        tl.store(
            p_Aqk21, (b_Aqk21 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )

        p_b2 = tl.make_block_ptr(
            beta + bos * HV + i_hv, (T,), (HV,), (i_tc2,), (BC,), (0,)
        )
        b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
        b_Akk20 = b_Akk20 * b_b2[:, None]
        b_Akk21 = b_Akk21 * b_b2[:, None]
    if i_tc3 < T:
        p_Aqk30 = tl.make_block_ptr(
            Aqk, (T, BT), (HV * BT, 1), (i_tc3, 0), (BC, BC), (1, 0)
        )
        p_Aqk31 = tl.make_block_ptr(
            Aqk, (T, BT), (HV * BT, 1), (i_tc3, BC), (BC, BC), (1, 0)
        )
        p_Aqk32 = tl.make_block_ptr(
            Aqk, (T, BT), (HV * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0)
        )
        tl.store(
            p_Aqk30, (b_Aqk30 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )
        tl.store(
            p_Aqk31, (b_Aqk31 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )
        tl.store(
            p_Aqk32, (b_Aqk32 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )

        p_b3 = tl.make_block_ptr(
            beta + bos * HV + i_hv, (T,), (HV,), (i_tc3,), (BC,), (0,)
        )
        b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)
        b_Akk30 = b_Akk30 * b_b3[:, None]
        b_Akk31 = b_Akk31 * b_b3[:, None]
        b_Akk32 = b_Akk32 * b_b3[:, None]

    p_Akk00 = tl.make_block_ptr(
        Akkd, (T, BC), (HV * BC, 1), (i_tc0, 0), (BC, BC), (1, 0)
    )
    p_Akk11 = tl.make_block_ptr(
        Akkd, (T, BC), (HV * BC, 1), (i_tc1, 0), (BC, BC), (1, 0)
    )
    p_Akk22 = tl.make_block_ptr(
        Akkd, (T, BC), (HV * BC, 1), (i_tc2, 0), (BC, BC), (1, 0)
    )
    p_Akk33 = tl.make_block_ptr(
        Akkd, (T, BC), (HV * BC, 1), (i_tc3, 0), (BC, BC), (1, 0)
    )
    b_Ai00 = tl.load(p_Akk00, boundary_check=(0, 1)).to(tl.float32)
    b_Ai11 = tl.load(p_Akk11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai22 = tl.load(p_Akk22, boundary_check=(0, 1)).to(tl.float32)
    b_Ai33 = tl.load(p_Akk33, boundary_check=(0, 1)).to(tl.float32)

    ################################################################################
    # forward substitution on diagonals
    ################################################################################

    if not USE_SAFE_GATE:
        m_A = o_i[:, None] > o_i[None, :]
        m_I = o_i[:, None] == o_i[None, :]

        b_Ai00 = -tl.where(m_A, b_Ai00, 0)
        b_Ai11 = -tl.where(m_A, b_Ai11, 0)
        b_Ai22 = -tl.where(m_A, b_Ai22, 0)
        b_Ai33 = -tl.where(m_A, b_Ai33, 0)

        for i in range(2, min(BC, T - i_tc0)):
            b_a00 = -tl.load(Akkd + (i_tc0 + i) * HV * BC + o_i)
            b_a00 = tl.where(o_i < i, b_a00, 0.0)
            b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
            b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
        for i in range(BC + 2, min(2 * BC, T - i_tc0)):
            b_a11 = -tl.load(Akkd + (i_tc0 + i) * HV * BC + o_i)
            b_a11 = tl.where(o_i < i - BC, b_a11, 0.0)
            b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
            b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
        for i in range(2 * BC + 2, min(3 * BC, T - i_tc0)):
            b_a22 = -tl.load(Akkd + (i_tc0 + i) * HV * BC + o_i)
            b_a22 = tl.where(o_i < i - 2 * BC, b_a22, 0.0)
            b_a22 += tl.sum(b_a22[:, None] * b_Ai22, 0)
            b_Ai22 = tl.where((o_i == i - 2 * BC)[:, None], b_a22, b_Ai22)
        for i in range(3 * BC + 2, min(4 * BC, T - i_tc0)):
            b_a33 = -tl.load(Akkd + (i_tc0 + i) * HV * BC + o_i)
            b_a33 = tl.where(o_i < i - 3 * BC, b_a33, 0.0)
            b_a33 += tl.sum(b_a33[:, None] * b_Ai33, 0)
            b_Ai33 = tl.where((o_i == i - 3 * BC)[:, None], b_a33, b_Ai33)

        b_Ai00 += m_I
        b_Ai11 += m_I
        b_Ai22 += m_I
        b_Ai33 += m_I

    ################################################################################
    # compute merged inverse using off-diagonals
    ################################################################################

    # we used tf32 to maintain matrix inverse's precision whenever possible.
    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_Akk10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai00,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_Akk21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai11,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_Akk32, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai22,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_Akk20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    ################################################################################
    # store full Akk_inv to Akk
    ################################################################################

    p_Akk00 = tl.make_block_ptr(
        Akk, (T, BT), (HV * BT, 1), (i_tc0, 0), (BC, BC), (1, 0)
    )
    p_Akk10 = tl.make_block_ptr(
        Akk, (T, BT), (HV * BT, 1), (i_tc1, 0), (BC, BC), (1, 0)
    )
    p_Akk11 = tl.make_block_ptr(
        Akk, (T, BT), (HV * BT, 1), (i_tc1, BC), (BC, BC), (1, 0)
    )
    p_Akk20 = tl.make_block_ptr(
        Akk, (T, BT), (HV * BT, 1), (i_tc2, 0), (BC, BC), (1, 0)
    )
    p_Akk21 = tl.make_block_ptr(
        Akk, (T, BT), (HV * BT, 1), (i_tc2, BC), (BC, BC), (1, 0)
    )
    p_Akk22 = tl.make_block_ptr(
        Akk, (T, BT), (HV * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0)
    )
    p_Akk30 = tl.make_block_ptr(
        Akk, (T, BT), (HV * BT, 1), (i_tc3, 0), (BC, BC), (1, 0)
    )
    p_Akk31 = tl.make_block_ptr(
        Akk, (T, BT), (HV * BT, 1), (i_tc3, BC), (BC, BC), (1, 0)
    )
    p_Akk32 = tl.make_block_ptr(
        Akk, (T, BT), (HV * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0)
    )
    p_Akk33 = tl.make_block_ptr(
        Akk, (T, BT), (HV * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0)
    )

    tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk11, b_Ai11.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk20, b_Ai20.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk21, b_Ai21.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk22, b_Ai22.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk30, b_Ai30.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk31, b_Ai31.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk32, b_Ai32.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk33, b_Ai33.to(Akk.dtype.element_ty), boundary_check=(0, 1))


# ---------------------------------------------------------------------------
# Full intra wrapper (chunk_kda_fwd_intra, HCU): step-1 variants + step-2 inter/solve
# ---------------------------------------------------------------------------
def chunk_kda_fwd_intra_hcu(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    safe_gate: bool = False,
):
    B, T, H, K, HV = *k.shape, gk.shape[2]
    BT = chunk_size
    BC = 16
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)

    Aqk = torch.empty(B, T, HV, BT, device=k.device, dtype=k.dtype)
    # Akk must be zero-initialized - kernel only writes lower triangular
    Akk = torch.zeros(B, T, HV, BT, device=k.device, dtype=k.dtype)
    # Separate fp32 buffer for diagonal 16x16 blocks (for precision in solve_tril)
    Akkd = torch.empty(B, T, HV, BC, device=k.device, dtype=torch.float32)

    # Compute diagonal blocks into Akkd in fp32.  For the model-shape class the
    # polynomial sibling writes the already-inverted 16x16 diagonal blocks, so
    # the inter kernel can skip four serial forward-substitution loops.  The
    # public token-parallel wrapper still retains its raw-Akkd contract.
    use_poly_solve = K == 128 and BT == 64 and BC == 16
    if use_poly_solve:
        grid = (NT, NC, B * HV if cu_seqlens is None else HV)
        config = get_chunk_kda_intra_sub_chunk_hcu_config(
            T=T,
            H=H,
            HV=HV,
            is_varlen=cu_seqlens is not None,
        )
        chunk_kda_fwd_kernel_intra_sub_chunk_poly_solve_optimized_hcu[grid](
            q=q,
            k=k,
            g=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            HV=HV,
            K=K,
            BT=BT,
            BC=BC,
            **config,
        )
    elif safe_gate:
        grid = (NT, NC, B * HV)
        BK = triton.next_power_of_2(K)
        fallback_config = {"num_warps": 1, "num_stages": 2}
        chunk_kda_fwd_kernel_intra_sub_chunk_hcu[grid](
            q=q,
            k=k,
            g=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            HV=HV,
            K=K,
            BT=BT,
            BC=BC,
            BK=BK,
            USE_GATHER=is_gather_supported,
            **fallback_config,
        )
    else:
        Aqk, Akkd = chunk_kda_fwd_intra_token_parallel_hcu(
            q=q,
            k=k,
            gk=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=BT,
            sub_chunk_size=BC,
        )

    # Step 2: Fused inter + solve_tril (works for both fixed-len and varlen)
    solve_tril_dot_precision = (
        "tf32"
        if is_tf32_supported
        else "ieee"
    )
    grid = (NT, B * HV)
    diagonal_is_presolved = safe_gate or use_poly_solve
    config = get_chunk_kda_inter_solve_hcu_config(
        T=T,
        H=H,
        HV=HV,
        is_varlen=cu_seqlens is not None,
        safe_gate=diagonal_is_presolved,
    )
    chunk_kda_fwd_kernel_inter_solve_fused_hcu[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        Aqk=Aqk,
        Akkd=Akkd,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BC=BC,
        USE_SAFE_GATE=diagonal_is_presolved,
        SOLVE_TRIL_DOT_PRECISION=solve_tril_dot_precision,
        **config,
    )
    return Aqk, Akk

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code copied from the flash-linear-attention project.
# The original source was licensed under the MIT license.
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Forward-only adaptation of flash-linear-attention 0.5.0.
# ruff: noqa: E501

# Token-parallel implementation of KDA intra chunk kernel
CHUNK_KDA_INTRA_TOKEN_PARALLEL_CONFIGS: tuple[dict[str, int], ...] = (
    {"BK": 128, "num_warps": 1, "num_stages": 1},
    {"BK": 128, "num_warps": 1, "num_stages": 2},
    {"BK": 128, "num_warps": 1, "num_stages": 3},
)


# Offline candidates retained on BW1100.  The new recompute/output champions
# were selected at B=1, T=8192, H=24, K=V=128; wrappers select one config
# directly and never run online autotuning.
RECOMPUTE_W_U_FWD_CONFIGS = (
    {"BK": 64, "BV": 128, "num_warps": 4, "num_stages": 2},
    {"BK": 64, "BV": 128, "num_warps": 4, "num_stages": 1},
    {"BK": 64, "BV": 64, "num_warps": 4, "num_stages": 2},
)
CHUNK_GLA_FWD_O_CONFIGS = (
    {"BK": 128, "BV": 128, "num_warps": 4, "num_stages": 2},
    {"BK": 128, "BV": 128, "num_warps": 8, "num_stages": 2},
    {"BK": 64, "BV": 64, "num_warps": 8, "num_stages": 2},
)


def get_recompute_w_u_fwd_hcu_config(
    _T: int, _H: int, _K: int, V: int, _is_varlen: bool
) -> dict:
    if V < 128:
        return RECOMPUTE_W_U_FWD_CONFIGS[2]
    return RECOMPUTE_W_U_FWD_CONFIGS[0]


def get_chunk_gla_fwd_o_hcu_config(
    T: int, H: int, K: int, V: int, is_varlen: bool
) -> dict:
    del T, H, is_varlen
    if K >= 128 and V >= 128:
        return CHUNK_GLA_FWD_O_CONFIGS[0]
    return CHUNK_GLA_FWD_O_CONFIGS[2]




@triton.heuristics(
    {
        "STORE_QG": lambda args: args["qg"] is not None,
        "STORE_KG": lambda args: args["kg"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel_hcu(
    q,
    k,
    qg,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    p_b = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,)).to(tl.float32)

    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_A = tl.load(p_A, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, input_precision=DOT_PRECISION)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_b[:, None]

        p_gk = tl.make_block_ptr(
            gk + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kb *= exp2(b_gk)
        if STORE_QG:
            p_q = tl.make_block_ptr(
                q + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            p_qg = tl.make_block_ptr(
                qg + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_qg = b_q * exp2(b_gk)
            tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), boundary_check=(0, 1))
        if STORE_KG:
            last_idx = min(i_t * BT + BT, T) - 1

            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            b_gn = tl.load(
                gk + ((bos + last_idx) * H + i_h) * K + o_k, mask=m_k, other=0.0
            )
            b_kg = b_k * exp2(b_gn - b_gk)

            p_kg = tl.make_block_ptr(
                kg + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))

        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))




@triton.heuristics(
    {
        "STORE_QG": lambda args: args["qg"] is not None,
        "STORE_KG": lambda args: args["kg"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel_beta_factored_head_first_hcu(
    q,
    k,
    qg,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """Head-first sibling that factors beta into the shared A tile."""
    i_bh, i_tc = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_tc * 2).to(tl.int32),
            tl.load(chunk_indices + i_tc * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        i_t = i_tc
        bos, eos = i_b * T, i_b * T + T

    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h,
        (T,),
        (H,),
        (i_t * BT,),
        (BT,),
        (0,),
    )
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT,
        (T, BT),
        (H * BT, 1),
        (i_t * BT, 0),
        (BT, BT),
        (1, 0),
    )
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
    b_Ab = (
        tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
        * b_beta[None, :]
    ).to(tl.bfloat16)

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_u = tl.dot(b_Ab, b_v, input_precision=DOT_PRECISION)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_gk = tl.make_block_ptr(
            gk + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
        b_kgate = (b_k * exp2(b_gk)).to(b_k.dtype)
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_w = tl.dot(b_Ab, b_kgate, input_precision=DOT_PRECISION)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))

        if STORE_QG:
            p_q = tl.make_block_ptr(
                q + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            p_qg = tl.make_block_ptr(
                qg + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            b_q = tl.load(p_q, boundary_check=(0, 1))
            tl.store(
                p_qg,
                (b_q * exp2(b_gk)).to(p_qg.dtype.element_ty),
                boundary_check=(0, 1),
            )

        if STORE_KG:
            last_idx = min(i_t * BT + BT, T) - 1
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            b_gn = tl.load(
                gk + ((bos + last_idx) * H + i_h) * K + o_k,
                mask=m_k,
                other=0.0,
            ).to(tl.float32)
            p_kg = tl.make_block_ptr(
                kg + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            b_kg = b_k * exp2(b_gn[None, :] - b_gk)
            tl.store(
                p_kg,
                b_kg.to(p_kg.dtype.element_ty),
                boundary_check=(0, 1),
            )




def recompute_w_u_fwd_hcu(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    q: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor | None]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    kg = torch.empty_like(k) if gk is not None else None
    config = get_recompute_w_u_fwd_hcu_config(
        T, H, K, V, cu_seqlens is not None
    )
    use_beta_factored_head_first = (
        k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and A.dtype == torch.bfloat16
        and gk is not None
        and BT == 64
        and K <= 128
        and V <= 128
        and (cu_seqlens is None or B == 1)
    )
    kernel = (
        recompute_w_u_fwd_kernel_beta_factored_head_first_hcu
        if use_beta_factored_head_first
        else recompute_w_u_fwd_kernel_hcu
    )
    grid = (B * H, NT) if use_beta_factored_head_first else (NT, B * H)
    kernel[grid](
        q=q,
        k=k,
        qg=None,
        kg=kg,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        DOT_PRECISION="ieee",
        **config,
    )
    return w, u, None, kg




@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_kernel_o_hcu(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_g = tl.make_block_ptr(
            g + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_h = tl.make_block_ptr(
            h + (i_tg * H + i_h) * K * V,
            (V, K),
            (K, 1),
            (i_v * BV, i_k * BK),
            (BV, BK),
            (1, 0),
        )

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BK]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        # [BT, BK]
        b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BT, BV]
        if i_k >= 0:
            b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    # [BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))




@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_kernel_o_local_first_hcu(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Chunk output with the causal local term scheduled before state."""
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        i_tg = i_b * tl.cdiv(T, BT) + i_t
        bos, eos = i_b * T, i_b * T + T

    rows = tl.arange(0, BT)
    causal = rows[:, None] >= rows[None, :]
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT,
        (T, BT),
        (H * BT, 1),
        (i_t * BT, 0),
        (BT, BT),
        (1, 0),
    )
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(causal, b_A, 0.0).to(b_v.dtype)
    b_o = tl.dot(b_A, b_v, allow_tf32=False)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_g = tl.make_block_ptr(
            g + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_h = tl.make_block_ptr(
            h + (i_tg * H + i_h) * K * V,
            (V, K),
            (K, 1),
            (i_v * BV, i_k * BK),
            (BV, BK),
            (1, 0),
        )
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))

    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))




def chunk_gla_fwd_o_gk_hcu(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    o: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
):
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    config = get_chunk_gla_fwd_o_hcu_config(
        T, H, K, V, cu_seqlens is not None
    )
    grid = (cdiv(V, config["BV"]), NT, B * H)

    kernel = (
        chunk_gla_fwd_kernel_o_local_first_hcu
        if K >= 128 and V >= 128 and BT == 64
        else chunk_gla_fwd_kernel_o_hcu
    )
    kernel[grid](
        q=q,
        v=v,
        g=g,
        h=h,
        o=o,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        **config,
    )
    return o




FUSED_RECURRENT_KDA_PACKED_DECODE_CONFIGS = (
    {"BV": 8, "num_warps": 1, "num_stages": 2},  # B=8
    {"BV": 8, "num_warps": 1, "num_stages": 1},  # B<=6
    {"BV": 32, "num_warps": 2, "num_stages": 2},  # B=1 or B>=16
)




def get_fused_recurrent_kda_packed_decode_hcu_config(
    B: int,
    H: int,
    K: int,
    V: int,
) -> dict[str, int]:
    if K < 128 or V < 128:
        return FUSED_RECURRENT_KDA_PACKED_DECODE_CONFIGS[1]
    if B <= 1:
        return FUSED_RECURRENT_KDA_PACKED_DECODE_CONFIGS[2]
    if B <= 6:
        return FUSED_RECURRENT_KDA_PACKED_DECODE_CONFIGS[1]
    if B < 16:
        return FUSED_RECURRENT_KDA_PACKED_DECODE_CONFIGS[0]
    return FUSED_RECURRENT_KDA_PACKED_DECODE_CONFIGS[2]


@triton.jit
def fused_recurrent_kda_packed_decode_kernel_hcu(
    mixed_qkv,
    raw_g,
    raw_beta,
    A_log,
    dt_bias,
    out,
    state,
    state_indices,
    lower_bound,
    scale: tl.constexpr,
    stride_mixed_token: tl.constexpr,
    stride_g_token: tl.constexpr,
    stride_beta_token: tl.constexpr,
    stride_state_token: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(state_indices + i_n).to(tl.int64)
    p_out = out + (i_n * H + i_h) * V + o_v
    if state_idx <= 0:
        tl.store(p_out, tl.zeros([BV], dtype=tl.float32), mask=mask_v)
        return

    p_state = state + state_idx * stride_state_token
    p_state += i_h * V * K + o_v[:, None] * K + o_k[None, :]
    b_state = tl.load(p_state, mask=mask_state, other=0).to(tl.float32)

    # Q, K, and V occupy consecutive channel ranges, while the token stride
    # may also include the output-gate projection that follows packed QKV.
    p_mixed = mixed_qkv + i_n * stride_mixed_token
    b_q = tl.load(p_mixed + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(
        p_mixed + H * K + i_h * K + o_k,
        mask=mask_k,
        other=0,
    ).to(tl.float32)
    b_v = tl.load(
        p_mixed + 2 * H * K + i_h * V + o_v,
        mask=mask_v,
        other=0,
    ).to(tl.float32)

    b_q /= tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k /= tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q *= scale

    p_g = raw_g + i_n * stride_g_token + i_h * K + o_k
    b_g = tl.load(p_g, mask=mask_k, other=0).to(tl.float32)
    b_bias = tl.load(dt_bias + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
    b_a = exp(tl.load(A_log + i_h).to(tl.float32))
    b_g += b_bias
    if USE_LOWER_BOUND:
        b_gate = lower_bound * tl.sigmoid(b_a * b_g)
    else:
        b_softplus = tl.where(
            b_g > SOFTPLUS_THRESHOLD,
            b_g,
            log(1.0 + tl.exp(b_g)),
        )
        b_gate = -b_a * b_softplus

    b_state *= exp(b_gate[None, :])
    b_v -= tl.sum(b_state * b_k[None, :], axis=1)
    b_beta = tl.sigmoid(
        tl.load(raw_beta + i_n * stride_beta_token + i_h).to(tl.float32)
    )
    b_v *= b_beta
    b_state += b_v[:, None] * b_k[None, :]
    b_out = tl.sum(b_state * b_q[None, :], axis=1)

    tl.store(p_out, b_out.to(p_out.dtype.element_ty), mask=mask_v)
    tl.store(p_state, b_state.to(p_state.dtype.element_ty), mask=mask_state)


def fused_recurrent_kda_packed_decode_hcu(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
    lower_bound: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SGLang-contract adapter over the operator-team packed-decode kernels.

    ``a`` is the raw per-K gate ``[B, HV*K]`` and ``b`` the raw per-head beta
    logits ``[B, HV]`` (sigmoid is applied inside the kernels), matching
    ``fused_recurrent_kda_packed_decode`` in ``fla/fused_recurrent.py``.  The
    teacher kernels assume HV == H and always L2-normalize Q/K, so callers
    should only route here when those conditions hold (see the dispatch).

    Returns ``(out, initial_state)`` with ``out`` left in the SGLang layout
    ``[B, 1, H, V]``; the teacher kernels only index it by ``(token, head)``
    so the same pointer arithmetic covers both contiguous layouts.
    """
    if mixed_qkv.ndim != 2 or mixed_qkv.stride(-1) != 1:
        raise ValueError("`mixed_qkv` must be 2D and contiguous in its last dim.")
    if a.ndim != 2 or a.stride(-1) != 1:
        raise ValueError("`a` must be 2D and contiguous in the last dim.")
    if b.ndim != 2 or b.stride(-1) != 1:
        raise ValueError("`b` must be 2D and contiguous in the last dim.")
    if initial_state.ndim != 4 or initial_state.stride(-1) != 1:
        raise ValueError("`initial_state` must be a contiguous 4D cache pool.")
    if ssm_state_indices.ndim != 1 or ssm_state_indices.stride(0) != 1:
        raise ValueError("`ssm_state_indices` must be contiguous and 1D.")
    if not out.is_contiguous():
        raise ValueError("`out` must be contiguous.")
    if A_log.ndim != 1 or not A_log.is_contiguous():
        raise ValueError("`A_log` must be contiguous and one-dimensional.")
    if not dt_bias.is_contiguous():
        raise ValueError("`dt_bias` must be contiguous.")

    device = mixed_qkv.device
    if any(
        x.device != device
        for x in (a, b, A_log, dt_bias, initial_state, out, ssm_state_indices)
    ):
        raise ValueError("All packed KDA inputs must be on the same device.")

    B = mixed_qkv.shape[0]
    H = b.shape[1]  # teacher kernels assume num_q_heads == num_kv_heads
    _, V, K = initial_state.shape[-3:]
    if a.shape != (B, H * K):
        raise ValueError(f"`a` must have shape [B, {H * K}] (got {tuple(a.shape)}).")
    if b.shape != (B, H):
        raise ValueError(f"`b` must have shape [B, {H}] (got {tuple(b.shape)}).")
    if mixed_qkv.shape[1] != 2 * H * K + H * V:
        raise ValueError(f"Unexpected packed QKV shape {tuple(mixed_qkv.shape)}.")
    if A_log.numel() != H or dt_bias.numel() != H * K:
        raise ValueError("`A_log` or `dt_bias` has an incompatible shape.")
    if ssm_state_indices.shape[0] != B:
        raise ValueError("`ssm_state_indices` must contain one entry per token.")
    if out.shape != (B, 1, H, V):
        raise ValueError(
            f"`out` must have shape {(B, 1, H, V)} (got out.shape={tuple(out.shape)})."
        )

    # Teacher kernels consume raw [1, B, H, K] / [1, B, H] inputs.
    raw_g = a.view(B, H, K).unsqueeze(0)
    raw_beta = b.view(B, H).unsqueeze(0)

    BK = next_power_of_2(K)
    if scale is None:
        scale = K ** -0.5
    config = get_fused_recurrent_kda_packed_decode_hcu_config(B, H, K, V)
    use_direct_3d = (
        lower_bound is not None
        and K == 128
        and V == 128
        and initial_state.dtype == torch.float32
    )
    if use_direct_3d:
        grid = (cdiv(V, config["BV"]), H, B)
        fused_recurrent_kda_packed_decode_3d_kernel_hcu[grid](
            mixed_qkv=mixed_qkv,
            raw_g=raw_g,
            raw_beta=raw_beta,
            A_log=A_log,
            dt_bias=dt_bias,
            out=out,
            state=initial_state,
            state_indices=ssm_state_indices,
            lower_bound=lower_bound,
            scale=scale,
            stride_mixed_token=mixed_qkv.stride(0),
            stride_g_token=raw_g.stride(1),
            stride_beta_token=raw_beta.stride(1),
            stride_state_token=initial_state.stride(0),
            H=H,
            K=K,
            V=V,
            BK=BK,
            **config,
        )
    else:
        config = {"BV": 16, "num_warps": 1, "num_stages": 3}
        grid = (cdiv(V, config["BV"]), B * H)
        fused_recurrent_kda_packed_decode_kernel_hcu[grid](
            mixed_qkv=mixed_qkv,
            raw_g=raw_g,
            raw_beta=raw_beta,
            A_log=A_log,
            dt_bias=dt_bias,
            out=out,
            state=initial_state,
            state_indices=ssm_state_indices,
            lower_bound=lower_bound or 0.0,
            scale=scale,
            stride_mixed_token=mixed_qkv.stride(0),
            stride_g_token=raw_g.stride(1),
            stride_beta_token=raw_beta.stride(1),
            stride_state_token=initial_state.stride(0),
            H=H,
            K=K,
            V=V,
            BK=BK,
            SOFTPLUS_THRESHOLD=20.0,
            USE_LOWER_BOUND=lower_bound is not None,
            **config,
        )
    return out, initial_state


@triton.jit
def fused_recurrent_kda_packed_decode_3d_kernel_hcu(
    mixed_qkv,
    raw_g,
    raw_beta,
    A_log,
    dt_bias,
    out,
    state,
    state_indices,
    lower_bound,
    scale: tl.constexpr,
    stride_mixed_token: tl.constexpr,
    stride_g_token: tl.constexpr,
    stride_beta_token: tl.constexpr,
    stride_state_token: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """Bounded-gate packed decode with direct (V tile, head, token) IDs."""
    i_v, i_h, i_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V

    state_idx = tl.load(state_indices + i_n).to(tl.int64)
    p_out = out + (i_n * H + i_h) * V + o_v
    if state_idx <= 0:
        tl.store(p_out, 0.0, mask=mask_v)
        return

    p_mixed = mixed_qkv + i_n * stride_mixed_token
    b_q = tl.load(
        p_mixed + i_h * K + o_k, mask=mask_k, other=0.0
    ).to(tl.float32)
    b_k = tl.load(
        p_mixed + H * K + i_h * K + o_k,
        mask=mask_k,
        other=0.0,
    ).to(tl.float32)
    b_q /= tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k /= tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q *= scale

    b_g = tl.load(
        raw_g + i_n * stride_g_token + i_h * K + o_k,
        mask=mask_k,
        other=0.0,
    ).to(tl.float32)
    b_bias = tl.load(
        dt_bias + i_h * K + o_k, mask=mask_k, other=0.0
    ).to(tl.float32)
    b_a = exp(tl.load(A_log + i_h).to(tl.float32))
    b_gate = lower_bound * tl.sigmoid(b_a * (b_g + b_bias))
    b_beta = tl.sigmoid(
        tl.load(raw_beta + i_n * stride_beta_token + i_h).to(tl.float32)
    )

    p_state = (
        state
        + state_idx * stride_state_token
        + i_h * V * K
        + o_v[:, None] * K
        + o_k[None, :]
    )
    mask_state = mask_v[:, None] & mask_k[None, :]
    b_state = tl.load(p_state, mask=mask_state, other=0.0).to(tl.float32)
    b_state *= exp(b_gate)[None, :]
    b_v = tl.load(
        p_mixed + 2 * H * K + i_h * V + o_v,
        mask=mask_v,
        other=0.0,
    ).to(tl.float32)
    b_v = (b_v - tl.sum(b_state * b_k[None, :], axis=1)) * b_beta
    b_state += b_v[:, None] * b_k[None, :]
    b_out = tl.sum(b_state * b_q[None, :], axis=1)

    tl.store(p_out, b_out.to(p_out.dtype.element_ty), mask=mask_v)
    tl.store(p_state, b_state.to(p_state.dtype.element_ty), mask=mask_state)
