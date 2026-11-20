# SPDX-License-Identifier: Apache-2.0
"""Opt-in HCU VAE Q/K RMSNorm + partial NeoX RoPE.

Specialized for FP16, head_dim=64, rotary_dim=48 and batch size one;
token and head counts are dynamic. Read interleaved QKV views directly
and return contiguous Q/K outputs without mutating Q, K or V.

This kernel is not bit-exact to the native split chain: reduction and
floating-point evaluation can differ. It explicitly rounds normalized
values to FP16 before rotation. Numerical tolerance checks do not certify
full-model output equivalence.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_hcu

if TYPE_CHECKING:
    from torch import nn

_HEAD_DIM = 64
_ROTARY_DIM = 48


@triton.jit
def _minimax_h3_vae_qknorm_rope_kernel(
    Q,
    K,
    QOUT,
    KOUT,
    CACHE,
    POS,
    IN_ROW: tl.constexpr,
    IN_HEAD: tl.constexpr,
    OUT_ROW: tl.constexpr,
    OUT_HEAD: tl.constexpr,
    EPS: tl.constexpr,
    D: tl.constexpr = _HEAD_DIM,
    R: tl.constexpr = _ROTARY_DIM,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    i = tl.arange(0, D)
    q = tl.load(Q + token * IN_ROW + head * IN_HEAD + i).to(tl.float32)
    k = tl.load(K + token * IN_ROW + head * IN_HEAD + i).to(tl.float32)
    # Unaffine RMSNorm (elementwise_affine=False): no weight multiply.
    qn = (
        (q * tl.rsqrt(tl.sum(q * q, 0) / D + EPS))
        .to(QOUT.dtype.element_ty)
        .to(tl.float32)
    )
    kn = (
        (k * tl.rsqrt(tl.sum(k * k, 0) / D + EPS))
        .to(KOUT.dtype.element_ty)
        .to(tl.float32)
    )
    partner = tl.where(i < R // 2, i + R // 2, tl.where(i < R, i - R // 2, i))
    qp = tl.gather(qn, partner, 0)
    kp = tl.gather(kn, partner, 0)
    pos = tl.load(POS + token)
    freq = i % (R // 2)
    cos = tl.load(CACHE + pos * R + freq).to(tl.float32)
    sin = tl.load(CACHE + pos * R + R // 2 + freq).to(tl.float32)
    sign = tl.where(i < R // 2, -1.0, 1.0)
    qo = tl.where(i < R, qn * cos + sign * qp * sin, qn)
    ko = tl.where(i < R, kn * cos + sign * kp * sin, kn)
    tl.store(QOUT + token * OUT_ROW + head * OUT_HEAD + i, qo)
    tl.store(KOUT + token * OUT_ROW + head * OUT_HEAD + i, ko)


def _supports_inputs(query, key, cache, positions):
    if (
        not is_hcu()
        or not query.is_cuda
        or query.ndim != 4
        or query.shape[0] != 1
        or query.shape[-1] != _HEAD_DIM
        or query.shape[1] == 0
        or query.shape[2] == 0
        or key.shape != query.shape
        or query.dtype != torch.float16
        or key.dtype != query.dtype
        or key.device != query.device
        or query.stride(-1) != 1
        or key.stride() != query.stride()
        or any(stride < 0 for stride in query.stride())
    ):
        return False
    return (
        isinstance(cache, torch.Tensor)
        and isinstance(positions, torch.Tensor)
        and cache.device == query.device
        and cache.dtype == query.dtype
        and cache.shape == (query.shape[1], _ROTARY_DIM)
        and cache.is_contiguous()
        and positions.device == query.device
        and positions.dtype in (torch.int32, torch.int64)
        and positions.shape == (query.shape[1],)
        and positions.is_contiguous()
    )


def can_use_minimax_h3_vae_qknorm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    rotary_pos_emb: tuple | None,
    norm_q: nn.Module,
    norm_k: nn.Module,
) -> bool:
    """Check the supported norm/layout/cache metadata without GPU reads.

    Position values must be valid cache row indices, as in the native RoPE
    contract. The caller retains native execution for unsupported metadata.
    """
    if not all(
        isinstance(norm, torch.nn.RMSNorm)
        and tuple(norm.normalized_shape) == (_HEAD_DIM,)
        and norm.weight is None
        for norm in (norm_q, norm_k)
    ):
        return False
    if (
        norm_q.eps is None
        or norm_q.eps != norm_k.eps
        or not math.isfinite(norm_q.eps)
        or norm_q.eps <= 0
        or not isinstance(rotary_pos_emb, (tuple, list))
        or len(rotary_pos_emb) != 4
    ):
        return False
    return _supports_inputs(query, key, *rotary_pos_emb[2:])


def minimax_h3_vae_qknorm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fused unaffine RMSNorm + partial NeoX RoPE for the H3 VAE ViT.

    ``query``/``key`` are the ``[1, S, H, 64]`` strided chunks of the fused
    QKV projection (head stride 192). ``cache`` is the ``[S, 48]``
    ``[cos_half(24) | sin_half(24)]`` table, ``positions`` is ``arange(S)``.
    Returns contiguous ``(query, key)``, or ``None`` when the caller should
    fall back to the split RMSNorm + rotary chain (call
    ``can_use_minimax_h3_vae_qknorm_rope`` first; this function does not
    re-validate every condition).
    """
    if not _supports_inputs(query, key, cache, positions):
        return None
    if eps is None or not math.isfinite(eps) or eps <= 0:
        return None
    with torch.cuda.device(query.device):
        qout = torch.empty(query.shape, dtype=query.dtype, device=query.device)
        kout = torch.empty_like(qout)
        grid = (query.shape[1], query.shape[2])
        _minimax_h3_vae_qknorm_rope_kernel[grid](
            query,
            key,
            qout,
            kout,
            cache,
            positions,
            query.stride(1),
            query.stride(2),
            qout.stride(1),
            qout.stride(2),
            eps,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return qout, kout


__all__ = ["can_use_minimax_h3_vae_qknorm_rope", "minimax_h3_vae_qknorm_rope"]
