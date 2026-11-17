# SPDX-License-Identifier: Apache-2.0
"""Opt-in HCU affine Q/K RMSNorm + partial NeoX RoPE for the H3 DiT.

Specialized for BF16, head_dim=128 and rotary_dim=96. Token and head
counts are dynamic, and actual Q/K row and head strides are used.
Normalize and rotate in place while preserving the V slice of packed QKV.

This kernel is not bit-exact to the native split chain: reduction and
floating-point evaluation can differ. Explicitly round normalized values
to BF16 before rotation. Numerical tolerance checks do not establish
full-model output equivalence.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_hcu
from sglang.srt.utils.custom_op import register_custom_op

_HEAD_DIM = 128
_ROTARY_DIM = 96


@triton.jit
def _minimax_h3_dit_qknorm_rope_kernel(
    Q,
    K,
    WQ,
    WK,
    CACHE,
    POS,
    Q_ROW: tl.constexpr,
    K_ROW: tl.constexpr,
    Q_HEAD: tl.constexpr,
    K_HEAD: tl.constexpr,
    EPS: tl.constexpr,
    D: tl.constexpr = _HEAD_DIM,
    R: tl.constexpr = _ROTARY_DIM,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    i = tl.arange(0, D)
    qptr = Q + token * Q_ROW + head * Q_HEAD + i
    kptr = K + token * K_ROW + head * K_HEAD + i
    q = tl.load(qptr).to(tl.float32)
    k = tl.load(kptr).to(tl.float32)
    wq = tl.load(WQ + i).to(tl.float32)
    wk = tl.load(WK + i).to(tl.float32)
    qscale = tl.rsqrt(tl.sum(q * q, 0) / D + EPS)
    kscale = tl.rsqrt(tl.sum(k * k, 0) / D + EPS)
    # Round to the activation dtype before RoPE, matching the affine
    # RMSNorm's own output dtype at that point in the split chain.
    qn = (q * qscale * wq).to(tl.bfloat16).to(tl.float32)
    kn = (k * kscale * wk).to(tl.bfloat16).to(tl.float32)
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
    tl.store(qptr, qo)
    tl.store(kptr, ko)


def can_use_minimax_h3_dit_qknorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps_q: float,
    eps_k: float,
    positions: torch.Tensor | None = None,
) -> bool:
    """Check metadata; position values follow the native cache-index contract."""
    if (
        not is_hcu()
        or not q.is_cuda
        or q.ndim != 3
        or q.shape != k.shape
        or q.shape[-1] != _HEAD_DIM
        or q.shape[0] == 0
        or q.shape[1] == 0
        or q.dtype != torch.bfloat16
        or k.dtype != q.dtype
        or k.device != q.device
        or eps_q is None
        or eps_q != eps_k
        or not math.isfinite(eps_q)
        or eps_q <= 0
    ):
        return False
    # In-place output requires non-overlapping rows/heads. Packed QKV views
    # satisfy these constraints without a copy.
    if not all(
        t.stride(-1) == 1
        and t.stride(1) >= _HEAD_DIM
        and t.stride(0) >= (t.shape[1] - 1) * t.stride(1) + _HEAD_DIM
        for t in (q, k)
    ):
        return False
    if not all(
        isinstance(w, torch.Tensor)
        and w.device == q.device
        and w.dtype == q.dtype
        and w.shape == (_HEAD_DIM,)
        and w.is_contiguous()
        for w in (q_weight, k_weight)
    ):
        return False
    return (
        isinstance(cos_sin_cache, torch.Tensor)
        and cos_sin_cache.device == q.device
        and cos_sin_cache.dtype == q.dtype
        and cos_sin_cache.ndim == 2
        and cos_sin_cache.shape[0] >= q.shape[0]
        and cos_sin_cache.shape[1] == _ROTARY_DIM
        and cos_sin_cache.is_contiguous()
        and isinstance(positions, torch.Tensor)
        and positions.device == q.device
        and positions.dtype in (torch.int32, torch.int64)
        and positions.shape == (q.shape[0],)
        and positions.is_contiguous()
    )


@register_custom_op(mutates_args=["q", "k"])
def minimax_h3_dit_qknorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
) -> None:
    """In-place fused affine RMSNorm + partial NeoX RoPE for the H3 DiT.

    ``q``/``k`` are ``[tokens, heads, 128]`` contiguous-in-last-dim views;
    rotated and normalized in place. ``cos_sin_cache`` is
    ``[tokens, 96] = [cos_half(48) | sin_half(48)]``. Call
    ``can_use_minimax_h3_dit_qknorm_rope`` first; this function does not
    re-validate every condition.
    """
    if not can_use_minimax_h3_dit_qknorm_rope(
        q, k, q_weight, k_weight, cos_sin_cache, eps, eps, positions=positions
    ):
        raise ValueError("Unsupported H3 DiT QK-norm/RoPE input metadata")
    with torch.cuda.device(q.device):
        grid = (q.shape[0], q.shape[1])
        _minimax_h3_dit_qknorm_rope_kernel[grid](
            q,
            k,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            q.stride(0),
            k.stride(0),
            q.stride(1),
            k.stride(1),
            eps,
            num_warps=4,
            enable_fp_fusion=False,
        )


__all__ = ["can_use_minimax_h3_dit_qknorm_rope", "minimax_h3_dit_qknorm_rope"]
