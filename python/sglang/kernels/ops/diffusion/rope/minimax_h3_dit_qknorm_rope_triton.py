# SPDX-License-Identifier: Apache-2.0
"""Opt-in HCU affine Q/K RMSNorm + partial NeoX RoPE for the H3 DiT.

Specialized for BF16, head_dim=128 and rotary_dim=96. Token and head
counts are dynamic, and actual Q/K row and head strides are used.
Normalize and rotate in place while preserving the V slice of packed QKV.
Each program handles four heads from one token and shares its RoPE cache.

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
_HEADS_PER_BLOCK = 4


@triton.jit
def _normalize_and_rotate(
    ptr,
    weight,
    col,
    valid_head,
    cos,
    sin,
    EPS: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
):
    half_rotary: tl.constexpr = R // 2
    rotary_mask = col < half_rotary
    tail_mask = col < D - R

    # Pair the two rotary halves directly, and include the unrotated tail in
    # the same per-head RMS reduction. Each column belongs to one pair.
    first = tl.load(ptr + col, valid_head & rotary_mask, 0).to(tl.float32)
    second = tl.load(ptr + half_rotary + col, valid_head & rotary_mask, 0).to(
        tl.float32
    )
    tail = tl.load(ptr + R + col, valid_head & tail_mask, 0).to(tl.float32)
    sum_squares = tl.sum(first * first + second * second + tail * tail, axis=1)
    inv = tl.rsqrt(sum_squares / D + EPS)[:, None]

    first_weight = tl.load(weight + col, rotary_mask, 0).to(tl.float32)
    second_weight = tl.load(weight + half_rotary + col, rotary_mask, 0).to(tl.float32)
    tail_weight = tl.load(weight + R + col, tail_mask, 0).to(tl.float32)

    # Preserve BF16 rounding between affine RMSNorm and RoPE.
    first = (first * inv * first_weight).to(tl.bfloat16).to(tl.float32)
    second = (second * inv * second_weight).to(tl.bfloat16).to(tl.float32)
    tl.store(ptr + col, first * cos - second * sin, valid_head & rotary_mask)
    tl.store(
        ptr + half_rotary + col, second * cos + first * sin, valid_head & rotary_mask
    )
    tl.store(ptr + R + col, tail * inv * tail_weight, valid_head & tail_mask)


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
    HEADS: tl.constexpr,
    EPS: tl.constexpr,
    D: tl.constexpr = _HEAD_DIM,
    R: tl.constexpr = _ROTARY_DIM,
    HEADS_PER_BLOCK: tl.constexpr = _HEADS_PER_BLOCK,
):
    blocks_per_token: tl.constexpr = (HEADS + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK
    block = tl.program_id(0)
    token = (block // blocks_per_token).to(tl.int64)
    head_group = block % blocks_per_token
    head = (head_group * HEADS_PER_BLOCK + tl.arange(0, HEADS_PER_BLOCK))[:, None]
    col = tl.arange(0, D // 2)[None, :]
    pos = tl.load(POS + token).to(tl.int64)
    cos = tl.load(CACHE + pos * R + col, col < R // 2, 0).to(tl.float32)
    sin = tl.load(CACHE + pos * R + R // 2 + col, col < R // 2, 0).to(tl.float32)

    q_ptr = Q + token * Q_ROW + head.to(tl.int64) * Q_HEAD
    k_ptr = K + token * K_ROW + head.to(tl.int64) * K_HEAD
    _normalize_and_rotate(q_ptr, WQ, col, head < HEADS, cos, sin, EPS, D, R)
    _normalize_and_rotate(k_ptr, WK, col, head < HEADS, cos, sin, EPS, D, R)


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
        grid = (q.shape[0] * triton.cdiv(q.shape[1], _HEADS_PER_BLOCK),)
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
            q.shape[1],
            eps,
            num_warps=1,
            enable_fp_fusion=False,
        )


__all__ = ["can_use_minimax_h3_dit_qknorm_rope", "minimax_h3_dit_qknorm_rope"]
