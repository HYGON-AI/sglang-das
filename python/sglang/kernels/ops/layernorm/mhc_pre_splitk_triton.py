"""Split-K Triton implementation of the mHC pre GEMM with fused square-sum.

``mhc_pre`` needs ``gemm_out_mul[n_split, tokens, hc_mult3]`` and
``gemm_out_sqrsum[n_split, tokens]`` for the downstream ``big_fuse`` kernel.
The TileLang split-K kernel that ships with the tree covers only
``num_tokens <= 2048`` with ``hc_hidden_size in {16384, 28672}``; prefill
batches are far larger (tens of thousands of tokens per step), so they fall
back to the non-split-K variant that serialises the whole K reduction in one
CTA and leaves most of the HCUs idle.

This module adds a Triton split-K kernel for that regime.  It keeps the exact
output layout the fuse kernel already consumes, so it is a drop-in replacement
for the existing split-K branch, and it computes the row square-sum of ``x`` in
the same pass (``big_fuse`` needs ``sum(x^2)`` for the RMS denominator).

Two weight representations are supported:

* ``ieee``   - ``tl.dot(..., input_precision="ieee")`` against the FP32 weights.
* ``split3`` - the FP32 weights are split into hi/mid/lo BF16 terms and the
  accumulation is repeated three times on the BF16 matrix units.  ``hi``,
  ``mid`` and ``lo`` together reproduce the FP32 weight to about 24 mantissa
  bits, so the result tracks the IEEE path while using the matrix units.

Only contiguous ``x[M, K]`` (BF16) and ``fn[N, K]`` (FP32) with ``N <= 32`` and
``K % (split * BLOCK_K) == 0`` are accepted; ``is_supported_mhc_pre_splitk``
mirrors those conditions so callers can keep their existing fallback.
"""

from __future__ import annotations

import weakref

import torch
import triton
import triton.language as tl

_MAX_MULT3 = 32
_BLOCK_K = 128
# id(weight) -> (weakref, version, parts). Keying on the data pointer alone is
# wrong: a freed weight can hand the same address to a new tensor, and the stale
# hi/mid/lo terms would then be used for different weights.
_WEIGHT_CACHE: dict[int, tuple[weakref.ref, int, tuple[torch.Tensor, ...]]] = {}
_WEIGHT_CACHE_LIMIT = 512


@triton.jit
def _mhc_pre_partial_ieee(
    X,
    FN,
    OUT,
    SQ,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    NP: tl.constexpr,
):
    pm = tl.program_id(0)
    ps = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    cols = tl.arange(0, NP)
    reduction = tl.arange(0, BK)
    accumulator = tl.zeros((BM, NP), dtype=tl.float32)
    square = tl.zeros((BM,), dtype=tl.float32)
    k_base = ps * SPLIT_SIZE
    for offset in range(0, SPLIT_SIZE, BK):
        indices = k_base + offset + reduction
        x = tl.load(
            X + rows[:, None] * K + indices[None, :],
            mask=rows[:, None] < M,
            other=0,
        ).to(tl.float32)
        w = tl.load(
            FN + cols[None, :] * K + indices[:, None],
            mask=cols[None, :] < N,
            other=0,
        )
        accumulator = tl.dot(x, w, accumulator, input_precision="ieee")
        square += tl.sum(x * x, axis=1)
    tl.store(
        OUT + ps * M * N + rows[:, None] * N + cols[None, :],
        accumulator,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )
    tl.store(SQ + ps * M + rows, square, mask=rows < M)


@triton.jit
def _mhc_pre_partial_split3(
    X,
    WH,
    WM,
    WL,
    OUT,
    SQ,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    NP: tl.constexpr,
):
    pm = tl.program_id(0)
    ps = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    cols = tl.arange(0, NP)
    reduction = tl.arange(0, BK)
    acc_h = tl.zeros((BM, NP), dtype=tl.float32)
    acc_m = tl.zeros((BM, NP), dtype=tl.float32)
    acc_l = tl.zeros((BM, NP), dtype=tl.float32)
    square = tl.zeros((BM,), dtype=tl.float32)
    k_base = ps * SPLIT_SIZE
    for offset in range(0, SPLIT_SIZE, BK):
        indices = k_base + offset + reduction
        x = tl.load(
            X + rows[:, None] * K + indices[None, :],
            mask=rows[:, None] < M,
            other=0,
        )
        w_offsets = cols[None, :] * K + indices[:, None]
        w_mask = cols[None, :] < N
        acc_h = tl.dot(x, tl.load(WH + w_offsets, mask=w_mask, other=0), acc_h)
        acc_m = tl.dot(x, tl.load(WM + w_offsets, mask=w_mask, other=0), acc_m)
        acc_l = tl.dot(x, tl.load(WL + w_offsets, mask=w_mask, other=0), acc_l)
        xf = x.to(tl.float32)
        square += tl.sum(xf * xf, axis=1)
    accumulator = acc_h + (acc_m + acc_l)
    tl.store(
        OUT + ps * M * N + rows[:, None] * N + cols[None, :],
        accumulator,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )
    tl.store(SQ + ps * M + rows, square, mask=rows < M)


def _split_weight(fn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split an FP32 weight into hi/mid/lo BF16 terms (cached per tensor version)."""
    entry = _WEIGHT_CACHE.get(id(fn))
    if entry is not None:
        ref, version, parts = entry
        if ref() is fn and fn._version == version:
            return parts
    if len(_WEIGHT_CACHE) > _WEIGHT_CACHE_LIMIT:
        _WEIGHT_CACHE.clear()
    hi = fn.to(torch.bfloat16)
    rest = fn - hi.float()
    mid = rest.to(torch.bfloat16)
    lo = (rest - mid.float()).to(torch.bfloat16)
    parts = (hi.contiguous(), mid.contiguous(), lo.contiguous())
    _WEIGHT_CACHE[id(fn)] = (weakref.ref(fn), fn._version, parts)
    return parts


def choose_split(k: int, max_split: int = 8, block_k: int = _BLOCK_K) -> int:
    """Largest split in {1, 2, 4, max_split} that divides K into whole blocks."""
    for split in (max_split, 4, 2, 1):
        if k % (split * block_k) == 0:
            return split
    return 0


def is_supported_mhc_pre_splitk(x: torch.Tensor, fn: torch.Tensor) -> bool:
    """Mirror of the shapes the kernels accept, so callers can keep a fallback."""
    if x.dtype != torch.bfloat16 or fn.dtype != torch.float32:
        return False
    if x.ndim != 2 or fn.ndim != 2:
        return False
    if x.shape[1] != fn.shape[1] or fn.shape[0] > _MAX_MULT3:
        return False
    if not x.is_contiguous() or not fn.is_contiguous():
        return False
    return choose_split(x.shape[1]) > 0


def mhc_pre_partial_splitk(
    x: torch.Tensor,
    fn: torch.Tensor,
    split: int | None = None,
    variant: str = "split3",
    bm: int | None = None,
    block_k: int = _BLOCK_K,
    warps: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split-K mHC pre GEMM plus row square-sum.

    Returns ``(out_partial[split, M, N], sqrsum_partial[split, M])`` in the
    layout ``mhc_pre_big_fuse_tilelang`` consumes; reduce over dim 0 for the
    full ``x @ fn.T`` and ``sum(x^2, -1)``.
    """
    if x.dtype != torch.bfloat16 or fn.dtype != torch.float32:
        raise ValueError("mhc_pre_partial_splitk expects BF16 x and FP32 fn")
    if x.ndim != 2 or fn.ndim != 2:
        raise ValueError("mhc_pre_partial_splitk expects x[M, K] and fn[N, K]")
    if x.shape[1] != fn.shape[1] or not x.is_contiguous() or not fn.is_contiguous():
        raise ValueError("mhc_pre_partial_splitk expects contiguous x and fn")
    m, k = x.shape
    n = fn.shape[0]
    if n > _MAX_MULT3:
        raise ValueError(f"mhc_pre_partial_splitk supports N <= {_MAX_MULT3}, got {n}")
    split = choose_split(k, max_split=split or 8, block_k=block_k)
    if split == 0:
        raise ValueError(
            f"mhc_pre_partial_splitk: K={k} is not divisible by split * {block_k}"
        )
    if variant not in ("ieee", "split3"):
        raise ValueError(f"unknown variant {variant!r}; expected 'ieee' or 'split3'")

    out = torch.empty((split, m, n), device=x.device, dtype=torch.float32)
    sqrsum = torch.empty((split, m), device=x.device, dtype=torch.float32)

    if variant == "ieee":
        bm = bm or 32
        warps = warps or 4
        _mhc_pre_partial_ieee[(triton.cdiv(m, bm), split)](
            x,
            fn,
            out,
            sqrsum,
            m,
            K=k,
            N=n,
            SPLIT_SIZE=k // split,
            BM=bm,
            BK=block_k,
            NP=_MAX_MULT3,
            num_warps=warps,
            num_stages=2,
        )
        return out, sqrsum

    bm = bm or 128
    warps = warps or 8
    hi, mid, lo = _split_weight(fn)
    _mhc_pre_partial_split3[(triton.cdiv(m, bm), split)](
        x,
        hi,
        mid,
        lo,
        out,
        sqrsum,
        m,
        K=k,
        N=n,
        SPLIT_SIZE=k // split,
        BM=bm,
        BK=block_k,
        NP=_MAX_MULT3,
        num_warps=warps,
        num_stages=2,
    )
    return out, sqrsum
