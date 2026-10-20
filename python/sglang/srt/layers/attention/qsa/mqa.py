"""Weight-free MQA operators for the simple QSA indexer.

The public entry points try BoltOPs TileLang first, then the in-tree TileLang
kernels, then the Torch reference.
"""

import logging
import math
from functools import lru_cache
from typing import Optional, Tuple

import torch

from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _warn_tilelang_fallback(kind: str, reason: str) -> None:
    logger.warning("QSA MQA TileLang %s falls back to torch: %s", kind, reason)

try:
    import flashinfer.comm  # noqa: F401
except ImportError:
    pass

try:
    import tilelang
    from tilelang import language as T

    HAS_TILELANG = True
except ImportError:
    tilelang = None
    T = None
    HAS_TILELANG = False


def _validate_q(q: torch.Tensor) -> None:
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError(f"QSA requires q [tokens, heads, head_dim], got {q.shape}")


def _validate_k(k: torch.Tensor) -> None:
    if k.ndim != 3 or k.shape[1] != 1 or k.shape[2] <= 0:
        raise ValueError(f"QSA MQA requires k [tokens, 1, head_dim], got {k.shape}")


def torch_qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Torch reference for packed, variable-length prefill MQA."""

    _validate_q(q)
    _validate_k(k)
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")
    scores = torch.einsum("mhd,nd->mnh", q.float(), k[:, 0].float())
    logits = torch.relu(scores).sum(dim=-1) / (score_scale or math.sqrt(q.shape[-1]))
    columns = torch.arange(k.shape[0], device=q.device).unsqueeze(0)
    valid = (columns >= row_starts.to(q.device).reshape(-1, 1)) & (
        columns < row_ends.to(q.device).reshape(-1, 1)
    )
    return logits.masked_fill(~valid, -float("inf"))


def _validate_decode_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
) -> None:
    _validate_q(q)
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError(
            "QSA decode cache must be [pages, page_size, 1, head_dim], "
            f"got {tuple(k_cache.shape)}"
        )
    if k_cache.shape[-1] != q.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")
    if page_table.ndim != 2 or page_table.shape[0] != q.shape[0]:
        raise ValueError("QSA decode page table must have one row per query")
    if context_lens.numel() != q.shape[0]:
        raise ValueError("QSA decode context lengths must have one entry per query")


def torch_qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Torch reference for variable-length paged decode MQA."""

    _validate_decode_inputs(q, k_cache, page_table, context_lens)
    batch = q.shape[0]
    page_size = k_cache.shape[1]
    total = page_table.shape[1] * page_size
    gathered = k_cache[page_table.long().clamp_min(0).reshape(-1), :, 0].reshape(
        batch, total, q.shape[-1]
    )
    scores = torch.einsum("bhd,bnd->bnh", q.float(), gathered.float())
    scores = torch.relu(scores).sum(dim=-1) / (score_scale or math.sqrt(q.shape[-1]))
    positions = torch.arange(total, device=q.device).unsqueeze(0)
    scores.masked_fill_(
        positions >= context_lens.to(q.device).reshape(-1, 1), -float("inf")
    )
    logits = torch.full(
        (batch, max_model_len), -float("inf"), dtype=torch.float32, device=q.device
    )
    copy_len = min(total, max_model_len)
    if copy_len:
        logits[:, :copy_len] = scores[:, :copy_len]
    return logits


if HAS_TILELANG:

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    def _tilelang_qsa_mqa_prefill_kernel(
        heads: int,
        head_dim: int,
        block_n: int = 64,
        block_q: int = 32,
        num_stages: int = 3,
        threads: int = 512,
    ):
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")

        @T.prim_func
        def kernel(
            Q: T.Tensor([rows * heads, head_dim], T.bfloat16),  # type: ignore
            K: T.Tensor([keys, head_dim], T.bfloat16),  # type: ignore
            Logits: T.Tensor([rows, keys], T.float32),  # type: ignore
            Starts: T.Tensor([rows], T.int32),  # type: ignore
            Ends: T.Tensor([rows], T.int32),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(rows, block_q), threads=threads) as bx:
                q_shared = T.alloc_shared([block_q * heads, head_dim], T.bfloat16)
                k_shared = T.alloc_shared([block_n, head_dim], T.bfloat16)
                scores = T.alloc_fragment([block_n, block_q * heads], T.float32)
                scores_3d = T.reshape(scores, (block_n, block_q, heads))
                reduced = T.alloc_fragment([block_n, block_q], T.float32)
                row_base = bx * block_q
                start_min = T.alloc_var(T.int32)
                end_max = T.alloc_var(T.int32)
                start_min = 2147483647
                end_max = -2147483648
                for qi in T.serial(block_q):
                    start_min = T.min(start_min, T.min(Starts[row_base + qi], keys))
                    end_max = T.max(end_max, T.min(Ends[row_base + qi], keys))

                T.copy(Q[row_base * heads, 0], q_shared)
                for ni in T.Pipelined(
                    T.ceildiv(end_max - start_min, block_n), num_stages=num_stages
                ):
                    T.copy(K[start_min + ni * block_n, 0], k_shared)
                    T.gemm(
                        k_shared,
                        q_shared,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    for n, qi, head in T.Parallel(block_n, block_q, heads):
                        scores_3d[n, qi, head] = T.max(scores_3d[n, qi, head], 0.0)
                    T.reduce_sum(scores_3d, reduced, dim=-1, clear=True)
                    for qi, n in T.Parallel(block_q, block_n):
                        Logits[row_base + qi, start_min + ni * block_n + n] = reduced[
                            n, qi
                        ]

        return kernel

    @tilelang.jit
    def _tilelang_qsa_mqa_mask_kernel(threads: int = 512, block_k: int = 4096):
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")

        @T.prim_func
        def kernel(
            Logits: T.Tensor([rows, keys], T.float32),  # type: ignore
            Starts: T.Tensor([rows], T.int32),  # type: ignore
            Ends: T.Tensor([rows], T.int32),  # type: ignore
        ):
            with T.Kernel(rows, threads=threads) as bx:
                tx = T.thread_binding(0, threads, thread="threadIdx.x")
                for block in T.Pipelined(T.ceildiv(keys, block_k)):
                    for item in T.serial(block_k // threads):
                        column = block * block_k + item * threads + tx
                        if column < Starts[bx] or column >= Ends[bx]:
                            Logits[bx, column] = -T.infinity(T.float32)

        return kernel

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        }
    )
    def _tilelang_qsa_mqa_decode_kernel(
        heads: int,
        head_dim: int,
        page_size: int = 64,
        groups_per_cta: int = 1,
        num_stages: int = 3,
        threads: int = 128,
    ):
        # The validated MMA layout wants 64 GEMM rows; pages narrower than
        # that (full_page // ratio compressed views) are packed in sub-page
        # quadrants of one 64-row tile.
        GROUP = 64
        assert GROUP % page_size == 0, page_size
        sub_pages = GROUP // page_size
        batch = T.dynamic("batch")
        pages = T.dynamic("pages")
        max_pages = T.dynamic("max_pages")
        max_model_len = T.dynamic("max_model_len")

        @T.prim_func
        def kernel(
            Q: T.Tensor([batch, 1, heads, head_dim], T.bfloat16),  # type: ignore
            KCache: T.Tensor([pages, page_size, 1, head_dim], T.bfloat16),  # type: ignore
            PageTable: T.Tensor([batch, max_pages], T.int32),  # type: ignore
            ContextLens: T.Tensor([batch], T.int32),  # type: ignore
            Logits: T.Tensor([batch, max_model_len], T.float32),  # type: ignore
            Scale: T.float32,
        ):
            with T.Kernel(
                batch,
                T.ceildiv(T.ceildiv(max_pages, sub_pages), groups_per_cta),
                threads=threads,
            ) as (bx, group_block):
                q_shared = T.alloc_shared([heads, head_dim], T.bfloat16)
                k_shared = T.alloc_shared([GROUP, head_dim], T.bfloat16)
                scores = T.alloc_fragment([GROUP, heads], T.float32)
                reduced = T.alloc_fragment([GROUP], T.float32)
                T.copy(Q[bx, 0, :, :], q_shared)
                context_len = ContextLens[bx]

                for gi in T.Pipelined(groups_per_cta, num_stages=num_stages):
                    group = group_block * groups_per_cta + gi
                    if group * GROUP < context_len:
                        # Python-level unroll: sub_pages is a compile-time
                        # constant and TileLang's pipeliner rejects dynamic
                        # inner loops around shared-memory copies.
                        for sp in range(sub_pages):
                            if (group * sub_pages + sp) * page_size < context_len:
                                T.copy(
                                    KCache[
                                        PageTable[bx, group * sub_pages + sp],
                                        :,
                                        0,
                                        :,
                                    ],
                                    k_shared[
                                        sp * page_size : (sp + 1) * page_size, :
                                    ],
                                )
                        T.gemm(
                            k_shared,
                            q_shared,
                            scores,
                            transpose_B=True,
                            clear_accum=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        for token, head in T.Parallel(GROUP, heads):
                            scores[token, head] = T.max(scores[token, head], 0.0)
                        T.reduce_sum(scores, reduced, dim=1, clear=True)
                        for token in T.Parallel(GROUP):
                            position = group * GROUP + token
                            if position < context_len:
                                Logits[bx, position] = reduced[token] / Scale

        return kernel


def tilelang_qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Validated TileLang packed prefill kernel with weights removed."""

    if not HAS_TILELANG:
        raise RuntimeError("TileLang is unavailable")
    _validate_q(q)
    _validate_k(k)
    rows, keys = q.shape[0], k.shape[0]
    if not rows or not keys:
        logits = torch.zeros((rows, keys), dtype=torch.float32, device=q.device)
        return logits.masked_fill_(
            torch.ones_like(logits, dtype=torch.bool), -float("inf")
        )
    heads, head_dim = q.shape[1:]
    # HIP/HCU LDS is 64KB. Default CUDA launch (threads=512, num_stages=3,
    # block_q=128//heads) pipelines three K tiles and overflows (~80KB).
    # Keep GEMM N = block_q * heads = 128 (MFMA-aligned) and drop pipeline
    # replicas; match the decode kernel's 128-thread CTA.
    if is_hip():
        threads = 128
        num_stages = 1
        block_q = 32
    else:
        threads = 512
        num_stages = 3
        block_q = max(1, 128 // heads)
    padding = (-rows) % block_q
    padded_rows = rows + padding
    # Allocate the padded output once. Appending even a few padding rows with
    # torch.cat would allocate and copy the entire [rows, keys] FP32 matrix,
    # temporarily doubling the dominant prefill buffer for long contexts.
    logits = torch.zeros(
        (padded_rows, keys), dtype=torch.float32, device=q.device
    )
    q_padded = q.to(torch.bfloat16).contiguous()
    starts = row_starts.to(device=q.device, dtype=torch.int32).contiguous()
    ends = row_ends.to(device=q.device, dtype=torch.int32).contiguous()
    if padding:
        q_padded = torch.cat([q_padded, q_padded.new_zeros(padding, heads, head_dim)])
        starts = torch.cat([starts, starts[-1:].expand(padding)])
        ends = torch.cat([ends, ends[-1:].expand(padding)])

    _tilelang_qsa_mqa_prefill_kernel(
        heads=heads,
        head_dim=head_dim,
        block_q=block_q,
        num_stages=num_stages,
        threads=threads,
    )(
        q_padded.reshape(-1, head_dim),
        k[:, 0].to(torch.bfloat16).contiguous(),
        logits,
        starts,
        ends,
    )
    # A leading-dimension slice that retains every column is already
    # contiguous, so do not copy this large matrix again when removing padding.
    logits = logits[:rows]
    logits.div_(score_scale or math.sqrt(head_dim))
    # HIP: default mask CTA (threads=512, block_k=4096) has been implicated in
    # warmup VMFaults; use a smaller launch matching the prefill GEMM CTA.
    if is_hip():
        _tilelang_qsa_mqa_mask_kernel(threads=128, block_k=1024)(
            logits, starts[:rows], ends[:rows]
        )
    else:
        _tilelang_qsa_mqa_mask_kernel()(logits, starts[:rows], ends[:rows])
    return logits


def tilelang_qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Validated TileLang paged decode kernel with weights removed."""

    if not HAS_TILELANG:
        raise RuntimeError("TileLang is unavailable")
    _validate_decode_inputs(q, k_cache, page_table, context_lens)
    page_size = int(k_cache.shape[1])
    if page_size < 8 or 64 % page_size != 0:
        raise ValueError(
            "TileLang QSA decode requires a compressed page size of "
            f"8/16/32/64 (64-row GEMM sub-page packing), got {page_size}"
        )
    logits = torch.full(
        (q.shape[0], max_model_len),
        -float("inf"),
        dtype=torch.float32,
        device=q.device,
    )
    if not q.shape[0] or not max_model_len:
        return logits
    # The validated MMA layout requires N (the Q-head dimension) to be a
    # multiple of eight. Zero-padding preserves the weight-free head sum.
    # CUDA MMA accepts an eight-wide N dimension, while ROCm/HCU MFMA
    # requires sixteen. Zero-padding preserves the weight-free head sum.
    query_heads, head_dim = q.shape[1:]
    head_alignment = 16 if is_hip() else 8
    kernel_heads = max(
        head_alignment,
        ((query_heads + head_alignment - 1) // head_alignment) * head_alignment,
    )
    q_kernel = q.to(torch.bfloat16)
    if kernel_heads != query_heads:
        q_kernel = torch.cat(
            [
                q_kernel,
                q_kernel.new_zeros(q.shape[0], kernel_heads - query_heads, head_dim),
            ],
            dim=1,
        )
    _tilelang_qsa_mqa_decode_kernel(
        heads=kernel_heads, head_dim=head_dim, page_size=page_size
    )(
        q_kernel.unsqueeze(1).contiguous(),
        k_cache.to(torch.bfloat16).contiguous(),
        page_table.to(device=q.device, dtype=torch.int32).contiguous(),
        context_lens.to(device=q.device, dtype=torch.int32).contiguous(),
        logits,
        float(score_scale or math.sqrt(head_dim)),
    )
    return logits


# ---------------------------------------------------------------------------
# BoltOPs FP8 paged-decode packing (gfx938)
#
# The BoltOPs `tilelang_qsa_mqa_decode` dispatches on input layout: a bf16
# [pages, page_size, 1, D] cache selects the BF16 kernel, while Q/K in
# float8_e4m3fn with the aiter-packed cache [pages, page_size, 1, D+4]
# (per-token fp32 scale in the trailing 4 bytes of every row) selects the
# dedicated FP8 kernel.  The BF16 decode kernel does not compile on the DTK
# tilelang toolchain ([ThreadSync] lowering failure with a non-warp-multiple
# thread count), which silently degraded every QSA decode step to the full
# gather + fp32 einsum Torch reference.  The FP8 kernel compiles and runs, so
# pack the decode inputs at this boundary (per-token amax/448 scale, exact
# inverse applied by the kernel's `reduced * scale` epilogue) and let the
# vendor dispatch choose it.  All ops are shape-static tensor math -- no host
# syncs -- so packing is CUDA-graph capture safe.
# ---------------------------------------------------------------------------

_FP8_DECODE_ENV = "SGLANG_QSA_INDEXER_FP8_DECODE"
_FP8_E4M3_MAX = 448.0
# BoltOPs FP8 decode kernel constraints (gfx938): query heads in {4,8,16,32,64},
# head_dim exactly 128, compressed page size one of 8/16/32/64.
_FP8_DECODE_HEADS = (4, 8, 16, 32, 64)
_FP8_DECODE_HEAD_DIM = 128
_FP8_DECODE_PAGE_SIZES = (8, 16, 32, 64)


def _fp8_decode_enabled() -> bool:
    import os

    return os.environ.get(_FP8_DECODE_ENV, "1").strip().lower() not in (
        "0",
        "off",
        "false",
    )


def _maybe_pack_fp8_decode(
    q: torch.Tensor, k_cache: torch.Tensor
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Quantize decode MQA inputs into the aiter packed FP8 layout.

    Returns ``(q_fp8, k_cache_packed)`` with the cache shaped
    ``[pages, page_size, 1, D+4]`` (per-token fp32 scale in the trailing 4
    bytes of every row), or ``None`` when the inputs do not match the FP8
    kernel contract or packing is disabled.
    """
    if not _fp8_decode_enabled() or not is_hip():
        return None
    if k_cache.dtype is not torch.bfloat16 or k_cache.dtype is not q.dtype:
        return None
    if (
        k_cache.ndim != 4
        or k_cache.shape[2] != 1
        or q.ndim != 3
        or k_cache.shape[-1] != q.shape[-1]
    ):
        return None
    page_size = int(k_cache.shape[1])
    heads = int(q.shape[1])
    if (
        q.shape[-1] != _FP8_DECODE_HEAD_DIM
        or page_size not in _FP8_DECODE_PAGE_SIZES
        or heads not in _FP8_DECODE_HEADS
    ):
        return None

    pages = int(k_cache.shape[0])
    dim = int(k_cache.shape[-1])
    rows = pages * page_size
    k = k_cache.reshape(rows, dim).float()
    amax = k.abs().amax(dim=-1, keepdim=True)
    # Per-token scale; zero rows keep a unit scale (their fp8 payload is 0,
    # so the kernel's `reduced * scale` still yields a 0 logit).
    scale = torch.where(
        amax > 0, amax / _FP8_E4M3_MAX, torch.ones_like(amax)
    )
    k_fp8 = (k / scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(
        torch.float8_e4m3fn
    )
    q_fp8 = q.float().clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)

    k_bytes = k_fp8.view(torch.uint8).reshape(pages, page_size * dim)
    s_bytes = (
        scale.view(torch.float32)
        .view(torch.uint8)
        .reshape(pages, page_size * 4)
    )
    packed = torch.empty(
        (pages, page_size * (dim + 4)), dtype=torch.uint8, device=k_cache.device
    )
    packed[:, : page_size * dim] = k_bytes
    packed[:, page_size * dim :] = s_bytes
    return q_fp8, packed.view(pages, page_size, 1, dim + 4)


def qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    from sglang.srt.layers.attention.qsa.boltops_mqa import try_boltops_qsa_mqa_prefill

    boltops_logits = try_boltops_qsa_mqa_prefill(
        q, k, row_starts, row_ends, score_scale
    )
    if boltops_logits is not None:
        return boltops_logits
    # HIP: TileLang QSA MQA has been observed to VMFault during warmup
    # (async Invalid address; stacks often surface later at apply_rope).
    # Prefer the Torch reference until the HIP TileLang path is stable.
    if q.is_cuda and HAS_TILELANG and not is_hip():
        try:
            return tilelang_qsa_mqa_prefill(q, k, row_starts, row_ends, score_scale)
        except Exception as exc:
            _warn_tilelang_fallback("prefill", str(exc))
    return torch_qsa_mqa_prefill(q, k, row_starts, row_ends, score_scale)


def qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    from sglang.srt.layers.attention.qsa.boltops_mqa import try_boltops_qsa_mqa_decode

    fp8_packed = _maybe_pack_fp8_decode(q, k_cache)
    if fp8_packed is not None:
        q_fp8, cache_fp8 = fp8_packed
        boltops_logits = try_boltops_qsa_mqa_decode(
            q_fp8, cache_fp8, page_table, context_lens, max_model_len, score_scale
        )
        if boltops_logits is not None:
            return boltops_logits
        # BoltOPs fp8 decode unavailable: fall through to the legacy bf16
        # path below with the original tensors.

    boltops_logits = try_boltops_qsa_mqa_decode(
        q, k_cache, page_table, context_lens, max_model_len, score_scale
    )
    if boltops_logits is not None:
        return boltops_logits
    if q.is_cuda and HAS_TILELANG and not is_hip():
        try:
            return tilelang_qsa_mqa_decode(
                q, k_cache, page_table, context_lens, max_model_len, score_scale
            )
        except Exception as exc:
            _warn_tilelang_fallback("decode", str(exc))
    return torch_qsa_mqa_decode(
        q, k_cache, page_table, context_lens, max_model_len, score_scale
    )


__all__ = [
    "HAS_TILELANG",
    "qsa_mqa_decode",
    "qsa_mqa_prefill",
    "tilelang_qsa_mqa_decode",
    "tilelang_qsa_mqa_prefill",
    "torch_qsa_mqa_decode",
    "torch_qsa_mqa_prefill",
]
