from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils import is_hcu

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsv4.candidate_indexer_deep_gemm import (
        DeepGemmCandidateIndexer,
    )
    from sglang.srt.layers.attention.dsv4.candidate_indexer_lightop import (
        LightopCandidateIndexer,
    )

# HIP/DCU resolves the block-level top-k to a prebuilt lightop op; NVIDIA to JIT.
_IS_HCU = is_hcu()

# Each GPU has its own worker process. The shared candidate stream is created
# once at backend init, after device selection and before any capture, so that
# no layer ever creates a stream from inside a captured region.
_CANDIDATE_STREAM: Optional[torch.cuda.Stream] = None


def init_candidate_stream() -> None:
    """Create the shared candidate stream for this worker's device.

    Call from backend init, before capture. Idempotent, so the eager and the
    capture paths both observe the same stream object.
    """
    global _CANDIDATE_STREAM
    if _CANDIDATE_STREAM is None:
        # Each SGLang worker owns one device, so one process-wide stream is
        # enough for every candidate indexer instance in that worker.
        _CANDIDATE_STREAM = torch.cuda.Stream()


def get_candidate_stream() -> torch.cuda.Stream:
    """The shared candidate stream, creating it on first use.

    Callers that run inside capture must not reach the creating path; the
    backend calls ``init_candidate_stream`` before capture so this only ever
    returns the stream that was created at init.
    """
    if _CANDIDATE_STREAM is None:
        init_candidate_stream()
    return _CANDIDATE_STREAM


def get_candidate_stream_if_initialized() -> Optional[torch.cuda.Stream]:
    """The shared candidate stream, or None when it was never created.

    Lets the scheduler avoid allocating a colliding side stream without
    forcing the candidate stream into existence on workers that never use it.
    """
    return _CANDIDATE_STREAM


class CandidateMetadata:
    """Base of an implementation's published state on
    ``DSV4Metadata.candidate_metadata``."""


@dataclass(frozen=True)
class IndexerInputs:
    """One index-source layer's operands on the paged fp4 decode path (one query
    row per request, or per draft token under verify)."""

    q_fp4: torch.Tensor  # [rows, 1, heads, 64] int8, packed fp4
    q_sf: torch.Tensor  # [rows, 1, heads] int32, packed ue8m0
    k_cache: torch.Tensor  # [pages, page_size, 1, 68] uint8, the layer's index-K pool
    weights: torch.Tensor  # [rows, heads] bf16/fp32 head weights
    metadata: PagedIndexerMetadata  # this ratio's lengths, page table and plans
    # [rows] int, one request id per query row, the rows of one request
    # consecutive (verify: its draft tokens); None = every row its own request
    request_ids: Optional[torch.Tensor] = None

    @property
    def num_rows(self) -> int:
        return self.q_fp4.shape[0]


def make_candidate_indexer(
    topk_blocks: int,
    block_size: int,
) -> Optional[Union[DeepGemmCandidateIndexer, LightopCandidateIndexer]]:
    """The paged fp4 decode path's two-level indexer; None on Hopper, whose decode
    indexer selects through masks inline. HCU uses the LightOp implementation
    when the installed LightOp provides it, else None (slot-based path)."""
    if topk_blocks <= 0:
        return None
    if block_size <= 0:
        raise ValueError(
            "candidate_block_size must be positive when candidate_topk_blocks is enabled"
        )
    from sglang.srt.utils import is_hcu

    if is_hcu():
        from sglang.kernels.ops.attention.dsv4.lightop_indexer import (
            lightop_paged_indexer_available,
        )

        if not lightop_paged_indexer_available():
            return None
        from sglang.srt.layers.attention.dsv4.candidate_indexer_lightop import (
            LightopCandidateIndexer,
        )

        return LightopCandidateIndexer(topk_blocks, block_size)
    if get_platform().device_sm < 100:
        return None
    from sglang.srt.layers.deep_gemm_wrapper.configurer import (
        DEEPGEMM_PAGED_SPARSE_MQA_LOGITS,
    )

    if not DEEPGEMM_PAGED_SPARSE_MQA_LOGITS:
        raise RuntimeError(
            "the candidate indexer needs DeepGEMM's paged sparse MQA logits "
            "(sgl-deep-gemm >= 0.2.0 with SGLANG_ENABLE_JIT_DEEPGEMM on)"
        )
    from sglang.srt.layers.attention.dsv4.candidate_indexer_deep_gemm import (
        DeepGemmCandidateIndexer,
    )

    return DeepGemmCandidateIndexer(topk_blocks, block_size)


# TODO(candidate): Hopper decode and prefill still select through these masks
# inline in the backend; move them behind the protocol as publish/select_prefill.
@dataclass
class CandidateMasks(CandidateMetadata):
    mask: Optional[torch.Tensor] = None  # decode: [rows, width] bool
    request_masks: Optional[List[torch.Tensor]] = None  # prefill: [rows_b, lc_b] each


def published_masks(candidate) -> CandidateMasks:
    assert isinstance(candidate, CandidateMasks), "candidate masks missing"
    return candidate


@triton.jit
def _mask_topk_kernel(
    indices_ptr,
    scores_ptr,
    offsets_ptr,
    out_ptr,
    W,
    K,
    stride_ir,
    stride_ik,
    stride_sr,
    stride_sc,
    stride_or,
    stride_ok,
    HAS_OFF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per row: fuses the whole valid-mask + gather + masked_fill of
    ``mask_topk_scores`` into a single kernel instead of ~10 eager elementwise
    ops (cast, sub, clamp, gather, three compares, two ands, not, masked_fill)."""
    r = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    m = k < K
    idx = tl.load(indices_ptr + r * stride_ir + k * stride_ik, mask=m, other=0)
    col = idx.to(tl.int64)
    if HAS_OFF:
        col = col - tl.load(offsets_ptr + r).to(tl.int64)
    in_range = (col >= 0) & (col < W)
    col_c = tl.minimum(tl.maximum(col, 0), W - 1)
    s = tl.load(
        scores_ptr + r * stride_sr + col_c * stride_sc, mask=m, other=-float("inf")
    )
    valid = in_range & (s > -float("inf"))
    tl.store(out_ptr + r * stride_or + k * stride_ok, tl.where(valid, idx, -1), mask=m)


def mask_topk_scores(
    scores: torch.Tensor,
    indices: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Keep masked indexer scores out of attention even when top-k underfills."""
    if indices.is_cuda and indices.ndim == 2 and scores.ndim == 2:
        rows, k_width = indices.shape
        out = torch.empty_like(indices)
        if rows and k_width:
            _mask_topk_kernel[(rows,)](
                indices,
                scores,
                indices if offsets is None else offsets,
                out,
                scores.shape[1],
                k_width,
                indices.stride(0),
                indices.stride(1),
                scores.stride(0),
                scores.stride(1),
                out.stride(0),
                out.stride(1),
                HAS_OFF=offsets is not None,
                BLOCK=triton.next_power_of_2(k_width),
            )
        return out
    columns = indices.to(torch.int64)
    if offsets is not None:
        columns = columns - offsets[:, None]
    selected_scores = scores.gather(1, columns.clamp(0, scores.shape[1] - 1))
    valid = (
        (columns >= 0) & (columns < scores.shape[1]) & (selected_scores > -torch.inf)
    )
    return indices.masked_fill(~valid, -1)


# HCU lightop top-k reads a per-row identity page table and a cu_seqlens; both
# depend only on (bs, wq), so cache them instead of rebuilding two aranges per
# decode step. Never evicted: a captured graph replays the buffers it saw, so a
# new (bs, wq) may not be materialised mid-capture -- warm the shape up first.
_HCU_TOPK_TABLES: dict[
    tuple[int, int, torch.device], tuple[torch.Tensor, torch.Tensor]
] = {}


def _hcu_topk_tables(
    bs: int, wq: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contiguous ``[bs, wq]`` identity page table (row-repeated ``arange(wq)``; the
    native op indexes it by raw offset, so a stride-0 broadcast would read past
    row 0) and ``arange(bs + 1)`` cu_seqlens, both int32."""
    key = (bs, wq, device)
    cached = _HCU_TOPK_TABLES.get(key)
    if cached is not None:
        return cached
    assert not torch.cuda.is_current_stream_capturing(), (
        f"HCU top-k tables for bs={bs}, wq={wq} built inside a CUDA graph capture; "
        "warm up this shape before capture"
    )
    identity = torch.arange(wq, dtype=torch.int32, device=device).repeat(bs, 1)
    cu_seqlens_q = torch.arange(bs + 1, dtype=torch.int32, device=device)
    tables = (identity, cu_seqlens_q)
    _HCU_TOPK_TABLES[key] = tables
    return tables


def _topk_block_ids(
    keys: torch.Tensor, nblocks: torch.Tensor, topk_blocks: int
) -> torch.Tensor:
    """Top-k over the ``[bs, nblocks_amax]`` block-amax keys -> ``[bs, K]`` int32
    raw block ids, ``-1`` past each row's ``nblocks[i]``.

    HIP/DCU uses lightop's prebuilt exact-adaptive Top2048 op (no JIT); NVIDIA the
    JIT v2 kernel with ``page_tables=None``. ``K`` is ``topk_blocks`` on HIP (the
    lightop op is fixed width), ``min(topk_blocks, nblocks_amax)`` on NVIDIA."""
    bs, nblocks_amax = keys.shape
    if _IS_HCU:
        import lightop

        # SGL_USE_LIGHTOP_TOPK_BACKAND unset/0 selects the exact adaptive Top2048.
        # The op needs a score width >= topk and reads each row only up to
        # nblocks[i]; pad so an identity page table spans [0, topk) with an
        # unread -inf tail, page_size 1 mapping selected index i to raw block i.
        wq = max((nblocks_amax + 3) & ~3, topk_blocks)
        if wq != nblocks_amax:
            keys = F.pad(keys, (0, wq - nblocks_amax), value=-torch.inf)
        identity, cu_seqlens_q = _hcu_topk_tables(bs, wq, keys.device)
        return lightop.fast_topk_transform_fused(
            keys, nblocks, identity, cu_seqlens_q, topk_blocks
        )

    from sglang.kernels.ops.attention.dsv4.topk import (
        plan_topk_v2,
        topk_transform_paged_v2,
    )

    # v2 needs the score row stride to be a multiple of 4; pad with -inf.
    key_pad = (-nblocks_amax) % 4
    if key_pad:
        keys = F.pad(keys, (0, key_pad), value=-torch.inf)
    k = min(topk_blocks, nblocks_amax)
    block_ids = torch.full((bs, k), -1, dtype=torch.int32, device=keys.device)
    # page_tables=None yields raw block ids; the plan must precede the kernel.
    plan = plan_topk_v2(nblocks)
    topk_transform_paged_v2(keys, nblocks, None, block_ids, 1, plan)
    return block_ids


def _select_candidate_blocks_fused(
    logits: torch.Tensor,
    compress_lens: torch.Tensor,
    topk_blocks: int,
) -> torch.Tensor:
    """Fast path for block_size==8: replaces the block-level ``torch.topk`` (which
    lowers to the multi-kernel mbtopk radix reduction) with a fused top-k, then
    expands the selected block ids to a position-level bool mask.

    The block-amax step stays in torch: ``amax`` over a size-8 dim is a plain
    reduction and never triggers mbtopk."""
    bs, width = logits.shape

    # Per-block amax: pad to a block boundary, then max over each group of 8.
    pad = (-width) % 8
    logits_aligned = (
        F.pad(logits, (0, pad), value=-torch.inf) if pad else logits
    )
    # [bs, nblocks_amax]; the kernel needs fp32, contiguous on dim 1.
    keys = logits_aligned.unflatten(-1, (-1, 8)).amax(dim=-1).to(torch.float32)

    # Per-row block count; the top-k reads each row only up to nblocks[i].
    lens1d = compress_lens.reshape(-1).to(torch.int32)  # [bs]
    nblocks = lens1d.add(7).div(8, rounding_mode="floor").to(torch.int32)  # [bs]

    # Force the newest block (+inf sentinel) so it is always selected.
    last_block = (nblocks - 1).clamp(min=0).to(torch.int64)  # [bs]
    keys.scatter_(1, last_block.unsqueeze(1), torch.inf)

    block_ids = _topk_block_ids(keys, nblocks, topk_blocks)  # [bs, K] int32

    # Expand block ids back to a position-level bool mask [bs, width].
    k = block_ids.shape[1]
    mask = torch.zeros(bs, width, dtype=torch.bool, device=logits.device)
    valid = block_ids >= 0  # [bs, K]
    _expand_blocks_kernel[(bs,)](
        mask,
        block_ids,
        valid,
        k,
        width,
        mask.stride(0),
        block_ids.stride(0),
        valid.stride(0),
        BLOCK_K=triton.next_power_of_2(k),
    )
    return mask


@triton.jit
def _expand_blocks_kernel(
    mask_ptr,
    block_ids_ptr,
    valid_ptr,
    K,
    W,
    stride_mr,
    stride_br,
    stride_vr,
    BLOCK_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 8,
):
    """One program per row: scatter each valid block id into 8 consecutive bool positions."""
    r = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    m = k < K
    bid = tl.load(block_ids_ptr + r * stride_br + k, mask=m, other=-1)
    ok = tl.load(valid_ptr + r * stride_vr + k, mask=m, other=0).to(tl.int1)
    pos_base = bid * BLOCK_SIZE
    # For each of the k blocks write 8 True entries; unroll over inner dim.
    for i in tl.static_range(BLOCK_SIZE):
        pos = pos_base + i  # [BLOCK_K]
        in_range = ok & m & (pos >= 0) & (pos < W)
        tl.store(mask_ptr + r * stride_mr + pos, tl.full([BLOCK_K], 1, tl.int1), mask=in_range)


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: Union[torch.Tensor, int],
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of the two-level top-k: a bool mask over positions keeping the
    topk_blocks best-scoring blocks per query. Unreachable positions are already -inf
    in logits, so an all -inf block means not reachable yet; the block holding the
    query's newest position is always kept."""
    # Fast path: block_size==8 replaces the block-level torch.topk (mbtopk) with
    # a fused top-k -- lightop's prebuilt Top2048 on HIP/DCU, the JIT v2 kernel
    # on NVIDIA -- to avoid the multi-kernel reduction over the position dim.
    if (
        block_size == 8
        and logits.is_cuda
        and logits.ndim == 2
        and isinstance(compress_lens, torch.Tensor)
    ):
        return _select_candidate_blocks_fused(logits, compress_lens, topk_blocks)

    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last, torch.inf
    )

    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]
