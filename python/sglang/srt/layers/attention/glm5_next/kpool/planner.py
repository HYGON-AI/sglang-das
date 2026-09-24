"""Kpool planner: build the layer-invariant kpool plans + decode paged-MQA
metadata once per forward batch.

The ``init_*`` builders return a new ``DSAMetadata`` (frozen) via
``dataclasses.replace``; the ``update_*`` variant mutates in place
because cuda-graph replay requires the captured tensors to stay put.

Class index (which dataclass lives where):
    Row-fanout (one row per closed pool / tail-write):
        PoolWriteRows      -- extend: per-closed-pool compress, mixes
                              tail prefix + chunk K
        TailWriteRows      -- extend: per-batch tail-buffer append
    Top-level plans (stashed on DSAMetadata, layer-invariant):
        KPoolExtendPlan    -- extend (incl. draft_extend v1/v2)
        KPoolWritePlan     -- decode + target_verify (shared schema)
        KPoolCpInfo        -- CP-ownership rider on KPoolExtendPlan
    Build-side scratch:
        _KPoolCpuPlan      -- extend CPU intermediate (one H2D before
                              GPU plan)
        _KPoolDecompose    -- shared splice/bulk/tail row count helper
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, List, NamedTuple, Optional

import torch

from sglang.kernels.jit.kpool_write_plan import (
    kpool_write_plan_cuda,
    kpool_write_plan_multi_decode_cuda,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.glm5_next.kpool.kernels import (
    INDEX_HEAD_DIM,
    kpool_build_ragged_layout,
    update_kpool_write_plan_cuda_graph,
    update_kpool_write_plan_cuda_graph_multi_decode,
)
from sglang.srt.layers.attention.glm5_next.utils import dsa_use_prefill_cp
from sglang.srt.model_executor.forward_context import (
    get_req_to_token_pool,
    get_token_to_kv_pool,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_cuda, is_hcu


@lru_cache(maxsize=1)
def _get_deep_gemm():
    try:
        import deep_gemm

        return deep_gemm
    except Exception:
        return None


@lru_cache(maxsize=1)
def _get_aot_kpool_write_plan():
    try:
        from sgl_kernel import kpool_write_plan

        return kpool_write_plan
    except Exception:
        return None


# Forward-persistent ragged compress scratch (uint8 K + fp32 scale). Lazy-grown
# to the largest seen ``total_k_rows`` and sliced per forward, so prefill no
# longer pays two cudaMallocs per call.
_RAGGED_SCRATCH_K_U8: Optional[torch.Tensor] = None
_RAGGED_SCRATCH_K_SCALE: Optional[torch.Tensor] = None
logger = logging.getLogger(__name__)
_KPOOL_WRITE_PLAN_SINGLE_JIT_WARNED = False
_KPOOL_WRITE_PLAN_JIT_WARNED = False


def _disable_jit_kpool_write_plan() -> bool:
    return os.getenv("SGLANG_DISABLE_JIT_KPOOL_WRITE_PLAN", "0") == "1"


def _get_ragged_scratch(
    total_k_rows: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``[total_k_rows, INDEX_HEAD_DIM]`` u8 + ``[total_k_rows]`` fp32
    slices of a persistent scratch pair, growing it if needed.
    """
    global _RAGGED_SCRATCH_K_U8, _RAGGED_SCRATCH_K_SCALE
    # Compare device.type / index explicitly: ``torch.device('cuda') !=
    # torch.device('cuda:0')`` even when they alias the same GPU.
    cur = _RAGGED_SCRATCH_K_U8
    grow = (
        cur is None
        or cur.device.type != device.type
        or (device.index is not None and cur.device.index != device.index)
        or cur.shape[0] < total_k_rows
    )
    if grow:
        _RAGGED_SCRATCH_K_U8 = torch.empty(
            (total_k_rows, INDEX_HEAD_DIM), dtype=torch.uint8, device=device
        )
        _RAGGED_SCRATCH_K_SCALE = torch.empty(
            (total_k_rows,), dtype=torch.float32, device=device
        )
    return _RAGGED_SCRATCH_K_U8[:total_k_rows], _RAGGED_SCRATCH_K_SCALE[:total_k_rows]


if TYPE_CHECKING:
    from sglang.srt.layers.attention.glm5_next.dsa_backend import (
        DSAMetadata,
        TopkTransformMethod,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


@dataclass(frozen=True)
class PoolWriteRows:
    """All pool slots to compress + write this forward, flattened across batches.

    ``n_from_tail`` is 0 for bulk pools and equals the splice prefix
    length for splice pools. ``tail_logical_base`` is the logical position
    of the in-progress pool's first slot, used by the kernel to ring-address
    the saved tail prefix.
    """

    req: torch.Tensor  # int64 [N]
    pool_id: torch.Tensor  # int64 [N]
    n_from_tail: torch.Tensor  # int32 [N]
    chunk_src: torch.Tensor  # int64 [N]
    tail_logical_base: torch.Tensor  # int32 [N]
    write_loc: torch.Tensor  # int64 [N]

    @property
    def is_empty(self) -> bool:
        return self.pool_id.shape[0] == 0


@dataclass(frozen=True)
class TailWriteRows:
    """Per-request tail-buffer write; one row per batch with leftover chunk tokens.

    ``dst_logical_start`` is the logical position where the new tail tokens
    begin; the kernel ring-addresses ``(dst_logical_start + slot) % TAIL_SIZE``.
    """

    req: torch.Tensor  # int64 [B]
    dst_logical_start: torch.Tensor  # int32 [B]
    chunk_src: torch.Tensor  # int64 [B]
    n_write: torch.Tensor  # int32 [B]

    @property
    def is_empty(self) -> bool:
        return self.req.shape[0] == 0


@dataclass(frozen=True)
class KPoolCpInfo:
    """CP-ownership for kpool writes under dsa_enable_prefill_cp.

    Each row i is owned by ``owner_rank[i]`` -- the rank that writes its
    compressed slot. ``all_gather_and_scatter_pool_slots`` then
    propagates writes so every rank's buf converges. ``local_write_mask``
    is the per-row ``owner_rank == rank`` bool, materialized once per
    forward so every DSA layer can reuse it.
    """

    size: int
    rank: int
    owner_rank: torch.Tensor  # int32 [N]
    local_write_mask: torch.Tensor  # bool [N]


@dataclass(frozen=True)
class KPoolExtendPlan:
    """Precomputed kpool extend metadata, layer-invariant.

    Compress side (``writes`` / ``tails``) is always full-seq: K is
    all-gathered before compress under CP, so write_locs index the full
    pool table.

    Local view (``ragged_*``, ``pooled_seq_lens_expanded``,
    ``seq_lens_expanded``) is rank-local under CP round-robin-split,
    full-seq otherwise. It is sized to match this rank's q
    (``forward_cuda`` passes the rank-local ``q_fp8`` straight into
    ``_get_topk_ragged`` without any all_gather).
    """

    writes: PoolWriteRows
    tails: TailWriteRows

    pooled_seq_lens_expanded: torch.Tensor  # int32 [sum_q]
    seq_lens_expanded: torch.Tensor  # int32 [sum_q]

    # page_indices for a single gather_index_k_scale_prefix_into into a
    # flat [total_k_rows, head_dim] K buffer.
    ragged_concat_page_table: torch.Tensor  # int32 [sum_pool_pages]
    # Per-q K start row; page-aligned so the gather kernel's
    # page_indices[token_id // page_size] lookup is correct.
    # Per-q K end row; equals ``ragged_q_ks + pooled_seq_lens_expanded``.
    # Precomputed here so every DSA layer reuses the same tensor instead
    # of re-adding the two each forward.
    ragged_q_ks: torch.Tensor  # int32 [sum_q]
    ragged_q_ke: torch.Tensor  # int32 [sum_q]
    # Request-local Q/K slices as (q_start, q_end, k_start, k_end). These are
    # CPU constants so the per-layer HCU path can process each request without
    # synchronizing GPU metadata back to the host.
    ragged_request_slices: tuple[tuple[int, int, int, int], ...]
    # Reused zero lower bounds for request-local MQA calls.
    ragged_q_local_ks: torch.Tensor  # int32 [sum_q]
    ragged_total_k_rows: int  # sum_pool_pages * page_size
    # Layer-shared scratch (alloc out of the per-layer hot path).
    ragged_k_u8: Optional[torch.Tensor]  # uint8 [total_k_rows, head_dim]
    ragged_k_scale: Optional[torch.Tensor]  # fp32 [total_k_rows]
    # Zero-copy reference to ``req_to_token`` ([pool_size, max_context_len]);
    # None unless PAGED + fuse-topk on. The fused topk kernel looks up each
    # q-token's row via ``ragged_paged_page_table_row_index`` (below) instead
    # of a dense [sum_q, max_seq_len] per-q replication.
    ragged_paged_page_table: Optional[torch.Tensor]  # int32 [pool_size, max_ctx]
    # Per-q page-table row = its request's req_pool_index; int32 [sum_q].
    ragged_paged_page_table_row_index: Optional[torch.Tensor]  # int32 [sum_q]

    cp: Optional[KPoolCpInfo] = None


@dataclass(frozen=True)
class KPoolWritePlan:
    """Layer-invariant plan for kpool tail-write + closed-pool compress.

    Shared by decode (N=1) and target_verify (N=num_draft_tokens). ``req`` and
    ``write_start`` stay per-batch, while close-pool addressing is flattened as
    ``[B * max_closed_pools]`` so EAGLE topk=1 chain-only verify can close more
    than one kpool when ``N > POOL_SIZE``.
    """

    req: torch.Tensor  # int64 [B]
    write_start: torch.Tensor  # int32 [B] -- decode: positions; verify: committed
    tail_logical_start: torch.Tensor  # int32 [B * max_closed_pools]
    write_loc: torch.Tensor  # int64 [B * max_closed_pools]

    num_draft_tokens: int
    max_closed_pools: int = 1

    # Verify-only (decode leaves these None).
    paged_page_table: Optional[torch.Tensor] = None  # int32 [B*N, max_seq_pages]
    pool_seqlens_per_q: Optional[torch.Tensor] = None  # int32 [B*N]
    seqlens_per_q: Optional[torch.Tensor] = None  # int32 [B*N]
    pool_schedule_metadata: Optional[torch.Tensor] = None

    # V2 draft_extend only: the just-finished verify's ``accept_length``
    # (which v2 emits already incremented to include the bonus "next"
    # token -- see eagle_info_v2.sample's trailing ``accept_length.add_(1)``),
    # used to gate the close-pool compress to the REAL advance rather than
    # the full N drafts. Skipping compress on the speculative tail
    # (positions past committed+effective_n) is safe because those
    # tail-ring slots will be overwritten by the next round's V2 write
    # before any reader needs them; deferring the FP8 quantize to the
    # round that actually commits past the pool boundary uses real (not
    # speculative) K, improving cache quality. ``None`` for verify/decode
    # -> kernel falls back to N as the gating window.
    effective_n_per_batch: Optional[torch.Tensor] = None  # int32 [B]


@dataclass
class _KPoolCpuPlan:
    """Raw per-batch lists; converted to GPU tensors in ``_kpool_plan_to_gpu``."""

    pool_batch_idx: List[int] = field(default_factory=list)
    pool_req: List[int] = field(default_factory=list)
    pool_pool_id: List[int] = field(default_factory=list)
    pool_n_from_tail: List[int] = field(default_factory=list)
    pool_chunk_src: List[int] = field(default_factory=list)
    # Logical position of the in-progress pool's first slot (per closed pool
    # row). Used by the ring-addressed tail read in the assemble kernel.
    pool_tail_logical_base: List[int] = field(default_factory=list)

    tail_req: List[int] = field(default_factory=list)
    # Logical position where this batch's new tail tokens start.
    tail_dst_logical_start: List[int] = field(default_factory=list)
    tail_chunk_src: List[int] = field(default_factory=list)
    tail_n_write: List[int] = field(default_factory=list)
    ragged_q_len: List[int] = field(default_factory=list)
    ragged_pool_pages: List[int] = field(default_factory=list)
    # Exclusive prefix sums (per batch start offsets). Built on CPU and
    # H2D'd alongside the other i32 counters so the Triton ragged-layout
    # kernel can read them directly without a GPU cumsum.
    cu_pages_excl: List[int] = field(default_factory=list)
    cu_q_len_excl: List[int] = field(default_factory=list)
    # Rolling sum of ragged_pool_pages, kept here so _kpool_plan_to_gpu
    # avoids a redundant Python sum() pass.
    total_pool_pages: int = 0
    # n_rag = len(ragged_q_len) = batch_size (every batch contributes a row).
    # ragged_batch_idx is therefore arange(batch_size); recomputed in
    # _kpool_plan_to_gpu instead of being H2D'd as a redundant list.


def _build_ragged_request_slices(
    q_starts: List[int],
    q_lens: List[int],
    page_starts: List[int],
    pool_pages: List[int],
    slots_per_page: int,
) -> tuple[tuple[int, int, int, int], ...]:
    """Build request-local ``(q_start, q_end, k_start, k_end)`` slices."""
    return tuple(
        (
            q_start,
            q_start + q_len,
            page_start * slots_per_page,
            (page_start + request_pool_pages) * slots_per_page,
        )
        for q_start, q_len, page_start, request_pool_pages in zip(
            q_starts,
            q_lens,
            page_starts,
            pool_pages,
            strict=True,
        )
    )


class _KPoolDecompose(NamedTuple):
    """Splice/bulk/tail decomposition of one req's compress write.

    For ``length`` new tokens appended at ``start`` into a pool layout of
    ``pool_size``:
      * ``first_slot``: in-progress tail offset at ``start`` (0..pool_size-1)
      * ``base_pool``: pool index containing ``start``
      * ``n_pool``: number of *closed* pools produced by these new tokens.
        Row 0 splices ``first_slot`` saved-tail tokens with
        ``pool_size - first_slot`` chunk tokens; rows 1..n_pool-1 are bulk
        pools (``n_from_tail == 0``).
      * ``tail_n_write``: residual tokens left in the in-progress pool
        after the closed rows. Zero when fully consumed.
    """

    first_slot: int
    base_pool: int
    n_pool: int
    tail_n_write: int


def _is_kpool_layout_enabled(pool_size: int, real_page_size: int) -> bool:
    """The 3 hardware/layout preconditions every kpool init/update gate on.

    Single source of truth: pool actually on (``pool_size > 1``), the
    DeepGEMM-required 64-token page, and pool slots evenly packed into
    one page (``page_size % pool_size == 0``). Mode-specific predicates
    (decode / extend / target_verify / cuda availability) stay at the
    call site.
    """
    return pool_size > 1 and real_page_size == 64 and real_page_size % pool_size == 0


def _decompose_compress(start: int, length: int, pool_size: int) -> _KPoolDecompose:
    first_slot = start % pool_size
    base_pool = start // pool_size
    n_pool = (start + length) // pool_size - base_pool
    consumed = max(0, n_pool * pool_size - first_slot)
    tail_n = length - consumed
    return _KPoolDecompose(
        first_slot=first_slot,
        base_pool=base_pool,
        n_pool=n_pool,
        tail_n_write=tail_n,
    )


def _append_compress_rows(
    plan: _KPoolCpuPlan,
    pool_size: int,
    batch_size: int,
    extend_seq_lens_cpu: List[int],
    seq_lens_cpu: List[int],
    req_pool_indices_cpu: List[int],
) -> None:
    """Compress side: per-pool write rows + per-batch tail rows.

    Under CP round-robin-split, ``key`` is all-gathered to full-seq before
    compress, so the compress side still operates on the *full* batch /
    seq layout regardless of CP split mode.

    Row 0 may "splice" ``first_slot`` saved-tail tokens with
    ``pool_size - first_slot`` chunk tokens to close the mid-pool the
    prefix started in; subsequent rows are aligned-bulk pools. When
    ``first_slot == 0`` the splice degenerates to a plain bulk row
    (``n_from_tail = 0``), so a single uniform code path covers both.
    """
    q_offset = 0
    for i in range(batch_size):
        q_len = extend_seq_lens_cpu[i]
        assert q_len > 0, f"extend_seq_lens_cpu[{i}] = {q_len}; expected > 0"

        seq_len = seq_lens_cpu[i]
        req = req_pool_indices_cpu[i]
        d = _decompose_compress(seq_len - q_len, q_len, pool_size)

        if d.n_pool > 0:
            plan.pool_batch_idx.extend([i] * d.n_pool)
            plan.pool_req.extend([req] * d.n_pool)
            plan.pool_pool_id.extend(range(d.base_pool, d.base_pool + d.n_pool))
            plan.pool_n_from_tail.append(d.first_slot)
            plan.pool_n_from_tail.extend([0] * (d.n_pool - 1))
            bulk_start = q_offset + pool_size - d.first_slot
            plan.pool_chunk_src.append(q_offset)
            plan.pool_chunk_src.extend(
                range(bulk_start, bulk_start + (d.n_pool - 1) * pool_size, pool_size)
            )
            # Logical base for each closed pool (= pool_id * pool_size).
            plan.pool_tail_logical_base.extend(
                range(
                    d.base_pool * pool_size,
                    (d.base_pool + d.n_pool) * pool_size,
                    pool_size,
                )
            )

        if d.tail_n_write > 0:
            consumed = q_len - d.tail_n_write
            plan.tail_req.append(req)
            # New in-progress tail tokens start at this logical position.
            plan.tail_dst_logical_start.append(seq_len - q_len + consumed)
            plan.tail_chunk_src.append(q_offset + consumed)
            plan.tail_n_write.append(d.tail_n_write)

        q_offset += q_len


def _append_local_rows(
    plan: _KPoolCpuPlan,
    pool_size: int,
    page_size: int,
    local_extend_seq_lens_cpu: List[int],
    local_seq_lens_cpu: List[int],
) -> None:
    """Local view: per-batch ragged layout (q_len + pool_pages + prefix sums).

    Under CP round-robin-split, these are the *rank-local* per-batch
    counts (only batches present on this rank, with this rank's q
    counts); under non-CP they equal the full-seq counts. q is NOT
    gathered, so the ragged layout sizes match this rank's q footprint.
    """
    slots_per_page = page_size // pool_size
    q_offset = 0
    for q_len, seq_len in zip(
        local_extend_seq_lens_cpu, local_seq_lens_cpu, strict=True
    ):
        assert q_len > 0, f"local_extend_seq_lens_cpu has non-positive {q_len = }"
        plan.ragged_q_len.append(q_len)
        pool_seq_len = seq_len // pool_size
        pool_pages_i = (pool_seq_len + slots_per_page - 1) // slots_per_page
        plan.ragged_pool_pages.append(pool_pages_i)
        plan.cu_pages_excl.append(plan.total_pool_pages)
        plan.cu_q_len_excl.append(q_offset)
        plan.total_pool_pages += pool_pages_i
        q_offset += q_len


def _kpool_cpu_plan(
    forward_batch: ForwardBatch,
    pool_size: int,
    page_size: int = 64,
    *,
    local_extend_seq_lens_cpu: Optional[List[int]] = None,
    local_seq_lens_cpu: Optional[List[int]] = None,
) -> _KPoolCpuPlan:
    """Build the combined compress + local CPU plan.

    Compress side: always full-seq (K is all-gathered before compress
    under CP, so write_locs index the full pool table).

    Local view: same full layout when no override is provided; otherwise
    the rank-local slice for CP round-robin-split, which lets
    ``_get_topk_ragged`` build per-q ks/ke for this rank's tokens
    instead of all-gathering q. ``local_*`` come paired (both None or
    both set).
    """
    plan = _KPoolCpuPlan()

    extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
    if isinstance(extend_seq_lens_cpu, torch.Tensor):
        extend_seq_lens_cpu = extend_seq_lens_cpu.tolist()
    seq_lens_cpu = forward_batch.seq_lens_cpu.tolist()
    req_pool_indices_cpu = forward_batch.req_pool_indices.tolist()

    _append_compress_rows(
        plan,
        pool_size,
        forward_batch.batch_size,
        extend_seq_lens_cpu,
        seq_lens_cpu,
        req_pool_indices_cpu,
    )

    if local_extend_seq_lens_cpu is None:
        local_extend_seq_lens_cpu = extend_seq_lens_cpu
        local_seq_lens_cpu = seq_lens_cpu

    _append_local_rows(
        plan,
        pool_size,
        page_size,
        local_extend_seq_lens_cpu,
        local_seq_lens_cpu,
    )
    return plan


def _kpool_plan_to_gpu(
    cpu: _KPoolCpuPlan,
    forward_batch: ForwardBatch,
    full_real_page_table: torch.Tensor,
    local_real_page_table: torch.Tensor,
    local_seqlens_expanded: torch.Tensor,
    local_req_pool_indices: torch.Tensor,
    pool_size: int,
    page_size: int,
    topk_transform_method: TopkTransformMethod,
) -> KPoolExtendPlan:
    """Pack lists into two pinned tensors (int64 indices + int32 small
    counts) for one H2D each; ragged-topk ``src_idx`` is computed on
    GPU from per-batch ``pool_pages``.

    ``full_real_page_table`` is used for the compress-side ``write_locs``
    (the FP8 cache index is full-seq because K is gathered before
    compress). ``local_real_page_table`` / ``local_seqlens_expanded`` are
    rank-local under CP round-robin-split, matching this rank's q
    footprint -- see ``_get_topk_ragged`` for how they're consumed.
    """
    from sglang.srt.layers.attention.glm5_next.dsa_backend import TopkTransformMethod

    device = forward_batch.seq_lens.device
    n_pool = len(cpu.pool_pool_id)
    n_tail = len(cpu.tail_req)
    n_rag = len(cpu.ragged_q_len)

    slots_per_page = page_size // pool_size
    total_pool_pages = cpu.total_pool_pages
    ragged_total_k_rows = total_pool_pages * slots_per_page
    ragged_request_slices = _build_ragged_request_slices(
        cpu.cu_q_len_excl,
        cpu.ragged_q_len,
        cpu.cu_pages_excl,
        cpu.ragged_pool_pages,
        slots_per_page,
    )

    need_paged = (
        topk_transform_method == TopkTransformMethod.PAGED
        and envs.SGLANG_DSA_FUSE_TOPK.get()
        and n_rag > 0
    )

    # int64 H2D: pool_req | pool_pool_id | pool_chunk_src | pool_batch_idx
    #          | tail_req | tail_chunk_src
    i64_total = 4 * n_pool + 2 * n_tail
    if i64_total > 0:
        i64_cpu = torch.tensor(
            cpu.pool_req
            + cpu.pool_pool_id
            + cpu.pool_chunk_src
            + cpu.pool_batch_idx
            + cpu.tail_req
            + cpu.tail_chunk_src,
            dtype=torch.int64,
            pin_memory=True,
        )
        i64_gpu = i64_cpu.to(device, non_blocking=True)
        c = 0
        pool_req_t = i64_gpu[c : c + n_pool]
        c += n_pool
        pool_pool_id_t = i64_gpu[c : c + n_pool]
        c += n_pool
        pool_chunk_src_t = i64_gpu[c : c + n_pool]
        c += n_pool
        pool_batch_idx_t = i64_gpu[c : c + n_pool]
        c += n_pool
        tail_req_t = i64_gpu[c : c + n_tail]
        c += n_tail
        tail_chunk_src_t = i64_gpu[c : c + n_tail]
    else:
        empty_i64 = torch.empty((0,), dtype=torch.int64, device=device)
        pool_req_t = pool_pool_id_t = pool_chunk_src_t = pool_batch_idx_t = empty_i64
        tail_req_t = tail_chunk_src_t = empty_i64

    # int32 H2D: pool_n_from_tail | pool_tail_logical_base
    #          | tail_dst_logical_start | tail_n_write
    #          | ragged_pool_pages | ragged_q_len
    #          | cu_pages_excl | cu_q_len_excl
    i32_total = 2 * n_pool + 2 * n_tail + 4 * n_rag
    if i32_total > 0:
        i32_cpu = torch.tensor(
            cpu.pool_n_from_tail
            + cpu.pool_tail_logical_base
            + cpu.tail_dst_logical_start
            + cpu.tail_n_write
            + cpu.ragged_pool_pages
            + cpu.ragged_q_len
            + cpu.cu_pages_excl
            + cpu.cu_q_len_excl,
            dtype=torch.int32,
            pin_memory=True,
        )
        i32_gpu = i32_cpu.to(device, non_blocking=True)
        c = 0
        pool_n_from_tail_t = i32_gpu[c : c + n_pool]
        c += n_pool
        pool_tail_logical_base_t = i32_gpu[c : c + n_pool]
        c += n_pool
        tail_dst_logical_start_t = i32_gpu[c : c + n_tail]
        c += n_tail
        tail_n_write_t = i32_gpu[c : c + n_tail]
        c += n_tail
        ragged_pool_pages_t = i32_gpu[c : c + n_rag]
        c += n_rag
        ragged_q_len_t = i32_gpu[c : c + n_rag]
        c += n_rag
        cu_pages_excl_t = i32_gpu[c : c + n_rag]
        c += n_rag
        cu_q_len_excl_t = i32_gpu[c : c + n_rag]
    else:
        empty_i32 = torch.empty((0,), dtype=torch.int32, device=device)
        pool_n_from_tail_t = pool_tail_logical_base_t = empty_i32
        tail_dst_logical_start_t = tail_n_write_t = empty_i32
        ragged_pool_pages_t = ragged_q_len_t = empty_i32
        cu_pages_excl_t = cu_q_len_excl_t = empty_i32

    # Compress side reads ``full_real_page_table``; local view reads
    # ``local_real_page_table`` / ``local_seqlens_expanded``. They diverge
    # under CP round-robin-split: K is gathered before compress (so the
    # compress plan is full-seq), but q stays rank-local for topk.

    if n_pool > 0:
        # Compressed pool slots are packed `page_size`-per-page (DeepGEMM's
        # fp8_paged_mqa_logits historically expects 64-token pages). Each
        # logical pooled-K id maps to `packed_page * slots_per_page +
        # pool_id % slots_per_page`, with `packed_page` looked up via the
        # per-batch page table row chosen by `pool_batch_idx_t`.
        pool_page_group = torch.div(
            pool_pool_id_t, slots_per_page, rounding_mode="floor"
        )
        packed_page = full_real_page_table[pool_batch_idx_t, pool_page_group].to(
            torch.int64
        )
        pool_write_locs = packed_page * slots_per_page + torch.remainder(
            pool_pool_id_t, slots_per_page
        )
    else:
        pool_write_locs = torch.empty((0,), dtype=torch.int64, device=device)

    pooled_seq_lens_expanded = torch.div(
        local_seqlens_expanded, pool_size, rounding_mode="floor"
    ).to(torch.int32)

    if n_rag > 0:
        # Single Triton kernel writes concat_page_table / ragged_q_ks / ragged_q_ke.
        # Per-batch prefix sums (cu_pages_excl, cu_q_len_excl) were built on CPU
        # and arrived via the i32 H2D above, so no GPU cumsum is needed here.
        (
            ragged_concat_page_table,
            ragged_q_ks,
            ragged_q_ke,
        ) = kpool_build_ragged_layout(
            full_page_table=local_real_page_table,
            cu_pages_excl=cu_pages_excl_t,
            ragged_pool_pages=ragged_pool_pages_t,
            cu_q_len_excl=cu_q_len_excl_t,
            ragged_q_len=ragged_q_len_t,
            pooled_seq_lens_expanded=pooled_seq_lens_expanded,
            slots_per_page=slots_per_page,
            total_pool_pages=total_pool_pages,
            total_q=pooled_seq_lens_expanded.shape[0],
        )
    else:
        empty_i32_dev = torch.empty((0,), dtype=torch.int32, device=device)
        ragged_concat_page_table = empty_i32_dev
        ragged_q_ks = empty_i32_dev
        ragged_q_ke = empty_i32_dev
    ragged_q_local_ks = torch.zeros_like(ragged_q_ks)

    # Build once per forward to avoid an O(B*layers) Python loop in the indexer.
    ragged_paged_page_table = None
    ragged_paged_page_table_row_index = None
    if need_paged:
        req_to_token = get_req_to_token_pool().req_to_token
        # repeat_interleave with GPU repeats stays on-device (no CPU sync).
        ragged_paged_page_table_row_index = torch.repeat_interleave(
            local_req_pool_indices.to(torch.int32), ragged_q_len_t
        )
        ragged_paged_page_table = req_to_token

    # Layer-shared scratch backed by a forward-persistent lazy-grow buffer
    # (avoids two allocations per forward; sliced to actual size below).
    if ragged_total_k_rows > 0:
        ragged_k_u8, ragged_k_scale = _get_ragged_scratch(ragged_total_k_rows, device)
    else:
        ragged_k_u8 = None
        ragged_k_scale = None

    return KPoolExtendPlan(
        writes=PoolWriteRows(
            req=pool_req_t,
            pool_id=pool_pool_id_t,
            n_from_tail=pool_n_from_tail_t,
            chunk_src=pool_chunk_src_t,
            tail_logical_base=pool_tail_logical_base_t,
            write_loc=pool_write_locs,
        ),
        tails=TailWriteRows(
            req=tail_req_t,
            dst_logical_start=tail_dst_logical_start_t,
            chunk_src=tail_chunk_src_t,
            n_write=tail_n_write_t,
        ),
        pooled_seq_lens_expanded=pooled_seq_lens_expanded,
        seq_lens_expanded=local_seqlens_expanded,
        ragged_concat_page_table=ragged_concat_page_table,
        ragged_q_ks=ragged_q_ks,
        ragged_q_ke=ragged_q_ke,
        ragged_request_slices=ragged_request_slices,
        ragged_q_local_ks=ragged_q_local_ks,
        ragged_total_k_rows=ragged_total_k_rows,
        ragged_k_u8=ragged_k_u8,
        ragged_k_scale=ragged_k_scale,
        ragged_paged_page_table=ragged_paged_page_table,
        ragged_paged_page_table_row_index=ragged_paged_page_table_row_index,
        cp=_kpool_cp_owner_rank(forward_batch, n_pool, device),
    )


def _kpool_cp_owner_rank(
    forward_batch: ForwardBatch,
    n_pool: int,
    device: torch.device,
) -> Optional[KPoolCpInfo]:
    """Owner rank = ``row_idx % cp_size``; balances writes by flat row
    index regardless of per-request pool distribution.
    """
    if not dsa_use_prefill_cp(forward_batch):
        return None

    cp_size = get_parallel().attn_cp_size
    if cp_size <= 1:
        return None

    cp_rank = get_parallel().attn_cp_rank
    if n_pool > 0:
        owner = torch.arange(n_pool, dtype=torch.int32, device=device) % cp_size
        local_write_mask = owner == cp_rank
    else:
        owner = torch.empty((0,), dtype=torch.int32, device=device)
        local_write_mask = torch.empty((0,), dtype=torch.bool, device=device)
    return KPoolCpInfo(
        size=cp_size,
        rank=cp_rank,
        owner_rank=owner,
        local_write_mask=local_write_mask,
    )


def init_kpool_extend_metadata(
    metadata: DSAMetadata,
    forward_batch: ForwardBatch,
    *,
    pool_size: int,
    real_page_size: int,
    topk_transform_method: TopkTransformMethod,
    full_real_page_table: torch.Tensor,
    full_seqlens_expanded: torch.Tensor,
    local_real_page_table: Optional[torch.Tensor] = None,
    local_seqlens_expanded: Optional[torch.Tensor] = None,
    local_extend_seq_lens_cpu: Optional[List[int]] = None,
    local_seq_lens_cpu: Optional[List[int]] = None,
    local_req_pool_indices: Optional[torch.Tensor] = None,
) -> DSAMetadata:
    """Build the layer-invariant kpool extend plan once per forward.

    Two views:
      * Compress (full-seq) always reads ``full_real_page_table`` --
        K is gathered before compress under CP, so write_locs index the
        full pool table.
      * Local view defaults to the full layout; under CP round-robin-split
        the caller passes the rank-local ``local_*`` tensors and per-batch
        counts so topk runs on this rank's q slice without a q all_gather.

    Returns input unchanged when the gate fails: extend-like mode
    (extend / draft_extend v2), valid seq lens, and the shared
    kpool layout (see ``_is_kpool_layout_enabled``).
    """
    mode = forward_batch.forward_mode
    is_extend_like = mode.is_extend_without_speculative() or mode.is_draft_extend_v2()
    if (
        not _is_kpool_layout_enabled(pool_size, real_page_size)
        or not is_extend_like
        or forward_batch.extend_seq_lens_cpu is None
        or forward_batch.seq_lens_cpu is None
    ):
        return metadata

    if local_real_page_table is None:
        local_real_page_table = full_real_page_table
    if local_seqlens_expanded is None:
        local_seqlens_expanded = full_seqlens_expanded
    if local_req_pool_indices is None:
        local_req_pool_indices = forward_batch.req_pool_indices

    cpu = _kpool_cpu_plan(
        forward_batch,
        pool_size,
        real_page_size,
        local_extend_seq_lens_cpu=local_extend_seq_lens_cpu,
        local_seq_lens_cpu=local_seq_lens_cpu,
    )
    plan = _kpool_plan_to_gpu(
        cpu,
        forward_batch,
        full_real_page_table,
        local_real_page_table,
        local_seqlens_expanded,
        local_req_pool_indices,
        pool_size,
        real_page_size,
        topk_transform_method,
    )
    return dataclasses.replace(metadata, kpool_extend_plan=plan)


def init_pooled_paged_mqa_metadata(
    metadata: DSAMetadata,
    seqlens_32: torch.Tensor,
    forward_mode: ForwardMode,
    *,
    pool_size: int,
    real_page_size: int,
) -> DSAMetadata:
    """Build decode-side pooled cache seqlens + deep_gemm schedule.

    Returns the (possibly updated) metadata. No-op when gate fails:
    CUDA + decode/idle mode + the shared kpool layout
    (see ``_is_kpool_layout_enabled``).
    """
    if (
        not _is_kpool_layout_enabled(pool_size, real_page_size)
        or not (is_cuda() or is_hcu())
        or not forward_mode.is_decode_or_idle()
    ):
        return metadata

    pooled_cache_seqlens = torch.div(seqlens_32, pool_size, rounding_mode="floor").to(
        torch.int32
    )
    deep_gemm = _get_deep_gemm()
    if deep_gemm is not None:
        slots_per_page = real_page_size // pool_size
        pooled_schedule = deep_gemm.get_paged_mqa_logits_metadata(
            pooled_cache_seqlens.unsqueeze(-1),
            slots_per_page,
            deep_gemm.get_num_sms(),
        )
    else:
        pooled_schedule = None

    return dataclasses.replace(
        metadata,
        pooled_index_kpool=pool_size,
        pooled_cache_seqlens_int32=pooled_cache_seqlens,
        pooled_paged_mqa_schedule_metadata=pooled_schedule,
    )


def update_pooled_paged_mqa_metadata(
    metadata: DSAMetadata,
    seqlens_32: torch.Tensor,
    forward_mode: ForwardMode,
    *,
    pool_size: int,
    real_page_size: int,
) -> None:
    """In-place refresh of decode-side pooled metadata (cuda-graph replay).

    Precondition: the pooled buffers were either both allocated (and
    pooled_index_kpool set to pool_size) or both left None during cuda-graph
    capture under the same gating conditions. Replay must observe the same
    gating, otherwise the captured graph's tensor addresses are invalid --
    so we either copy_ into the existing buffers or no-op; we never alloc /
    reset fields here.
    """
    if (
        not _is_kpool_layout_enabled(pool_size, real_page_size)
        or not (is_cuda() or is_hcu())
        or not forward_mode.is_decode_or_idle()
        or metadata.pooled_cache_seqlens_int32 is None
        or metadata.pooled_index_kpool != pool_size
    ):
        return

    pool_seqlens = metadata.pooled_cache_seqlens_int32[: seqlens_32.shape[0]]
    torch.div(
        seqlens_32,
        pool_size,
        rounding_mode="floor",
        out=pool_seqlens,
    )

    deep_gemm = _get_deep_gemm()
    if deep_gemm is None:
        return
    if metadata.pooled_paged_mqa_schedule_metadata is None:
        return
    try:
        new_schedule = deep_gemm.get_paged_mqa_logits_metadata(
            pool_seqlens.unsqueeze(-1),
            real_page_size // pool_size,
            deep_gemm.get_num_sms(),
        )
        metadata.pooled_paged_mqa_schedule_metadata.copy_(
            new_schedule, non_blocking=True
        )
    except Exception:
        # capture saw deep_gemm absent -> schedule buffer is None already
        pass


# ---------------------------------------------------------------------------
# Unified write plan (decode + target_verify): per-batch tail write + closed-
# pool compress addressing. Decode is the N=1 special case of verify.
# Dataclass ``KPoolWritePlan`` lives at the top of the file alongside the
# extend Plan classes.
# ---------------------------------------------------------------------------


def _alloc_kpool_write_plan_buffers(
    *,
    max_bs: int,
    pool_size: int,
    num_draft_tokens: int,
    device: torch.device,
    is_verify: bool,
    is_v2: bool = False,
) -> KPoolWritePlan:
    """Allocate worst-case ``[max_bs]`` write-plan buffers; values filled by
    ``update_kpool_write_plan_cuda_graph``. Eager uses ``max_bs = bs``;
    capture uses the cuda-graph max batch.
    """
    max_closed_pools = max(1, (num_draft_tokens + pool_size - 1) // pool_size)
    n_rows = max_bs * num_draft_tokens
    n_close_rows = max_bs * max_closed_pools
    verify_extras = {}
    if is_verify:
        verify_extras = dict(
            pool_seqlens_per_q=torch.zeros(n_rows, dtype=torch.int32, device=device),
            seqlens_per_q=torch.zeros(n_rows, dtype=torch.int32, device=device),
        )
    if is_v2:
        verify_extras["effective_n_per_batch"] = torch.zeros(
            max_bs, dtype=torch.int32, device=device
        )
    return KPoolWritePlan(
        req=torch.zeros(max_bs, dtype=torch.int64, device=device),
        write_start=torch.zeros(max_bs, dtype=torch.int32, device=device),
        tail_logical_start=torch.zeros(n_close_rows, dtype=torch.int32, device=device),
        write_loc=torch.zeros(n_close_rows, dtype=torch.int64, device=device),
        num_draft_tokens=num_draft_tokens,
        max_closed_pools=max_closed_pools,
        **verify_extras,
    )


def _compute_pool_schedule_metadata(
    pool_seqlens_per_q: torch.Tensor,
    *,
    slots_per_page: int,
) -> Optional[torch.Tensor]:
    """Per-pool DeepGEMM schedule metadata; None if deep_gemm absent."""
    if not (is_cuda() or is_hcu()):
        return None
    deep_gemm = _get_deep_gemm()
    if deep_gemm is None:
        return None
    try:
        return deep_gemm.get_paged_mqa_logits_metadata(
            pool_seqlens_per_q.unsqueeze(-1),
            slots_per_page,
            deep_gemm.get_num_sms(),
        )
    except Exception:
        return None


def init_kpool_write_plan_capture(
    metadata: DSAMetadata,
    *,
    max_bs: int,
    pool_size: int,
    real_page_size: int,
    real_page_table: torch.Tensor,
    num_draft_tokens: int,
    device: torch.device,
    is_verify: bool,
    is_v2: bool = False,
) -> DSAMetadata:
    """Pre-allocate the kpool write plan at the worst-case ``[max_bs]`` shape.

    Shared by decode (``num_draft_tokens=1``, ``is_verify=False``),
    target_verify (``num_draft_tokens=N``, ``is_verify=True``) and
    draft_extend_v2 (``num_draft_tokens=N``, ``is_verify=True``,
    ``is_v2=True``). For verify/v2, ``real_page_table`` is the
    ``repeat_interleave(N)``-shaped ``[max_bs*N, max_pages]`` table; for
    decode it's the plain ``[max_bs, max_pages]``.

    Verify-only fields (``pool_seqlens_per_q`` / ``seqlens_per_q`` /
    ``pool_schedule_metadata`` / ``paged_page_table``) are allocated and
    wired only when ``is_verify`` is True; for decode they stay None.
    ``effective_n_per_batch`` is v2-only.
    """
    if not _is_kpool_layout_enabled(pool_size, real_page_size) or num_draft_tokens == 0:
        return metadata

    plan = _alloc_kpool_write_plan_buffers(
        max_bs=max_bs,
        pool_size=pool_size,
        num_draft_tokens=num_draft_tokens,
        device=device,
        is_verify=is_verify,
        is_v2=is_v2,
    )
    if is_verify:
        slots_per_page = real_page_size // pool_size
        schedule = _compute_pool_schedule_metadata(
            plan.pool_seqlens_per_q,
            slots_per_page=slots_per_page,
        )
        plan = dataclasses.replace(
            plan,
            paged_page_table=real_page_table,
            pool_schedule_metadata=schedule,
        )
    return dataclasses.replace(metadata, kpool_write_plan=plan)


def update_kpool_write_plan(
    metadata: DSAMetadata,
    *,
    write_start: torch.Tensor,
    req_pool_indices: torch.Tensor,
    real_page_table: torch.Tensor,
    pool_size: int,
    real_page_size: int,
    num_draft_tokens: int,
    forward_mode: ForwardMode,
    accept_length: Optional[torch.Tensor] = None,
) -> None:
    """Rebuild the kpool write plan device-side, in-place.

    Capture (or eager) must have already allocated ``metadata.kpool_write_plan``
    via ``init_kpool_write_plan_capture``. This call fires the GPU
    plan-build kernel and (for verify / v2 only) refreshes
    ``pool_schedule_metadata``.

    Gate: shared kpool layout + cuda + ring-write mode
    (decode / target_verify / draft_extend_v2).

    ``write_start`` carries the per-batch logical write position:
      decode: forward_batch.positions[:bs] (= seq_lens - 1)
      verify: forward_batch.seq_lens[:bs]  (= committed)
      v2:     forward_batch.seq_lens - N   (= extend_prefix_lens)

    ``accept_length`` (v2 only): int32 [B] of the just-finished verify's
    accept_length, already incremented in eagle_info_v2.sample to include
    the bonus "next" token (so it IS the real advance, not raw accepted
    drafts). Stored on the plan as ``effective_n_per_batch = accept_length``
    so the write kernel can gate compress on the real advance rather than
    the full N drafts (safe because rejected draft K in the tail ring gets
    overwritten by the next round's V2 write before any reader needs it).
    """
    if not _is_kpool_layout_enabled(pool_size, real_page_size) or not (
        is_cuda() or is_hcu()
    ):
        return
    is_verify = forward_mode.is_target_verify()
    is_decode = forward_mode.is_decode_or_idle()
    is_v2 = forward_mode.is_draft_extend_v2()
    if not (is_verify or is_decode or is_v2):
        return
    plan = metadata.kpool_write_plan
    assert plan is not None, (
        "kpool_write_plan must be pre-allocated before update; "
        "see init_kpool_write_plan_capture"
    )
    slots_per_page = real_page_size // pool_size
    kernel_kwargs = dict(
        write_start=write_start,
        req_pool_indices=req_pool_indices,
        real_page_table=real_page_table,
        req_out=plan.req,
        write_start_out=plan.write_start,
        tail_logical_start_out=plan.tail_logical_start,
        write_loc_out=plan.write_loc,
        pool_seqlens_per_q_out=plan.pool_seqlens_per_q,
        seqlens_per_q_out=plan.seqlens_per_q,
        pool_size=pool_size,
        num_draft_tokens=num_draft_tokens,
        max_closed_pools=plan.max_closed_pools,
        slots_per_page=slots_per_page,
    )
    if plan.max_closed_pools == 1:
        single_close_kwargs = dict(kernel_kwargs)
        single_close_kwargs.pop("max_closed_pools")
        aot_kpool_write_plan = _get_aot_kpool_write_plan()
        try:
            if aot_kpool_write_plan is not None:
                aot_kpool_write_plan(**single_close_kwargs)
            else:
                if _disable_jit_kpool_write_plan():
                    raise RuntimeError(
                        "disabled by SGLANG_DISABLE_JIT_KPOOL_WRITE_PLAN"
                    )
                kpool_write_plan_cuda(**single_close_kwargs)
        except Exception as e:
            global _KPOOL_WRITE_PLAN_SINGLE_JIT_WARNED
            if aot_kpool_write_plan is not None:
                try:
                    if _disable_jit_kpool_write_plan():
                        raise RuntimeError(
                            "disabled by SGLANG_DISABLE_JIT_KPOOL_WRITE_PLAN"
                        )
                    kpool_write_plan_cuda(**single_close_kwargs)
                    e = None
                except Exception as jit_e:
                    e = jit_e
            if e is not None:
                if (
                    not _KPOOL_WRITE_PLAN_SINGLE_JIT_WARNED
                    and not _disable_jit_kpool_write_plan()
                ):
                    logger.warning(
                        "AOT/JIT kpool write-plan update failed; "
                        "falling back to Triton update: %s",
                        e,
                    )
                    _KPOOL_WRITE_PLAN_SINGLE_JIT_WARNED = True
                update_kpool_write_plan_cuda_graph(**kernel_kwargs)
    else:
        try:
            if _disable_jit_kpool_write_plan():
                raise RuntimeError("disabled by SGLANG_DISABLE_JIT_KPOOL_WRITE_PLAN")
            update_kpool_write_plan_cuda_graph(**kernel_kwargs)
        except Exception:
            # Preserve the same exception surface as the single-close fallback.
            raise
    if is_v2 and accept_length is not None and plan.effective_n_per_batch is not None:
        # effective_n = accept_length directly. v2's accept_length already
        # includes the bonus "next" token (eagle_info_v2.sample applies an
        # in-place ``accept_length.add_(1)`` before return), so it IS the
        # real advance; adding +1 here would over-count by one slot and
        # spuriously fire compress on rounds that didn't truly cross.
        plan.effective_n_per_batch.copy_(accept_length.to(torch.int32))
    if plan.pool_schedule_metadata is not None:
        new_schedule = _compute_pool_schedule_metadata(
            plan.pool_seqlens_per_q,
            slots_per_page=slots_per_page,
        )
        if new_schedule is not None:
            plan.pool_schedule_metadata.copy_(new_schedule)


def update_kpool_write_plan_multi_decode(
    metadata0: DSAMetadata,
    metadata1: DSAMetadata,
    metadata2: DSAMetadata,
    metadata3: Optional[DSAMetadata] = None,
    *,
    write_start: torch.Tensor,
    req_pool_indices: torch.Tensor,
    real_page_table: torch.Tensor,
    pool_size: int,
    real_page_size: int,
) -> None:
    """Refresh three backend-local decode write plans in one launch."""
    if not _is_kpool_layout_enabled(pool_size, real_page_size) or not (
        is_cuda() or is_hcu()
    ):
        return

    plan0 = metadata0.kpool_write_plan
    plan1 = metadata1.kpool_write_plan
    plan2 = metadata2.kpool_write_plan
    plan3 = metadata3.kpool_write_plan if metadata3 is not None else None
    assert plan0 is not None and plan1 is not None and plan2 is not None, (
        "kpool_write_plan must be pre-allocated before multi update; "
        "see init_kpool_write_plan_capture"
    )
    if metadata3 is not None:
        assert plan3 is not None, (
            "fourth kpool_write_plan must be pre-allocated before multi update; "
            "see init_kpool_write_plan_capture"
        )

    slots_per_page = real_page_size // pool_size
    if plan3 is not None and not _disable_jit_kpool_write_plan():
        try:
            kpool_write_plan_multi_decode_cuda(
                write_start=write_start,
                req_pool_indices=req_pool_indices,
                real_page_table=real_page_table,
                req_out0=plan0.req,
                write_start_out0=plan0.write_start,
                tail_logical_start_out0=plan0.tail_logical_start,
                write_loc_out0=plan0.write_loc,
                req_out1=plan1.req,
                write_start_out1=plan1.write_start,
                tail_logical_start_out1=plan1.tail_logical_start,
                write_loc_out1=plan1.write_loc,
                req_out2=plan2.req,
                write_start_out2=plan2.write_start,
                tail_logical_start_out2=plan2.tail_logical_start,
                write_loc_out2=plan2.write_loc,
                req_out3=plan3.req,
                write_start_out3=plan3.write_start,
                tail_logical_start_out3=plan3.tail_logical_start,
                write_loc_out3=plan3.write_loc,
                pool_size=pool_size,
                slots_per_page=slots_per_page,
            )
            return
        except Exception as e:
            global _KPOOL_WRITE_PLAN_JIT_WARNED
            if not _KPOOL_WRITE_PLAN_JIT_WARNED:
                logger.warning(
                    "JIT kpool write-plan multi decode failed; "
                    "falling back to Triton multi update: %s",
                    e,
                )
                _KPOOL_WRITE_PLAN_JIT_WARNED = True

    update_kpool_write_plan_cuda_graph_multi_decode(
        write_start=write_start,
        req_pool_indices=req_pool_indices,
        real_page_table=real_page_table,
        req_out0=plan0.req,
        write_start_out0=plan0.write_start,
        tail_logical_start_out0=plan0.tail_logical_start,
        write_loc_out0=plan0.write_loc,
        req_out1=plan1.req,
        write_start_out1=plan1.write_start,
        tail_logical_start_out1=plan1.tail_logical_start,
        write_loc_out1=plan1.write_loc,
        req_out2=plan2.req,
        write_start_out2=plan2.write_start,
        tail_logical_start_out2=plan2.tail_logical_start,
        write_loc_out2=plan2.write_loc,
        req_out3=plan3.req if plan3 is not None else None,
        write_start_out3=plan3.write_start if plan3 is not None else None,
        tail_logical_start_out3=(
            plan3.tail_logical_start if plan3 is not None else None
        ),
        write_loc_out3=plan3.write_loc if plan3 is not None else None,
        pool_size=pool_size,
        slots_per_page=slots_per_page,
    )


def init_kpool_write_plan(
    metadata: DSAMetadata,
    forward_batch: ForwardBatch,
    *,
    pool_size: int,
    real_page_size: int,
    real_page_table: torch.Tensor,
    num_draft_tokens: int,
    write_start: torch.Tensor,
    accept_length: Optional[torch.Tensor] = None,
) -> DSAMetadata:
    """Build the layer-invariant kpool write plan (eager path).

    Allocates worst-case buffers at ``batch_size`` and fills them via
    the shared GPU plan-build kernel -- same code path as the cuda-graph
    replay update. Returns input unchanged when the gate fails.

    Inputs:
      real_page_table: int32 ``[B, max_pages]`` (decode) or
        ``[B*N, max_pages]`` (verify / v2 draft_extend, repeat-interleave'd).
      write_start: int32 ``[B]`` -- per-batch logical write position
        (decode: positions; verify: committed_seq_lens = seq_lens;
        v2 draft_extend: seq_lens - N = extend_prefix_lens).
      accept_length: int32 ``[B]`` -- v2 only, just-finished verify's
        accept_length (already includes the bonus token from
        eagle_info_v2.sample); lets the write kernel gate compress on
        this real-advance window instead of the full N drafts.
    """
    forward_mode = forward_batch.forward_mode
    # "ring write" modes share the [B] write-plan: decode (N=1), target_verify
    # and draft_extend_v2 (N=num_draft_tokens, fixed per batch). v1 draft_extend
    # has variable q_len and still goes through the extend planner.
    is_verify = forward_mode.is_target_verify()
    is_v2 = forward_mode.is_draft_extend_v2()
    is_ring_write = forward_mode.is_decode_or_idle() or is_verify or is_v2
    if not _is_kpool_layout_enabled(pool_size, real_page_size) or not is_ring_write:
        return metadata

    if is_verify or is_v2:
        pool = get_token_to_kv_pool()
        assert pool.tail_extra_slots == num_draft_tokens, (
            f"tail_extra_slots mismatch: pool={pool.tail_extra_slots}, "
            f"forward={num_draft_tokens}"
        )
    assert real_page_table.dtype == torch.int32, (
        f"real_page_table must be int32, got {real_page_table.dtype}"
    )

    batch_size = forward_batch.seq_lens.shape[0]
    device = forward_batch.seq_lens.device

    metadata = init_kpool_write_plan_capture(
        metadata,
        max_bs=batch_size,
        pool_size=pool_size,
        real_page_size=real_page_size,
        real_page_table=real_page_table,
        num_draft_tokens=num_draft_tokens,
        device=device,
        is_verify=is_verify or is_v2,
        is_v2=is_v2,
    )
    update_kpool_write_plan(
        metadata,
        write_start=write_start,
        req_pool_indices=forward_batch.req_pool_indices,
        real_page_table=real_page_table,
        pool_size=pool_size,
        real_page_size=real_page_size,
        num_draft_tokens=num_draft_tokens,
        forward_mode=forward_mode,
        accept_length=accept_length,
    )
    return metadata
