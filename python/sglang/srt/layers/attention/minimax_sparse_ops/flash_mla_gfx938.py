"""gfx938 FlashMLA MSA128 adapter for MiniMax-M3 sparse attention.

The adapter can replace all three compute stages: index score, exact Top16,
and sparse main attention.  Small page-table/forced-local-page glue operations
remain in SGLang because they encode serving metadata and MiniMax semantics.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.minimax_sparse.common.utils import unit_scale


class FlashMLAGfx938UnavailableError(RuntimeError):
    """The opt-in gfx938 FlashMLA path cannot satisfy its runtime contract."""


@functools.lru_cache(maxsize=1)
def _load_flash_mla():
    try:
        from flash_mla import (
            flash_mla_sparse_fwd,
            flash_mla_with_kvcache,
            get_mla_metadata,
            msa128_expand_block_indexes,
            msa128_stage1_page_max,
            msa128_stage2_topk_select,
        )
    except Exception as err:
        raise FlashMLAGfx938UnavailableError(
            "flash_mla with the MSA128 interface is not importable"
        ) from err
    funcs = (
        flash_mla_sparse_fwd,
        flash_mla_with_kvcache,
        get_mla_metadata,
        msa128_expand_block_indexes,
        msa128_stage1_page_max,
        msa128_stage2_topk_select,
    )
    if not all(callable(fn) for fn in funcs):
        raise FlashMLAGfx938UnavailableError(
            "flash_mla is missing one or more required MSA128 callables"
        )
    return funcs


@functools.lru_cache(maxsize=1)
def flash_mla_gfx938_available() -> bool:
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        arch = str(
            getattr(props, "gcnArchName", "") or getattr(props, "gcn_arch_name", "")
        )
        if "gfx938" not in arch:
            return False
        _load_flash_mla()
        return True
    except Exception:
        return False


def new_flash_mla_decode_metadata():
    """Return one scheduler object to keep alive for one layer/graph shape."""
    _, _, get_mla_metadata, *_ = _load_flash_mla()
    metadata, _ = get_mla_metadata()
    return metadata


@functools.lru_cache(maxsize=64)
def _cpu_i32(values: tuple[int, ...]) -> torch.Tensor:
    """Cache immutable host metadata used by the FlashMLA dispatcher."""
    return torch.tensor(values, dtype=torch.int32)


def _paged_index_k(k_cache: torch.Tensor, page_size: int) -> torch.Tensor:
    """Expose SGLang slot-major index K as FlashMLA Stage-1 paged K."""
    if k_cache.dim() != 3 or k_cache.shape[1:] != (1, 128):
        raise FlashMLAGfx938UnavailableError(
            "FlashMLA MSA128 indexer requires index K [slots, 1, 128], got "
            f"{tuple(k_cache.shape)}"
        )
    if k_cache.shape[0] % page_size:
        raise FlashMLAGfx938UnavailableError(
            f"index K slots={k_cache.shape[0]} is not page{page_size}-aligned"
        )
    # FlashMLA Stage-1 consumes [pages, Hkv, page, D].  Hkv is one for the
    # MiniMax indexer, so this permutation remains a contiguous zero-copy view.
    return k_cache.view(-1, page_size, 1, 128).permute(0, 2, 1, 3)


@triton.jit
def _force_local_pages_kernel(
    score,
    k_end,
    stride_h,
    stride_page,
    stride_q,
    num_pages: tl.constexpr,
    block_size: tl.constexpr,
    local_blocks: tl.constexpr,
):
    query = tl.program_id(0)
    head = tl.program_id(1)
    last_page = (tl.load(k_end + query).to(tl.int64) - 1) // block_size
    for offset in range(local_blocks):
        page = last_page - offset
        if page >= 0 and page < num_pages:
            tl.store(
                score + head * stride_h + page * stride_page + query * stride_q,
                1.0e30 - offset,
            )


def _force_local_pages(
    score: torch.Tensor,
    k_end: torch.Tensor,
    block_size: int,
    local_blocks: int,
) -> None:
    """Apply MiniMax's per-query forced local-page rule before Stage-2."""
    if local_blocks <= 0:
        return
    _force_local_pages_kernel[(score.shape[2], score.shape[0])](
        score,
        k_end,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        num_pages=score.shape[1],
        block_size=block_size,
        local_blocks=local_blocks,
        num_warps=1,
        num_stages=1,
    )


def _run_flash_mla_indexer(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    q_lens: tuple[int, ...],
    k_end: torch.Tensor,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    safe_pages: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run gfx938 FlashMLA MSA128 Stage-1 + Stage-2 exactly."""
    *_, stage1, stage2 = _load_flash_mla()
    if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16:
        raise FlashMLAGfx938UnavailableError(
            "FlashMLA MSA128 indexer requires BF16 index Q/K"
        )
    if q.shape[1:] != (4, 128):
        raise FlashMLAGfx938UnavailableError(
            "FlashMLA MSA128 MiniMax indexer requires index Q [rows, 4, 128], "
            f"got {tuple(q.shape)}"
        )
    if block_size != 128 or topk != 16:
        raise FlashMLAGfx938UnavailableError(
            f"FlashMLA MSA128 indexer requires block=128/topk=16, got "
            f"block={block_size}/topk={topk}"
        )
    if page_table.dim() != 2 or page_table.dtype != torch.int32:
        raise FlashMLAGfx938UnavailableError(
            "FlashMLA MSA128 indexer requires a 2D int32 page table"
        )
    if len(q_lens) != page_table.shape[0] or sum(q_lens) != q.shape[0]:
        raise FlashMLAGfx938UnavailableError(
            "FlashMLA MSA128 indexer query segmentation mismatch: "
            f"q_lens={q_lens}, rows={q.shape[0]}, batch={page_table.shape[0]}"
        )
    if k_end.numel() != q.shape[0]:
        raise FlashMLAGfx938UnavailableError(
            f"FlashMLA MSA128 k_end rows={k_end.numel()} != q rows={q.shape[0]}"
        )

    max_pages = int(page_table.shape[1])
    # A fixed declared KV extent keeps the page-table shape invariant under a
    # decode CUDA graph.  k_end is the live, device-side exclusive bound, so
    # padded page-table entries are never visible to the score computation.
    fixed_k_len = max_pages * block_size
    if safe_pages is None:
        safe_pages = page_table.clamp_min(0).reshape(-1).contiguous()
    elif (
        safe_pages.device != page_table.device
        or safe_pages.dtype != page_table.dtype
        or safe_pages.numel() != page_table.numel()
    ):
        raise FlashMLAGfx938UnavailableError(
            "FlashMLA MSA128 cached safe-page metadata does not match the "
            "current page table"
        )
    score = stage1(
        q,
        _paged_index_k(k_cache, block_size),
        _cpu_i32(q_lens),
        _cpu_i32((fixed_k_len,) * len(q_lens)),
        causal=True,
        kv_indices=safe_pages,
        k_end=k_end.to(torch.int32),
    )
    _force_local_pages(score, k_end, block_size, local_blocks)
    # Stage-2 can force common prefix pages directly.  Local pages are
    # query-dependent and were marked above.
    selected = stage2(
        score,
        topk,
        num_valid_pages=max_pages,
        force_begin_blocks=init_blocks,
        force_end_blocks=0,
    )
    # Preserve the established SGLang contract [index_head, query, TopK].
    return selected.permute(1, 0, 2)


def flash_mla_sparse_prefill_indexer(
    idx_q: torch.Tensor,
    idx_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
    prefix_lens: torch.Tensor,
    q_lens_cpu,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    k_end: Optional[torch.Tensor] = None,
    safe_pages: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    q_lens = tuple(int(x) for x in q_lens_cpu)
    if k_end is None:
        # Compatibility path for callers that do not provide forward-local
        # metadata. The MiniMax backend caches this value across sparse layers.
        lens = torch.tensor(q_lens, device=idx_q.device, dtype=torch.long)
        k_end = build_flash_mla_sparse_prefill_k_end(
            cu_seqlens,
            prefix_lens,
            lens,
            idx_q.shape[0],
            idx_q.device,
        )
    return _run_flash_mla_indexer(
        q=idx_q,
        k_cache=idx_k_cache,
        page_table=page_table,
        q_lens=q_lens,
        k_end=k_end,
        block_size=block_size,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        safe_pages=safe_pages,
    )


def build_flash_mla_sparse_prefill_k_end(
    cu_seqlens: torch.Tensor,
    prefix_lens: torch.Tensor,
    q_lens: torch.Tensor,
    num_queries: int,
    device: torch.device,
) -> torch.Tensor:
    """Build device-side exclusive KV bounds once for one ragged segment.

    ``q_lens`` is already a device tensor in the serving backend. Accepting it
    directly avoids the tiny pageable H2D copy that otherwise serializes the
    host once per sparse layer and CP segment.
    """
    cu = cu_seqlens.to(device=device, dtype=torch.long)
    lens = q_lens.to(device=device, dtype=torch.long)
    if lens.numel() != prefix_lens.numel() or int(num_queries) < 0:
        raise FlashMLAGfx938UnavailableError(
            "FlashMLA MSA128 cached prefill metadata has inconsistent batch "
            f"dimensions: q_lens={lens.numel()}, prefix_lens={prefix_lens.numel()}, "
            f"queries={num_queries}"
        )
    batch_ids = torch.arange(
        lens.numel(), device=device, dtype=torch.long
    ).repeat_interleave(lens)
    if batch_ids.numel() != int(num_queries):
        raise FlashMLAGfx938UnavailableError(
            "FlashMLA MSA128 cached prefill metadata has inconsistent query rows: "
            f"q_lens rows={batch_ids.numel()}, queries={num_queries}"
        )
    within = torch.arange(int(num_queries), device=device) - cu[:-1].repeat_interleave(
        lens
    )
    return (prefix_lens.to(torch.long)[batch_ids] + within + 1).to(torch.int32)


def flash_mla_sparse_decode_indexer(
    idx_q: torch.Tensor,
    idx_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    return _run_flash_mla_indexer(
        q=idx_q,
        k_cache=idx_k_cache,
        page_table=page_table,
        q_lens=(1,) * idx_q.shape[0],
        k_end=seq_lens,
        block_size=block_size,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )


def update_flash_mla_page_table(
    output: torch.Tensor,
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
) -> None:
    """Refresh ``logical page -> physical page`` for a serving forward.

    ``output`` is persistent so CUDA-graph replay observes a stable address.
    SGLang allocates a page as contiguous physical slots, hence looking up the
    first logical token of each page is sufficient.
    """
    if output.dtype != torch.int32 or output.dim() != 2:
        raise ValueError("FlashMLA page-table output must be a 2D int32 tensor")
    batch, max_pages = output.shape
    if slot_ids.numel() != batch or seq_lens.numel() != batch:
        raise ValueError(
            "FlashMLA page-table batch mismatch: "
            f"output={batch}, slots={slot_ids.numel()}, seq_lens={seq_lens.numel()}"
        )
    logical_pages = torch.arange(max_pages, device=output.device, dtype=torch.long)
    logical_first = (logical_pages * page_size).clamp_max(req_to_token.shape[1] - 1)
    rows = slot_ids.to(torch.long).reshape(-1, 1)
    physical_pages = torch.div(
        req_to_token[rows, logical_first.reshape(1, -1)],
        page_size,
        rounding_mode="floor",
    ).to(torch.int32)
    live_pages = torch.div(
        seq_lens.to(torch.long) + page_size - 1,
        page_size,
        rounding_mode="floor",
    ).reshape(-1, 1)
    output.copy_(
        torch.where(
            logical_pages.reshape(1, -1) < live_pages,
            physical_pages,
            physical_pages.new_full((), -1),
        )
    )


def _validate_contract(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    page_table: Optional[torch.Tensor],
    block_size_k: int,
) -> None:
    if not flash_mla_gfx938_available():
        raise FlashMLAGfx938UnavailableError(
            "gfx938 flash_mla MSA128 kernels are unavailable"
        )
    if q.dtype != torch.bfloat16:
        raise FlashMLAGfx938UnavailableError(
            f"gfx938 FlashMLA MSA128 requires BF16 Q, got {q.dtype}"
        )
    if k_cache.dtype != torch.bfloat16 or v_cache.dtype != torch.bfloat16:
        raise FlashMLAGfx938UnavailableError(
            "gfx938 FlashMLA MSA128 currently requires BF16 K/V cache, got "
            f"K={k_cache.dtype}, V={v_cache.dtype}"
        )
    if q.shape[-2:] != (64, 128):
        raise FlashMLAGfx938UnavailableError(
            "gfx938 FlashMLA MSA128 requires attention TP1 with local Q shape "
            f"[..., 64, 128], got {tuple(q.shape)}"
        )
    if k_cache.shape[-2:] != (4, 128) or v_cache.shape[-2:] != (4, 128):
        raise FlashMLAGfx938UnavailableError(
            "gfx938 FlashMLA MSA128 requires local K/V shape "
            f"[slots, 4, 128], got K={tuple(k_cache.shape)}, V={tuple(v_cache.shape)}"
        )
    if block_size_k != 128 or topk_idx.shape[-1] != 16:
        raise FlashMLAGfx938UnavailableError(
            "gfx938 FlashMLA MSA128 requires block_size=128 and TopK=16, got "
            f"block_size={block_size_k}, topk={topk_idx.shape[-1]}"
        )
    if page_table is None or page_table.dtype != torch.int32:
        raise FlashMLAGfx938UnavailableError(
            "gfx938 FlashMLA MSA128 requires a prepared int32 page table"
        )


def flash_mla_sparse_prefill_main(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    page_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_size_k: int,
    indices_output: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
) -> torch.Tensor:
    """Run packed prefill through FlashMLA after the native MiniMax indexer."""
    flash_mla_sparse_fwd, _, _, expand, *_ = _load_flash_mla()
    _validate_contract(q, k_cache, v_cache, topk_idx, page_table, block_size_k)
    batch = int(cu_seqlens.numel()) - 1
    if page_table.shape[0] != batch:
        raise FlashMLAGfx938UnavailableError(
            "FlashMLA prefill page-table/query batch mismatch: "
            f"page_table={tuple(page_table.shape)}, packed_batch={batch}"
        )
    if (
        seq_lens.numel() != batch
        or prefix_lens.numel() != batch
        or int(cu_seqlens.numel()) != batch + 1
    ):
        raise FlashMLAGfx938UnavailableError(
            "FlashMLA prefill ragged metadata mismatch: "
            f"batch={batch}, seq_lens={seq_lens.numel()}, "
            f"prefix_lens={prefix_lens.numel()}, cu={cu_seqlens.numel()}"
        )
    if topk_idx.shape[0] != 4 or topk_idx.shape[1] != q.shape[0]:
        raise FlashMLAGfx938UnavailableError(
            "FlashMLA prefill needs one exact Top16 row per query and KV head, "
            f"got topk={tuple(topk_idx.shape)}, q={tuple(q.shape)}"
        )
    blocks = topk_idx.permute(1, 0, 2).contiguous().to(torch.int32)
    indices = expand(
        blocks,
        causal=True,
        qo_offset=prefix_lens.to(torch.int32),
        seqused_k=seq_lens.to(torch.int32),
        page_table=page_table,
        cu_seqlens_q=cu_seqlens.to(torch.int32),
        output=indices_output,
        s_q_axis=0,
    )
    scale = (
        (128**-0.5 if sm_scale is None else sm_scale)
        * unit_scale(q_scale)
        * unit_scale(k_scale)
    )
    output, _, _ = flash_mla_sparse_fwd(
        q,
        k_cache,
        indices,
        sm_scale=scale,
        d_v=128,
        v=v_cache,
    )
    value_scale = unit_scale(v_scale)
    return output if value_scale == 1.0 else output * value_scale


def flash_mla_sparse_decode_main(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size_k: int,
    sched_meta,
    indices_output: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    verify_group_size: int = 1,
) -> torch.Tensor:
    """Run decode or target verify through the native multi-Q interface.

    Target-verify tensors arrive flattened in request-major order. Keeping the
    speculative positions in ``s_q`` lets FlashMLA schedule them as one request
    instead of treating every position as an unrelated decode row.
    """
    _, flash_mla_with_kvcache, _, expand, *_ = _load_flash_mla()
    _validate_contract(q, k_cache, v_cache, topk_idx, page_table, block_size_k)
    rows = q.shape[0]
    verify_group_size = int(verify_group_size)
    if verify_group_size <= 0 or rows % verify_group_size:
        raise FlashMLAGfx938UnavailableError(
            "FlashMLA decode rows must be divisible by verify_group_size, got "
            f"rows={rows}, verify_group_size={verify_group_size}"
        )
    batch = rows // verify_group_size
    if topk_idx.shape[:2] != (4, rows) or seq_lens.numel() != rows:
        raise FlashMLAGfx938UnavailableError(
            "FlashMLA decode shape mismatch: expected topk=[4, rows, 16] and "
            f"seq_lens=[rows], got topk={tuple(topk_idx.shape)}, "
            f"seq_lens={tuple(seq_lens.shape)}, rows={rows}"
        )
    if page_table.shape[0] != batch:
        raise FlashMLAGfx938UnavailableError(
            "FlashMLA grouped decode needs one page-table row per request, got "
            f"page_table={tuple(page_table.shape)}, batch={batch}, "
            f"verify_group_size={verify_group_size}"
        )
    q_grouped = q.reshape(batch, verify_group_size, 64, 128)
    blocks = (
        topk_idx.permute(1, 0, 2)
        .reshape(batch, verify_group_size, 4, 16)
        .contiguous()
        .to(torch.int32)
    )
    grouped_seq_lens = seq_lens.reshape(batch, verify_group_size).to(torch.int32)
    indices = expand(
        blocks,
        causal=True,
        seqused_k=grouped_seq_lens,
        page_table=page_table,
        output=indices_output,
        s_q_axis=1,
    )
    num_pages = k_cache.shape[0] // block_size_k
    k_paged = k_cache.view(num_pages, block_size_k, 4, 128)
    v_paged = v_cache.view(num_pages, block_size_k, 4, 128)
    scale = (
        (128**-0.5 if sm_scale is None else sm_scale)
        * unit_scale(q_scale)
        * unit_scale(k_scale)
    )
    output, _ = flash_mla_with_kvcache(
        q_grouped,
        k_paged,
        None,
        None,
        128,
        sched_meta,
        softmax_scale=scale,
        causal=False,
        is_fp8_kvcache=False,
        indices=indices,
        v_cache=v_paged,
    )
    output = output.reshape(rows, 64, 128)
    value_scale = unit_scale(v_scale)
    return output if value_scale == 1.0 else output * value_scale
