"""Consume published Prefill candidates from the paged index-K pool on HCU.

The source layer and decode keep their existing paths.  This adapter only turns
an existing candidate mask/table into sorted logical blocks, scores those blocks
with FP32 head weights and logits, and restores the backend's sorted selections.
"""

from typing import Callable, Optional

import torch

from sglang.srt.environ import envs

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # CPU contract tests need neither Triton nor a device.
    triton = tl = libdevice = None


SPARSE_BLOCK_KV = 8
INVALID_BLOCK = torch.iinfo(torch.int32).max
# Same floor as torch_quant.fake_quant_fp4, before the divide by six.
FP4_AMAX_FLOOR = 6 * 2.0**-126


def prefill_paged_sparse_enabled() -> bool:
    """Enable the HCU Prefill paged sparse consumer by default."""
    return envs.SGLANG_HCU_PREFILL_PAGED_SPARSE.get()


if triton is not None:

    @triton.jit
    def _pack_query_kernel(
        Q,
        FREQS,
        POS,
        OUT,
        SF,
        HEADS: tl.constexpr,
        FREQ_STRIDE: tl.constexpr,
        ROPE_DIM: tl.constexpr,
        AMAX_FLOOR: tl.constexpr,
    ):
        row = tl.program_id(0)
        token = row // HEADS
        position = tl.load(POS + token).to(tl.int64)
        offsets = tl.arange(0, 128)
        values = tl.load(Q + row * 128 + offsets).to(tl.float32)
        tail = offsets >= 128 - ROPE_DIM
        pair = (offsets - (128 - ROPE_DIM)) // 2
        real = tl.load(Q + row * 128 + 128 - ROPE_DIM + 2 * pair, tail, 0).to(
            tl.float32
        )
        imag = tl.load(Q + row * 128 + 128 - ROPE_DIM + 2 * pair + 1, tail, 0).to(
            tl.float32
        )
        cosine = tl.load(FREQS + position * FREQ_STRIDE + 2 * pair, tail, 1)
        sine = tl.load(FREQS + position * FREQ_STRIDE + 2 * pair + 1, tail, 0)
        rotated = tl.where(
            (offsets - (128 - ROPE_DIM)) % 2 == 0,
            real * cosine - imag * sine,
            real * sine + imag * cosine,
        )
        # queries() rounds the rotated tail to BF16 before fake quantization.
        values = tl.where(tail, rotated.to(tl.bfloat16).to(tl.float32), values)
        groups = tl.reshape(values, (4, 32))
        amax = tl.maximum(tl.max(tl.abs(groups), axis=1), AMAX_FLOOR)
        bits = (amax * (1.0 / 6.0)).to(tl.int32, bitcast=True)
        exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        scale = (exponent << 23).to(tl.float32, bitcast=True)
        scaled = tl.minimum(tl.maximum(groups / scale[:, None], -6.0), 6.0)
        magnitude = tl.abs(scaled)
        step = tl.where(magnitude < 2.0, 0.5, tl.where(magnitude < 4.0, 1.0, 2.0))
        sign = tl.where(scaled > 0, 1.0, tl.where(scaled < 0, -1.0, 0.0))
        rounded = tl.reshape(libdevice.rint(magnitude / step) * step * sign, (128,))
        # Keep the original fake-quant scale.  Do not requantize a BF16 buffer
        # using the standalone index-K packer's different 1e-4 scale floor.
        magnitude = tl.abs(rounded)
        code = (magnitude > 0.25).to(tl.uint8)
        code += (magnitude > 0.75).to(tl.uint8)
        code += (magnitude > 1.25).to(tl.uint8)
        code += (magnitude > 1.75).to(tl.uint8)
        code += (magnitude > 2.5).to(tl.uint8)
        code += (magnitude > 3.5).to(tl.uint8)
        code += (magnitude > 5.0).to(tl.uint8)
        code |= ((rounded < 0) & (code != 0)).to(tl.uint8) << 3
        low, high = tl.split(tl.reshape(code, (64, 2)))
        tl.store(OUT + row * 64 + tl.arange(0, 64), low | (high << 4))
        packed_scale = tl.sum(exponent.to(tl.uint32) << (tl.arange(0, 4) * 8), 0)
        tl.store(SF + row, packed_scale.to(tl.int32))


def pack_prefill_query(
    q_before_rope: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
):
    """Raw BF16 ``[R,H,128]`` Q -> packed ``[R,1,H,64]`` and int32 scales.

    GPU uses one RoPE/fake-quant/pack launch with no intermediate BF16 Q buffer.
    CPU implements the same contract for mocks and independent shape tests.
    ``freqs_cis`` is the full complex table, indexed by ``positions``.
    """
    rows, heads, dim = q_before_rope.shape
    if dim != 128 or q_before_rope.dtype != torch.bfloat16:
        raise ValueError("Prefill query pack requires BF16 Q with head_dim=128")
    if not 0 <= rope_dim <= 128 or rope_dim % 2:
        raise ValueError("rope_dim must be even and between 0 and 128")
    if positions.numel() != rows or not freqs_cis.is_complex():
        raise ValueError(
            "one position per Q row and a complex frequency table required"
        )
    q = q_before_rope.contiguous()
    payload = torch.empty((rows, 1, heads, 64), device=q.device, dtype=torch.int8)
    sf = torch.empty((rows, 1, heads), device=q.device, dtype=torch.int32)
    if rows == 0:
        return payload, sf
    if q.is_cuda:
        if triton is None:
            raise RuntimeError("Triton is required for HCU Prefill query packing")
        freqs = torch.view_as_real(freqs_cis.contiguous()).flatten(-2)
        _pack_query_kernel[(rows * heads,)](
            q,
            freqs,
            positions.contiguous(),
            payload,
            sf,
            HEADS=heads,
            FREQ_STRIDE=freqs.stride(0),
            ROPE_DIM=rope_dim,
            AMAX_FLOOR=FP4_AMAX_FLOOR,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return payload, sf
    values = q.float().clone()
    if rope_dim:
        tail = torch.view_as_complex(
            values[..., -rope_dim:].reshape(rows, heads, -1, 2)
        )
        rotated = torch.view_as_real(
            tail * freqs_cis[positions.long(), None, :]
        ).flatten(-2)
        values[..., -rope_dim:] = rotated.to(torch.bfloat16).float()
    groups = values.reshape(rows, heads, 4, 32)
    bits = (
        (groups.abs().amax(-1).clamp_min(FP4_AMAX_FLOOR) * (1.0 / 6.0))
        .contiguous()
        .view(torch.int32)
    )
    exponents = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    scale = (exponents << 23).view(torch.float32)
    scaled = (groups / scale[..., None]).clamp(-6.0, 6.0)
    magnitude = scaled.abs()
    step = torch.where(magnitude < 2.0, 0.5, torch.where(magnitude < 4.0, 1.0, 2.0))
    rounded = (torch.round(magnitude / step) * step * scaled.sign()).reshape(
        rows, heads, 128
    )
    magnitude = rounded.abs()
    code = torch.zeros_like(rounded, dtype=torch.uint8)
    for boundary in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        code += (magnitude > boundary).to(torch.uint8)
    code |= ((rounded < 0) & (code != 0)).to(torch.uint8) << 3
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    payload.copy_(packed.view(torch.int8).unsqueeze(1))
    shifts = torch.arange(4, device=q.device, dtype=torch.int64) * 8
    sf.copy_((exponents.to(torch.int64) << shifts).sum(-1).to(torch.int32).unsqueeze(1))
    return payload, sf


def candidate_blocks_from_mask(
    mask: torch.Tensor, topk_blocks: int, *, budget_bytes: int = 1 << 30
) -> torch.Tensor:
    """Recover the source's logical 8-position blocks; never select new blocks.

    Fixed width, ascending int32 IDs, INT32_MAX padding.  A causally truncated
    last block is retained if any of its valid mask positions is selected.
    """
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("candidate mask must be a bool matrix")
    if topk_blocks <= 0 or topk_blocks % 4 or topk_blocks > 4096:
        raise ValueError(
            "candidate table width must be a positive multiple of four <=4096"
        )
    rows, width = mask.shape
    nblocks = (width + 7) // 8
    blocks = torch.full(
        (rows, topk_blocks), INVALID_BLOCK, device=mask.device, dtype=torch.int32
    )
    if rows == 0 or nblocks == 0:
        return blocks
    step = max(1, budget_bytes // max(1, width + nblocks * 12 + topk_blocks * 4))
    ids = torch.arange(nblocks, device=mask.device, dtype=torch.int32)
    for begin in range(0, rows, step):
        chunk = mask[begin : begin + step]
        if width % 8:
            chunk = torch.nn.functional.pad(chunk, (0, -width % 8), value=False)
        kept = chunk.reshape(chunk.shape[0], nblocks, 8).any(-1)
        ranked = torch.where(kept, ids[None, :], INVALID_BLOCK)
        count = min(topk_blocks, nblocks)
        blocks[begin : begin + chunk.shape[0], :count] = ranked.topk(
            count, dim=-1, largest=False, sorted=True
        ).values
    return blocks


def compact_valid_lens(context_lens: torch.Tensor, topk_blocks: int):
    """Valid compact prefix when sorted candidates include the newest block."""
    lengths = context_lens.reshape(-1).to(torch.int32)
    kept = torch.minimum((lengths + 7) // 8, lengths.new_full((), topk_blocks))
    return torch.where(
        lengths > 0, 8 * (kept - 1) + (lengths - 1) % 8 + 1, 0
    ).contiguous()


def request_page_table(
    req_to_token: torch.Tensor,
    request_pool_index: torch.Tensor,
    context_len: int,
    compress_ratio: int,
    page_size: int,
):
    """Logical index-K pages mapped through the real pool's compressed slots."""
    columns = torch.arange(
        (context_len + page_size - 1) // page_size,
        device=req_to_token.device,
        dtype=torch.int64,
    )
    full_positions = columns * page_size * compress_ratio
    # A zero-dimensional device index is converted to a Python scalar by
    # Tensor.__getitem__, causing a synchronous device-to-host transfer.
    # A singleton vector uses device-side advanced indexing and broadcasts
    # over the page columns without reading the request ID on the CPU.
    request_index = request_pool_index.reshape(1).to(torch.int64)
    slots = (
        req_to_token[request_index, full_positions].to(torch.int64) // compress_ratio
    )
    return (slots // page_size).to(torch.int32).contiguous()


def _lightop_fused_consumer(x):
    """GPU implementation lives in LightOp; keep the original adapter as fallback."""
    if not envs.SGLANG_HCU_PREFILL_SPARSE_FUSION.get() or not x.is_cuda:
        return None
    try:
        from lightop.attention.mqa import (
            prefill_sparse_mqa_fused,
            supports_prefill_sparse_fusion,
        )
    except (ImportError, OSError):
        return None
    return prefill_sparse_mqa_fused if supports_prefill_sparse_fusion(x) else None


def _lightop_functions():
    try:
        from lightop import topk_transform_ragged_v2
        from lightop.attention import (
            sparse_fp8_fp4_paged_mqa_logits,
            get_paged_sparse_mqa_logits_metadata,
        )
    except (ImportError, OSError):
        return None
    if not all(
        callable(fn)
        for fn in (
            sparse_fp8_fp4_paged_mqa_logits,
            get_paged_sparse_mqa_logits_metadata,
            topk_transform_ragged_v2,
        )
    ):
        return None
    return (
        get_paged_sparse_mqa_logits_metadata,
        sparse_fp8_fp4_paged_mqa_logits,
        topk_transform_ragged_v2,
    )


def select_prefill_sparse_rows(
    *,
    q_fp4: torch.Tensor,
    q_sf: torch.Tensor,
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    blocks: torch.Tensor,
    paged_table: torch.Tensor,
    context_lens: torch.Tensor,
    index_topk: int,
    budget_bytes: int = 1 << 30,
    num_sms: Optional[int] = None,
    schedule_fn: Optional[Callable] = None,
    logits_fn: Optional[Callable] = None,
    topk_fn: Optional[Callable] = None,
    schedule_cache: Optional[dict] = None,
):
    """Score rows of ONE request, returning sorted logical positions/-1 padding.

    Hooks allow CPU mocks; defaults call the production metadata, FP32 scorer and
    FP32 fused TopK, which clips causal tails before ranking. The scorer
    returns whole-block logits; invalid tail columns are not a masked output
    contract of this adapter. Every memory-budget row chunk has its own metadata.  A
    metadata byte buffer is never sliced to obtain a smaller Q row set.
    """
    rows, _, heads, payload_bytes = q_fp4.shape
    width = blocks.shape[1]
    if q_fp4.shape[1] != 1 or payload_bytes != 64 or not 0 < heads <= 32 or heads % 4:
        raise ValueError(
            "packed Prefill Q requires next_n=1, head_dim=128 and 4..32 heads in multiples of four"
        )
    if blocks.shape[0] != rows or width <= 0 or width % 4 or width > 4096:
        raise ValueError("invalid sparse candidate table shape")
    if weights.shape != (rows, heads) or weights.dtype != torch.float32:
        raise ValueError("Prefill head weights must be [rows,heads] FP32")
    if context_lens.numel() != rows or paged_table.shape[0] != rows:
        raise ValueError("lengths and per-row page tables must match Q rows")
    if index_topk <= 0 or budget_bytes <= 0:
        raise ValueError("index_topk and budget_bytes must be positive")
    selected = torch.full(
        (rows, index_topk), -1, device=q_fp4.device, dtype=torch.int32
    )
    if rows == 0:
        return selected
    if schedule_fn is None or logits_fn is None or topk_fn is None:
        functions = _lightop_functions()
        if functions is None:
            raise RuntimeError(
                "LightOp paged sparse Prefill and FP32 ragged TopK are required"
            )
        schedule_fn = schedule_fn or functions[0]
        logits_fn = logits_fn or functions[1]
        topk_fn = topk_fn or functions[2]
    if num_sms is None:
        num_sms = (
            torch.cuda.get_device_properties(q_fp4.device).multi_processor_count
            if q_fp4.is_cuda
            else 1
        )
    page_size = k_cache.shape[1]
    # Conservative per-row resident/scratch accounting, including metadata and
    # the contiguous per-Q page table; the source mask stays owned by its source.
    splits = (width + 79) // 80
    row_bytes = (
        width * (8 * 4 + 4) + paged_table.shape[1] * 4 + splits * (16 + 80 * 8 + 16)
    )
    row_bytes += heads * (64 + 4 + 4) + index_topk * 16
    step = max(1, budget_bytes // max(1, row_bytes))
    for begin in range(0, rows, step):
        end = min(rows, begin + step)
        lengths = context_lens[begin:end].reshape(-1).to(torch.int32).contiguous()
        block_chunk = blocks[begin:end].contiguous()
        page_chunk = paged_table[begin:end].contiguous()
        # One ID per pair prevents the scheduler's backward request scan from
        # becoming quadratic.  Caller groups requests; no pair crosses a request.
        pair_ids = (
            torch.arange(end - begin, device=q_fp4.device, dtype=torch.int32) // 2
        )
        cache_key = (begin, end, width, page_size, num_sms)
        metadata = None if schedule_cache is None else schedule_cache.get(cache_key)
        if metadata is None:
            metadata = schedule_fn(
                lengths,
                page_chunk,
                pair_ids,
                page_size,
                block_chunk,
                q_fp4.dtype,
                SPARSE_BLOCK_KV,
                num_sms=num_sms,
            )
            if schedule_cache is not None:
                schedule_cache[cache_key] = metadata
        packed_q = q_fp4[begin:end].contiguous()
        packed_sf = q_sf[begin:end].contiguous()
        head_weights = weights[begin:end].contiguous()
        # Score whole candidate blocks. The fused TopK below uses valid_lens
        # to exclude the partial causal tail before ranking; the source keeps
        # the newest block and blocks are sorted, so valid tokens are a prefix.
        logits = logits_fn(
            (packed_q, packed_sf),
            k_cache,
            head_weights,
            metadata,
            width,
            SPARSE_BLOCK_KV,
            num_sms=num_sms,
            is_prefill=True,
        )
        if logits.dtype != torch.float32:
            raise RuntimeError("paged sparse Prefill logits must be FP32")
        valid_lens = compact_valid_lens(lengths, width)
        compact = torch.empty(
            (end - begin, index_topk), device=q_fp4.device, dtype=torch.int32
        )
        topk_fn(
            logits,
            valid_lens,
            out_offsets=torch.zeros_like(lengths),
            out_indices=compact,
        )
        valid = (compact >= 0) & (compact < valid_lens[:, None])
        slots = (compact.clamp_min(0) // 8).clamp_max(width - 1)
        logical = block_chunk.gather(1, slots) * 8 + compact.clamp_min(0) % 8
        valid &= (logical >= 0) & (logical < lengths[:, None])
        logical = torch.where(valid, logical, INVALID_BLOCK).sort(-1).values
        selected[begin:end] = torch.where(logical != INVALID_BLOCK, logical, -1)
        # Release score/selection scratch before the next chunk's allocation.
        del logits, compact, page_chunk
    return selected


def consume_prefill_candidate_sparse(
    *,
    pool,
    indexer,
    layer_id: int,
    x: torch.Tensor,
    q_lora: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    req_to_token: torch.Tensor,
    request_pool_indices: torch.Tensor,
    lc_per_request: list,
    rows_per_request: list,
    candidate_metadata,
    compress_ratio: int,
    page_indices: torch.Tensor,
    raw_indices: Optional[torch.Tensor],
    budget_bytes: int = 1 << 30,
) -> bool:
    """True when this consumer was handled, False for the existing fallback.

    There is no context-length dispatch threshold.  Capability/format checks
    precede Q preparation and any full-K gather.  Published masks are unchanged.
    """
    if not indexer.uses_candidates or indexer.is_candidate_source:
        return False
    if indexer.candidate_block_size != 8 or indexer.index_head_dim != 128:
        return False
    if not 0 < indexer.n_local_heads <= 32 or indexer.n_local_heads % 4:
        return False
    width = indexer.candidate_topk_blocks
    if width <= 0 or width % 4 or width > 4096:
        return False
    masks = getattr(candidate_metadata, "request_masks", None)
    if masks is None or len(masks) != len(rows_per_request):
        return False
    functions = _lightop_functions()
    if functions is None:
        return False
    if (
        len(lc_per_request) != len(rows_per_request)
        or sum(rows_per_request) != positions.numel()
    ):
        raise ValueError("ragged request rows do not match Prefill Q")
    page_size = pool.get_index_k_page_size(compress_ratio)
    if page_size <= 0 or page_size % 8:
        return False
    k_cache = pool.get_index_k_with_scale_buffer(layer_id)
    if k_cache is None:
        return False
    k_cache = k_cache.view(k_cache.shape[0], page_size, 1, 68)
    fused_consumer = _lightop_fused_consumer(x)
    retained = getattr(candidate_metadata, "request_block_ids", None)
    if (
        fused_consumer is not None
        and retained is not None
        and callable(getattr(indexer, "head_weights_raw", None))
        and indexer.index_topk in (512, 1024, 2048)
    ):
        # Fused writeback covers every live output cell. Preserve any padding
        # owned by the backend without clearing the live matrix first.
        for output in (page_indices, raw_indices):
            if output is None:
                continue
            if output.shape[0] > positions.numel():
                output[positions.numel():].fill_(-1)
            if output.shape[1] > indexer.index_topk:
                output[:positions.numel(), indexer.index_topk:].fill_(-1)
        # Keep both GEMMs and the exact BF16 weight-rounding contract.
        raw_weights = indexer.head_weights_raw(x)
        q, _ = indexer.wq_b(q_lora)
        q = q.view(positions.numel(), indexer.n_local_heads, indexer.index_head_dim)
        fused_consumer(
            q, raw_weights, freqs_cis, positions,
            weight_scale=indexer.head_weight_scale,
            rope_dim=indexer.rope_head_dim,
            k_cache=k_cache,
            req_to_token=req_to_token,
            request_pool_indices=request_pool_indices,
            context_lens_per_request=lc_per_request,
            rows_per_request=rows_per_request,
            request_block_ids=retained,
            compress_ratio=compress_ratio,
            page_size=page_size,
            num_max_sparse_blocks=width,
            index_topk=indexer.index_topk,
            out_page_indices=page_indices,
            out_raw_indices=raw_indices,
            budget_bytes=budget_bytes,
        )
        return True
    # The original adapter needs -1 for skipped requests and output padding.
    page_indices.fill_(-1)
    if raw_indices is not None:
        raw_indices.fill_(-1)
    cache = getattr(candidate_metadata, "_lightop_prefill_sparse_cache", None)
    if cache is None:
        cache = {}
        candidate_metadata._lightop_prefill_sparse_cache = cache
    key = (
        width,
        tuple(lc_per_request),
        tuple(rows_per_request),
        compress_ratio,
        page_size,
        req_to_token.data_ptr(),
        request_pool_indices.data_ptr(),
    )
    prepared = cache.get(key)
    if prepared is None:
        retained = getattr(candidate_metadata, "request_block_ids", None)
        prepared = []
        for request, (length, count) in enumerate(
            zip(lc_per_request, rows_per_request)
        ):
            if length == 0 or count == 0:
                prepared.append(None)
                continue
            if retained is not None:
                source = retained[request]
                if (
                    source.shape[0] != count
                    or source.ndim != 2
                    or source.dtype != torch.int32
                ):
                    raise ValueError(
                        "published candidate block IDs do not match Prefill rows"
                    )
                blocks = torch.full(
                    (count, width),
                    INVALID_BLOCK,
                    device=positions.device,
                    dtype=torch.int32,
                )
                n = min(source.shape[1], width)
                blocks[:, :n] = torch.where(
                    source[:, :n] >= 0, source[:, :n], INVALID_BLOCK
                )
                blocks = blocks.sort(-1).values.contiguous()
            else:
                if masks[request].shape != (count, length):
                    raise ValueError(
                        "published candidate mask does not match Prefill request"
                    )
                blocks = candidate_blocks_from_mask(
                    masks[request], width, budget_bytes=budget_bytes
                )
            pages = request_page_table(
                req_to_token,
                request_pool_indices[request : request + 1],
                length,
                compress_ratio,
                page_size,
            )
            prepared.append((blocks, pages, {}))
        cache[key] = prepared
    # Preserve the current weight calculation and BF16 multiply-before-float
    # order; do not substitute raw projection weights scaled in FP32.
    weights = indexer.head_weights(x).float().contiguous()
    q, _ = indexer.wq_b(q_lora)
    q = q.view(positions.numel(), indexer.n_local_heads, indexer.index_head_dim)
    q_fp4, q_sf = pack_prefill_query(q, freqs_cis, positions, indexer.rope_head_dim)
    lengths = ((positions + 1) // compress_ratio).to(torch.int32)
    offset = 0
    for request, (count, operands) in enumerate(zip(rows_per_request, prepared)):
        end = offset + count
        if operands is not None:
            blocks, pages, schedules = operands
            selected = select_prefill_sparse_rows(
                q_fp4=q_fp4[offset:end],
                q_sf=q_sf[offset:end],
                k_cache=k_cache,
                weights=weights[offset:end],
                blocks=blocks,
                paged_table=pages[None, :].expand(count, -1),
                context_lens=lengths[offset:end],
                index_topk=indexer.index_topk,
                budget_bytes=budget_bytes,
                schedule_fn=functions[0],
                logits_fn=functions[1],
                topk_fn=functions[2],
                schedule_cache=schedules,
            )
            chosen = selected >= 0
            # Gather only selected slots, never the request's full index-K.
            slots = req_to_token[
                request_pool_indices[request : request + 1].to(torch.int64),
                selected.clamp_min(0).to(torch.int64) * compress_ratio,
            ]
            slots = slots.to(torch.int64) // compress_ratio
            page_indices[offset:end, : indexer.index_topk] = torch.where(
                chosen, slots, -1
            ).to(torch.int32)
            if raw_indices is not None:
                raw_indices[offset:end, : indexer.index_topk] = selected
        offset = end
    return True
