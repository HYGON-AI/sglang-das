import json
import os
import threading
import time
from typing import Optional

import torch
import triton
import triton.language as tl

INDEX_HEAD_DIM = 128
KPOOL_SCORE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_KPOOL_CALL_LOG_LOCK = threading.Lock()
_KPOOL_CALL_LOG_COUNTS: dict[str, int] = {}


def _kpool_call_log_value(value):
    if isinstance(value, torch.Tensor):
        return {
            "type": "torch.Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "stride": list(value.stride()),
            "is_contiguous": value.is_contiguous(),
            "requires_grad": value.requires_grad,
        }

    value_type = f"{type(value).__module__}.{type(value).__qualname__}"
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"type": value_type, "value": value}

    result = {"type": value_type, "value": repr(value)}
    for attr in ("index_kpool", "slots_per_page"):
        if hasattr(value, attr):
            attr_value = getattr(value, attr)
            if isinstance(attr_value, (bool, int, float, str)):
                result[attr] = attr_value
            else:
                result[attr] = repr(attr_value)
    return result


def _log_kpool_wrapper_call(function_name: str, **kwargs) -> None:
    log_path = os.environ.get("SGLANG_KPOOL_CALL_LOG")
    if not log_path:
        return

    try:
        max_calls = int(os.environ.get("SGLANG_KPOOL_CALL_LOG_MAX", "512"))
    except ValueError:
        max_calls = 512

    with _KPOOL_CALL_LOG_LOCK:
        call_index = _KPOOL_CALL_LOG_COUNTS.get(function_name, 0)
        if call_index > max_calls:
            return
        _KPOOL_CALL_LOG_COUNTS[function_name] = call_index + 1

        if call_index == max_calls:
            record = {
                "event": "limit_reached",
                "function": function_name,
                "pid": os.getpid(),
                "max_calls": max_calls,
            }
        else:
            try:
                is_capturing = bool(torch.cuda.is_current_stream_capturing())
            except Exception as exc:
                is_capturing = f"unavailable:{type(exc).__name__}"
            record = {
                "event": "call",
                "timestamp_ns": time.time_ns(),
                "function": function_name,
                "call_index": call_index,
                "pid": os.getpid(),
                "rank": os.environ.get("RANK"),
                "local_rank": os.environ.get("LOCAL_RANK"),
                "is_cuda_graph_capturing": is_capturing,
                "args": {
                    name: _kpool_call_log_value(value) for name, value in kwargs.items()
                },
            }

        line = (
            json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n"
        ).encode()
        fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)


def kpool_bf16_paged_mqa_logits(
    q: torch.Tensor,
    buf: torch.Tensor,
    weights: torch.Tensor,
    pool_seqlens: torch.Tensor,
    page_table: torch.Tensor,
    max_seq_len: int,
    slots_per_page: int,
) -> torch.Tensor:
    """Run kpool decode MQA logits directly from the packed FP8 page cache.

    KPool stores each physical page as ``[all K bytes][all FP32 scales]``.
    The generic LightOp paged kernel requires 64 K slots per page, while a
    kpool page has ``page_size / index_kpool`` slots. This Triton fallback
    keeps the native layout and dequantizes each FP8 K slot to BF16 in
    registers. The caller has already applied the matching normalized
    Hadamard rotation to Q before entering this kernel.
    """

    assert q.ndim == 4 and q.shape[1] == 1
    assert q.shape[-1] == INDEX_HEAD_DIM
    assert buf.dtype == torch.uint8 and buf.ndim == 2 and buf.is_contiguous()
    assert slots_per_page > 0
    assert buf.shape[1] == slots_per_page * (INDEX_HEAD_DIM + 4)
    assert weights.shape == (q.shape[0], q.shape[2])
    assert pool_seqlens.shape == (q.shape[0],)
    assert page_table.shape[0] == q.shape[0]
    assert page_table.shape[1] * slots_per_page == max_seq_len
    assert page_table.is_contiguous()

    q = q.squeeze(1).to(torch.bfloat16).contiguous()
    weights = weights.to(torch.float32).contiguous()
    pool_seqlens = pool_seqlens.to(torch.int32).contiguous()
    page_table = page_table.to(torch.int32).contiguous()
    logits = torch.empty(
        (q.shape[0], max_seq_len), dtype=torch.float32, device=q.device
    )
    if max_seq_len == 0 or q.shape[0] == 0:
        return logits

    _kpool_bf16_paged_mqa_logits_kernel[(q.shape[0], triton.cdiv(max_seq_len, 4))](
        q,
        buf,
        buf.view(torch.float32),
        weights,
        pool_seqlens,
        page_table,
        logits,
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        weights.stride(1),
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        max_seq_len,
        NUM_HEADS=q.shape[1],
        SLOTS_PER_PAGE=slots_per_page,
        BUF_NUMEL_PER_PAGE=buf.shape[1],
        HEAD_DIM=INDEX_HEAD_DIM,
        BLOCK_K=4,
        BLOCK_D=triton.next_power_of_2(INDEX_HEAD_DIM),
        num_warps=4,
    )
    return logits


def bf16_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    seqlens: torch.Tensor,
    page_table: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """Triton fallback for the dense BF16 DSA index cache on HCU."""
    assert q.ndim == 4 and q.shape[1] == 1
    assert kv_cache.ndim == 4 and kv_cache.shape[2] == 1
    assert q.shape[-1] == kv_cache.shape[-1]
    assert weights.shape == (q.shape[0], q.shape[2])

    q = q.squeeze(1).to(torch.bfloat16).contiguous()
    kv_cache = kv_cache.contiguous()
    weights = weights.to(torch.float32).contiguous()
    seqlens = seqlens.reshape(-1).to(torch.int32).contiguous()
    page_table = page_table.to(torch.int32).contiguous()
    logits = torch.empty(
        (q.shape[0], max_seq_len), dtype=torch.float32, device=q.device
    )
    if max_seq_len == 0 or q.shape[0] == 0:
        return logits

    page_size = kv_cache.shape[1]
    head_dim = kv_cache.shape[-1]
    _bf16_paged_mqa_logits_kernel[(q.shape[0], triton.cdiv(max_seq_len, 4))](
        q,
        kv_cache,
        weights,
        seqlens,
        page_table,
        logits,
        q.stride(0),
        q.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(3),
        weights.stride(0),
        weights.stride(1),
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        max_seq_len,
        NUM_HEADS=q.shape[1],
        PAGE_SIZE=page_size,
        HEAD_DIM=head_dim,
        BLOCK_K=4,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return logits


def kpool_dequantize_fp8_paged_kv_cache(kv_cache_fp8: torch.Tensor) -> torch.Tensor:
    """Dequantize a kpool FP8 paged cache view into BF16 K cache.

    ``kv_cache_fp8`` is the 4D view used by paged MQA kernels:
    ``[num_pages, slots_per_page, 1, INDEX_HEAD_DIM + 4]``. The underlying
    kpool page layout is still ``[all K bytes][all FP32 scales]`` rather than
    interleaved per slot, so this kernel reads K rows and scales using the
    physical page layout and writes compact BF16 K rows:
    ``[num_pages, slots_per_page, 1, INDEX_HEAD_DIM]``.
    """

    assert kv_cache_fp8.dtype == torch.uint8
    assert kv_cache_fp8.ndim == 4 and kv_cache_fp8.shape[2] == 1
    assert kv_cache_fp8.shape[-1] == INDEX_HEAD_DIM + 4
    assert kv_cache_fp8.is_contiguous()

    num_pages = kv_cache_fp8.shape[0]
    slots_per_page = kv_cache_fp8.shape[1]
    k_out = torch.empty(
        (num_pages, slots_per_page, 1, INDEX_HEAD_DIM),
        dtype=torch.bfloat16,
        device=kv_cache_fp8.device,
    )
    if num_pages == 0 or slots_per_page == 0:
        return k_out

    _kpool_dequantize_fp8_paged_kv_cache_kernel[
        (num_pages, triton.cdiv(slots_per_page, 4))
    ](
        kv_cache_fp8,
        kv_cache_fp8.view(torch.float32),
        k_out,
        kv_cache_fp8.stride(0),
        k_out.stride(0),
        k_out.stride(1),
        k_out.stride(3),
        SLOTS_PER_PAGE=slots_per_page,
        HEAD_DIM=INDEX_HEAD_DIM,
        BLOCK_SLOTS=4,
        BLOCK_D=triton.next_power_of_2(INDEX_HEAD_DIM),
        num_warps=4,
    )
    return k_out


@triton.jit
def _bf16_paged_mqa_logits_kernel(
    q_ptr,
    kv_ptr,
    weights_ptr,
    seqlens_ptr,
    page_table_ptr,
    logits_ptr,
    q_stride_0,
    q_stride_1,
    kv_stride_0,
    kv_stride_1,
    kv_stride_3,
    weights_stride_0,
    weights_stride_1,
    page_table_stride_0,
    page_table_stride_1,
    logits_stride_0,
    max_seq_len,
    NUM_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    token_offsets = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    valid_output = token_offsets < max_seq_len
    seqlen = tl.load(seqlens_ptr + batch_idx).to(tl.int32)
    valid_token = valid_output & (token_offsets < seqlen)
    logical_page = token_offsets // PAGE_SIZE
    slot = token_offsets % PAGE_SIZE
    physical_page = tl.load(
        page_table_ptr
        + batch_idx * page_table_stride_0
        + logical_page * page_table_stride_1,
        mask=valid_token,
        other=0,
    ).to(tl.int32)
    physical_page = tl.maximum(physical_page, 0)

    dims = tl.arange(0, BLOCK_D)
    scores = tl.zeros([BLOCK_K], dtype=tl.float32)
    for head_idx in tl.static_range(0, NUM_HEADS):
        q = tl.load(
            q_ptr + batch_idx * q_stride_0 + head_idx * q_stride_1 + dims,
            mask=dims < HEAD_DIM,
            other=0.0,
        ).to(tl.bfloat16)
        k_offsets = (
            physical_page[:, None] * kv_stride_0
            + slot[:, None] * kv_stride_1
            + dims[None, :] * kv_stride_3
        )
        k = tl.load(
            kv_ptr + k_offsets,
            mask=valid_token[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.bfloat16)
        dot = tl.sum((k * q[None, :]).to(tl.float32), axis=1)
        weight = tl.load(
            weights_ptr + batch_idx * weights_stride_0 + head_idx * weights_stride_1
        ).to(tl.float32)
        scores += tl.maximum(dot, 0.0) * weight

    tl.store(
        logits_ptr + batch_idx * logits_stride_0 + token_offsets,
        scores,
        mask=valid_output,
    )


@triton.jit
def _decode_e4m3fn(raw):
    """Decode raw E4M3FN bytes without Triton's FP8 load conversion.

    HCU's current Triton FP8 pointer cast does not preserve E4M3FN values.
    ``raw`` contains the cache byte so decode it using the IEEE-like E4M3FN
    layout: sign[7], exponent[6:3] with bias 7, and mantissa[2:0].
    """

    raw = raw.to(tl.int32)
    sign = tl.where((raw & 0x80) != 0, -1.0, 1.0)
    exponent = (raw >> 3) & 0x0F
    mantissa = raw & 0x07
    normal = tl.exp2(exponent.to(tl.float32) - 7.0) * (
        1.0 + mantissa.to(tl.float32) * 0.125
    )
    subnormal = mantissa.to(tl.float32) * 0.001953125
    value = tl.where(exponent == 0, subnormal, normal)
    # E4M3FN reserves exponent=15,mantissa=7 as NaN. Valid cache writes are
    # clamped to the largest finite value, but treat an unexpected NaN byte as
    # zero so it cannot poison a sparse score row.
    value = tl.where((exponent == 15) & (mantissa == 7), 0.0, value)
    return sign * value


@triton.jit
def _kpool_dequantize_fp8_paged_kv_cache_kernel(
    kv_u8_ptr,
    kv_fp32_ptr,
    k_out_ptr,
    kv_page_stride,
    out_page_stride,
    out_slot_stride,
    out_dim_stride,
    SLOTS_PER_PAGE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SLOTS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    page_idx = tl.program_id(0)
    slot = tl.program_id(1) * BLOCK_SLOTS + tl.arange(0, BLOCK_SLOTS)
    valid_slot = slot < SLOTS_PER_PAGE
    dims = tl.arange(0, BLOCK_D)
    valid_dim = dims < HEAD_DIM

    k_offsets = page_idx * kv_page_stride + slot[:, None] * HEAD_DIM + dims[None, :]
    raw_k = tl.load(
        kv_u8_ptr + k_offsets,
        mask=valid_slot[:, None] & valid_dim[None, :],
        other=0.0,
    )

    scale_byte_offsets = (
        page_idx * kv_page_stride + SLOTS_PER_PAGE * HEAD_DIM + slot * 4
    )
    scale_offsets = scale_byte_offsets // 4
    scales = tl.load(kv_fp32_ptr + scale_offsets, mask=valid_slot, other=0.0).to(
        tl.float32
    )

    k = (_decode_e4m3fn(raw_k) * scales[:, None]).to(tl.bfloat16)
    out_offsets = (
        page_idx * out_page_stride
        + slot[:, None] * out_slot_stride
        + dims[None, :] * out_dim_stride
    )
    tl.store(
        k_out_ptr + out_offsets,
        k,
        mask=valid_slot[:, None] & valid_dim[None, :],
    )


@triton.jit
def _kpool_bf16_paged_mqa_logits_kernel(
    q_ptr,
    buf_u8_ptr,
    buf_fp32_ptr,
    weights_ptr,
    pool_seqlens_ptr,
    page_table_ptr,
    logits_ptr,
    q_stride_0,
    q_stride_1,
    weights_stride_0,
    weights_stride_1,
    page_table_stride_0,
    page_table_stride_1,
    logits_stride_0,
    max_seq_len,
    NUM_HEADS: tl.constexpr,
    SLOTS_PER_PAGE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    pool_offset = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    valid_output = pool_offset < max_seq_len
    pool_seqlen = tl.load(pool_seqlens_ptr + batch_idx).to(tl.int32)
    valid_pool = valid_output & (pool_offset < pool_seqlen)

    logical_page = pool_offset // SLOTS_PER_PAGE
    slot = pool_offset % SLOTS_PER_PAGE
    physical_page = tl.load(
        page_table_ptr
        + batch_idx * page_table_stride_0
        + logical_page * page_table_stride_1,
        mask=valid_pool,
        other=0,
    ).to(tl.int32)
    physical_page = tl.maximum(physical_page, 0)

    dims = tl.arange(0, BLOCK_D)
    scores = tl.zeros([BLOCK_K], dtype=tl.float32)
    scale_offsets = (
        physical_page * (BUF_NUMEL_PER_PAGE // 4)
        + (SLOTS_PER_PAGE * HEAD_DIM // 4)
        + slot
    )
    scales = tl.load(buf_fp32_ptr + scale_offsets, mask=valid_pool, other=0.0).to(
        tl.float32
    )

    for head_idx in tl.static_range(0, NUM_HEADS):
        q = tl.load(
            q_ptr + batch_idx * q_stride_0 + head_idx * q_stride_1 + dims,
            mask=dims < HEAD_DIM,
            other=0.0,
        ).to(tl.bfloat16)
        k_offsets = (
            physical_page[:, None] * BUF_NUMEL_PER_PAGE
            + slot[:, None] * HEAD_DIM
            + dims[None, :]
        )
        raw_k = tl.load(
            buf_u8_ptr + k_offsets,
            mask=valid_pool[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
        )
        k = _decode_e4m3fn(raw_k)
        k = (k * scales[:, None]).to(tl.bfloat16)
        # Match _bf16_paged_mqa_logits_kernel (line 281): accumulate the
        # BF16 elementwise product directly in FP32 without an extra
        # bf16 round on (k*q), which would drop mantissa bits and
        # systematically flatten the head-dim reduction.
        dot = tl.sum((k * q[None, :]).to(tl.float32), axis=1)
        weight = tl.load(
            weights_ptr + batch_idx * weights_stride_0 + head_idx * weights_stride_1
        ).to(tl.float32)
        scores += tl.maximum(dot, 0.0) * weight

    tl.store(
        logits_ptr + batch_idx * logits_stride_0 + pool_offset,
        tl.where(valid_pool, scores, 0.0),
        mask=valid_output,
    )


def gather_index_k_scale_prefix_into(
    pool,
    buf: torch.Tensor,
    page_indices: torch.Tensor,
    seq_len: int,
    k_out: torch.Tensor,
    scale_out: torch.Tensor,
) -> None:
    if seq_len == 0:
        return
    # Contiguity is silent-corruption territory; the rest of the shape/dtype
    # checks are upstream guarantees.
    assert buf.is_contiguous()
    assert page_indices.is_contiguous()
    assert k_out.is_contiguous()
    assert scale_out.is_contiguous()

    slots_per_page = pool.slots_per_page
    _gather_index_k_scale_prefix_into_kernel[(seq_len,)](
        buf.view(torch.int8),
        buf.view(torch.float32),
        page_indices,
        k_out.view(torch.int8),
        scale_out,
        SLOTS_PER_PAGE=slots_per_page,
        BUF_NUMEL_PER_PAGE=buf.shape[1],
        HEAD_DIM=INDEX_HEAD_DIM,
        S_OFFSET_NBYTES_IN_PAGE=slots_per_page * INDEX_HEAD_DIM,
        BLOCK_D=triton.next_power_of_2(INDEX_HEAD_DIM),
    )


@triton.jit
def _gather_index_k_scale_prefix_into_kernel(
    buf_ptr,
    buf_fp32_ptr,
    page_indices_ptr,
    k_out_ptr,
    scale_out_ptr,
    SLOTS_PER_PAGE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_id = tl.program_id(0)
    page_idx = token_id // SLOTS_PER_PAGE
    token_offset_in_page = token_id % SLOTS_PER_PAGE
    page = tl.load(page_indices_ptr + page_idx)

    offs = tl.arange(0, BLOCK_D)
    mask = offs < HEAD_DIM
    src_k_offsets = page * BUF_NUMEL_PER_PAGE + token_offset_in_page * HEAD_DIM + offs
    dst_k_offsets = token_id * HEAD_DIM + offs
    k = tl.load(buf_ptr + src_k_offsets, mask=mask)
    tl.store(k_out_ptr + dst_k_offsets, k, mask=mask)

    src_s_offset = (
        page * BUF_NUMEL_PER_PAGE // 4
        + S_OFFSET_NBYTES_IN_PAGE // 4
        + token_offset_in_page
    )
    scale = tl.load(buf_fp32_ptr + src_s_offset)
    tl.store(scale_out_ptr + token_id, scale)


def kpool_build_ragged_layout(
    full_page_table: torch.Tensor,
    cu_pages_excl: torch.Tensor,
    ragged_pool_pages: torch.Tensor,
    cu_q_len_excl: torch.Tensor,
    ragged_q_len: torch.Tensor,
    pooled_seq_lens_expanded: torch.Tensor,
    slots_per_page: int,
    total_pool_pages: int,
    total_q: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse the ragged-topk layout build: gather padded page-table rows into
    a flat ``concat_page_table`` and broadcast per-batch K offsets to
    per-q ``q_ks`` / ``q_ke``.

    Replaces ~8 small torch ops (cumsum / arange / repeat_interleave /
    index_select / add) with a single kernel; the prefix sums come from
    the CPU planner via the existing i32 H2D channel so no GPU cumsum is
    needed.
    """
    device = full_page_table.device
    n_rag = cu_pages_excl.shape[0]
    concat_page_table = torch.empty(
        (total_pool_pages,), dtype=full_page_table.dtype, device=device
    )
    q_ks = torch.empty((total_q,), dtype=torch.int32, device=device)
    q_ke = torch.empty((total_q,), dtype=torch.int32, device=device)
    if n_rag == 0:
        return concat_page_table, q_ks, q_ke

    max_pool_pages = full_page_table.shape[1]
    _kpool_build_ragged_layout_kernel[(n_rag,)](
        full_page_table,
        cu_pages_excl,
        ragged_pool_pages,
        cu_q_len_excl,
        ragged_q_len,
        pooled_seq_lens_expanded,
        concat_page_table,
        q_ks,
        q_ke,
        max_pool_pages,
        slots_per_page,
        BLOCK_PAGE=128,
        BLOCK_Q=128,
    )
    return concat_page_table, q_ks, q_ke


# GLM NOTE: the planner tensors are slices of the shared i32 H2D channel, so
# their data-pointer alignment (and MAX_POOL_PAGES divisibility) changes per
# batch. Each new combination is a new Triton specialization = a full HCU JIT
# recompile on every rank, leaking native heap per compile until cgroup OOM.
@triton.jit(
    do_not_specialize=["MAX_POOL_PAGES"],
    do_not_specialize_on_alignment=[
        "cu_pages_excl_ptr",
        "ragged_pool_pages_ptr",
        "cu_q_len_excl_ptr",
        "ragged_q_len_ptr",
        "pooled_seq_lens_ptr",
    ],
)
def _kpool_build_ragged_layout_kernel(
    full_page_table_ptr,
    cu_pages_excl_ptr,
    ragged_pool_pages_ptr,
    cu_q_len_excl_ptr,
    ragged_q_len_ptr,
    pooled_seq_lens_ptr,
    concat_page_table_ptr,
    q_ks_ptr,
    q_ke_ptr,
    MAX_POOL_PAGES,
    SLOTS_PER_PAGE: tl.constexpr,
    BLOCK_PAGE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    k = tl.program_id(0)
    page_start = tl.load(cu_pages_excl_ptr + k)
    n_pages = tl.load(ragged_pool_pages_ptr + k)
    q_start = tl.load(cu_q_len_excl_ptr + k)
    q_count = tl.load(ragged_q_len_ptr + k)
    ks_val = page_start * SLOTS_PER_PAGE

    for p_off in tl.range(0, BLOCK_PAGE * tl.cdiv(n_pages, BLOCK_PAGE), BLOCK_PAGE):
        p_offs = p_off + tl.arange(0, BLOCK_PAGE)
        p_mask = p_offs < n_pages
        pages = tl.load(
            full_page_table_ptr + k * MAX_POOL_PAGES + p_offs,
            mask=p_mask,
            other=0,
        )
        tl.store(concat_page_table_ptr + page_start + p_offs, pages, mask=p_mask)

    for q_off in tl.range(0, BLOCK_Q * tl.cdiv(q_count, BLOCK_Q), BLOCK_Q):
        q_offs = q_off + tl.arange(0, BLOCK_Q)
        q_mask = q_offs < q_count
        plen = tl.load(pooled_seq_lens_ptr + q_start + q_offs, mask=q_mask, other=0)
        tl.store(
            q_ks_ptr + q_start + q_offs,
            tl.full([BLOCK_Q], ks_val, tl.int32),
            mask=q_mask,
        )
        tl.store(q_ke_ptr + q_start + q_offs, ks_val + plen, mask=q_mask)


def update_kpool_decode_cuda_graph_page_tables(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    page_table_1: torch.Tensor,
    real_page_table: torch.Tensor,
    max_len: int,
    pool_size: int,
    page_size: int,
) -> None:
    """Update CUDA graph decode page tables for kpool in one pass."""
    # Algorithm invariant + contiguity (silent-corruption guard); other
    # ndim/dtype/shape checks are upstream guarantees.
    assert page_size % pool_size == 0
    assert page_table_1.is_contiguous()
    assert real_page_table.is_contiguous()

    bs = req_pool_indices.shape[0]
    if bs == 0 or max_len == 0:
        return

    real_cols = triton.cdiv(max_len, page_size)

    block_n = 1024
    block_p = block_n // page_size
    grid = (bs, triton.cdiv(max_len, block_n))
    _update_kpool_decode_cuda_graph_page_tables_kernel[grid](
        req_to_token,
        req_pool_indices,
        page_table_1,
        real_page_table,
        req_to_token.stride(0),
        page_table_1.stride(0),
        real_page_table.stride(0),
        max_len,
        real_cols,
        PAGE_SIZE=page_size,
        BLOCK_N=block_n,
        BLOCK_P=block_p,
    )


@triton.jit
def _update_kpool_decode_cuda_graph_page_tables_kernel(
    req_to_token_ptr,
    req_pool_indices_ptr,
    page_table_1_ptr,
    real_page_table_ptr,
    req_to_token_stride_0,
    page_table_1_stride_0,
    real_page_table_stride_0,
    max_len,
    real_cols,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    req = tl.load(req_pool_indices_ptr + row)

    token_cols = block * BLOCK_N + tl.arange(0, BLOCK_N)
    token_mask = token_cols < max_len
    token_values = tl.load(
        req_to_token_ptr + req * req_to_token_stride_0 + token_cols,
        mask=token_mask,
        other=0,
    ).to(tl.int32)
    tl.store(
        page_table_1_ptr + row * page_table_1_stride_0 + token_cols,
        token_values,
        mask=token_mask,
    )

    page_cols = block * BLOCK_P + tl.arange(0, BLOCK_P)
    page_token_cols = page_cols * PAGE_SIZE
    page_mask = page_cols < real_cols
    page_values = tl.load(
        req_to_token_ptr + req * req_to_token_stride_0 + page_token_cols,
        mask=page_mask,
        other=0,
    ).to(tl.int32)
    tl.store(
        real_page_table_ptr + row * real_page_table_stride_0 + page_cols,
        page_values // PAGE_SIZE,
        mask=page_mask,
    )


def update_kpool_write_plan_cuda_graph(
    write_start: torch.Tensor,
    req_pool_indices: torch.Tensor,
    real_page_table: torch.Tensor,
    req_out: torch.Tensor,
    write_start_out: torch.Tensor,
    tail_logical_start_out: torch.Tensor,
    write_loc_out: torch.Tensor,
    pool_seqlens_per_q_out: Optional[torch.Tensor],
    seqlens_per_q_out: Optional[torch.Tensor],
    *,
    pool_size: int,
    num_draft_tokens: int,
    max_closed_pools: int = 1,
    slots_per_page: int,
) -> None:
    """Build the kpool write plan (decode + target_verify) device-side in one launch.

    Grid ``(B,)``, per program computes everything for one batch:
      decode (N=1):  write_start = positions[b]
      verify (N>=1): write_start = committed_seq_lens[b] (= seq_lens)

    ``req`` / ``write_start`` are per-batch. Close-pool addressing tensors
    are flattened as ``[B * max_closed_pools]`` so chain-only speculative
    verify can close more than one pool when ``num_draft_tokens > pool_size``.

    ``pool_seqlens_per_q_out`` / ``seqlens_per_q_out`` are verify-only
    (shape ``[B*N]``); pass ``None`` for decode and the kernel skips them.
    """
    required_closed_pools = max(1, (num_draft_tokens + pool_size - 1) // pool_size)
    assert max_closed_pools >= required_closed_pools, (
        f"kpool write plan needs max_closed_pools >= {required_closed_pools} "
        f"for N={num_draft_tokens}, pool_size={pool_size}; got {max_closed_pools}."
    )

    bs = write_start.shape[0]
    if bs == 0 or num_draft_tokens == 0:
        return

    has_per_q_outputs = pool_seqlens_per_q_out is not None
    assert has_per_q_outputs == (seqlens_per_q_out is not None), (
        "pool_seqlens_per_q_out and seqlens_per_q_out must be both set or both None"
    )

    _update_kpool_write_plan_kernel[(bs,)](
        write_start,
        req_pool_indices,
        real_page_table,
        req_out,
        write_start_out,
        tail_logical_start_out,
        write_loc_out,
        pool_seqlens_per_q_out,
        seqlens_per_q_out,
        real_page_table.stride(0),
        POOL_SIZE=pool_size,
        N=num_draft_tokens,
        MAX_CLOSED_POOLS=max_closed_pools,
        SLOTS_PER_PAGE=slots_per_page,
        HAS_PER_Q=has_per_q_outputs,
    )


def update_kpool_write_plan_cuda_graph_multi_decode(
    write_start: torch.Tensor,
    req_pool_indices: torch.Tensor,
    real_page_table: torch.Tensor,
    req_out0: torch.Tensor,
    write_start_out0: torch.Tensor,
    tail_logical_start_out0: torch.Tensor,
    write_loc_out0: torch.Tensor,
    req_out1: torch.Tensor,
    write_start_out1: torch.Tensor,
    tail_logical_start_out1: torch.Tensor,
    write_loc_out1: torch.Tensor,
    req_out2: torch.Tensor,
    write_start_out2: torch.Tensor,
    tail_logical_start_out2: torch.Tensor,
    write_loc_out2: torch.Tensor,
    req_out3: Optional[torch.Tensor] = None,
    write_start_out3: Optional[torch.Tensor] = None,
    tail_logical_start_out3: Optional[torch.Tensor] = None,
    write_loc_out3: Optional[torch.Tensor] = None,
    *,
    pool_size: int,
    slots_per_page: int,
) -> None:
    """Build three or four decode kpool write plans in one Triton launch.

    Multi-step EAGLE v2 replay copies identical decode metadata into the first
    draft backends. Their kpool write-plan inputs are also identical, but the
    destination graph buffers are backend-local. Updating the plans in one
    launcher removes repeated Python/Triton scheduling trips between spec steps.
    """
    bs = write_start.shape[0]
    if bs == 0:
        return
    has_fourth = req_out3 is not None
    assert has_fourth == (
        write_start_out3 is not None
        and tail_logical_start_out3 is not None
        and write_loc_out3 is not None
    ), "fourth decode plan outputs must be all set or all None"

    _update_kpool_write_plan_multi_decode_kernel[(bs,)](
        write_start,
        req_pool_indices,
        real_page_table,
        req_out0,
        write_start_out0,
        tail_logical_start_out0,
        write_loc_out0,
        req_out1,
        write_start_out1,
        tail_logical_start_out1,
        write_loc_out1,
        req_out2,
        write_start_out2,
        tail_logical_start_out2,
        write_loc_out2,
        req_out3 if has_fourth else req_out2,
        write_start_out3 if has_fourth else write_start_out2,
        tail_logical_start_out3 if has_fourth else tail_logical_start_out2,
        write_loc_out3 if has_fourth else write_loc_out2,
        real_page_table.stride(0),
        POOL_SIZE=pool_size,
        SLOTS_PER_PAGE=slots_per_page,
        HAS_FOURTH=has_fourth,
    )


@triton.jit
def _update_kpool_write_plan_kernel(
    write_start_ptr,
    req_pool_indices_ptr,
    real_page_table_ptr,
    req_out_ptr,
    write_start_out_ptr,
    tail_logical_start_out_ptr,
    write_loc_out_ptr,
    pool_seqlens_per_q_out_ptr,
    seqlens_per_q_out_ptr,
    real_page_table_stride_0,
    POOL_SIZE: tl.constexpr,
    N: tl.constexpr,
    MAX_CLOSED_POOLS: tl.constexpr,
    SLOTS_PER_PAGE: tl.constexpr,
    HAS_PER_Q: tl.constexpr,
):
    b = tl.program_id(0)
    ws = tl.load(write_start_ptr + b).to(tl.int32)
    req = tl.load(req_pool_indices_ptr + b)

    # _decompose_compress(ws, N, POOL_SIZE): the consumer re-derives the
    # actual close count in-kernel from write_start and gate_n, while the
    # plan precomputes the maximum candidate close addresses for this N.
    base_pool = ws // POOL_SIZE

    # Per-q expanded fields (verify only): cover all N draft positions.
    if HAS_PER_Q:
        for k in tl.static_range(0, N):
            row = b * N + k
            seqlen_per_q = ws + k + 1
            tl.store(seqlens_per_q_out_ptr + row, seqlen_per_q)
            tl.store(pool_seqlens_per_q_out_ptr + row, seqlen_per_q // POOL_SIZE)

    tl.store(write_start_out_ptr + b, ws)

    tl.store(req_out_ptr + b, req)
    # Addressing fields stored unconditionally; the consumer gates each
    # candidate on `(ws + gate_n) // P - ws // P` recomputed in-kernel.
    for i_pool in tl.static_range(0, MAX_CLOSED_POOLS):
        pool_id = base_pool + i_pool
        pool_page_group = pool_id // SLOTS_PER_PAGE
        # Read packed page from row (b * N): for verify the page table is
        # repeat_interleave'd to [B*N, max_pages] so this picks the first-of-N
        # (identical across N); for decode N=1 so this is just row b.
        packed_page = tl.load(
            real_page_table_ptr + (b * N) * real_page_table_stride_0 + pool_page_group
        ).to(tl.int64)
        write_loc = packed_page * SLOTS_PER_PAGE + (pool_id % SLOTS_PER_PAGE)
        out_row = b * MAX_CLOSED_POOLS + i_pool
        tl.store(
            tail_logical_start_out_ptr + out_row,
            (pool_id * POOL_SIZE).to(tl.int32),
        )
        tl.store(write_loc_out_ptr + out_row, write_loc.to(tl.int64))


@triton.jit
def _update_kpool_write_plan_multi_decode_kernel(
    write_start_ptr,
    req_pool_indices_ptr,
    real_page_table_ptr,
    req_out0_ptr,
    write_start_out0_ptr,
    tail_logical_start_out0_ptr,
    write_loc_out0_ptr,
    req_out1_ptr,
    write_start_out1_ptr,
    tail_logical_start_out1_ptr,
    write_loc_out1_ptr,
    req_out2_ptr,
    write_start_out2_ptr,
    tail_logical_start_out2_ptr,
    write_loc_out2_ptr,
    req_out3_ptr,
    write_start_out3_ptr,
    tail_logical_start_out3_ptr,
    write_loc_out3_ptr,
    real_page_table_stride_0,
    POOL_SIZE: tl.constexpr,
    SLOTS_PER_PAGE: tl.constexpr,
    HAS_FOURTH: tl.constexpr,
):
    b = tl.program_id(0)
    ws = tl.load(write_start_ptr + b).to(tl.int32)
    req = tl.load(req_pool_indices_ptr + b)
    base_pool = ws // POOL_SIZE
    tail_logical_start = base_pool * POOL_SIZE
    pool_page_group = base_pool // SLOTS_PER_PAGE
    packed_page = tl.load(
        real_page_table_ptr + b * real_page_table_stride_0 + pool_page_group
    ).to(tl.int64)
    write_loc = packed_page * SLOTS_PER_PAGE + (base_pool % SLOTS_PER_PAGE)

    tl.store(req_out0_ptr + b, req)
    tl.store(write_start_out0_ptr + b, ws)
    tl.store(tail_logical_start_out0_ptr + b, tail_logical_start.to(tl.int32))
    tl.store(write_loc_out0_ptr + b, write_loc.to(tl.int64))

    tl.store(req_out1_ptr + b, req)
    tl.store(write_start_out1_ptr + b, ws)
    tl.store(tail_logical_start_out1_ptr + b, tail_logical_start.to(tl.int32))
    tl.store(write_loc_out1_ptr + b, write_loc.to(tl.int64))

    tl.store(req_out2_ptr + b, req)
    tl.store(write_start_out2_ptr + b, ws)
    tl.store(tail_logical_start_out2_ptr + b, tail_logical_start.to(tl.int32))
    tl.store(write_loc_out2_ptr + b, write_loc.to(tl.int64))

    if HAS_FOURTH:
        tl.store(req_out3_ptr + b, req)
        tl.store(write_start_out3_ptr + b, ws)
        tl.store(tail_logical_start_out3_ptr + b, tail_logical_start.to(tl.int32))
        tl.store(write_loc_out3_ptr + b, write_loc.to(tl.int64))


def expand_pooled_groups_to_topk(
    group_ids: torch.Tensor,
    group_valid: torch.Tensor,
    topk: int,
    pool_size: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand selected full-pool ids to a strict-width token topk tensor."""
    # Correctness invariants (cheap, structural).
    assert topk % pool_size == 0
    assert page_table is None or topk_offsets is None

    device = group_ids.device
    offsets = torch.arange(pool_size, device=device, dtype=torch.int64)
    token_ids = group_ids.to(torch.int64).unsqueeze(-1) * pool_size + offsets
    token_ids = token_ids.reshape(group_ids.shape[0], topk)
    valid = (
        group_valid.unsqueeze(-1)
        .expand(-1, -1, pool_size)
        .reshape(group_ids.shape[0], topk)
    )

    if page_table is not None:
        safe_ids = token_ids.clamp(min=0, max=page_table.shape[1] - 1)
        output = torch.gather(page_table, dim=1, index=safe_ids).to(torch.int32)
    elif topk_offsets is not None:
        if topk_offsets.ndim == 2:
            topk_offsets = topk_offsets.squeeze(1)
        output = (token_ids + topk_offsets.to(torch.int64).unsqueeze(1)).to(torch.int32)
    else:
        output = token_ids.to(torch.int32)

    return torch.where(valid, output, torch.full_like(output, -1))


def append_kpool_tail_to_topk(
    topk_result: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_lens: torch.Tensor,
    pool_size: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Append non-pooled tail tokens after selected expanded-history tokens."""
    rows, n_cols = topk_result.shape
    out_cols = n_cols + pool_size - 1
    device = topk_result.device
    cols = torch.arange(out_cols, device=device, dtype=torch.int64).unsqueeze(0)
    seq_lens = seq_lens.to(device=device, dtype=torch.int64).reshape(rows, 1)
    pool_lens = pool_lens.to(device=device, dtype=torch.int64).reshape(rows, 1)

    history_len = (pool_lens * pool_size).clamp(max=n_cols)
    history_mask = cols < history_len
    safe_history_cols = cols.clamp(min=0, max=max(n_cols - 1, 0)).expand(rows, -1)
    history_value = torch.gather(topk_result, dim=1, index=safe_history_cols)
    out = torch.where(
        history_mask,
        history_value,
        torch.full((rows, out_cols), -1, dtype=topk_result.dtype, device=device),
    )

    tail_offset = cols - history_len
    tail_count = seq_lens % pool_size
    tail_mask = (tail_offset >= 0) & (tail_offset < tail_count)
    tail_raw = pool_lens * pool_size + tail_offset

    if page_table is not None:
        if page_table.shape[0] != rows:
            if page_table.shape[0] == 1:
                page_table = page_table.expand(rows, -1)
            else:
                page_table = page_table[:rows]
        safe_tail = tail_raw.clamp(min=0, max=page_table.shape[1] - 1)
        tail_value = torch.gather(page_table, dim=1, index=safe_tail)
    elif topk_offsets is not None:
        if topk_offsets.ndim == 2:
            topk_offsets = topk_offsets.squeeze(1)
        topk_offsets = topk_offsets.to(device=device, dtype=torch.int64).reshape(
            rows, 1
        )
        tail_value = tail_raw + topk_offsets
    else:
        tail_value = tail_raw

    tail_value = tail_value.to(dtype=topk_result.dtype)
    return torch.where(tail_mask, tail_value, out)


@triton.jit
def _append_kpool_tail_to_topk_kernel(
    topk_ptr,
    seq_lens_ptr,
    pool_lens_ptr,
    page_table_ptr,
    topk_offsets_ptr,
    out_ptr,
    topk_stride_0,
    topk_stride_1,
    page_table_stride_0,
    page_table_stride_1,
    out_stride_0,
    out_stride_1,
    N_COLS: tl.constexpr,
    OUT_COLS: tl.constexpr,
    PAGE_TABLE_COLS: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    HAS_PAGE_TABLE: tl.constexpr,
    HAS_TOPK_OFFSETS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_COLS)
    mask = cols < OUT_COLS

    seq_len = tl.load(seq_lens_ptr + row).to(tl.int32)
    pool_len = tl.load(pool_lens_ptr + row).to(tl.int32)
    tail_start = pool_len * POOL_SIZE
    history_len = tl.minimum(tail_start, N_COLS)
    tail_count = seq_len % POOL_SIZE

    is_history = cols < history_len
    safe_history_cols = tl.minimum(cols, N_COLS - 1)
    history_value = tl.load(
        topk_ptr + row * topk_stride_0 + safe_history_cols * topk_stride_1,
        mask=mask & is_history,
        other=-1,
    )

    tail_offset = cols - history_len
    is_tail = (tail_offset >= 0) & (tail_offset < tail_count)
    tail_raw = tail_start + tail_offset
    tail_value = tail_raw
    if HAS_PAGE_TABLE:
        safe_tail = tl.minimum(tl.maximum(tail_raw, 0), PAGE_TABLE_COLS - 1)
        tail_value = tl.load(
            page_table_ptr
            + row * page_table_stride_0
            + safe_tail * page_table_stride_1,
            mask=mask & is_tail,
            other=-1,
        ).to(tl.int32)
    if HAS_TOPK_OFFSETS:
        offset = tl.load(topk_offsets_ptr + row).to(tl.int32)
        tail_value = tail_raw + offset

    value = tl.where(is_history, history_value, -1)
    value = tl.where(is_tail, tail_value, value)
    tl.store(out_ptr + row * out_stride_0 + cols * out_stride_1, value, mask=mask)


def _torch_topk_pooled_history(
    logits: torch.Tensor,
    group_lengths: torch.Tensor,
    pool_size: int,
    topk: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
    seq_lens: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
    out_rows: int | None = None,
    page_table_row_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Deterministic torch.topk-based fallback for the kpool history topk.

    Mirrors ``topk_from_pooled_history_logits`` semantics for use under
    ``enable_deterministic_inference``: the topk *selection* runs through
    ``torch.topk`` (deterministic on CUDA for fp32 input), while the
    downstream expand / page-table gather / tail-append helpers are reused
    verbatim (already deterministic).
    """
    rows, cols = logits.shape
    group_topk = topk // pool_size
    device = logits.device

    col_idx = torch.arange(cols, device=device, dtype=torch.int32)
    if row_starts is None:
        valid_lo = torch.zeros(rows, device=device, dtype=torch.int32)
    else:
        valid_lo = row_starts.to(torch.int32)
    valid_hi = valid_lo + group_lengths.to(torch.int32)
    valid_mask = (col_idx.unsqueeze(0) >= valid_lo.unsqueeze(1)) & (
        col_idx.unsqueeze(0) < valid_hi.unsqueeze(1)
    )

    masked = torch.where(valid_mask, logits, torch.full_like(logits, float("-inf")))
    k = min(group_topk, cols)
    _, group_ids = torch.topk(masked, k=k, dim=1)
    if k < group_topk:
        pad = torch.zeros((rows, group_topk - k), device=device, dtype=group_ids.dtype)
        group_ids = torch.cat([group_ids, pad], dim=1)
    group_ids = group_ids.to(torch.int32)

    if row_starts is not None:
        group_ids = group_ids - valid_lo.unsqueeze(1)

    if page_table is not None and page_table_row_index is not None:
        page_table = page_table.index_select(0, page_table_row_index.to(torch.int64))

    max_valid_groups = min(cols, group_topk)
    rank = torch.arange(group_topk, device=device, dtype=torch.int32)
    valid_counts = group_lengths.to(torch.int32).clamp(max=max_valid_groups)
    group_valid = rank.unsqueeze(0) < valid_counts.unsqueeze(1)

    expanded = expand_pooled_groups_to_topk(
        group_ids.contiguous(),
        group_valid,
        topk=topk,
        pool_size=pool_size,
        page_table=page_table,
        topk_offsets=topk_offsets,
    )
    if seq_lens is None:
        result = expanded
    else:
        result = append_kpool_tail_to_topk(
            expanded,
            seq_lens=seq_lens,
            pool_lens=group_lengths,
            pool_size=pool_size,
            page_table=page_table,
            topk_offsets=topk_offsets,
        )

    if out_rows is None or out_rows == result.shape[0]:
        return result
    assert out_rows >= result.shape[0], (
        f"out_rows ({out_rows}) must be >= topk rows ({result.shape[0]})"
    )
    padded = torch.full(
        (out_rows, result.shape[1]),
        -1,
        dtype=result.dtype,
        device=result.device,
    )
    padded[: result.shape[0]] = result
    return padded


def topk_from_pooled_history_logits(
    logits: torch.Tensor,
    group_lengths: torch.Tensor,
    pool_size: int,
    topk: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
    seq_lens: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
    out_rows: int | None = None,
    page_table_row_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select full-pool groups, expand to tokens, and optionally append tail.

    ``row_starts`` lets the caller pass a single flat ``[total_q, K]``
    logits matrix where each row's valid scores live at columns
    ``[row_starts[i], row_starts[i] + group_lengths[i])`` instead of at
    column 0. Used by the kpool ragged-extend path to do one fused
    topk over concatenated batches.

    ``out_rows`` lets the caller request more output rows than
    ``logits.shape[0]``; the trailing rows are filled with -1. On the
    fused (group_topk in 128..512) path this happens in-kernel; on the
    slow ``fast_topk_v2`` path it falls back to a host-side ``torch.full``
    pad. Used when q is right-padded for mlp-sync (TP/CP) and the topk
    output should be sized against the padded q row count.
    """
    assert topk > 0
    assert topk % pool_size == 0

    _, cols = logits.shape
    group_topk = topk // pool_size
    if topk_offsets is not None and topk_offsets.ndim == 2:
        topk_offsets = topk_offsets.squeeze(1)

    if group_topk not in (128, 160, 192, 224, 256, 512, 2048):
        raise NotImplementedError(
            "index_kpool topk only supports pooled group_topk in "
            f"(128, 160, 192, 224, 256, 512, 2048), got {group_topk} "
            f"(topk={topk}, pool_size={pool_size})."
        )
    if not logits.is_cuda or logits.dtype != torch.float32:
        return _torch_topk_pooled_history(
            logits=logits,
            group_lengths=group_lengths,
            pool_size=pool_size,
            topk=topk,
            page_table=page_table,
            topk_offsets=topk_offsets,
            seq_lens=seq_lens,
            row_starts=row_starts,
            out_rows=out_rows,
            page_table_row_index=page_table_row_index,
        )

    if group_topk in (128, 160, 192, 224, 256, 512):
        try:
            from sgl_kernel import fast_kpool_topk_transform_fused
        except (ImportError, AttributeError):
            return _torch_topk_pooled_history(
                logits=logits,
                group_lengths=group_lengths,
                pool_size=pool_size,
                topk=topk,
                page_table=page_table,
                topk_offsets=topk_offsets,
                seq_lens=seq_lens,
                row_starts=row_starts,
                out_rows=out_rows,
                page_table_row_index=page_table_row_index,
            )

        return fast_kpool_topk_transform_fused(
            score=logits,
            lengths=group_lengths.to(torch.int32),
            pool_size=pool_size,
            topk=topk,
            page_table=page_table,
            topk_indices_offset=topk_offsets,
            row_starts=row_starts,
            seq_lens=seq_lens.to(torch.int32) if seq_lens is not None else None,
            out_rows=out_rows,
            page_table_row_index=page_table_row_index,
        )

    # Slow path: ``fast_topk_v2`` + manual expand. The fused fast_kpool
    # kernel handles ``out_rows`` in-kernel; here we replicate it with a
    # host-side -1 pad so callers see the same contract regardless of
    # which path runs.
    from sgl_kernel import fast_topk_v2

    # The slow path's expand/append helpers gather through ``page_table``
    # directly and have no row-index indirection. The kpool ragged extend
    # path (the only caller passing page_table_row_index) always lands on
    # the fast group_topk in (128..512) branch above, so this never fires.
    assert page_table_row_index is None, (
        "page_table_row_index requires the fused fast_kpool group_topk path"
    )

    group_lengths_i32 = group_lengths.to(torch.int32)
    selected_groups = fast_topk_v2(
        logits,
        group_lengths_i32,
        group_topk,
        row_starts=row_starts,
    )

    rank = torch.arange(group_topk, device=logits.device, dtype=torch.int32)
    max_valid_groups = min(cols, group_topk)
    valid_counts = torch.minimum(
        group_lengths_i32,
        torch.full_like(group_lengths_i32, max_valid_groups),
    )
    group_valid = rank.unsqueeze(0) < valid_counts.unsqueeze(1)
    expanded = expand_pooled_groups_to_topk(
        selected_groups.contiguous(),
        group_valid,
        topk=topk,
        pool_size=pool_size,
        page_table=page_table,
        topk_offsets=topk_offsets,
    )
    if seq_lens is None:
        result = expanded
    else:
        result = append_kpool_tail_to_topk(
            expanded,
            seq_lens=seq_lens,
            pool_lens=group_lengths,
            pool_size=pool_size,
            page_table=page_table,
            topk_offsets=topk_offsets,
        )

    if out_rows is None or out_rows == result.shape[0]:
        return result
    assert out_rows >= result.shape[0], (
        f"out_rows ({out_rows}) must be >= topk rows ({result.shape[0]})"
    )
    padded = torch.full(
        (out_rows, result.shape[1]),
        -1,
        dtype=result.dtype,
        device=result.device,
    )
    padded[: result.shape[0]] = result
    return padded


@triton.jit
def _hadamard128_stage(x, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    x3 = tl.reshape(x, (GROUPS, 2, STRIDE))
    x3 = tl.trans(x3, 0, 2, 1)
    a, b = tl.split(x3)
    x3 = tl.join(a + b, a - b)
    x3 = tl.trans(x3, 0, 2, 1)
    return tl.reshape(x3, (128,))


@triton.jit
def _hadamard128(x):
    x = _hadamard128_stage(x, 64, 1)
    x = _hadamard128_stage(x, 32, 2)
    x = _hadamard128_stage(x, 16, 4)
    x = _hadamard128_stage(x, 8, 8)
    x = _hadamard128_stage(x, 4, 16)
    x = _hadamard128_stage(x, 2, 32)
    x = _hadamard128_stage(x, 1, 64)
    return x * 0.08838834764831845


@triton.jit
def _hadamard_quantize_fp8(acc, denom, ROUND_SCALE: tl.constexpr):
    """Normalize -> bf16 round-trip -> Hadamard rotate -> bf16 round-trip
    -> fp8-e4m3 quantize. Returns (quantized, scale)."""
    x = (acc / denom).to(tl.bfloat16).to(tl.float32)
    x = _hadamard128(x).to(tl.bfloat16).to(tl.float32)

    fp8_max_inv = 1.0 / 448.0
    absmax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-4)
    if ROUND_SCALE:
        scale = tl.exp2(tl.ceil(tl.log2(absmax * fp8_max_inv)))
    else:
        scale = absmax * fp8_max_inv

    quantized = tl.minimum(tl.maximum(x / scale, -448.0), 448.0)
    return quantized, scale


@triton.jit
def _round_to_nearest_int(x):
    return tl.where(x >= 0.0, tl.floor(x + 0.5), tl.ceil(x - 0.5))


@triton.jit
def _hadamard_quantize_int8(acc, denom, ROUND_SCALE: tl.constexpr):
    """Normalize -> bf16 round-trip -> Hadamard rotate -> bf16 round-trip
    -> int8 quantize. Returns (quantized, scale)."""
    x = (acc / denom).to(tl.bfloat16).to(tl.float32)
    x = _hadamard128(x).to(tl.bfloat16).to(tl.float32)

    int8_max_inv = 1.0 / 127.0
    absmax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-4)
    if ROUND_SCALE:
        scale = tl.exp2(tl.ceil(tl.log2(absmax * int8_max_inv)))
    else:
        scale = absmax * int8_max_inv

    quantized = tl.minimum(tl.maximum(x / scale, -127.0), 127.0)
    quantized = _round_to_nearest_int(quantized).to(tl.int8)
    return quantized, scale


@triton.jit
def _kpool_assemble_softmax_rotate_write_cache_kernel(
    buf_i8_ptr,
    buf_fp32_ptr,
    chunk_k_ptr,
    chunk_score_ptr,
    tail_k_ptr,
    tail_score_ptr,
    req_pool_idx_ptr,
    n_from_tail_ptr,
    chunk_src_start_ptr,
    tail_logical_base_ptr,
    ape_ptr,
    loc_ptr,
    write_mask_ptr,
    chunk_stride_0,
    tail_stride_0,
    tail_stride_1,
    ape_stride_0,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    HAS_WRITE_MASK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SLOTS_PER_PAGE: tl.constexpr = 64,
):
    """Per-pool program. Slot src dispatch:
    if slot < n_from_tail[row]:
        src = tail[req_pool_idx[row], (tail_logical_base[row] + slot) % TAIL_SIZE]
    else:
        chunk_off = chunk_src_start[row] + (slot - n_from_tail[row])
        src = chunk[chunk_off]
    """
    row = tl.program_id(0)
    # CP rows masked out have no result to compute; skip the full
    # POOL_SIZE-load + Hadamard + fp8 quant pipeline (the previous
    # implementation paid them and only masked the final store).
    if HAS_WRITE_MASK:
        if not tl.load(write_mask_ptr + row):
            return

    offs = tl.arange(0, BLOCK_D)
    mask = offs < HEAD_DIM

    n_tail = tl.load(n_from_tail_ptr + row)
    req = tl.load(req_pool_idx_ptr + row)
    chunk_src = tl.load(chunk_src_start_ptr + row)
    tail_base = tl.load(tail_logical_base_ptr + row)

    # Online softmax over POOL_SIZE slots; (max, denom, acc) rescaled
    # in place to halve the load traffic vs a two-pass version.
    #
    # `from_tail = slot < n_tail` is a scalar runtime bool (n_tail is a
    # per-row scalar). Using a real `if/else` here -- instead of the
    # previous `tl.where(from_tail, load_tail, load_chunk)` -- means
    # only the active branch's loads are emitted. The dead pointer
    # arithmetic (e.g. ``tail_ptr + chunk_off``) is never computed, so
    # we cannot read into unmapped memory even when max_running_requests
    # is small.
    m = tl.full((BLOCK_D,), -float("inf"), tl.float32)
    acc = tl.full((BLOCK_D,), 0.0, tl.float32)
    denom = tl.full((BLOCK_D,), 0.0, tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        if slot < n_tail:
            phys = (tail_base + slot) % TAIL_SIZE
            off = req * tail_stride_0 + phys * tail_stride_1 + offs
            score = tl.load(tail_score_ptr + off, mask=mask, other=0.0).to(tl.float32)
            k = tl.load(tail_k_ptr + off, mask=mask, other=0.0).to(tl.float32)
        else:
            off = (chunk_src + (slot - n_tail)) * chunk_stride_0 + offs
            score = tl.load(chunk_score_ptr + off, mask=mask, other=0.0).to(tl.float32)
            k = tl.load(chunk_k_ptr + off, mask=mask, other=0.0).to(tl.float32)

        score += tl.load(ape_ptr + slot * ape_stride_0 + offs, mask=mask, other=0.0).to(
            tl.float32
        )

        new_m = tl.maximum(m, score)
        rescale = tl.exp(m - new_m)
        prob = tl.exp(score - new_m)
        denom = denom * rescale + prob
        acc = acc * rescale + k * prob
        m = new_m

    quantized, scale = _hadamard_quantize_int8(acc, denom, ROUND_SCALE)

    loc = tl.load(loc_ptr + row)
    loc_page_index = loc // SLOTS_PER_PAGE
    loc_token_offset_in_page = loc % SLOTS_PER_PAGE
    out_k_offsets = (
        loc_page_index * BUF_NUMEL_PER_PAGE + loc_token_offset_in_page * HEAD_DIM + offs
    )
    out_s_offset = (
        loc_page_index * BUF_NUMEL_PER_PAGE // 4
        + S_OFFSET_NBYTES_IN_PAGE // 4
        + loc_token_offset_in_page
    )

    tl.store(buf_i8_ptr + out_k_offsets, quantized, mask=mask)
    tl.store(buf_fp32_ptr + out_s_offset, scale)


def kpool_assemble_softmax_rotate_write_cache(
    pool,
    buf: torch.Tensor,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    n_from_tail: torch.Tensor,
    chunk_src_start: torch.Tensor,
    tail_logical_base: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: torch.Tensor | None = None,
    round_scale: bool = False,
) -> None:
    """Fused gather + softmax + Hadamard rotate + int8 quant + cache write.

    For each output pool row r, slots are gathered from tail or chunk
    based on n_from_tail[r] (see _kpool_assemble_softmax_rotate_write_cache_kernel).
    ``tail_logical_base[r]`` is the logical position of the in-progress pool's
    first slot (used to ring-address the saved tail prefix).
    """
    _log_kpool_wrapper_call(
        "kpool_assemble_softmax_rotate_write_cache",
        pool=pool,
        buf=buf,
        chunk_k=chunk_k,
        chunk_score=chunk_score,
        tail_k=tail_k,
        tail_score=tail_score,
        req_pool_idx=req_pool_idx,
        n_from_tail=n_from_tail,
        chunk_src_start=chunk_src_start,
        tail_logical_base=tail_logical_base,
        ape=ape,
        loc=loc,
        write_mask=write_mask,
        round_scale=round_scale,
    )
    pool_size = pool.index_kpool
    n_pools = req_pool_idx.shape[0]

    if n_pools == 0:
        return

    chunk_k = chunk_k.contiguous()
    chunk_score = chunk_score.contiguous()
    ape = ape.contiguous()
    loc = loc.contiguous()
    if write_mask is None:
        write_mask = torch.empty((1,), dtype=torch.bool, device=chunk_k.device)
        has_write_mask = False
    else:
        write_mask = write_mask.contiguous()
        has_write_mask = True

    buf_i8 = buf.view(torch.int8)
    buf_fp32 = buf.view(torch.float32)
    slots_per_page = pool.slots_per_page

    _kpool_assemble_softmax_rotate_write_cache_kernel[(n_pools,)](
        buf_i8,
        buf_fp32,
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        req_pool_idx,
        n_from_tail,
        chunk_src_start,
        tail_logical_base,
        ape,
        loc,
        write_mask,
        chunk_k.stride(0),
        tail_k.stride(0),
        tail_k.stride(1),
        ape.stride(0),
        BUF_NUMEL_PER_PAGE=buf.shape[1],
        POOL_SIZE=pool_size,
        TAIL_SIZE=tail_k.shape[1],
        HEAD_DIM=INDEX_HEAD_DIM,
        S_OFFSET_NBYTES_IN_PAGE=slots_per_page * INDEX_HEAD_DIM,
        ROUND_SCALE=round_scale,
        HAS_WRITE_MASK=has_write_mask,
        BLOCK_D=triton.next_power_of_2(INDEX_HEAD_DIM),
        SLOTS_PER_PAGE=slots_per_page,
    )


def scatter_kpool_tail_updates(
    pool,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    dst_logical_start: torch.Tensor,
    chunk_src_start: torch.Tensor,
    n_write: torch.Tensor,
) -> None:
    """Per-batch tail buffer writes folded into one kernel, ring-addressed.

    For each batch row r, writes ``n_write[r]`` tail slots at logical positions
    ``[dst_logical_start[r], dst_logical_start[r] + n_write[r])``, physical
    ``(dst_logical_start[r] + s) % TAIL_SIZE`` for ``s ?[0, n_write[r])``.
    """
    # Per-row n_write is at most pool_size (the rest closes pools instead),
    # so the kernel grid covers pool_size slots and masks past n_write.
    pool_size = pool.index_kpool
    n_rows = req_pool_idx.shape[0]
    if n_rows == 0:
        return

    # Kernel passes a single ``chunk_stride_0`` and uses it for BOTH
    # chunk_k and chunk_score loads; make both contiguous so the shared
    # stride is correct even if a caller hands a view-of-permute in.
    chunk_k = chunk_k.contiguous()
    chunk_score = chunk_score.contiguous()
    _scatter_kpool_tail_updates_kernel[(n_rows, pool_size)](
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        req_pool_idx,
        dst_logical_start,
        chunk_src_start,
        n_write,
        chunk_k.stride(0),
        tail_k.stride(0),
        tail_k.stride(1),
        POOL_SIZE=pool_size,
        TAIL_SIZE=tail_k.shape[1],
        HEAD_DIM=INDEX_HEAD_DIM,
        BLOCK_D=triton.next_power_of_2(INDEX_HEAD_DIM),
    )


# GLM NOTE: the per-row metadata tensors are H2D-channel slices with
# batch-dependent alignment; see _kpool_build_ragged_layout_kernel.
@triton.jit(
    do_not_specialize_on_alignment=[
        "req_pool_idx_ptr",
        "dst_logical_start_ptr",
        "chunk_src_start_ptr",
        "n_write_ptr",
    ],
)
def _scatter_kpool_tail_updates_kernel(
    chunk_k_ptr,
    chunk_score_ptr,
    tail_k_ptr,
    tail_score_ptr,
    req_pool_idx_ptr,
    dst_logical_start_ptr,
    chunk_src_start_ptr,
    n_write_ptr,
    chunk_stride_0,
    tail_stride_0,
    tail_stride_1,
    POOL_SIZE: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.program_id(1)

    n_w = tl.load(n_write_ptr + row)
    if slot >= n_w:
        return

    req = tl.load(req_pool_idx_ptr + row)
    dst_logical_start = tl.load(dst_logical_start_ptr + row)
    src_off = tl.load(chunk_src_start_ptr + row) + slot

    offs = tl.arange(0, BLOCK_D)
    mask = offs < HEAD_DIM
    k = tl.load(chunk_k_ptr + src_off * chunk_stride_0 + offs, mask=mask)
    s = tl.load(chunk_score_ptr + src_off * chunk_stride_0 + offs, mask=mask)

    dst = (
        req * tail_stride_0
        + ((dst_logical_start + slot) % TAIL_SIZE) * tail_stride_1
        + offs
    )
    tl.store(tail_k_ptr + dst, k, mask=mask)
    tl.store(tail_score_ptr + dst, s, mask=mask)


@triton.jit
def _pack_pool_slots_to_payload_kernel(
    buf_ptr,  # uint8 [num_pages, page_bytes]
    locs_ptr,  # int64 [N]
    payload_ptr,  # uint8 [N, payload_bytes]
    payload_bytes: tl.constexpr,
    slots_per_page: tl.constexpr,
    head_dim: tl.constexpr,
    page_bytes: tl.constexpr,  # slots_per_page * head_dim + slots_per_page * 4
    scale_region_off: tl.constexpr,  # slots_per_page * head_dim
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    loc = tl.load(locs_ptr + row).to(tl.int64)
    page = loc // slots_per_page
    slot = loc % slots_per_page
    page_base = page * page_bytes

    # Copy head_dim fp8 bytes.
    offs = tl.arange(0, BLOCK_D)
    mask = offs < head_dim
    src = page_base + slot * head_dim + offs
    val = tl.load(buf_ptr + src, mask=mask, other=0).to(tl.uint8)
    tl.store(payload_ptr + row * payload_bytes + offs, val, mask=mask)

    # Copy 4 fp32-scale bytes (positions head_dim .. head_dim+4).
    s_offs = tl.arange(0, 4)
    s_src = page_base + scale_region_off + slot * 4 + s_offs
    s_val = tl.load(buf_ptr + s_src).to(tl.uint8)
    tl.store(payload_ptr + row * payload_bytes + head_dim + s_offs, s_val)


@triton.jit
def _select_and_scatter_pool_slots_kernel(
    recv_ptr,  # uint8 [cp_size, N, payload_bytes]
    owner_ptr,  # int32 [N]
    locs_ptr,  # int64 [N]
    buf_ptr,  # uint8 [num_pages, page_bytes]
    cp_rank: tl.constexpr,
    payload_bytes: tl.constexpr,
    slots_per_page: tl.constexpr,
    head_dim: tl.constexpr,
    page_bytes: tl.constexpr,
    scale_region_off: tl.constexpr,
    n_total: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    owner = tl.load(owner_ptr + row).to(tl.int64)
    # Self-owned rows were already written by this rank's compress kernel
    # (the local writes that fed pack_pool_slots). Skipping them avoids a
    # redundant recv -> buf copy of bytes we already have correct in buf.
    if owner == cp_rank:
        return
    loc = tl.load(locs_ptr + row).to(tl.int64)
    page = loc // slots_per_page
    slot = loc % slots_per_page
    page_base = page * page_bytes

    recv_row_base = (owner * n_total + row) * payload_bytes

    # Scatter head_dim fp8 bytes back into the fp8 region.
    offs = tl.arange(0, BLOCK_D)
    mask = offs < head_dim
    val = tl.load(recv_ptr + recv_row_base + offs, mask=mask, other=0).to(tl.uint8)
    dst = page_base + slot * head_dim + offs
    tl.store(buf_ptr + dst, val, mask=mask)

    # Scatter 4 fp32-scale bytes back into the scale region.
    s_offs = tl.arange(0, 4)
    s_val = tl.load(recv_ptr + recv_row_base + head_dim + s_offs).to(tl.uint8)
    s_dst = page_base + scale_region_off + slot * 4 + s_offs
    tl.store(buf_ptr + s_dst, s_val)


def all_gather_and_scatter_pool_slots(
    buf: torch.Tensor,
    local_locs: torch.Tensor,
    owner_rank: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    slots_per_page: int,
) -> None:
    """Replicate this layer's freshly written pool slots across all CP ranks.

    Each rank wrote a subset of N total pool slots (FP8 key + fp32 scale)
    at physical locations ``local_locs[owner_rank == cp_rank]`` into its
    own copy of ``buf``. To make every rank see the same post-write cache
    state, we pack each slot's (fp8 key + fp32 scale) into a contiguous
    payload, all-gather across ranks, then scatter the owner's payload
    for each row back into buf.

    Layout of ``buf`` (per page, see DSATokenToKVPool init):
        buf[p, 0 : slots_per_page * head_dim]                  uint8 fp8 keys
        buf[p, slots_per_page * head_dim : ].view(fp32)        fp32 scales

    Two triton kernels (pack and select-scatter) replace four advanced-
    indexing launches and avoid the (cp_size, N, payload) intermediate
    contiguous gather buffer copy.

    buf:               (num_pages, slots_per_page * head_dim + slots_per_page * 4) uint8
    local_locs:        (N,) int64. Physical slot locations for all pool
                       rows in the plan (identical on every rank).
    owner_rank:        (N,) int32. Which CP rank owns each row (identical
                       on every rank). Derived from a row-index-mod-cp_size
                       scheme in ``_kpool_cp_owner_rank``.
    cp_size:           Number of CP ranks.
    """
    from sglang.srt.layers.dp_attention import attn_cp_all_gather_into_tensor

    # Contiguity is silent-corruption territory; other shape/dtype guards
    # are upstream guarantees in CP setup.
    assert buf.is_contiguous()

    n_total = local_locs.shape[0]
    if n_total == 0 or cp_size <= 1:
        return

    head_dim = INDEX_HEAD_DIM
    payload_bytes = head_dim + 4
    scale_region_off = slots_per_page * head_dim
    page_bytes = buf.shape[1]
    device = buf.device

    send_payload = torch.empty(
        (n_total, payload_bytes), dtype=torch.uint8, device=device
    )
    _pack_pool_slots_to_payload_kernel[(n_total,)](
        buf,
        local_locs,
        send_payload,
        payload_bytes=payload_bytes,
        slots_per_page=slots_per_page,
        head_dim=head_dim,
        page_bytes=page_bytes,
        scale_region_off=scale_region_off,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )

    recv = torch.empty(
        (cp_size, n_total, payload_bytes), dtype=torch.uint8, device=device
    )
    attn_cp_all_gather_into_tensor(
        recv.view(cp_size * n_total, payload_bytes), send_payload
    )

    _select_and_scatter_pool_slots_kernel[(n_total,)](
        recv,
        owner_rank,
        local_locs,
        buf,
        cp_rank=cp_rank,
        payload_bytes=payload_bytes,
        slots_per_page=slots_per_page,
        head_dim=head_dim,
        page_bytes=page_bytes,
        scale_region_off=scale_region_off,
        n_total=n_total,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )


# ---------------------------------------------------------------------------
# target_verify: draft-tail write + closed-pool compress (tail-only) + commit
# ---------------------------------------------------------------------------


@triton.jit
def _kpool_write_tail_and_maybe_compress_kernel(
    # Inputs.
    key_ptr,  # bf16 [B*N, head_dim]
    score_ptr,  # bf16 [B*N, head_dim]
    tail_k_ptr,  # bf16 [req_pool, TAIL_SIZE, head_dim]
    tail_score_ptr,  # bf16 same shape
    ape_ptr,  # fp32 [POOL_SIZE, head_dim]
    # Plan tensors from update_kpool_write_plan_cuda_graph.
    req_pool_indices_ptr,  # int64 [B]
    write_start_ptr,  # int32 [B] -- decode: positions[b]; verify: committed[b]
    tail_logical_start_ptr,  # int32 [B * MAX_CLOSED_POOLS]
    write_loc_ptr,  # int64 [B * MAX_CLOSED_POOLS]
    # Padding sentinel: out_cache_loc[b*N] == 0 means batch b is a
    # cuda-graph padded slot (the cache allocator reserves slot 0 as a
    # dummy sink, see DSATokenToKVPool init -- "padded slot 0 is used for
    # writing dummy outputs from padded tokens"). Without this gate,
    # padded batches would write garbage drafts into tail[req=0, ...] and
    # poison req 0's tail ring across the whole forward.
    out_cache_loc_ptr,  # int64 [B*N]
    # V2-only: effective_n_per_batch[b] = accept_length[b] (which v2
    # emits already including the bonus "next" token; the real advance),
    # used to gate compress to the REAL advance (vs the full N drafts).
    # nullptr -> ``HAS_EFFECTIVE_N=False`` and the kernel falls back to N
    # as the gate window (verify / decode behavior).
    effective_n_ptr,  # int32 [B] or null
    # Compress sink.
    buf_i8_ptr,
    buf_fp32_ptr,
    # Strides.
    key_stride_0,
    score_stride_0,
    tail_stride_0,
    tail_stride_1,
    ape_stride_0,
    # Constexprs.
    N: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SLOTS_PER_PAGE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    HAS_EFFECTIVE_N: tl.constexpr,
    MAX_CLOSED_POOLS: tl.constexpr,
):
    """Per-batch program: write N tokens to tail ring then compress closed pools.

    Pool-close gating: ``(write_start + N) // POOL_SIZE > write_start // POOL_SIZE``.
    Under EAGLE topk=1 chain-only, each batch closes up to
    ``ceil(N / POOL_SIZE)`` pools. Padded batches (``out_cache_loc[b*N] == 0``,
    the reserved sink) early-return to avoid poisoning req=0's tail.
    """
    b = tl.program_id(0)
    # Padded-batch gate: any draft of this batch landing at the reserved
    # sink slot 0 means the whole batch is padding. Skip both write and
    # compress so req=0's real tail data stays untouched.
    cache_loc_0 = tl.load(out_cache_loc_ptr + b * N)
    if cache_loc_0 == 0:
        return

    req = tl.load(req_pool_indices_ptr + b)
    write_start = tl.load(write_start_ptr + b)

    offs = tl.arange(0, BLOCK_D)
    dim_mask = offs < HEAD_DIM

    # ----- Step 1: write N tokens into the tail ring. -----
    for i_n in tl.static_range(0, N):
        row = b * N + i_n
        k = tl.load(key_ptr + row * key_stride_0 + offs, mask=dim_mask)
        s = tl.load(score_ptr + row * score_stride_0 + offs, mask=dim_mask)
        phys = (write_start + i_n) % TAIL_SIZE
        dst = req * tail_stride_0 + phys * tail_stride_1 + offs
        tl.store(tail_k_ptr + dst, k, mask=dim_mask)
        tl.store(tail_score_ptr + dst, s, mask=dim_mask)

    # ----- Step 2: if a pool closed, compress it from the tail. -----
    # gate_n: window we treat as "committed" for compress purposes.
    #   verify/decode: full N drafts ("assume all accepted").
    #   v2: accept_length (already includes the bonus token; the REAL
    #       advance). Rejected drafts past the accept frontier get their
    #       tail K overwritten by the next round's V2 write before any
    #       reader needs them, so deferring compress is safe and uses
    #       real (not speculative) K when the pool finally crosses.
    if HAS_EFFECTIVE_N:
        gate_n = tl.load(effective_n_ptr + b).to(tl.int32)
    else:
        gate_n = N
    gate_n = tl.minimum(tl.maximum(gate_n, 0), N)
    base_pool = write_start // POOL_SIZE
    n_pool = (write_start + gate_n) // POOL_SIZE - base_pool
    if n_pool == 0:
        return

    for i_pool in tl.static_range(0, MAX_CLOSED_POOLS):
        if i_pool < n_pool:
            plan_row = b * MAX_CLOSED_POOLS + i_pool
            base = tl.load(tail_logical_start_ptr + plan_row)
            m = tl.full((BLOCK_D,), -float("inf"), tl.float32)
            acc = tl.full((BLOCK_D,), 0.0, tl.float32)
            denom = tl.full((BLOCK_D,), 0.0, tl.float32)
            for slot in tl.static_range(0, POOL_SIZE):
                phys = (base + slot) % TAIL_SIZE
                off = req * tail_stride_0 + phys * tail_stride_1 + offs
                score = tl.load(tail_score_ptr + off, mask=dim_mask, other=0.0).to(
                    tl.float32
                )
                k_ld = tl.load(tail_k_ptr + off, mask=dim_mask, other=0.0).to(
                    tl.float32
                )
                score += tl.load(
                    ape_ptr + slot * ape_stride_0 + offs, mask=dim_mask, other=0.0
                ).to(tl.float32)
                new_m = tl.maximum(m, score)
                rescale = tl.exp(m - new_m)
                prob = tl.exp(score - new_m)
                denom = denom * rescale + prob
                acc = acc * rescale + k_ld * prob
                m = new_m

            quantized, scale = _hadamard_quantize_int8(acc, denom, ROUND_SCALE)
            loc = tl.load(write_loc_ptr + plan_row)
            loc_page_index = loc // SLOTS_PER_PAGE
            loc_token_offset_in_page = loc % SLOTS_PER_PAGE
            out_k_offsets = (
                loc_page_index * BUF_NUMEL_PER_PAGE
                + loc_token_offset_in_page * HEAD_DIM
                + offs
            )
            out_s_offset = (
                loc_page_index * BUF_NUMEL_PER_PAGE // 4
                + S_OFFSET_NBYTES_IN_PAGE // 4
                + loc_token_offset_in_page
            )
            tl.store(buf_i8_ptr + out_k_offsets, quantized, mask=dim_mask)
            tl.store(buf_fp32_ptr + out_s_offset, scale)


def kpool_write_tail_and_maybe_compress(
    pool,
    buf: torch.Tensor,
    key: torch.Tensor,
    score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    ape: torch.Tensor,
    req_pool_indices: torch.Tensor,
    write_start: torch.Tensor,
    tail_logical_start: torch.Tensor,
    write_loc: torch.Tensor,
    out_cache_loc: torch.Tensor,
    num_draft_tokens: int,
    round_scale: bool,
    effective_n_per_batch: Optional[torch.Tensor] = None,
) -> None:
    """Fused kpool tail-write + closed-pool compress.

    Unified entry point for decode (``num_draft_tokens=1``),
    target_verify and draft_extend_v2 (``num_draft_tokens=N``). Per-batch
    program: write N tokens to tail then compress a closed pool if the
    "real advance" window crossed a pool boundary.

    ``effective_n_per_batch`` (v2 only): int32 [B], the per-batch real
    advance ?for v2 this is ``spec_info.accept_length``, which v2 emits
    already including the bonus "next" token. When provided, compress is
    gated on ``(write_start + effective_n[b]) // P > write_start // P``;
    when None the gate uses full N (verify "assume all accepted" semantics).

    ``write_start``: decode = positions[b] (= seq_lens[b] - 1);
    verify = committed_seq_lens[b] (= seq_lens[b]);
    v2 = seq_lens[b] - N (= committed_after_verify).

    Cuda-graph padding: ``out_cache_loc[b*N] == 0`` flags padded batches
    (cache slot 0 is the reserved sink); the kernel skips both write and
    compress so padded batches don't poison req=0's tail ring.
    """
    _log_kpool_wrapper_call(
        "kpool_write_tail_and_maybe_compress",
        pool=pool,
        buf=buf,
        key=key,
        score=score,
        tail_k=tail_k,
        tail_score=tail_score,
        ape=ape,
        req_pool_indices=req_pool_indices,
        write_start=write_start,
        tail_logical_start=tail_logical_start,
        write_loc=write_loc,
        out_cache_loc=out_cache_loc,
        num_draft_tokens=num_draft_tokens,
        round_scale=round_scale,
        effective_n_per_batch=effective_n_per_batch,
    )
    bn = key.shape[0]
    if bn == 0:
        return
    bs = bn // num_draft_tokens

    # Plan tensors (req_pool_indices / write_start / tail_logical_start /
    # write_loc) and out_cache_loc are always allocated contiguous; only
    # key/score may be views from the caller, so guard those.
    key = key.contiguous()
    score = score.contiguous()
    slots_per_page = pool.slots_per_page
    buf_i8 = buf.view(torch.int8)
    buf_fp32 = buf.view(torch.float32)

    _kpool_write_tail_and_maybe_compress_kernel[(bs,)](
        key,
        score,
        tail_k,
        tail_score,
        ape,
        req_pool_indices,
        write_start,
        tail_logical_start,
        write_loc,
        out_cache_loc,
        effective_n_per_batch,
        buf_i8,
        buf_fp32,
        key.stride(0),
        score.stride(0),
        tail_k.stride(0),
        tail_k.stride(1),
        ape.stride(0),
        N=num_draft_tokens,
        POOL_SIZE=pool.index_kpool,
        TAIL_SIZE=tail_k.shape[1],
        HEAD_DIM=INDEX_HEAD_DIM,
        BLOCK_D=triton.next_power_of_2(INDEX_HEAD_DIM),
        SLOTS_PER_PAGE=slots_per_page,
        BUF_NUMEL_PER_PAGE=buf.shape[1],
        S_OFFSET_NBYTES_IN_PAGE=slots_per_page * INDEX_HEAD_DIM,
        ROUND_SCALE=round_scale,
        HAS_EFFECTIVE_N=effective_n_per_batch is not None,
        MAX_CLOSED_POOLS=max(1, write_loc.numel() // bs),
    )
