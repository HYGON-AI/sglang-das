# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

fp8_dtype = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn

# v4 KV cache layout (see dsv4.index_buf_accessor._set_k_and_s_triton_kernel):
#   per-token: 448 fp8 nope + 64 bf16 rope (= 576 contiguous bytes) +
#              7 ue8m0 scales padded to 8 bytes.
#   per-page:  [token 0..P-1 nope+rope (P*576 bytes)] [token 0..P-1 scale (P*8 bytes)]
#              padded up to a multiple of 576.
DIM_NOPE = 448
DIM_ROPE = 64
TILE_SIZE = 64  # one nope scale tile = 64 fp8 values
NUM_SCALE_TILES = DIM_NOPE // TILE_SIZE  # 7
NOPE_ROPE_BYTES = DIM_NOPE + DIM_ROPE * 2  # 576
PADDED_SCALE_PER_TOKEN = NUM_SCALE_TILES + 1  # 8


def dequantize_k_cache_paged(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    page_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dequantize the DeepSeek v4 paged KV cache for a list of token IDs.

    Args:
        quant_k_cache: (num_pages, bytes_per_page_padded) uint8.
        page_table_1_flattened: (num_tokens,) int — token IDs into the cache.
        page_size: number of tokens per page.
        out: optional (num_tokens, 1, DIM_NOPE + DIM_ROPE) bf16 destination.
            May be a slice of a larger workspace; the kernel uses out.stride(0)
            so contiguous-along-dim-0 slices work.

    Returns:
        (num_tokens, 1, DIM_NOPE + DIM_ROPE) bfloat16.
    """
    assert quant_k_cache.is_contiguous()
    assert page_table_1_flattened.dtype in (torch.int32, torch.int64)

    # The buffer's dtype is whatever the pool exposes (often bf16); the
    # underlying storage is uint8. Reinterpret to byte-space first.
    quant_k_cache_u8 = quant_k_cache.view(torch.uint8)
    num_tokens = page_table_1_flattened.shape[0]
    bytes_per_page = quant_k_cache_u8.shape[-1]
    s_offset_bytes = page_size * NOPE_ROPE_BYTES

    # Three typed views over the same underlying bytes.
    buf_fp8 = quant_k_cache_u8.view(fp8_dtype).reshape(-1)
    buf_bf16 = quant_k_cache_u8.view(torch.bfloat16).reshape(-1)
    buf_uint8 = quant_k_cache_u8.reshape(-1)

    if out is None:
        out = torch.empty(
            (num_tokens, 1, DIM_NOPE + DIM_ROPE),
            dtype=torch.bfloat16,
            device=quant_k_cache.device,
        )
    else:
        assert out.shape == (num_tokens, 1, DIM_NOPE + DIM_ROPE)
        assert out.dtype == torch.bfloat16

    _dequantize_k_cache_paged_kernel[(num_tokens,)](
        out,
        buf_fp8,
        buf_bf16,
        buf_uint8,
        page_table_1_flattened,
        out.stride(0),
        BYTES_PER_PAGE=bytes_per_page,
        PAGE_SIZE=page_size,
        DIM_NOPE=DIM_NOPE,
        DIM_ROPE=DIM_ROPE,
        TILE_SIZE=TILE_SIZE,
        NUM_SCALE_TILES=NUM_SCALE_TILES,
        NOPE_ROPE_BYTES=NOPE_ROPE_BYTES,
        PADDED_SCALE_PER_TOKEN=PADDED_SCALE_PER_TOKEN,
        S_OFFSET_BYTES=s_offset_bytes,
    )
    return out


def gather_dequant_requant_fp8_paged(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    page_size: int,
    extra_rows: int = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gather DeepSeek-V4 paged KV cache into a flat FP8 workspace.

    This is the Q8KV8 sparse-prefill adapter for the DeepSeek-V4 packed layout.
    It gathers token IDs from the existing paged cache, dequantizes the 448-dim
    nope region with its UE8M0 per-64 scales, casts the 64-dim BF16 rope tail to
    FP8, and writes the result as ``(num_tokens + extra_rows, 1, 512)`` FP8.

    ``extra_rows`` appends zero rows for kernels that map masked sparse indices
    to a valid zero landing pad.
    """
    assert quant_k_cache.is_contiguous()
    assert page_table_1_flattened.dtype in (torch.int32, torch.int64)
    assert extra_rows >= 0

    quant_k_cache_u8 = quant_k_cache.view(torch.uint8)
    num_tokens = page_table_1_flattened.shape[0]
    total_rows = num_tokens + extra_rows
    bytes_per_page = quant_k_cache_u8.shape[-1]
    s_offset_bytes = page_size * NOPE_ROPE_BYTES

    buf_fp8 = quant_k_cache_u8.view(fp8_dtype).reshape(-1)
    buf_bf16 = quant_k_cache_u8.view(torch.bfloat16).reshape(-1)
    buf_uint8 = quant_k_cache_u8.reshape(-1)

    if out is None:
        out = torch.zeros(
            (total_rows, 1, DIM_NOPE + DIM_ROPE),
            dtype=fp8_dtype,
            device=quant_k_cache.device,
        )
    else:
        assert out.shape == (total_rows, 1, DIM_NOPE + DIM_ROPE)
        assert out.dtype == fp8_dtype
        if extra_rows:
            out[num_tokens:].zero_()

    if num_tokens == 0:
        return out

    _gather_dequant_requant_fp8_paged_kernel[(num_tokens,)](
        out,
        buf_fp8,
        buf_bf16,
        buf_uint8,
        page_table_1_flattened,
        out.stride(0),
        BYTES_PER_PAGE=bytes_per_page,
        PAGE_SIZE=page_size,
        DIM_NOPE=DIM_NOPE,
        DIM_ROPE=DIM_ROPE,
        TILE_SIZE=TILE_SIZE,
        NUM_SCALE_TILES=NUM_SCALE_TILES,
        NOPE_ROPE_BYTES=NOPE_ROPE_BYTES,
        PADDED_SCALE_PER_TOKEN=PADDED_SCALE_PER_TOKEN,
        S_OFFSET_BYTES=s_offset_bytes,
    )
    return out


def gather_upconvert_k_cache_paged(
    quant_k_cache: torch.Tensor,
    token_indices: torch.Tensor,
    topk_lengths: torch.Tensor,
    page_size: int,
    out: Optional[torch.Tensor] = None,
    compact_indices: Optional[torch.Tensor] = None,
    output_offsets: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather packed DSV4 FP8 KV rows and up-convert them to BF16.

    This is the Triton equivalent of the LightOp decode gather used by SparkV1.
    Each query gets a private, contiguous page in ``out``.  The returned indices
    address that compact cache and can therefore be passed directly to the
    existing BF16 ``flash_mla_with_kvcache`` sparse-decode path.

    Args:
        quant_k_cache: Packed DSV4 cache. Accepted layouts are the raw
            ``(num_pages, bytes_per_page_padded)`` allocation and its FlashMLA
            view ``(num_pages, page_size, 1, 584)``. Page padding is preserved
            through ``stride(0)``.
        token_indices: ``(num_queries, topk)`` or ``(num_queries, 1, topk)``
            physical token locations in ``quant_k_cache``.
        topk_lengths: ``(num_queries,)`` valid prefix lengths.
        page_size: source cache page size.
        out: optional BF16 ``(num_queries, output_topk, 1, 512)`` workspace.
            ``output_topk`` may be larger than the source ``topk`` when two
            caches are packed into one FlashMLA input.
        compact_indices: optional int32 ``(num_queries, 1, output_topk)``
            workspace.
        output_offsets: optional int32 ``(num_queries,)`` destination column
            for each query. This lets a second cache start immediately after
            the first cache's valid prefix instead of after its padded width.
    """
    assert quant_k_cache.dim() in (2, 4), f"unexpected {quant_k_cache.shape=}"
    assert quant_k_cache.is_cuda
    assert token_indices.dtype in (torch.int32, torch.int64)
    if token_indices.dim() == 3:
        assert token_indices.shape[1] == 1
        token_indices_2d = token_indices[:, 0, :]
    else:
        assert token_indices.dim() == 2
        token_indices_2d = token_indices
    token_indices_2d = token_indices_2d.contiguous()

    topk_lengths = topk_lengths.reshape(-1).contiguous()
    assert topk_lengths.dtype == torch.int32
    num_queries, topk = token_indices_2d.shape
    assert topk_lengths.numel() == num_queries
    assert page_size > 0

    quant_k_cache_u8 = quant_k_cache.view(torch.uint8)
    num_pages = quant_k_cache_u8.shape[0]
    page_stride_bytes = quant_k_cache_u8.stride(0)
    assert page_stride_bytes >= page_size * (
        NOPE_ROPE_BYTES + PADDED_SCALE_PER_TOKEN
    )

    if out is None:
        output_topk = topk
        out = torch.empty(
            (num_queries, output_topk, 1, DIM_NOPE + DIM_ROPE),
            dtype=torch.bfloat16,
            device=quant_k_cache.device,
        )
    else:
        assert out.shape[0] == num_queries
        assert out.shape[2:] == (1, DIM_NOPE + DIM_ROPE)
        assert out.dtype == torch.bfloat16
        output_topk = out.shape[1]
        assert output_topk >= topk

    if compact_indices is None:
        compact_indices = torch.empty(
            (num_queries, 1, output_topk),
            dtype=torch.int32,
            device=token_indices.device,
        )
    else:
        assert compact_indices.shape == (num_queries, 1, output_topk)
        assert compact_indices.dtype == torch.int32

    use_output_offsets = output_offsets is not None
    if output_offsets is None:
        # The kernel ignores this pointer when USE_OUTPUT_OFFSETS=False.
        output_offsets = topk_lengths
    else:
        output_offsets = output_offsets.reshape(-1).contiguous()
        assert output_offsets.dtype == torch.int32
        assert output_offsets.numel() == num_queries

    if num_queries == 0 or topk == 0:
        return out, compact_indices

    # Keep the original page stride.  The 4-D FlashMLA view excludes the
    # per-page padding from its logical shape and is therefore non-contiguous;
    # reshape() would silently copy it and invalidate PAGE_STRIDE_BYTES.
    buf_fp8 = quant_k_cache_u8.view(fp8_dtype)
    buf_bf16 = quant_k_cache_u8.view(torch.bfloat16)
    buf_uint8 = quant_k_cache_u8
    _gather_upconvert_k_cache_paged_kernel[(num_queries, topk)](
        out,
        compact_indices,
        buf_fp8,
        buf_bf16,
        buf_uint8,
        token_indices_2d,
        topk_lengths,
        output_offsets,
        out.stride(0),
        out.stride(1),
        compact_indices.stride(0),
        compact_indices.stride(2),
        token_indices_2d.stride(0),
        token_indices_2d.stride(1),
        PAGE_STRIDE_BYTES=page_stride_bytes,
        PAGE_SIZE=page_size,
        NUM_PAGES=num_pages,
        OUTPUT_TOPK=output_topk,
        USE_OUTPUT_OFFSETS=use_output_offsets,
        DIM_NOPE=DIM_NOPE,
        DIM_ROPE=DIM_ROPE,
        TILE_SIZE=TILE_SIZE,
        NUM_SCALE_TILES=NUM_SCALE_TILES,
        NOPE_ROPE_BYTES=NOPE_ROPE_BYTES,
        PADDED_SCALE_PER_TOKEN=PADDED_SCALE_PER_TOKEN,
        S_OFFSET_BYTES=page_size * NOPE_ROPE_BYTES,
    )
    return out, compact_indices


def q8kv8_padded_num_heads(num_heads: int) -> int:
    """Return a Q-head count supported by the SM90 Q8KV8 kernel."""
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if num_heads <= 64:
        return 64
    if num_heads <= 128:
        return 128
    raise ValueError(
        "DeepSeek-V4 Q8KV8 sparse prefill supports at most 128 local "
        f"query heads, got {num_heads}"
    )


def cast_q_fp8_for_q8kv8_prefill(
    q: torch.Tensor,
    padded_num_heads: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast DeepSeek-V4 sparse-prefill Q to the Q8KV8 kernel format.

    The incoming Q is the model-produced BF16/FP16 tensor already shaped as
    ``(num_tokens, num_heads, 512)`` after removing the singleton MQA axis.

    The SM90 kernel processes query heads in 64-head blocks. Tensor parallelism
    commonly leaves fewer than 64 local heads, so the active heads are copied
    into a zero-padded 64/128-head FP8 tensor.
    """
    assert q.ndim == 3
    assert q.shape[-1] == DIM_NOPE + DIM_ROPE

    num_tokens, num_heads, head_dim = q.shape
    if padded_num_heads is None:
        padded_num_heads = q8kv8_padded_num_heads(num_heads)

    if padded_num_heads not in (64, 128) or padded_num_heads < num_heads:
        raise ValueError(
            f"invalid padded_num_heads={padded_num_heads} for num_heads={num_heads}"
        )

    expected_shape = (num_tokens, padded_num_heads, head_dim)

    if out is None:
        q_fp8 = torch.zeros(
            expected_shape,
            dtype=fp8_dtype,
            device=q.device,
        )
    else:
        if (
            out.shape != expected_shape
            or out.dtype != fp8_dtype
            or out.device != q.device
        ):
            raise ValueError(
                "Q8KV8 Q output must have shape/dtype/device "
                f"{expected_shape}/{fp8_dtype}/{q.device}, got "
                f"{tuple(out.shape)}/{out.dtype}/{out.device}"
            )
        q_fp8 = out
        if padded_num_heads > num_heads:
            q_fp8[:, num_heads:].zero_()

    q_fp8[:, :num_heads].copy_(q)
    q_scale = torch.ones((), dtype=torch.float32, device=q.device)
    return q_fp8, q_scale


@triton.jit
def _dequantize_k_cache_paged_kernel(
    output_ptr,
    buf_fp8_ptr,
    buf_bf16_ptr,
    buf_uint8_ptr,
    page_table_ptr,
    output_stride_0,
    BYTES_PER_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_SCALE_TILES: tl.constexpr,
    NOPE_ROPE_BYTES: tl.constexpr,
    PADDED_SCALE_PER_TOKEN: tl.constexpr,
    S_OFFSET_BYTES: tl.constexpr,
):
    # One program per token: load page_table[token_id] once and emit all
    # NUM_SCALE_TILES nope tiles + rope tail via tl.static_range.
    token_id = tl.program_id(0)
    loc = tl.load(page_table_ptr + token_id).to(tl.int64)
    page_idx = loc // PAGE_SIZE
    in_page = loc % PAGE_SIZE
    page_byte_base = page_idx * BYTES_PER_PAGE
    token_data_base = page_byte_base + in_page * NOPE_ROPE_BYTES
    token_scale_base = (
        page_byte_base + S_OFFSET_BYTES + in_page * PADDED_SCALE_PER_TOKEN
    )
    out_row_base = token_id * output_stride_0

    nope_offs = tl.arange(0, TILE_SIZE)
    for tile_id in tl.static_range(NUM_SCALE_TILES):
        fp8_off = token_data_base + tile_id * TILE_SIZE + nope_offs
        fp8_vals = tl.load(buf_fp8_ptr + fp8_off).to(tl.float32)

        scale_u8 = tl.load(buf_uint8_ptr + token_scale_base + tile_id).to(tl.int32)
        scale_pow2 = tl.exp2((scale_u8 - 127).to(tl.float32))

        out_off = out_row_base + tile_id * TILE_SIZE + nope_offs
        tl.store(
            output_ptr + out_off,
            (fp8_vals * scale_pow2).to(output_ptr.dtype.element_ty),
        )

    rope_offs = tl.arange(0, DIM_ROPE)
    bf16_off = (token_data_base + DIM_NOPE) // 2 + rope_offs
    rope_data = tl.load(buf_bf16_ptr + bf16_off)
    tl.store(output_ptr + out_row_base + DIM_NOPE + rope_offs, rope_data)


@triton.jit
def _gather_dequant_requant_fp8_paged_kernel(
    output_ptr,
    buf_fp8_ptr,
    buf_bf16_ptr,
    buf_uint8_ptr,
    page_table_ptr,
    output_stride_0,
    BYTES_PER_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_SCALE_TILES: tl.constexpr,
    NOPE_ROPE_BYTES: tl.constexpr,
    PADDED_SCALE_PER_TOKEN: tl.constexpr,
    S_OFFSET_BYTES: tl.constexpr,
):
    token_id = tl.program_id(0)
    loc = tl.load(page_table_ptr + token_id).to(tl.int64)
    page_idx = loc // PAGE_SIZE
    in_page = loc % PAGE_SIZE
    page_byte_base = page_idx * BYTES_PER_PAGE
    token_data_base = page_byte_base + in_page * NOPE_ROPE_BYTES
    token_scale_base = (
        page_byte_base + S_OFFSET_BYTES + in_page * PADDED_SCALE_PER_TOKEN
    )
    out_row_base = token_id * output_stride_0

    nope_offs = tl.arange(0, TILE_SIZE)
    for tile_id in tl.static_range(NUM_SCALE_TILES):
        fp8_off = token_data_base + tile_id * TILE_SIZE + nope_offs
        fp8_vals = tl.load(buf_fp8_ptr + fp8_off).to(tl.float32)

        scale_u8 = tl.load(buf_uint8_ptr + token_scale_base + tile_id).to(tl.int32)
        scale_pow2 = tl.exp2((scale_u8 - 127).to(tl.float32))

        out_off = out_row_base + tile_id * TILE_SIZE + nope_offs
        tl.store(
            output_ptr + out_off,
            (fp8_vals * scale_pow2).to(output_ptr.dtype.element_ty),
        )

    rope_offs = tl.arange(0, DIM_ROPE)
    bf16_off = (token_data_base + DIM_NOPE) // 2 + rope_offs
    rope_data = tl.load(buf_bf16_ptr + bf16_off)
    tl.store(
        output_ptr + out_row_base + DIM_NOPE + rope_offs,
        rope_data.to(output_ptr.dtype.element_ty),
    )


@triton.jit
def _gather_upconvert_k_cache_paged_kernel(
    output_ptr,
    compact_indices_ptr,
    buf_fp8_ptr,
    buf_bf16_ptr,
    buf_uint8_ptr,
    token_indices_ptr,
    topk_lengths_ptr,
    output_offsets_ptr,
    output_stride_0,
    output_stride_1,
    compact_indices_stride_0,
    compact_indices_stride_2,
    indices_stride_0,
    indices_stride_1,
    PAGE_STRIDE_BYTES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_PAGES: tl.constexpr,
    OUTPUT_TOPK: tl.constexpr,
    USE_OUTPUT_OFFSETS: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_SCALE_TILES: tl.constexpr,
    NOPE_ROPE_BYTES: tl.constexpr,
    PADDED_SCALE_PER_TOKEN: tl.constexpr,
    S_OFFSET_BYTES: tl.constexpr,
):
    # HCU Triton otherwise keeps program IDs and runtime strides in i32.  For
    # the full CP-local prefill batch, query_id * output_stride_0 can exceed
    # 2**31 elements (notably the combined SWA+C128 BF16 workspace), wrapping
    # the destination address and causing an HSA VMFault.  Promote every index
    # participating in pointer arithmetic before the multiplication.
    query_id = tl.program_id(0).to(tl.int64)
    topk_id = tl.program_id(1).to(tl.int64)
    valid_length = tl.load(topk_lengths_ptr + query_id).to(tl.int64)
    output_offset = query_id * 0
    if USE_OUTPUT_OFFSETS:
        output_offset = tl.load(output_offsets_ptr + query_id).to(tl.int64)
    output_col = output_offset + topk_id
    in_valid_prefix = topk_id < valid_length
    loc = tl.load(
        token_indices_ptr
        + query_id * indices_stride_0
        + topk_id * indices_stride_1,
        mask=in_valid_prefix,
        other=0,
    ).to(tl.int64)
    valid = in_valid_prefix & (loc >= 0) & (loc < NUM_PAGES * PAGE_SIZE)
    safe_loc = tl.where(valid, loc, 0)
    page_idx = safe_loc // PAGE_SIZE
    in_page = safe_loc % PAGE_SIZE
    page_byte_base = page_idx * PAGE_STRIDE_BYTES
    token_data_base = page_byte_base + in_page * NOPE_ROPE_BYTES
    token_scale_base = (
        page_byte_base + S_OFFSET_BYTES + in_page * PADDED_SCALE_PER_TOKEN
    )
    out_row_base = query_id * output_stride_0 + output_col * output_stride_1

    nope_offs = tl.arange(0, TILE_SIZE)
    for tile_id in tl.static_range(NUM_SCALE_TILES):
        fp8_off = token_data_base + tile_id * TILE_SIZE + nope_offs
        fp8_vals = tl.load(buf_fp8_ptr + fp8_off, mask=valid, other=0.0).to(
            tl.float32
        )
        scale_u8 = tl.load(
            buf_uint8_ptr + token_scale_base + tile_id,
            mask=valid,
            other=127,
        ).to(tl.int32)
        scale_pow2 = tl.exp2((scale_u8 - 127).to(tl.float32))
        out_off = out_row_base + tile_id * TILE_SIZE + nope_offs
        tl.store(
            output_ptr + out_off,
            (fp8_vals * scale_pow2).to(output_ptr.dtype.element_ty),
        )

    rope_offs = tl.arange(0, DIM_ROPE)
    bf16_off = (token_data_base + DIM_NOPE) // 2 + rope_offs
    rope_data = tl.load(buf_bf16_ptr + bf16_off, mask=valid, other=0.0)
    tl.store(output_ptr + out_row_base + DIM_NOPE + rope_offs, rope_data)

    compact_loc = (query_id * OUTPUT_TOPK + output_col).to(tl.int32)
    tl.store(
        compact_indices_ptr
        + query_id * compact_indices_stride_0
        + output_col * compact_indices_stride_2,
        tl.where(in_valid_prefix, compact_loc, 0),
    )


def dequantize_k_cache_paged_ref(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Pure-torch reference for :func:`dequantize_k_cache_paged`.

    Decodes the same v4 paged layout with vectorized torch indexing instead of
    a Triton kernel. Used to validate the kernel (see the ``__main__`` block
    below); not on any hot path.
    """
    assert page_table_1_flattened.dtype in (torch.int32, torch.int64)
    u8 = quant_k_cache.view(torch.uint8)
    bytes_per_page = u8.shape[-1]
    s_offset_bytes = page_size * NOPE_ROPE_BYTES

    flat_u8 = u8.reshape(-1)
    flat_fp8 = u8.view(fp8_dtype).reshape(-1)
    flat_bf16 = u8.view(torch.bfloat16).reshape(-1)

    loc = page_table_1_flattened.to(torch.int64)
    page_idx = loc // page_size
    in_page = loc % page_size
    page_byte_base = page_idx * bytes_per_page
    token_data_base = page_byte_base + in_page * NOPE_ROPE_BYTES
    token_scale_base = (
        page_byte_base + s_offset_bytes + in_page * PADDED_SCALE_PER_TOKEN
    )

    device = quant_k_cache.device
    nope_byte = (
        token_data_base[:, None] + torch.arange(DIM_NOPE, device=device)[None, :]
    )
    nope_fp8 = flat_fp8[nope_byte].to(torch.float32)
    scale_byte = (
        token_scale_base[:, None]
        + torch.arange(NUM_SCALE_TILES, device=device)[None, :]
    )
    scale_u8 = flat_u8[scale_byte].to(torch.int32)
    scale_pow2 = torch.exp2((scale_u8 - 127).to(torch.float32))
    scale_pow2 = torch.where(
        scale_pow2 < (2.0**-126), torch.zeros_like(scale_pow2), scale_pow2
    )
    scale_full = scale_pow2.repeat_interleave(TILE_SIZE, dim=1)
    nope = nope_fp8 * scale_full

    rope_bf16_base = (token_data_base + DIM_NOPE) // 2
    rope_idx = rope_bf16_base[:, None] + torch.arange(DIM_ROPE, device=device)[None, :]
    rope = flat_bf16[rope_idx]

    out = torch.empty(
        (loc.shape[0], 1, DIM_NOPE + DIM_ROPE),
        dtype=torch.bfloat16,
        device=device,
    )
    out[:, 0, :DIM_NOPE] = nope.to(torch.bfloat16)
    out[:, 0, DIM_NOPE:] = rope
    return out


def gather_dequant_requant_fp8_paged_ref(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    page_size: int,
    extra_rows: int = 0,
) -> torch.Tensor:
    """Torch reference for :func:`gather_dequant_requant_fp8_paged`."""
    active = dequantize_k_cache_paged_ref(
        quant_k_cache,
        page_table_1_flattened,
        page_size,
    ).to(fp8_dtype)
    if extra_rows == 0:
        return active
    out = torch.zeros(
        (active.shape[0] + extra_rows, 1, DIM_NOPE + DIM_ROPE),
        dtype=fp8_dtype,
        device=active.device,
    )
    out[: active.shape[0]] = active
    return out


if __name__ == "__main__":
    assert torch.cuda.is_available(), "this self-test needs a CUDA device"
    torch.manual_seed(0)
    device = "cuda"

    page_size = 64
    num_pages = 8
    num_tokens = 333
    raw_bytes = page_size * (NOPE_ROPE_BYTES + PADDED_SCALE_PER_TOKEN)
    bytes_per_page = (
        (raw_bytes + NOPE_ROPE_BYTES - 1) // NOPE_ROPE_BYTES
    ) * NOPE_ROPE_BYTES

    quant_k_cache = torch.randint(
        0, 256, (num_pages, bytes_per_page), dtype=torch.uint8, device=device
    )
    page_table = torch.randint(
        0, num_pages * page_size, (num_tokens,), dtype=torch.int32, device=device
    )

    out_kernel = dequantize_k_cache_paged(quant_k_cache, page_table, page_size)
    out_ref = dequantize_k_cache_paged_ref(quant_k_cache, page_table, page_size)

    torch.testing.assert_close(out_kernel, out_ref, atol=0, rtol=0, equal_nan=True)
    print(
        f"OK: kernel matches torch ref for {num_tokens} tokens "
        f"(page_size={page_size}, bytes_per_page={bytes_per_page})"
    )
