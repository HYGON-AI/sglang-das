from typing import List, Optional

import torch
import triton
import triton.language as tl


def transform_index_page_table_prefill(**kwargs):
    return transform_index_page_table_prefill_fast(**kwargs)


def transform_index_page_table_decode(**kwargs):
    return transform_index_page_table_decode_fast(**kwargs)


# GLM NOTE: topk_indices / page_table_row_indices / result can be views with
# batch-dependent alignment, and the numel argument changes per batch; each
# new specialization combo forces a full HCU JIT recompile on every rank
# (native-heap residue per compile), so keep these out of the cache key.
@triton.jit(
    do_not_specialize=[
        "page_table_stride_0",
        "page_table_num_rows",
        "page_table_num_cols",
        "page_table_row_indices_numel",
    ],
    do_not_specialize_on_alignment=[
        "page_table_ptr",
        "topk_indices_ptr",
        "page_table_row_indices_ptr",
        "result_ptr",
    ],
)
def transform_index_page_table_batched_kernel(
    page_table_ptr,
    topk_indices_ptr,
    page_table_row_indices_ptr,
    result_ptr,
    page_table_stride_0,
    page_table_stride_1,
    topk_indices_stride_0,
    topk_indices_stride_1,
    page_table_row_indices_stride_0,
    result_stride_0,
    result_stride_1,
    page_table_num_rows,
    page_table_num_cols,
    page_table_row_indices_numel,
    num_cols,
    HAS_PAGE_TABLE_ROW_INDICES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Map logical token positions to allocator-local slots in one launch."""
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_mask = cols < num_cols

    page_table_row = row
    row_valid = row < page_table_num_rows
    if HAS_PAGE_TABLE_ROW_INDICES:
        row_has_index = row < page_table_row_indices_numel
        page_table_row = tl.load(
            page_table_row_indices_ptr + row * page_table_row_indices_stride_0,
            mask=row_has_index,
            other=-1,
        )
        row_valid = (
            row_has_index
            & (page_table_row >= 0)
            & (page_table_row < page_table_num_rows)
        )
    safe_page_table_row = tl.where(row_valid, page_table_row, 0)

    logical_indices = tl.load(
        topk_indices_ptr + row * topk_indices_stride_0 + cols * topk_indices_stride_1,
        mask=col_mask,
        other=-1,
    )
    valid = (
        col_mask
        & row_valid
        & (logical_indices >= 0)
        & (logical_indices < page_table_num_cols)
    )
    safe_logical_indices = tl.where(valid, logical_indices, 0)
    physical_indices = tl.load(
        page_table_ptr
        + safe_page_table_row * page_table_stride_0
        + safe_logical_indices * page_table_stride_1,
        mask=valid,
        other=-1,
    )
    tl.store(
        result_ptr + row * result_stride_0 + cols * result_stride_1,
        physical_indices,
        mask=col_mask,
    )


def transform_index_page_table_batched(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    result: Optional[torch.Tensor] = None,
    page_size: int = 1,
    page_table_row_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Map logical token positions to physical slots in one Triton launch.

    When ``page_table_row_indices`` is provided, output row ``i`` reads page
    table row ``page_table_row_indices[i]``.  A shorter row-index tensor is
    allowed: remaining output rows are filled with ``-1``, which covers
    device-side q padding without allocating another padded mapping tensor.
    Without row indirection, output row ``i`` reads page-table row ``i`` and
    rows beyond the page table are likewise filled with ``-1``.
    """
    assert page_size == 1
    assert page_table.ndim == 2
    assert topk_indices.ndim == 2
    assert page_table.device == topk_indices.device
    assert page_table.dtype in (torch.int32, torch.int64)
    assert topk_indices.dtype in (torch.int32, torch.int64)

    if page_table_row_indices is not None:
        assert page_table_row_indices.ndim == 1
        assert page_table_row_indices.shape[0] <= topk_indices.shape[0]
        assert page_table_row_indices.device == topk_indices.device
        assert page_table_row_indices.dtype in (torch.int32, torch.int64)

    if result is None:
        result = torch.empty_like(topk_indices, dtype=torch.int32)
    assert result.ndim == 2
    assert result.shape == topk_indices.shape
    assert result.device == topk_indices.device
    assert result.dtype in (torch.int32, torch.int64)

    num_rows, num_cols = topk_indices.shape
    if num_rows == 0 or num_cols == 0:
        return result

    block_size = min(256, triton.next_power_of_2(num_cols))
    num_warps = max(1, min(4, block_size // 64))
    grid = (num_rows, triton.cdiv(num_cols, block_size))
    transform_index_page_table_batched_kernel[grid](
        page_table,
        topk_indices,
        page_table_row_indices,
        result,
        page_table.stride(0),
        page_table.stride(1),
        topk_indices.stride(0),
        topk_indices.stride(1),
        (page_table_row_indices.stride(0) if page_table_row_indices is not None else 0),
        result.stride(0),
        result.stride(1),
        page_table.shape[0],
        page_table.shape[1],
        (page_table_row_indices.shape[0] if page_table_row_indices is not None else 0),
        num_cols,
        HAS_PAGE_TABLE_ROW_INDICES=page_table_row_indices is not None,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return result


def transform_index_page_table_decode_fast(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    result: Optional[torch.Tensor] = None,
    page_size: int = 1,
) -> torch.Tensor:
    """
    Transform the page table according to topk indices for sparse topk attention.
    Args:
        page_table: [qo_len, max_seqlen_k], the original page table
        topk_indices: [qo_len, topk], the topk indices for each query position
    Returns:
        transformed_page_table: [qo_len, topk], the transformed page table
        For out-of-bound indices in topk_indices, this should be filled with -1.
    """
    return transform_index_page_table_batched(
        page_table=page_table,
        topk_indices=topk_indices,
        result=result,
        page_size=page_size,
    )


def transform_index_page_table_prefill_fast(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    extend_lens_cpu: List[int],
    page_size: int = 1,
    page_table_row_indices: Optional[torch.Tensor] = None,
    result: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert page_size == 1
    assert len(extend_lens_cpu) == page_table.shape[0]
    assert all(extend_len >= 0 for extend_len in extend_lens_cpu)
    num_logical_rows = sum(extend_lens_cpu)
    assert num_logical_rows <= topk_indices.shape[0]
    if result is not None:
        assert result.ndim == 2
        assert result.shape == topk_indices.shape
        assert result.device == topk_indices.device
        assert result.dtype in (torch.int32, torch.int64)

    if page_table_row_indices is not None:
        assert page_table_row_indices.shape[0] == num_logical_rows
        return transform_index_page_table_batched(
            page_table=page_table,
            topk_indices=topk_indices,
            result=result,
            page_size=page_size,
            page_table_row_indices=page_table_row_indices,
        )

    if num_logical_rows == page_table.shape[0] and all(
        extend_len == 1 for extend_len in extend_lens_cpu
    ):
        return transform_index_page_table_batched(
            page_table=page_table,
            topk_indices=topk_indices,
            result=result,
            page_size=page_size,
        )

    if result is None:
        if num_logical_rows < topk_indices.shape[0]:
            result = torch.full_like(topk_indices, -1, dtype=torch.int32)
        else:
            result = torch.empty_like(topk_indices, dtype=torch.int32)
    elif num_logical_rows < topk_indices.shape[0]:
        result[num_logical_rows:].fill_(-1)

    offset = 0
    for i, l in enumerate(extend_lens_cpu):
        transform_index_page_table_decode_fast(
            page_table[i].unsqueeze(0).expand(l, -1),
            topk_indices[offset : offset + l],
            result=result[offset : offset + l],
        )
        offset += l
    assert offset == num_logical_rows
    return result


def transform_index_page_table_decode_ref(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    result: Optional[torch.Tensor] = None,
    page_size: int = 1,
) -> torch.Tensor:
    assert page_size == 1
    assert page_table.shape[0] == topk_indices.shape[0]
    if result is None:
        result = torch.empty_like(topk_indices, dtype=torch.int32)
    assert result.shape == topk_indices.shape
    torch.gather(
        page_table.to(result.dtype),
        dim=1,
        index=topk_indices.clamp(min=0),
        out=result,
    )
    result[topk_indices < 0] = -1
    return result


def transform_index_page_table_prefill_ref(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    extend_lens_cpu: List[int],
    page_size: int = 1,
) -> torch.Tensor:
    assert page_size == 1
    result = torch.empty_like(topk_indices, dtype=torch.int32)
    assert len(extend_lens_cpu) == page_table.shape[0]
    offset = 0
    for i, l in enumerate(extend_lens_cpu):
        transform_index_page_table_decode_ref(
            page_table[i].unsqueeze(0).expand(l, -1),
            topk_indices[offset : offset + l],
            result=result[offset : offset + l],
        )
        offset += l
    assert offset == topk_indices.shape[0]
    return result


if __name__ == "__main__":
    bs, topk, max_seqlen = 10, 2048, 3000
    page_table = torch.randint(0, 100, (bs, max_seqlen), device="cuda")
    topk_indices = torch.full((bs, topk), -1, device="cuda")
    topk_indices[:, :1600] = torch.arange(1600).unsqueeze(0).repeat(bs, 1)
    ref_result = transform_index_page_table_decode_ref(page_table, topk_indices)
    result = transform_index_page_table_decode_fast(page_table, topk_indices)
    assert torch.all(result == ref_result)
    print("Passed")
