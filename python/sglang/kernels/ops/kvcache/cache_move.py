import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_cpu

_is_cpu = is_cpu()

if _is_cpu:
    from sgl_kernel import copy_all_layer_kv_cache_cpu


@triton.jit
def set_kv_buffer_prefix_valid_tiled(
    src_k_ptr,
    src_v_ptr,
    dst_k_ptr,
    dst_v_ptr,
    loc_2d_ptr,
    commit_len_ptr,
    src_k_row_stride,
    src_v_row_stride,
    dst_k_row_stride,
    dst_v_row_stride,
    block_size,
    ROW_BYTES: tl.constexpr,
    BYTES_PER_TILE: tl.constexpr,
):
    bid = tl.program_id(0)
    row = tl.program_id(1)
    tid = tl.program_id(2)

    commit_len = tl.load(commit_len_ptr + bid)
    if row >= commit_len:
        return

    byte_off = tid * BYTES_PER_TILE + tl.arange(0, BYTES_PER_TILE)
    mask_byte = byte_off < ROW_BYTES
    tl.multiple_of(byte_off, 16)

    loc = tl.load(loc_2d_ptr + bid * block_size + row)
    src_row = bid * block_size + row

    src_k_ptr = tl.cast(src_k_ptr, tl.pointer_type(tl.uint8))
    src_v_ptr = tl.cast(src_v_ptr, tl.pointer_type(tl.uint8))
    dst_k_ptr = tl.cast(dst_k_ptr, tl.pointer_type(tl.uint8))
    dst_v_ptr = tl.cast(dst_v_ptr, tl.pointer_type(tl.uint8))

    src_k_row_ptr = src_k_ptr + src_row * src_k_row_stride + byte_off
    src_v_row_ptr = src_v_ptr + src_row * src_v_row_stride + byte_off
    dst_k_row_ptr = dst_k_ptr + loc * dst_k_row_stride + byte_off
    dst_v_row_ptr = dst_v_ptr + loc * dst_v_row_stride + byte_off

    k_val = tl.load(src_k_row_ptr, mask=mask_byte, other=0)
    v_val = tl.load(src_v_row_ptr, mask=mask_byte, other=0)
    tl.store(dst_k_row_ptr, k_val, mask=mask_byte)
    tl.store(dst_v_row_ptr, v_val, mask=mask_byte)


# This path handles the HCU FA page-major layout:
# K is [page, head, token-in-page, dim], while V is [page, head, dim,
# token-in-page]. Each head therefore needs its own grid axis.
@triton.jit
def set_kv_buffer_prefix_valid_hcu_fa_kernel(
    src_ptr,
    dst_ptr,
    loc_2d_ptr,
    commit_len_ptr,
    src_row_stride,
    src_head_stride,
    src_dim_stride,
    dst_page_stride,
    dst_head_stride,
    dst_token_stride,
    dst_dim_stride,
    block_size,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bid = tl.program_id(0)
    row = tl.program_id(1)
    head_tile = tl.program_id(2)
    head = head_tile // NUM_TILES
    tile = head_tile % NUM_TILES

    commit_len = tl.load(commit_len_ptr + bid)
    valid_row = row < commit_len
    row_idx = bid * block_size + row
    loc = tl.load(loc_2d_ptr + row_idx, mask=valid_row, other=0).to(tl.int64)
    page_id = loc // PAGE_SIZE
    tok_in_page = loc % PAGE_SIZE

    dim_offset = tile * BLOCK + tl.arange(0, BLOCK)
    mask = valid_row & (dim_offset < HEAD_DIM)
    src_row_ptr = (
        src_ptr
        + row_idx * src_row_stride
        + head * src_head_stride
        + dim_offset * src_dim_stride
    )
    dst_row_ptr = (
        dst_ptr
        + page_id * dst_page_stride
        + head * dst_head_stride
        + tok_in_page * dst_token_stride
        + dim_offset * dst_dim_stride
    )
    values = tl.load(src_row_ptr, mask=mask, other=0)
    tl.store(dst_row_ptr, values, mask=mask)


def set_kv_buffer_prefix_valid_hcu_fa(
    k_view: torch.Tensor,
    v_view: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    loc_2d: torch.Tensor,
    commit_lens: torch.Tensor,
    page_size: int,
) -> None:
    """Scatter committed rows into the HCU FA K/V page layout.

    The rectangular launch gates rows in the kernel with ``commit_lens``. It
    avoids materializing a dynamic ``nonzero`` result and its implicit DtoH
    synchronization on the scheduler thread.
    """
    if loc_2d.numel() == 0:
        return
    if k_view.ndim != 4 or v_view.ndim != 4:
        raise ValueError(
            "HCU FA KV views must be rank-4, got "
            f"{k_view.ndim}/{v_view.ndim}."
        )
    if cache_k.ndim != 3 or cache_v.ndim != 3:
        raise ValueError(
            "HCU FA source KV tensors must be rank-3, got "
            f"{cache_k.ndim}/{cache_v.ndim}."
        )
    if loc_2d.ndim != 2 or commit_lens.ndim != 1:
        raise ValueError(
            "HCU FA prefix metadata must be loc_2d=[B, W] and commit_lens=[B]."
        )
    # The kernel linearizes both metadata tensors; normalize only their layout,
    # never their values, so non-contiguous scheduler views stay asynchronous.
    if not loc_2d.is_contiguous():
        loc_2d = loc_2d.contiguous()
    if not commit_lens.is_contiguous():
        commit_lens = commit_lens.contiguous()
    batch_size, block_size = loc_2d.shape
    if commit_lens.shape[0] != batch_size:
        raise ValueError(
            "HCU FA commit_lens batch size mismatch: "
            f"{commit_lens.shape[0]} != {batch_size}."
        )
    num_rows = batch_size * block_size
    if cache_k.shape[0] != num_rows or cache_v.shape[0] != num_rows:
        raise ValueError(
            "HCU FA source KV rows must match loc_2d: "
            f"{cache_k.shape[0]}/{cache_v.shape[0]} != {num_rows}."
        )
    if k_view.dtype != cache_k.dtype or v_view.dtype != cache_v.dtype:
        raise ValueError(
            "HCU FA source/destination dtype mismatch: "
            f"K {cache_k.dtype}/{k_view.dtype}, V {cache_v.dtype}/{v_view.dtype}."
        )

    block = 128
    for src, dst, token_axis, dim_axis in (
        (cache_k, k_view, 2, 3),
        (cache_v, v_view, 3, 2),
    ):
        head_num = dst.shape[1]
        head_dim = src.shape[2]
        if src.shape[1] != head_num or dst.shape[dim_axis] != head_dim:
            raise ValueError(
                "HCU FA source/destination shape mismatch: "
                f"src={tuple(src.shape)}, dst={tuple(dst.shape)}."
            )
        grid = (
            batch_size,
            block_size,
            head_num * triton.cdiv(head_dim, block),
        )
        set_kv_buffer_prefix_valid_hcu_fa_kernel[grid](
            src,
            dst,
            loc_2d,
            commit_lens,
            src.stride(0),
            src.stride(1),
            src.stride(2),
            dst.stride(0),
            dst.stride(1),
            dst.stride(token_axis),
            dst.stride(dim_axis),
            block_size,
            HEAD_DIM=head_dim,
            PAGE_SIZE=page_size,
            NUM_TILES=triton.cdiv(head_dim, block),
            BLOCK=block,
            num_warps=4,
            num_stages=2,
        )


@triton.jit
def set_kv_buffer_prefix_valid_tiled_fp8(
    src_k_ptr,
    src_v_ptr,
    dst_k_ptr,
    dst_v_ptr,
    loc_2d_ptr,
    commit_len_ptr,
    k_scale,
    v_scale,
    src_k_row_stride,
    src_v_row_stride,
    dst_k_row_stride,
    dst_v_row_stride,
    block_size,
    ROW_ELEMS: tl.constexpr,
    ELEMS_PER_TILE: tl.constexpr,
    K_SCALE_IS_TENSOR: tl.constexpr = False,
    V_SCALE_IS_TENSOR: tl.constexpr = False,
):
    bid = tl.program_id(0)
    row = tl.program_id(1)
    tid = tl.program_id(2)

    commit_len = tl.load(commit_len_ptr + bid)
    if row >= commit_len:
        return

    elem_off = tid * ELEMS_PER_TILE + tl.arange(0, ELEMS_PER_TILE)
    mask_elem = elem_off < ROW_ELEMS

    loc = tl.load(loc_2d_ptr + bid * block_size + row)
    src_row = bid * block_size + row

    src_k_row_ptr = src_k_ptr + src_row * src_k_row_stride + elem_off
    src_v_row_ptr = src_v_ptr + src_row * src_v_row_stride + elem_off
    dst_k_row_ptr = dst_k_ptr + loc * dst_k_row_stride + elem_off
    dst_v_row_ptr = dst_v_ptr + loc * dst_v_row_stride + elem_off

    # Tensor division rounds the divisor first and uses correctly rounded division.
    k_val = tl.load(src_k_row_ptr, mask=mask_elem, other=0).to(tl.float32)
    if K_SCALE_IS_TENSOR:
        k_scale = k_scale.to(src_k_ptr.dtype.element_ty).to(tl.float32)
        k_val = tl.div_rn(k_val, k_scale)
    else:
        # Eager host-scalar division multiplies by a rounded FP32 reciprocal.
        k_val = k_val * tl.div_rn(1.0, k_scale)
    k_val = k_val.to(src_k_ptr.dtype.element_ty).to(dst_k_ptr.dtype.element_ty)

    v_val = tl.load(src_v_row_ptr, mask=mask_elem, other=0).to(tl.float32)
    if V_SCALE_IS_TENSOR:
        v_scale = v_scale.to(src_v_ptr.dtype.element_ty).to(tl.float32)
        v_val = tl.div_rn(v_val, v_scale)
    else:
        v_val = v_val * tl.div_rn(1.0, v_scale)
    v_val = v_val.to(src_v_ptr.dtype.element_ty).to(dst_v_ptr.dtype.element_ty)

    tl.store(dst_k_row_ptr, k_val, mask=mask_elem)
    tl.store(dst_v_row_ptr, v_val, mask=mask_elem)


@triton.jit
def copy_all_layer_kv_cache_tiled(
    data_ptrs,
    strides,
    tgt_loc_ptr,
    src_loc_ptr,
    num_locs,
    num_locs_upper: tl.constexpr,
    BYTES_PER_TILE: tl.constexpr,
):
    """2D tiled kernel. Safe for in-place copy."""
    bid = tl.program_id(0)
    tid = tl.program_id(1)

    stride = tl.load(strides + bid)
    base_ptr = tl.load(data_ptrs + bid)
    base_ptr = tl.cast(base_ptr, tl.pointer_type(tl.uint8))

    byte_off = tid * BYTES_PER_TILE + tl.arange(0, BYTES_PER_TILE)
    mask_byte = byte_off < stride
    tl.multiple_of(byte_off, 16)

    loc_idx = tl.arange(0, num_locs_upper)
    mask_loc = loc_idx < num_locs

    src = tl.load(src_loc_ptr + loc_idx, mask=mask_loc, other=0)
    tgt = tl.load(tgt_loc_ptr + loc_idx, mask=mask_loc, other=0)

    src_ptr = base_ptr + src[:, None] * stride + byte_off[None, :]
    tgt_ptr = base_ptr + tgt[:, None] * stride + byte_off[None, :]

    mask = mask_loc[:, None] & mask_byte[None, :]
    vals = tl.load(src_ptr, mask=mask)
    tl.store(tgt_ptr, vals, mask=mask)


def copy_all_layer_kv_cache_func(
    data_ptrs: torch.Tensor,
    strides: torch.Tensor,
    tgt_loc: torch.Tensor,
    src_loc: torch.Tensor,
    num_locs: int,
    num_locs_upper: int,
    kv_copy_config: dict,
):
    if _is_cpu:
        copy_all_layer_kv_cache_cpu(
            data_ptrs,
            strides,
            tgt_loc[:num_locs],
            src_loc[:num_locs],
        )
        return
    grid = (data_ptrs.numel(), kv_copy_config["byte_tiles"])
    copy_all_layer_kv_cache_tiled[grid](
        data_ptrs,
        strides,
        tgt_loc,
        src_loc,
        num_locs,
        num_locs_upper,
        BYTES_PER_TILE=kv_copy_config["bytes_per_tile"],
        num_warps=kv_copy_config["num_warps"],
        num_stages=2,
    )
