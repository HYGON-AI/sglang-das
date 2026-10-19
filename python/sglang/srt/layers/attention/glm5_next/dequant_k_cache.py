from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

if envs.SGLANG_USE_LIGHTOP_PREFILL_DEQUANT.get():
    try:
        from lightop import op as _lightop_op
    except ImportError:
        _lightop_op = None
else:
    _lightop_op = None

_has_lightop_prefill_dequant = _lightop_op is not None and hasattr(
    _lightop_op, "prefill_gather_and_upconvert_fp8_kv_cache"
)


def dequantize_k_cache(quant_k_cache):
    return _dequantize_k_cache_fast_wrapped(quant_k_cache)


def _dequantize_k_cache_ref(
    quant_k_cache: torch.Tensor,  # (num_blocks, block_size, 1, bytes_per_token)
    dv: int = 512,
    tile_size: int = 128,
    d: int = 576,
) -> torch.Tensor:
    """
    De-quantize the k-cache
    """
    assert dv % tile_size == 0
    original_ndim = quant_k_cache.ndim
    if original_ndim == 3:
        # set block_size = 1
        quant_k_cache = quant_k_cache.unsqueeze(1)
    num_tiles = dv // tile_size
    num_blocks, block_size, h_k, _ = quant_k_cache.shape
    assert h_k == 1
    result = torch.empty(
        (num_blocks, block_size, d), dtype=torch.bfloat16, device=quant_k_cache.device
    )

    quant_k_cache = quant_k_cache.view(num_blocks, block_size, -1)

    input_nope = quant_k_cache[..., :dv]
    input_scale = quant_k_cache[..., dv : dv + num_tiles * 4].view(torch.float32)
    input_rope = quant_k_cache[..., dv + num_tiles * 4 :].view(torch.bfloat16)
    result[..., dv:] = input_rope

    for tile_idx in range(0, num_tiles):
        cur_nope = input_nope[
            ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
        ].to(torch.float32)
        cur_scales = input_scale[..., tile_idx].unsqueeze(-1)
        result[..., tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
            cur_nope * cur_scales
        )

    if original_ndim == 3:
        return result.view(num_blocks, 1, -1)
    else:
        return result.view(num_blocks, block_size, 1, -1)


def _dequantize_k_cache_fast_wrapped(
    quant_k_cache: torch.Tensor,
    dv: int = 512,
    tile_size: int = 128,
) -> torch.Tensor:
    original_ndim = quant_k_cache.ndim
    if original_ndim == 3:
        # set block_size = 1
        quant_k_cache = quant_k_cache.unsqueeze(1)
    num_blocks, block_size, _, dim_quant = quant_k_cache.shape
    assert dv == 512
    assert tile_size == 128
    scale_bytes = dv // tile_size * 4
    rope_bytes = dim_quant - dv - scale_bytes
    assert rope_bytes >= 0 and rope_bytes % torch.bfloat16.itemsize == 0
    dim_rope = rope_bytes // torch.bfloat16.itemsize
    assert dim_rope in (0, 64)
    quant_k_cache = quant_k_cache.view((-1, dim_quant))

    output = _dequantize_k_cache_fast(
        quant_k_cache, group_size=tile_size, dim_rope=dim_rope
    )

    if original_ndim == 3:
        return output.view(num_blocks, 1, -1)
    else:
        return output.view(num_blocks, block_size, 1, -1)


def _dequantize_k_cache_fast(quant_k_cache, group_size: int = 128, dim_rope: int = 64):
    num_tokens, dim_quant = quant_k_cache.shape

    assert quant_k_cache.dtype == torch.float8_e4m3fn
    dim_nope = 512
    assert dim_rope in (0, 64)
    num_tiles = dim_nope // group_size
    assert dim_quant == dim_nope + num_tiles * 4 + dim_rope * 2

    output = torch.empty(
        (num_tokens, dim_nope + dim_rope),
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )

    num_blocks_per_token = triton.cdiv(dim_nope + dim_rope, group_size)
    assert dim_nope % group_size == 0

    input_nope_q = quant_k_cache[:, :dim_nope]
    input_nope_s = quant_k_cache[:, dim_nope : dim_nope + num_tiles * 4].view(
        torch.float32
    )
    if dim_rope == 0:
        input_rope = quant_k_cache[:, :2].view(torch.bfloat16)
    else:
        input_rope = quant_k_cache[:, dim_nope + num_tiles * 4 :].view(torch.bfloat16)

    _dequantize_k_cache_fast_kernel[(num_tokens, num_blocks_per_token)](
        output,
        input_nope_q,
        input_nope_s,
        input_rope,
        output.stride(0),
        input_nope_q.stride(0),
        input_nope_s.stride(0),
        input_rope.stride(0),
        NUM_NOPE_BLOCKS=num_tiles,
        GROUP_SIZE=group_size,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
    )

    return output


@triton.jit
def _dequantize_k_cache_fast_kernel(
    output_ptr,
    input_nope_q_ptr,
    input_nope_s_ptr,
    input_rope_ptr,
    output_stride_0: int,
    input_nope_q_stride_0: int,
    input_nope_s_stride_0: int,
    input_rope_stride_0: int,
    NUM_NOPE_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
):
    token_id = tl.program_id(0)
    raw_block_id = tl.program_id(1)

    if raw_block_id < NUM_NOPE_BLOCKS:
        # a. dequant nope
        effective_block_id = raw_block_id

        offs_q = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs_q < DIM_NOPE
        ptr_q = input_nope_q_ptr + token_id * input_nope_q_stride_0 + offs_q
        ptr_s = input_nope_s_ptr + token_id * input_nope_s_stride_0 + effective_block_id

        y_q = tl.load(ptr_q, mask=mask, other=0.0).to(tl.float32)
        y_s = tl.load(ptr_s)

        y = (y_q * y_s).to(output_ptr.dtype.element_ty)

        dst_ptr = output_ptr + token_id * output_stride_0 + offs_q
        tl.store(dst_ptr, y, mask=mask)
    else:
        # b. copy rope
        effective_block_id = raw_block_id - NUM_NOPE_BLOCKS

        offs = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_ROPE

        src_ptr = input_rope_ptr + token_id * input_rope_stride_0 + offs
        dst_ptr = output_ptr + token_id * output_stride_0 + DIM_NOPE + offs

        data = tl.load(src_ptr, mask=mask).to(tl.bfloat16)
        tl.store(dst_ptr, data, mask=mask)


def dequantize_k_cache_paged(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    group_size: int = 128,
    target_dim_rope: Optional[int] = None,
) -> torch.Tensor:
    """
    De-quantize the k-cache with paged layout
    Args:
        quant_k_cache: [total_num_tokens, 1, dim_quant] or [num_blocks, block_size, 1, dim_quant], the quantized k-cache in paged layout
        page_table_1_flattened: [num_tokens], the flattened page_table_1 with the page indices in each requests concatenated together
        target_dim_rope: override the output rope dimension after validating
            the packed storage layout. The no-rope FlashMLA sparse path uses 0
            to require a 512-dimensional BF16 result.
    Returns:
        output: [num_tokens, 1, dim_nope + dim_rope], the de-quantized k-cache
    """
    if target_dim_rope is not None:
        assert target_dim_rope in (0, 64)

    dim_quant = quant_k_cache.shape[-1]
    lightop_output_dim = None
    if target_dim_rope is None:
        if dim_quant == 656:
            lightop_output_dim = 576
        elif dim_quant == 528:
            lightop_output_dim = 512
    elif target_dim_rope == 0 and dim_quant == 528:
        lightop_output_dim = 512

    use_lightop = (
        _has_lightop_prefill_dequant
        and lightop_output_dim is not None
        and group_size == 128
        and quant_k_cache.dtype == torch.float8_e4m3fn
        and quant_k_cache.is_contiguous()
        and quant_k_cache.dim() in (3, 4)
        and quant_k_cache.shape[-2] == 1
        and page_table_1_flattened.dtype == torch.int32
        and page_table_1_flattened.is_contiguous()
    )
    if use_lightop:
        output = torch.empty(
            (page_table_1_flattened.numel(), 1, lightop_output_dim),
            dtype=torch.bfloat16,
            device=quant_k_cache.device,
        )
        _lightop_op.prefill_gather_and_upconvert_fp8_kv_cache(
            quant_k_cache,
            page_table_1_flattened,
            output,
        )
        return output

    quant_k_cache = quant_k_cache.view((-1, dim_quant))

    # num_tokens can exceed kv_cache_size due to prefix sharing (multiple seqs share same KV slots)
    # Index bounds validated in dsa_backend.init_forward_metadata
    num_tokens = page_table_1_flattened.shape[0]
    assert quant_k_cache.dtype == torch.float8_e4m3fn
    dim_nope = 512
    num_tiles = dim_nope // group_size  # 512 // 128 = 4
    scale_bytes = num_tiles * 4
    rope_bytes = dim_quant - dim_nope - scale_bytes
    assert rope_bytes >= 0 and rope_bytes % torch.bfloat16.itemsize == 0
    storage_dim_rope = rope_bytes // torch.bfloat16.itemsize
    assert storage_dim_rope in (
        0,
        64,
    ), f"Unsupported rope dimension {storage_dim_rope} for packed MLA KV cache"
    dim_rope = storage_dim_rope if target_dim_rope is None else target_dim_rope
    assert dim_rope <= storage_dim_rope

    output = torch.empty(
        (num_tokens, 1, dim_nope + dim_rope),
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )

    num_blocks_per_token = triton.cdiv(dim_nope + dim_rope, group_size)

    assert dim_nope % group_size == 0

    input_nope_q = quant_k_cache[:, :dim_nope]
    # [:, 512:512+4*4] = [:, 512:528]
    input_nope_s = quant_k_cache[:, dim_nope : dim_nope + num_tiles * 4].view(
        torch.float32
    )
    # [:, 528:]
    if dim_rope == 0:
        input_rope = quant_k_cache[:, :2].view(torch.bfloat16)
    else:
        input_rope = quant_k_cache[:, dim_nope + num_tiles * 4 :].view(torch.bfloat16)

    _dequantize_k_cache_paged_kernel[(num_tokens, num_blocks_per_token)](
        output,
        input_nope_q,
        input_nope_s,
        input_rope,
        page_table_1_flattened,
        output.stride(0),
        input_nope_q.stride(0),
        input_nope_s.stride(0),
        input_rope.stride(0),
        NUM_NOPE_BLOCKS=num_tiles,
        GROUP_SIZE=group_size,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
    )

    return output


@triton.jit
def _dequantize_k_cache_paged_kernel(
    output_ptr,
    input_nope_q_ptr,
    input_nope_s_ptr,
    input_rope_ptr,
    page_table_1_ptr,
    output_stride_0: int,
    input_nope_q_stride_0: int,
    input_nope_s_stride_0: int,
    input_rope_stride_0: int,
    NUM_NOPE_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
):
    token_id = tl.program_id(0)
    # Physical KV slots can exceed the int32 byte-offset range on large
    # FP8-KV pools, so keep the slot in int64 before multiplying by strides.
    token_id_paged = tl.load(page_table_1_ptr + token_id).to(tl.int64)
    raw_block_id = tl.program_id(1)

    if raw_block_id < NUM_NOPE_BLOCKS:
        # a. dequant nope
        effective_block_id = raw_block_id

        offs_q = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs_q < DIM_NOPE
        ptr_q = input_nope_q_ptr + token_id_paged * input_nope_q_stride_0 + offs_q
        ptr_s = (
            input_nope_s_ptr
            + token_id_paged * input_nope_s_stride_0
            + effective_block_id
        )

        y_q = tl.load(ptr_q, mask=mask, other=0.0).to(tl.float32)
        y_s = tl.load(ptr_s)

        y = (y_q * y_s).to(output_ptr.dtype.element_ty)

        dst_ptr = output_ptr + token_id * output_stride_0 + offs_q
        tl.store(dst_ptr, y, mask=mask)
    else:
        # b. copy rope
        effective_block_id = raw_block_id - NUM_NOPE_BLOCKS

        offs = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_ROPE

        src_ptr = input_rope_ptr + token_id_paged * input_rope_stride_0 + offs
        dst_ptr = output_ptr + token_id * output_stride_0 + DIM_NOPE + offs

        data = tl.load(src_ptr, mask=mask).to(tl.bfloat16)
        tl.store(dst_ptr, data, mask=mask)


if __name__ == "__main__":
    raise Exception("UT is in quant_k_cache.py")
