from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn

# DSV4 KV cache byte layout used by DeepSeekV4KVCache:
#   token data: 448 fp8 nope bytes + 64 bf16 rope values
#   scale data: 7 UE8M0 exponent bytes padded to 8 bytes per token
#   page data:  token data for every token, then scale data for every token,
#               padded to a multiple of the 576-byte token data width.
DIM_NOPE = 448
DIM_ROPE = 64
TILE_SIZE = 64
NUM_SCALE_TILES = DIM_NOPE // TILE_SIZE
NOPE_ROPE_BYTES = DIM_NOPE + DIM_ROPE * 2
PADDED_SCALE_PER_TOKEN = NUM_SCALE_TILES + 1


def dequantize_k_cache_paged(
    quant_k_cache: torch.Tensor,
    token_ids: torch.Tensor,
    page_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert quant_k_cache.is_contiguous()
    assert token_ids.dtype in (torch.int32, torch.int64)

    quant_k_cache_u8 = quant_k_cache.view(torch.uint8)
    num_tokens = token_ids.shape[0]
    bytes_per_page = quant_k_cache_u8.shape[-1]
    scale_offset_bytes = page_size * NOPE_ROPE_BYTES

    buf_fp8 = quant_k_cache_u8.view(FP8_DTYPE).reshape(-1)
    buf_bf16 = quant_k_cache_u8.view(torch.bfloat16).reshape(-1)
    buf_u8 = quant_k_cache_u8.reshape(-1)

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
        buf_u8,
        token_ids,
        out.stride(0),
        BYTES_PER_PAGE=bytes_per_page,
        PAGE_SIZE=page_size,
        DIM_NOPE=DIM_NOPE,
        DIM_ROPE=DIM_ROPE,
        TILE_SIZE=TILE_SIZE,
        NUM_SCALE_TILES=NUM_SCALE_TILES,
        NOPE_ROPE_BYTES=NOPE_ROPE_BYTES,
        PADDED_SCALE_PER_TOKEN=PADDED_SCALE_PER_TOKEN,
        SCALE_OFFSET_BYTES=scale_offset_bytes,
    )
    return out


@triton.jit
def _dequantize_k_cache_paged_kernel(
    out_ptr,
    buf_fp8_ptr,
    buf_bf16_ptr,
    buf_u8_ptr,
    token_ids_ptr,
    out_stride_0,
    BYTES_PER_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_SCALE_TILES: tl.constexpr,
    NOPE_ROPE_BYTES: tl.constexpr,
    PADDED_SCALE_PER_TOKEN: tl.constexpr,
    SCALE_OFFSET_BYTES: tl.constexpr,
):
    token_idx = tl.program_id(0)
    loc = tl.load(token_ids_ptr + token_idx).to(tl.int64)
    page_idx = loc // PAGE_SIZE
    in_page = loc - page_idx * PAGE_SIZE

    page_base = page_idx * BYTES_PER_PAGE
    token_data_base = page_base + in_page * NOPE_ROPE_BYTES
    token_scale_base = page_base + SCALE_OFFSET_BYTES + in_page * PADDED_SCALE_PER_TOKEN
    out_base = token_idx * out_stride_0

    offs = tl.arange(0, TILE_SIZE)
    for tile_id in tl.static_range(NUM_SCALE_TILES):
        fp8_vals = tl.load(buf_fp8_ptr + token_data_base + tile_id * TILE_SIZE + offs)
        scale_u8 = tl.load(buf_u8_ptr + token_scale_base + tile_id).to(tl.int32)
        scale = tl.exp2((scale_u8 - 127).to(tl.float32))
        tl.store(
            out_ptr + out_base + tile_id * TILE_SIZE + offs,
            (fp8_vals.to(tl.float32) * scale).to(out_ptr.dtype.element_ty),
        )

    rope_offs = tl.arange(0, DIM_ROPE)
    rope_bf16_base = (token_data_base + DIM_NOPE) // 2
    rope = tl.load(buf_bf16_ptr + rope_bf16_base + rope_offs)
    tl.store(out_ptr + out_base + DIM_NOPE + rope_offs, rope)
