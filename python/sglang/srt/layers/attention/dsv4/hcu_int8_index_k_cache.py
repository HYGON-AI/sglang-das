"""INT8 page-planar cache helpers for the DeepSeek-V4 C4 indexer.

The persistent layout intentionally keeps the existing scaled-FP8 ABI:
one 64-token page stores 64x128 signed INT8 K bytes followed by 64 FP32
scales.  The serving path consumes this layout directly with native LightOp
INT8 Paged MQA; it does not allocate or use a BF16 dequant workspace.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.utils import is_hcu


INDEX_K_PAGE_SIZE = 64
INDEX_K_HEAD_DIM = 128
INDEX_K_SCALE_BYTES = 4
INDEX_K_BYTES_PER_TOKEN = INDEX_K_HEAD_DIM + INDEX_K_SCALE_BYTES
INDEX_K_EPSILON = 1e-6


def is_hcu_gfx936() -> bool:
    if not is_hcu():
        return False
    try:
        arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
        return "gfx936" in arch
    except Exception:
        return False


def int8_index_k_cache_enabled() -> bool:
    return envs.SGLANG_DSV4_HCU_INT8_INDEX_K_CACHE.get()


def validate_int8_index_k_cache(
    page_size: int,
    index_head_dim: int,
    *,
    use_fp4_indexer: bool,
) -> None:
    if not int8_index_k_cache_enabled():
        return
    if not is_hcu() or not is_hcu_gfx936():
        raise ValueError(
            "SGLANG_DSV4_HCU_INT8_INDEX_K_CACHE=1 is supported only on "
            "HCU gfx936."
        )
    if use_fp4_indexer:
        raise ValueError(
            "SGLANG_DSV4_HCU_INT8_INDEX_K_CACHE=1 cannot be combined with "
            "the DeepSeek-V4 FP4 indexer."
        )
    if page_size != INDEX_K_PAGE_SIZE or index_head_dim != INDEX_K_HEAD_DIM:
        raise ValueError(
            "SGLANG_DSV4_HCU_INT8_INDEX_K_CACHE=1 requires "
            f"page_size={INDEX_K_PAGE_SIZE} and index_head_dim={INDEX_K_HEAD_DIM}; "
            f"got page_size={page_size}, index_head_dim={index_head_dim}."
        )



def create_index_k_int8_aliases(
    packed_cache: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if packed_cache.dtype != torch.uint8 or packed_cache.ndim != 2:
        raise ValueError("packed INT8 index-K cache must be a 2D uint8 tensor")
    expected_page_bytes = INDEX_K_PAGE_SIZE * INDEX_K_BYTES_PER_TOKEN
    if packed_cache.shape[1] != expected_page_bytes:
        raise ValueError(
            f"packed INT8 index-K page width must be {expected_page_bytes}, "
            f"got {packed_cache.shape[1]}"
        )
    if not packed_cache.is_contiguous():
        raise ValueError("packed INT8 index-K cache must be contiguous")

    num_pages = packed_cache.shape[0]
    k_bytes = INDEX_K_PAGE_SIZE * INDEX_K_HEAD_DIM
    int8_k = packed_cache[:, :k_bytes].view(torch.int8).view(
        num_pages, INDEX_K_PAGE_SIZE, INDEX_K_HEAD_DIM
    )
    scales = packed_cache[:, k_bytes:].view(torch.float32).view(
        num_pages, INDEX_K_PAGE_SIZE
    )
    return int8_k, scales


def _validate_quantize_inputs(
    key: torch.Tensor,
    packed_cache: torch.Tensor,
    out_cache_loc: torch.Tensor,
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if page_size != INDEX_K_PAGE_SIZE:
        raise ValueError(f"INT8 index-K cache requires page_size={INDEX_K_PAGE_SIZE}")
    if key.dtype != torch.bfloat16:
        raise ValueError(f"INT8 index-K cache expects BF16 K, got {key.dtype}")
    key = key.reshape(-1, key.shape[-1])
    if key.shape[1] != INDEX_K_HEAD_DIM:
        raise ValueError(
            f"INT8 index-K cache requires head_dim={INDEX_K_HEAD_DIM}, "
            f"got {key.shape[1]}"
        )
    if key.shape[0] != out_cache_loc.numel():
        raise ValueError("key and out_cache_loc must contain the same token count")
    if out_cache_loc.dtype not in (torch.int32, torch.int64):
        raise ValueError("out_cache_loc must be int32 or int64")
    if not key.is_contiguous():
        key = key.contiguous()
    if not out_cache_loc.is_contiguous():
        out_cache_loc = out_cache_loc.contiguous()
    create_index_k_int8_aliases(packed_cache)
    return key, out_cache_loc


@triton.jit
def _quantize_and_store_index_k_int8_kernel(
    key_ptr,
    k_cache_ptr,
    scale_cache_ptr,
    loc_ptr,
    key_stride_0: tl.constexpr,
    k_page_stride_0: tl.constexpr,
    scale_page_stride_0: tl.constexpr,
    epsilon: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    offsets = tl.arange(0, HEAD_DIM)
    key = tl.load(key_ptr + token_idx * key_stride_0 + offsets).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(key)), epsilon) / 127.0
    scaled = key / scale
    rounded = tl.where(
        scaled >= 0.0, tl.floor(scaled + 0.5), tl.ceil(scaled - 0.5)
    )
    quantized = tl.clamp(rounded, -127.0, 127.0).to(tl.int8)

    loc = tl.load(loc_ptr + token_idx).to(tl.int64)
    page = loc // PAGE_SIZE
    token_offset = loc % PAGE_SIZE
    tl.store(
        k_cache_ptr
        + page * k_page_stride_0
        + token_offset * HEAD_DIM
        + offsets,
        quantized,
    )
    tl.store(
        scale_cache_ptr + page * scale_page_stride_0 + token_offset,
        scale,
    )


def _quantize_and_store_reference(
    key: torch.Tensor,
    int8_k: torch.Tensor,
    scales: torch.Tensor,
    out_cache_loc: torch.Tensor,
    epsilon: float,
) -> None:
    key_fp32 = key.float()
    token_scales = key_fp32.abs().amax(dim=-1).clamp_min(epsilon) / 127.0
    scaled = key_fp32 / token_scales[:, None]
    quantized = torch.clamp(
        torch.where(
            scaled >= 0,
            torch.floor(scaled + 0.5),
            torch.ceil(scaled - 0.5),
        ),
        -127,
        127,
    ).to(torch.int8)
    pages = (out_cache_loc // INDEX_K_PAGE_SIZE).long()
    offsets = (out_cache_loc % INDEX_K_PAGE_SIZE).long()
    int8_k[pages, offsets] = quantized
    scales[pages, offsets] = token_scales


def quantize_and_store_index_k_int8(
    key: torch.Tensor,
    packed_cache: torch.Tensor,
    out_cache_loc: torch.Tensor,
    page_size: int = INDEX_K_PAGE_SIZE,
    epsilon: float = INDEX_K_EPSILON,
    *,
    int8_k: Optional[torch.Tensor] = None,
    fp32_scales: Optional[torch.Tensor] = None,
) -> None:
    key, out_cache_loc = _validate_quantize_inputs(
        key, packed_cache, out_cache_loc, page_size
    )
    if int8_k is None or fp32_scales is None:
        int8_k, fp32_scales = create_index_k_int8_aliases(packed_cache)
    if key.numel() == 0:
        return
    if key.is_cuda:
        _quantize_and_store_index_k_int8_kernel[(key.shape[0],)](
            key,
            int8_k,
            fp32_scales,
            out_cache_loc,
            key.stride(0),
            int8_k.stride(0),
            fp32_scales.stride(0),
            epsilon=epsilon,
            HEAD_DIM=INDEX_K_HEAD_DIM,
            PAGE_SIZE=INDEX_K_PAGE_SIZE,
            num_warps=4,
        )
        return
    _quantize_and_store_reference(
        key, int8_k, fp32_scales, out_cache_loc, epsilon
    )
