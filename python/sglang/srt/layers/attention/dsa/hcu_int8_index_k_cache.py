"""INT8 page-planar cache helpers for the HCU DSA indexer.

The persistent packed ABI is shared with the scaled FP8 indexer cache: every
physical page stores 64 K vectors followed by 64 FP32 token scales.  In INT8
mode the K bytes are signed INT8 values.  Referenced pages are dequantized into
one reusable BF16 workspace before they are consumed by LightOp.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.utils.common import is_hcu, is_hcu_native_fp8_supported


INDEX_K_PAGE_SIZE = 64
INDEX_K_HEAD_DIM = 128
INDEX_K_SCALE_BYTES = 4
INDEX_K_BYTES_PER_TOKEN = INDEX_K_HEAD_DIM + INDEX_K_SCALE_BYTES
INDEX_K_EPSILON = 1e-6


class IndexKCacheMode(str, Enum):
    BF16 = "bf16"
    FP8_SCALED = "fp8_scaled"
    INT8_SCALED = "int8_scaled"


def is_hcu_gfx936() -> bool:
    """Return whether the active HCU device is the validated INT8 target."""
    if not is_hcu():
        return False
    try:
        gcn_arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
        return "gfx936" in gcn_arch
    except Exception:
        return False


def resolve_index_k_cache_mode(
    dtype: torch.dtype,
    page_size: int,
    index_head_dim: int,
    *,
    int8_enabled: Optional[bool] = None,
    hcu_device: Optional[bool] = None,
    gfx936_device: Optional[bool] = None,
    native_fp8_supported: Optional[bool] = None,
) -> IndexKCacheMode:
    """Resolve the index-K format and reject non-gfx936 INT8 opt-ins."""
    if int8_enabled is None:
        int8_enabled = envs.SGLANG_DSA_HCU_INT8_INDEX_K_CACHE.get()
    if hcu_device is None:
        hcu_device = is_hcu()
    if gfx936_device is None:
        gfx936_device = is_hcu_gfx936()

    if int8_enabled:
        if not hcu_device or not gfx936_device:
            raise ValueError(
                "SGLANG_DSA_HCU_INT8_INDEX_K_CACHE=1 is only supported on "
                "HCU gfx936; gfx938 continues to use the native FP8 index-K cache."
            )
        if page_size != INDEX_K_PAGE_SIZE or index_head_dim != INDEX_K_HEAD_DIM:
            raise ValueError(
                "SGLANG_DSA_HCU_INT8_INDEX_K_CACHE=1 requires "
                f"page_size={INDEX_K_PAGE_SIZE} and "
                f"index_head_dim={INDEX_K_HEAD_DIM}; got page_size={page_size} "
                f"and index_head_dim={index_head_dim}."
            )
        return IndexKCacheMode.INT8_SCALED

    if native_fp8_supported is None:
        native_fp8_supported = is_hcu_native_fp8_supported()
    fp8_dtype = dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    if not hcu_device or (fp8_dtype and native_fp8_supported):
        return IndexKCacheMode.FP8_SCALED
    return IndexKCacheMode.BF16


def index_k_cache_bytes_per_token(mode: IndexKCacheMode) -> int:
    if mode is IndexKCacheMode.BF16:
        return INDEX_K_HEAD_DIM * torch.bfloat16.itemsize
    return INDEX_K_BYTES_PER_TOKEN


def index_k_workspace_bytes_per_token(mode: IndexKCacheMode) -> float:
    """Return the shared workspace and page-claim reservation for INT8."""
    if mode is IndexKCacheMode.INT8_SCALED:
        return INDEX_K_HEAD_DIM * torch.bfloat16.itemsize + 4 / INDEX_K_PAGE_SIZE
    return 0.0


def create_index_k_int8_aliases(
    packed_cache: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create zero-copy INT8-K and FP32-scale aliases of packed storage."""
    if packed_cache.dtype != torch.uint8 or packed_cache.ndim != 2:
        raise ValueError("packed INT8 index-K cache must be a 2D torch.uint8 tensor")
    expected_page_bytes = INDEX_K_PAGE_SIZE * INDEX_K_BYTES_PER_TOKEN
    if packed_cache.shape[1] != expected_page_bytes:
        raise ValueError(
            "packed INT8 index-K cache has invalid page width: "
            f"expected {expected_page_bytes}, got {packed_cache.shape[1]}"
        )
    if not packed_cache.is_contiguous():
        raise ValueError("packed INT8 index-K cache must be contiguous")

    num_pages = packed_cache.shape[0]
    k_bytes = INDEX_K_PAGE_SIZE * INDEX_K_HEAD_DIM
    int8_k = packed_cache[:, :k_bytes].view(torch.int8).view(
        num_pages, INDEX_K_PAGE_SIZE, INDEX_K_HEAD_DIM
    )
    fp32_scales = packed_cache[:, k_bytes:].view(torch.float32).view(
        num_pages, INDEX_K_PAGE_SIZE
    )
    return int8_k, fp32_scales


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
            f"INT8 index-K cache requires head dim {INDEX_K_HEAD_DIM}, "
            f"got {key.shape[1]}"
        )
    if key.shape[0] != out_cache_loc.numel():
        raise ValueError("key and out_cache_loc must contain the same number of tokens")
    if out_cache_loc.dtype not in (torch.int32, torch.int64):
        raise ValueError("out_cache_loc must be an int32 or int64 tensor")
    if not key.is_contiguous():
        key = key.contiguous()
    if not out_cache_loc.is_contiguous():
        out_cache_loc = out_cache_loc.contiguous()
    expected_page_bytes = INDEX_K_PAGE_SIZE * INDEX_K_BYTES_PER_TOKEN
    if (
        packed_cache.dtype != torch.uint8
        or packed_cache.ndim != 2
        or packed_cache.shape[1] != expected_page_bytes
        or not packed_cache.is_contiguous()
    ):
        raise ValueError("packed INT8 index-K cache has an invalid page-planar layout")
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
    k_ptr = (
        k_cache_ptr + page * k_page_stride_0 + token_offset * HEAD_DIM + offsets
    )
    scale_ptr = scale_cache_ptr + page * scale_page_stride_0 + token_offset
    tl.store(k_ptr, quantized)
    tl.store(scale_ptr, scale)


def _quantize_and_store_index_k_int8_reference(
    key: torch.Tensor,
    int8_k: torch.Tensor,
    fp32_scales: torch.Tensor,
    out_cache_loc: torch.Tensor,
    epsilon: float,
) -> None:
    key_fp32 = key.float()
    scales = key_fp32.abs().amax(dim=-1).clamp_min(epsilon) / 127.0
    scaled = key_fp32 / scales[:, None]
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
    fp32_scales[pages, offsets] = scales


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
    """Symmetrically quantize BF16 K and scatter it into packed storage."""
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
    _quantize_and_store_index_k_int8_reference(
        key, int8_k, fp32_scales, out_cache_loc, epsilon
    )


@triton.jit
def _dequantize_index_k_int8_paged_kernel(
    int8_k_ptr,
    scale_ptr,
    block_tables_ptr,
    context_lens_ptr,
    workspace_ptr,
    page_claims_ptr,
    block_table_stride_0: tl.constexpr,
    k_page_stride_0: tl.constexpr,
    scale_page_stride_0: tl.constexpr,
    workspace_page_stride_0: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_PHYSICAL_PAGES: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    logical_page_idx = tl.program_id(1)
    context_len = tl.load(context_lens_ptr + batch_idx).to(tl.int64)
    valid_page_count = (context_len + PAGE_SIZE - 1) // PAGE_SIZE
    if logical_page_idx < valid_page_count:
        physical_page = tl.load(
            block_tables_ptr + batch_idx * block_table_stride_0 + logical_page_idx
        ).to(tl.int64)
        if (physical_page >= 0) & (physical_page < NUM_PHYSICAL_PAGES):
            previous_claim = tl.atomic_cas(page_claims_ptr + physical_page, 0, 1)
            if previous_claim == 0:
                offsets = tl.arange(0, HEAD_DIM)
                for token_offset in tl.static_range(0, PAGE_SIZE):
                    quantized = tl.load(
                        int8_k_ptr
                        + physical_page * k_page_stride_0
                        + token_offset * HEAD_DIM
                        + offsets
                    ).to(tl.float32)
                    scale = tl.load(
                        scale_ptr
                        + physical_page * scale_page_stride_0
                        + token_offset
                    )
                    tl.store(
                        workspace_ptr
                        + physical_page * workspace_page_stride_0
                        + token_offset * HEAD_DIM
                        + offsets,
                        (quantized * scale).to(tl.bfloat16),
                    )


def _dequantize_index_k_int8_paged_reference(
    int8_k: torch.Tensor,
    fp32_scales: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    workspace: torch.Tensor,
    page_claims: torch.Tensor,
) -> torch.Tensor:
    page_claims.zero_()
    for batch_idx, context_len in enumerate(context_lens.tolist()):
        page_count = (int(context_len) + INDEX_K_PAGE_SIZE - 1) // INDEX_K_PAGE_SIZE
        for physical_page in block_tables[batch_idx, :page_count].tolist():
            if physical_page < 0:
                continue
            if physical_page >= int8_k.shape[0]:
                raise ValueError(f"invalid physical index-K page ID {physical_page}")
            if page_claims[physical_page] != 0:
                continue
            page_claims[physical_page] = 1
            workspace[physical_page, :, 0, :] = (
                int8_k[physical_page].float() * fp32_scales[physical_page, :, None]
            ).to(torch.bfloat16)
    return workspace


def dequantize_index_k_int8_paged(
    packed_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    workspace: torch.Tensor,
    page_claims: torch.Tensor,
    *,
    int8_k: Optional[torch.Tensor] = None,
    fp32_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dequantize referenced physical pages once into a BF16 workspace."""
    expected_page_bytes = INDEX_K_PAGE_SIZE * INDEX_K_BYTES_PER_TOKEN
    if (
        packed_cache.dtype != torch.uint8
        or packed_cache.ndim != 2
        or packed_cache.shape[1] != expected_page_bytes
        or not packed_cache.is_contiguous()
    ):
        raise ValueError("packed INT8 index-K cache has an invalid page-planar layout")
    if workspace.dtype != torch.bfloat16 or workspace.ndim != 4:
        raise ValueError(
            "INT8 index-K workspace must have BF16 [pages, 64, 1, 128] layout"
        )
    expected_workspace_shape = (
        packed_cache.shape[0],
        INDEX_K_PAGE_SIZE,
        1,
        INDEX_K_HEAD_DIM,
    )
    if tuple(workspace.shape) != expected_workspace_shape:
        raise ValueError(
            "INT8 index-K workspace has invalid shape: "
            f"expected {expected_workspace_shape}, got {tuple(workspace.shape)}"
        )
    if page_claims.dtype != torch.int32 or page_claims.numel() != packed_cache.shape[0]:
        raise ValueError("INT8 index-K page_claims must be one int32 value per page")
    if block_tables.ndim != 2 or context_lens.ndim != 1:
        raise ValueError("block_tables must be 2D and context_lens must be 1D")
    if block_tables.shape[0] != context_lens.shape[0]:
        raise ValueError(
            "block_tables and context_lens must have matching batch dimensions"
        )

    if int8_k is None or fp32_scales is None:
        int8_k, fp32_scales = create_index_k_int8_aliases(packed_cache)
    page_claims.zero_()
    if context_lens.numel() == 0 or block_tables.shape[1] == 0:
        return workspace

    if packed_cache.is_cuda:
        _dequantize_index_k_int8_paged_kernel[
            (block_tables.shape[0], block_tables.shape[1])
        ](
            int8_k,
            fp32_scales,
            block_tables,
            context_lens,
            workspace,
            page_claims,
            block_tables.stride(0),
            int8_k.stride(0),
            fp32_scales.stride(0),
            workspace.stride(0),
            HEAD_DIM=INDEX_K_HEAD_DIM,
            PAGE_SIZE=INDEX_K_PAGE_SIZE,
            NUM_PHYSICAL_PAGES=packed_cache.shape[0],
            num_warps=4,
        )
        return workspace
    return _dequantize_index_k_int8_paged_reference(
        int8_k, fp32_scales, block_tables, context_lens, workspace, page_claims
    )


def validate_hcu_int8_index_k_cache_server_args(
    server_args, *, allow_mooncake_pd: bool
) -> None:
    """Reject combinations whose INT8 cache lifetime is not implemented."""
    if not envs.SGLANG_DSA_HCU_INT8_INDEX_K_CACHE.get():
        return

    unsupported = []
    if getattr(server_args, "enable_hisparse", False):
        unsupported.append("--enable-hisparse")
    if getattr(server_args, "enable_hierarchical_cache", False):
        unsupported.append("--enable-hierarchical-cache")
    if getattr(server_args, "cpu_offload_gb", 0) > 0:
        unsupported.append("--cpu-offload-gb")
    if getattr(server_args, "enable_two_batch_overlap", False):
        unsupported.append("--enable-two-batch-overlap")

    disaggregation_mode = getattr(server_args, "disaggregation_mode", "null")
    if disaggregation_mode != "null":
        backend = getattr(server_args, "disaggregation_transfer_backend", "mooncake")
        if not allow_mooncake_pd or backend not in ("mooncake", "mooncake_tcp"):
            unsupported.append(f"--disaggregation-transfer-backend={backend}")

    if unsupported:
        raise ValueError(
            "SGLANG_DSA_HCU_INT8_INDEX_K_CACHE=1 does not support "
            + ", ".join(unsupported)
            + "."
        )
