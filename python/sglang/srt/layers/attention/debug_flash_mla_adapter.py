from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz


_DIM_NOPE = 448
_DIM_ROPE = 64
_DIM_K = _DIM_NOPE + _DIM_ROPE
_TILE_SIZE = 64
_NUM_TILES = _DIM_NOPE // _TILE_SIZE
_BYTES_PER_K_TOKEN = _DIM_NOPE + _DIM_ROPE * 2
_BYTES_PER_SCALE_TOKEN = _NUM_TILES + 1
_BYTES_PER_TOKEN = _BYTES_PER_K_TOKEN + _BYTES_PER_SCALE_TOKEN
_FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
_NATIVE_TILE_SIZE = int(os.environ.get("SGLANG_TORCH_NATIVE_FLASHMLA_TILE_SIZE", "64"))


def _unpack_nope_fp8_rope_bf16_parts(
    k_cache: torch.Tensor,
    loc: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if k_cache.dtype != torch.uint8:
        k_cache = k_cache.view(torch.uint8)
    assert k_cache.shape[-1] == _BYTES_PER_TOKEN, f"{k_cache.shape=}"

    num_pages, page_size = k_cache.shape[:2]
    page_bytes = k_cache.reshape(num_pages, -1)

    loc = loc.to(torch.long)
    safe_loc = loc.clamp(min=0)
    page_idx = safe_loc // page_size
    token_offset = safe_loc % page_size

    nope_offsets = (
        token_offset.unsqueeze(-1) * _BYTES_PER_K_TOKEN
        + torch.arange(_DIM_NOPE, device=loc.device)
    )
    k_nope_fp8 = page_bytes[page_idx.unsqueeze(-1), nope_offsets].view(_FP8_DTYPE)

    rope_byte_offsets = (
        token_offset.unsqueeze(-1) * _BYTES_PER_K_TOKEN
        + _DIM_NOPE
        + torch.arange(_DIM_ROPE * 2, device=loc.device)
    )
    k_rope_bf16 = page_bytes[page_idx.unsqueeze(-1), rope_byte_offsets].view(
        torch.bfloat16
    )

    scale_offsets = (
        page_size * _BYTES_PER_K_TOKEN
        + token_offset.unsqueeze(-1) * _BYTES_PER_SCALE_TOKEN
        + torch.arange(_NUM_TILES, device=loc.device)
    )
    scale_u8 = page_bytes[page_idx.unsqueeze(-1), scale_offsets]
    scale = torch.pow(2.0, scale_u8.to(torch.float32) - 127.0)

    k_nope = k_nope_fp8.float().view(*loc.shape, _NUM_TILES, _TILE_SIZE)
    k_nope = (k_nope * scale.unsqueeze(-1)).flatten(-2)
    return k_nope, k_rope_bf16.float()


def _accumulate_cache_tiles(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    head_dim_v: int,
    softmax_scale: float,
    m: torch.Tensor,
    l: torch.Tensor,
    o: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if indices.ndim == 2:
        indices = indices.unsqueeze(1)
    if topk_length.ndim == 1:
        topk_length = topk_length.unsqueeze(1)

    total_len = indices.shape[-1]
    q_float = q.float()
    q_nope = q_float[..., :_DIM_NOPE]
    q_rope = q_float[..., _DIM_NOPE:]
    arange_full = torch.arange(total_len, device=indices.device).view(1, 1, -1)

    for start in range(0, total_len, _NATIVE_TILE_SIZE):
        end = min(start + _NATIVE_TILE_SIZE, total_len)
        tile_indices = indices[..., start:end]
        tile_positions = arange_full[..., start:end]

        valid = (tile_indices >= 0) & (
            tile_positions < topk_length.to(torch.long).unsqueeze(-1)
        )
        k_nope, k_rope = _unpack_nope_fp8_rope_bf16_parts(k_cache, tile_indices)
        logits = torch.einsum("bqhd,bqld->bqhl", q_nope, k_nope)
        logits = logits + torch.einsum("bqhd,bqld->bqhl", q_rope, k_rope)
        logits = logits * softmax_scale
        logits = logits.masked_fill(~valid.unsqueeze(2), float("-inf"))

        tile_m = logits.amax(dim=-1, keepdim=True)
        new_m = torch.maximum(m, tile_m)
        old_scale = torch.where(
            torch.isfinite(new_m),
            torch.exp(m - new_m),
            torch.zeros_like(new_m),
        )
        p = torch.exp(logits - new_m).masked_fill(~valid.unsqueeze(2), 0.0)

        o.mul_(old_scale)
        nope_v_dim = min(head_dim_v, _DIM_NOPE)
        if nope_v_dim:
            o[..., :nope_v_dim].add_(
                torch.einsum("bqhl,bqld->bqhd", p, k_nope[..., :nope_v_dim])
            )
        rope_v_dim = head_dim_v - nope_v_dim
        if rope_v_dim > 0:
            o[..., nope_v_dim:head_dim_v].add_(
                torch.einsum("bqhl,bqld->bqhd", p, k_rope[..., :rope_v_dim])
            )
        l = l * old_scale + p.sum(dim=-1, keepdim=True)
        m = new_m

    return m, l, o


def torch_native_flash_mla_with_kvcache(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    head_dim_v: int,
    softmax_scale: float,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    **_,
):
    if q.ndim == 3:
        q = q.unsqueeze(1)
    assert q.shape[-1] == _DIM_K, f"{q.shape=}"

    acc_shape = (*q.shape[:-1], 1)
    if attn_sink is not None:
        m = attn_sink.to(torch.float32).view(1, 1, -1, 1).expand(acc_shape)
        l = torch.ones(acc_shape, dtype=torch.float32, device=q.device)
    else:
        m = torch.full(acc_shape, float("-inf"), dtype=torch.float32, device=q.device)
        l = torch.zeros(acc_shape, dtype=torch.float32, device=q.device)
    o = torch.zeros(*q.shape[:-1], head_dim_v, dtype=torch.float32, device=q.device)

    m, l, o = _accumulate_cache_tiles(
        q=q,
        k_cache=k_cache,
        indices=indices,
        topk_length=topk_length,
        head_dim_v=head_dim_v,
        softmax_scale=softmax_scale,
        m=m,
        l=l,
        o=o,
    )

    if extra_k_cache is not None:
        assert extra_indices_in_kvcache is not None
        assert extra_topk_length is not None
        m, l, o = _accumulate_cache_tiles(
            q=q,
            k_cache=extra_k_cache,
            indices=extra_indices_in_kvcache,
            topk_length=extra_topk_length,
            head_dim_v=head_dim_v,
            softmax_scale=softmax_scale,
            m=m,
            l=l,
            o=o,
        )

    out = o / l.clamp_min(1e-20)
    return out.to(q.dtype), None


def flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):
    if backend in {"torch", "native", "torch_native"}:
        return torch_native_flash_mla_with_kvcache(**kwargs)

    assert backend == "kernel", f"unsupported backend {backend!r}"
    import flash_mla

    return flash_mla.flash_mla_with_kvcache(**kwargs)
