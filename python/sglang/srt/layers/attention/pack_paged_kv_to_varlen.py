from __future__ import annotations

from typing import Optional

import torch
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import get_bool_env_var

_kv_layout_dcu_fa = get_bool_env_var("SGLANG_KV_LAYOUT_DCU_FA", default="true")
_max_batch_size_by_kv_heads = {1: 128, 2: 4}


def can_pack_paged_kv_to_varlen(
    *,
    forward_batch,
    metadata,
    layer,
    window_size: tuple,
    sinks: Optional[torch.Tensor],
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    page_size: int,
) -> bool:
    server_args = get_global_server_args()
    pack_mode = server_args.pack_paged_kv_to_varlen
    if pack_mode == "off":
        return False

    seq_lens_cpu = forward_batch.seq_lens_cpu
    if seq_lens_cpu is None:
        return False

    batch_size = forward_batch.batch_size
    seq_lens_cpu = seq_lens_cpu[:batch_size]
    total_kv_tokens = int(seq_lens_cpu.sum().item())

    correctness_ok = (
        _kv_layout_dcu_fa
        and not forward_batch.mha_return_lse
        and layer.logit_cap == 0.0
        and window_size == (-1, -1)
        and sinks is None
        and key_cache.dtype in (torch.float16, torch.bfloat16)
        and value_cache.dtype in (torch.float16, torch.bfloat16)
        and metadata.page_table is not None
        and metadata.cu_seqlens_k is not None
        and metadata.page_table.shape[0] >= batch_size
        and metadata.page_table.shape[1]
        >= (metadata.max_seq_len_k + page_size - 1) // page_size
        and metadata.cu_seqlens_k.shape[0] >= batch_size + 1
        and metadata.max_seq_len_k > 0
        and int(seq_lens_cpu.min().item()) > 0
    )
    if not correctness_ok:
        return False

    if pack_mode == "on":
        return True

    if layer.tp_k_head_num > 4:
        return False

    max_batch_size = _max_batch_size_by_kv_heads.get(layer.tp_k_head_num, 1)
    return (
        batch_size <= max_batch_size
        and total_kv_tokens >= server_args.pack_paged_kv_to_varlen_min_kv_tokens
        and metadata.max_seq_len_q >= server_args.pack_paged_kv_to_varlen_min_q_tokens
    )


def pack_paged_kv_to_varlen(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens_k_cpu: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    packed_k_list = []
    packed_v_list = []
    for batch_idx, seq_len_k in enumerate(seq_lens_k_cpu.tolist()):
        page_count = (seq_len_k + page_size - 1) // page_size
        pages = page_table[batch_idx, :page_count].to(torch.long)
        k = key_cache.index_select(0, pages)
        v = value_cache.index_select(0, pages)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 3, 1, 2)
        k = k.reshape(-1, k.shape[2], k.shape[3])
        v = v.reshape(-1, v.shape[2], v.shape[3])
        packed_k_list.append(k[:seq_len_k])
        packed_v_list.append(v[:seq_len_k])

    return torch.cat(packed_k_list, dim=0), torch.cat(packed_v_list, dim=0)


def try_pack_paged_kv_to_varlen_attention(
    *,
    q: torch.Tensor,
    forward_batch,
    metadata,
    layer,
    window_size: tuple,
    sinks: Optional[torch.Tensor],
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    page_size: int,
    k_descale,
    v_descale,
    **kwargs,
):
    from sglang.srt.layers.attention.flashattention_interface import (
        flash_attn_varlen_func,
    )

    if not can_pack_paged_kv_to_varlen(
        forward_batch=forward_batch,
        metadata=metadata,
        layer=layer,
        window_size=window_size,
        sinks=sinks,
        key_cache=key_cache,
        value_cache=value_cache,
        page_size=page_size,
    ):
        return None

    packed_k, packed_v = pack_paged_kv_to_varlen(
        key_cache.view(-1, layer.tp_k_head_num, page_size, layer.head_dim),
        value_cache.view(-1, layer.tp_v_head_num, layer.v_head_dim, page_size),
        metadata.page_table,
        forward_batch.seq_lens_cpu[: forward_batch.batch_size],
        page_size,
    )
    return flash_attn_varlen_func(
        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
        k=packed_k,
        v=packed_v,
        cu_seqlens_q=metadata.cu_seqlens_q,
        cu_seqlens_k=metadata.cu_seqlens_k,
        max_seqlen_q=metadata.max_seq_len_q,
        max_seqlen_k=metadata.max_seq_len_k,
        softmax_scale=layer.scaling,
        causal=True,
        k_descale=k_descale,
        v_descale=v_descale,
        return_softmax_lse=False,
        **kwargs,
    )
