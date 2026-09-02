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

from typing import Optional, Union

from flash_attn import flash_attn_varlen_func as flash_attn_varlen_func_interface
from flash_attn import flash_attn_with_kvcache as flash_attn_with_kvcache_interface
from flash_attn import (
    vllm_flash_attn_varlen_func as vllm_flash_attn_varlen_func_interface,
)
from flash_attn import (
    vllm_flash_attn_with_kvcache as vllm_flash_attn_with_kvcache_interface,
)

from sglang.srt.utils import is_hcu
from sglang.srt.utils.common import get_bool_env_var

_use_triton_vllm_fa = get_bool_env_var("SGLANG_USE_TRITON_VLLM_FA")
_is_hcu = is_hcu()
_kv_layout_hcu_fa = _is_hcu and get_bool_env_var(
    "SGLANG_KV_LAYOUT_HCU_FA", default="true"
)

if _is_hcu and _use_triton_vllm_fa:
    from sglang.srt.layers.attention.triton_vllm_flash_attn import (
        triton_vllm_flash_attn_varlen_func,
        triton_vllm_flash_attn_with_kvcache,
    )

import torch

_SERVER_ARGS = None
IS_SLIMQUANT_W4A8 = None
IS_KVCACHE_FP8_E4M3 = None


def is_nmz_fp8(dtype: torch.dtype) -> bool:
    if is_hcu():
        props = torch.cuda.get_device_properties(0)
        gcn_arch = getattr(props, "gcnArchName", "")
        if "gfx938" in gcn_arch and (
            dtype == torch.float8_e4m3fn or dtype == torch.float8_e5m2
        ):
            return True
    return False


@torch._dynamo.disable()
def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    attention_chunk: Optional[int] = None,
    softcap=0.0,  # 0.0 means deactivated
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,  # Can be tuned for speed
    pack_gqa=None,  # Can be tuned for speed
    sm_margin=0,  # Can be tuned if some SMs are used for communication
    return_softmax_lse=False,
    sinks=None,
    ver=3,
    out=None,
    layout=None,
):
    if layout == "bhsd":
        if not _is_hcu:
            raise ValueError("bhsd layout is reserved for HCU HND KV cache")
        if qv is not None:
            raise NotImplementedError("HND BHSD attention does not support qv")
        if page_table is None or cu_seqlens_q is None or max_seqlen_q is None:
            raise ValueError("HND BHSD attention requires paged varlen metadata")
        if not torch.is_tensor(cache_seqlens):
            cache_seqlens = torch.full(
                (cu_seqlens_q.numel() - 1,),
                int(cache_seqlens),
                dtype=torch.int32,
                device=q.device,
            )
        cu_seqlens_k = torch.cat(
            [cache_seqlens.new_zeros(1), torch.cumsum(cache_seqlens, dim=0)]
        )
        if _is_hcu and _use_triton_vllm_fa and not return_softmax_lse:
            result = triton_vllm_flash_attn_varlen_func(
                q=q,
                k=k_cache,
                v=v_cache,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=cache_seqlens,
                max_seqlen_k=page_table.shape[1] * k_cache.shape[2],
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                block_table=page_table,
                fa_version=ver,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                layout="bhsd",
            )
            return _apply_flash_attn_varlen_out(result, out, return_softmax_lse)

        result = flash_attn_varlen_func_interface(
            q=q,
            k=k_cache,
            v=v_cache,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=page_table.shape[1] * k_cache.shape[2],
            seqused_k=cache_seqlens,
            block_table=page_table,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            num_splits=num_splits,
            return_softmax_lse=return_softmax_lse,
            fa_version=ver,
            layout="bhsd",
        )
        return _apply_flash_attn_varlen_out(result, out, return_softmax_lse)

    if _is_hcu and _use_triton_vllm_fa and is_nmz_fp8(k_cache.dtype):
        result = triton_vllm_flash_attn_with_kvcache(
            q=q.contiguous().view(-1, max_seqlen_q, q.shape[-2], q.shape[-1]),
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=page_table,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            softcap=softcap,
            return_softmax_lse=return_softmax_lse,
            layout=layout,
        )
        return _apply_flash_attn_varlen_out(result, out, return_softmax_lse)

    k_cache = k_cache.to(q.dtype) if not is_nmz_fp8(k_cache.dtype) else k_cache
    v_cache = v_cache.to(q.dtype) if not is_nmz_fp8(k_cache.dtype) else v_cache
    result = flash_attn_with_kvcache_interface(
        q=q.contiguous().view(-1, max_seqlen_q, q.shape[-2], q.shape[-1]),
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=page_table,
        cache_seqlens=cache_seqlens,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        return_softmax_lse=return_softmax_lse,
        num_splits=num_splits,
    )
    return _apply_flash_attn_varlen_out(result, out, return_softmax_lse)


def vllm_flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    # max_seqlen_k: Optional[int] = 0,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    attention_chunk: Optional[int] = None,
    softcap=0.0,  # 0.0 means deactivated
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,  # Can be tuned for speed
    pack_gqa=None,  # Can be tuned for speed
    sm_margin=0,  # Can be tuned if some SMs are used for communication
    return_softmax_lse=False,
    sinks=None,
    ver=3,
    layout=None,
):
    if _is_hcu and _use_triton_vllm_fa:
        return triton_vllm_flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=page_table,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            softcap=softcap,
            return_softmax_lse=return_softmax_lse,
            layout=layout,
        )

    return vllm_flash_attn_with_kvcache_interface(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=page_table,
        cache_seqlens=cache_seqlens,
        softmax_scale=softmax_scale,
        # max_seqlen_k=max_seqlen_k,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        return_softmax_lse=return_softmax_lse,
        num_splits=num_splits,
    )


def _apply_flash_attn_varlen_out(result, out, return_softmax_lse):
    if out is None:
        return result
    if return_softmax_lse:
        attn_out, lse, *rest = result
        out.copy_(attn_out)
        return (out, lse, *rest)
    out.copy_(result)
    return out


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,
    max_seqlen_k=None,
    seqused_q=None,
    seqused_k=None,
    page_table=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    sm_margin=0,
    return_softmax_lse=False,
    sinks=None,
    ver=3,
    out=None,
    layout=None,
):
    global _SERVER_ARGS, IS_SLIMQUANT_W4A8, IS_KVCACHE_FP8_E4M3
    if IS_KVCACHE_FP8_E4M3 is None:
        from sglang.srt.server_args import get_global_server_args

        _SERVER_ARGS = get_global_server_args()
        IS_SLIMQUANT_W4A8 = _SERVER_ARGS.quantization == "slimquant_w4a8_marlin"
        IS_KVCACHE_FP8_E4M3 = _SERVER_ARGS.kv_cache_dtype == "fp8_e4m3"

    if is_nmz_fp8(k.dtype) and not IS_SLIMQUANT_W4A8 and not IS_KVCACHE_FP8_E4M3:
        q_descale = torch.ones_like(k_descale)
        result = flash_attn_varlen_func_interface(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            causal=causal,
            return_attn_probs=return_softmax_lse,
            softcap=softcap,
            **({"layout": layout} if layout is not None else {}),
        )
        return _apply_flash_attn_varlen_out(result, out, return_softmax_lse)

    result = flash_attn_varlen_func_interface(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        return_attn_probs=return_softmax_lse,
        softcap=softcap,
        **({"layout": layout} if layout is not None else {}),
    )
    return _apply_flash_attn_varlen_out(result, out, return_softmax_lse)


def vllm_flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    fa_version,
    q_descale,
    k_descale,
    v_descale,
    layout=None,
):
    if _is_hcu and _use_triton_vllm_fa:
        return triton_vllm_flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            block_table=block_table,
            fa_version=fa_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            layout=layout,
        )

    return vllm_flash_attn_varlen_func_interface(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        seqused_k=seqused_k,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        block_table=block_table,
        fa_version=fa_version,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )
