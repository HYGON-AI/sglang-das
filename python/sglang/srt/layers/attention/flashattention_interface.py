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

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_spec
from sglang.srt.utils import is_hcu
from sglang.srt.utils.common import get_bool_env_var

_use_triton_vllm_fa = get_bool_env_var("SGLANG_USE_TRITON_VLLM_FA")
_is_hcu = is_hcu()
_kv_layout_hcu_fa = _is_hcu and get_bool_env_var(
    "SGLANG_KV_LAYOUT_HCU_FA", default="true"
)

if _is_hcu:
    from flash_attn import varlen_fwd_unified as varlen_fwd_unified_interface

    from sglang.srt.layers.attention.triton_vllm_flash_attn import (
        triton_vllm_flash_attn_varlen_func,
        triton_vllm_flash_attn_with_kvcache,
    )

import logging
import os

import torch

logger = logging.getLogger(__name__)

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

        result = varlen_fwd_unified_interface(
            q=q,
            k=k_cache,
            v=v_cache,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=cache_seqlens,
            block_table=page_table,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=page_table.shape[1] * k_cache.shape[2],
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            out=out,
            return_softmax_lse=return_softmax_lse,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            s_aux=sinks,
            layout="bhsd",
        )
        return _apply_flash_attn_varlen_out(result, out, return_softmax_lse)

    # The ordinary HCU/NHD path also receives flattened variable-length
    # queries when multiple requests are scheduled together.  Do not reshape
    # them to ``(-1, max_seqlen_q, ...)``: the sum of sequence lengths is not
    # generally divisible by the batch maximum.  Use the paged varlen
    # interface directly, which consumes ``cu_seqlens_q`` and the page table.
    if (
        q.dim() == 3
        and cu_seqlens_q is not None
        and max_seqlen_q is not None
        and page_table is not None
        and cache_seqlens is not None
    ):
        if not torch.is_tensor(cache_seqlens):
            cache_seqlens = torch.full(
                (cu_seqlens_q.numel() - 1,),
                int(cache_seqlens),
                dtype=torch.int32,
                device=q.device,
            )
        cu_seqlens_k = torch.cat(
            [
                cache_seqlens.new_zeros(1),
                torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32),
            ]
        )
        k_cache = k_cache.to(q.dtype) if not is_nmz_fp8(k_cache.dtype) else k_cache
        v_cache = v_cache.to(q.dtype) if not is_nmz_fp8(v_cache.dtype) else v_cache
        # BSHD paged caches are [pages, page_size, heads, head_dim], whereas
        # the HND/BHSD layout is [pages, heads, page_size, head_dim].
        page_size = k_cache.shape[1] if layout != "bhsd" else k_cache.shape[2]
        if (
            _is_hcu
            and _use_triton_vllm_fa
            and is_nmz_fp8(k_cache.dtype)
            and not return_softmax_lse
        ):
            result = triton_vllm_flash_attn_varlen_func(
                q=q,
                k=k_cache,
                v=v_cache,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=cache_seqlens,
                max_seqlen_k=page_table.shape[1] * page_size,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                block_table=page_table,
                fa_version=ver,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                layout="bshd" if layout is None else layout,
            )
            return _apply_flash_attn_varlen_out(result, out, return_softmax_lse)
        if _is_hcu:
            result = varlen_fwd_unified_interface(
                q=q,
                k=k_cache,
                v=v_cache,
                cu_seqlens_q=cu_seqlens_q,
                seqused_k=cache_seqlens,
                block_table=page_table,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=page_table.shape[1] * page_size,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                out=out,
                return_softmax_lse=return_softmax_lse,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                s_aux=sinks,
                layout="bshd" if layout is None else layout,
            )
        else:
            result = flash_attn_varlen_func_interface(
                q=q,
                k=k_cache,
                v=v_cache,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=page_table.shape[1] * page_size,
                seqused_k=cache_seqlens,
                block_table=page_table,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                num_splits=num_splits,
                return_softmax_lse=return_softmax_lse,
                fa_version=ver,
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
    if WELM_Q_PRESCALED and softmax_scale is not None:
        softmax_scale = q.shape[-1] ** -0.5
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

    if _is_hcu and _use_welm_decode_attention(
        q, k_cache, v_cache, page_table, window_size, softcap, return_softmax_lse
    ):
        from sglang.srt.layers.welmv4_fast_op import welm_decode_attention

        B, _, H, D = q.shape
        window_left = (
            window_size[0]
            if window_size is not None and tuple(window_size) != (-1, -1)
            else -1
        )
        out = welm_decode_attention(
            q.view(B, H, D),
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            softmax_scale if softmax_scale is not None else D**-0.5,
            page_table.shape[1] * k_cache.shape[2],
            sinks=sinks,
            window_left=window_left,
            **({"chunk_pages": 1, "num_warps": 2} if window_left >= 0 else {}),
        )
        return out.view(B, 1, H, D)
    if _is_hcu:
        q, softmax_scale = _hcu_prescale_q_for_paged_attention(q, softmax_scale)
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
        s_aux=sinks,
        num_splits=num_splits,
    )


# Set by models that already folded softmax_scale / sqrt(1/head_dim) into q
# (WeLM fuses it into its RoPE kernel); the HCU paged-attention wrappers then
# use the default scale instead of rescaling q again.
WELM_Q_PRESCALED = False


_HCU_SHORT_Q_AS_DECODE_MAX = int(
    os.environ.get("SGLANG_HCU_SHORT_Q_AS_DECODE_MAX", "8")
)
_WELM_PFX_ATTN = get_bool_env_var("WELM_PFX_ATTN", default="true")
_WELM_PFX_ATTN_MAX_Q = int(os.environ.get("WELM_PFX_ATTN_MAX_Q", "256"))


_WELM_DEC_ATTN = get_bool_env_var("WELM_DEC_ATTN", default="true")
# Verify / draft-extend: use the paged kernel's multi-token (mtp) mode.
_WELM_PA_MTP = get_bool_env_var("WELM_PA_MTP", default="true")
# Decode rows use the vendor paged-attention kernel unless it would drop
# tokens (see _hcu_paged_attention_covers); batch sizes up to this always use
# welm_decode_attention. In serving it is not faster than the vendor kernel,
# so by default it is only the correctness fallback.
_WELM_DEC_ATTN_MAX_BS = int(os.environ.get("WELM_DEC_ATTN_MAX_BS", "0"))
# Sliding-window decode rows use welm_decode_attention up to this batch size.
_WELM_DEC_ATTN_SWA_MAX_BS = int(os.environ.get("WELM_DEC_ATTN_SWA_MAX_BS", "24"))


def _hcu_paged_attention_covers(num_blocks, page_size, num_seqs, max_len):
    """Whether flash_attn_cuda.paged_attention attends to all max_len tokens.

    Its host dispatch (src/paged_attention.hip) replaces any max_seq_len <= 10
    or > 8192 with num_blocks*block_size/num_seqs + 8192 and sizes the
    partition grid from that; tokens past it are silently dropped.
    """
    return (num_blocks * page_size) // num_seqs + 8192 >= max_len


def _use_welm_decode_attention(
    q, k_cache, v_cache, page_table, window_size, softcap, return_softmax_lse
):
    if not _WELM_DEC_ATTN or page_table is None or return_softmax_lse or softcap:
        return False
    windowed = window_size is not None and tuple(window_size) != (-1, -1)
    if windowed and not (
        window_size[0] >= 0
        and window_size[1] in (0, -1)
        and q.shape[0] <= _WELM_DEC_ATTN_SWA_MAX_BS
    ):
        return False
    if q.dim() != 4 or q.shape[1] != 1 or q.dtype != torch.bfloat16:
        return False
    if k_cache.dtype != torch.bfloat16 or k_cache.dim() != 4 or v_cache.dim() != 4:
        return False
    # HCU legacy layout: K [blocks, KVH, PAGE, D], V [blocks, KVH, D, PAGE].
    page, d = k_cache.shape[2], k_cache.shape[3]
    if tuple(v_cache.shape[2:]) != (d, page) or q.shape[3] != d:
        return False
    bs = q.shape[0]
    if windowed:
        # The vendor kernel runs one workgroup per row for a short window.
        return True
    if bs <= _WELM_DEC_ATTN_MAX_BS:
        return True
    return not _hcu_paged_attention_covers(
        k_cache.shape[0], page, bs, page_table.shape[1] * page
    )


def _hcu_prescale_q_for_paged_attention(q, softmax_scale):
    """Fold a non-default softmax scale into q for the HCU paged-attention kernel.

    The vendor ``flash_attn_cuda.paged_attention`` kernel (used by
    ``vllm_flash_attn_with_kvcache`` and by the varlen interface whenever
    ``max_seqlen_q < 16``) ignores ``softmax_scale`` and always uses
    1/sqrt(head_dim). Scaling q instead gives identical scores (and LSE).
    """
    if softmax_scale is None or q.dtype not in (torch.float16, torch.bfloat16):
        return q, softmax_scale
    default = q.shape[-1] ** -0.5
    if abs(softmax_scale - default) <= 1e-6 * default:
        return q, softmax_scale
    return (q.float() * (softmax_scale / default)).to(q.dtype), default


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

    # gfx938 FP8 varlen attention dequantizes with these scales. The kernel
    # requires shape (batch, num_kv_heads); a missing or 1-element scale makes
    # it treat the FP8 prefill K/V as unscaled bits and the logits collapse.
    if is_nmz_fp8(k.dtype) and not IS_SLIMQUANT_W4A8:
        descale_shape = (cu_seqlens_q.shape[0] - 1, k.shape[-2])

        def _as_descale(scale):
            if scale is None or tuple(scale.shape) != descale_shape:
                return torch.ones(descale_shape, dtype=torch.float32, device=k.device)
            return scale

        q_descale = _as_descale(q_descale)
        k_descale = _as_descale(k_descale)
        v_descale = _as_descale(v_descale)
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


def _hcu_paged_prefill_no_sync(
    q, k, v, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
    softmax_scale, causal, window_size, block_table, total_k, sinks,
    return_softmax_lse=False, out=None, host_seq_lens=None,
):
    """The vendor paged-prefill path (paged -> contiguous KV, then varlen_fwd)
    with the workspace sized from a host-side KV total.

    flash_attn's wrapper computes ``seqused_k.sum()`` and branches on it in
    every layer, which blocks the CPU until all queued GPU work drains. Same
    kernels and arguments otherwise, so results are bit-identical.
    """
    from flash_attn.flash_attn_interface import flash_attn_cuda

    workspace_tokens = max(int(total_k), 256 * 1024)
    nheads_kv = k.shape[1]
    kt = torch.empty((workspace_tokens, nheads_kv, q.shape[-1]), device=q.device, dtype=k.dtype)
    vt = torch.empty((workspace_tokens, nheads_kv, v.shape[-2]), device=q.device, dtype=v.dtype)
    cu_k = torch.empty_like(cu_seqlens_q)
    if (
        _WELM_FAST_P2C
        and k.dtype == torch.bfloat16
        and k.dim() == 4
        and v.dim() == 4
        and tuple(v.shape[2:]) == (k.shape[3], k.shape[2])
        and cu_k.dtype == torch.int32
        and seqused_k.dtype == torch.int32
    ):
        from sglang.srt.layers.welmv4_fast_op import welm_pagedkv_to_contiguous

        welm_pagedkv_to_contiguous(
            kt, vt, k, v, block_table, seqused_k, cu_k, max_seqlen_k
        )
    else:
        flash_attn_cuda.pagedkv_to_contiguouskv(
            kt, vt, k, v, block_table, seqused_k, cu_k, max_seqlen_k, 0
        )
    if window_size is None:
        window_size = (-1, -1)
    fa256 = _welm_fa256()
    if (
        fa256 is not None
        and causal
        and window_size[0] < 0
        and q.dtype == torch.bfloat16
        and q.shape[-1] == 256
        and vt.shape[-1] == 256
    ):
        # Faster hdim-256 tile (128x64, 2 stages) for full-attention layers.
        if host_seq_lens is not None and not return_softmax_lse and _WELM_FA_SPLIT:
            from sglang.srt.layers.attention import welm_fa_split

            q_lens, k_lens = host_seq_lens
            if welm_fa_split.should_split(q_lens, k_lens):
                return welm_fa_split.fa256_split_prefill(
                    fa256, q, kt, vt, q_lens, k_lens,
                    softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5,
                    sinks, _WELM_FA256_CFG, out=out,
                )
        o, lse = fa256.varlen_fwd256(
            q, kt, vt, out, cu_seqlens_q, cu_k, max_seqlen_q, max_seqlen_k,
            softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5,
            True, -1, -1, sinks, _WELM_FA256_CFG,
        )
        if return_softmax_lse:
            return o, lse
        return o
    res = flash_attn_cuda.varlen_fwd(
        q, kt, vt, out, cu_seqlens_q, cu_k, None, None, None, None,
        max_seqlen_q, max_seqlen_k, 0.0,
        softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5,
        False, causal, window_size[0], window_size[1], 0.0, False,
        None, None, None, None, sinks,
    )
    if return_softmax_lse:
        return res[0], res[5]
    return res[0]


_WELM_FA256_CFG = int(os.environ.get("SGLANG_WELM_FA256_CFG", "20"))
# Triton paged->contiguous KV copy (bit-identical, ~4x faster).
_WELM_FAST_P2C = get_bool_env_var("WELM_FAST_P2C", default="true")
# Split-KV prefill for short new segments over long cached prefixes.
_WELM_FA_SPLIT = get_bool_env_var("WELM_FA_SPLIT", default="true")
_welm_fa256_mod = None


class _VendorFA256:
    """fa256-compatible wrapper around the vendor varlen_fwd. Builds that carry
    the WeLM patches (paged_attention_partial, dim-256 sink epilogue) are
    1.3-3.8x faster than welm_fa256.so on WeLM prefill shapes."""

    @staticmethod
    def varlen_fwd256(q, k, v, out, cu_q, cu_k, max_q, max_k, scale, causal,
                      wl, wr, sinks, cfg):
        from flash_attn.flash_attn_interface import flash_attn_cuda

        r = flash_attn_cuda.varlen_fwd(
            q, k, v, out, cu_q, cu_k, None, None, None, None, max_q, max_k, 0.0,
            scale, False, causal, wl, wr, 0.0, False, None, None, None, None, sinks,
        )
        return r[0], r[5]


def _vendor_fa_has_welm_patches():
    try:
        from flash_attn.flash_attn_interface import flash_attn_cuda

        return hasattr(flash_attn_cuda, "paged_attention_partial")
    except Exception:
        return False


# auto: use the vendor kernel when the installed flash_attn carries the WeLM patches.
_WELM_VENDOR_FA = os.environ.get("WELM_VENDOR_FA", "auto").lower()


def _welm_fa256():
    """Standalone hdim-256 bf16 forward (welm_fa256.so); None when disabled/unavailable.
    SGLANG_WELM_FA256_CFG: 20 = 128x32 tile (fastest, bit-exact with cfg 1), 4 = 128x64,
    1 = 64x32 (bit-exact with vendor), 0 = off."""
    global _welm_fa256_mod
    if _welm_fa256_mod is None:
        _welm_fa256_mod = False
        if _WELM_VENDOR_FA in ("1", "true") or (
            _WELM_VENDOR_FA == "auto" and _vendor_fa_has_welm_patches()
        ):
            _welm_fa256_mod = _VendorFA256
            return _welm_fa256_mod
        if _WELM_FA256_CFG in (1, 4, 20):
            try:
                from sglang.srt.layers.attention import welm_fa256

                _welm_fa256_mod = welm_fa256
            except ImportError as e:
                logger.warning(f"welm_fa256 unavailable, using vendor kernel: {e}")
    return _welm_fa256_mod or None


def _can_use_hcu_prefill_no_sync(q, k, v, max_seqlen_q, block_table, seqused_k,
                                 total_k_hint, layout, q_descale, k_descale, v_descale):
    return (
        _is_hcu
        and not _use_triton_vllm_fa
        and _HCU_PREFILL_NO_SYNC
        and isinstance(total_k_hint, int)
        and total_k_hint > 0
        and layout == "legacy_bhsd"
        and block_table is not None
        and seqused_k is not None
        and max_seqlen_q >= 16
        and q.dtype in (torch.bfloat16, torch.float16)
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and k.dim() == 4
        and k.shape[-2] == 64
        and q.shape[-1] == v.shape[-2]
        and q_descale is None
        and k_descale is None
        and v_descale is None
    )


_HCU_PREFILL_NO_SYNC = get_bool_env_var("SGLANG_HCU_PREFILL_NO_SYNC", default="true")


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
    out=None,
    total_k_hint: Optional[int] = None,
    sinks=None,
    host_seq_lens=None,
):
    if WELM_Q_PRESCALED and softmax_scale is not None:
        softmax_scale = q.shape[-1] ** -0.5
    # Prefill already has an exact host KV-length sum. The vendor paged
    # prefill wrapper recomputes seqused_k.sum() and branches on that device
    # scalar for every layer, serializing all preceding GPU work. For the
    # validated Qwen DFLASH layout, use the same conversion and FA kernels
    # with the host count instead. Decode and all other layouts stay unchanged.
    # E4M3 is intentionally excluded from this shortcut: retain its existing
    # vendor/full-attention or Triton/SWA dispatch, including descale arguments.
    # --kv-cache-dtype fp8_e4m3 remains supported; it is not coerced to E5M2.
    if (
        _is_hcu
        and not _use_triton_vllm_fa
        and get_bool_env_var("SGLANG_HCU_DFLASH_PREFILL_KV_HINT", default="true")
        and get_spec().speculative_algorithm == "DFLASH"
        and isinstance(total_k_hint, int)
        and total_k_hint > 0
        and layout == "legacy_bhsd"
        and block_table is not None
        and seqused_k is not None
        and max_seqlen_q >= 16
        and q.dtype == torch.bfloat16
        and k.dtype == v.dtype
        and k.dtype in (torch.bfloat16, torch.float8_e5m2)
        and q.shape[1:] == (8, 256)
        and k.shape[1:] == (1, 64, 256)
        and v.shape[1:] == (1, 256, 64)
        and causal
        and window_size == (-1, -1)
        and "gfx936" in torch.cuda.get_device_properties(q.device).gcnArchName
    ):
        from flash_attn.flash_attn_interface import flash_attn_cuda

        # Retain the vendor's minimum workspace size (smaller buffers can
        # fault in this installed runtime). Only the size calculation changes.
        workspace_tokens = max(total_k_hint, 256 * 1024)
        kt = torch.empty((workspace_tokens, 1, 256), device=q.device, dtype=q.dtype)
        vt = torch.empty_like(kt)
        cu_k = torch.empty_like(cu_seqlens_q)
        flash_attn_cuda.pagedkv_to_contiguouskv(
            kt, vt, k, v, block_table, seqused_k, cu_k, max_seqlen_k, 0
        )
        return flash_attn_cuda.varlen_fwd(
            q, kt, vt, out, cu_seqlens_q, cu_k, None, None, None, None,
            max_seqlen_q, max_seqlen_k, 0.0,
            softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5,
            False, causal, -1, -1, 0.0, False,
            q_descale, k_descale, v_descale, None, None,
        )[0]

    use_hcu_fp8_swa_fallback = (
        _is_hcu
        and k.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        and window_size is not None
        and window_size[0] >= 0
    )
    # The dedicated model switch isolates vendor-specific behavior. The block
    # size comes from published service configuration, including its alias.
    use_dflash_native_block = (
        _is_hcu
        and not _use_triton_vllm_fa
        and get_bool_env_var("SGLANG_USE_QWEN_DFLASH2")
        and get_spec().speculative_algorithm == "DFLASH"
        and (
            get_spec().speculative_num_draft_tokens == max_seqlen_q
            or get_spec().speculative_dflash_block_size == max_seqlen_q
        )
        and layout == "legacy_bhsd"
        and block_table is not None
        and seqused_k is not None
        and max_seqlen_q in (4, 8, 16)
        and q.shape[0] == (cu_seqlens_q.numel() - 1) * max_seqlen_q
        and q.dtype == torch.bfloat16
        and k.dtype == v.dtype
        and k.dtype
        in (
            torch.bfloat16,
            torch.float8_e5m2,
        )
        and k.shape[2] == v.shape[3] == 64
        and q.shape[2] == k.shape[3] == v.shape[2]
        and "gfx936" in torch.cuda.get_device_properties(q.device).gcnArchName
    )
    use_dflash_native_draft = (
        use_dflash_native_block
        and q.shape[2] == 128
        and q.shape[1] * max_seqlen_q <= k.shape[1] * 64
        and window_size in ((-1, -1), (4095, 0), (4095, 4095))
    )
    use_dflash_native_target = (
        use_dflash_native_block
        and max_seqlen_q in (8, 16)
        and q.shape[2] == 256
        and q.shape[1] == 8
        and k.shape[1] == v.shape[1] == 1
        and causal
        and window_size == (-1, -1)
    )
    if use_dflash_native_draft or use_dflash_native_target:
        # Call the existing vendor kernel, without extending the installed
        # flash_attn Python API. Keep all model/shape guards above local.
        from flash_attn.flash_attn_interface import flash_attn_cuda

        batch_size = cu_seqlens_q.numel() - 1
        kv_bound = max(max_seqlen_k or 0, block_table.shape[1] * k.shape[2])
        scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
        if out is None:
            out = torch.empty_like(q)
        queries = q.reshape(batch_size, max_seqlen_q, *q.shape[1:])
        if use_dflash_native_draft:
            out.zero_()  # Native attention skips empty graph-padding rows.
            flash_attn_cuda.paged_attention(
                out,
                queries,
                k,
                v,
                scale,
                block_table,
                seqused_k,
                None,
                "",
                q_descale,
                k_descale,
                v_descale,
                kv_bound,
                None,
                0,
                window_size[0],
                window_size[1],
                causal,
            )
        else:
            outputs = out.reshape(batch_size, max_seqlen_q, *q.shape[1:])
            for begin in range(0, max_seqlen_q, 4):
                chunk_q = queries[:, begin : begin + 4].contiguous()
                chunk_out = torch.zeros_like(chunk_q)
                # Preserve bottom-right causal positions within the full block.
                chunk_lengths = (seqused_k - (max_seqlen_q - 4 - begin)).clamp_min(0)
                flash_attn_cuda.paged_attention(
                    chunk_out,
                    chunk_q,
                    k,
                    v,
                    scale,
                    block_table,
                    chunk_lengths,
                    None,
                    "",
                    q_descale,
                    k_descale,
                    v_descale,
                    kv_bound,
                    None,
                    0,
                    -1,
                    -1,
                    True,
                )
                outputs[:, begin : begin + 4].copy_(chunk_out)
        return out
    if (
        use_hcu_fp8_swa_fallback
        and not use_dflash_native_draft
        and not _use_triton_vllm_fa
        and get_bool_env_var("SGLANG_USE_QWEN_DFLASH2")
        and get_spec().speculative_algorithm == "DFLASH"
        and layout == "legacy_bhsd"
        and q.dtype == torch.bfloat16
        and k.dtype == v.dtype == torch.float8_e5m2
        and q.shape[-1] == k.shape[-1] == v.shape[-2] == 128
        and k.shape[2] == v.shape[3] == 64
        and 1 <= max_seqlen_q <= 16
        and q.shape[0] == (cu_seqlens_q.numel() - 1) * max_seqlen_q
        and 0 <= window_size[0] <= 8192
    ):
        from sglang.srt.layers.attention.hcu_native_swa import (
            native_hcu_sliding_attention,
        )

        return native_hcu_sliding_attention(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            window_size=window_size,
            block_table=block_table,
            softmax_scale=softmax_scale,
            causal=causal,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            out=out,
        )
    if (
        _is_hcu
        and (_use_triton_vllm_fa or use_hcu_fp8_swa_fallback)
        and not use_dflash_native_draft
    ):
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
            out=out,
        )

    # HCU paged MTP folds query tokens into heads and supports at most
    # 64 query heads per KV head. Split uniform causal verification blocks
    # without changing their bottom-right causal alignment or the KV cache.
    if (
        _is_hcu
        and envs.SGLANG_USE_QWEN_DSPARK.get()
        and get_spec().speculative_algorithm == "DSPARK"
        and layout == "legacy_bhsd"
        and k.shape[2] == 64
        and q.shape[2] == v.shape[2]
        and causal
        and window_size == (-1, -1)
        and 1 < max_seqlen_q <= 16
        and q.shape[0] == (cu_seqlens_q.numel() - 1) * max_seqlen_q
        and q.shape[1] * max_seqlen_q > k.shape[1] * 64
        and q.shape[1] <= k.shape[1] * 64
    ):
        batch_size = cu_seqlens_q.numel() - 1
        chunk_size = min(16, k.shape[1] * 64 // q.shape[1])
        queries = q.reshape(batch_size, max_seqlen_q, *q.shape[1:])
        chunks = []
        for begin in range(0, max_seqlen_q, chunk_size):
            end = min(begin + chunk_size, max_seqlen_q)
            chunk_q = queries[:, begin:end].contiguous().flatten(0, 1)
            chunk_cu = torch.arange(
                batch_size + 1, device=cu_seqlens_q.device, dtype=cu_seqlens_q.dtype
            ) * (end - begin)
            chunk_lengths = (seqused_k - (max_seqlen_q - end)).clamp_min(0)
            chunk_out = vllm_flash_attn_varlen_func_interface(
                q=chunk_q,
                k=k,
                v=v,
                cu_seqlens_q=chunk_cu,
                max_seqlen_q=end - begin,
                seqused_k=chunk_lengths,
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
            chunks.append(
                chunk_out.reshape(batch_size, end - begin, *chunk_out.shape[1:])
            )
        result = torch.cat(chunks, dim=1).flatten(0, 1)
        if out is not None:
            out.copy_(result)
            return out
        return result

    # Short queries over a long paged prefix (prefill with prefix-cache hits):
    # split-KV kernel instead of the query-block-parallel varlen kernel, which
    # leaves most of the device idle here.  Verify-sized queries keep the
    # short-q-as-decode path below.
    if (
        _is_hcu
        and _WELM_PFX_ATTN
        and causal
        and layout == "legacy_bhsd"
        and tuple(window_size) == (-1, -1)
        and block_table is not None
        and seqused_k is not None
        and (
            _HCU_SHORT_Q_AS_DECODE_MAX < max_seqlen_q <= _WELM_PFX_ATTN_MAX_Q
            # One query row per request: 1-token prefills and the KV-mirror
            # layers (last token per request against the full sequence).
            or max_seqlen_q == 1
            # Under a prefill CUDA graph the vendor varlen path would bake the
            # capture-time KV total (total_k_hint) into the launch; this
            # kernel reads every length on device, so prefix replays are exact.
            or torch.cuda.is_current_stream_capturing()
        )
        and q.dtype == torch.bfloat16
        and k.dtype == v.dtype == torch.bfloat16
        and q_descale is None
        and k.dim() == 4
        and q.shape[1] % k.shape[1] == 0
    ):
        from sglang.srt.layers.welmv4_fast_op import welm_prefix_attention

        return welm_prefix_attention(
            q, k, v, block_table, seqused_k, cu_seqlens_q, max_seqlen_q,
            softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5,
            sinks=sinks, out=out,
        )

    # The HCU paged "MTP" branch (selected for uniform 1 < max_seqlen_q < 16
    # batches) folds query tokens into heads and rejects or ignores
    # per-head attention sinks. ``max_seqlen_q`` is only an upper bound for
    # the varlen kernel, so raising it routes these batches to the regular
    # paged-prefill kernel, which applies sinks correctly.
    # Short uniform queries (target verify, draft extend): run each query
    # token as its own single-token decode row against the same page-table
    # row with cache length prefix + i + 1. This is exact bottom-right causal
    # attention, supports sinks and windows, and avoids the slow
    # paged->contiguous KV conversion of the varlen path (~8x faster here).
    if (
        _is_hcu
        and causal
        and layout == "legacy_bhsd"
        and 1 < max_seqlen_q <= _HCU_SHORT_Q_AS_DECODE_MAX
        and block_table is not None
        and seqused_k is not None
        and q.shape[0] == (cu_seqlens_q.shape[0] - 1) * max_seqlen_q
        and q.dtype in (torch.float16, torch.bfloat16)
        and k.dtype == q.dtype
    ):
        n = max_seqlen_q
        bs = cu_seqlens_q.shape[0] - 1
        if (
            _WELM_PA_MTP
            and q.dtype == torch.bfloat16
            and k.dim() == 4
            and q.shape[1] * n <= k.shape[1] * 64
            # The multi-token mode applies per-folded-head sinks only for
            # n == 2; for n > 2 it silently drops them (kt2/pa_mtp_map.py).
            and (sinks is None or n == 2)
            and _hcu_paged_attention_covers(
                k.shape[0], k.shape[2], bs, block_table.shape[1] * k.shape[2]
            )
        ):
            # The paged kernel's multi-token mode reads each row's KV once for
            # all n queries (bottom-right causal); the
            # expansion below reads it n times.
            result = vllm_flash_attn_with_kvcache(
                q=q.view(bs, n, q.shape[1], q.shape[2]),
                k_cache=k,
                v_cache=v,
                page_table=block_table,
                # graph-padding rows may be shorter than n
                cache_seqlens=torch.clamp(seqused_k, min=n),
                softmax_scale=softmax_scale,
                causal=True,
                window_size=window_size,
                sinks=sinks.repeat(n) if sinks is not None else None,
            ).view(q.shape[0], q.shape[1], v.shape[2])
            if out is not None:
                out.copy_(result)
                return out
            return result
        from sglang.srt.layers.welmv4_fast_op import short_q_expand

        # Graph-padding rows may be shorter than n; keep them >= 1.
        row_lens, row_page_table = short_q_expand(seqused_k, block_table, n)
        result = vllm_flash_attn_with_kvcache(
            q=q.view(bs * n, 1, q.shape[1], q.shape[2]),
            k_cache=k,
            v_cache=v,
            page_table=row_page_table,
            cache_seqlens=row_lens,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=window_size,
            sinks=sinks,
        ).view(q.shape[0], q.shape[1], v.shape[2])
        if out is not None:
            out.copy_(result)
            return out
        return result

    if _is_hcu and sinks is not None and 1 < max_seqlen_q < 16:
        max_seqlen_q = 16
    # Below 16 query tokens the vendor varlen interface dispatches to the
    # paged-attention kernel, which ignores softmax_scale.
    if _is_hcu and max_seqlen_q < 16:
        q, softmax_scale = _hcu_prescale_q_for_paged_attention(q, softmax_scale)

    # The HCU sliding-window paged-prefill kernel silently ignores s_aux.
    # Apply the sinks exactly from the softmax LSE instead: a softmax over
    # [scores, sink] equals the sink-free output scaled by sigmoid(lse - sink).
    if (
        _is_hcu
        and sinks is not None
        and max_seqlen_q > 1
        and window_size is not None
        and window_size[0] >= 0
    ):
        if _can_use_hcu_prefill_no_sync(
            q, k, v, max_seqlen_q, block_table, seqused_k, total_k_hint,
            layout, q_descale, k_descale, v_descale,
        ):
            result, lse = _hcu_paged_prefill_no_sync(
                q, k, v, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
                softmax_scale, causal, window_size, block_table, total_k_hint,
                None, return_softmax_lse=True,
            )
        else:
            result, lse = vllm_flash_attn_varlen_func_interface(
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
                return_softmax_lse=True,
            )[:2]
        # lse: [num_heads, total_q] -> [total_q, num_heads, 1]
        scale = torch.sigmoid(lse.t().float() - sinks.float()[None, :])
        return (result.float() * scale.unsqueeze(-1)).to(result.dtype)

    if _can_use_hcu_prefill_no_sync(
        q, k, v, max_seqlen_q, block_table, seqused_k, total_k_hint,
        layout, q_descale, k_descale, v_descale,
    ):
        return _hcu_paged_prefill_no_sync(
            q, k, v, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
            softmax_scale, causal, window_size, block_table, total_k_hint,
            sinks, out=out, host_seq_lens=host_seq_lens,
        )
    # Capture may leave the host bound at zero; live lengths remain on GPU.
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
        s_aux=sinks,
    )
