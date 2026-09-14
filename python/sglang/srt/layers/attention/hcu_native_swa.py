"""Graph-safe FP8 sliding-window adapter for HCU native FlashAttention.

The page-64 vendor wrapper materializes a full contiguous KV cache using a
GPU-derived allocation size (and underallocates for some long contexts).
Stage only the union of keys visible to the query block instead. Allocation
sizes depend on tensor shapes and the window, while live lengths stay on GPU.
Attention itself is evaluated by the vendor native varlen kernel.
"""

import logging
from functools import lru_cache

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=1)
def _log_native_path():
    logging.getLogger(__name__).info(
        "DFLASH FP8 SWA: native FlashAttention with bounded BF16 KV staging"
    )


@triton.jit
def _gather_window(
    K,
    V,
    TABLE,
    SEQ,
    CUQ,
    CUK,
    KS,
    VS,
    KO,
    VO,
    CAP: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    LEFT: tl.constexpr,
    TB0: tl.constexpr,
    TB1: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    K3: tl.constexpr,
    V0: tl.constexpr,
    V1: tl.constexpr,
    V2: tl.constexpr,
    V3: tl.constexpr,
    B: tl.constexpr,
):
    batch = tl.program_id(1)
    x = tl.program_id(0) * B + tl.arange(0, B)
    token = x // (H * D)
    head = x // D % H
    dim = x % D
    n = tl.load(SEQ + batch)
    qlen = tl.load(CUQ + batch + 1) - tl.load(CUQ + batch)
    start = tl.maximum(n - qlen - LEFT, 0)
    count = n - start
    valid = (x < CAP * H * D) & (token < count)
    logical = start + token
    physical = tl.load(TABLE + batch * TB0 + (logical // PAGE) * TB1, valid, 0)
    slot = logical % PAGE
    kval = tl.load(K + physical * K0 + head * K1 + slot * K2 + dim * K3, valid, 0.0).to(
        tl.float32
    )
    vval = tl.load(V + physical * V0 + head * V1 + dim * V2 + slot * V3, valid, 0.0).to(
        tl.float32
    )
    kval *= tl.load(KS + batch * H + head)
    vval *= tl.load(VS + batch * H + head)
    base = tl.load(CUK + batch)
    dst = (base + token) * H * D + head * D + dim
    tl.store(KO + dst, kval, valid)
    tl.store(VO + dst, vval, valid)


def _scale_rows(scale, batch, heads, device):
    if scale is None:
        return torch.ones((batch, heads), dtype=torch.float32, device=device)
    scale = scale.float()
    if scale.numel() == 1:
        scale = scale.reshape(1, 1)
    elif scale.ndim == 1:
        scale = scale.reshape(1, -1)
    return scale.expand(batch, heads).contiguous()


def native_hcu_sliding_attention(
    *,
    q,
    k,
    v,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    window_size,
    block_table,
    softmax_scale,
    causal,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    out=None
):
    # The caller admits uniform query blocks and the asymmetric HCU KV layout.
    # All sizes below are host shape constants, including under graph capture.
    from flash_attn.flash_attn_interface import flash_attn_cuda

    _log_native_path()
    batch = cu_seqlens_q.numel() - 1
    heads = k.shape[1]
    dim = q.shape[-1]
    # Removing a common left prefix preserves bottom-right query/key alignment.
    # Even noncausal draft queries need only this union of their local windows.
    capacity = int(window_size[0]) + max_seqlen_q
    lengths = torch.minimum(
        seqused_k, cu_seqlens_q[1:] - cu_seqlens_q[:-1] + window_size[0]
    )
    cu_k = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=q.device),
            lengths.cumsum(0, dtype=torch.int32),
        )
    )
    staged_k = torch.empty(
        (batch * capacity, heads, dim), dtype=torch.bfloat16, device=q.device
    )
    staged_v = torch.empty_like(staged_k)
    ks = _scale_rows(k_descale, batch, heads, q.device)
    vs = _scale_rows(v_descale, batch, heads, q.device)
    _gather_window[(triton.cdiv(capacity * heads * dim, 256), batch)](
        k,
        v,
        block_table,
        seqused_k,
        cu_seqlens_q,
        cu_k,
        ks,
        vs,
        staged_k,
        staged_v,
        capacity,
        heads,
        dim,
        k.shape[2],
        window_size[0],
        *block_table.stride(),
        *k.stride(),
        *v.stride(),
        256,
    )
    q_native = q
    if q_descale is not None:
        scale = q_descale
        nheads = 1 if scale.numel() == 1 else scale.shape[-1]
        qs = _scale_rows(scale, batch, nheads, q.device)
        qs = qs.repeat_interleave(q.shape[1] // nheads, dim=1)
        q_native = (
            (
                q.view(batch, max_seqlen_q, q.shape[1], dim).float()
                * qs[:, None, :, None]
            )
            .to(torch.bfloat16)
            .view_as(q)
        )
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    return flash_attn_cuda.varlen_fwd(
        q_native,
        staged_k,
        staged_v,
        out,
        cu_seqlens_q,
        cu_k,
        None,
        None,
        None,
        None,
        max_seqlen_q,
        capacity,
        0.0,
        softmax_scale,
        False,
        causal,
        window_size[0],
        window_size[1],
        0.0,
        False,
        None,
        None,
        None,
        None,
        None,
    )[0]
