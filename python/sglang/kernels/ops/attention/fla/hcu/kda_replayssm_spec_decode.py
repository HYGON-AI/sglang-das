# SPDX-License-Identifier: Apache-2.0
"""ReplaySSM accepted-prefix commit for KDA speculative verification.

The recurrent verify kernel still computes every speculative output. It also
stages the current window's raw ``v``, pre-normalization ``k``, per-K fp32 gate,
and fp32 beta. After sampling, this module replays only the accepted prefix into
the persistent fp32 state. The next verify step overwrites the staging window.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def kda_replayssm_exact_fold_kernel(
    h0,
    rawv_cache,
    rawk_cache,
    g_cache,
    beta_cache,
    ssm_state_indices,
    accept_lens,
    mamba_track_indices,
    mamba_steps_to_track,
    stride_state_slot: tl.constexpr,
    stride_rawv_slot: tl.constexpr,
    stride_rawk_slot: tl.constexpr,
    stride_g_slot: tl.constexpr,
    stride_beta_slot: tl.constexpr,
    stride_state_layer: tl.constexpr,
    stride_rawv_layer: tl.constexpr,
    stride_rawk_layer: tl.constexpr,
    stride_g_layer: tl.constexpr,
    stride_beta_layer: tl.constexpr,
    stride_indices: tl.constexpr,
    stride_accept: tl.constexpr,
    stride_track: tl.constexpr,
    stride_track_step: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    WINDOW_LEN: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    HAS_TRACK: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_n = tl.program_id(1)
    i_hvl = tl.program_id(2)
    i_layer = (i_hvl // HV).to(tl.int64)
    i_hv = i_hvl % HV
    i_h = i_hv // (HV // H)

    h0 += i_layer * stride_state_layer
    rawv_cache += i_layer * stride_rawv_layer
    rawk_cache += i_layer * stride_rawk_layer
    g_cache += i_layer * stride_g_layer
    beta_cache += i_layer * stride_beta_layer

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices).to(tl.int64)
    if state_idx <= NULL_BLOCK_ID:
        return
    n_commit = tl.load(accept_lens + i_n * stride_accept).to(tl.int32)
    if n_commit <= 0:
        return

    if HAS_TRACK:
        track_idx = tl.load(mamba_track_indices + i_n * stride_track).to(tl.int64)
        track_step = tl.load(mamba_steps_to_track + i_n * stride_track_step).to(
            tl.int32
        )
    else:
        track_idx = NULL_BLOCK_ID
        track_step = -1

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    p_h0 = (
        h0
        + state_idx * stride_state_slot
        + i_hv * V * K
        + o_v[None, :] * K
        + o_k[:, None]
    )
    h = tl.load(p_h0, mask=mask_h, other=0.0).to(tl.float32)

    # Match fused_sigmoid_gating_delta_rule_update_kernel_opt exactly. In
    # particular, HCU's recurrent kernel normalizes K with rsqrt.
    for t in range(0, n_commit):
        step = t.to(tl.int64)
        rawk = tl.load(
            rawk_cache
            + state_idx * stride_rawk_slot
            + (i_h * WINDOW_LEN + step) * K
            + o_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        rawv = tl.load(
            rawv_cache
            + state_idx * stride_rawv_slot
            + (i_hv * WINDOW_LEN + step) * V
            + o_v,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        gk = tl.load(
            g_cache + state_idx * stride_g_slot + (i_hv * WINDOW_LEN + step) * K + o_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        beta = tl.load(
            beta_cache + state_idx * stride_beta_slot + i_hv * WINDOW_LEN + step
        ).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            rawk *= tl.rsqrt(tl.sum(rawk * rawk) + 1e-6)
        h *= tl.exp(gk[:, None])
        rawv -= tl.sum(h * rawk[:, None], axis=0)
        rawv *= beta
        h += rawk[:, None] * rawv[None, :]

        if HAS_TRACK:
            if (t == track_step) and (track_idx > NULL_BLOCK_ID):
                tl.store(
                    h0
                    + track_idx * stride_state_slot
                    + i_hv * V * K
                    + o_v[None, :] * K
                    + o_k[:, None],
                    h.to(h0.dtype.element_ty),
                    mask=mask_h,
                )

    tl.store(p_h0, h.to(p_h0.dtype.element_ty), mask=mask_h)


def commit_kda_replayssm_spec_all_layers(
    *,
    checkpoint_state: torch.Tensor,
    rawv_cache: torch.Tensor,
    rawk_cache: torch.Tensor,
    g_cache: torch.Tensor,
    beta_cache: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    accept_lens: torch.Tensor,
    mamba_track_indices: torch.Tensor | None = None,
    mamba_steps_to_track: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    null_block_id: int = -1,
) -> None:
    """Replay every layer's accepted window into its persistent checkpoint."""
    if checkpoint_state.ndim != 5:
        raise ValueError("KDA ReplaySSM checkpoint must have rank 5")
    if (
        rawv_cache.ndim != 5
        or rawk_cache.ndim != 5
        or g_cache.ndim != 5
        or beta_cache.ndim != 4
    ):
        raise ValueError("KDA ReplaySSM expects all-layer input-window views")
    num_layers, _, value_heads, value_dim, key_dim = checkpoint_state.shape
    key_heads = rawk_cache.shape[2]
    window_len = rawv_cache.shape[-2]
    if checkpoint_state.dtype != torch.float32:
        raise ValueError("KDA ReplaySSM requires a float32 checkpoint")
    if g_cache.dtype != torch.float32 or beta_cache.dtype != torch.float32:
        raise ValueError("KDA ReplaySSM g/beta windows must use float32")
    if accept_lens.shape != ssm_state_indices.shape:
        raise ValueError("KDA ReplaySSM accept lengths and state indices must align")
    if (mamba_track_indices is None) != (mamba_steps_to_track is None):
        raise ValueError("KDA ReplaySSM tracking indices and steps must be paired")
    if mamba_track_indices is not None and (
        mamba_track_indices.shape != ssm_state_indices.shape
        or mamba_steps_to_track.shape != ssm_state_indices.shape
    ):
        raise ValueError("KDA ReplaySSM tracking tensors must align with requests")
    if value_heads % key_heads:
        raise ValueError("KDA ReplaySSM value heads must be divisible by key heads")
    if (
        rawv_cache.shape[:2] != checkpoint_state.shape[:2]
        or rawv_cache.shape[2:] != (value_heads, window_len, value_dim)
        or rawk_cache.shape[:2] != checkpoint_state.shape[:2]
        or rawk_cache.shape[3:] != (window_len, key_dim)
        or g_cache.shape
        != (*checkpoint_state.shape[:2], value_heads, window_len, key_dim)
        or beta_cache.shape != (*checkpoint_state.shape[:2], value_heads, window_len)
    ):
        raise ValueError("KDA ReplaySSM received incompatible input-window shapes")

    has_track = mamba_track_indices is not None and mamba_steps_to_track is not None
    if has_track:
        track_indices = mamba_track_indices
        track_steps = mamba_steps_to_track
        stride_track = track_indices.stride(0)
        stride_track_step = track_steps.stride(0)
    else:
        track_indices = ssm_state_indices
        track_steps = accept_lens
        stride_track = 0
        stride_track_step = 0

    BK = triton.next_power_of_2(key_dim)
    BV = min(triton.next_power_of_2(value_dim), 32)
    batch = ssm_state_indices.shape[0]
    grid = (triton.cdiv(value_dim, BV), batch, value_heads * num_layers)
    kda_replayssm_exact_fold_kernel[grid](
        checkpoint_state,
        rawv_cache,
        rawk_cache,
        g_cache,
        beta_cache,
        ssm_state_indices,
        accept_lens,
        track_indices,
        track_steps,
        checkpoint_state.stride(1),
        rawv_cache.stride(1),
        rawk_cache.stride(1),
        g_cache.stride(1),
        beta_cache.stride(1),
        checkpoint_state.stride(0),
        rawv_cache.stride(0),
        rawk_cache.stride(0),
        g_cache.stride(0),
        beta_cache.stride(0),
        ssm_state_indices.stride(0),
        accept_lens.stride(0),
        stride_track,
        stride_track_step,
        H=key_heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BK=BK,
        BV=BV,
        WINDOW_LEN=window_len,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        NULL_BLOCK_ID=null_block_id,
        HAS_TRACK=has_track,
        num_warps=1,
        num_stages=3,
    )
