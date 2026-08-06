# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code adapted from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
"""Kimi-K3 attention-residual mix kernel (HCU single-kernel variant).

Vendored from the operator team's implementation.  Computes the same pre-norm
softmax mixture as sglang's ``_mix_fused`` (score -> softmax -> weighted sum
over ``[block_0 .. block_{num_blocks-1}, prefix]``) but in a single Triton
kernel with an online softmax.  Static configs were selected by an offline
sweep on gfx938 (BW1100); a lazy-prefix sibling shortens prefix live ranges.

Used on HCU via ``SGLANG_K3_ATTN_RES_HCU`` (see
``sglang.srt.layers.attn_residual``).
"""

import torch
import triton
import triton.language as tl

# Three static configs selected by an offline sweep on gfx938 (BW1100).
# Keep this list explicit: production calls must not pay Triton's runtime
# autotune cost.
_ATTN_RES_HCU_CONFIGS = (
    {"BLOCK_L": 4, "num_warps": 8, "num_stages": 2},
    {"BLOCK_L": 4, "num_warps": 4, "num_stages": 2},
    {"BLOCK_L": 2, "num_warps": 4, "num_stages": 2},
)


def _get_attn_res_hcu_config(num_tokens: int, num_blocks: int) -> dict[str, int]:
    if num_tokens < 320:
        return _ATTN_RES_HCU_CONFIGS[0]
    if 2 <= num_blocks <= 3:
        return _ATTN_RES_HCU_CONFIGS[1]
    return _ATTN_RES_HCU_CONFIGS[2]


def _use_attn_res_hcu_lazy_prefix(num_tokens: int, num_blocks: int) -> bool:
    return num_tokens >= 320 and num_blocks >= 4


@triton.jit
def _attn_res_hcu_kernel(
    prefix_ptr,
    blocks_ptr,
    norm_weight_ptr,
    qk_weight_ptr,
    output_ptr,
    stride_prefix_m: tl.constexpr,
    stride_block_m: tl.constexpr,
    stride_block_r: tl.constexpr,
    stride_output_m: tl.constexpr,
    num_blocks: tl.constexpr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    d_offsets = tl.max_contiguous(tl.arange(0, BLOCK_D), BLOCK_D)
    d_mask = d_offsets < hidden_size

    prefix = tl.load(
        prefix_ptr + row_idx * stride_prefix_m + d_offsets,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    input_qk_weight = tl.load(norm_weight_ptr + d_offsets, mask=d_mask, other=0.0).to(
        tl.float32
    ) * tl.load(qk_weight_ptr + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

    max_logit = tl.full((), -float("inf"), tl.float32)
    denominator = tl.zeros((), tl.float32)
    mixed = tl.zeros((BLOCK_D,), tl.float32)
    num_sources = num_blocks + 1

    for source_tile in range(tl.cdiv(num_sources, BLOCK_L)):
        source_offsets = source_tile * BLOCK_L + tl.arange(0, BLOCK_L)
        source_mask = source_offsets < num_sources
        is_prefix = source_offsets == num_blocks
        block_ptrs = (
            blocks_ptr
            + row_idx * stride_block_m
            + source_offsets[:, None] * stride_block_r
            + d_offsets[None, :]
        )
        block_values = tl.load(
            block_ptrs,
            mask=(source_mask[:, None] & ~is_prefix[:, None] & d_mask[None, :]),
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        values = tl.where(is_prefix[:, None], prefix[None, :], block_values)
        reciprocal_std = tl.rsqrt(
            tl.sum(values * values, axis=1) * (1.0 / hidden_size) + eps
        )
        logits = tl.sum(values * input_qk_weight[None, :], axis=1) * reciprocal_std
        scores = tl.where(source_mask, logits, -float("inf"))

        new_max_logit = tl.maximum(max_logit, tl.max(scores, axis=0))
        old_scale = tl.exp(max_logit - new_max_logit)
        block_scales = tl.exp(scores - new_max_logit)
        denominator = denominator * old_scale + tl.sum(block_scales, axis=0)
        mixed = mixed * old_scale + tl.sum(block_scales[:, None] * values, axis=0)
        max_logit = new_max_logit

    output = mixed / denominator
    tl.store(
        output_ptr + row_idx * stride_output_m + d_offsets,
        output,
        mask=d_mask,
    )


@triton.jit
def _attn_res_hcu_kernel_lazy_prefix(
    prefix_ptr,
    blocks_ptr,
    norm_weight_ptr,
    qk_weight_ptr,
    output_ptr,
    stride_prefix_m: tl.constexpr,
    stride_block_m: tl.constexpr,
    stride_block_r: tl.constexpr,
    stride_output_m: tl.constexpr,
    num_blocks: tl.constexpr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Keep the upstream kernel intact while shortening prefix live ranges."""
    row_idx = tl.program_id(0).to(tl.int64)
    d_offsets = tl.max_contiguous(tl.arange(0, BLOCK_D), BLOCK_D)
    d_mask = d_offsets < hidden_size
    input_qk_weight = tl.load(
        norm_weight_ptr + d_offsets, mask=d_mask, other=0.0
    ).to(tl.float32) * tl.load(
        qk_weight_ptr + d_offsets, mask=d_mask, other=0.0
    ).to(tl.float32)

    max_logit = tl.full((), -float("inf"), tl.float32)
    denominator = tl.zeros((), tl.float32)
    mixed = tl.zeros((BLOCK_D,), tl.float32)
    num_sources: tl.constexpr = num_blocks + 1
    num_tiles: tl.constexpr = tl.cdiv(num_sources, BLOCK_L)

    for source_tile in range(num_tiles):
        source_offsets = source_tile * BLOCK_L + tl.arange(0, BLOCK_L)
        source_mask = source_offsets < num_sources
        block_ptrs = (
            blocks_ptr
            + row_idx * stride_block_m
            + source_offsets[:, None] * stride_block_r
            + d_offsets[None, :]
        )
        if source_tile + 1 == num_tiles:
            is_prefix = source_offsets == num_blocks
            block_values = tl.load(
                block_ptrs,
                mask=(source_mask[:, None] & ~is_prefix[:, None] & d_mask[None, :]),
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            prefix = tl.load(
                prefix_ptr + row_idx * stride_prefix_m + d_offsets,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            values = tl.where(is_prefix[:, None], prefix[None, :], block_values)
        else:
            values = tl.load(
                block_ptrs,
                mask=(source_mask[:, None] & d_mask[None, :]),
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

        reciprocal_std = tl.rsqrt(
            tl.sum(values * values, axis=1) * (1.0 / hidden_size) + eps
        )
        logits = tl.sum(values * input_qk_weight[None, :], axis=1) * reciprocal_std
        scores = tl.where(source_mask, logits, -float("inf"))
        new_max_logit = tl.maximum(max_logit, tl.max(scores, axis=0))
        old_scale = tl.exp(max_logit - new_max_logit)
        block_scales = tl.exp(scores - new_max_logit)
        denominator = denominator * old_scale + tl.sum(block_scales, axis=0)
        mixed = mixed * old_scale + tl.sum(block_scales[:, None] * values, axis=0)
        max_logit = new_max_logit

    tl.store(
        output_ptr + row_idx * stride_output_m + d_offsets,
        mixed / denominator,
        mask=d_mask,
    )


def attn_res_hcu(
    prefix: torch.Tensor,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    num_blocks: int,
    eps: float,
) -> torch.Tensor:
    """HCU single-kernel mix: score -> online softmax -> weighted sum.

    Equivalent to sglang's ``_mix_fused`` / ``aggregate_stream`` (pre-norm
    mixture over ``[blocks[.., :num_blocks], prefix]``); one kernel instead of
    two, with an offline-selected config and lazy-prefix dispatch.
    """
    num_tokens, hidden_size = prefix.shape
    assert 0 < num_blocks <= blocks.shape[1]
    assert blocks.shape[0] == num_tokens
    assert norm_weight.numel() == hidden_size
    assert qk_weight.numel() == hidden_size
    assert prefix.stride(-1) == 1
    assert blocks.stride(-1) == 1
    assert norm_weight.stride(-1) == 1
    assert qk_weight.stride(-1) == 1

    output = prefix.new_empty(prefix.shape)
    if num_tokens == 0:
        return output

    config = _get_attn_res_hcu_config(num_tokens, num_blocks)
    kernel = (
        _attn_res_hcu_kernel_lazy_prefix
        if _use_attn_res_hcu_lazy_prefix(num_tokens, num_blocks)
        else _attn_res_hcu_kernel
    )
    kernel[(num_tokens,)](
        prefix,
        blocks,
        norm_weight,
        qk_weight,
        output,
        prefix.stride(0),
        blocks.stride(0),
        blocks.stride(1),
        output.stride(0),
        num_blocks,
        hidden_size,
        eps,
        BLOCK_D=triton.next_power_of_2(hidden_size),
        **config,
    )
    return output
