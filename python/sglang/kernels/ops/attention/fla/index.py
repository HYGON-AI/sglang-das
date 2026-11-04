# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Modified by Hygon Information Technology Co., Ltd., 2026.

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton

from sglang.kernels.ops.attention.fla.utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    # Eager GDN metadata owns a fresh cumulative-length tensor per forward.
    # Its host lengths let every chunk-size variant avoid a shape readback.
    lengths_cpu = getattr(cu_seqlens, "_fla_lengths_cpu", None)
    num_chunks = (
        [triton.cdiv(length, chunk_size) for length in lengths_cpu]
        if lengths_cpu is not None
        else triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
    )
    indices = torch.cat([torch.arange(n) for n in num_chunks])
    return (
        torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)
        .pin_memory()
        .to(cu_seqlens, non_blocking=True)
    )


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    lengths_cpu = getattr(cu_seqlens, "_fla_lengths_cpu", None)
    if lengths_cpu is not None:
        offsets = [0]
        for length in lengths_cpu:
            offsets.append(offsets[-1] + triton.cdiv(length, chunk_size))
        return torch.tensor(
            offsets, dtype=torch.int64, pin_memory=True
        ).to(device=cu_seqlens.device, non_blocking=True)
    return torch.cat(
        [cu_seqlens.new_zeros(1), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
