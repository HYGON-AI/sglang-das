# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0.

import torch

from sglang.srt.models.hunyuan_v3 import (
    _merge_hy3_sp_attention_output,
    _pack_hy3_sp_qkv,
)


def test_pack_hy3_sp_qkv_gqa_tp8():
    tokens_per_rank = 2
    tp_size = 8
    head_dim = 2
    total_num_q_heads = 64
    total_num_kv_heads = 8
    q_size = total_num_q_heads // tp_size * head_dim
    kv_size = total_num_kv_heads // tp_size * head_dim

    q = torch.arange(
        tokens_per_rank * total_num_q_heads * head_dim, dtype=torch.float32
    ).view(tokens_per_rank, -1)
    k = 1000 + torch.arange(
        tokens_per_rank * total_num_kv_heads * head_dim, dtype=torch.float32
    ).view(tokens_per_rank, -1)
    v = 2000 + torch.arange(
        tokens_per_rank * total_num_kv_heads * head_dim, dtype=torch.float32
    ).view(tokens_per_rank, -1)

    packed = _pack_hy3_sp_qkv(
        q,
        k,
        v,
        tp_size,
        q_size,
        kv_size,
        total_num_kv_heads,
        head_dim,
    )

    assert packed.shape == (tp_size, tokens_per_rank, q_size + 2 * kv_size)
    q_by_rank = q.view(tokens_per_rank, tp_size, q_size)
    k_by_rank = k.view(tokens_per_rank, tp_size, kv_size)
    v_by_rank = v.view(tokens_per_rank, tp_size, kv_size)
    for rank in range(tp_size):
        expected = torch.cat(
            [q_by_rank[:, rank], k_by_rank[:, rank], v_by_rank[:, rank]], dim=-1
        )
        torch.testing.assert_close(packed[rank], expected)


def test_pack_hy3_sp_qkv_replicates_kv_heads():
    tokens_per_rank = 1
    tp_size = 8
    head_dim = 2
    total_num_q_heads = 16
    total_num_kv_heads = 2
    q_size = total_num_q_heads // tp_size * head_dim
    kv_size = head_dim

    q = torch.arange(total_num_q_heads * head_dim, dtype=torch.float32).view(1, -1)
    k = 100 + torch.arange(total_num_kv_heads * head_dim, dtype=torch.float32).view(
        1, -1
    )
    v = 200 + torch.arange(total_num_kv_heads * head_dim, dtype=torch.float32).view(
        1, -1
    )

    packed = _pack_hy3_sp_qkv(
        q,
        k,
        v,
        tp_size,
        q_size,
        kv_size,
        total_num_kv_heads,
        head_dim,
    )

    k_by_head = k.view(tokens_per_rank, total_num_kv_heads, head_dim)
    v_by_head = v.view(tokens_per_rank, total_num_kv_heads, head_dim)
    replicas_per_kv_head = tp_size // total_num_kv_heads
    for rank in range(tp_size):
        kv_head = rank // replicas_per_kv_head
        torch.testing.assert_close(
            packed[rank, :, q_size : q_size + kv_size], k_by_head[:, kv_head]
        )
        torch.testing.assert_close(
            packed[rank, :, q_size + kv_size :], v_by_head[:, kv_head]
        )


def test_merge_hy3_sp_attention_output_restores_full_q_width():
    output = torch.arange(8 * 3 * 4, dtype=torch.float32).view(8, 3, 4)
    merged = _merge_hy3_sp_attention_output(output)

    assert merged.shape == (3, 32)
    torch.testing.assert_close(merged, output.permute(1, 0, 2).reshape(3, 32))
