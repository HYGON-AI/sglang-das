# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0.

import torch

from sglang.srt.models.hunyuan_v3 import (
    _apply_hy3_qk_norm,
    _merge_hy3_sp_attention_output,
    _pack_hy3_sp_qkv,
    _reshape_hy3_sp_attention_output,
)


class _PerHeadRMSNorm(torch.nn.Module):
    def __init__(self, head_dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(1, head_dim + 1).float())
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x.float() * torch.rsqrt(variance + self.eps) * self.weight).to(
            x.dtype
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


def test_apply_hy3_qk_norm_is_per_head_and_preserves_width():
    head_dim = 2
    q = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    k = torch.tensor([[5.0, 6.0]])
    q_norm = _PerHeadRMSNorm(head_dim)
    k_norm = _PerHeadRMSNorm(head_dim)

    q_actual, k_actual = _apply_hy3_qk_norm(
        q, k, q_norm, k_norm, head_dim
    )
    q_expected = q_norm(q.view(-1, head_dim)).view_as(q)
    k_expected = k_norm(k.view(-1, head_dim)).view_as(k)

    assert q_actual.shape == q.shape
    assert k_actual.shape == k.shape
    torch.testing.assert_close(q_actual, q_expected)
    torch.testing.assert_close(k_actual, k_expected)


def test_reshape_hy3_sp_attention_output_uses_actual_token_count():
    attn_output = torch.arange(8 * 3 * 4, dtype=torch.float32).view(24, 4)
    reshaped = _reshape_hy3_sp_attention_output(
        attn_output, tp_size=8, q_size=4
    )

    assert reshaped.shape == (8, 3, 4)
    torch.testing.assert_close(reshaped.flatten(0, 1), attn_output)


def test_reshape_hy3_sp_attention_output_rejects_invalid_token_count():
    attn_output = torch.empty(23, 4)
    try:
        _reshape_hy3_sp_attention_output(attn_output, tp_size=8, q_size=4)
    except ValueError as exc:
        assert "token count must be divisible" in str(exc)
    else:
        raise AssertionError("Expected an invalid SP token count to be rejected")
