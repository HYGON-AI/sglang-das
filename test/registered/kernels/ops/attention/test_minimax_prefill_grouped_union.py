"""MiniMax prefill grouped-union numerical equivalence.

The grouped main-attention path may visit blocks selected by another query in
the same group.  Those rows are masked by ``per_query_topk_idx`` and must leave
the online-softmax state bit-for-bit unchanged.
"""

import pytest
import torch
import triton

from sglang.kernels.ops.attention.minimax_sparse.prefill.flash_with_topk_idx import (
    _prune_prefill_score_configs,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill.topk_sparse import (
    build_query_group_topk_union,
    flash_prefill_with_gqa_share_sparse,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def test_prefill_score_override_is_limited_to_long_context_exact_q1(monkeypatch):
    configs = [
        triton.Config(
            {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 128},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 128, "BLOCK_SIZE_K": 128},
            num_warps=8,
            num_stages=3,
        ),
    ]
    monkeypatch.setenv("SGLANG_MINIMAX_PREFILL_SCORE_CONFIG", "128,128,8,3")
    monkeypatch.setenv("SGLANG_MINIMAX_PREFILL_SCORE_CONFIG_MIN_K", "131072")

    short_or_grouped = _prune_prefill_score_configs(
        configs,
        {"Q_SAMPLE_INTERVAL": 16, "autotune_k_bucket": 262144},
    )
    short_k = _prune_prefill_score_configs(
        configs,
        {"Q_SAMPLE_INTERVAL": 1, "autotune_k_bucket": 1024},
    )
    winner_shape = _prune_prefill_score_configs(
        configs,
        {"Q_SAMPLE_INTERVAL": 1, "autotune_k_bucket": 262144},
    )

    assert short_or_grouped == configs
    assert short_k == configs
    assert winner_shape == [configs[1]]


@pytest.mark.parametrize("fused_union", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_grouped_union_preserves_q1_output_bit_exact(monkeypatch, fused_union):
    monkeypatch.setenv(
        "SGLANG_MINIMAX_PREFILL_FUSED_TOPK_UNION",
        "1" if fused_union else "0",
    )
    torch.manual_seed(7)
    device = torch.device("cuda")

    num_queries = 16
    prefix_len = 2048
    sparse_block_size = 64
    num_kv_heads = 1
    num_query_heads = 16
    qk_head_dim = 192
    v_head_dim = 128
    topk = 16
    seq_len = prefix_len + num_queries

    q = torch.randn(
        num_queries,
        num_query_heads,
        qk_head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        seq_len,
        num_kv_heads,
        qk_head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    v = torch.randn(
        seq_len,
        num_kv_heads,
        v_head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    req_to_token = torch.zeros((1, 4096), device=device, dtype=torch.int32)
    req_to_token[0, :seq_len] = torch.arange(seq_len, device=device, dtype=torch.int32)
    slot_ids = torch.tensor([0], device=device, dtype=torch.int32)
    cu_seqlens = torch.tensor([0, num_queries], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)
    prefix_lens = torch.tensor([prefix_len], device=device, dtype=torch.int32)

    # Give adjacent queries overlapping but non-identical Top-K sets.  The
    # grouped union consequently contains inactive blocks for every query.
    rows = []
    for query_id in range(num_queries):
        selected = set(range(12))
        selected.update(
            {
                12 + query_id % 8,
                20 + (query_id * 3) % 10,
                seq_len // sparse_block_size,
            }
        )
        for candidate in range(12, seq_len // sparse_block_size):
            if len(selected) >= topk:
                break
            selected.add(candidate)
        row = sorted(selected)[: topk - 1] + [seq_len // sparse_block_size]
        # AITER Top-K emits score-ranked ids.  The fused producer/union path
        # deliberately skips the old standalone id-sort kernel, so exercise
        # that production contract instead of accidentally validating only an
        # already-sorted input.
        rows.append(row[::2] + row[1::2])
    unsorted_topk = torch.tensor(rows, device=device, dtype=torch.int32).unsqueeze(0)
    sorted_topk = torch.sort(unsorted_topk, dim=-1).values
    # Legacy production performs a separate id-sort before union.  The fused
    # path must accept AITER's unsorted result and establish the same in-place
    # contract itself.
    per_query_topk = unsorted_topk.clone() if fused_union else sorted_topk.clone()

    q1_cu_seqblocks = torch.tensor([0, num_queries], device=device, dtype=torch.int32)
    reference = flash_prefill_with_gqa_share_sparse(
        q=q,
        k_cache=k,
        v_cache=v,
        sink=None,
        req_to_token=req_to_token,
        slot_ids=slot_ids,
        topk_idx=sorted_topk,
        block_size_q=1,
        block_size_k=sparse_block_size,
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        max_seqlen_q=num_queries,
        cu_seqblocks_q=q1_cu_seqblocks,
        max_seqblock_q=num_queries,
    )

    grouped_q = 8
    grouped_cu_seqblocks = torch.tensor([0, 2], device=device, dtype=torch.int32)
    grouped_topk, query_group_mask = build_query_group_topk_union(
        topk_idx=per_query_topk,
        cu_seqlens_q=cu_seqlens,
        cu_seqblocks_q=grouped_cu_seqblocks,
        block_size_q=grouped_q,
        all_groups=2,
        max_num_blocks=(seq_len + sparse_block_size - 1) // sparse_block_size,
        max_union=grouped_q * topk,
        return_query_mask=True,
    )
    assert torch.equal(per_query_topk, sorted_topk)
    grouped = flash_prefill_with_gqa_share_sparse(
        q=q,
        k_cache=k,
        v_cache=v,
        sink=None,
        req_to_token=req_to_token,
        slot_ids=slot_ids,
        topk_idx=grouped_topk,
        per_query_topk_idx=per_query_topk,
        query_group_mask=query_group_mask,
        block_size_q=grouped_q,
        block_size_k=sparse_block_size,
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        max_seqlen_q=num_queries,
        cu_seqblocks_q=grouped_cu_seqblocks,
        max_seqblock_q=2,
    )

    torch.cuda.synchronize()
    assert torch.equal(grouped, reference)
