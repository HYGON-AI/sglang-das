# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic Triton decode-attention checks against a Torch reference."""

import unittest

import torch

from sglang.kernels.ops.attention.decode_attention import (
    decode_attention_fwd,
)
from sglang.test.ci.ci_register import register_hcu_ci

register_hcu_ci(
    est_time=120,
    suite="nightly-hcu-core-functional",
    nightly=True,
)


def _torch_decode_reference(q, k, v, kv_indptr, kv_indices, scale):
    batch, query_heads, dim = q.shape
    kv_heads = k.shape[1]
    group_size = query_heads // kv_heads
    output = torch.empty(batch, query_heads, dim, device=q.device, dtype=torch.float32)
    for batch_index in range(batch):
        start = int(kv_indptr[batch_index].item())
        end = int(kv_indptr[batch_index + 1].item())
        indices = kv_indices[start:end]
        keys = k.index_select(0, indices).repeat_interleave(group_size, dim=1).float()
        values = v.index_select(0, indices).repeat_interleave(group_size, dim=1).float()
        logits = torch.einsum("hd,lhd->hl", q[batch_index].float(), keys) * scale
        probabilities = torch.softmax(logits, dim=-1)
        output[batch_index] = torch.einsum("hl,lhd->hd", probabilities, values)
    return output


class TestBW1100AttentionReferenceHCU(unittest.TestCase):
    def test_decode_attention_matches_torch(self):
        torch.manual_seed(2026)
        for batch, query_heads, kv_heads, dim in (
            (2, 4, 4, 64),
            (3, 8, 2, 128),
        ):
            with self.subTest(
                batch=batch,
                query_heads=query_heads,
                kv_heads=kv_heads,
                dim=dim,
            ):
                seq_len = 17
                total_tokens = batch * seq_len
                scale = dim**-0.5
                q = torch.randn(
                    batch, query_heads, dim, device="cuda", dtype=torch.bfloat16
                )
                k = torch.randn(
                    total_tokens, kv_heads, dim, device="cuda", dtype=torch.bfloat16
                )
                v = torch.randn_like(k)
                output = torch.zeros_like(q)
                kv_indptr = torch.arange(
                    0,
                    total_tokens + 1,
                    seq_len,
                    device="cuda",
                    dtype=torch.int32,
                )
                kv_indices = torch.arange(total_tokens, device="cuda", dtype=torch.int32)
                max_kv_splits = 8
                num_kv_splits = torch.full(
                    (batch,), 4, device="cuda", dtype=torch.int32
                )
                attn_logits = torch.empty(
                    batch,
                    query_heads,
                    max_kv_splits,
                    dim,
                    device="cuda",
                    dtype=torch.float32,
                )
                attn_lse = torch.empty(
                    batch,
                    query_heads,
                    max_kv_splits,
                    device="cuda",
                    dtype=torch.float32,
                )

                decode_attention_fwd(
                    q,
                    k,
                    v,
                    output,
                    kv_indptr,
                    kv_indices,
                    attn_logits,
                    attn_lse,
                    num_kv_splits,
                    max_kv_splits,
                    scale,
                    1.0,
                    1.0,
                )
                reference = _torch_decode_reference(
                    q, k, v, kv_indptr, kv_indices, scale
                )
                torch.testing.assert_close(
                    output.float(), reference, rtol=1e-2, atol=1e-2
                )


if __name__ == "__main__":
    unittest.main()
