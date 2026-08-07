# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic HCU MoE routing checks without random top-k ties."""

import unittest

import torch
from sgl_kernel import topk_sigmoid, topk_softmax

from sglang.test.ci.ci_register import register_hcu_ci

register_hcu_ci(
    est_time=90,
    suite="nightly-hcu-core-functional",
    nightly=True,
)


def _routing_scores(tokens: int, experts: int, dtype: torch.dtype) -> torch.Tensor:
    expert_scores = torch.arange(experts, device="cuda", dtype=torch.float32) * 0.25
    token_offsets = torch.arange(tokens, device="cuda", dtype=torch.float32)[:, None]
    return (expert_scores[None, :] + token_offsets * 0.03125).to(dtype)


class TestBW1100MoeTopKReferenceHCU(unittest.TestCase):
    def _check(self, op, activation, dtype):
        scores = _routing_scores(32, 64, dtype)
        weights = torch.empty(32, 4, device="cuda", dtype=torch.float32)
        indices = torch.empty(32, 4, device="cuda", dtype=torch.int32)
        op(weights, indices, scores, renormalize=True)

        activated = activation(scores.float())
        expected_weights, expected_indices = torch.topk(activated, 4, dim=-1)
        expected_weights /= expected_weights.sum(dim=-1, keepdim=True)

        sorted_indices, order = torch.sort(indices, dim=-1)
        sorted_weights = torch.gather(weights, 1, order)
        expected_sorted_indices, expected_order = torch.sort(
            expected_indices.int(), dim=-1
        )
        expected_sorted_weights = torch.gather(
            expected_weights, 1, expected_order
        )

        torch.testing.assert_close(
            sorted_indices, expected_sorted_indices, rtol=0, atol=0
        )
        torch.testing.assert_close(
            sorted_weights, expected_sorted_weights, rtol=1e-3, atol=1e-3
        )

    def test_sigmoid_topk_matches_torch(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self._check(topk_sigmoid, torch.sigmoid, dtype)

    def test_softmax_topk_matches_torch(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self._check(
                    topk_softmax,
                    lambda value: torch.softmax(value, dim=-1),
                    dtype,
                )


if __name__ == "__main__":
    unittest.main()
