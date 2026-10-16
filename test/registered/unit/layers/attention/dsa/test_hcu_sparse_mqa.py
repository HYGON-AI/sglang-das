"""CPU routing tests for paired LightOp sparse Page-MQA/mask TopK."""

import unittest

import torch

from sglang.srt.layers.attention.dsa.hcu_sparse_mqa import (
    select_lightop_sparse_mqa_route,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


@unittest.skipUnless(
    hasattr(torch, "float8_e4m3fn"), "PyTorch build does not expose FP8 E4M3FN"
)
class TestLightOpSparseMQARoute(CustomTestCase):
    def _inputs(self, rows: int):
        q = torch.empty((rows, 1, 32, 128), dtype=torch.float8_e4m3fn)
        fused_kv_cache = torch.empty((2, 64, 1, 132), dtype=torch.uint8)
        weights = torch.empty((rows, 32), dtype=torch.float32)
        context_lens = torch.ones(rows, dtype=torch.int32)
        block_table = torch.zeros((rows, 2), dtype=torch.int32)
        return q, fused_kv_cache, weights, context_lens, block_table

    def _select(self, rows: int, **overrides):
        q, fused_kv_cache, weights, context_lens, block_table = self._inputs(rows)
        kwargs = dict(
            enabled=True,
            api_available=True,
            is_hcu=True,
            arch_name="gfx938",
            num_cus=64,
            page_size=64,
            topk=2048,
            fuse_topk=True,
            force_unfused_topk=False,
            topk_transform_method_name="PAGED",
            q=q,
            fused_kv_cache=fused_kv_cache,
            weights=weights,
            context_lens=context_lens,
            block_table=block_table,
            max_context_len=128,
            batch_size=rows,
            is_target_verify=False,
            is_draft_extend_v2=False,
            mtp_group_size=None,
            grouping_lens_cpu=[1] * rows,
        )
        kwargs.update(overrides)
        return select_lightop_sparse_mqa_route(**kwargs)

    def test_decode_selects_independent_row_sparse_mqa(self):
        route = self._select(8)

        self.assertIsNotNone(route)
        self.assertEqual(route.group_size, 1)

    def test_target_verify_selects_each_supported_mtp_group(self):
        for group_size in (3, 4, 5):
            for graph_expanded in (False, True):
                with self.subTest(group_size=group_size, graph_expanded=graph_expanded):
                    batch_size = 8
                    route = self._select(
                        batch_size * group_size,
                        batch_size=batch_size,
                        is_target_verify=True,
                        mtp_group_size=group_size,
                        grouping_lens_cpu=(
                            [1] * (batch_size * group_size)
                            if graph_expanded
                            else [group_size] * batch_size
                        ),
                    )

                    self.assertIsNotNone(route)
                    self.assertEqual(route.group_size, group_size)

    def test_grouped_route_requires_per_request_row_contract(self):
        route = self._select(
            40,
            batch_size=8,
            is_draft_extend_v2=True,
            mtp_group_size=5,
            grouping_lens_cpu=[1] * 39 + [2],
        )

        self.assertIsNone(route)

    def test_b40_without_mtp_contract_does_not_guess_grouping(self):
        self.assertIsNone(self._select(40, batch_size=8))

    def test_consumer_or_hardware_miss_falls_back_before_sparse_mqa(self):
        for overrides in (
            {"api_available": False},
            {"fuse_topk": False},
            {"force_unfused_topk": True},
            {"topk_transform_method_name": "RAGGED"},
            {"topk": 1024},
            {"arch_name": "gfx936", "num_cus": 80},
        ):
            with self.subTest(overrides=overrides):
                self.assertIsNone(self._select(8, **overrides))


if __name__ == "__main__":
    unittest.main()
