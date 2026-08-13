"""Unit coverage for the HCU persistent paged-MQA safety gates."""

import unittest
from unittest.mock import MagicMock, patch, sentinel

import torch

from sglang.srt.layers.attention.dsa import hcu_persistent_mqa
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _mock_tensor(*, dtype, shape, stride, device=torch.device("cuda:0")):
    tensor = MagicMock()
    tensor.dtype = dtype
    tensor.device = device
    tensor.shape = shape
    tensor.dim.return_value = len(shape)
    tensor.stride.return_value = stride
    tensor.stride.side_effect = lambda dim=None: stride if dim is None else stride[dim]
    tensor.is_contiguous.return_value = True
    return tensor


class TestHCUPersistentMQAGates(CustomTestCase):
    def setUp(self):
        self.saved_operation = hcu_persistent_mqa._selected_operation
        self.saved_failure = hcu_persistent_mqa._preload_failure
        self.saved_hardware = hcu_persistent_mqa._preloaded_hardware
        hcu_persistent_mqa._selected_operation = None
        hcu_persistent_mqa._preload_failure = None
        hcu_persistent_mqa._preloaded_hardware = None
        hcu_persistent_mqa._logged_hits.clear()
        hcu_persistent_mqa._logged_misses.clear()

    def tearDown(self):
        hcu_persistent_mqa._selected_operation = self.saved_operation
        hcu_persistent_mqa._preload_failure = self.saved_failure
        hcu_persistent_mqa._preloaded_hardware = self.saved_hardware
        hcu_persistent_mqa._logged_hits.clear()
        hcu_persistent_mqa._logged_misses.clear()

    def test_auto_ctas_are_graph_stable_and_bounded(self):
        self.assertEqual(
            hcu_persistent_mqa.resolve_persistent_ctas(
                0, batch_size=8, max_context_len=65536
            ),
            256,
        )
        self.assertEqual(
            hcu_persistent_mqa.resolve_persistent_ctas(
                96, batch_size=8, max_context_len=65536
            ),
            96,
        )
        with self.assertRaisesRegex(ValueError, r"\[1, 4096\]"):
            hcu_persistent_mqa.resolve_persistent_ctas(
                4097, batch_size=8, max_context_len=65536
            )

    def test_hardware_gate_is_exactly_gfx938_with_64_cus(self):
        self.assertIsNone(
            hcu_persistent_mqa._hardware_gate_miss_reason(
                arch_name="gfx938:sramecc+", num_cus=64
            )
        )
        self.assertIn(
            "gfx938",
            hcu_persistent_mqa._hardware_gate_miss_reason(
                arch_name="gfx936", num_cus=64
            ),
        )
        self.assertIn(
            "64 CUs",
            hcu_persistent_mqa._hardware_gate_miss_reason(
                arch_name="gfx938", num_cus=60
            ),
        )

    def test_input_gate_accepts_only_native_fp8_page64_layout(self):
        batch_size = 2
        q = _mock_tensor(
            dtype=torch.float8_e4m3fn,
            shape=(batch_size, 1, 32, 128),
            stride=(4096, 4096, 128, 1),
        )
        kv_cache = _mock_tensor(
            dtype=torch.uint8,
            shape=(4, 64, 1, 132),
            stride=(8448, 132, 132, 1),
        )
        weights = _mock_tensor(
            dtype=torch.float32,
            shape=(batch_size, 32),
            stride=(32, 1),
        )
        context_lens = _mock_tensor(
            dtype=torch.int32,
            shape=(batch_size,),
            stride=(1,),
        )
        block_table = _mock_tensor(
            dtype=torch.int32,
            shape=(batch_size, 4),
            stride=(4, 1),
        )

        kwargs = dict(
            is_hcu=True,
            page_size=64,
            q=q,
            fused_kv_cache=kv_cache,
            weights=weights,
            context_lens=context_lens,
            block_table=block_table,
            schedule_meta=None,
            max_context_len=256,
        )
        self.assertIsNone(hcu_persistent_mqa._input_gate_miss_reason(**kwargs))

        kv_cache.stride.side_effect = lambda dim=None: (
            (1, 1, 1, 1) if dim is None else 1
        )
        self.assertIn(
            "packed stride",
            hcu_persistent_mqa._input_gate_miss_reason(**kwargs),
        )

    def test_consumer_requires_the_exact_context_length_tensor(self):
        context_lens = sentinel.context_lens
        common = dict(
            fuse_topk=True,
            force_unfused_topk=False,
            topk_transform_method_name="PAGED",
            index_topk=2048,
            score_context_lens=context_lens,
        )
        self.assertIsNone(
            hcu_persistent_mqa._consumer_gate_miss_reason(
                **common,
                topk_context_lens=context_lens,
            )
        )
        self.assertIn(
            "exact tensor",
            hcu_persistent_mqa._consumer_gate_miss_reason(
                **common,
                topk_context_lens=sentinel.other_context_lens,
            ),
        )

    def test_consumer_mismatch_falls_back_without_calling_package(self):
        operation = MagicMock()
        hcu_persistent_mqa._selected_operation = operation
        hcu_persistent_mqa._preloaded_hardware = ("gfx938", 64)

        with (
            patch.object(
                hcu_persistent_mqa.envs.SGLANG_DSA_HCU_PERSISTENT_MQA_FASTPATH,
                "get",
                return_value=True,
            ),
            patch.object(
                hcu_persistent_mqa,
                "_input_gate_miss_reason",
                return_value=None,
            ),
            patch.object(
                hcu_persistent_mqa,
                "_consumer_gate_miss_reason",
                return_value="context_lens mismatch",
            ),
        ):
            result = hcu_persistent_mqa.paged_mqa_logits_length_masked(
                sentinel.q,
                sentinel.kv_cache,
                sentinel.weights,
                sentinel.context_lens,
                sentinel.block_table,
                None,
                65536,
                is_hcu=True,
                page_size=64,
                fuse_topk=True,
                force_unfused_topk=False,
                topk_transform_method_name="PAGED",
                index_topk=2048,
                topk_context_lens=sentinel.other_context_lens,
            )

        self.assertIsNone(result)
        operation.assert_not_called()

    def test_eligible_call_uses_packaged_operation(self):
        q = MagicMock()
        q.shape = (8, 1, 32, 128)
        operation = MagicMock(return_value=sentinel.logits)
        hcu_persistent_mqa._selected_operation = operation
        hcu_persistent_mqa._preloaded_hardware = ("gfx938", 64)

        with (
            patch.object(
                hcu_persistent_mqa.envs.SGLANG_DSA_HCU_PERSISTENT_MQA_FASTPATH,
                "get",
                return_value=True,
            ),
            patch.object(
                hcu_persistent_mqa.envs.SGLANG_DSA_HCU_PERSISTENT_MQA_CTAS,
                "get",
                return_value=0,
            ),
            patch.object(
                hcu_persistent_mqa,
                "_input_gate_miss_reason",
                return_value=None,
            ),
            patch.object(
                hcu_persistent_mqa,
                "_consumer_gate_miss_reason",
                return_value=None,
            ),
        ):
            result = hcu_persistent_mqa.paged_mqa_logits_length_masked(
                q,
                sentinel.kv_cache,
                sentinel.weights,
                sentinel.context_lens,
                sentinel.block_table,
                None,
                65536,
                is_hcu=True,
                page_size=64,
                fuse_topk=True,
                force_unfused_topk=False,
                topk_transform_method_name="PAGED",
                index_topk=2048,
                topk_context_lens=sentinel.context_lens,
            )

        self.assertIs(result, sentinel.logits)
        operation.assert_called_once_with(
            q,
            sentinel.kv_cache,
            sentinel.weights,
            sentinel.context_lens,
            sentinel.block_table,
            None,
            65536,
            True,
            4,
            256,
        )


if __name__ == "__main__":
    unittest.main()
