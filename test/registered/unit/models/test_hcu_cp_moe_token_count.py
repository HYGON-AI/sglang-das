"""HCU legacy CP metadata must not be consumed as generic strategy metadata."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.layers.cp.interleave import InterleaveCPStrategy
from sglang.srt.layers.utils.hcu_cp_utils import ContextParallelMetadata
from sglang.srt.model_executor import forward_batch_info as fb
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, stage="base-a", runner_config="cpu")


@pytest.mark.parametrize("real,padded", [(21, 24), (8, 8), (0, 8)])
def test_hcu_interleave_dispatch_masks_padding(real, padded):
    for rank in range(8):
        batch = SimpleNamespace(
            attn_cp_metadata=ContextParallelMetadata(),
            forward_mode=fb.ForwardMode.EXTEND,
            input_ids=torch.arange(padded),
            global_num_token_non_padded_cpu=real,
            extend_seq_lens_cpu=[real],
            num_token_non_padded=torch.tensor(real, dtype=torch.int32),
        )
        parallel = SimpleNamespace(attn_cp_size=8, attn_cp_rank=rank)
        with (
            patch.object(fb, "_is_hcu", True),
            patch.object(fb, "get_parallel", return_value=parallel),
            patch("sglang.srt.layers.cp.utils.enable_cp_v2", return_value=False),
            patch("sglang.srt.layers.attention.dsa.utils.dsa_use_prefill_cp", return_value=True),
            patch("sglang.srt.layers.attention.dsa.utils.is_dsa_prefill_cp_round_robin_split", return_value=True),
            patch("sglang.srt.layers.moe.utils.is_moe_input_scattered_across_dp_ranks", return_value=True),
            patch("sglang.srt.layers.cp.base.get_cp_strategy", return_value=InterleaveCPStrategy(8)),
        ):
            count = fb.ForwardBatch.moe_num_token_non_padded(batch)
            assert count.item() == len(range(rank, real, 8))
            assert count.dtype == torch.int32
            assert fb.ForwardBatch.moe_num_token_non_padded(batch) is count


def test_generic_strategy_still_owns_its_count():
    result = torch.tensor(3, dtype=torch.int32)
    strategy = SimpleNamespace(moe_num_token_non_padded=Mock(return_value=result))
    batch = SimpleNamespace(attn_cp_metadata=object(), forward_mode=fb.ForwardMode.EXTEND)
    with (
        patch.object(fb, "_is_hcu", False),
        patch("sglang.srt.layers.cp.utils.enable_cp_v2", return_value=True),
        patch("sglang.srt.layers.moe.utils.is_moe_input_scattered_across_dp_ranks", return_value=True),
        patch("sglang.srt.layers.cp.base.get_cp_strategy", return_value=strategy),
    ):
        assert fb.ForwardBatch.moe_num_token_non_padded(batch) is result
    strategy.moe_num_token_non_padded.assert_called_once_with(batch)


class TestHcuMoeCountDeviceFill(CustomTestCase):
    def _batch(self, real, *, extend_lens=None):
        return SimpleNamespace(
            attn_cp_metadata=ContextParallelMetadata(),
            forward_mode=fb.ForwardMode.EXTEND,
            input_ids=torch.empty(0, dtype=torch.int64),
            global_num_token_non_padded_cpu=real,
            extend_seq_lens_cpu=extend_lens,
            num_token_non_padded=torch.full((), 123, dtype=torch.int32),
        )

    def _count(self, batch, rank, *, round_robin=True):
        with (
            patch.object(fb, "_is_hcu", True),
            patch.object(
                fb, "get_parallel",
                return_value=SimpleNamespace(attn_cp_size=8, attn_cp_rank=rank),
            ),
            patch("sglang.srt.layers.cp.utils.enable_cp_v2", return_value=False),
            patch("sglang.srt.layers.attention.dsa.utils.dsa_use_prefill_cp", return_value=True),
            patch("sglang.srt.layers.attention.dsa.utils.is_dsa_prefill_cp_round_robin_split", return_value=round_robin),
            patch("sglang.srt.layers.moe.utils.is_moe_input_scattered_across_dp_ranks", return_value=True),
            patch.object(
                fb.torch, "tensor",
                side_effect=AssertionError("MoE count must not upload a host scalar"),
            ),
        ):
            return fb.ForwardBatch.moe_num_token_non_padded(batch)

    def test_all_rank_counts_without_host_tensor_factory(self):
        for real in (0, 1, 5, 8, 21, 32768, 1048576):
            for rank in range(8):
                with self.subTest(real=real, rank=rank):
                    batch = self._batch(real)
                    count = self._count(batch, rank)
                    self.assertEqual(count.item(), len(range(rank, real, 8)))
                    self.assertEqual(count.shape, torch.Size([]))
                    self.assertEqual(count.dtype, torch.int32)
                    self.assertEqual(count.device, batch.input_ids.device)
                    self.assertIs(self._count(batch, rank), count)

    def test_full_and_tail_metadata_keep_separate_cached_counts(self):
        full, tail = self._batch(32768), self._batch(128)
        for batch, expected in ((full, 4096), (tail, 16)):
            count = self._count(batch, 7)
            self.assertEqual(count.item(), expected)
            self.assertIs(self._count(batch, 7), count)
        self.assertIsNot(full.attn_cp_metadata.moe_local_token_count,
                         tail.attn_cp_metadata.moe_local_token_count)
        self.assertEqual(full.attn_cp_metadata.moe_local_token_count.item(), 4096)

    def test_cpu_length_fallback_keeps_exact_rank_counts(self):
        for rank in range(8):
            with self.subTest(rank=rank):
                batch = self._batch(None, extend_lens=[1, 3, 17])
                self.assertEqual(self._count(batch, rank).item(), len(range(rank, 21, 8)))

    def test_zigzag_keeps_original_bound_without_creating_cp_count(self):
        batch = self._batch(21)
        self.assertIs(self._count(batch, 0, round_robin=False), batch.num_token_non_padded)
        self.assertIsNone(batch.attn_cp_metadata.moe_local_token_count)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
