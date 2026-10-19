"""HCU legacy CP metadata must not be consumed as generic strategy metadata."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.layers.cp.interleave import InterleaveCPStrategy
from sglang.srt.layers.utils.hcu_cp_utils import ContextParallelMetadata
from sglang.srt.model_executor import forward_batch_info as fb
from sglang.test.ci.ci_register import register_cpu_ci

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
