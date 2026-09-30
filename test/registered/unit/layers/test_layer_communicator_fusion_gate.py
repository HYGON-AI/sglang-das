import types
import unittest
from unittest.mock import patch

from sglang.srt.layers import communicator as comm
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fake_communicator():
    return types.SimpleNamespace(
        _speculative_algo=None,
        layer_scatter_modes=types.SimpleNamespace(mlp_mode=ScatterMode.TP_ATTN_FULL),
        is_last_layer=False,
        _context=types.SimpleNamespace(tp_size=4),
    )


class TestFuseMlpAllReduceGate(CustomTestCase):
    """Hybrid EP+TP must not fuse the post-experts all-reduce away.

    The fused residual+LN reduces over a single group, but with moe_ep_size > 1
    and moe_tp_size > 1 the post-experts reduction spans two disjoint groups
    (_MOE_EP then _MOE_TP) and should_skip_post_experts_all_reduce() drops both
    once fusion is published. The result is activations reduced over only half
    the peers -- wrong output, no crash. Observed as garbage completions on
    Qwen3-30B-A3B with --tp-size 4 --ep-size 2.
    """

    def _should_fuse(self, *, moe_ep_size, moe_tp_size):
        forward_batch = types.SimpleNamespace(
            input_ids=types.SimpleNamespace(shape=(8,))
        )
        with (
            patch.object(comm, "is_enable_moe_cp_allgather", return_value=False),
            patch.object(comm, "apply_flashinfer_allreduce_fusion", return_value=True),
            patch.object(
                comm,
                "get_attn_tp_context",
                return_value=types.SimpleNamespace(input_scattered=False),
            ),
            get_parallel().override(
                moe_ep_size=moe_ep_size, moe_tp_size=moe_tp_size, tp_size=4
            ),
        ):
            return LayerCommunicator.should_fuse_mlp_allreduce_with_next_layer(
                _fake_communicator(), forward_batch
            )

    def test_hybrid_ep_tp_does_not_fuse(self):
        self.assertFalse(self._should_fuse(moe_ep_size=2, moe_tp_size=2))

    def test_pure_tp_still_fuses(self):
        self.assertTrue(self._should_fuse(moe_ep_size=1, moe_tp_size=4))

    def test_pure_ep_still_fuses(self):
        self.assertTrue(self._should_fuse(moe_ep_size=4, moe_tp_size=1))


class TestAlignedMoeDpLayout(CustomTestCase):
    def _modes(self, **parallel_overrides):
        values = dict(
            tp_size=8,
            attn_dp_size=4,
            attn_tp_size=2,
            attn_cp_size=1,
            moe_dp_size=4,
            moe_tp_size=2,
            moe_ep_size=1,
        )
        values.update(parallel_overrides)
        with (
            patch.object(comm, "is_dp_attention_enabled", return_value=True),
            patch.object(
                comm,
                "get_moe_a2a_backend",
                return_value=types.SimpleNamespace(is_none=lambda: True),
            ),
            patch.object(comm, "is_enable_moe_cp_allgather", return_value=False),
            patch.object(
                comm,
                "should_use_flashinfer_cutlass_moe_fp4_allgather",
                return_value=False,
            ),
            patch.object(comm, "enable_dwdp", return_value=False),
            get_parallel().override(**values),
        ):
            return LayerScatterModes.init_new(
                num_layers=60,
                layer_id=10,
                is_layer_sparse=True,
                is_previous_layer_sparse=True,
                is_next_layer_sparse=True,
            )

    def test_matching_dp_and_tp_keep_attention_local_layout(self):
        modes = self._modes()
        self.assertEqual(modes.mlp_mode, ScatterMode.TP_ATTN_FULL)
        self.assertEqual(modes.middle_residual_mode, ScatterMode.TP_ATTN_FULL)
        self.assertEqual(modes.layer_output_mode, ScatterMode.TP_ATTN_FULL)

    def test_mismatched_moe_dp_falls_back_to_global_full(self):
        modes = self._modes(moe_dp_size=1, moe_tp_size=8)
        self.assertEqual(modes.mlp_mode, ScatterMode.FULL)

    def test_ep_layout_is_not_assumed_to_match(self):
        modes = self._modes(moe_dp_size=2, moe_tp_size=2, moe_ep_size=2)
        self.assertEqual(modes.mlp_mode, ScatterMode.FULL)


if __name__ == "__main__":
    unittest.main()
