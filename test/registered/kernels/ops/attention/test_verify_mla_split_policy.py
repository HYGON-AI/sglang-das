from unittest.mock import patch

from sglang.kernels.ops.attention import verify_mla


def _policy(*, head_dim: int, kv_group_num: int, n_head_blocks: int):
    policy = verify_mla.VerifyMLA.__new__(verify_mla.VerifyMLA)
    policy.head_dim = head_dim
    policy.kv_group_num = kv_group_num
    policy.n_head_blocks = n_head_blocks
    return policy


def test_minimax_gqa16_adaptive_split_preserves_small_batches():
    policy = _policy(head_dim=128, kv_group_num=16, n_head_blocks=4)
    with (
        patch.object(verify_mla, "MINIMAX_GQA16_TARGET_PROGRAMS", 128),
        patch.object(verify_mla, "MINIMAX_GQA16_MAX_SPLITS_MIN_BS", 8),
    ):
        assert policy._num_splits(4) == 8
        assert policy._num_splits(7) == 4
        assert policy._num_splits(8) == verify_mla.MAX_N_SPLITS


def test_adaptive_override_is_minimax_gqa16_only():
    policy = _policy(head_dim=128, kv_group_num=8, n_head_blocks=4)
    with (
        patch.object(verify_mla, "MINIMAX_GQA16_TARGET_PROGRAMS", 128),
        patch.object(verify_mla, "MINIMAX_GQA16_MAX_SPLITS_MIN_BS", 8),
    ):
        assert policy._num_splits(8) == 16
