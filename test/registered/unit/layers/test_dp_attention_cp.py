import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import communicator, dp_attention
from sglang.srt.layers.cp import utils as cp_utils
from sglang.srt.layers.cp.zigzag import compute_zigzag_cp_physical_token_count


def test_max_len_cp_gather_uses_full_tp_all_reduce():
    forward_batch = SimpleNamespace(dp_padding_mode=dp_attention.DpPaddingMode.MAX_LEN)
    global_tokens = torch.empty((2, 3))
    local_tokens = torch.empty((1, 3))

    with (
        patch.object(dp_attention, "configured_attn_cp_size", return_value=4),
        patch.object(dp_attention, "_dp_gather_via_all_reduce") as all_reduce,
        patch.object(dp_attention, "_dp_gather_via_all_gather") as all_gather,
    ):
        dp_attention.dp_gather_replicate(global_tokens, local_tokens, forward_batch)

    all_reduce.assert_called_once_with(
        global_tokens, local_tokens, forward_batch, False
    )
    all_gather.assert_not_called()


def test_cp_combine_all_reduces_then_selects_attention_dp_slot():
    global_tokens = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    local_tokens = torch.empty((3, 2), dtype=torch.float32)

    with (
        patch.object(dp_attention, "configured_attn_cp_size", return_value=4),
        patch.object(
            dp_attention,
            "tensor_model_parallel_all_reduce",
            side_effect=lambda tensor: tensor,
        ) as all_reduce,
        patch.object(dp_attention, "get_attention_dp_rank", return_value=1),
    ):
        dp_attention.dp_reduce_scatter_tensor(local_tokens, global_tokens)

    all_reduce.assert_called_once_with(global_tokens)
    torch.testing.assert_close(local_tokens, global_tokens[3:6])


def test_expanded_cp_dp_local_index_selects_dp_cp_slot():
    forward_batch = SimpleNamespace(
        global_num_tokens_cpu=[2, 2, 2, 2, 3, 3, 3, 3],
        global_num_tokens_gpu=torch.tensor([2, 2, 2, 2, 3, 3, 3, 3]),
        dp_local_start_pos=None,
        dp_local_num_tokens=None,
        dp_local_token_index=6,
    )

    start, count = dp_attention.get_dp_local_info(forward_batch)

    assert start.item() == 14
    assert count.item() == 3


def test_dp_layernorm_before_gather_preserves_residual_addition():
    hidden_states = torch.tensor([[1.0, 2.0]])
    residual = torch.tensor([[10.0, 20.0]])
    gathered = torch.empty_like(hidden_states)
    forward_batch = SimpleNamespace()
    context = SimpleNamespace(
        attn_dp_size=2,
        attn_tp_size=1,
        attn_tp_rank=0,
        force_layernorm_before_dp_gather=False,
    )

    class AddNorm:
        def __call__(self, x, residual_input):
            updated_residual = x + residual_input
            return updated_residual * 2, updated_residual

    def gather_replicate(output, local, _forward_batch):
        output.copy_(local)

    with (
        patch.object(
            communicator.CommunicateWithAllReduceAndLayerNormFn,
            "_skip_layernorm",
            return_value=False,
        ),
        patch.object(
            communicator,
            "use_symmetric_memory",
            return_value=contextlib.nullcontext(),
        ),
        patch.object(communicator, "get_tp_group", return_value=object()),
        patch.object(communicator, "get_global_dp_buffer", return_value=gathered),
        patch.object(
            communicator, "dp_gather_replicate", side_effect=gather_replicate
        ),
    ):
        output, updated_residual = (
            communicator.CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual(
                hidden_states,
                residual,
                forward_batch,
                AddNorm(),
                context,
                residual_input_mode=communicator.ScatterMode.TP_ATTN_FULL,
            )
        )

    torch.testing.assert_close(updated_residual, torch.tensor([[11.0, 22.0]]))
    torch.testing.assert_close(output, torch.tensor([[22.0, 44.0]]))


def test_zigzag_cp_physical_count_handles_multi_sequence_skew():
    # Remainders 1 and 7 favor opposite zigzag ranks.  The helper must mirror
    # metadata's aggregate-per-rank maximum before 2*CP alignment.
    assert compute_zigzag_cp_physical_token_count([1, 7], 4) == 8
    assert compute_zigzag_cp_physical_token_count([65536], 4) == 16384


def test_dp_cp_layout_falls_back_when_non_idle_peer_is_not_cp_eligible():
    assert cp_utils.normalize_dp_cp_token_counts([16384, 1], [4096, 0]) == [0, 0]
    assert cp_utils.normalize_dp_cp_token_counts([16384, 0], [4096, 0]) == [4096, 0]
    assert cp_utils.normalize_dp_cp_token_counts([16384, 16384], [4096, 4096]) == [
        4096,
        4096,
    ]


def test_synchronized_dp_fallback_disables_locally_eligible_cp_batch():
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_context_parallel_extend=lambda: True),
        input_ids=torch.empty(16384, dtype=torch.int64),
        extend_seq_lens_cpu=[16384],
        global_cp_num_tokens_cpu=[0, 0],
    )

    with patch.object(cp_utils, "enable_cp_v2", return_value=True):
        assert not cp_utils.is_cp_v2_active(forward_batch)


def test_cp_local_dp_state_uses_per_dp_shards_and_restores_counts():
    original_gpu = torch.tensor([64, 96], dtype=torch.int64)
    forward_batch = SimpleNamespace(
        global_cp_num_tokens_cpu=[16, 24],
        global_num_tokens_cpu=[64, 96],
        global_num_tokens_gpu=original_gpu,
        global_dp_buffer_len=160,
        dp_padding_mode=dp_attention.DpPaddingMode.SUM_LEN,
        dp_local_start_pos=None,
        dp_local_num_tokens=None,
        dp_local_token_index=None,
        cp_local_dp_layout=False,
    )
    parallel = SimpleNamespace(attn_dp_rank=1, attn_cp_rank=2, attn_cp_size=4)

    with (
        patch.object(cp_utils, "get_parallel", return_value=parallel),
        patch.object(dp_attention, "set_dp_buffer_len") as set_buffer,
    ):
        state = cp_utils._enter_cp_local_dp_state(forward_batch)
        assert forward_batch.global_num_tokens_cpu == [16, 24]
        assert forward_batch.dp_local_token_index == 1
        assert forward_batch.cp_local_dp_layout
        assert forward_batch.global_dp_buffer_len == 40
        set_buffer.assert_called_with(
            40,
            24,
            False,
            [16, 24],
            forward_batch.global_num_tokens_gpu,
        )

        cp_utils._restore_cp_local_dp_state(forward_batch, state)

    assert forward_batch.global_num_tokens_cpu == [64, 96]
    assert forward_batch.global_num_tokens_gpu is original_gpu
    assert forward_batch.dp_local_token_index is None
    assert not forward_batch.cp_local_dp_layout


def test_cp_v2_mla_tp_moe_uses_full_moe_token_layout():
    context = SimpleNamespace(is_layer_sparse=True)

    with (
        patch.object(
            communicator,
            "get_moe_a2a_backend",
            return_value=SimpleNamespace(is_none=lambda: True),
        ),
        patch.object(
            communicator,
            "should_use_flashinfer_cutlass_moe_fp4_allgather",
            return_value=False,
        ),
        patch.object(communicator, "enable_dwdp", return_value=False),
        patch.object(communicator, "is_enable_moe_cp_allgather", return_value=True),
        patch.object(communicator, "is_dsa_enable_prefill_cp", return_value=False),
        patch.object(communicator, "is_mla_prefill_cp_enabled", return_value=True),
        patch.object(cp_utils, "enable_cp_v2", return_value=True),
    ):
        mode = communicator.LayerScatterModes._compute_mlp_mode(context)

    assert mode is communicator.ScatterMode.MOE_FULL


def test_legacy_mla_cp_keeps_existing_full_layout():
    context = SimpleNamespace(is_layer_sparse=True)

    with (
        patch.object(
            communicator,
            "get_moe_a2a_backend",
            return_value=SimpleNamespace(is_none=lambda: True),
        ),
        patch.object(
            communicator,
            "should_use_flashinfer_cutlass_moe_fp4_allgather",
            return_value=False,
        ),
        patch.object(communicator, "enable_dwdp", return_value=False),
        patch.object(communicator, "is_enable_moe_cp_allgather", return_value=True),
        patch.object(communicator, "is_dsa_enable_prefill_cp", return_value=False),
        patch.object(communicator, "is_mla_prefill_cp_enabled", return_value=True),
        patch.object(cp_utils, "enable_cp_v2", return_value=False),
    ):
        mode = communicator.LayerScatterModes._compute_mlp_mode(context)

    assert mode is communicator.ScatterMode.FULL


def test_moe_full_postprocess_requires_normal_moe_tp_all_reduce():
    layer_communicator = object.__new__(communicator.LayerCommunicator)
    layer_communicator.allow_reduce_scatter = True
    layer_communicator._communicate_summable_tensor_pair_fn = (
        communicator.CommunicateSummableTensorPairFn._scatter_hidden_states_moe
    )

    assert not layer_communicator.should_use_reduce_scatter(SimpleNamespace())


def test_moe_full_gathers_all_cp_shards_before_experts():
    hidden_states = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    residual = torch.full_like(hidden_states, -1)
    forward_batch = SimpleNamespace(
        cp_local_dp_layout=True,
        forward_mode=SimpleNamespace(is_context_parallel_extend=lambda: True),
        attn_cp_metadata=SimpleNamespace(per_rank_actual_token=[2, 2, 2, 2]),
    )

    def all_gather(output, local):
        output.copy_(torch.cat([local + rank * 100 for rank in range(4)]))

    with (
        patch.object(
            communicator.CommunicateWithAllReduceAndLayerNormFn,
            "_gather_hidden_states_and_residual",
            return_value=(hidden_states, residual),
        ),
        patch.object(communicator, "get_moe_cp_size", return_value=4),
        patch.object(
            communicator, "moe_cp_all_gather_into_tensor", side_effect=all_gather
        ) as gather,
    ):
        gathered, returned_residual = (
            communicator.CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual_moe(
                hidden_states,
                residual,
                forward_batch,
                layernorm=object(),
                context=object(),
                residual_input_mode=communicator.ScatterMode.TP_ATTN_FULL,
            )
        )

    gather.assert_called_once()
    assert gathered.shape == (8, 3)
    torch.testing.assert_close(gathered[:2], hidden_states)
    torch.testing.assert_close(gathered[6:], hidden_states + 300)
    assert returned_residual is residual


def test_moe_full_gathers_and_scatters_dp1_cp_v2_tokens():
    hidden_states = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    residual = torch.full_like(hidden_states, -1)
    forward_batch = SimpleNamespace(
        cp_local_dp_layout=False,
        forward_mode=SimpleNamespace(is_context_parallel_extend=lambda: True),
        attn_cp_metadata=SimpleNamespace(per_rank_actual_token=[2, 2, 2, 2]),
    )

    def all_gather(output, local):
        output.copy_(torch.cat([local + rank * 100 for rank in range(4)]))

    with (
        patch.object(
            communicator.CommunicateWithAllReduceAndLayerNormFn,
            "_gather_hidden_states_and_residual",
            return_value=(hidden_states, residual),
        ),
        patch.object(communicator, "get_moe_cp_size", return_value=4),
        patch.object(communicator, "get_moe_cp_rank", return_value=2),
        patch.object(
            communicator, "moe_cp_all_gather_into_tensor", side_effect=all_gather
        ),
    ):
        gathered, _ = (
            communicator.CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual_moe(
                hidden_states,
                residual,
                forward_batch,
                layernorm=object(),
                context=object(),
                residual_input_mode=communicator.ScatterMode.TP_ATTN_FULL,
            )
        )
        scattered, _ = (
            communicator.CommunicateSummableTensorPairFn._scatter_hidden_states_moe(
                gathered,
                residual,
                forward_batch,
                context=SimpleNamespace(attn_dp_size=1),
            )
        )

    assert gathered.shape == (8, 3)
    torch.testing.assert_close(scattered, hidden_states + 200)
