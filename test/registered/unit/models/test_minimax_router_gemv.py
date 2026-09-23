import pytest
import torch

from sglang.kernels.ops.moe.minimax_router_gemv import (
    can_use_minimax_router_gemv,
    minimax_router_gemv,
)


def test_minimax_router_gemv_rejects_cpu_tensors():
    hidden_states = torch.empty((1, 6144), dtype=torch.bfloat16)
    router_weight = torch.empty((128, 6144), dtype=torch.bfloat16)

    assert not can_use_minimax_router_gemv(hidden_states, router_weight)
    with pytest.raises(ValueError, match="CUDA/HIP contiguous BF16"):
        minimax_router_gemv(hidden_states, router_weight)


def test_minimax_router_gemv_rejects_wrong_shape_before_launch():
    hidden_states = torch.empty((65, 6144), dtype=torch.bfloat16)
    router_weight = torch.empty((128, 6144), dtype=torch.bfloat16)

    assert not can_use_minimax_router_gemv(hidden_states, router_weight)
