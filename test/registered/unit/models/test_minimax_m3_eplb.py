from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.models.minimax_m3 import MiniMaxM3MoE


def test_standard_ep_router_applies_expert_location_dispatch():
    hidden_states = torch.randn(2, 8)
    router_logits = torch.randn(2, 128)
    topk_output = object()
    expected_output = torch.randn(2, 8)
    dispatch_info = object()
    moe = SimpleNamespace(
        layer_id=11,
        _compute_router_logits=Mock(return_value=router_logits),
        topk=Mock(return_value=topk_output),
        experts=Mock(return_value=expected_output),
    )

    with patch(
        "sglang.srt.models.minimax_m3.ExpertLocationDispatchInfo.init_new",
        return_value=dispatch_info,
    ) as init_dispatch:
        output = MiniMaxM3MoE._forward_router_experts(moe, hidden_states)

    init_dispatch.assert_called_once_with(layer_id=11)
    moe.topk.assert_called_once_with(
        hidden_states,
        router_logits,
        expert_location_dispatch_info=dispatch_info,
    )
    moe.experts.assert_called_once_with(hidden_states, topk_output)
    assert output is expected_output
