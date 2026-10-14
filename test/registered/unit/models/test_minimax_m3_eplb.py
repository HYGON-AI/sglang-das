from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from sglang.srt.layers.moe.moe_runner.aiter import AiterRunnerCore
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.models import minimax_m3
from sglang.srt.models.minimax_m3 import MiniMaxM3MoE
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-c-test-cpu")


def test_minimax_m3_marks_its_runner_for_cached_weight_layouts(monkeypatch):
    class FakeExperts(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.moe_runner_config = MoeRunnerConfig()
            self.runner_core = AiterRunnerCore(self.moe_runner_config)

    parallel = SimpleNamespace(
        moe_ep_size=1, moe_tp_size=1, tp_size=1, moe_tp_rank=0, tp_rank=0
    )
    backend = SimpleNamespace(is_none=lambda: True, is_deepep=lambda: False)
    monkeypatch.setattr(minimax_m3, "get_parallel", lambda: parallel)
    monkeypatch.setattr(minimax_m3, "get_moe_a2a_backend", lambda: backend)
    monkeypatch.setattr(
        minimax_m3,
        "get_exec",
        lambda: SimpleNamespace(moe=SimpleNamespace(ep_num_redundant_experts=0)),
    )
    monkeypatch.setattr(minimax_m3, "is_shared_experts_fusion_disabled", lambda: True)
    monkeypatch.setattr(minimax_m3, "has_per_rank_fused_shared_slots", lambda _: False)
    monkeypatch.setattr(minimax_m3, "get_moe_impl_class", lambda _: FakeExperts)
    monkeypatch.setattr(minimax_m3, "TopK", lambda **kwargs: nn.Identity())
    monkeypatch.setattr(
        minimax_m3,
        "ReplicatedLinear",
        lambda in_features, out_features, **kwargs: nn.Linear(
            in_features, out_features, bias=False
        ),
    )
    config = SimpleNamespace(
        num_local_experts=2,
        n_shared_experts=None,
        num_experts_per_tok=1,
        hidden_size=4,
        intermediate_size=8,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        scoring_func="softmax",
    )

    model = MiniMaxM3MoE(config, layer_id=0)

    assert model.experts.moe_runner_config.is_minimax_m3
    assert model.experts.runner_core.config is model.experts.moe_runner_config
    assert not MoeRunnerConfig().is_minimax_m3


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
