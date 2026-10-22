from types import SimpleNamespace

import pytest
import torch.nn as nn

import sglang.srt.models.minimax_m3 as minimax_m3
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-c-test-cpu")


@pytest.mark.parametrize("use_deepep", [False, True])
def test_minimax_model_passes_marker_during_deepep_construction(
    monkeypatch, use_deepep
):
    received = {}

    class FakeDeepEPMoE(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            assert kwargs.pop("is_minimax_m3") is True
            received.update(kwargs)
            self.moe_runner_config = SimpleNamespace(is_minimax_m3=True)

    class OtherMoE(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            assert "is_minimax_m3" not in kwargs
            received.update(kwargs)
            self.moe_runner_config = SimpleNamespace(is_minimax_m3=False)

    parallel = SimpleNamespace(
        tp_size=8, tp_rank=0, moe_ep_size=8, moe_tp_size=1, moe_tp_rank=0
    )
    a2a = SimpleNamespace(is_none=lambda: not use_deepep, is_deepep=lambda: use_deepep)
    monkeypatch.setattr(minimax_m3, "DeepEPMoE", FakeDeepEPMoE)
    monkeypatch.setattr(
        minimax_m3,
        "get_moe_impl_class",
        lambda quant: FakeDeepEPMoE if use_deepep else OtherMoE,
    )
    monkeypatch.setattr(minimax_m3, "get_parallel", lambda: parallel)
    monkeypatch.setattr(minimax_m3, "get_moe_a2a_backend", lambda: a2a)
    monkeypatch.setattr(
        minimax_m3,
        "get_exec",
        lambda: SimpleNamespace(moe=SimpleNamespace(ep_num_redundant_experts=0)),
    )
    monkeypatch.setattr(minimax_m3, "is_shared_experts_fusion_disabled", lambda: True)
    monkeypatch.setattr(minimax_m3, "has_per_rank_fused_shared_slots", lambda n: False)
    monkeypatch.setattr(minimax_m3, "_is_hip", False)
    monkeypatch.setattr(minimax_m3, "TopK", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(
        minimax_m3, "ReplicatedLinear", lambda *args, **kwargs: nn.Identity()
    )
    config = SimpleNamespace(
        hidden_size=6144,
        intermediate_size=3072,
        num_local_experts=128,
        num_experts_per_tok=4,
        n_shared_experts=None,
        use_routing_bias=False,
        scoring_func="sigmoid",
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
    )
    layer = minimax_m3.MiniMaxM3MoE(config, layer_id=3)
    assert layer.experts.moe_runner_config.is_minimax_m3
    assert received["layer_id"] == 3
    assert received["num_experts"] == 128
