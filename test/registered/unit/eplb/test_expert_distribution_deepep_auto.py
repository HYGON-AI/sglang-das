from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.eplb import expert_distribution as ed
from sglang.srt.runtime_context import get_context


def test_deepep_auto_selects_mode_switching_gatherer():
    args = SimpleNamespace(
        expert_distribution_recorder_mode="per_pass",
        moe_a2a_backend="deepep",
        deepep_mode="auto",
        elastic_ep_backend=None,
    )
    sentinel = object()
    with get_context().override_server_args(
        expert_distribution_recorder_mode="per_pass",
        moe_a2a_backend="deepep",
        deepep_mode="auto",
        elastic_ep_backend=None,
    ), patch.object(ed, "_DeepepAutoSinglePassGatherer", return_value=sentinel):
        assert ed._SinglePassGatherer.init_new(args, object(), 3) is sentinel


def test_deepep_auto_collects_normal_and_low_latency_counts():
    gatherer = ed._DeepepAutoSinglePassGatherer.__new__(
        ed._DeepepAutoSinglePassGatherer
    )
    gatherer._expert_location_metadata = SimpleNamespace(
        num_layers=2,
        num_local_physical_experts=3,
        num_physical_experts=6,
    )
    gatherer._rank = 1
    gatherer._elastic_ep_enabled = False
    gatherer._enable_global_physical_experts = False
    gatherer._data = torch.zeros((2, 3), dtype=torch.int32)

    gatherer.on_deepep_dispatch_normal(0, [1, 2, 3], None, None, None)
    gatherer.on_deepep_dispatch_low_latency(1, torch.tensor([4, 5, 6]))

    expected = torch.tensor([[0, 0, 0, 1, 2, 3], [0, 0, 0, 4, 5, 6]], dtype=torch.int32)
    assert torch.equal(gatherer.collect()["global_physical_count"], expected)
