from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.layers.moe import mega_moe
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("use_draft_weights", [False, True])
def test_topk_uses_the_layout_of_the_loaded_experts(monkeypatch, use_draft_weights):
    # Target and draft layer 0 share an index but have different expert counts.
    target_map = torch.tensor([415])
    lookup = Mock(return_value=target_map)
    monkeypatch.setattr(mega_moe.ExpertLocationDispatchInfo, "init_new", lookup)

    def topk(hidden, logits, *, num_token_non_padded, expert_location_dispatch_info):
        logical_ids = logits.argmax(dim=-1)
        ids = logical_ids if expert_location_dispatch_info is None else target_map[logical_ids]
        return SimpleNamespace(topk_ids=ids, topk_weights=torch.ones_like(ids))

    gate = Mock(return_value=torch.zeros(1, 1), e_score_correction_bias_vl=None)
    moe = SimpleNamespace(
        gate=gate, topk=topk, layer_id=0, is_hash=False,
        experts=SimpleNamespace(_skip_expert_location=use_draft_weights),
    )
    ids, _ = mega_moe._mega_topk(moe, torch.zeros(1, 4), None, None)
    if use_draft_weights:
        lookup.assert_not_called()
        assert ids.tolist() == [0]
    else:
        lookup.assert_called_once_with(layer_id=0)
        assert ids.tolist() == [415]
