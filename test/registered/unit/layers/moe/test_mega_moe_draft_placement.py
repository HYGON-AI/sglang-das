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


@pytest.mark.parametrize(
    "extend,images,dp_size,capture,use_vision",
    [
        (True, False, 1, False, False),
        (True, True, 1, False, True),
        (True, False, 2, False, True),
        (True, False, 1, True, True),
        (False, False, 1, False, True),
    ],
)
def test_text_prefill_router(monkeypatch, extend, images, dp_size, capture, use_vision):
    result = SimpleNamespace(topk_ids=None, topk_weights=None)
    vision = Mock(return_value=result)
    text = Mock(return_value=result)
    monkeypatch.setattr(mega_moe, "vision_topk", vision)
    monkeypatch.setattr(mega_moe, "_is_hcu", True)
    monkeypatch.setattr(mega_moe, "get_attention_dp_size", lambda: dp_size)
    monkeypatch.setattr(mega_moe, "get_is_capture_mode", lambda: capture)
    monkeypatch.setattr(mega_moe.ExpertLocationDispatchInfo, "init_new", Mock())
    moe = SimpleNamespace(
        gate=Mock(e_score_correction_bias_vl=object()), experts=object(),
        topk=text, layer_id=0, is_hash=False,
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: extend),
        input_embeds=None, mm_input_embeds=None, contains_mm_inputs=lambda: images,
        moe_num_token_non_padded=lambda: None,
    )
    mega_moe._mega_topk(moe, None, batch, None)
    assert vision.call_count == int(use_vision)
    assert text.call_count == int(not use_vision)
