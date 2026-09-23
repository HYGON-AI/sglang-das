from types import SimpleNamespace

import torch.nn as nn

from sglang.srt.models.minimax_m3 import MiniMaxM3SparseForCausalLM
from sglang.srt.models.minimax_m3_vl import (
    MiniMaxM3SparseForConditionalGeneration,
)


def _make_target(cls, config):
    target = cls.__new__(cls)
    nn.Module.__init__(target)
    target.pp_group = SimpleNamespace(is_last_rank=True)
    target.config = config
    target.model = SimpleNamespace(
        layers_to_capture=[],
        layers=[SimpleNamespace() for _ in range(60)],
    )
    target.capture_aux_hidden_states = False
    return target


def _assert_minimax_m3_default_capture(target):
    target.set_eagle3_layers_to_capture()

    assert target.capture_aux_hidden_states
    assert target.model.layers_to_capture == [2, 30, 57]
    assert [
        i
        for i, layer in enumerate(target.model.layers)
        if getattr(layer, "_is_layer_to_capture", False)
    ] == [2, 30, 57]


def _assert_minimax_m3_explicit_capture(target):
    target.set_eagle3_layers_to_capture([2, 30, 57])

    assert target.capture_aux_hidden_states
    assert target.model.layers_to_capture == [3, 31, 58]
    assert [
        i
        for i, layer in enumerate(target.model.layers)
        if getattr(layer, "_is_layer_to_capture", False)
    ] == [3, 31, 58]


def test_minimax_m3_causal_eagle3_default_capture_layers():
    target = _make_target(
        MiniMaxM3SparseForCausalLM,
        SimpleNamespace(num_hidden_layers=60),
    )
    _assert_minimax_m3_default_capture(target)


def test_minimax_m3_vl_eagle3_default_capture_layers():
    target = _make_target(
        MiniMaxM3SparseForConditionalGeneration,
        SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=60)),
    )
    _assert_minimax_m3_default_capture(target)


def test_minimax_m3_causal_eagle3_explicit_capture_layers():
    target = _make_target(
        MiniMaxM3SparseForCausalLM,
        SimpleNamespace(num_hidden_layers=60),
    )
    _assert_minimax_m3_explicit_capture(target)


def test_minimax_m3_vl_eagle3_explicit_capture_layers():
    target = _make_target(
        MiniMaxM3SparseForConditionalGeneration,
        SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=60)),
    )
    _assert_minimax_m3_explicit_capture(target)
