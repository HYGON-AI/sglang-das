"""Checkpoint encoding contract for compressed-tensors to SlimQuant W4A8."""

from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    _resolve_weight_param_name,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-c-test-cpu")


def _convert(param, loaded, name):
    captured = []

    def stop_after_conversion(param, value, shard, expert):
        captured.append(value)
        return True

    owner = SimpleNamespace(_load_gguf_weight=stop_after_conversion)
    FusedMoE._weight_loader_impl(owner, param, loaded, name, "w1", 0)
    return captured[0]


def test_all_signed_int4_values_keep_k_order():
    values = torch.arange(-8, 8, dtype=torch.int64)
    offset = (values + 8).reshape(2, 8)
    shifts = 4 * torch.arange(8, dtype=torch.int64)
    checkpoint = (offset << shifts).sum(-1).to(torch.int32).reshape(1, 2)
    param = torch.nn.Parameter(
        torch.empty((1, 8), dtype=torch.int8), requires_grad=False
    )
    param.compressed_tensors_packed_int4 = True
    converted = _convert(param, checkpoint, "weight")
    byte = converted.to(torch.uint8).flatten().long()
    decoded = torch.stack((byte >> 4, byte & 15), dim=-1).flatten()
    decoded = torch.where(decoded >= 8, decoded - 16, decoded)
    torch.testing.assert_close(decoded, values)
    assert converted.shape == param.shape


def test_checkpoint_scales_match_native_factor_sixteen():
    scale = torch.tensor([[0.25], [2.5]], dtype=torch.float32)
    param = torch.nn.Parameter(torch.empty_like(scale), requires_grad=False)
    param.compressed_tensors_packed_int4 = True
    converted = _convert(param, scale, "weight_scale")
    torch.testing.assert_close(converted * 16, scale, rtol=0, atol=0)


def test_unmarked_weights_and_parameter_names_are_unchanged():
    param = torch.nn.Parameter(
        torch.empty((1, 8), dtype=torch.int8), requires_grad=False
    )
    raw = torch.ones((1, 2), dtype=torch.int32)
    assert _convert(param, raw, "weight") is raw
    assert (
        _resolve_weight_param_name("layer.weight_packed", {"layer.weight": param})
        is None
    )
    param.compressed_tensors_packed_int4 = True
    assert (
        _resolve_weight_param_name("layer.weight_packed", {"layer.weight": param})
        == "layer.weight"
    )
