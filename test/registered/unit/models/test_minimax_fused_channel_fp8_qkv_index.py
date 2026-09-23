from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.models import minimax_m3
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeCompressedTensorMethod:
    def apply(self, layer, x, bias=None):
        output = x.float() @ layer.weight.float().t()
        return output * layer.weight_scale.float().view(1, -1)


class _Projection(nn.Module):
    def __init__(self, weight, scale, quant_method, scheme):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale = nn.Parameter(scale, requires_grad=False)
        self.quant_method = quant_method
        self.scheme = scheme
        self.input_scale = None
        self.input_size = weight.shape[1]
        self.input_size_per_partition = weight.shape[1]
        self.output_size_per_partition = weight.shape[0]
        self.orig_dtype = torch.bfloat16
        self.params_dtype = torch.bfloat16


def _make_attention(*, static_input=False):
    method = _FakeCompressedTensorMethod()
    scheme = SimpleNamespace(
        strategy=SimpleNamespace(value="channel"),
        is_static_input_scheme=static_input,
    )
    q_weight = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 16
    i_weight = torch.arange(12, dtype=torch.float32).reshape(2, 6) / 8
    q_scale = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    i_scale = torch.tensor([[5.0], [6.0]])

    attention = minimax_m3.MiniMaxM3Attention.__new__(minimax_m3.MiniMaxM3Attention)
    nn.Module.__init__(attention)
    attention._fuse_qkv_index_enabled = True
    attention._fused_qkv_index = None
    attention.qkv_proj = _Projection(q_weight, q_scale, method, scheme)
    attention.index_qkv_proj = _Projection(i_weight, i_scale, method, scheme)
    return attention


def test_lightop_channel_fp8_qkv_index_fusion_preserves_rows_and_scales(monkeypatch):
    monkeypatch.setattr(minimax_m3, "_fuse_lightop_channel_fp8_qkv_index", True)
    attention = _make_attention()
    x = torch.arange(18, dtype=torch.float32).reshape(3, 6) / 10
    expected = torch.cat(
        [
            attention.qkv_proj.quant_method.apply(attention.qkv_proj, x),
            attention.index_qkv_proj.quant_method.apply(attention.index_qkv_proj, x),
        ],
        dim=-1,
    )

    assert attention.maybe_build_fused_qkv_index()
    fused = attention._fused_qkv_index
    torch.testing.assert_close(fused(x), expected)
    assert fused.weight.shape == (6, 6)
    assert fused.weight_scale.shape == (6, 1)
    assert fused.scheme is attention.qkv_proj.scheme
    assert attention.qkv_proj.weight.numel() == 0
    assert attention.index_qkv_proj.weight_scale.numel() == 0


def test_lightop_channel_fp8_qkv_index_fusion_rejects_static_input(monkeypatch):
    monkeypatch.setattr(minimax_m3, "_fuse_lightop_channel_fp8_qkv_index", True)
    attention = _make_attention(static_input=True)
    assert not attention.maybe_build_fused_qkv_index()
    assert attention._fused_qkv_index is None
