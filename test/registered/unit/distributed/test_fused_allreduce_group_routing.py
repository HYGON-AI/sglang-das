import torch

from sglang.srt.distributed import communication_op
from sglang.srt.layers import layernorm


class _FakeGroup:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def fused_allreduce_rmsnorm(self, *args):
        self.calls.append((self.name, "rmsnorm"))
        return args[0], args[1]

    def fused_allreduce_rmsnorm_quant_per_group(self, *args, **kwargs):
        self.calls.append((self.name, "quant"))
        return args[0], args[1], torch.ones(1)


def test_fused_allreduce_routes_to_requested_group(monkeypatch):
    calls = []
    groups = {
        "tp": _FakeGroup("tp", calls),
        "attn_tp": _FakeGroup("attn_tp", calls),
        "moe_tp": _FakeGroup("moe_tp", calls),
    }
    monkeypatch.setattr(communication_op, "get_tp_group", lambda: groups["tp"])
    monkeypatch.setattr(
        communication_op, "get_attn_tp_group", lambda: groups["attn_tp"]
    )
    monkeypatch.setattr(communication_op, "get_moe_tp_group", lambda: groups["moe_tp"])
    x = torch.ones(1, 4)

    for group in groups:
        communication_op.tensor_model_parallel_fused_allreduce_rmsnorm(
            x, x, x[0], 1e-6, group=group
        )
        communication_op.tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group(
            x, x, x[0], 1e-6, group=group
        )

    assert calls == [
        ("tp", "rmsnorm"),
        ("tp", "quant"),
        ("attn_tp", "rmsnorm"),
        ("attn_tp", "quant"),
        ("moe_tp", "rmsnorm"),
        ("moe_tp", "quant"),
    ]


def test_gemma_allreduce_fusion_preserves_requested_group(monkeypatch):
    requested = []

    def fake_forward(*args, **kwargs):
        requested.append(kwargs["use_attn_tp_group"])
        return args[1], args[2]

    monkeypatch.setattr(layernorm, "_forward_with_allreduce_fusion", fake_forward)
    norm = layernorm.GemmaRMSNorm(4, eps=1e-6)
    x = torch.ones(1, 4)
    residual = torch.zeros_like(x)

    norm.forward_with_allreduce_fusion(x, residual, use_attn_tp_group=False)

    assert requested == [False]
