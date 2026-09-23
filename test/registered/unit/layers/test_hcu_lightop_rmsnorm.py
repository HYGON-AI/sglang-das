import torch

import sglang.srt.layers.layernorm as layernorm


def test_gemma_lightop_fp8_quant_preserves_residual_contract(monkeypatch):
    calls = []

    def fake_quant(x, weight, eps, fp8type, residual, update_input):
        calls.append((fp8type, update_input))
        if residual is not None:
            residual.add_(x)
            source = residual
        else:
            source = x
        scale = source.abs().amax(dim=-1, keepdim=True).float().clamp_min(1e-6)
        return source / scale, scale

    monkeypatch.setattr(layernorm, "_use_hcu_lightop_gemma_rmsnorm", True)
    monkeypatch.setattr(
        layernorm, "gemma_rms_norm_fp8_quant_hcu", fake_quant, raising=False
    )
    norm = layernorm.GemmaRMSNorm(4, eps=1e-6)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    residual = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    post = torch.tensor([[0.25, 0.25, 0.25, 0.25]])

    quantized, residual_out = norm.forward_with_lightop_fp8_quant(x, residual, post)

    assert calls == [(0, False)]
    assert residual_out is not residual
    torch.testing.assert_close(residual_out, torch.tensor([[1.75, 2.75, 3.75, 4.75]]))
    assert len(quantized) == 2
