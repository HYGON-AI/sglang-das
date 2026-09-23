import torch

import sglang.srt.layers.layernorm as layernorm


def _fake_fused_add_rms_norm(x, residual, weight, eps):
    summed = x + residual
    residual.copy_(summed)
    variance = summed.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = summed.float() * torch.rsqrt(variance + eps)
    x.copy_((normalized * weight.float()).to(x.dtype))


def _fake_legacy_fused_add_rms_norm(out, x, residual_out, residual, weight, eps):
    summed = x + residual
    residual_out.copy_(summed)
    variance = summed.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = summed.float() * torch.rsqrt(variance + eps)
    out.copy_((normalized * weight.float()).to(x.dtype))


def test_rmsnorm_hip_uses_vllm_four_argument_inplace_contract(monkeypatch):
    monkeypatch.setattr(layernorm, "_has_vllm_rms_norm", True)
    monkeypatch.setattr(layernorm, "_is_hcu", False)
    monkeypatch.setattr(layernorm, "fused_add_rms_norm", _fake_fused_add_rms_norm)
    norm = layernorm.RMSNorm(4, eps=1e-6)
    norm.weight.data.fill_(1)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    residual = torch.tensor([[0.5, 0.5, 0.5, 0.5]])

    out, residual_out = norm.forward_hip(x, residual)

    assert out is x
    assert residual_out is residual
    torch.testing.assert_close(residual, torch.tensor([[1.5, 2.5, 3.5, 4.5]]))


def test_gemma_rmsnorm_hip_uses_vllm_four_argument_inplace_contract(monkeypatch):
    monkeypatch.setattr(layernorm, "_has_vllm_rms_norm", True)
    monkeypatch.setattr(layernorm, "_is_hcu", False)
    monkeypatch.setattr(layernorm, "_use_aiter", False)
    monkeypatch.setattr(layernorm, "_use_hcu_lightop_gemma_rmsnorm", False)
    monkeypatch.setattr(layernorm, "fused_add_rms_norm", _fake_fused_add_rms_norm)
    norm = layernorm.GemmaRMSNorm(4, eps=1e-6)
    norm.weight.data.zero_()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    residual = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    post = torch.tensor([[0.25, 0.25, 0.25, 0.25]])

    out, residual_out = norm.forward_hip(x, residual, post)

    assert out is x
    assert residual_out is not residual
    torch.testing.assert_close(residual_out, torch.tensor([[1.75, 2.75, 3.75, 4.75]]))


def test_rmsnorm_hip_without_residual_uses_vllm_op(monkeypatch):
    calls = []

    def fake_rms_norm(x, weight, eps):
        calls.append((x, weight, eps))
        return x + 1

    monkeypatch.setattr(layernorm, "_has_vllm_rms_norm", True)
    monkeypatch.setattr(layernorm, "_is_hcu", False)
    monkeypatch.setattr(layernorm, "rms_norm", fake_rms_norm)
    norm = layernorm.RMSNorm(4, eps=1e-6)
    x = torch.ones((1, 4))

    out = norm.forward_hip(x)

    torch.testing.assert_close(out, x + 1)
    assert len(calls) == 1
    assert calls[0][0] is x
    torch.testing.assert_close(calls[0][1], norm.weight.data)
    assert calls[0][2] == norm.variance_epsilon


def test_gemma_rmsnorm_hip_without_residual_uses_vllm_op(monkeypatch):
    calls = []

    def fake_rms_norm(x, weight, eps):
        calls.append((x, weight, eps))
        return x + 1

    monkeypatch.setattr(layernorm, "_has_vllm_rms_norm", True)
    monkeypatch.setattr(layernorm, "_is_hcu", False)
    monkeypatch.setattr(layernorm, "_use_aiter", False)
    monkeypatch.setattr(layernorm, "rms_norm", fake_rms_norm)
    norm = layernorm.GemmaRMSNorm(4, eps=1e-6)
    x = torch.ones((1, 4))

    out = norm.forward_hip(x)

    torch.testing.assert_close(out, x + 1)
    assert len(calls) == 1
    assert calls[0][0] is x
    torch.testing.assert_close(calls[0][1], norm.gemma_weight)
    assert calls[0][2] == norm.variance_epsilon


def test_rmsnorm_hip_supports_legacy_six_argument_contract(monkeypatch):
    monkeypatch.setattr(layernorm, "_has_vllm_rms_norm", True)
    monkeypatch.setattr(layernorm, "_is_hcu", False)
    monkeypatch.setattr(
        layernorm, "fused_add_rms_norm", _fake_legacy_fused_add_rms_norm
    )
    norm = layernorm.RMSNorm(4, eps=1e-6)
    norm.weight.data.fill_(1)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    residual = torch.tensor([[0.5, 0.5, 0.5, 0.5]])

    out, residual_out = norm.forward_hip(x, residual)

    assert out is not x
    assert residual_out is not residual
    torch.testing.assert_close(residual_out, torch.tensor([[1.5, 2.5, 3.5, 4.5]]))
