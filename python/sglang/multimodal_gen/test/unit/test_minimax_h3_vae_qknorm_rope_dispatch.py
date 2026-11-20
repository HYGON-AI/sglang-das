# SPDX-License-Identifier: Apache-2.0
"""Opt-in dispatch and metadata safety for H3 VAE QK-norm/RoPE fusion."""

from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops import diffusion
from sglang.kernels.ops.diffusion import can_use_minimax_h3_vae_qknorm_rope
from sglang.srt.environ import envs
from sglang.srt.utils import is_hcu

requires_hcu = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_hcu()), reason="Requires HCU"
)
FLAG = "SGLANG_ENABLE_DIFFUSION_MINIMAX_H3_VAE_QKNORM_ROPE"


def test_vae_qknorm_rope_flag_is_opt_in(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert not envs.SGLANG_ENABLE_DIFFUSION_MINIMAX_H3_VAE_QKNORM_ROPE.get()
    monkeypatch.setenv(FLAG, "1")
    assert envs.SGLANG_ENABLE_DIFFUSION_MINIMAX_H3_VAE_QKNORM_ROPE.get()
    monkeypatch.setenv(FLAG, "false")
    assert not envs.SGLANG_ENABLE_DIFFUSION_MINIMAX_H3_VAE_QKNORM_ROPE.get()


@requires_hcu
@pytest.mark.parametrize(
    "case",
    [
        "positions_dtype",
        "positions_size",
        "cache_stride",
        "head_dim",
        "norm_shape",
        "eps_none",
        "eps_mismatch",
        "cpu_positions",
    ],
)
def test_vae_qknorm_rope_rejects_unsupported_metadata(case):
    qkv = torch.randn((1, 17, 2, 192), device="cuda", dtype=torch.float16)
    q, k, _ = qkv.chunk(3, dim=-1)
    cache = torch.randn((17, 48), device=q.device, dtype=q.dtype)
    positions = torch.arange(17, device=q.device)
    norm_q = torch.nn.RMSNorm(64, eps=1e-5, elementwise_affine=False)
    norm_k = torch.nn.RMSNorm(64, eps=1e-5, elementwise_affine=False)
    assert can_use_minimax_h3_vae_qknorm_rope(
        q, k, (None, None, cache, positions), norm_q, norm_k
    )
    if case == "positions_dtype":
        positions = positions.float()
    elif case == "positions_size":
        positions = positions[:-1]
    elif case == "cache_stride":
        cache = torch.empty((17, 96), device=q.device, dtype=q.dtype)[:, ::2]
    elif case == "head_dim":
        q, k = q[..., :32], k[..., :32]
    elif case == "norm_shape":
        norm_q = torch.nn.RMSNorm(32, eps=1e-5, elementwise_affine=False)
    elif case == "eps_none":
        norm_q.eps = norm_k.eps = None
    elif case == "eps_mismatch":
        norm_k.eps *= 2
    elif case == "cpu_positions":
        positions = positions.cpu()
    assert not can_use_minimax_h3_vae_qknorm_rope(
        q, k, (None, None, cache, positions), norm_q, norm_k
    )


@requires_hcu
@pytest.mark.parametrize("enabled", [False, True])
def test_vae_attention_call_site(monkeypatch, enabled):
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.attention import (
        Attention,
    )

    torch.manual_seed(5)
    module = (
        Attention(heads=2, dim_head=64, qk_norm_type="rms_norm", qk_norm_affine=False)
        .cuda()
        .half()
    )
    module.attn = None
    x = torch.randn((1, 17, 128), device="cuda", dtype=torch.float16)
    angles = torch.randn((17, 24), device=x.device)
    cache = torch.cat((angles.cos(), angles.sin()), dim=1).half().contiguous()
    rope = (None, None, cache, torch.arange(17, device=x.device))
    with torch.inference_mode():
        monkeypatch.delenv(FLAG, raising=False)
        expected = module(x, rope)
        monkeypatch.setenv(FLAG, "1" if enabled else "0")
        with patch.object(
            diffusion,
            "minimax_h3_vae_qknorm_rope",
            wraps=diffusion.minimax_h3_vae_qknorm_rope,
        ) as fused:
            actual = module(x, rope)
        assert fused.call_count == int(enabled)
    torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.02)
