# SPDX-License-Identifier: Apache-2.0
"""Opt-in dispatch and in-place metadata safety for H3 DiT QK-norm/RoPE."""

from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    can_use_minimax_h3_dit_qknorm_rope,
    minimax_h3_dit_qknorm_rope,
)
from sglang.multimodal_gen import envs
from sglang.srt.utils import is_hcu

requires_hcu = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_hcu()), reason="Requires HCU"
)
FLAG = "SGLANG_DIFFUSION_ENABLE_MINIMAX_H3_DIT_QKNORM_ROPE"


def test_dit_qknorm_rope_flag_is_opt_in(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert not envs.SGLANG_DIFFUSION_ENABLE_MINIMAX_H3_DIT_QKNORM_ROPE
    monkeypatch.setenv(FLAG, "1")
    assert envs.SGLANG_DIFFUSION_ENABLE_MINIMAX_H3_DIT_QKNORM_ROPE
    monkeypatch.setenv(FLAG, "false")
    assert not envs.SGLANG_DIFFUSION_ENABLE_MINIMAX_H3_DIT_QKNORM_ROPE


@requires_hcu
@pytest.mark.parametrize(
    "case",
    [
        "positions_dtype",
        "positions_size",
        "cache_stride",
        "weight_stride",
        "head_dim",
        "eps_mismatch",
        "cpu_positions",
        "overlapping_rows",
    ],
)
def test_dit_qknorm_rope_rejects_unsupported_metadata(case):
    qkv = torch.randn((17, 3, 2, 128), device="cuda", dtype=torch.bfloat16)
    q, k, _ = qkv.unbind(1)
    wq = torch.ones(128, device=q.device, dtype=q.dtype)
    wk = torch.ones_like(wq)
    cache = torch.randn((17, 96), device=q.device, dtype=q.dtype)
    positions = torch.arange(17, device=q.device)
    eps_q = eps_k = 1e-5
    assert can_use_minimax_h3_dit_qknorm_rope(
        q, k, wq, wk, cache, eps_q, eps_k, positions=positions
    )
    if case == "positions_dtype":
        positions = positions.float()
    elif case == "positions_size":
        positions = positions[:-1]
    elif case == "cache_stride":
        cache = torch.empty((17, 192), device=q.device, dtype=q.dtype)[:, ::2]
    elif case == "weight_stride":
        wq = torch.ones(256, device=q.device, dtype=q.dtype)[::2]
    elif case == "head_dim":
        q, k = q[..., :64], k[..., :64]
    elif case == "eps_mismatch":
        eps_k *= 2
    elif case == "cpu_positions":
        positions = positions.cpu()
    elif case == "overlapping_rows":
        q = q[:1].expand_as(q)
    assert not can_use_minimax_h3_dit_qknorm_rope(
        q, k, wq, wk, cache, eps_q, eps_k, positions=positions
    )


@requires_hcu
@pytest.mark.parametrize("enabled", [False, True])
def test_dit_attention_call_site(monkeypatch, enabled):
    from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
        MiniMaxH3DiTArchConfig,
    )
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3Attention
    from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
        ensure_distributed_env_defaults,
    )

    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    torch.manual_seed(7)
    arch = MiniMaxH3DiTArchConfig(
        hidden_size=128, num_attention_heads=2, attention_head_dim=128
    )
    with torch.device("cuda"):
        module = MiniMaxH3Attention(
            arch, None, prefix="blocks.0.attn", bcg_breakpoint=False
        )
    with torch.no_grad():
        for name, value in module.named_parameters():
            if name.endswith("norm.weight"):
                value.fill_(1.0)
            else:
                value.normal_(0.0, 0.05)
    module._attention_impl = SDPAImpl(
        num_heads=2, head_size=128, causal=False, softmax_scale=128**-0.5
    )
    assert not module._use_fused_qknorm_rope
    x = torch.randn((17, 128), device="cuda", dtype=torch.bfloat16)
    angles = torch.randn((17, 48), device=x.device)
    cache = torch.cat((angles.cos(), angles.sin()), dim=1).bfloat16().contiguous()
    kwargs = dict(
        rope_cache=(cache, torch.arange(17, device=x.device)),
        cu_seqlens=torch.tensor([0, 17], device=x.device, dtype=torch.int32),
        cu_seqlens_host=(0, 17),
        max_seqlen=17,
    )
    target = "sglang.kernels.ops.diffusion.minimax_h3_dit_qknorm_rope"
    with torch.inference_mode():
        monkeypatch.delenv(FLAG, raising=False)
        expected = module(x, **kwargs)
        monkeypatch.setenv(FLAG, "1" if enabled else "0")
        with patch(target, wraps=minimax_h3_dit_qknorm_rope) as fused:
            actual = module(x, **kwargs)
        assert fused.call_count == int(enabled)
    torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.02)
