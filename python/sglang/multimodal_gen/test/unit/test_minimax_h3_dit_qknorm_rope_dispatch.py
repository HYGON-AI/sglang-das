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
FLAG_4WAVE = "SGLANG_DIFFUSION_ENABLE_MINIMAX_H3_DIT_QKNORM_ROPE"
FLAG_2WAVE = FLAG_4WAVE + "_2WAVE"
FLAG_4HEADS = FLAG_4WAVE + "_4HEADS"


@pytest.mark.parametrize("flag", [FLAG_4WAVE, FLAG_2WAVE, FLAG_4HEADS])
def test_dit_qknorm_rope_flag_is_opt_in(monkeypatch, flag):
    monkeypatch.delenv(flag, raising=False)
    assert not getattr(envs, flag)
    monkeypatch.setenv(flag, "1")
    assert getattr(envs, flag)
    monkeypatch.setenv(flag, "false")
    assert not getattr(envs, flag)


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
@pytest.mark.parametrize(
    ("enabled_flags", "variant"),
    [
        ((), None),
        ((FLAG_4WAVE,), "4wave"),
        ((FLAG_2WAVE,), "2wave"),
        ((FLAG_4HEADS,), "4heads"),
        ((FLAG_4WAVE, FLAG_2WAVE), "2wave"),
        ((FLAG_4WAVE, FLAG_4HEADS), "4heads"),
        ((FLAG_4WAVE, FLAG_2WAVE, FLAG_4HEADS), "4heads"),
    ],
)
def test_dit_attention_call_site(monkeypatch, enabled_flags, variant):
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
        for flag in (FLAG_4WAVE, FLAG_2WAVE, FLAG_4HEADS):
            monkeypatch.delenv(flag, raising=False)
        expected = module(x, **kwargs)
        for flag in enabled_flags:
            monkeypatch.setenv(flag, "1")
        with patch(target, wraps=minimax_h3_dit_qknorm_rope) as fused:
            actual = module(x, **kwargs)
        if variant is None:
            fused.assert_not_called()
        else:
            assert fused.call_count == 1
            assert fused.call_args.kwargs["variant"] == variant
    torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.02)
