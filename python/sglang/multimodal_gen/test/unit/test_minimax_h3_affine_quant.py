# SPDX-License-Identifier: Apache-2.0
"""Exact AdaLN/INT8 fusion, dynamic shapes and model dispatch fallbacks."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.quantization.minimax_h3_quant import (
    MiniMaxH3Int8LinearMethod,
    affine,
    linear,
    quant_affine_rows,
    quant_rows,
    supports_int8_affine_quant,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import _can_fuse_int8_affine
from sglang.srt.environ import envs

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")


def test_affine_quant_environment_flag(monkeypatch):
    name = "SGLANG_ENABLE_DIFFUSION_MINIMAX_H3_AFFINE_QUANT_FUSION"
    monkeypatch.delenv(name, raising=False)
    assert not envs.SGLANG_ENABLE_DIFFUSION_MINIMAX_H3_AFFINE_QUANT_FUSION.get()
    monkeypatch.setenv(name, "false")
    assert not envs.SGLANG_ENABLE_DIFFUSION_MINIMAX_H3_AFFINE_QUANT_FUSION.get()
    monkeypatch.setenv(name, "1")
    assert envs.SGLANG_ENABLE_DIFFUSION_MINIMAX_H3_AFFINE_QUANT_FUSION.get()


def test_affine_quant_model_dispatch_guards(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_DIFFUSION_MINIMAX_H3_AFFINE_QUANT_FUSION", "true")
    layer = SimpleNamespace(
        tp_size=1,
        bias=None,
        gather_output=False,
        quant_method=MiniMaxH3Int8LinearMethod(fused=True),
    )
    target = "sglang.multimodal_gen.runtime.layers.quantization.minimax_h3_quant.supports_int8_affine_quant"
    with patch(target, return_value=True), torch.inference_mode():
        assert _can_fuse_int8_affine(layer, None, None, None, None)
        for attr, value in [
            ("tp_size", 2),
            ("bias", object()),
            ("gather_output", True),
        ]:
            old = getattr(layer, attr)
            setattr(layer, attr, value)
            assert not _can_fuse_int8_affine(layer, None, None, None, None)
            setattr(layer, attr, old)
        layer.quant_method.fused = False
        assert not _can_fuse_int8_affine(layer, None, None, None, None)
        layer.quant_method = object()
        assert not _can_fuse_int8_affine(layer, None, None, None, None)
        layer.quant_method = MiniMaxH3Int8LinearMethod(fused=True)
        with patch("torch.compiler.is_compiling", return_value=True):
            assert not _can_fuse_int8_affine(layer, None, None, None, None)
        monkeypatch.setenv(
            "SGLANG_ENABLE_DIFFUSION_MINIMAX_H3_AFFINE_QUANT_FUSION", "false"
        )
        assert not _can_fuse_int8_affine(layer, None, None, None, None)
    monkeypatch.setenv("SGLANG_ENABLE_DIFFUSION_MINIMAX_H3_AFFINE_QUANT_FUSION", "true")
    with torch.enable_grad():
        assert not _can_fuse_int8_affine(layer, None, None, None, None)


def _inputs(m, k, index_dtype=torch.int64, strided_params=False):
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    shift = torch.randn((4, k), device=x.device, dtype=torch.float32) * 0.1
    scale = torch.randn_like(shift) * 0.1
    if strided_params:
        shift_storage = torch.empty((4, k + 16), device=x.device)
        scale_storage = torch.empty((4, k + 32), device=x.device)
        shift_storage[:, :k].copy_(shift)
        scale_storage[:, :k].copy_(scale)
        shift, scale = shift_storage[:, :k], scale_storage[:, :k]
    indices = torch.arange(m, device=x.device, dtype=index_dtype) % 4
    return x, shift, scale, indices


@requires_gpu
@pytest.mark.parametrize(
    "m,k", [(1, 32), (17, 112), (65, 256), (17, 8208), (33, 14336), (17, 16384)]
)
@pytest.mark.parametrize(
    "index_dtype,strided", [(torch.int32, False), (torch.int64, True)]
)
def test_affine_quant_matches_bf16_rounded_reference(m, k, index_dtype, strided):
    torch.manual_seed(19)
    x, shift, scale, indices = _inputs(m, k, index_dtype, strided)
    x_before, shift_before, scale_before = x.clone(), shift.clone(), scale.clone()
    modulated = (
        x.float() * (1.0 + scale.index_select(0, indices))
        + shift.index_select(0, indices)
    ).to(torch.bfloat16)
    expected_q, expected_s = quant_rows(modulated)
    q, sx = quant_affine_rows(x, shift, scale, indices)
    assert torch.equal(q, expected_q)
    assert torch.equal(sx, expected_s)
    assert torch.equal(x, x_before)
    assert torch.equal(shift, shift_before)
    assert torch.equal(scale, scale_before)


@requires_gpu
@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 24, 32),
        (17, 80, 112),
        (65, 384, 256),
        (33, 24, 8208),
        (17, 24, 14336),
        (33, 24, 16384),
    ],
)
@pytest.mark.parametrize("has_bias", [False, True])
def test_affine_linear_uses_existing_gemm_with_exact_chunked_output(m, n, k, has_bias):
    torch.manual_seed(23)
    x, shift, scale, indices = _inputs(m, k)
    weight = torch.randint(-127, 128, (n, k), device=x.device, dtype=torch.int8)
    sw = torch.rand((n, 1), device=x.device, dtype=torch.float32) * 0.01
    bias = torch.randn(n, device=x.device, dtype=torch.bfloat16) if has_bias else None
    with torch.inference_mode():
        expected = linear(affine(x, shift, scale, indices), weight, sw, bias, chunk=16)
        actual = linear(
            x, weight, sw, bias, chunk=16, input_affine=(shift, scale, indices)
        )
    assert torch.equal(actual, expected)


@requires_gpu
@pytest.mark.parametrize(
    "case",
    [
        "large_k",
        "shift_dtype",
        "indices_dtype",
        "short_indices",
        "noncontiguous",
        "empty_params",
    ],
)
def test_affine_quant_rejects_unsupported_metadata(case):
    args = list(_inputs(17, 16400 if case == "large_k" else 32))
    if case == "shift_dtype":
        args[1] = args[1].to(torch.bfloat16)
    elif case == "indices_dtype":
        args[3] = args[3].float()
    elif case == "short_indices":
        args[3] = args[3][:-1]
    elif case == "noncontiguous":
        args[0] = args[0][:, ::2]
    elif case == "empty_params":
        args[1], args[2] = args[1][:0], args[2][:0]
    assert not supports_int8_affine_quant(*args)
    with pytest.raises(ValueError, match="Unsupported indexed affine"):
        quant_affine_rows(*args)


@requires_gpu
@pytest.mark.parametrize("m", [1, 17])
def test_affine_quant_zero_rows_and_padding(m):
    x, shift, scale, indices = _inputs(m, 48)
    x.zero_()
    shift.zero_()
    scale.zero_()
    q, sx = quant_affine_rows(x, shift, scale, indices)
    expected_q, expected_s = quant_rows(x)
    assert torch.equal(q, expected_q)
    assert torch.equal(sx, expected_s)


@requires_gpu
def test_affine_quant_cuda_graph_replay():
    args = _inputs(17, 112)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            quant_affine_rows(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_q, actual_s = quant_affine_rows(*args)
    args[0].add_(0.25)
    graph.replay()
    expected_q, expected_s = quant_rows(affine(*args))
    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_s, expected_s)


@requires_gpu
@pytest.mark.parametrize("rows", [17, 65])
def test_int8_dit_block_affine_fusion_is_exact(monkeypatch, rows):
    from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
        MiniMaxH3DiTArchConfig,
    )
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl
    from sglang.multimodal_gen.runtime.layers.quantization.minimax_h3_quant import (
        MiniMaxH3Int8Config,
    )
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3DiTBlock
    from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
        ensure_distributed_env_defaults,
    )

    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    monkeypatch.setenv("SGLANG_MINIMAX_H3_QUANT_OPTIMIZE", "true")
    arch = MiniMaxH3DiTArchConfig(
        hidden_size=128,
        num_attention_heads=2,
        attention_head_dim=64,
        ffn_hidden_size=256,
        time_embed_dim=8,
        adaln_curve_grid=1025,
        adaln_out_features=18 * 128,
    )
    config = MiniMaxH3Int8Config(
        targets={
            f"blocks.0.{name}"
            for name in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
        },
        compressed_adaln=True,
    )
    torch.manual_seed(31)
    with torch.device("cuda"):
        block = MiniMaxH3DiTBlock(arch, config, prefix="blocks.0")
    with torch.no_grad():
        for name, value in block.named_parameters():
            if value.dtype == torch.int8:
                value.random_(-127, 128)
            elif name.endswith("weight_scale_inv"):
                value.fill_(0.001)
            else:
                value.uniform_(0.1, 0.2)
    block.attn._attention_impl = SDPAImpl(
        num_heads=2, head_size=64, causal=False, softmax_scale=64**-0.5
    )
    x = torch.randn((rows, arch.hidden_size), device="cuda", dtype=torch.bfloat16)
    params = tuple(
        torch.randn((4, arch.hidden_size), device=x.device, dtype=torch.float32) * 0.1
        for _ in range(6)
    )
    kwargs = dict(
        adaln_input=torch.zeros((1, 8), device=x.device),
        combined_indices=torch.arange(rows, device=x.device) % 4,
        rope_cache=None,
        cu_seqlens=torch.tensor([0, rows], device=x.device, dtype=torch.int32),
        cu_seqlens_host=(0, rows),
        max_seqlen=rows,
        adaln_params=params,
    )
    target = "sglang.multimodal_gen.runtime.layers.quantization.minimax_h3_quant.quant_affine_rows"
    with torch.inference_mode():
        monkeypatch.setenv(
            "SGLANG_ENABLE_DIFFUSION_MINIMAX_H3_AFFINE_QUANT_FUSION", "false"
        )
        expected = block(x.clone(), **kwargs)
        monkeypatch.setenv(
            "SGLANG_ENABLE_DIFFUSION_MINIMAX_H3_AFFINE_QUANT_FUSION", "true"
        )
        with patch(target, wraps=quant_affine_rows) as fused:
            actual = block(x.clone(), **kwargs)
        assert fused.call_count == 2  # QKV and FC1 both exercise the fused path.
    assert torch.equal(actual, expected)
