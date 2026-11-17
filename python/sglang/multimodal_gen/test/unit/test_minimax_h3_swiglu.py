# SPDX-License-Identifier: Apache-2.0
"""BF16-native SwiGLU parity, stream/capture lifetime and fallbacks."""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.layers import minimax_h3_swiglu as swiglu
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import _silu_mul

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")


@pytest.fixture(autouse=True)
def enable_swiglu_fusion(monkeypatch):
    # Numerical tests explicitly opt in; the environment test removes it.
    monkeypatch.setenv("SGLANG_DIFFUSION_ENABLE_MINIMAX_H3_SWIGLU_FUSION", "true")


def _reference(x):
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


def _assert_same_bits(actual, expected):
    # NaNs have no stable payload contract across device kernels.
    assert torch.equal(actual.isnan(), expected.isnan())
    mask = ~expected.isnan()
    assert torch.equal(actual.view(torch.int16)[mask], expected.view(torch.int16)[mask])


def test_swiglu_environment_flag(monkeypatch):
    name = "SGLANG_DIFFUSION_ENABLE_MINIMAX_H3_SWIGLU_FUSION"
    monkeypatch.delenv(name, raising=False)
    assert not envs.SGLANG_DIFFUSION_ENABLE_MINIMAX_H3_SWIGLU_FUSION
    monkeypatch.setenv(name, "false")
    assert not envs.SGLANG_DIFFUSION_ENABLE_MINIMAX_H3_SWIGLU_FUSION
    monkeypatch.setenv(name, "1")
    assert envs.SGLANG_DIFFUSION_ENABLE_MINIMAX_H3_SWIGLU_FUSION


def test_cpu_fallback_preserves_autograd():
    x = torch.randn((3, 14), requires_grad=True)
    assert swiglu.try_fused_swiglu(x) is None
    actual = _silu_mul(x, reuse_input=False)
    expected = _reference(x)
    assert torch.equal(actual, expected)
    actual_grad = torch.autograd.grad(actual.sum(), x)[0]
    expected_grad = torch.autograd.grad(expected.sum(), x)[0]
    assert torch.equal(actual_grad, expected_grad)


@requires_gpu
@pytest.mark.parametrize(
    "m,h", [(1, 1), (3, 7), (17, 48), (65, 112), (17, 4097), (33, 14336), (7, 16385)]
)
@pytest.mark.parametrize("scale", [1.0, 16.0])
def test_dynamic_shapes_are_bitwise_exact_and_preserve_input(m, h, scale):
    torch.manual_seed(5)
    x = torch.randn((m, 2 * h), device="cuda", dtype=torch.bfloat16) * scale
    original = x.clone()
    expected = _reference(x)
    with torch.inference_mode():
        actual = swiglu.try_fused_swiglu(x)
    assert actual is not None
    _assert_same_bits(actual, expected)
    assert torch.equal(x, original)
    assert actual.is_contiguous()
    assert actual.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()


@requires_gpu
@pytest.mark.parametrize("up", [1.0, -0.75, 0.0, -0.0, float("inf")])
def test_all_bf16_gate_patterns(up):
    bits = torch.arange(
        1 << torch.finfo(torch.bfloat16).bits, device="cuda", dtype=torch.int32
    ).to(torch.int16)
    gate = bits.view(torch.bfloat16).reshape(-1, 1)
    x = torch.cat((gate, torch.full_like(gate, up)), dim=1)
    expected = _reference(x)
    with torch.inference_mode():
        actual = swiglu.try_fused_swiglu(x)
    _assert_same_bits(actual, expected)


@requires_gpu
@pytest.mark.parametrize(
    "case",
    ["disabled", "grad", "compile", "float32", "noncontiguous", "odd_width", "empty"],
)
def test_unsupported_calls_retain_native_path(monkeypatch, case):
    x = torch.randn((17, 96), device="cuda", dtype=torch.bfloat16)
    if case == "float32":
        x = x.float()
    elif case == "noncontiguous":
        x = x[:, ::2]
    elif case == "odd_width":
        x = x[:, :95].contiguous()
    elif case == "empty":
        x = x[:0]
    elif case == "disabled":
        monkeypatch.setenv("SGLANG_DIFFUSION_ENABLE_MINIMAX_H3_SWIGLU_FUSION", "false")
    with (
        torch.no_grad(),
        patch("torch.compiler.is_compiling", return_value=case == "compile"),
    ):
        if case == "grad":
            with torch.enable_grad():
                assert swiglu.try_fused_swiglu(x) is None
        else:
            assert swiglu.try_fused_swiglu(x) is None


@requires_gpu
def test_existing_inplace_path_is_retained():
    x = torch.randn((3, 32), device="cuda", dtype=torch.bfloat16)
    sentinel = torch.empty((3, 16), device=x.device, dtype=x.dtype)
    with (
        torch.inference_mode(),
        patch.object(
            swiglu,
            "try_fused_swiglu",
            side_effect=AssertionError("Unexpected lookup path"),
        ),
        patch(
            "sglang.multimodal_gen.runtime.models.dits.minimax_h3.silu_and_mul_with_activation_rounding_",
            return_value=sentinel,
        ) as inplace,
    ):
        assert _silu_mul(x, reuse_input=True) is sentinel
        inplace.assert_called_once_with(x)


@requires_gpu
def test_model_helper_uses_lookup_when_enabled(monkeypatch):
    x = torch.randn((17, 96), device="cuda", dtype=torch.bfloat16)
    with (
        torch.inference_mode(),
        patch.object(
            swiglu, "try_fused_swiglu", wraps=swiglu.try_fused_swiglu
        ) as fused,
    ):
        actual = _silu_mul(x, reuse_input=False)
        fused.assert_called_once_with(x)
    _assert_same_bits(actual, _reference(x))
    monkeypatch.setenv("SGLANG_DIFFUSION_ENABLE_MINIMAX_H3_SWIGLU_FUSION", "false")
    with (
        torch.inference_mode(),
        patch.object(
            swiglu,
            "_silu_table",
            side_effect=AssertionError("Unexpected table allocation"),
        ),
    ):
        assert torch.equal(_silu_mul(x, reuse_input=False), _reference(x))


@requires_gpu
def test_cached_table_is_safe_on_another_stream():
    x = torch.randn((65, 96), device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        swiglu.try_fused_swiglu(x)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actual = swiglu.try_fused_swiglu(x)
        torch.cuda.current_stream().wait_stream(stream)
    _assert_same_bits(actual, _reference(x))


@requires_gpu
def test_warm_graph_replay_uses_stable_lookup_table():
    x = torch.randn((17, 96), device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                swiglu.try_fused_swiglu(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = _silu_mul(x, reuse_input=False)
        x.add_(0.25)
        graph.replay()
    _assert_same_bits(actual, _reference(x))


@requires_gpu
def test_cold_graph_capture_retains_native_path(monkeypatch):
    x = torch.randn((17, 96), device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        _reference(x)  # Warm the native operations before capture.
        monkeypatch.delitem(swiglu._tables, x.device.index, raising=False)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = _silu_mul(x, reuse_input=False)
        assert x.device.index not in swiglu._tables
        x.add_(0.25)
        graph.replay()
    _assert_same_bits(actual, _reference(x))


@requires_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two GPUs")
def test_lookup_tables_are_separate_for_each_device():
    with torch.inference_mode():
        for device in (torch.device("cuda:0"), torch.device("cuda:1")):
            x = torch.randn((17, 96), device=device, dtype=torch.bfloat16)
            actual = swiglu.try_fused_swiglu(x)
            _assert_same_bits(actual, _reference(x))
        assert swiglu._tables[0][0].device != swiglu._tables[1][0].device


@requires_gpu
@pytest.mark.parametrize("use_affine", [False, True])
def test_int8_mlp_fused_swiglu_matches_native(monkeypatch, use_affine):
    from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
        MiniMaxH3DiTArchConfig,
    )
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.multimodal_gen.runtime.layers.quantization.minimax_h3_quant import (
        MiniMaxH3Int8Config,
    )
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3MLP
    from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
        ensure_distributed_env_defaults,
    )

    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    monkeypatch.setenv("SGLANG_MINIMAX_H3_QUANT_OPTIMIZE", "true")
    arch = MiniMaxH3DiTArchConfig(hidden_size=64, ffn_hidden_size=128)
    config = MiniMaxH3Int8Config(
        targets={"blocks.0.mlp.fc1", "blocks.0.mlp.fc2"}, compressed_adaln=True
    )
    torch.manual_seed(19)
    with torch.device("cuda"):
        mlp = MiniMaxH3MLP(arch, config, prefix="blocks.0.mlp")
    with torch.no_grad():
        for value in mlp.parameters():
            if value.dtype == torch.int8:
                value.random_(-127, 128)
            else:
                value.fill_(0.001)
    x = torch.randn((33, arch.hidden_size), device="cuda", dtype=torch.bfloat16)
    params = None
    if use_affine:
        params = (
            torch.randn((4, arch.hidden_size), device=x.device) * 0.1,
            torch.randn((4, arch.hidden_size), device=x.device) * 0.1,
            torch.arange(x.shape[0], device=x.device) % 4,
        )
    with torch.inference_mode():
        monkeypatch.setenv("SGLANG_DIFFUSION_ENABLE_MINIMAX_H3_SWIGLU_FUSION", "false")
        expected = mlp(x, input_affine=params)
        monkeypatch.setenv("SGLANG_DIFFUSION_ENABLE_MINIMAX_H3_SWIGLU_FUSION", "true")
        with patch.object(
            swiglu, "try_fused_swiglu", wraps=swiglu.try_fused_swiglu
        ) as fused:
            actual = mlp(x, input_affine=params)
            assert fused.call_count == 1
    _assert_same_bits(actual, expected)
