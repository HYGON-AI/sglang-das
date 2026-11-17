# SPDX-License-Identifier: Apache-2.0
"""Exact SwiGLU/INT8 fusion, dynamic shapes and Linear bypass contracts."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.layers import minimax_h3_swiglu as swiglu
from sglang.multimodal_gen.runtime.layers.quantization import (
    minimax_h3_swiglu_quant as fused,
)
from sglang.multimodal_gen.runtime.layers.quantization.minimax_h3_quant import (
    MAX_ROW_QUANT_BLOCK_SIZE,
    MiniMaxH3Int8LinearMethod,
    quant_rows,
)

FLAG = "SGLANG_DIFFUSION_ENABLE_MINIMAX_H3_SWIGLU_QUANT_FUSION"
GEMM_FLAG = "SGLANG_DIFFUSION_ENABLE_MINIMAX_H3_LIGHTOP_GEMM"
WHOLE_FLAG = GEMM_FLAG + "_WHOLE"
requires_hcu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None, reason="Requires HCU"
)


def _native(x):
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


def _enable(monkeypatch):
    for name in (FLAG, GEMM_FLAG, WHOLE_FLAG):
        monkeypatch.setenv(name, "true")


def test_default_off_and_cpu_fallback(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert not envs.SGLANG_DIFFUSION_ENABLE_MINIMAX_H3_SWIGLU_QUANT_FUSION
    monkeypatch.setenv(FLAG, "true")
    assert envs.SGLANG_DIFFUSION_ENABLE_MINIMAX_H3_SWIGLU_QUANT_FUSION
    assert fused.quant_swiglu_rows(torch.zeros((3, 32), dtype=torch.bfloat16)) is None


@requires_hcu
@pytest.mark.parametrize(
    "m,k", [(1, 1), (17, 112), (65, 384), (33, 14336), (17, 16384)]
)
@pytest.mark.parametrize("scale", [0.0, 1.0, 16.0])
def test_quantized_values_scales_and_padding_are_exact(m, k, scale):
    torch.manual_seed(59)
    x = torch.randn((m, 2 * k), device="cuda", dtype=torch.bfloat16) * scale
    original = x.clone()
    with torch.inference_mode():
        expected_q, expected_s = quant_rows(_native(x))
        actual_q, actual_s = fused.quant_swiglu_rows(x)
    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_s, expected_s)
    assert torch.equal(x, original)


@requires_hcu
@pytest.mark.parametrize(
    "case", ["dtype", "odd_width", "stride", "empty", "reduction_limit"]
)
def test_unsupported_activation_metadata(case):
    x = torch.zeros((17, 224), device="cuda", dtype=torch.bfloat16)
    if case == "dtype":
        x = x.float()
    elif case == "odd_width":
        x = x[:, :223].contiguous()
    elif case == "stride":
        x = x[:, ::2]
    elif case == "empty":
        x = x[:0]
    else:
        x = torch.empty(
            (1, 2 * (MAX_ROW_QUANT_BLOCK_SIZE + 1)), device=x.device, dtype=x.dtype
        )
    assert fused.quant_swiglu_rows(x) is None


@requires_hcu
def test_cross_stream_table_read_and_warm_graph_replay(monkeypatch):
    x = torch.randn((17, 224), device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        monkeypatch.delitem(swiglu._tables, x.device.index, raising=False)
        producer = torch.cuda.Stream()
        with torch.cuda.stream(producer):
            swiglu._silu_table(x.device)
        # quant_swiglu_rows must acquire table readiness on this stream too.
        q, sx = fused.quant_swiglu_rows(x)
        expected_q, expected_s = quant_rows(_native(x))
        assert torch.equal(q, expected_q) and torch.equal(sx, expected_s)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fused.quant_swiglu_rows(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            q, sx = fused.quant_swiglu_rows(x)
        x.add_(0.25)
        graph.replay()
        expected_q, expected_s = quant_rows(_native(x))
    assert torch.equal(q, expected_q) and torch.equal(sx, expected_s)


@requires_hcu
def test_cold_capture_defers_table_initialization(monkeypatch):
    x = torch.randn((17, 224), device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        quant_rows(_native(x))
        monkeypatch.delitem(swiglu._tables, x.device.index, raising=False)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            assert fused.quant_swiglu_rows(x) is None
            q, sx = quant_rows(_native(x))
        assert x.device.index not in swiglu._tables
        graph.replay()
        expected_q, expected_s = quant_rows(_native(x))
    assert torch.equal(q, expected_q) and torch.equal(sx, expected_s)


@requires_hcu
@pytest.mark.parametrize(
    "case",
    [
        "disabled",
        "whole_disabled",
        "gemm_disabled",
        "grad",
        "compile",
        "tp",
        "rank",
        "input_parallel",
        "bias",
        "method",
        "unfused",
        "weight",
        "scale",
    ],
)
def test_model_dispatch_guards(monkeypatch, case):
    _enable(monkeypatch)
    x = torch.zeros((17, 256), device="cuda", dtype=torch.bfloat16)
    layer = SimpleNamespace(
        tp_size=1,
        tp_rank=0,
        input_is_parallel=True,
        bias=None,
        quant_method=MiniMaxH3Int8LinearMethod(fused=True),
        weight=torch.zeros((256, 128), device=x.device, dtype=torch.int8),
        weight_scale_inv=torch.ones((256, 1), device=x.device),
    )
    changes = {
        "tp": ("tp_size", 2),
        "rank": ("tp_rank", 1),
        "input_parallel": ("input_is_parallel", False),
        "bias": ("bias", object()),
        "method": ("quant_method", object()),
        "unfused": ("quant_method", MiniMaxH3Int8LinearMethod(fused=False)),
        "weight": ("weight", layer.weight[:, :64]),
        "scale": ("weight_scale_inv", layer.weight_scale_inv.view(-1)),
    }
    if case in changes:
        key, value = changes[case]
        setattr(layer, key, value)
    for kind, name in [
        ("disabled", FLAG),
        ("whole_disabled", WHOLE_FLAG),
        ("gemm_disabled", GEMM_FLAG),
    ]:
        if case == kind:
            monkeypatch.setenv(name, "false")
    with (
        torch.set_grad_enabled(case == "grad"),
        patch("torch.compiler.is_compiling", return_value=case == "compile"),
        patch.object(
            fused,
            "quant_swiglu_rows",
            side_effect=AssertionError("Unexpected quantization"),
        ),
    ):
        assert fused.try_swiglu_fc2(layer, x) is None


@requires_hcu
@pytest.mark.parametrize("m", [1, 17, 65])
@pytest.mark.parametrize("use_affine", [False, True])
@pytest.mark.parametrize("hidden,ffn,supported", [(256, 384, True), (80, 112, False)])
def test_int8_mlp_fusion_matches_native(
    monkeypatch, m, use_affine, hidden, ffn, supported
):
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

    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx936":
        pytest.skip("Ordered Lightop ASM requires gfx936")
    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    monkeypatch.setenv("SGLANG_MINIMAX_H3_QUANT_OPTIMIZE", "true")
    arch = MiniMaxH3DiTArchConfig(hidden_size=hidden, ffn_hidden_size=ffn)
    config = MiniMaxH3Int8Config(
        targets={"blocks.0.mlp.fc1", "blocks.0.mlp.fc2"}, compressed_adaln=True
    )
    with torch.device("cuda"):
        mlp = MiniMaxH3MLP(arch, config, prefix="blocks.0.mlp")
    with torch.no_grad():
        for value in mlp.parameters():
            if value.dtype == torch.int8:
                value.random_(-127, 128)
            else:
                value.fill_(0.001)
    x = torch.randn((m, arch.hidden_size), device="cuda", dtype=torch.bfloat16)
    params = None
    if use_affine:
        params = (
            torch.randn((2, arch.hidden_size), device=x.device) * 0.1,
            torch.randn((2, arch.hidden_size), device=x.device) * 0.1,
            torch.arange(m, device=x.device) % 2,
        )
    with torch.inference_mode():
        for flag in (FLAG, GEMM_FLAG, WHOLE_FLAG):
            monkeypatch.setenv(flag, "false")
        expected = mlp(x, input_affine=params)
        _enable(monkeypatch)
        with patch.object(fused, "try_swiglu_fc2", wraps=fused.try_swiglu_fc2) as calls:
            actual = mlp(x, input_affine=params)
        assert calls.call_count == 1
        assert (
            fused.try_swiglu_fc2(
                mlp.fc2,
                torch.zeros(
                    (m, 2 * arch.ffn_hidden_size), device=x.device, dtype=x.dtype
                ),
            )
            is not None
        ) == supported
    assert torch.equal(actual, expected)
