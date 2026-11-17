# SPDX-License-Identifier: Apache-2.0
"""Ordered Lightop scaling, dynamic Linear inputs and optional-dependency fallback."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.layers.quantization import (
    minimax_h3_lightop_gemm as adapter,
)
from sglang.multimodal_gen.runtime.layers.quantization.minimax_h3_quant import (
    linear,
    quant_rows,
)

FLAG = "SGLANG_DIFFUSION_ENABLE_MINIMAX_H3_LIGHTOP_GEMM"
requires_hcu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None, reason="Requires HCU"
)


@pytest.fixture(autouse=True)
def reset_capability_cache():
    adapter._ordered_gemm.cache_clear()
    yield
    adapter._ordered_gemm.cache_clear()


def test_environment_flag(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert not envs.SGLANG_DIFFUSION_ENABLE_MINIMAX_H3_LIGHTOP_GEMM
    monkeypatch.setenv(FLAG, "true")
    assert envs.SGLANG_DIFFUSION_ENABLE_MINIMAX_H3_LIGHTOP_GEMM


def test_missing_and_legacy_lightop():
    with patch.object(adapter.importlib, "import_module", side_effect=ImportError):
        assert adapter._ordered_gemm() is None
    adapter._ordered_gemm.cache_clear()
    legacy = lambda a, b, sa, sb, bias, dtype: (False, None)
    with patch.object(
        adapter.importlib,
        "import_module",
        return_value=SimpleNamespace(gemm_w8a8_smooth=legacy),
    ):
        assert adapter._ordered_gemm() is None


def _inputs(m=33, n=256, k=128):
    torch.manual_seed(47)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    w = torch.randint(-127, 128, (n, k), device=x.device, dtype=torch.int8)
    sw = torch.rand((n, 1), device=x.device) * 0.01
    return x, w, sw


@requires_hcu
@pytest.mark.parametrize("m,n,k", [(1, 256, 128), (17, 768, 384), (65, 512, 256)])
@pytest.mark.parametrize("affine", [False, True])
def test_real_lightop_matches_linear_reference(monkeypatch, m, n, k, affine):
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx936":
        pytest.skip("Ordered Lightop ASM requires gfx936")
    monkeypatch.setenv(FLAG, "true")
    x, w, sw = _inputs(m, n, k)
    params = None
    reference_x = x
    if affine:
        shift = torch.randn((2, k), device=x.device)
        scale = torch.randn_like(shift) * 0.1
        indices = torch.arange(m, device=x.device) % 2
        params = (shift, scale, indices)
        reference_x = (x.float() * (1.0 + scale[indices]) + shift[indices]).to(x.dtype)
    with torch.inference_mode():
        q, sx = quant_rows(reference_x)
        assert adapter.try_lightop_gemm(q, w, sx, sw) is not None
        expected = linear(reference_x, w, sw, mode="reference", chunk=16)
        with patch.object(
            adapter, "try_lightop_gemm", wraps=adapter.try_lightop_gemm
        ) as calls:
            actual = linear(x, w, sw, chunk=32, input_affine=params)
        assert calls.call_count > 0
    assert torch.equal(actual, expected)


@requires_hcu
@pytest.mark.parametrize(
    "case", ["disabled", "grad", "compile", "bias", "unsupported", "legacy"]
)
def test_fallback_preserves_reference(monkeypatch, case):
    monkeypatch.setenv(FLAG, "false" if case == "disabled" else "true")
    x, w, sw = _inputs()
    bias = (
        torch.randn(w.shape[0], device=x.device, dtype=x.dtype)
        if case == "bias"
        else None
    )
    with torch.inference_mode():
        expected = linear(x, w, sw, bias, mode="reference", chunk=32)
    gemm = lambda *args, **kwargs: (False, None)
    with (
        patch.object(
            adapter, "_ordered_gemm", return_value=None if case == "legacy" else gemm
        ) as capability,
        patch("torch.compiler.is_compiling", return_value=case == "compile"),
        torch.set_grad_enabled(case == "grad"),
    ):
        actual = linear(x, w, sw, bias, chunk=32)
    if case in ("disabled", "grad", "compile", "bias"):
        capability.assert_not_called()
    assert torch.equal(actual, expected)


@requires_hcu
def test_ordered_keyword_and_runtime_errors(monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    x, w, sw = _inputs()
    q, sx = quant_rows(x)
    with torch.inference_mode(), patch.object(adapter, "_ordered_gemm") as capability:
        gemm = capability.return_value
        gemm.return_value = (False, None)
        assert adapter.try_lightop_gemm(q, w, sx, sw) is None
        assert gemm.call_args.kwargs == {"input_scale_first": True}
        gemm.side_effect = RuntimeError("kernel failure")
        with pytest.raises(RuntimeError, match="kernel failure"):
            adapter.try_lightop_gemm(q, w, sx, sw)


@requires_hcu
def test_real_ordered_bf16_rounding_boundary(monkeypatch):
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx936":
        pytest.skip("Ordered Lightop ASM requires gfx936")
    monkeypatch.setenv(FLAG, "true")
    # This exact FP32 scale pair makes an accumulator of 137 cross a BF16 tie
    # when the scales are folded. Values are regression data, not dispatch sizes.
    q = torch.ones((257, 128), device="cuda", dtype=torch.int8)
    w = torch.zeros((256, 128), device=q.device, dtype=q.dtype)
    w[:, 0], w[:, 1] = 127, 10
    sx = torch.full((len(q), 1), 0.06446823477745056, device=q.device)
    sw = torch.full((len(w), 1), 0.07585038244724274, device=q.device)
    acc = torch._int_mm(q[:256], w.t()).float()
    expected = ((acc * sx[:256]) * sw.t()).bfloat16()
    folded = (acc * (sx[:256] * sw.t())).bfloat16()
    assert torch.all(expected != folded)
    with torch.inference_mode():
        actual = adapter.try_lightop_gemm(q, w, sx, sw)
    assert actual is not None
    assert torch.equal(actual[:256], expected)


@requires_hcu
def test_real_unsupported_alignment_falls_back(monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    x, w, sw = _inputs(17, 80, 112)
    with torch.inference_mode():
        q, sx = quant_rows(x)
        assert adapter.try_lightop_gemm(q, w, sx, sw) is None
        assert torch.equal(linear(x, w, sw), linear(x, w, sw, mode="reference"))
