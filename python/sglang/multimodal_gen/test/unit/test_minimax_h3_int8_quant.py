# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 row-quantization boundaries and exact INT8 Linear parity."""

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.quantization.minimax_h3_quant import (
    MAX_ROW_QUANT_BLOCK_SIZE,
    MAX_ROW_QUANT_LINEAR_OFFSET,
    _row_quant_block_size,
    linear,
    quant_rows,
)

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")


def _metadata(m, k, *, row_stride=None, col_stride=1, dtype=torch.bfloat16):
    strides = (k if row_stride is None else row_stride, col_stride)
    return SimpleNamespace(
        ndim=2,
        layout=torch.strided,
        dtype=dtype,
        is_cuda=True,
        shape=(m, k),
        stride=lambda dim: strides[dim],
    )


@pytest.mark.parametrize(
    "k,expected",
    [(8192, 8192), (8208, 16384), (14336, 16384), (16384, 16384), (16400, None)],
)
def test_row_quant_reduction_boundary(k, expected):
    assert _row_quant_block_size(_metadata(17, k), "int8") == expected


@pytest.mark.parametrize(
    "m,k,row_stride,col_stride",
    [(0, 32, 32, 1), (17, 0, 0, 1), (17, 32, -32, 1), (17, 32, 64, 2)],
)
def test_row_quant_rejects_unsupported_metadata(m, k, row_stride, col_stride):
    assert (
        _row_quant_block_size(
            _metadata(m, k, row_stride=row_stride, col_stride=col_stride), "int8"
        )
        is None
    )


def test_row_quant_checks_relative_input_and_padded_output_offsets():
    assert (
        _row_quant_block_size(
            _metadata(17, 32, row_stride=MAX_ROW_QUANT_LINEAR_OFFSET), "int8"
        )
        is None
    )
    rows = MAX_ROW_QUANT_LINEAR_OFFSET // MAX_ROW_QUANT_BLOCK_SIZE + 2
    assert (
        _row_quant_block_size(
            _metadata(rows, MAX_ROW_QUANT_BLOCK_SIZE, row_stride=0), "int8"
        )
        is None
    )
    assert _row_quant_block_size(_metadata(17, 32, row_stride=0), "int8") == 32
    assert _row_quant_block_size(_metadata(17, 32, dtype=torch.float16), "int8") is None
    assert _row_quant_block_size(_metadata(17, 32, dtype=torch.float16), "fp8") == 32


def _reference_quant(x):
    values = x.float()
    scales = values.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127.0
    quant = (values / scales).round().clamp(-127, 127).to(torch.int8)
    return quant, scales


@requires_gpu
@pytest.mark.parametrize(
    "m,k", [(1, 32), (17, 112), (33, 8192), (17, 8208), (33, 14336), (17, 16384)]
)
@pytest.mark.parametrize("layout", ["contiguous", "row_stride", "broadcast"])
def test_row_quant_exact_parity(m, k, layout):
    torch.manual_seed(7)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    if layout == "row_stride":
        storage = torch.empty((m, k + 16), device="cuda", dtype=x.dtype)
        storage[:, :k].copy_(x)
        x = storage[:, :k]
    elif layout == "broadcast":
        x = x[:1].expand(m, k)
    expected_q, expected_s = _reference_quant(x)
    q, scales = quant_rows(x)
    assert torch.equal(q[:m], expected_q)
    assert torch.equal(scales[:m], expected_s)
    assert not torch.count_nonzero(q[m:])
    if q.shape[0] > m:
        _, zero_scale = _reference_quant(
            torch.zeros((1, k), device=x.device, dtype=x.dtype)
        )
        assert torch.equal(scales[m:], zero_scale.expand_as(scales[m:]))


@requires_gpu
@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 24, 32),
        (17, 80, 112),
        (33, 24, 8208),
        (17, 24, 14336),
        (33, 24, 16384),
        (17, 24, 16400),
    ],
)
@pytest.mark.parametrize("has_bias", [False, True])
def test_int8_linear_exact_parity_and_large_k_fallback(m, n, k, has_bias):
    torch.manual_seed(11)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (n, k), device=x.device, dtype=torch.int8)
    scale = torch.rand((n, 1), device=x.device, dtype=torch.float32) * 0.01
    bias = torch.randn(n, device=x.device, dtype=torch.bfloat16) if has_bias else None
    with torch.inference_mode():
        expected = linear(x, weight, scale, bias, mode="reference", chunk=16)
        actual = linear(x, weight, scale, bias, chunk=16)
    assert torch.equal(actual, expected)
