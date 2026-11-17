# SPDX-License-Identifier: Apache-2.0
"""Native BF16-rounded SwiGLU fused with per-row INT8 quantization."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.layers.minimax_h3_swiglu import _silu_table
from sglang.multimodal_gen.runtime.layers.quantization.minimax_h3_lightop_gemm import (
    lightop_gemm_enabled,
    try_lightop_gemm,
)
from sglang.multimodal_gen.runtime.layers.quantization.minimax_h3_quant import (
    MAX_ROW_QUANT_LINEAR_OFFSET,
    MiniMaxH3Int8LinearMethod,
    _row_quant_block_size,
)


def _swiglu_quant_block_size(x):
    if (
        x.ndim != 2
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
        or x.shape[0] < 1
        or x.shape[1] < 2
        or x.shape[1] % 2
        or x.numel() - 1 > MAX_ROW_QUANT_LINEAR_OFFSET
    ):
        return None
    # Reuse the existing row reduction and padded-output indexing limits.
    return _row_quant_block_size(x[:, : x.shape[1] // 2], "int8")


@triton.jit
def _swiglu_quant_rows(
    X, TABLE, Q, SX, M: tl.constexpr, K: tl.constexpr, B: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    mask = (row < M) & (col < K)
    gate = tl.load(X + row * (2 * K) + col, mask, 0)
    up = tl.load(X + row * (2 * K) + K + col, mask, 0).to(tl.float32)
    bits = gate.to(tl.uint16, bitcast=True).to(tl.int32)
    activation = tl.load(TABLE + bits).to(tl.float32)
    # The lookup retains SiLU's BF16 rounding; retain product rounding too,
    # before computing absmax and the existing ties-to-even quantization.
    product = (activation * up).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(product), 0), 1.0e-12)
    sx = amax * (1.0 / 127.0)
    quant = libdevice.nearbyint(tl.div_rn(product, sx))
    quant = tl.minimum(tl.maximum(quant, -127.0), 127.0).to(tl.int8)
    tl.store(Q + row * K + col, quant, col < K)
    tl.store(SX + row, sx)


def quant_swiglu_rows(x):
    """Return padded INT8 rows/scales, or None for unsupported/cold capture.

    Shares the native SiLU table's device, stream and graph-capture lifetime.
    Padding and launch parameters match quant_rows(), independently of H3 sizes.
    """
    block_size = _swiglu_quant_block_size(x)
    if block_size is None:
        return None
    with torch.cuda.device(x.device):
        table = _silu_table(x.device)
        if table is None:
            return None
        m, width = x.shape
        k = width // 2
        # Same _int_mm padding contract as quant_rows(), including small M.
        padded_m = max(32, triton.cdiv(m, 16) * 16)
        q = torch.empty((padded_m, k), device=x.device, dtype=torch.int8)
        sx = torch.empty((padded_m, 1), device=x.device, dtype=torch.float32)
        _swiglu_quant_rows[(padded_m,)](
            x,
            table,
            q,
            sx,
            m,
            k,
            block_size,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return q, sx


def try_swiglu_fc2(layer, x):
    """Bypass activation materialization only for an eligible local INT8 fc2."""
    if (
        not envs.SGLANG_DIFFUSION_ENABLE_MINIMAX_H3_SWIGLU_QUANT_FUSION
        or not envs.SGLANG_DIFFUSION_ENABLE_MINIMAX_H3_LIGHTOP_GEMM_WHOLE
        or not lightop_gemm_enabled()
        # Bypassing RowParallelLinear must preserve its TP/reduction contract.
        or getattr(layer, "tp_size", 0) != 1
        or getattr(layer, "tp_rank", -1) != 0
        or not getattr(layer, "input_is_parallel", False)
        or layer.bias is not None
        or not isinstance(layer.quant_method, MiniMaxH3Int8LinearMethod)
        or not layer.quant_method.fused
        or _swiglu_quant_block_size(x) is None
    ):
        return None
    weight, scale = layer.weight, layer.weight_scale_inv
    if (
        weight.ndim != 2
        or weight.shape[1] != x.shape[1] // 2
        or weight.dtype != torch.int8
        or scale.dtype != torch.float32
        or scale.shape != (weight.shape[0], 1)
        or not all(
            t.is_cuda and t.device == x.device and t.is_contiguous()
            for t in (weight, scale)
        )
    ):
        return None
    quantized = quant_swiglu_rows(x)
    if quantized is None:
        return None
    q, sx = quantized
    result = try_lightop_gemm(q, weight, sx, scale)
    return None if result is None else result[: x.shape[0]]
