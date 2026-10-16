"""Small-token K3 shared-expert W8A8 GEMM for serial communication insertion.

The native projection remains the fallback. This kernel changes neither the
INT8 dot product nor the channelwise scaling order used by strategy 3.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_hcu

_is_hcu = is_hcu()


@triton.jit
def _k3_shared_w8a8(
    A,
    W,
    AS,
    WS,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SA0: tl.constexpr,
    SA1: tl.constexpr,
    SW0: tl.constexpr,
    SW1: tl.constexpr,
    PROGRAMS: tl.constexpr,
):
    BM: tl.constexpr = 16
    BN: tl.constexpr = 64
    BK: tl.constexpr = 256
    nt = tl.cdiv(N, BN)
    for tile in range(tl.program_id(0), tl.cdiv(M, BM) * nt, PROGRAMS):
        rows = tile // nt * BM + tl.arange(0, BM)
        cols = tile % nt * BN + tl.arange(0, BN)
        ks = tl.arange(0, BK)
        acc = tl.full((BN, BM), 0, tl.int32)
        for start in range(tl.cdiv(K, BK)):
            k = start * BK + ks
            w = tl.load(
                W + cols[:, None] * SW0 + k[None, :] * SW1,
                (cols[:, None] < N) & (k[None, :] < K),
                0,
            )
            x = tl.load(
                A + rows[None, :] * SA0 + k[:, None] * SA1,
                (rows[None, :] < M) & (k[:, None] < K),
                0,
            )
            acc = tl.dot(w, x, acc, out_dtype=tl.int32)
        sa = tl.load(AS + rows, rows < M, 0)
        sw = tl.load(WS + cols, cols < N, 0)
        result = acc.to(tl.float32) * (sa[None, :] * sw[:, None])
        tl.store(
            Y + rows[None, :] * N + cols[:, None],
            result,
            (rows[None, :] < M) & (cols[:, None] < N),
        )


@lru_cache(maxsize=None)
def _supported_device(device_index):
    if not _is_hcu or not torch.version.hip:
        return False
    arch = getattr(torch.cuda.get_device_properties(device_index), "gcnArchName", "")
    return arch.split(":", 1)[0] == "gfx938"


def _eligible(linear, x):
    weight = getattr(linear, "weight", None)
    scale = getattr(linear, "weight_scale", None)
    quant_method = getattr(linear, "quant_method", None)
    if not (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and 0 < x.shape[0] <= 16
        and not x.requires_grad
        and x.stride(1) == 1
        and getattr(linear, "tp_size", None) == 1
        and getattr(linear, "bias", None) is None
        and not getattr(linear, "gather_output", False)
        and getattr(quant_method, "w8a8_strategy", None) == 3
        and weight is not None
        and weight.dtype == torch.int8
        and tuple(weight.shape) in ((12288, 7168), (7168, 6144))
        and weight.shape[1] == x.shape[1]
        and weight.device == x.device
        and weight.is_contiguous()
        and scale is not None
        and scale.dtype == torch.float32
        and scale.numel() == weight.shape[0]
        and scale.device == x.device
        and scale.is_contiguous()
        and not scale.requires_grad
        and _supported_device(x.device.index)
    ):
        return False
    # Do not bypass smoothing, zero-points, or another method's weight layout.
    from sglang.srt.layers.quantization.slimquant_w4a8 import (
        SlimQuantW4A8Int8LinearMethod,
    )

    return type(quant_method) is SlimQuantW4A8Int8LinearMethod


def project(linear, x, input_quant_args=None):
    """Return the normal linear result tuple, with an exact native fallback."""
    quant_supported = input_quant_args is None or (
        isinstance(input_quant_args, (tuple, list))
        and len(input_quant_args) == 2
        and isinstance(input_quant_args[0], torch.Tensor)
        and isinstance(input_quant_args[1], torch.Tensor)
        and input_quant_args[0].device == x.device
        and input_quant_args[1].device == x.device
        and input_quant_args[0].is_contiguous()
        and not input_quant_args[1].requires_grad
        and input_quant_args[0].dtype == torch.int8
        and input_quant_args[0].shape == x.shape
        and input_quant_args[1].dtype == torch.float32
        and input_quant_args[1].shape == (x.shape[0], 1)
        and input_quant_args[1].is_contiguous()
    )
    if not quant_supported or not _eligible(linear, x):
        if input_quant_args is None:
            return linear(x)
        return linear(x, input_quant_args=input_quant_args)
    if input_quant_args is None:
        from lightop.quant import per_token_quant_int8

        input_quant_args = per_token_quant_int8(x)
    q, scale = input_quant_args
    m, k = q.shape
    n = linear.weight.shape[0]
    assert q.dtype == torch.int8 and scale.shape == (m, 1)
    assert scale.is_contiguous()
    out = torch.empty((m, n), device=q.device, dtype=torch.bfloat16)
    grid = min(192, triton.cdiv(n, 64))
    _k3_shared_w8a8[(grid,)](
        q,
        linear.weight,
        scale,
        linear.weight_scale,
        out,
        m,
        n,
        k,
        *q.stride(),
        *linear.weight.stride(),
        grid,
        num_warps=4,
        num_stages=2,
        matrix_instr_nonkdim=16,
        kpack=2,
        mmac_layout_force=3,
    )
    return out, None
