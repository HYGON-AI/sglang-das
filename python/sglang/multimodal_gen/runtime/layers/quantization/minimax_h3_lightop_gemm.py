# SPDX-License-Identifier: Apache-2.0
"""Optional Lightop INT8 GEMM preserving H3's ordered FP32 scaling."""

import importlib
import inspect
from functools import lru_cache

import torch

from sglang.srt.environ import envs


@lru_cache(maxsize=1)
def _ordered_gemm():
    # Older wheels have the same base version but lack ordered scaling. Never
    # retry them with the legacy scale-product semantics.
    try:
        lightop = importlib.import_module("lightop")
    except ImportError:
        return None
    gemm = getattr(lightop, "gemm_w8a8_smooth", None)
    if gemm is None:
        return None
    try:
        parameter = inspect.signature(gemm).parameters.get("input_scale_first")
    except (TypeError, ValueError):
        return None
    if parameter is None or parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
        return None
    return gemm


def lightop_gemm_enabled():
    """Check opt-in and API capability before allocating whole-input buffers."""
    return (
        envs.SGLANG_ENABLE_DIFFUSION_MINIMAX_H3_LIGHTOP_GEMM.get()
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and torch.version.hip is not None
        and _ordered_gemm() is not None
    )


def try_lightop_gemm(qx, qw, sx, sw, bias=None):
    """Return BF16 output or None; Lightop owns architecture/alignment checks.

    No model dimensions, request lengths or tuning tables participate in
    dispatch. Runtime kernel errors propagate rather than silently falling back.
    """
    if (
        not envs.SGLANG_ENABLE_DIFFUSION_MINIMAX_H3_LIGHTOP_GEMM.get()
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or torch.version.hip is None
        or bias is not None
        or qx.ndim != 2
        or qw.ndim != 2
        or qx.dtype != torch.int8
        or qw.dtype != torch.int8
        or qw.shape[1] != qx.shape[1]
        or sx.dtype != torch.float32
        or sw.dtype != torch.float32
        or sx.shape != (qx.shape[0], 1)
        or sw.numel() != qw.shape[0]
        or not all(
            t.is_cuda and t.device == qx.device and t.is_contiguous()
            for t in (qx, qw, sx, sw)
        )
    ):
        return None
    gemm = _ordered_gemm()
    if gemm is None:
        return None
    supported, output = gemm(
        qx,
        qw.t(),
        sx,
        sw.reshape(-1, 1),
        None,
        torch.bfloat16,
        input_scale_first=True,
    )
    if not supported:
        return None
    if (
        output.dtype != torch.bfloat16
        or output.device != qx.device
        or output.shape != (qx.shape[0], qw.shape[0])
        or not output.is_contiguous()
    ):
        raise RuntimeError("Lightop returned an invalid MiniMax-H3 INT8 GEMM output")
    return output
