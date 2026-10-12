"""K3 MLA Q concat/cast and direct KV cache store for HCU."""

from __future__ import annotations

import ctypes
import functools
from pathlib import Path

import torch


@functools.cache
def _module():
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("csrc") / "mla_post_hip.cu"
    binary = load(
        name="sglang_k3_mla_post_hip",
        sources=[str(source)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        is_python_module=False,
    )
    module = ctypes.CDLL(binary)
    module.k3_mla_post.argtypes = (
        [ctypes.c_void_p] * 7
        + [ctypes.c_int] * 2
        + [ctypes.c_int64] * 8
        + [ctypes.c_int] * 2
        + [ctypes.c_void_p, ctypes.c_int]
    )
    module.k3_mla_post.restype = ctypes.c_int
    module.k3_mla_error.argtypes = [ctypes.c_int]
    module.k3_mla_error.restype = ctypes.c_char_p
    return module


def fused_mla_post(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    cache: torch.Tensor,
    loc: torch.Tensor,
    *,
    threads: int = 256,
) -> torch.Tensor:
    """Return contiguous Q in cache dtype; store valid KV rows directly.

    Slot 0 is reserved for padding and is not written. This kernel uses
    the plain 576-element MLA cache, not DSA's structured/scaled cache format.
    """
    dtype_codes = {torch.bfloat16: 0, torch.float8_e4m3fn: 1, torch.float8_e4m3fnuz: 2}
    if cache.dtype not in dtype_codes:
        raise ValueError("K3 MLA post requires BF16/E4M3 plain KV cache")
    if q_nope.ndim != 3 or q_nope.shape[-1] != 512:
        raise ValueError("q_nope must be [tokens, heads, 512]")
    tokens, heads, _ = q_nope.shape
    if q_pe.shape != (tokens, heads, 64):
        raise ValueError("q_pe must match Q tokens/heads and have width 64")
    if k_nope.shape != (tokens, 1, 512) or k_pe.shape != (tokens, 1, 64):
        raise ValueError("K3 KV inputs must be [tokens, 1, 512/64]")
    inputs = (q_nope, q_pe, k_nope, k_pe)
    if any(x.dtype != torch.bfloat16 or x.stride(-1) != 1 for x in inputs):
        raise ValueError("K3 MLA post requires BF16 inputs with unit feature stride")
    if not q_nope.is_cuda or any(
        x.device != q_nope.device for x in (*inputs, cache, loc)
    ):
        raise ValueError("K3 MLA post tensors must share one HCU device")
    if (
        cache.ndim != 3
        or cache.shape[1:] != (1, 576)
        or cache.stride(-1) != 1
        or cache.stride(0) % 4 != 0
        or cache.data_ptr() % (4 * cache.element_size()) != 0
    ):
        raise ValueError("K3 MLA post requires plain [slots, 1, 576] cache")
    if loc.dtype != torch.int64 or loc.ndim != 1 or loc.numel() != tokens:
        raise ValueError("cache locations must be INT64 [tokens]")
    if threads not in (64, 128, 256):
        raise ValueError("threads must be 64, 128 or 256")
    if tokens == 0:
        return torch.empty((0, heads, 576), device=q_nope.device, dtype=cache.dtype)
    with torch.cuda.device(q_nope.device):
        q_out = torch.empty(
            (tokens, heads, 576), device=q_nope.device, dtype=cache.dtype
        )
        module = _module()
        error = module.k3_mla_post(
            *(x.data_ptr() for x in inputs),
            q_out.data_ptr(),
            cache.data_ptr(),
            loc.data_ptr(),
            tokens,
            heads,
            q_nope.stride(0),
            q_nope.stride(1),
            q_pe.stride(0),
            q_pe.stride(1),
            k_nope.stride(0),
            k_pe.stride(0),
            cache.stride(0),
            cache.shape[0],
            loc.stride(0),
            dtype_codes[cache.dtype],
            torch.cuda.current_stream(q_nope.device).cuda_stream,
            threads,
        )
        if error:
            raise RuntimeError(
                "K3 MLA HIP launch failed: " + module.k3_mla_error(error).decode()
            )
    return q_out
