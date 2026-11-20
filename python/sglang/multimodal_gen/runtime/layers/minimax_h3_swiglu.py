# SPDX-License-Identifier: Apache-2.0
"""BF16-rounded SwiGLU using a device-native SiLU lookup table."""

import threading

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.environ import envs

# Enumerate every BF16 bit pattern, not a sampled approximation of SiLU.
_BF16_PATTERN_COUNT = 1 << (torch.finfo(torch.bfloat16).bits)
# Pointwise launch parameters, independent of model/request dimensions.
_BLOCK_SIZE = 1024
_NUM_WARPS = 4
_tables = {}
_tables_lock = threading.Lock()


def _silu_table(device):
    """Cache one 128-KiB table per device, with cross-stream readiness.

    Initialize during eager warmup. A cold graph capture takes the native
    path rather than publishing a table whose initialization needs replay.
    """
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        with _tables_lock:
            entry = _tables.get(device.index)
            if entry is None:
                if torch.cuda.is_current_stream_capturing():
                    return None
                patterns = torch.arange(
                    _BF16_PATTERN_COUNT, device=device, dtype=torch.int32
                ).to(torch.int16)
                table = F.silu(patterns.view(torch.bfloat16))
                ready = torch.cuda.Event()
                ready.record(stream)
                entry = (table, ready, stream.cuda_stream)
                _tables[device.index] = entry
        table, ready, producer = entry
        if stream.cuda_stream != producer:
            stream.wait_event(ready)
        return table


@triton.jit
def _swiglu_lookup(
    X, TABLE, OUT, N: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr
):
    # int64 offsets keep large packed sequences independent of int32 indexing.
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    row, col = i // H, i % H
    gate = tl.load(X + row * (2 * H) + col, mask, 0)
    up = tl.load(X + row * (2 * H) + H + col, mask, 0).to(tl.float32)
    bits = gate.to(tl.uint16, bitcast=True).to(tl.int32)
    # The table already contains the native BF16 SiLU-output rounding.
    activation = tl.load(TABLE + bits).to(tl.float32)
    # Store into BF16 retains the native product's second rounding.
    tl.store(OUT + i, activation * up, mask)


def try_fused_swiglu(x):
    """Return a fresh BF16 output, or None to retain the caller's native path."""
    if (
        not envs.SGLANG_ENABLE_DIFFUSION_MINIMAX_H3_SWIGLU_FUSION.get()
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.ndim != 2
        or not x.is_contiguous()
        or x.numel() == 0
        or x.shape[-1] % 2
    ):
        return None
    with torch.cuda.device(x.device):
        table = _silu_table(x.device)
        if table is None:
            return None
        out = torch.empty((x.shape[0], x.shape[1] // 2), device=x.device, dtype=x.dtype)
        _swiglu_lookup[(triton.cdiv(out.numel(), _BLOCK_SIZE),)](
            x,
            table,
            out,
            out.numel(),
            out.shape[1],
            _BLOCK_SIZE,
            num_warps=_NUM_WARPS,
            enable_fp_fusion=False,
        )
        return out
