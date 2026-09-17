"""JIT loader for the fused C4 indexer Q INT8 kernel.

The kernel combines trailing RoPE, the 128-point Hadamard transform, INT8
quantization, and Q-scale folding into one HCU launch.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from sglang.srt.utils import is_hcu


@functools.cache
def _module():
    import torch.utils.cpp_extension

    source_path = Path(__file__).with_name("csrc") / "q_indexer_int8.cu"
    return torch.utils.cpp_extension.load_inline(
        name="sgl_q_indexer_int8_jit",
        cpp_sources="",
        cuda_sources=source_path.read_text(),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-w", "-DUSE_ROCM=1"],
        verbose=False,
    )


def fused_q_indexer_rope_hadamard_quant_int8(
    q_input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one-shot INT8 C4 Q and weights with Q scale folded in."""
    if not q_input.is_cuda or not is_hcu():
        raise ValueError("fused INT8 C4 Q requires an HCU tensor")
    if q_input.dtype != torch.bfloat16:
        raise ValueError(f"fused INT8 C4 Q requires BF16 input, got {q_input.dtype}")
    if q_input.ndim != 3 or q_input.shape[-1] != 128:
        raise ValueError(
            "fused INT8 C4 Q requires q_input shaped (batch, heads, 128)"
        )
    if weight.dtype != torch.bfloat16:
        raise ValueError(f"fused INT8 C4 Q requires BF16 weight, got {weight.dtype}")
    if weight.numel() != q_input.numel() // q_input.shape[-1]:
        raise ValueError("weight must contain one value per query head")
    if not weight.is_contiguous():
        raise ValueError("fused INT8 C4 Q requires contiguous weight")
    if freqs_cis.device != q_input.device or weight.device != q_input.device:
        raise ValueError("Q, weight, and RoPE frequencies must share a device")
    if positions.device != q_input.device:
        raise ValueError("positions must share the Q device")

    freqs_real = torch.view_as_real(freqs_cis).flatten(-2).contiguous()
    if freqs_real.dtype != torch.float32 or freqs_real.shape[-1] != 64:
        raise ValueError("freqs_cis must materialize a float32 (max_pos, 64) table")

    q_int8 = torch.empty(q_input.shape, dtype=torch.int8, device=q_input.device)
    weights_out = torch.empty(
        (*q_input.shape[:-1], 1), dtype=torch.float32, device=q_input.device
    )
    q_contig = q_input if q_input.is_contiguous() else q_input.contiguous()
    positions_i32 = positions.to(torch.int32).contiguous()
    _module().fused_q_int8(
        q_contig,
        q_int8,
        weight,
        weights_out,
        float(weight_scale),
        freqs_real,
        positions_i32,
    )
    return q_int8, weights_out
