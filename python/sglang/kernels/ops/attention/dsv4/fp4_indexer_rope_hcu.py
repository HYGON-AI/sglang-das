"""Exact Torch RoPE plus fused FP4 pack in the HCU LightOp indexer layout."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.fp4_indexer import _index_k_rope_pack_kernel
from sglang.kernels.ops.attention.dsv4.torch_quant import FP4_AMAX_FLOOR
from sglang.srt.layers.attention.dsv4.dsv41_sparse import rope_tail


@triton.jit
def _index_q_fp4_pack_weights_hcu_kernel(
    Q,
    F,
    Positions,
    RawWeights,
    Payload,
    Scale,
    Weights,
    weight_scale,
    F_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    RD: tl.constexpr,
    AMAX_FLOOR: tl.constexpr,
):
    # RoPE is already computed by the exact Torch reference. RD=0 keeps
    # the two quantization stages while eliding the packer's own rotation.
    _index_k_rope_pack_kernel(
        Q,
        F,
        Positions,
        Payload,
        Scale,
        Payload,
        Positions,
        F_STRIDE=F_STRIDE,
        HEADS=HEADS,
        RD=0,
        INDEXED=False,
        STORE_CACHE=False,
        PAGE_SIZE=0,
        CACHE_STRIDE=0,
        AMAX_FLOOR=AMAX_FLOOR,
    )
    row = tl.program_id(0)
    raw = tl.load(RawWeights + row).to(tl.float32)
    # head_weights returns BF16 before its caller converts to FP32.
    scaled = (raw * weight_scale).to(tl.bfloat16).to(tl.float32)
    tl.store(Weights + row, scaled)


def index_q_rope_pack_weights_hcu(
    q: torch.Tensor,
    freqs: torch.Tensor,
    positions: torch.Tensor,
    raw_weights: torch.Tensor,
    weight_scale: float,
    rope_dim: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack BF16 [T,H,128] Q plus BF16 [T,H] raw weights for the HCU indexer.

    Output: int8 [T*H,64] payload, int32 [T*H] packed scale, FP32 [T,H]
    weights. Positions are valid nonnegative indices into the complex64 table.
    """
    assert q.ndim == 3 and q.dtype == torch.bfloat16 and q.shape[-1] == 128
    tokens, heads, _ = q.shape
    assert raw_weights.shape == (tokens, heads) and raw_weights.dtype == torch.bfloat16
    assert positions.shape == (tokens,) and positions.dtype in (
        torch.int32,
        torch.int64,
    )
    assert rope_dim == 64 and freqs.dtype == torch.complex64 and freqs.ndim == 2
    assert freqs.shape[1] == rope_dim // 2
    assert q.device == raw_weights.device == positions.device == freqs.device
    q = q.contiguous()
    raw_weights = raw_weights.contiguous()
    positions = positions.contiguous()
    f = torch.view_as_real(freqs.contiguous())
    payload = torch.empty((tokens * heads, 64), dtype=torch.int8, device=q.device)
    scale = torch.empty((tokens * heads,), dtype=torch.int32, device=q.device)
    weights = torch.empty((tokens, heads), dtype=torch.float32, device=q.device)
    if tokens * heads:
        # Preserve Torch complex multiply ordering at BF16/FP4 boundaries.
        q = rope_tail(q, freqs[positions], rope_dim)
        _index_q_fp4_pack_weights_hcu_kernel[(tokens * heads,)](
            q,
            f,
            positions,
            raw_weights,
            payload,
            scale,
            weights,
            float(weight_scale),
            F_STRIDE=f.stride(0),
            HEADS=heads,
            RD=rope_dim,
            AMAX_FLOOR=FP4_AMAX_FLOOR,
            num_warps=4,
        )
    return payload, scale, weights
