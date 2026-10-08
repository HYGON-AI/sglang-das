"""Production FP8 draft LM-head followed by deterministic Top-1."""

from __future__ import annotations

import torch

from sglang.kernels.ops.speculative.topk1 import draft_topk1_argmax

_FP8_DTYPE = torch.float8_e4m3fn


def quantize_lm_head_weight_fp8_per_channel(
    weight: torch.Tensor, *, chunk_rows: int = 1024
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize row-major ``[local_vocab, hidden]`` weights once after load."""
    if weight.ndim != 2 or not weight.is_cuda:
        raise ValueError("draft FP8 LM head requires a 2-D device weight")
    qweight = torch.empty_like(weight, dtype=_FP8_DTYPE)
    scales = torch.empty(
        (weight.shape[0], 1), dtype=torch.float32, device=weight.device
    )
    fp8_max = torch.finfo(_FP8_DTYPE).max
    for start in range(0, weight.shape[0], chunk_rows):
        end = min(start + chunk_rows, weight.shape[0])
        chunk = weight[start:end]
        scale = chunk.abs().float().amax(dim=1).clamp_min_(1e-12) / fp8_max
        qweight[start:end].copy_((chunk.float() / scale[:, None]).to(_FP8_DTYPE))
        scales[start:end, 0].copy_(scale)
    return qweight, scales


def fp8_lm_head_top1(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    valid_vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local maximum values/ids without creating global vocab logits."""
    from sglang.srt.layers.quantization.fp8_utils import (
        apply_fp8_lightop_channelwise_linear,
    )

    if hidden_states.ndim != 2 or weight.ndim != 2:
        raise ValueError("hidden_states and weight must both be 2-D")
    if hidden_states.shape[1] != weight.shape[1]:
        raise ValueError(
            f"LM-head K mismatch: {hidden_states.shape[1]} vs {weight.shape[1]}"
        )
    if weight.dtype != _FP8_DTYPE:
        raise ValueError(f"expected {_FP8_DTYPE} LM-head weight, got {weight.dtype}")
    if not 0 < valid_vocab_size <= weight.shape[0]:
        raise ValueError(
            f"invalid local vocab {valid_vocab_size} for {weight.shape[0]} rows"
        )
    local_logits = apply_fp8_lightop_channelwise_linear(
        hidden_states.contiguous(), weight, weight_scale
    )
    # Padding rows are zero-filled by the vocab loader and can incorrectly win
    # if every real logit is negative, so exclude them from the reduction.
    return draft_topk1_argmax(local_logits[:, :valid_vocab_size].contiguous())
