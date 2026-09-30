"""Low-latency MiniMax MoE router GEMV for small decode batches."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _minimax_router_gemv_kernel(
    hidden_states_ptr,
    router_weight_ptr,
    output_ptr,
    NUM_TOKENS: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    TOKENS_PER_PROGRAM: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert_id = tl.program_id(0)
    token_block_id = tl.program_id(1)

    token_offsets = token_block_id * TOKENS_PER_PROGRAM + tl.arange(
        0, TOKENS_PER_PROGRAM
    )
    token_mask = token_offsets < NUM_TOKENS
    accumulator = tl.zeros((TOKENS_PER_PROGRAM,), dtype=tl.float32)

    # A program owns one expert and a small group of tokens. The expert vector
    # is loaded once and reused across up to four tokens. Keeping the token
    # group small limits register pressure while retaining weight reuse at
    # DP4/DP8.
    for k_start in range(0, HIDDEN_SIZE, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < HIDDEN_SIZE
        weight = tl.load(
            router_weight_ptr + expert_id * HIDDEN_SIZE + k_offsets,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        hidden = tl.load(
            hidden_states_ptr
            + token_offsets[:, None] * HIDDEN_SIZE
            + k_offsets[None, :],
            mask=token_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(hidden * weight[None, :], axis=1)

    tl.store(
        output_ptr + token_offsets * NUM_EXPERTS + expert_id,
        accumulator,
        mask=token_mask,
    )


def can_use_minimax_router_gemv(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
) -> bool:
    """Return whether the specialized small-M kernel supports this problem."""
    return (
        hidden_states.is_cuda
        and router_weight.is_cuda
        and hidden_states.dim() == 2
        and router_weight.dim() == 2
        # Target verify gathers the DP4 local Q2 rows before the MoE and
        # therefore presents M=32 at the strict c16 workload.  The specialized
        # kernel remains substantially faster than rocBLAS through M=64 on
        # gfx938; stopping at 16 made every sparse layer silently fall back.
        and 0 < hidden_states.shape[0] <= 64
        and hidden_states.shape[1] == router_weight.shape[1]
        and router_weight.shape[0] == 128
        and hidden_states.shape[1] == 6144
        and hidden_states.dtype == torch.bfloat16
        and router_weight.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and router_weight.is_contiguous()
    )


def minimax_router_gemv(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    *,
    block_k: int = 256,
    num_warps: int = 4,
) -> torch.Tensor:
    """Compute small-M MiniMax router logits with FP32 accumulation/output."""
    if not can_use_minimax_router_gemv(hidden_states, router_weight):
        raise ValueError(
            "MiniMax router GEMV requires CUDA/HIP contiguous BF16 tensors "
            "with M=1..64, N=128 and K=6144"
        )
    if block_k not in (128, 256, 512, 1024):
        raise ValueError(f"Unsupported MiniMax router GEMV block_k={block_k}")
    if num_warps not in (1, 2, 4, 8):
        raise ValueError(f"Unsupported MiniMax router GEMV num_warps={num_warps}")

    num_tokens = hidden_states.shape[0]
    num_experts = router_weight.shape[0]
    hidden_size = hidden_states.shape[1]
    # Reuse each router-weight load across up to eight rows.  At the important
    # M=32 verify shape this cuts the grid/load duplication in half while
    # retaining enough programs to fill gfx938.
    tokens_per_program = min(triton.next_power_of_2(num_tokens), 8)
    output = torch.empty(
        (num_tokens, num_experts),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    grid = (num_experts, triton.cdiv(num_tokens, tokens_per_program))
    _minimax_router_gemv_kernel[grid](
        hidden_states,
        router_weight,
        output,
        NUM_TOKENS=num_tokens,
        NUM_EXPERTS=num_experts,
        HIDDEN_SIZE=hidden_size,
        TOKENS_PER_PROGRAM=tokens_per_program,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return output


__all__ = ["can_use_minimax_router_gemv", "minimax_router_gemv"]
