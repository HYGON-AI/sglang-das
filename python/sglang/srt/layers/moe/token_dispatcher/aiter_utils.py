from __future__ import annotations

import torch

from sglang.srt.layers.moe.utils import get_moe_runner_backend
from sglang.srt.utils import get_bool_env_var, is_hip


def should_use_aiter_runner() -> bool:
    """Honor both the ROCm default and an explicit AITER runner choice."""
    backend = get_moe_runner_backend()
    return backend.is_aiter() or (
        backend.is_auto() and get_bool_env_var("SGLANG_USE_AITER") and is_hip()
    )


def build_aiter_sink_expert_metadata(
    num_local_experts: int, device: torch.device | int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build legacy and unified-AITER metadata for invalid expert slots."""
    expert_mask = torch.zeros(
        num_local_experts + 1,
        device=device,
        dtype=torch.int,
    )
    expert_mask[:-1] = 1
    return expert_mask, expert_mask.to(torch.bool)
