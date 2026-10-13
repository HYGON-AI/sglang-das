"""Select and prepare the replay state-index kernel during backend startup."""

from __future__ import annotations

import logging
import os

import torch

from .mamba_state_indices_triton import fused_replay_state_indices

logger = logging.getLogger(__name__)


def prepare_fused_replay_state_indices(device):
    """Return a ready callable; keep the original Triton fallback on CUDA."""
    backend = os.environ.get("SGLANG_REPLAY_STATE_INDICES_BACKEND", "auto").lower()
    if backend not in {"auto", "hip", "triton"}:
        raise ValueError(
            "SGLANG_REPLAY_STATE_INDICES_BACKEND must be auto, hip or triton"
        )
    if backend == "triton" or (backend == "auto" and torch.version.hip is None):
        return fused_replay_state_indices
    from .mamba_state_indices_hip import (
        fused_replay_state_indices_hip,
        prepare_replay_state_indices,
    )

    try:
        prepare_replay_state_indices(device)
    except Exception:
        if backend == "hip":
            raise
        logger.warning(
            "Replay indices HIP initialization failed; using Triton", exc_info=True
        )
        return fused_replay_state_indices
    logger.info("Replay state indices: HIP C++ kernel prepared before graph capture")
    return fused_replay_state_indices_hip
