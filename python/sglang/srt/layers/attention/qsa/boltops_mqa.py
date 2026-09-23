"""Default BoltOPs TileLang QSA MQA. Callers fall back when this returns None."""

from __future__ import annotations

import importlib
import logging
from functools import lru_cache
from typing import Optional

import torch

from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode

logger = logging.getLogger(__name__)

# Flash-Next indexer Q is 4 heads and is not sharded by TP. HIP fused prep
# zero-pads decode Q to a multiple of 8, so the tensor arriving here is 8
# (or 16 if a caller already applied the in-tree MFMA pad). BoltOPs accepts
# those widths, but the tuned input is the real 4 heads. Wider legal widths
# are not this padding and are passed through.
_INDEXER_HEADS = 4
_PADDED_DECODE_HEADS = (8, 16)


@lru_cache(maxsize=32)
def _warn(reason: str) -> None:
    logger.warning("QSA MQA BoltOPs falls back to in-tree TileLang: %s", reason)


@lru_cache(maxsize=1)
def _get_kernels():
    try:
        module = importlib.import_module("boltops.mqa_logits")
    except ImportError as exc:
        _warn(f"boltops.mqa_logits import unavailable: {exc}")
        return None
    prefill = getattr(module, "tilelang_qsa_mqa_prefill", None)
    decode = getattr(module, "tilelang_qsa_mqa_decode", None)
    if not callable(prefill) or not callable(decode):
        _warn("boltops.mqa_logits is missing tilelang_qsa_mqa_prefill/decode")
        return None
    return prefill, decode


@lru_cache(maxsize=2)
def _log_success(name: str) -> None:
    logger.info("QSA MQA BoltOPs call returned: boltops.mqa_logits.%s", name)


def _slice_padded_decode_heads(q: torch.Tensor) -> torch.Tensor:
    if q.shape[1] in _PADDED_DECODE_HEADS:
        return q[:, :_INDEXER_HEADS]
    return q


def _effective_key_range(
    row_starts: torch.Tensor, row_ends: torch.Tensor, keys: int
) -> Optional[int]:
    """Longest visible compressed-K span. Eager only; CUDA graph must not sync."""
    if get_is_capture_mode() or row_starts.numel() == 0 or keys <= 0:
        return None
    span = int((row_ends - row_starts).clamp_min(0).max().item())
    if span <= 0:
        return None
    return max(1, min(keys, span))


def try_boltops_qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
) -> Optional[torch.Tensor]:
    """Return BoltOPs prefill logits, or None when BoltOPs cannot run this input."""
    if q.device.type != "cuda":
        return None
    kernels = _get_kernels()
    if kernels is None:
        return None
    prefill, _ = kernels
    try:
        with torch.cuda.device(q.device):
            logits = prefill(
                q,
                k,
                row_starts,
                row_ends,
                score_scale,
                effective_range=_effective_key_range(row_starts, row_ends, k.shape[0]),
            )
    except Exception as exc:
        _warn(f"prefill failed: {exc}")
        return None
    _log_success("tilelang_qsa_mqa_prefill")
    return logits


def try_boltops_qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
) -> Optional[torch.Tensor]:
    """Return BoltOPs decode logits, or None when BoltOPs cannot run this input.

    Decode config is chosen from batch and max_model_len on the host. Do not
    read context_lens here; CUDA graph replay must not sync.
    """
    if q.device.type != "cuda":
        return None
    kernels = _get_kernels()
    if kernels is None:
        return None
    _, decode = kernels
    try:
        with torch.cuda.device(q.device):
            logits = decode(
                _slice_padded_decode_heads(q),
                k_cache,
                page_table,
                context_lens,
                max_model_len,
                score_scale,
            )
    except Exception as exc:
        _warn(f"decode failed: {exc}")
        return None
    _log_success("tilelang_qsa_mqa_decode")
    return logits
