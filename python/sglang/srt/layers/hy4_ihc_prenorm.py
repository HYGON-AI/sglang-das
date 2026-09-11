"""Optional HCU coefficient projection and RMS scaling for four-stream iHC."""

import importlib
import logging
from functools import lru_cache
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


@lru_cache(maxsize=16)
def _warn_fallback(reason: str) -> None:
    logger.warning("HY4 iHC prenorm uses eager torch: %s", reason)


@lru_cache(maxsize=1)
def _get_hcu_kernel():
    try:
        backend = importlib.import_module("deepgemm")
    except ImportError:
        _warn_fallback("the HCU deepgemm package or an import dependency is missing")
        return None
    kernel = getattr(backend, "tf32_hc_pernorm_gemm", None)
    if not callable(kernel):
        _warn_fallback("deepgemm.tf32_hc_pernorm_gemm is unavailable")
        return None
    logger.info("HY4 iHC prenorm uses deepgemm.tf32_hc_pernorm_gemm (HCU)")
    return kernel


def _unsupported_reason(hidden_states: torch.Tensor, weight: torch.Tensor):
    if hidden_states.ndim != 3 or hidden_states.shape[1] != 4:
        return "expected four residual streams [T, 4, D]"
    k = hidden_states.shape[1] * hidden_states.shape[2]
    if k == 0 or k % 64 != 0:
        return "flattened input width must be a positive multiple of 64"
    if weight.shape != (8, k):
        return "expected an unquantized coefficient weight [8, 4D]"
    if hidden_states.dtype != torch.bfloat16 or weight.dtype != torch.float32:
        return "expected BF16 activations and FP32 coefficient weights"
    if hidden_states.device != weight.device:
        return "activations and coefficient weights must share a device"
    if torch.is_grad_enabled() and (
        hidden_states.requires_grad or weight.requires_grad
    ):
        return "the fused kernel is inference-only"
    if torch.version.hip is None or hidden_states.device.type != "cuda":
        return "the fused kernel requires a HIP device"
    return None


def try_hcu_ihc_prenorm(
    hidden_states: torch.Tensor, weight: torch.Tensor, eps: float
) -> Optional[torch.Tensor]:
    """Return FP32 scaled coefficients, or None to retain the eager model path."""
    if not envs.SGLANG_OPT_HY4_IHC_PRENORM.get():
        return None
    reason = _unsupported_reason(hidden_states, weight)
    if reason is not None:
        _warn_fallback(reason)
        return None
    m = hidden_states.shape[0]
    if m == 0:
        return torch.empty((0, 8), dtype=torch.float32, device=hidden_states.device)
    kernel = _get_hcu_kernel()
    if kernel is None:
        return None

    a = hidden_states.flatten(1).contiguous()
    raw = torch.empty((m, 8), dtype=torch.float32, device=hidden_states.device)
    sqr_sum = torch.empty((m,), dtype=torch.float32, device=hidden_states.device)
    # The HCU launcher uses the current device's stream; keep it with the inputs.
    with torch.cuda.device(hidden_states.device):
        kernel(a, weight.contiguous(), raw, sqr_sum, num_splits=None)
    # The kernel writes a sum, whereas the model's RMS scale uses a mean over 4D.
    return raw * torch.rsqrt(sqr_sum / a.shape[1] + eps).unsqueeze(-1)
