"""Opt-in BoltOps TileLang iHC operators; installed packages are untouched."""

import importlib
import logging
from functools import lru_cache

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def _warn(reason):
    logger.warning("HY4 iHC TileLang falls back to eager torch: %s", reason)


@lru_cache(maxsize=1)
def _get_kernels():
    try:
        module = importlib.import_module("boltops.ihc")
    except ImportError as exc:
        _warn(f"boltops.ihc import unavailable: {exc}")
        return None
    names = ("ihc_pre", "ihc_post", "ihc_head")
    kernels = {name: getattr(module, name, None) for name in names}
    missing = [name for name, kernel in kernels.items() if not callable(kernel)]
    if missing:
        _warn(f"boltops.ihc requires all three APIs; unavailable: {', '.join(missing)}")
        return None
    return kernels


def _get_kernel(name):
    # A partial installation must never enable only part of the iHC backend.
    kernels = _get_kernels()
    return None if kernels is None else kernels[name]


@lru_cache(maxsize=3)
def _log_success(name="ihc_pre"):
    # Logged after the real API returns, not merely after resolving an import.
    logger.info("HY4 iHC TileLang call returned: boltops.ihc.%s", name)


def _unsupported(x, fn, scale, base):
    if x.ndim != 3 or x.shape[1] != 4:
        return "expected residual [T,4,D]"
    # The epilogue distributes gcd(D,512) values over 64 apply threads.
    if x.shape[2] <= 0 or x.shape[2] % 64:
        return "D must be a positive multiple of 64"
    if fn.shape != (8, 4 * x.shape[2]) or scale.numel() != 2 or base.shape != (8,):
        return "expected weight [8,4D], scale with 2 values, base [8]"
    if x.dtype != torch.bfloat16 or any(t.dtype != torch.float32 for t in (fn, scale, base)):
        return "expected BF16 residual and FP32 weight/scale/base"
    if any(t.device != x.device for t in (fn, scale, base)):
        return "all tensors must share a device"
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, fn, scale, base)):
        return "TileLang iHC is inference-only"
    if torch.version.hip is None or x.device.type != "cuda":
        return "requires HCU/HIP device"
    return None


def try_tilelang_ihc_pre(x, fn, scale, base, rms_eps, hc_eps, magnitude):
    """Return (BF16 reduced, FP32 post), or None for eager fallback.

    Compilation/execution errors propagate. This API includes coefficient
    normalization but not the independent input/post-attention RMSNorm.
    """
    if not envs.SGLANG_OPT_HY4_IHC_TILELANG.get():
        return None
    reason = _unsupported(x, fn, scale, base)
    if reason:
        _warn(reason)
        return None
    if x.shape[0] == 0:
        return x.new_empty((0, x.shape[2])), x.new_empty((0, 4), dtype=torch.float32)
    kernel = _get_kernel("ihc_pre")
    if kernel is None:
        return None
    with torch.cuda.device(x.device):
        result = kernel(
            x.contiguous(), fn.contiguous(), scale.contiguous().view(2),
            base.contiguous(), float(rms_eps), float(hc_eps), float(magnitude),
        )
    _log_success()
    return result


def try_tilelang_ihc_post(x, residual, post):
    """Return TileLang post result, or None for the eager caller."""
    if not envs.SGLANG_OPT_HY4_IHC_TILELANG.get():
        return None
    if (x.ndim != 2 or residual.ndim != 3 or residual.shape[1] != 4
            or x.shape != (residual.shape[0], residual.shape[2])
            or post.shape != (residual.shape[0], 4)
            or x.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16
            or post.dtype != torch.float32
            or any(t.device != residual.device for t in (x, post))):
        _warn("ihc_post input contract is unsupported")
        return None
    if residual.shape[2] <= 0 or residual.shape[2] % 64:
        _warn("ihc_post requires hidden size divisible by 64")
        return None
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, residual, post)):
        _warn("ihc_post is inference-only")
        return None
    if torch.version.hip is None or residual.device.type != "cuda":
        _warn("ihc_post requires HCU/HIP device")
        return None
    if residual.shape[0] == 0:
        return torch.empty_like(residual)
    kernel = _get_kernel("ihc_post")
    if kernel is None:
        return None
    with torch.cuda.device(residual.device):
        result = kernel(x.contiguous(), residual.contiguous(), post.contiguous())
    _log_success("ihc_post")
    return result


def try_tilelang_ihc_head(residual, head_fn, head_scale, head_base, rms_eps, hc_eps):
    """Return TileLang head result, or None for the eager caller."""
    if not envs.SGLANG_OPT_HY4_IHC_TILELANG.get():
        return None
    if (residual.ndim != 3 or residual.shape[1] != 4
            or head_fn.shape != (4, 4 * residual.shape[2])
            or head_base.shape != (4,) or head_scale.numel() != 1
            or residual.dtype != torch.bfloat16
            or any(t.dtype != torch.float32 for t in (head_fn, head_scale, head_base))
            or any(t.device != residual.device for t in (head_fn, head_scale, head_base))):
        _warn("ihc_head input contract is unsupported")
        return None
    if residual.shape[2] <= 0 or residual.shape[2] % 64:
        _warn("ihc_head requires hidden size divisible by 64")
        return None
    if torch.is_grad_enabled() and any(t.requires_grad for t in (residual, head_fn, head_scale, head_base)):
        _warn("ihc_head is inference-only")
        return None
    if torch.version.hip is None or residual.device.type != "cuda":
        _warn("ihc_head requires HCU/HIP device")
        return None
    if residual.shape[0] == 0:
        return residual.new_empty((0, residual.shape[2]))
    kernel = _get_kernel("ihc_head")
    if kernel is None:
        return None
    with torch.cuda.device(residual.device):
        result = kernel(residual.contiguous(), head_fn.contiguous(), head_scale.contiguous(), head_base.contiguous(), float(rms_eps), float(hc_eps))
    _log_success("ihc_head")
    return result
