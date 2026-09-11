"""Optional boltops TileLang kernels for the full four-stream iHC junction.

Each ``try_tilelang_ihc_*`` fuses one eager iHC operation in ``hunyuan_v4.py``
into a single boltops launch and returns ``None`` to keep the eager path:

- ``ihc_pre``:  coefficient GEMM + RMS + pre/post gates + weighted reduction.
- ``ihc_post``: ``post * x + residual``.
- ``ihc_head``: coefficient GEMM + RMS + head gate + weighted reduction.

Gated by ``SGLANG_OPT_HY4_IHC_TILELANG`` and separate from the deepgemm-only
``try_hcu_ihc_prenorm``. Any missing package, unsupported arch, or kernel error
falls back to eager torch, so a non-target device never crashes here.
"""

import logging
from functools import lru_cache
from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


@lru_cache(maxsize=16)
def _warn_fallback(reason: str) -> None:
    logger.warning("HY4 iHC TileLang kernels unused, eager torch: %s", reason)


@lru_cache(maxsize=1)
def _get_ops():
    try:
        from boltops import ihc
    except ImportError:
        _warn_fallback("the boltops package or an import dependency is missing")
        return None
    for name in ("ihc_pre", "ihc_post", "ihc_head"):
        if not callable(getattr(ihc, name, None)):
            _warn_fallback(f"boltops.ihc.{name} is unavailable")
            return None
    logger.info("HY4 iHC uses boltops.ihc TileLang kernels (HCU)")
    return ihc


def _device_ok(tensor: torch.Tensor) -> bool:
    # The TileLang kernels launch on the current HIP stream.
    return torch.version.hip is not None and tensor.device.type == "cuda"


def _enabled(stream_3d: torch.Tensor) -> bool:
    if not envs.SGLANG_OPT_HY4_IHC_TILELANG.get():
        return False
    if stream_3d.shape[0] == 0:
        # DP-attention idle ranks hand a 0-token batch; eager handles zero rows,
        # and a 0-block grid is not exercised by these kernels.
        return False
    return _device_ok(stream_3d)


def try_tilelang_ihc_pre(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    magnitude: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return (reduced [T,D] bf16, post [T,hc] fp32), or None for the eager path.

    ``reduced`` is unnormalized; the caller still applies its RMSNorm, matching
    the eager ``HYV4HCPreLayer`` contract.
    """
    if not _enabled(hidden_states):
        return None
    ops = _get_ops()
    if ops is None:
        return None
    if hidden_states.ndim != 3 or hidden_states.dtype != torch.bfloat16:
        _warn_fallback("ihc_pre expects a BF16 [T, hc, D] residual stream")
        return None
    hc, dim = hidden_states.shape[1], hidden_states.shape[2]
    if weight.dtype != torch.float32 or weight.shape != (2 * hc, hc * dim):
        _warn_fallback("ihc_pre expects an FP32 coefficient weight [2*hc, hc*D]")
        return None
    residual = hidden_states.contiguous()
    try:
        with torch.cuda.device(residual.device):
            reduced, post = ops.ihc_pre(
                residual,
                weight.contiguous(),
                scale.float().contiguous(),
                base.float().contiguous(),
                float(rms_eps),
                float(hc_eps),
                float(magnitude),
            )
    except Exception as exc:
        _warn_fallback(f"ihc_pre kernel raised: {exc!r}")
        return None
    return reduced, post


def try_tilelang_ihc_post(
    output: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Return ``post * output + residual`` as [T, hc, D] bf16, or None for eager."""
    if not _enabled(residual):
        return None
    ops = _get_ops()
    if ops is None:
        return None
    if (
        output.ndim != 2
        or residual.ndim != 3
        or output.dtype != torch.bfloat16
        or residual.dtype != torch.bfloat16
        or post.dtype != torch.float32
    ):
        _warn_fallback("ihc_post expects BF16 x [T,D] / residual [T,hc,D], FP32 post")
        return None
    try:
        with torch.cuda.device(residual.device):
            return ops.ihc_post(
                output.contiguous(), residual.contiguous(), post.contiguous()
            )
    except Exception as exc:
        _warn_fallback(f"ihc_post kernel raised: {exc!r}")
        return None


def try_tilelang_ihc_head(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    head_scale: torch.Tensor,
    head_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> Optional[torch.Tensor]:
    """Return the collapsed hidden state [T, D] bf16, or None for the eager path.

    Output is unnormalized; the caller still applies its optional RMSNorm.
    """
    if not _enabled(hidden_states):
        return None
    ops = _get_ops()
    if ops is None:
        return None
    if hidden_states.ndim != 3 or hidden_states.dtype != torch.bfloat16:
        _warn_fallback("ihc_head expects a BF16 [T, hc, D] residual stream")
        return None
    hc, dim = hidden_states.shape[1], hidden_states.shape[2]
    if weight.dtype != torch.float32 or weight.shape != (hc, hc * dim):
        _warn_fallback("ihc_head expects an FP32 coefficient weight [hc, hc*D]")
        return None
    residual = hidden_states.contiguous()
    try:
        with torch.cuda.device(residual.device):
            return ops.ihc_head(
                residual,
                weight.contiguous(),
                head_scale.float().contiguous(),
                head_base.float().contiguous(),
                float(rms_eps),
                float(hc_eps),
            )
    except Exception as exc:
        _warn_fallback(f"ihc_head kernel raised: {exc!r}")
        return None
