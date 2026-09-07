"""``npu_quant_matmul`` via ``lightop.gemm_w8a8_smooth`` + lightweight epilogue.

Ported from ``lightop/test/npu/npu_quant_matmul/npu_quant_matmul_w8a8.py``.

Core GEMM always goes through W8A8 when possible::

    out0 = (A @ B) * scale_a * scale_b     # fused in smooth GEMM

Then apply Triton-aligned epilogue in elementwise form::

    int32 bias:  out = out0 + bias * scale_b * scale_a
    float bias:  out = out0 + bias
    offset:      out = out + offset   (only when bias is None or int32)

Falls back to ``npu_quant_matmul_triton`` when W8A8 cannot run.

``gemm_w8a8_smooth`` requires B as a non-contiguous TN ``[K, N]`` view
(stride ``(1, K)``, storage contiguous ``[N, K]``). Do that **once** at
load with ``pack_int8_weight_as_tn`` so CUDA-graph replay does not recopy
every int8 weight. Prefill still goes through the same ``smooth()`` call
and keeps the large-M Tensile I8II kernel.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch

_OUT_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "int32": torch.int32,
    "int8": torch.int8,
}

_LAST_BACKEND = "uninitialized"

# Fallback cache for callers that skip load-time packing. Cap must cover unique
# dense GEMMs (~239/PP stage on Kimi-K3); 64 evicted the working set every token.
_TN_CACHE: Dict[Tuple[Any, ...], torch.Tensor] = {}
_TN_CACHE_MAX = int(os.environ.get("NPU_QUANT_MATMUL_TN_CACHE_MAX", "8192"))

# Per-device cached ones column for scale_a when pertoken is absent.
_ONES_A: Dict[Tuple[str, int], torch.Tensor] = {}


def last_backend() -> str:
    return _LAST_BACKEND


def clear_tn_cache() -> None:
    """Drop cached TN weight views (call if weights are updated in-place)."""
    _TN_CACHE.clear()


def _normalize_out_dtype(output_dtype: Optional[object]) -> torch.dtype:
    if output_dtype is None:
        return torch.float16
    if isinstance(output_dtype, torch.dtype):
        return output_dtype
    key = str(output_dtype).replace("torch.", "").strip().lower()
    if key not in _OUT_DTYPE:
        raise TypeError(f"unsupported output_dtype: {output_dtype!r}")
    return _OUT_DTYPE[key]


def _try_import_smooth():
    try:
        import lightop  # type: ignore

        fn = getattr(lightop, "gemm_w8a8_smooth", None)
        if callable(fn):
            return fn, "lightop.gemm_w8a8_smooth"
    except Exception as e:
        return None, f"lightop import failed: {type(e).__name__}: {e}"
    return None, "lightop.gemm_w8a8_smooth missing"


def _tn_cache_key(x2: torch.Tensor) -> Tuple[Any, ...]:
    return (
        x2.device,
        x2.data_ptr(),
        x2.storage_offset(),
        tuple(x2.shape),
        tuple(x2.stride()),
        x2.dtype,
    )


def is_tn_int8(x2: torch.Tensor) -> bool:
    """True if ``x2`` is already the TN layout ``gemm_w8a8_smooth`` wants."""
    return (
        x2.dim() == 2
        and (not x2.is_contiguous())
        and x2.stride(0) == 1
        and x2.stride(1) == x2.size(0)
    )


def pack_int8_weight_as_tn(x2: torch.Tensor) -> torch.Tensor:
    """Pack contiguous NN ``[K, N]`` int8 into a TN view of the same shape.

    Result has ``stride == (1, K)`` and is backed by a contiguous ``[N, K]``
    storage. Same number of bytes as ``x2``; call once at weight load and
    assign onto ``layer.weight.data`` so decode CUDA graphs never launch
    ``direct_copy`` on the weights.

    Idempotent if ``x2`` is already TN.
    """
    if is_tn_int8(x2):
        return x2
    if x2.dim() != 2:
        raise ValueError(f"pack_int8_weight_as_tn expects 2D, got {tuple(x2.shape)}")
    return x2.t().contiguous().t()


def prefetch_tn_weight(x2: torch.Tensor) -> torch.Tensor:
    """Warm the TN cache for a still-NN weight (graph capture must see a hit).

    Prefer ``pack_int8_weight_as_tn`` on the Parameter itself — that uses no
    extra memory. Use this only when the live Parameter must stay NN.
    """
    return _make_tn_b(x2)


def _make_tn_b(x2: torch.Tensor) -> torch.Tensor:
    """Return [K, N] int8 with contiguous==False (TN), as required by smooth GEMM."""
    if is_tn_int8(x2):
        return x2

    key = _tn_cache_key(x2)
    hit = _TN_CACHE.get(key)
    if hit is not None and hit.shape == x2.shape and hit.device == x2.device:
        return hit

    b = pack_int8_weight_as_tn(x2)
    if len(_TN_CACHE) >= _TN_CACHE_MAX:
        _TN_CACHE.pop(next(iter(_TN_CACHE)))
    _TN_CACHE[key] = b
    return b


def _ones_scale_a(M: int, device: torch.device) -> torch.Tensor:
    key = (str(device), M)
    t = _ONES_A.get(key)
    if t is None or t.device != device or t.shape[0] != M:
        t = torch.ones(M, 1, device=device, dtype=torch.float32)
        _ONES_A[key] = t
    return t


def _prepare_scales(
    M: int,
    N: int,
    device: torch.device,
    scale: torch.Tensor,
    pertoken_scale: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if pertoken_scale is None:
        scale_a = _ones_scale_a(M, device)
    else:
        scale_a = (
            pertoken_scale.to(device=device, dtype=torch.float32)
            .reshape(M, 1)
            .contiguous()
        )

    scale_b = scale.to(device=device, dtype=torch.float32).reshape(-1)
    if scale_b.numel() == 1:
        scale_b = scale_b.expand(N).contiguous()
    elif scale_b.numel() != N:
        raise ValueError(f"scale length must be 1 or N={N}, got {scale_b.numel()}")
    scale_b = scale_b.reshape(N, 1).contiguous()
    return scale_a, scale_b


def _apply_epilogue(
    out: torch.Tensor,
    *,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: Optional[torch.Tensor],
    offset: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Apply bias/offset in ``out_dtype`` (avoid full fp32 materialization)."""
    has_bias = bias is not None
    use_int32_bias = has_bias and bias.dtype == torch.int32
    apply_offset = offset is not None and ((not has_bias) or use_int32_bias)

    if not has_bias and not apply_offset:
        return out if out.dtype == out_dtype else out.to(out_dtype)

    y = out if out.dtype == out_dtype else out.to(out_dtype)

    if use_int32_bias:
        # (mm + bias) * sa * sb = out0 + bias * sa * sb
        b = bias.to(device=y.device, dtype=torch.float32).reshape(1, -1)
        term = b * scale_b.reshape(1, -1) * scale_a.reshape(-1, 1)
        y = y + term.to(out_dtype)
    elif has_bias:
        y = y + bias.to(device=y.device, dtype=out_dtype).reshape(1, -1)

    if apply_offset:
        off = offset.to(device=y.device, dtype=out_dtype).reshape(-1)
        if off.numel() == 1:
            y = y + off
        else:
            y = y + off.reshape(1, -1)
    return y


def _run_fallback(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: torch.Tensor,
    offset: Optional[torch.Tensor],
    pertoken_scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    output_dtype: Any,
    reason: str,
) -> torch.Tensor:
    global _LAST_BACKEND
    from sglang.kernels.npu_kernels.npu_quant_matmul_triton import (
        npu_quant_matmul_triton,
    )

    _LAST_BACKEND = f"fallback:npu_quant_matmul_triton ({reason})"
    od = (
        output_dtype
        if isinstance(output_dtype, torch.dtype)
        else _normalize_out_dtype(output_dtype)
    )
    return npu_quant_matmul_triton(
        x1,
        x2,
        scale,
        offset=offset,
        pertoken_scale=pertoken_scale,
        bias=bias,
        output_dtype=od,
    )


def _w8a8_2d(
    smooth,
    smooth_src: str,
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: torch.Tensor,
    offset: Optional[torch.Tensor],
    pertoken_scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Run W8A8 + epilogue on 2D inputs. Return None to signal caller fallback."""
    global _LAST_BACKEND
    M, K = int(x1.shape[0]), int(x1.shape[1])
    N = int(x2.shape[1])

    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None
    if M > 16 and K < 128:
        return None
    if os.environ.get("NPU_QUANT_MATMUL_FORCE_FALLBACK", "0") == "1":
        return None

    a = x1 if x1.is_contiguous() else x1.contiguous()
    b = _make_tn_b(x2)
    scale_a, scale_b = _prepare_scales(M, N, a.device, scale, pertoken_scale)

    status, out = smooth(a, b, scale_a, scale_b, None, out_dtype)
    if (not status) or out is None:
        return None

    out = _apply_epilogue(
        out,
        scale_a=scale_a,
        scale_b=scale_b,
        bias=bias,
        offset=offset,
        out_dtype=out_dtype,
    )
    tag = smooth_src
    if bias is not None or offset is not None:
        tag = f"{smooth_src}+epilogue"
    _LAST_BACKEND = tag
    return out


def npu_quant_matmul_w8a8(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: torch.Tensor,
    offset: Optional[torch.Tensor] = None,
    pertoken_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    output_dtype: Optional[object] = None,
) -> torch.Tensor:
    """``npu_quant_matmul`` via W8A8 smooth GEMM + epilogue when possible."""
    out_dtype = _normalize_out_dtype(output_dtype)

    smooth, smooth_src = _try_import_smooth()
    if smooth is None:
        return _run_fallback(
            x1,
            x2,
            scale,
            offset,
            pertoken_scale,
            bias,
            out_dtype,
            smooth_src,
        )

    if x1.dim() == 2 and x2.dim() == 2:
        if x2.shape[0] != x1.shape[1]:
            raise ValueError(
                f"K mismatch: x1 K={x1.shape[1]}, x2 K={x2.shape[0]}"
            )
        out = _w8a8_2d(
            smooth,
            smooth_src,
            x1,
            x2,
            scale,
            offset,
            pertoken_scale,
            bias,
            out_dtype,
        )
        if out is not None:
            return out
        return _run_fallback(
            x1,
            x2,
            scale,
            offset,
            pertoken_scale,
            bias,
            out_dtype,
            "gemm_w8a8_smooth returned False/None or shape gate",
        )

    if x1.dim() >= 2 and x2.dim() == 2:
        batch = x1.shape[:-2]
        M, K = int(x1.shape[-2]), int(x1.shape[-1])
        N = int(x2.shape[-1])
        if x2.shape[0] != K:
            raise ValueError(f"K mismatch: x1 K={K}, x2 K={x2.shape[0]}")
        a2 = x1.reshape(-1, K)
        pts = None
        if pertoken_scale is not None:
            pts = pertoken_scale.reshape(a2.shape[0])
        if bias is not None and bias.dim() > 1:
            return _run_fallback(
                x1,
                x2,
                scale,
                offset,
                pertoken_scale,
                bias,
                out_dtype,
                "batched bias not supported in W8A8 flatten path",
            )
        out2 = _w8a8_2d(
            smooth,
            smooth_src,
            a2,
            x2,
            scale,
            offset,
            pts,
            bias,
            out_dtype,
        )
        if out2 is not None:
            return out2.reshape(batch + (M, N))
        return _run_fallback(
            x1,
            x2,
            scale,
            offset,
            pertoken_scale,
            bias,
            out_dtype,
            "batched W8A8 failed",
        )

    return _run_fallback(
        x1,
        x2,
        scale,
        offset,
        pertoken_scale,
        bias,
        out_dtype,
        "unsupported input ranks for W8A8 path",
    )


npu_quant_matmul = npu_quant_matmul_w8a8
