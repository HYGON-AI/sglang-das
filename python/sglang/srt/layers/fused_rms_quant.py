from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional, Tuple

import msgspec
import torch


class Int8PerTokenInput(msgspec.Struct, frozen=True):
    """A ColumnParallelLinear input prequantized by a fused RMSNorm: the linear
    unpacks it into ``(x, input_quant_args=(q, scale))``."""

    x: torch.Tensor  # normalized [M, K] bf16/fp16; sets the GEMM output dtype
    q: torch.Tensor  # [M, K] int8
    scale: torch.Tensor  # [M, 1] fp32, per token


class Int8QuantizedInput(msgspec.Struct, frozen=True):
    """Contiguous GPU [M,K] INT8 activations with no floating carrier."""

    q: torch.Tensor
    scale: torch.Tensor
    orig_dtype: torch.dtype

    def unpack(self):
        # Produced by LightOp and consumed only by an opted-in linear.
        return self.q, self.scale, self.orig_dtype


def supports_int8_quantized_input(linear) -> bool:
    """Whether an opted-in linear consumes INT8 without a floating carrier."""
    return bool(
        getattr(linear.quant_method, "supports_prequantized_input", False)
        and (
            getattr(linear.quant_method, "supports_int8_quantized_input", False)
            or getattr(
                getattr(linear, "scheme", None), "supports_int8_quantized_input", False
            )
        )
    )


@lru_cache(maxsize=1)
def is_lightop_sglang_rms_quant_available() -> bool:
    """Return whether the audited LightOp entry point can run on this host."""

    if torch.version.hip is None:
        return False
    try:
        import lightop
    except Exception:
        # This is a capability probe: a stale ABI, missing shared object, or
        # partially installed package must select the native fallback.
        return False
    return hasattr(lightop, "rms_norm_dynamic_per_token_quant_sglang") and hasattr(
        getattr(lightop, "op", None), "rms_norm_dynamic_per_token_quant_sglang"
    )


@lru_cache(maxsize=1)
def is_lightop_rms_fp8_quant_available() -> bool:
    """Return whether LightOp's RMSNorm + per-token FP8 quant can run on this host."""

    if torch.version.hip is None:
        return False
    try:
        from lightop import norm
    except Exception:
        return False
    return hasattr(norm, "rms_norm_per_token_fp8_quant")


def supports_fused_rms_quant_input(
    input: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> bool:
    """Return whether the LightOp fast path can preserve the current semantics.

    The production GLM5-N path uses two-dimensional, contiguous BF16 tensors
    with hidden sizes 4096 and 1536.  Keep the initial integration deliberately
    narrow so an unsupported view falls back to native RMSNorm + quantization
    instead of silently taking one of LightOp's generic strided paths.
    """

    return is_lightop_sglang_rms_quant_available() and _is_narrow_rms_quant_input(
        input, weight, residual
    )


def _is_narrow_rms_quant_input(
    input: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    *,
    allow_strided: bool = False,
    max_width: int = 8192,
) -> bool:
    if input.numel() == 0 or input.dim() != 2 or not input.is_cuda:
        return False
    if input.dtype not in (torch.float16, torch.bfloat16):
        return False
    if (
        weight.dtype != input.dtype
        or weight.device != input.device
        or weight.shape != (input.shape[-1],)
    ):
        return False
    if input.shape[-1] % 16 != 0 or input.shape[-1] > max_width:
        return False
    if not weight.is_contiguous():
        return False
    if allow_strided:
        if (
            input.stride(1) != 1
            or not input.shape[1] <= input.stride(0) <= 2147483647
            or input.stride(0) % 8
        ):
            return False
    elif not input.is_contiguous():
        return False
    if input.data_ptr() % 16 != 0 or weight.data_ptr() % 16 != 0:
        return False
    if residual is not None:
        if (
            residual.shape != input.shape
            or residual.dtype != input.dtype
            or residual.device != input.device
            or not residual.is_contiguous()
            or residual.data_ptr() % 16 != 0
            or residual.untyped_storage().data_ptr()
            == input.untyped_storage().data_ptr()
        ):
            return False
    return True


def fused_rms_norm_per_token_quant(
    *,
    input: torch.Tensor,
    rms_weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype = torch.int8,
    residual: Optional[torch.Tensor] = None,
    update_input: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run LightOp RMSNorm + dynamic per-token INT8 quantization.

    ``update_input=True`` is required when the normalized BF16/FP16 value has
    consumers in addition to the quantized GEMM (for example GLM5 KDA and its
    four projections).  LightOp also updates ``residual`` in place when one is
    supplied, matching the fused-add RMSNorm contract.
    """

    if quant_dtype != torch.int8:
        raise ValueError(f"Only torch.int8 is supported, got {quant_dtype}")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError(f"epsilon must be finite and positive, got {epsilon}")
    if not supports_fused_rms_quant_input(input, rms_weight, residual):
        raise ValueError(
            "LightOp fused RMS+INT8 quant requires an aligned, contiguous 2D "
            "FP16/BF16 input and weight, with hidden size divisible by 16 and <= 8192"
        )

    # Import lazily so the environment switch remains optional and CPU-only
    # tooling can import SGLang without loading the HIP extension.
    from lightop import rms_norm_dynamic_per_token_quant_sglang

    return rms_norm_dynamic_per_token_quant_sglang(
        input=input,
        weight=rms_weight,
        epsilon=epsilon,
        quant_dtype=quant_dtype,
        residual=residual,
        update_input=update_input,
    )


def linear_takes_fp8_operand(linear) -> bool:
    """Whether linear consumes LightOp's per-token ``(fp8, scale, orig_dtype)`` input
    as is (the HCU channelwise W8A8-FP8 scheme)."""
    return linear is not None and bool(
        getattr(
            getattr(linear, "scheme", None), "supports_fp8_prequantized_input", False
        )
    )


def rms_norm_fp8_operand(
    x: torch.Tensor, norm
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.dtype]]:
    """RMSNorm ``x`` in place and quantize it to per-token FP8 in one LightOp launch,
    as the ``(fp8, scale, orig_dtype)`` input a ``linear_takes_fp8_operand`` linear
    reads; None, with ``x`` untouched, outside the narrow layout the INT8 sibling
    accepts."""
    if not (
        is_lightop_rms_fp8_quant_available()
        and _is_narrow_rms_quant_input(x, norm.weight)
    ):
        return None
    from lightop.norm import rms_norm_per_token_fp8_quant

    q, scale = rms_norm_per_token_fp8_quant(
        x,
        norm.weight,
        epsilon=norm.variance_epsilon,
        fp8type=0,
        residual=None,
        update_input=True,
    )
    return q, scale, x.dtype


def opt_into_int8_operand(linear, *, quantized_only=False) -> bool:
    """Opt linear into a per-token INT8 ``input_quant_args`` pair (W8A8-INT8,
    SlimQuant W4A8) and return whether it now takes one."""
    from sglang.srt.layers.quantization.base_config import LinearMethodBase

    return (
        isinstance(linear.quant_method, LinearMethodBase)
        and linear.quant_method.enable_prequantized_input(linear)
        and (not quantized_only or supports_int8_quantized_input(linear))
    )


def rms_norm_int8_operand(x: torch.Tensor, norm) -> Optional[Int8PerTokenInput]:
    """RMSNorm ``x`` in place and quantize it to per-token INT8 in one LightOp
    launch; None, with ``x`` untouched, outside the narrow layout the op accepts."""
    if not supports_fused_rms_quant_input(x, norm.weight):
        return None
    q, scale = fused_rms_norm_per_token_quant(
        input=x,
        rms_weight=norm.weight,
        epsilon=norm.variance_epsilon,
        update_input=True,
    )
    return Int8PerTokenInput(x=x, q=q, scale=scale)


@lru_cache(maxsize=1)
def is_lightop_sglang_mla_qkv_a_rms_quant_available() -> bool:
    """Return whether the packed MLA RMSNorm+INT8 entry point is available."""
    try:
        import lightop
    except Exception:
        return False
    name = "mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang"
    return bool(
        torch.version.hip is not None
        and hasattr(lightop, name)
        and hasattr(getattr(lightop, "op", None), name)
    )


def supports_fused_mla_qkv_a_rms_quant_input(
    packed_input: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
) -> bool:
    """Check the exact packed-layout and alignment contract of the MLA op."""
    weights = (q_weight, kv_weight)
    q_cols = q_weight.numel()
    kv_cols = kv_weight.numel()
    return bool(
        is_lightop_sglang_mla_qkv_a_rms_quant_available()
        and packed_input.numel() > 0
        and packed_input.dim() == 2
        and packed_input.is_cuda
        and packed_input.dtype in (torch.float16, torch.bfloat16)
        and packed_input.is_contiguous()
        and packed_input.data_ptr() % 16 == 0
        and packed_input.shape[-1] % 8 == 0
        and q_cols + kv_cols <= packed_input.shape[-1]
        and all(0 < weight.numel() <= 8192 for weight in weights)
        and all(weight.numel() % 16 == 0 for weight in weights)
        and all(weight.dim() == 1 for weight in weights)
        and all(weight.dtype == packed_input.dtype for weight in weights)
        and all(weight.device == packed_input.device for weight in weights)
        and all(weight.is_contiguous() for weight in weights)
        and all(weight.data_ptr() % 16 == 0 for weight in weights)
    )


def fused_mla_qkv_a_rms_norm_per_token_quant(
    *,
    packed_input: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    q_epsilon: float,
    kv_epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize packed MLA q/kv in place and return prequantized q."""

    if not all(
        math.isfinite(epsilon) and epsilon > 0 for epsilon in (q_epsilon, kv_epsilon)
    ):
        raise ValueError("q_epsilon and kv_epsilon must be finite and positive")
    if not supports_fused_mla_qkv_a_rms_quant_input(packed_input, q_weight, kv_weight):
        raise ValueError(
            "LightOp packed MLA fusion requires an aligned contiguous 2D "
            "FP16/BF16 [q_lora, kv_lora, q_rope] tensor and aligned weights"
        )

    from lightop import mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang

    return mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang(
        packed_input=packed_input,
        q_weight=q_weight,
        kv_weight=kv_weight,
        q_epsilon=q_epsilon,
        kv_epsilon=kv_epsilon,
    )


@lru_cache(maxsize=2)
def _strided_rms_quant_op(quant_dtype):
    if torch.version.hip is None:
        return None
    try:
        from lightop import rmsnorm
    except (ImportError, OSError, RuntimeError):
        return None
    if quant_dtype == torch.int8:
        version = getattr(rmsnorm, "SGLANG_NATIVE_RMS_QUANT_VERSION", 1)
        name = "rms_norm_dynamic_per_token_quant_sglang"
        if not is_lightop_sglang_rms_quant_available():
            return None
    else:
        version = getattr(rmsnorm, "RMS_NORM_FP8_QUANT_VERSION", 1)
        name = "rms_norm_per_token_fp8_quant"
    return getattr(rmsnorm, name, None) if version >= 2 else None


def _rms_norm_strided_quant(x, norm, quant_dtype):
    op = _strided_rms_quant_op(quant_dtype)
    max_width = 8192 if quant_dtype == torch.int8 else 16384
    if (
        op is None
        or not _is_narrow_rms_quant_input(
            x, norm.weight, allow_strided=True, max_width=max_width
        )
        # Native FP8 uses a signed 32-bit row-offset product.
        or x.shape[0] * x.stride(0) > 2147483647
        or not math.isfinite(norm.variance_epsilon)
        or norm.variance_epsilon <= 0
    ):
        return None
    normalized = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    q, scale = op(
        x,
        norm.weight,
        epsilon=norm.variance_epsilon,
        update_input=False,
        norm_output=normalized,
    )
    return normalized, q, scale


def rms_norm_strided_int8_operand(x: torch.Tensor, norm) -> Optional[Int8PerTokenInput]:
    """Read the Q slice without modifying the packed projection's KV storage."""
    result = _rms_norm_strided_quant(x, norm, torch.int8)
    return Int8PerTokenInput(*result) if result is not None else None


def rms_norm_strided_fp8_operand(x: torch.Tensor, norm):
    """FP8 sibling of the same strided QNorm preparation."""
    result = _rms_norm_strided_quant(x, norm, torch.float8_e4m3fn)
    if result is None:
        return None
    normalized, q, scale = result
    return normalized, (q, scale, x.dtype)
