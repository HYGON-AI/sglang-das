from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional, Tuple

import torch


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

    if (
        input.numel() == 0
        or input.dim() != 2
        or not input.is_cuda
        or not is_lightop_sglang_rms_quant_available()
    ):
        return False
    if input.dtype not in (torch.float16, torch.bfloat16):
        return False
    if (
        weight.dtype != input.dtype
        or weight.device != input.device
        or weight.shape != (input.shape[-1],)
    ):
        return False
    if input.shape[-1] % 16 != 0 or input.shape[-1] > 8192:
        return False
    if not input.is_contiguous() or not weight.is_contiguous():
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
