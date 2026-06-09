from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)
from sglang.srt.utils import is_dcu

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


class AiterQuantType(str, Enum):
    NONE = "No"
    PER_TOKEN = "per_Token"
    PER_128X128 = "per_128x128"
    PER_1X32 = "per_1x32"


_is_dcu = is_dcu()


@dataclass
class AiterMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    quant_type: AiterQuantType = AiterQuantType.NONE
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    a13_scale: Optional[torch.Tensor] = None
    a2_scale: Optional[torch.Tensor] = None
    b13: Optional[torch.Tensor] = None
    b2: Optional[torch.Tensor] = None
    expert_mask: Optional[torch.Tensor] = None
    doweight_stage1: bool = False
    hidden_pad: int = 0
    intermediate_pad: int = 0
    use_int8_w8a8: bool = False
    use_fp8_w8a8: bool = False
    global_num_experts: Optional[int] = None
    expert_map: Optional[torch.Tensor] = None
    moe_config_cache: Optional[dict] = None
    weight_cache: Optional[dict] = None
    moe_c_weight_layout: bool = False
    layer: Optional[torch.nn.Module] = None


_AITER_ACTIVATIONS = {"silu": "Silu", "swiglu": "Swiglu"}


def process_weights_after_loading_aiter_w8a8_int8(layer: torch.nn.Module) -> None:
    try:
        from aiter.moe import (  # noqa: F401
            MoeQuantType,
            aiter_moe,
            get_aiter_moe_config,
        )
        from aiter.ops.shuffle import (  # noqa: F401
            moe_layout_shuffle_gemm1,
            moe_layout_shuffle_gemm2,
        )
    except Exception as exc:
        raise RuntimeError(
            "AITER W8A8 INT8 MoE is enabled but required aiter modules are "
            "unavailable."
        ) from exc

    if not hasattr(MoeQuantType, "W8A8"):
        raise RuntimeError(
            "The installed aiter package does not expose MoeQuantType.W8A8."
        )
    moe_runner_config = getattr(layer, "moe_runner_config", None)
    if getattr(layer, "apply_router_weight_on_input", False) or (
        moe_runner_config is not None
        and moe_runner_config.apply_router_weight_on_input
    ):
        raise RuntimeError(
            "AITER W8A8 INT8 MoE does not support "
            "apply_router_weight_on_input=True."
        )

    with torch.no_grad():
        w13_weight = moe_layout_shuffle_gemm1(layer.w13_weight).view(
            *layer.w13_weight.shape
        )
    layer.w13_weight = Parameter(w13_weight, requires_grad=False)

    with torch.no_grad():
        w2_weight = moe_layout_shuffle_gemm2(layer.w2_weight).view(
            *layer.w2_weight.shape
        )
    layer.w2_weight = Parameter(w2_weight, requires_grad=False)

    setattr(layer, "_aiter_w8a8_int8_moe_c_w13_weight", layer.w13_weight)
    setattr(layer, "_aiter_w8a8_int8_moe_c_w2_weight", layer.w2_weight)
    setattr(layer, "_aiter_w8a8_int8_moe_config_cache", {})
    setattr(layer, "_aiter_w8a8_int8_weight_cache", {})
    setattr(layer, "_aiter_w8a8_int8_moe_c_weight_layout", True)


def process_weights_after_loading_aiter_w8a8_fp8(layer: torch.nn.Module) -> None:
    try:
        from aiter.moe import (  # noqa: F401
            MoeQuantType,
            aiter_moe,
            get_aiter_moe_config,
        )
        from aiter.ops.shuffle import (  # noqa: F401
            moe_layout_shuffle_gemm1,
            moe_layout_shuffle_gemm2,
        )
    except Exception as exc:
        raise RuntimeError(
            "AITER FP8 W8A8 MoE is enabled but required aiter modules are "
            "unavailable."
        ) from exc

    if not hasattr(MoeQuantType, "FP8_W8A8"):
        raise RuntimeError(
            "The installed aiter package does not expose MoeQuantType.FP8_W8A8."
        )
    moe_runner_config = getattr(layer, "moe_runner_config", None)
    if getattr(layer, "apply_router_weight_on_input", False) or (
        moe_runner_config is not None
        and moe_runner_config.apply_router_weight_on_input
    ):
        raise RuntimeError(
            "AITER FP8 W8A8 MoE does not support "
            "apply_router_weight_on_input=True."
        )

    with torch.no_grad():
        w13_weight = moe_layout_shuffle_gemm1(layer.w13_weight).view(
            *layer.w13_weight.shape
        )
    layer.w13_weight = Parameter(w13_weight, requires_grad=False)

    with torch.no_grad():
        w2_weight = moe_layout_shuffle_gemm2(layer.w2_weight).view(
            *layer.w2_weight.shape
        )
    layer.w2_weight = Parameter(w2_weight, requires_grad=False)

    setattr(layer, "_aiter_w8a8_fp8_moe_c_w13_weight", layer.w13_weight)
    setattr(layer, "_aiter_w8a8_fp8_moe_c_w2_weight", layer.w2_weight)
    setattr(layer, "_aiter_w8a8_fp8_moe_config_cache", {})
    setattr(layer, "_aiter_w8a8_fp8_weight_cache", {})
    setattr(layer, "_aiter_w8a8_fp8_moe_c_weight_layout", True)


def get_aiter_w8a8_int8_quant_info(layer: torch.nn.Module) -> AiterMoeQuantInfo:
    dispatcher = getattr(layer, "dispatcher", None)
    expert_map = (
        getattr(dispatcher, "local_expert_mapping", None)
        if getattr(dispatcher, "expert_mask_gpu", None) is not None
        else None
    )

    return AiterMoeQuantInfo(
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        w13_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        a13_scale=layer.w13_input_scale,
        a2_scale=layer.w2_input_scale,
        use_int8_w8a8=True,
        global_num_experts=getattr(layer, "num_experts", None),
        expert_map=expert_map,
        moe_config_cache=getattr(layer, "_aiter_w8a8_int8_moe_config_cache", None),
        weight_cache=getattr(layer, "_aiter_w8a8_int8_weight_cache", None),
        moe_c_weight_layout=getattr(
            layer, "_aiter_w8a8_int8_moe_c_weight_layout", False
        ),
        layer=layer,
    )


def get_aiter_w8a8_fp8_quant_info(layer: torch.nn.Module) -> AiterMoeQuantInfo:
    dispatcher = getattr(layer, "dispatcher", None)
    expert_map = (
        getattr(dispatcher, "local_expert_mapping", None)
        if getattr(dispatcher, "expert_mask_gpu", None) is not None
        else None
    )

    return AiterMoeQuantInfo(
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        w13_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        a13_scale=layer.w13_input_scale,
        a2_scale=layer.w2_input_scale,
        use_fp8_w8a8=True,
        global_num_experts=getattr(layer, "num_experts", None),
        expert_map=expert_map,
        moe_config_cache=getattr(layer, "_aiter_w8a8_fp8_moe_config_cache", None),
        weight_cache=getattr(layer, "_aiter_w8a8_fp8_weight_cache", None),
        moe_c_weight_layout=getattr(
            layer, "_aiter_w8a8_fp8_moe_c_weight_layout", False
        ),
        layer=layer,
    )


def _get_aiter_w8a8_quant_type(use_fp8_w8a8: bool = False):
    from aiter.moe import MoeQuantType

    quant_type_name = "FP8_W8A8" if use_fp8_w8a8 else "W8A8"
    quant_type = getattr(MoeQuantType, quant_type_name, None)
    if quant_type is None:
        raise RuntimeError(
            f"The installed aiter package does not expose MoeQuantType.{quant_type_name}."
        )
    return quant_type


def _get_aiter_w8a8_moe_config(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    quant_info: AiterMoeQuantInfo,
):
    from aiter.moe import AiterMoeConfig, MoeSolutionType, get_aiter_moe_config

    if hidden_states.dim() != 2:
        raise RuntimeError(
            "AITER W8A8 MoE expects 2D hidden_states, got "
            f"shape={tuple(hidden_states.shape)}."
        )
    if w1.dim() != 3 or w2.dim() != 3:
        raise RuntimeError(
            "AITER W8A8 MoE expects 3D expert weights, got "
            f"w1.shape={tuple(w1.shape)}, w2.shape={tuple(w2.shape)}."
        )

    M, K = hidden_states.shape
    E, N1, K1 = w1.shape
    E2, N2, _ = w2.shape
    if E != E2 or K != K1 or K != N2:
        raise RuntimeError(
            "AITER W8A8 MoE shape mismatch: "
            f"hidden_states={tuple(hidden_states.shape)}, "
            f"w1={tuple(w1.shape)}, w2={tuple(w2.shape)}."
        )

    top_k = topk_ids.shape[1]
    cache_key = (M, top_k, hidden_states.dtype, activation)
    if quant_info.moe_config_cache is not None:
        moe_config = quant_info.moe_config_cache.get(cache_key)
        if moe_config is not None:
            return moe_config

    quant_type = _get_aiter_w8a8_quant_type(quant_info.use_fp8_w8a8)
    config_kwargs = dict(
        M=M,
        E=E,
        N1=N1,
        N2=N2,
        K=K,
        top_k=top_k,
        block_size=0,
        dtype=hidden_states.dtype,
        quant_type=quant_type,
        activation=activation
    )
    try:
        status, moe_config = get_aiter_moe_config(**config_kwargs)
    except TypeError:
        config_kwargs.pop("activation", None)
        status, moe_config = get_aiter_moe_config(**config_kwargs)

    if not status:
        raise RuntimeError(
            "AITER W8A8 MoE did not find a valid backend config: "
            f"M={M}, N1={N1}, N2={N2}, K={K}, E={E}, topk={top_k}, "
            f"dtype={hidden_states.dtype}."
        )

    allowed_solution_types = {
        MoeSolutionType.MOE_C,
        MoeSolutionType.ASM,
        MoeSolutionType.TRITON,
        MoeSolutionType.CK,
    }
    if moe_config.solution_type not in allowed_solution_types:
        raise RuntimeError(
            f"Unsupported AITER MoE solution_type: {moe_config.solution_type}"
        )
    if moe_config.quant_type != quant_type:
        raise RuntimeError(f"Unexpected AITER MoE quant_type: {moe_config.quant_type}")

    if quant_info.moe_config_cache is not None:
        quant_info.moe_config_cache[cache_key] = moe_config
    return moe_config


def _get_aiter_w8a8_weights_for_solution(
    quant_info: AiterMoeQuantInfo,
    solution_type,
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.moe import MoeSolutionType
    from aiter.ops.shuffle import moe_layout_shuffle_gemm1, moe_layout_shuffle_gemm2

    if solution_type != MoeSolutionType.MOE_C:
        if quant_info.moe_c_weight_layout:
            raise RuntimeError(
                "AITER W8A8 weights were converted to MOE_C layout during "
                "weight loading, but AITER selected a non-MOE_C solution: "
                f"{solution_type}."
            )
        return quant_info.w13_weight, quant_info.w2_weight

    if quant_info.moe_c_weight_layout:
        return quant_info.w13_weight, quant_info.w2_weight

    cache_prefix = (
        "_aiter_w8a8_fp8" if quant_info.use_fp8_w8a8 else "_aiter_w8a8_int8"
    )
    layer = quant_info.layer
    if layer is not None:
        w1_moe_c = getattr(layer, f"{cache_prefix}_moe_c_w13_weight", None)
        w2_moe_c = getattr(layer, f"{cache_prefix}_moe_c_w2_weight", None)
        if w1_moe_c is not None and w2_moe_c is not None:
            return w1_moe_c, w2_moe_c

    if quant_info.weight_cache is not None:
        w1_moe_c = quant_info.weight_cache.get("moe_c_w13_weight")
        w2_moe_c = quant_info.weight_cache.get("moe_c_w2_weight")
        if w1_moe_c is not None and w2_moe_c is not None:
            return w1_moe_c, w2_moe_c

    with torch.no_grad():
        w1_moe_c = moe_layout_shuffle_gemm1(quant_info.w13_weight).view(
            *quant_info.w13_weight.shape
        )
        w2_moe_c = moe_layout_shuffle_gemm2(quant_info.w2_weight).view(
            *quant_info.w2_weight.shape
        )

    if layer is not None:
        setattr(layer, f"{cache_prefix}_moe_c_w13_weight", w1_moe_c)
        setattr(layer, f"{cache_prefix}_moe_c_w2_weight", w2_moe_c)
    if quant_info.weight_cache is not None:
        quant_info.weight_cache["moe_c_w13_weight"] = w1_moe_c
        quant_info.weight_cache["moe_c_w2_weight"] = w2_moe_c
    return w1_moe_c, w2_moe_c


def _fused_experts_none_to_aiter_w8a8(
    dispatch_output: StandardDispatchOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from aiter.moe import aiter_moe

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    assert not runner_config.no_combine, "no_combine=True is not supported by AITER"
    if runner_config.apply_router_weight_on_input:
        raise RuntimeError(
            "AITER W8A8 MoE does not support "
            "apply_router_weight_on_input=True."
        )

    hidden_states = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    activation = str(runner_config.activation)
    moe_config = _get_aiter_w8a8_moe_config(
        hidden_states,
        quant_info.w13_weight,
        quant_info.w2_weight,
        topk_ids,
        activation,
        quant_info,
    )
    w1, w2 = _get_aiter_w8a8_weights_for_solution(
        quant_info, moe_config.solution_type
    )
    routed_scaling_factor = (
        runner_config.routed_scaling_factor
        if runner_config.routed_scaling_factor is not None
        else 1.0
    )
    if quant_info.expert_map is not None:
        global_num_experts = quant_info.global_num_experts or w1.shape[0]
    else:
        global_num_experts = w1.shape[0]

    output = aiter_moe(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights.to(torch.float32),
        topk_ids=topk_ids.to(torch.int32),
        moe_config=moe_config,
        inplace=runner_config.inplace,
        activation=activation,
        w1_scale=quant_info.w13_scale,
        w2_scale=quant_info.w2_scale,
        w1_zp=None,
        w2_zp=None,
        a1_scale=quant_info.a13_scale,
        a2_scale=quant_info.a2_scale,
        block_shape=None,
        global_num_experts=global_num_experts,
        expert_map=quant_info.expert_map,
        routed_scaling_factor=float(routed_scaling_factor),
        output_dtype=hidden_states.dtype,
        gemm1_alpha=runner_config.gemm1_alpha,
        gemm1_limit=runner_config.gemm1_clamp_limit,
    )
    return StandardCombineInput(hidden_states=output)


@register_fused_func("none", "aiter")
def fused_experts_none_to_aiter(
    dispatch_output: StandardDispatchOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    if quant_info.use_int8_w8a8 or quant_info.use_fp8_w8a8:
        if _is_dcu:
            return _fused_experts_none_to_aiter_w8a8(
                dispatch_output,
                quant_info,
                runner_config,
            )
        raise RuntimeError(
            "AITER W8A8 MoE is only supported on DCU. "
            "Use the native AITER path for other quantization modes."
        )

    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    assert not runner_config.no_combine, "no_combine=True is not supported by AITER"

    hidden_states = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    topk_weights = topk_weights.to(torch.float32)

    if runner_config.apply_router_weight_on_input and not quant_info.doweight_stage1:
        # Pre-scale at the Python level for kernels that don't honor doweight_stage1.
        assert (
            topk_weights.dim() == 2 and topk_weights.shape[-1] == 1
        ), "apply_router_weight_on_input requires topk=1"
        hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
        topk_weights = torch.ones_like(topk_weights)

    activation = runner_config.activation
    output = fused_moe(
        hidden_states=hidden_states,
        w1=quant_info.w13_weight,
        w2=quant_info.w2_weight,
        topk_weight=topk_weights,
        topk_ids=topk_ids.to(torch.int32),
        quant_type=getattr(QuantType, quant_info.quant_type.value),
        activation=getattr(ActivationType, _AITER_ACTIVATIONS.get(activation, "Gelu")),
        w1_scale=quant_info.w13_scale,
        w2_scale=quant_info.w2_scale,
        a1_scale=quant_info.a13_scale,
        a2_scale=quant_info.a2_scale,
        bias1=quant_info.b13,
        bias2=quant_info.b2,
        expert_mask=quant_info.expert_mask,
        doweight_stage1=quant_info.doweight_stage1,
        hidden_pad=quant_info.hidden_pad,
        intermediate_pad=quant_info.intermediate_pad,
    )
    return StandardCombineInput(hidden_states=output)
