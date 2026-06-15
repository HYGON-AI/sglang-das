from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
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
    moe_c_weight_layout: bool = False
    original_w13_shape: Optional[tuple[int, ...]] = None
    original_w2_shape: Optional[tuple[int, ...]] = None
    layer: Optional[torch.nn.Module] = None


_AITER_ACTIVATIONS = {"silu": "Silu", "swiglu": "Swiglu"}


@dataclass
class AiterRunnerInput(RunnerInput):
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.AITER


@dataclass
class AiterRunnerOutput(RunnerOutput):
    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.AITER


def process_weights_after_loading_aiter_w8a8_int8(layer: torch.nn.Module) -> None:

    moe_runner_config = getattr(layer, "moe_runner_config", None)
    if getattr(layer, "apply_router_weight_on_input", False) or (
        moe_runner_config is not None
        and moe_runner_config.apply_router_weight_on_input
    ):
        raise RuntimeError(
            "AITER W8A8 INT8 MoE does not support "
            "apply_router_weight_on_input=True."
        )

    setattr(layer, "_aiter_w8a8_int8_original_w13_shape", tuple(layer.w13_weight.shape))
    setattr(layer, "_aiter_w8a8_int8_original_w2_shape", tuple(layer.w2_weight.shape))
    setattr(layer, "_aiter_w8a8_int8_moe_config_cache", {})
    setattr(layer, "_aiter_w8a8_int8_moe_c_weight_layout", False)


def process_weights_after_loading_aiter_w8a8_fp8(layer: torch.nn.Module) -> None:

    moe_runner_config = getattr(layer, "moe_runner_config", None)
    if getattr(layer, "apply_router_weight_on_input", False) or (
        moe_runner_config is not None
        and moe_runner_config.apply_router_weight_on_input
    ):
        raise RuntimeError(
            "AITER FP8 W8A8 MoE does not support "
            "apply_router_weight_on_input=True."
        )

    setattr(layer, "_aiter_w8a8_fp8_original_w13_shape", tuple(layer.w13_weight.shape))
    setattr(layer, "_aiter_w8a8_fp8_original_w2_shape", tuple(layer.w2_weight.shape))
    setattr(layer, "_aiter_w8a8_fp8_moe_config_cache", {})
    setattr(layer, "_aiter_w8a8_fp8_moe_c_weight_layout", False)


def get_aiter_w8a8_int8_quant_info(layer: torch.nn.Module) -> AiterMoeQuantInfo:
    dispatcher = getattr(layer, "dispatcher", None)
    expert_map = (
        getattr(dispatcher, "local_expert_mapping", None)
        if getattr(dispatcher, "expert_mask_gpu", None) is not None
        else None
    )
    if not hasattr(layer, "_aiter_w8a8_int8_original_w13_shape"):
        setattr(
            layer,
            "_aiter_w8a8_int8_original_w13_shape",
            tuple(layer.w13_weight.shape),
        )
    if not hasattr(layer, "_aiter_w8a8_int8_original_w2_shape"):
        setattr(
            layer,
            "_aiter_w8a8_int8_original_w2_shape",
            tuple(layer.w2_weight.shape),
        )
    if not hasattr(layer, "_aiter_w8a8_int8_moe_config_cache"):
        setattr(layer, "_aiter_w8a8_int8_moe_config_cache", {})
    if not hasattr(layer, "_aiter_w8a8_int8_moe_c_weight_layout"):
        setattr(layer, "_aiter_w8a8_int8_moe_c_weight_layout", False)

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
        moe_c_weight_layout=getattr(
            layer, "_aiter_w8a8_int8_moe_c_weight_layout", False
        ),
        original_w13_shape=getattr(
            layer, "_aiter_w8a8_int8_original_w13_shape", tuple(layer.w13_weight.shape)
        ),
        original_w2_shape=getattr(
            layer, "_aiter_w8a8_int8_original_w2_shape", tuple(layer.w2_weight.shape)
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
    if not hasattr(layer, "_aiter_w8a8_fp8_original_w13_shape"):
        setattr(
            layer,
            "_aiter_w8a8_fp8_original_w13_shape",
            tuple(layer.w13_weight.shape),
        )
    if not hasattr(layer, "_aiter_w8a8_fp8_original_w2_shape"):
        setattr(
            layer,
            "_aiter_w8a8_fp8_original_w2_shape",
            tuple(layer.w2_weight.shape),
        )
    if not hasattr(layer, "_aiter_w8a8_fp8_moe_config_cache"):
        setattr(layer, "_aiter_w8a8_fp8_moe_config_cache", {})
    if not hasattr(layer, "_aiter_w8a8_fp8_moe_c_weight_layout"):
        setattr(layer, "_aiter_w8a8_fp8_moe_c_weight_layout", False)

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
        moe_c_weight_layout=getattr(
            layer, "_aiter_w8a8_fp8_moe_c_weight_layout", False
        ),
        original_w13_shape=getattr(
            layer, "_aiter_w8a8_fp8_original_w13_shape", tuple(layer.w13_weight.shape)
        ),
        original_w2_shape=getattr(
            layer, "_aiter_w8a8_fp8_original_w2_shape", tuple(layer.w2_weight.shape)
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


def _get_aiter_w8a8_original_dims(
    hidden_states: torch.Tensor,
    quant_info: AiterMoeQuantInfo,
) -> tuple[int, int, int, int]:
    _, K = hidden_states.shape
    w1_shape = quant_info.original_w13_shape
    w2_shape = quant_info.original_w2_shape
    E, N1, K1 = w1_shape
    E2, N2, _ = w2_shape
    if E != E2 or K != K1 or K != N2:
        raise RuntimeError(
            "AITER W8A8 MoE shape mismatch: "
            f"hidden_states={tuple(hidden_states.shape)}, "
            f"w1_original={tuple(w1_shape)}, w2_original={tuple(w2_shape)}, "
            f"w1_current={tuple(quant_info.w13_weight.shape)}, "
            f"w2_current={tuple(quant_info.w2_weight.shape)}."
        )
    return E, N1, N2, K


def _get_aiter_w8a8_moe_config(
    hidden_states: torch.Tensor,
    E: int,
    N1: int,
    N2: int,
    K: int,
    topk_ids: torch.Tensor,
    activation: str,
    quant_info: AiterMoeQuantInfo,
):
    from aiter.moe import MoeSolutionType, get_aiter_moe_config

    if hidden_states.dim() != 2:
        raise RuntimeError(
            "AITER W8A8 MoE expects 2D hidden_states, got "
            f"shape={tuple(hidden_states.shape)}."
        )
    M, hidden_size = hidden_states.shape
    if hidden_size != K:
        raise RuntimeError(
            "AITER W8A8 MoE shape mismatch: "
            f"hidden_states={tuple(hidden_states.shape)}, K={K}."
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
        activation=activation,
    )

    try:
        status, moe_config = get_aiter_moe_config(**config_kwargs)
    except TypeError:
        config_kwargs.pop("activation", None)
        status, moe_config = get_aiter_moe_config(**config_kwargs)

    if os.environ.get("PRINT_MOE_ARGS", "0") == "1":
        print(
            "AITER W8A8 MoE args: "
            f"moe_config={moe_config}, "
            f"M={M}, N1={N1}, N2={N2}, K={K}, E={E}, topk={top_k}",
            flush=True,
        )

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
    moe_config,
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.moe import MoeSolutionType
    from aiter.ops.shuffle import moe_layout_shuffle_gemm1, moe_layout_shuffle_gemm2

    solution_type = moe_config.solution_type
    need_shuffle = getattr(
        moe_config, "need_shuffle", solution_type == MoeSolutionType.MOE_C
    )

    if not need_shuffle:
        if quant_info.moe_c_weight_layout:
            raise RuntimeError(
                "AITER W8A8 weights were converted to shuffled layout, but "
                "AITER selected a config that does not need shuffled weights: "
                f"{solution_type}."
            )
        return quant_info.w13_weight, quant_info.w2_weight

    if quant_info.moe_c_weight_layout:
        return quant_info.w13_weight, quant_info.w2_weight

    cache_prefix = (
        "_aiter_w8a8_fp8" if quant_info.use_fp8_w8a8 else "_aiter_w8a8_int8"
    )
    layer = quant_info.layer

    with torch.no_grad():
        w1_moe_c = moe_layout_shuffle_gemm1(quant_info.w13_weight).view(
            *quant_info.w13_weight.shape
        )
        w2_moe_c = moe_layout_shuffle_gemm2(quant_info.w2_weight).view(
            *quant_info.w2_weight.shape
        )

    if layer is not None:
        layer.w13_weight = Parameter(w1_moe_c, requires_grad=False)
        layer.w2_weight = Parameter(w2_moe_c, requires_grad=False)
        setattr(layer, f"{cache_prefix}_moe_c_weight_layout", True)
    quant_info.w13_weight = layer.w13_weight if layer is not None else w1_moe_c
    quant_info.w2_weight = layer.w2_weight if layer is not None else w2_moe_c
    quant_info.moe_c_weight_layout = True
    return quant_info.w13_weight, quant_info.w2_weight


def _run_aiter_w8a8(
    runner_input: AiterRunnerInput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> AiterRunnerOutput:
    from aiter.moe import aiter_moe

    assert not runner_config.no_combine, "no_combine=True is not supported by AITER"
    if runner_config.apply_router_weight_on_input:
        raise RuntimeError(
            "AITER W8A8 MoE does not support "
            "apply_router_weight_on_input=True."
        )

    hidden_states = runner_input.hidden_states
    topk_weights = runner_input.topk_weights
    topk_ids = runner_input.topk_ids
    activation = str(runner_config.activation)
    E, N1, N2, K = _get_aiter_w8a8_original_dims(hidden_states, quant_info)
    moe_config = _get_aiter_w8a8_moe_config(
        hidden_states,
        E,
        N1,
        N2,
        K,
        topk_ids,
        activation,
        quant_info,
    )
    w1, w2 = _get_aiter_w8a8_weights_for_solution(
        quant_info, moe_config
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
    return AiterRunnerOutput(hidden_states=output)


def _run_aiter_native(
    runner_input: AiterRunnerInput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> AiterRunnerOutput:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    assert not runner_config.no_combine, "no_combine=True is not supported by AITER"

    hidden_states = runner_input.hidden_states
    topk_weights = runner_input.topk_weights
    topk_ids = runner_input.topk_ids
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
    return AiterRunnerOutput(hidden_states=output)


class AiterRunnerCore(MoeRunnerCore):
    def run(
        self,
        runner_input: AiterRunnerInput,
        quant_info: AiterMoeQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> AiterRunnerOutput:
        assert hooks is None, "AITER MoE does not support LoRA hooks."

        if quant_info.use_int8_w8a8 or quant_info.use_fp8_w8a8:
            if _is_dcu:
                return _run_aiter_w8a8(runner_input, quant_info, self.config)
            raise RuntimeError(
                "AITER W8A8 MoE is only supported on DCU. "
                "Use the native AITER path for other quantization modes."
            )

        return _run_aiter_native(runner_input, quant_info, self.config)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.AITER


@register_pre_permute("standard", "aiter")
def pre_permute_standard_to_aiter(
    dispatch_output: StandardDispatchOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AiterRunnerInput:
    hidden_states = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    return AiterRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )


@register_post_permute("aiter", "standard")
def post_permute_aiter_to_standard(
    runner_output: AiterRunnerOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(hidden_states=runner_output.hidden_states)
