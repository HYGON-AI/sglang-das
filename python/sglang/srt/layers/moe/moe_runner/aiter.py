# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Hygon modifications to this file are licensed under the Apache License,
# Version 2.0 (the "License"); you may not use these modifications except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import functools
import inspect
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Union

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
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_bool_env_var, get_int_env_var, is_hcu

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.base import CombineInput
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLDispatchOutput,
        DeepEPNormalDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.moriep import (
        MoriEPLLDispatchOutput,
        MoriEPNormalDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


logger = logging.getLogger(__name__)


class AiterQuantType(str, Enum):
    NONE = "No"
    PER_TOKEN = "per_Token"
    PER_128X128 = "per_128x128"
    PER_1X32 = "per_1x32"


_is_hcu = is_hcu()


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
    swiglu_limit: float = 0.0
    use_int8_w8a8: bool = False
    use_fp8_w8a8: bool = False
    global_num_experts: Optional[int] = None
    expert_map: Optional[torch.Tensor] = None
    moe_config_cache: Optional[dict] = None
    moe_c_weight_layout: bool = False
    original_w13_shape: Optional[tuple[int, ...]] = None
    original_w2_shape: Optional[tuple[int, ...]] = None
    layer: Optional[torch.nn.Module] = None
    fused_moe_kwargs: Optional[dict[str, Any]] = None


# `AiterRunnerInput` / `AiterRunnerOutput` keep the HCU ordering and are
# defined below, after the activation/quant-type helpers they depend on.
_AITER_ACTIVATIONS = {
    "silu": "Silu",
    "swiglu": "Swiglu",
    "situ": "Situv2",
}


def _aiter_activation(activation: str):
    from aiter import ActivationType

    return getattr(ActivationType, _AITER_ACTIVATIONS.get(activation, "Gelu"))


def _aiter_quant_type(quant_type: AiterQuantType):
    from aiter import QuantType

    return getattr(QuantType, quant_type.value)


@dataclass
class AiterRunnerInput(RunnerInput):
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    quant_type: AiterQuantType = AiterQuantType.NONE
    a1_scale: Optional[torch.Tensor] = None
    num_local_tokens: Optional[torch.Tensor] = None
    output_dtype: Optional[torch.dtype] = None

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
        moe_runner_config is not None and moe_runner_config.apply_router_weight_on_input
    ):
        raise RuntimeError(
            "AITER W8A8 INT8 MoE does not support apply_router_weight_on_input=True."
        )

    setattr(layer, "_aiter_w8a8_int8_original_w13_shape", tuple(layer.w13_weight.shape))
    setattr(layer, "_aiter_w8a8_int8_original_w2_shape", tuple(layer.w2_weight.shape))
    setattr(layer, "_aiter_w8a8_int8_moe_config_cache", {})
    setattr(layer, "_aiter_w8a8_int8_moe_c_weight_layout", False)


def process_weights_after_loading_aiter_w8a8_fp8(layer: torch.nn.Module) -> None:

    moe_runner_config = getattr(layer, "moe_runner_config", None)
    if getattr(layer, "apply_router_weight_on_input", False) or (
        moe_runner_config is not None and moe_runner_config.apply_router_weight_on_input
    ):
        raise RuntimeError(
            "AITER FP8 W8A8 MoE does not support apply_router_weight_on_input=True."
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
    from aiter.ops.shuffle import moe_layout_shuffle_gemm2

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

    cache_prefix = "_aiter_w8a8_fp8" if quant_info.use_fp8_w8a8 else "_aiter_w8a8_int8"
    layer = quant_info.layer

    with torch.no_grad():
        w1_moe_c = moe_layout_shuffle_gemm2(quant_info.w13_weight).view(
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
            "AITER W8A8 MoE does not support apply_router_weight_on_input=True."
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
    w1, w2 = _get_aiter_w8a8_weights_for_solution(quant_info, moe_config)
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


@functools.cache
def _aiter_fused_moe_supports_no_combine() -> bool:
    """Return whether the installed AITER fused_moe supports no_combine."""
    from aiter.fused_moe import fused_moe

    return "no_combine" in inspect.signature(fused_moe).parameters


_RECV_BOUND_LOGGED: set[int] = set()
_RECV_BOUND_WARNED = False


def _warn_recv_bound_unavailable() -> None:
    global _RECV_BOUND_WARNED
    if not _RECV_BOUND_WARNED:
        _RECV_BOUND_WARNED = True
        logger.warning(
            "SGLANG_MORI_RECV_BOUND is set but the per-rank DP token counts do "
            "not cover every mori sender, so the receive fan-in is unknown; "
            "leaving the receive buffer unbounded."
        )


def _mori_decode_recv_bound(recv_rows: int, topk: int) -> int:
    """Live rows mori's receive buffer can hold in decode, or 0 for "do not bound".

    Worst case fan-in is every rank routing all of its tokens to this one, so
    `sum(per-rank tokens) * topk`, where topk already includes the fused shared
    expert. The per-rank counts come from the DP sync, so this is the fan-in for
    the batch actually being run rather than an upper bound over all batches.

    That is only sound because enabling this gate also makes
    `require_mlp_tp_gather()` true for mori, which gives every rank the same
    cuda-graph bucket. The value is baked into a captured graph and has to hold
    for every later replay; with per-rank buckets a rank on a narrow tier could
    be handed rows by a peer on a wider one, and the only bound valid under that
    is the widest tier's -- 4-16x looser than the batch being run, which costs
    more in expert-GEMM tiles (M 32/64 -> 128) than the trim saves.

    Two cases stay unbounded, because a bound below the real fan-in silently
    drops rows from the all-to-all -- wrong output rather than an error:

    * Prefill, whose per-rank counts are uneven and not knowable here.
    * Anything that leaves the per-rank counts unpopulated, or where the EP world
      is wider than the DP world so the counts do not cover every sender.
    """
    if not get_bool_env_var("SGLANG_MORI_RECV_BOUND", "false"):
        return 0

    from sglang.srt.layers.dp_attention import (
        get_dp_global_num_tokens,
        get_is_extend_in_batch,
    )

    if get_is_extend_in_batch():
        return 0

    per_rank_tokens = get_dp_global_num_tokens()
    ep_size = get_parallel().moe_ep_size
    if not per_rank_tokens or len(per_rank_tokens) < ep_size:
        # Either the DP sync did not publish counts, or they do not cover every
        # mori sender. Both mean the fan-in is unknown here.
        _warn_recv_bound_unavailable()
        return 0

    max_tokens = sum(per_rank_tokens)
    bound = max_tokens * topk
    # Never grow the tensor, and nothing to do when there is nothing to trim.
    if not 0 < bound < recv_rows:
        return 0

    # One INFO line the first time it engages, so an inert bound is not mistaken
    # for an active one in the results. Per-tier values go to DEBUG: capture
    # visits every tier, and at INFO on every rank that is dozens of lines.
    if get_parallel().tp_rank == 0 and bound not in _RECV_BOUND_LOGGED:
        first = not _RECV_BOUND_LOGGED
        _RECV_BOUND_LOGGED.add(bound)
        if first:
            logger.info(
                "mori recv bound active: %d rows -> %d for this tier "
                "(dp_tokens=%d ep=%d topk=%d); per-tier values at DEBUG",
                recv_rows,
                bound,
                max_tokens,
                get_parallel().moe_ep_size,
                topk,
            )
        else:
            logger.debug(
                "mori recv bound: %d rows -> %d (dp_tokens=%d ep=%d topk=%d)",
                recv_rows,
                bound,
                max_tokens,
                get_parallel().moe_ep_size,
                topk,
            )
    return bound


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
            if _is_hcu:
                return _run_aiter_w8a8(runner_input, quant_info, self.config)
            raise RuntimeError(
                "AITER W8A8 MoE is only supported on HCU. "
                "Use the native AITER path for other quantization modes."
            )

        if _is_hcu:
            return _run_aiter_native(runner_input, quant_info, self.config)

        if self.config.no_combine and not _aiter_fused_moe_supports_no_combine():
            raise NotImplementedError(
                "no_combine=True requested but the installed aiter.fused_moe does "
                "not accept a `no_combine` kwarg. Install an aiter build that "
                "supports fused_moe no_combine output."
            )

        if runner_input.hidden_states.shape[0] == 0:
            if self.config.no_combine:
                topk = runner_input.topk_ids.shape[-1]
                hidden_size = runner_input.hidden_states.shape[-1]
                return AiterRunnerOutput(
                    hidden_states=runner_input.hidden_states.new_empty(
                        (0, topk, hidden_size)
                    )
                )
            return AiterRunnerOutput(hidden_states=runner_input.hidden_states)

        from aiter.fused_moe import fused_moe

        from sglang.srt.environ import envs

        a1_scale = (
            runner_input.a1_scale
            if runner_input.a1_scale is not None
            else quant_info.a13_scale
        )

        extra: dict = {}
        if quant_info.fused_moe_kwargs:
            extra.update(quant_info.fused_moe_kwargs)
        if runner_input.num_local_tokens is not None:
            extra["num_local_tokens"] = runner_input.num_local_tokens
        if runner_input.output_dtype is not None:
            extra["dtype"] = runner_input.output_dtype
        if self.config.activation == "situ":
            from aiter.ops.flydsl.moe_common import GateMode

            extra["gate_mode"] = GateMode.SEPARATED.value
            if self.config.gemm1_alpha is not None:
                extra["beta"] = float(self.config.gemm1_alpha)
            if self.config.gemm1_clamp_limit is not None:
                extra["linear_beta"] = float(self.config.gemm1_clamp_limit)
        elif quant_info.swiglu_limit > 0:
            # GateMode is only needed for the gpt-oss MXFP4 swiglu_limit path.
            # Import lazily so models that don't use it (e.g. DeepSeek-V3 fp8,
            # swiglu_limit==0) still run on aiter builds where this module
            # lives elsewhere / is absent.
            from aiter.ops.flydsl.moe_common import GateMode

            # Default (INTERLEAVE) preserves the pre-fix behavior for paths
            # that prepare weights in the gate/up-interleaved layout. Set
            # `SGLANG_USE_AITER_MOE_GU_ITLV=0` to switch to SEPARATED, which
            # matches the layout produced by `Mxfp4MoEMethod` (gpt-oss
            # MXFP4) and the gptoss_fp4 tuned FlyDSL kernels.
            extra["gate_mode"] = (
                GateMode.INTERLEAVE.value
                if envs.SGLANG_USE_AITER_MOE_GU_ITLV.get()
                else GateMode.SEPARATED.value
            )
            extra["swiglu_limit"] = quant_info.swiglu_limit
        if self.config.no_combine:
            extra["no_combine"] = True

        output = fused_moe(
            hidden_states=runner_input.hidden_states,
            w1=quant_info.w13_weight,
            w2=quant_info.w2_weight,
            topk_weight=runner_input.topk_weights,
            topk_ids=runner_input.topk_ids,
            quant_type=_aiter_quant_type(runner_input.quant_type),
            activation=_aiter_activation(self.config.activation),
            w1_scale=quant_info.w13_scale,
            w2_scale=quant_info.w2_scale,
            a1_scale=a1_scale,
            a2_scale=quant_info.a2_scale,
            bias1=quant_info.b13,
            bias2=quant_info.b2,
            expert_mask=quant_info.expert_mask,
            doweight_stage1=quant_info.doweight_stage1,
            hidden_pad=quant_info.hidden_pad,
            intermediate_pad=quant_info.intermediate_pad,
            **extra,
        )
        return AiterRunnerOutput(hidden_states=output)

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
        quant_type=quant_info.quant_type,
    )


def _is_mori_dispatch_output(dispatch_output: Any) -> bool:
    # MoriEP{Normal,LL}DispatchOutput carry the post-mori-permute origin_topk_*
    # tensors that the standard DeepEP outputs lack.
    return hasattr(dispatch_output, "origin_topk_ids")


def _resolve_mori_quant_type(
    dispatch_a1_dtype: torch.dtype,
    dispatch_scale: Optional[torch.Tensor],
    weight_quant: AiterQuantType,
) -> AiterQuantType:
    """Pick the activation quant_type for AITER when the dispatch path may have
    pre-quantized hidden_states. Mirrors the original MoriEPMoE.run_moe_core
    decision tree."""
    is_fp8_quant = weight_quant in (
        AiterQuantType.PER_128X128,
        AiterQuantType.PER_TOKEN,
    )
    is_w4a4 = weight_quant == AiterQuantType.PER_1X32
    is_fp4_dispatch = dispatch_a1_dtype == torch.float4_e2m1fn_x2
    has_dispatch_scale = dispatch_scale is not None

    if is_w4a4:
        # W4A4 weights always run as per_1x32; FP8 dispatch is upscaled to BF16
        # before this point so dispatch_scale won't conflict.
        return AiterQuantType.PER_1X32
    if is_fp8_quant:
        return weight_quant
    # BF16 weights: lift to the dispatch-side quant type when scales are provided.
    if has_dispatch_scale and is_fp4_dispatch:
        return AiterQuantType.PER_1X32
    if has_dispatch_scale and not is_fp4_dispatch:
        return AiterQuantType.PER_128X128
    return AiterQuantType.NONE


def _pre_permute_deepep_to_aiter(
    dispatch_output: Union[
        DeepEPNormalDispatchOutput,
        DeepEPLLDispatchOutput,
        MoriEPNormalDispatchOutput,
        MoriEPLLDispatchOutput,
    ],
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AiterRunnerInput:
    is_mori = _is_mori_dispatch_output(dispatch_output)

    hidden_states = dispatch_output.hidden_states
    topk_ids = dispatch_output.topk_ids.to(torch.int32)
    topk_weights = dispatch_output.topk_weights.to(torch.float32)
    a1_scale: Optional[torch.Tensor] = None
    num_local_tokens: Optional[torch.Tensor] = None
    output_dtype: Optional[torch.dtype] = None
    quant_type = quant_info.quant_type

    if is_mori:
        from sglang.kernels.ops.moe.rocm_moe_utils import upscale, upscale_mxfp4

        a1_scale = dispatch_output.hidden_states_scale
        num_local_tokens = dispatch_output.num_recv_tokens_per_expert
        output_dtype = dispatch_output.out_dtype

        # Truncate dispatch tensors to the configured cap; mori combine only
        # reads [0, totalRecvTokenNum), so the truncated result needs no
        # padding back.
        mori_max = get_int_env_var("SGLANG_MORI_MOE_MAX_INPUT_TOKENS", 0)
        if mori_max <= 0:
            mori_max = _mori_decode_recv_bound(
                hidden_states.shape[0], topk_ids.shape[-1]
            )
        if mori_max > 0:
            hidden_states = hidden_states[:mori_max]
            if a1_scale is not None:
                a1_scale = a1_scale[:mori_max]
            topk_ids = topk_ids[:mori_max]
            topk_weights = topk_weights[:mori_max]

        # Upscale dispatched activations when there is no AITER kernel for the
        # weight/activation dtype pair.
        weight_quant = quant_info.quant_type
        is_fp8_quant = weight_quant in (
            AiterQuantType.PER_128X128,
            AiterQuantType.PER_TOKEN,
        )
        is_w4a4 = weight_quant == AiterQuantType.PER_1X32
        is_fp4_dispatch = hidden_states.dtype == torch.float4_e2m1fn_x2

        # AITER fused_moe Clamped-SwiGLU is dispatched with
        # gate_mode=INTERLEAVE, for which AITER picks a bf16/fp8 `q_dtype_a`
        # Refer to https://github.com/ROCm/aiter/blob/a2617c366dc7271a1662ecda2023d19f6ccefcec/aiter/fused_moe.py#L406-L412
        swiglu_interleave = quant_info.swiglu_limit > 0 and get_bool_env_var(
            "SGLANG_USE_AITER_MOE_GU_ITLV", "true"
        )

        # MXFP8 dispatch already carries fp8 data with group-32 e8m0 scales,
        # which is what per_1x32 wants, so hand it straight to fused_moe. Only
        # fp8 dispatch's group-128/fp32 scales need the dequant round trip; it is
        # distinguishable by scale dtype (fp32 there, e8m0 here).
        is_mx_fp8_dispatch = (
            a1_scale is not None
            and a1_scale.dtype == torch.float8_e8m0fnu
            and not is_fp4_dispatch
        )
        if (
            is_w4a4
            and a1_scale is not None
            and not is_fp4_dispatch
            and not is_mx_fp8_dispatch
        ):
            # W4A4 weights with FP8 dispatch: dequant FP8->BF16 first; the
            # FP4 per_1x32 path needs BF16 input.
            hidden_states = upscale(
                hidden_states, a1_scale, num_local_tokens, output_dtype
            )
            a1_scale = None
        elif is_w4a4 and is_fp4_dispatch and a1_scale is not None and swiglu_interleave:
            # W4A4 weights + FP4 dispatch on the clamped-SwiGLU/INTERLEAVE
            # path: AITER expects a bf16/fp8 activation here, not fp4x2.
            # Dequant FP4->BF16 and let fused_moe re-quantize internally.
            hidden_states = upscale_mxfp4(
                hidden_states, a1_scale, num_local_tokens, output_dtype
            )
            a1_scale = None
        elif is_fp8_quant and is_fp4_dispatch and a1_scale is not None:
            # FP8 weights + FP4 dispatch: no kernel for the fp4x2/fp8 pair;
            # dequant FP4->BF16 and let fused_moe re-quantize to FP8.
            hidden_states = upscale_mxfp4(
                hidden_states, a1_scale, num_local_tokens, output_dtype
            )
            a1_scale = None

        quant_type = _resolve_mori_quant_type(
            hidden_states.dtype, a1_scale, weight_quant
        )

        running_state["aiter_combine_topk_ids"] = dispatch_output.origin_topk_ids
        running_state["aiter_combine_topk_weights"] = (
            dispatch_output.origin_topk_weights
        )
    else:
        # DeepEP marks invalid topk slots with idx == -1; AITER cannot accept
        # negative ids, so reroute them to the sink slot at index
        # num_local_experts (masked off by quant_info.expert_mask which has
        # shape (num_local_experts + 1,)).
        topk_ids = torch.where(
            topk_ids == -1,
            torch.full_like(topk_ids, runner_config.num_local_experts),
            topk_ids,
        )
        running_state["aiter_combine_topk_ids"] = dispatch_output.topk_ids
        running_state["aiter_combine_topk_weights"] = dispatch_output.topk_weights

    running_state["aiter_combine_is_mori"] = is_mori

    return AiterRunnerInput(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        quant_type=quant_type,
        a1_scale=a1_scale,
        num_local_tokens=num_local_tokens,
        output_dtype=output_dtype,
    )


register_pre_permute("deepep_normal", "aiter")(_pre_permute_deepep_to_aiter)
register_pre_permute("deepep_ll", "aiter")(_pre_permute_deepep_to_aiter)


@register_post_permute("aiter", "standard")
def post_permute_aiter_to_standard(
    runner_output: AiterRunnerOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(hidden_states=runner_output.hidden_states)


def _post_permute_aiter_to_deepep(
    runner_output: AiterRunnerOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
    is_normal: bool,
) -> CombineInput:
    if running_state.get("aiter_combine_is_mori"):
        from sglang.srt.layers.moe.token_dispatcher.moriep import (
            MoriEPLLCombineInput,
            MoriEPNormalCombineInput,
        )

        cls = MoriEPNormalCombineInput if is_normal else MoriEPLLCombineInput
    else:
        from sglang.srt.layers.moe.token_dispatcher.deepep import (
            DeepEPLLCombineInput,
            DeepEPNormalCombineInput,
        )

        cls = DeepEPNormalCombineInput if is_normal else DeepEPLLCombineInput

    return cls(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["aiter_combine_topk_ids"],
        topk_weights=running_state["aiter_combine_topk_weights"],
    )


@register_post_permute("aiter", "deepep_normal")
def post_permute_aiter_to_deepep_normal(
    runner_output: AiterRunnerOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> CombineInput:
    return _post_permute_aiter_to_deepep(
        runner_output, quant_info, runner_config, running_state, is_normal=True
    )


@register_post_permute("aiter", "deepep_ll")
def post_permute_aiter_to_deepep_ll(
    runner_output: AiterRunnerOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> CombineInput:
    return _post_permute_aiter_to_deepep(
        runner_output, quant_info, runner_config, running_state, is_normal=False
    )
