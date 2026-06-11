from __future__ import annotations

from dataclasses import dataclass
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
from sglang.srt.layers.moe.utils import MoeRunnerBackend, get_moe_a2a_backend
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors_moe_marlin import (
    get_w8a8_int8_marlin_weights,
    weight8bit_nt_kpack2_marlin1,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


@dataclass
class LightOpRunnerInput(RunnerInput):
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.LIGHTOP


@dataclass
class LightOpRunnerOutput(RunnerOutput):
    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.LIGHTOP


@dataclass
class LightOpMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_scale: torch.Tensor
    w2_scale: torch.Tensor
    a13_scale: Optional[torch.Tensor] = None
    a2_scale: Optional[torch.Tensor] = None


def get_lightop_marlin_weight(weight: torch.Tensor) -> torch.Tensor:
    if get_moe_a2a_backend().is_deepep():
        return weight8bit_nt_kpack2_marlin1(weight)
    return get_w8a8_int8_marlin_weights(weight)


def process_weights_after_loading_lightop(layer: torch.nn.Module) -> None:
    w13_weight = torch.stack(
        [
            get_lightop_marlin_weight(layer.w13_weight[i])
            for i in range(layer.w13_weight.shape[0])
        ],
        dim=0,
    )
    w2_weight = torch.stack(
        [
            get_lightop_marlin_weight(layer.w2_weight[i])
            for i in range(layer.w2_weight.shape[0])
        ],
        dim=0,
    )
    layer.w13_weight = Parameter(w13_weight, requires_grad=False)
    layer.w2_weight = Parameter(w2_weight, requires_grad=False)


def get_lightop_quant_info(layer: torch.nn.Module) -> LightOpMoeQuantInfo:
    return LightOpMoeQuantInfo(
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        w13_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        a13_scale=layer.w13_input_scale,
        a2_scale=layer.w2_input_scale,
    )


class LightOpRunnerCore(MoeRunnerCore):
    def run(
        self,
        runner_input: LightOpRunnerInput,
        quant_info: LightOpMoeQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> LightOpRunnerOutput:
        assert hooks is None, "LightOp MoE does not support LoRA hooks."

        routed_scaling_factor = (
            self.config.routed_scaling_factor
            if self.config.routed_scaling_factor is not None
            else 1.0
        )

        output = torch.ops.sglang.fused_experts_impl_int8_marlin(
            runner_input.hidden_states,
            quant_info.w13_weight,
            quant_info.w2_weight,
            topk_weights=runner_input.topk_weights,
            topk_ids=runner_input.topk_ids,
            inplace=self.config.inplace,
            activation=self.config.activation,
            apply_router_weight_on_input=self.config.apply_router_weight_on_input,
            use_int8_w8a8=True,
            per_channel_quant=True,
            global_num_experts=self.config.num_experts,
            w1_scale=quant_info.w13_scale,
            w2_scale=quant_info.w2_scale,
            a1_scale=quant_info.a13_scale,
            a2_scale=quant_info.a2_scale,
            use_nn_moe=False,
            routed_scaling_factor=float(routed_scaling_factor),
        )
        return LightOpRunnerOutput(hidden_states=output)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.LIGHTOP


@register_pre_permute("standard", "lightop")
def pre_permute_standard_to_lightop(
    dispatch_output: StandardDispatchOutput,
    quant_info: LightOpMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> LightOpRunnerInput:
    from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

    hidden_states = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    hidden_states, topk_weights = apply_topk_weights_cpu(
        runner_config.apply_router_weight_on_input,
        topk_weights,
        hidden_states,
    )

    return LightOpRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )


@register_post_permute("lightop", "standard")
def post_permute_lightop_to_standard(
    runner_output: LightOpRunnerOutput,
    quant_info: LightOpMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(hidden_states=runner_output.hidden_states)
