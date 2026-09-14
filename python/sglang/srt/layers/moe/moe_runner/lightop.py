# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


def weight8bit_nt_kpack2_marlin1(
    weight, k_tile=16, k_tile1=4, n_tile=16, n_tile1=16  # [size_n, size_k// 2 ]
):
    assert weight.element_size() == 1, "weight 必须是 8 bit 类型"
    if weight.dim() == 2:
        size_n, size_k = weight.shape
        assert (
            size_n % k_tile == 0 and size_k % n_tile == 0
        ), "k_tile / n_tile 必须能整除对应维度"

        q = weight.reshape(
            (
                size_n // (n_tile * n_tile1),
                n_tile1,
                n_tile,
                size_k // (k_tile * k_tile1),
                k_tile1,
                k_tile,
            )
        )
        # q = q.permute((0, 2, 1, 3)).contiguous()
        q = q.permute((0, 3, 1, 4, 2, 5)).contiguous()
        # q = q.reshape((size_n // k_tile, size_k * k_tile))
    elif weight.dim() == 3:
        E, size_n, size_k = weight.shape
        assert (
            size_n % n_tile == 0 and size_k % k_tile == 0
        ), "k_tile / n_tile 必须能整除对应维度"

        q = weight.reshape(
            (
                E,
                size_n // (n_tile * n_tile1),
                n_tile1,
                n_tile,
                size_k // (k_tile * k_tile1),
                k_tile1,
                k_tile,
            )
        )
        q = q.permute((0, 1, 4, 2, 5, 3, 6)).contiguous()
        # q = q.reshape((E, size_n // k_tile, size_k * k_tile))
    return q


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
    use_fp8_w8a8: bool = False
    use_int8_w8a8: bool = True
    origin_w13_shape: Optional[torch.Size] = None
    origin_w2_shape: Optional[torch.Size] = None


def _is_moe_prefill_or_normal() -> bool:
    from sglang.srt.runtime_context import get_server_args

    server_args = get_server_args()
    return (
        server_args.disaggregation_mode == "prefill"
        or server_args.deepep_mode == "normal"
    )


def get_lightop_marlin_weight(
    weight: torch.Tensor, *, use_fp8_w8a8: bool = False
) -> torch.Tensor:
    if get_moe_a2a_backend().is_deepep():
        if use_fp8_w8a8 and _is_moe_prefill_or_normal():
            from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
                weight8bit_nt_kpack2_marlin,
            )

            return weight8bit_nt_kpack2_marlin(weight)
        return weight8bit_nt_kpack2_marlin1(weight)

    if use_fp8_w8a8:
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            w8a8_2_marlin_weight,
        )

        return w8a8_2_marlin_weight(weight)
    return get_w8a8_int8_marlin_weights(weight)


def process_weights_after_loading_lightop(
    layer: torch.nn.Module, *, use_fp8_w8a8: bool = False
) -> None:
    origin_w13_shape = layer.w13_weight.shape
    origin_w2_shape = layer.w2_weight.shape
    if use_fp8_w8a8:
        if (
            layer.w13_weight.dim() != 3
            or layer.w2_weight.dim() != 3
            or layer.w13_weight.size(0) != layer.w2_weight.size(0)
        ):
            raise RuntimeError("Unexpected MoE weight shapes")
        two_n, hidden_size = layer.w13_weight.shape[1:]
        if layer.w2_weight.size(1) != hidden_size:
            raise RuntimeError("Unexpected MoE w2 layout")
        intermediate_size = layer.w2_weight.size(2)
        if two_n != 2 * intermediate_size:
            raise RuntimeError("Unexpected MoE hidden dims")
        if (
            hidden_size % 32 != 0
            or intermediate_size % 16 != 0
            or two_n % 32 != 0
        ):
            raise RuntimeError("Marlin packing requires alignment")

    w13_weight = torch.stack(
        [
            get_lightop_marlin_weight(
                layer.w13_weight[i], use_fp8_w8a8=use_fp8_w8a8
            )
            for i in range(layer.w13_weight.shape[0])
        ],
        dim=0,
    )
    w2_weight = torch.stack(
        [
            get_lightop_marlin_weight(
                layer.w2_weight[i], use_fp8_w8a8=use_fp8_w8a8
            )
            for i in range(layer.w2_weight.shape[0])
        ],
        dim=0,
    )
    layer.w13_weight = Parameter(w13_weight, requires_grad=False)
    layer.w2_weight = Parameter(w2_weight, requires_grad=False)
    layer._lightop_origin_w13_shape = origin_w13_shape
    layer._lightop_origin_w2_shape = origin_w2_shape


def get_lightop_quant_info(
    layer: torch.nn.Module, *, use_fp8_w8a8: bool = False
) -> LightOpMoeQuantInfo:
    return LightOpMoeQuantInfo(
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        w13_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        a13_scale=layer.w13_input_scale,
        a2_scale=layer.w2_input_scale,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=not use_fp8_w8a8,
        origin_w13_shape=getattr(layer, "_lightop_origin_w13_shape", None),
        origin_w2_shape=getattr(layer, "_lightop_origin_w2_shape", None),
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

        if quant_info.use_fp8_w8a8:
            if (
                quant_info.origin_w13_shape is None
                or quant_info.origin_w2_shape is None
            ):
                raise RuntimeError("Missing original FP8 MoE weight shapes")

            from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
                fused_moe_fp8_w8a8,
            )

            output = fused_moe_fp8_w8a8(
                hidden_states=runner_input.hidden_states,
                w1=quant_info.w13_weight,
                w2=quant_info.w2_weight,
                w1_scale=quant_info.w13_scale,
                w2_scale=quant_info.w2_scale,
                topk_weights=runner_input.topk_weights,
                topk_ids=runner_input.topk_ids,
                global_num_experts=self.config.num_experts,
                inplace=self.config.inplace,
                origin_w1_shape=quant_info.origin_w13_shape,
                origin_w2_shape=quant_info.origin_w2_shape,
                routed_scaling_factor=routed_scaling_factor,
            )
        else:
            output = torch.ops.sglang.fused_experts_impl_int8_marlin(
                runner_input.hidden_states,
                quant_info.w13_weight,
                quant_info.w2_weight,
                topk_weights=runner_input.topk_weights,
                topk_ids=runner_input.topk_ids,
                inplace=self.config.inplace,
                activation=self.config.activation,
                apply_router_weight_on_input=self.config.apply_router_weight_on_input,
                use_fp8_w8a8=False,
                use_int8_w8a8=quant_info.use_int8_w8a8,
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
