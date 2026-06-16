      
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import enum
from enum import Enum
from typing import Callable, List, Optional
import torch

from compressed_tensors.quantization import (QuantizationStrategy)
import logging
from torch.nn.parameter import Parameter

from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase

from sglang.srt.utils import set_weight_attrs, direct_register_custom_op
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
try:
    from lmslim.layers.fused_moe.fuse_moe_int8_marlin import fused_experts_impl_int8_marlin
except Exception:
    print("INFO: Please install lmslim if you want to infer the quantitative model of moe.\n")

logger = logging.getLogger(__name__)

__all__ = [
    "CompressedTensorsW8A8Int8MarlinMoEMethod",
]

def fused_experts_impl_int8_marlin_wrapper(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    use_nn_moe: Optional[bool] = False,
    routed_scaling_factor: Optional[float] = 1.0,
    shared_output: Optional[torch.Tensor] = None,
    i_q: Optional[torch.Tensor] = None,
    i_s: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return fused_experts_impl_int8_marlin(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=inplace,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        use_nn_moe=use_nn_moe,
        routed_scaling_factor=routed_scaling_factor,
        shared_output=shared_output,
        i_q=i_q,
        i_s=i_s,
    )

def fused_experts_impl_int8_marlin_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    use_nn_moe: Optional[bool] = False,
    routed_scaling_factor: Optional[float] = 1.0,
    shared_output: Optional[torch.Tensor] = None,
    i_q: Optional[torch.Tensor] = None,
    i_s: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)

direct_register_custom_op(
    op_name="fused_experts_impl_int8_marlin",
    op_func=fused_experts_impl_int8_marlin_wrapper,
    mutates_args=[],
    fake_impl=fused_experts_impl_int8_marlin_fake,
)

    

def get_w8a8_int8_marlin_weights(
         weight,
         k_tile=64):
    # 7168, 512
    weight = weight.T
    size_k, size_n = weight.shape
    assert size_k // k_tile
    weight = weight.reshape(size_k // k_tile, k_tile, size_n)
    weight = weight.transpose(1, 2)
    weight = weight.reshape(size_k // k_tile, size_n * k_tile)

    return weight

def w8a8_nt_kpack2_marlin_weight(w8a8_w, # [size_n, size_k// 2 ]
                                k_tile=16,
                                n_tile=16, ):
    assert w8a8_w.dtype == torch.int8, "w8a8_w 必须是 int8 类型"
    size_n, size_k = w8a8_w.shape
    assert size_n % k_tile == 0 and size_k % n_tile == 0, "k_tile / n_tile 必须能整除对应维度"

    q = w8a8_w.reshape((size_n // n_tile,  n_tile, size_k // k_tile, k_tile))
    q = q.permute((0, 2, 1, 3)).contiguous()
    q = q.reshape((size_n // k_tile, size_k * k_tile))
    return q

def weight8bit_nt_kpack2_marlin1(weight, # [size_n, size_k// 2 ]
                                k_tile=16,
                                k_tile1=4,
                                n_tile=16, 
                                n_tile1=16):
    assert weight.element_size() == 1, "weight 必须是 8 bit 类型"
    if weight.dim() == 2:
        size_n, size_k = weight.shape
        assert size_n % k_tile == 0 and size_k % n_tile == 0, "k_tile / n_tile 必须能整除对应维度"

        q = weight.reshape((size_n // (n_tile*n_tile1), n_tile1, n_tile, size_k // (k_tile*k_tile1), k_tile1, k_tile))
        # q = q.permute((0, 2, 1, 3)).contiguous()
        q = q.permute((0, 3, 1, 4, 2, 5)).contiguous()
        # q = q.reshape((size_n // k_tile, size_k * k_tile))
    elif weight.dim() == 3:
        E, size_n, size_k = weight.shape
        assert size_n % n_tile == 0 and size_k % k_tile == 0, "k_tile / n_tile 必须能整除对应维度"

        q = weight.reshape((E, size_n // (n_tile*n_tile1), n_tile1, n_tile, size_k // (k_tile*k_tile1), k_tile1, k_tile))
        q = q.permute((0, 1, 4, 2, 5, 3, 6)).contiguous()
        # q = q.reshape((E, size_n // k_tile, size_k * k_tile))
    return q

class CompressedTensorsMarlinMoEMethod(FusedMoEMethodBase):
    @staticmethod
    def get_moe_method(
        quant_config: "SlimQuantCompressedTensorsMarlinConfig",  # type: ignore # noqa E501
        layer: torch.nn.Module,
    ) -> "CompressedTensorsMarlinMoEMethod":
        # are supported + check if the layer is being ignored.
        weight_quant = quant_config.target_scheme_map["Linear"].get("weights")
        input_quant = quant_config.target_scheme_map["Linear"].get(
            "input_activations")
        if quant_config._is_dynamic_token_w8a8(weight_quant, input_quant):
            return CompressedTensorsW8A8Int8MarlinMoEMethod(quant_config)
        else:
            raise RuntimeError(
                f"Slimquant_marlin does not support the FusedMoe scheme: {weight_quant}, {input_quant}")

class CompressedTensorsW8A8Int8MarlinMoEMethod(CompressedTensorsMarlinMoEMethod):
    def __init__(
            self,
            quant_config: "CompressedTensorsMarlinConfig"  # type: ignore # noqa E501
    ):
        self.quant_config = quant_config
        self.weight_quant = self.quant_config.target_scheme_map["Linear"].get(
            "weights")
        self.input_quant = self.quant_config.target_scheme_map["Linear"].get(
            "input_activations")
        self.use_deepep = get_moe_a2a_backend().is_deepep()
        per_channel = (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN)
        if not per_channel:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found "
                f"{self.weight_quant}, {self.input_quant}")

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found static input scales.")

        
    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        params_dtype = torch.int8

        # WEIGHTS
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        assert self.weight_quant.strategy == QuantizationStrategy.CHANNEL
        w13_weight_scale = torch.nn.Parameter(torch.ones(
            num_experts,
            2 * intermediate_size_per_partition,
            1,
            dtype=torch.float32),
                                              requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                        hidden_size,
                                                        1,
                                                        dtype=torch.float32),
                                             requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        assert not self.static_input_scales
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            weight8bit_nt_kpack2_marlin,
        )

        w1_marlin_list = []
        for ii in range(layer.w13_weight.shape[0]):
            if not self.use_deepep:
                w1_marlin_in = get_w8a8_int8_marlin_weights(layer.w13_weight[ii])
            else:
                w1_marlin_in = weight8bit_nt_kpack2_marlin(layer.w13_weight[ii])
            w1_marlin_list.append(w1_marlin_in)
        w1_marlin = torch.stack(w1_marlin_list, dim=0)

        w2_marlin_list = []
        for ii in range(layer.w2_weight.shape[0]):
            if not self.use_deepep:
                w2_marlin_in = get_w8a8_int8_marlin_weights(layer.w2_weight[ii])
            else:
                w2_marlin_in = weight8bit_nt_kpack2_marlin(layer.w2_weight[ii])
            w2_marlin_list.append(w2_marlin_in)
        w2_marlin = torch.stack(w2_marlin_list, dim=0)

        layer.w13_weight = Parameter(w1_marlin, requires_grad=False)
        layer.w2_weight = Parameter(w2_marlin, requires_grad=False)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    # def apply(
    #     self,
    #     layer: torch.nn.Module,
    #     x: torch.Tensor,
    #     router_logits: torch.Tensor,
    #     top_k: int,
    #     renormalize: bool,
    #     use_grouped_topk: bool = False,
    #     topk_group: Optional[int] = None,
    #     num_expert_group: Optional[int] = None,
    #     global_num_experts: int = -1,
    #     expert_map: Optional[torch.Tensor] = None,
    #     custom_routing_function: Optional[Callable] = None,
    #     scoring_func: str = "softmax",
    #     e_score_correction_bias: Optional[torch.Tensor] = None,
    #     apply_router_weight_on_input: bool = False,
    #     activation: str = "silu",
    #     enable_eplb: bool = False,
    #     use_nn_moe: Optional[bool] = False,
    #     routed_scaling_factor: Optional[float] = None,
    #     use_fused_gate: Optional[bool] = False,
    #     expert_load_view: Optional[torch.Tensor] = None,
    #     logical_to_physical_map: Optional[torch.Tensor] = None,
    #     logical_replica_count: Optional[torch.Tensor] = None,
    #     shared_output: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #     from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    #     if enable_eplb:
    #         raise NotImplementedError(
    #             "EPLB not supported for "
    #             "`CompressedTensorsW8A8Int8MoEMethod` yet.")


    #     topk_weights, topk_ids = FusedMoE.select_experts(
    #         hidden_states=x,
    #         router_logits=router_logits,
    #         use_grouped_topk=use_grouped_topk,
    #         top_k=top_k,
    #         renormalize=renormalize,
    #         topk_group=topk_group,
    #         num_expert_group=num_expert_group,
    #         custom_routing_function=custom_routing_function,
    #         scoring_func=scoring_func,
    #         routed_scaling_factor=routed_scaling_factor,
    #         use_fused_gate=use_fused_gate,
    #         e_score_correction_bias=e_score_correction_bias)

    #     return fused_experts_impl_int8_marlin(
    #         hidden_states=x,
    #         w1=layer.w13_weight,
    #         w2=layer.w2_weight,
    #         topk_weights=topk_weights,
    #         topk_ids=topk_ids,
    #         inplace=True,
    #         activation=activation,
    #         apply_router_weight_on_input=apply_router_weight_on_input,
    #         use_int8_w8a8=True,
    #         per_channel_quant=True,
    #         global_num_experts=global_num_experts,
    #         expert_map=expert_map,
    #         w1_scale=layer.w13_weight_scale,
    #         w2_scale=layer.w2_weight_scale,
    #         a1_scale=layer.w13_input_scale,
    #         a2_scale=layer.w2_input_scale,
    #         use_nn_moe=False,
    #         shared_output=shared_output,
    #         routed_scaling_factor=routed_scaling_factor)
    
    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output,
        # local_expert_mapping,
        i_q: Optional[torch.Tensor] = None,
        i_s: Optional[torch.Tensor] = None,
    ):
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
        from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        topk_weights, topk_ids, _ = topk_output
        x, topk_weights = apply_topk_weights_cpu(
            self.moe_runner_config.apply_router_weight_on_input, topk_weights, x
        )

        routed_scaling_factor = (
            self.moe_runner_config.routed_scaling_factor
            if self.moe_runner_config.routed_scaling_factor is not None
            else 1.0
        )

        output = torch.ops.sglang.fused_experts_impl_int8_marlin(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=layer.moe_runner_config.activation,
            apply_router_weight_on_input=self.moe_runner_config.apply_router_weight_on_input,
            use_int8_w8a8=True,
            per_channel_quant=True,
            global_num_experts=layer.moe_runner_config.num_experts,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            use_nn_moe=False,
            routed_scaling_factor=float(routed_scaling_factor),
            # expert_map=local_expert_mapping,
        )

        return StandardCombineInput(hidden_states=output)
    
    def apply_with_shared_output(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        activation: str = "silu",
        shared_output: Optional[torch.Tensor] = None,
        topk_output=None,
        i_q: Optional[torch.Tensor] = None,
        i_s: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids

        routed_scaling_factor = (
            self.moe_runner_config.routed_scaling_factor
            if self.moe_runner_config.routed_scaling_factor is not None
            else 1.0
        )

        output = torch.ops.sglang.fused_experts_impl_int8_marlin(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            apply_router_weight_on_input=self.moe_runner_config.apply_router_weight_on_input,
            use_int8_w8a8=True,
            per_channel_quant=True,
            global_num_experts=layer.moe_runner_config.num_experts,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            use_nn_moe=False,
            routed_scaling_factor=float(routed_scaling_factor),
            shared_output=shared_output,
            i_q=i_q,
            i_s=i_s,
        )

        return output
