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

import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import triton.language as tl
from lightop import gemm_ops as quant_tools
from lightop._lmslim_native.layers.fused_moe import w4a8 as w4a8_triton
from lightop._lmslim_native.vllm_compat.fused_moe_cache import get_moe_cache
from lightop.quant import per_token_quant_int8
from torch.nn.parameter import Parameter

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.linear import LinearBase, set_weight_attrs
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.parameter import (
    ChannelQuantScaleParameter,
    RowvLLMParameter,
    _ColumnvLLMParameter,
)
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.compressed_tensors import quant_ops
from sglang.srt.utils import W8a8GetCacheJSON, get_bool_env_var

_use_fused_rms_quant = get_bool_env_var("SGLANG_USE_FUSED_RMS_QUANT")
_use_fused_silu_mul_quant = get_bool_env_var("SGLANG_USE_FUSED_SILU_MUL_QUANT")


class ModelWeightParameter(_ColumnvLLMParameter, RowvLLMParameter):
    """
    Parameter class for linear layer weights. Uses both column and
    row parallelism.
    """

    pass


W8A8_TRITONJSON = W8a8GetCacheJSON()


def baseline_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    scales = scale_a * scale_b.T
    gemmout = torch.mm(a.to(dtype=torch.float32), b.to(dtype=torch.float32))
    output = (scales * gemmout).to(out_dtype)
    if bias is not None:
        output = output + bias
    return output.to(out_dtype)


def fused_experts_impl_w4a8_triton(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    cache13: torch.Tensor,
    *,
    activation: str,
    apply_router_weight_on_input: bool,
    global_num_experts: int,
    expert_map: Optional[torch.Tensor],
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    routed_scaling_factor: float,
    shared_output: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run SlimQuant W4A8 Triton GEMMs without whole-layer LightOp fusion."""
    assert hidden_states.ndim == 2 and hidden_states.is_contiguous()
    assert hidden_states.shape[1] == w1.shape[2] * 2
    assert topk_weights.shape == topk_ids.shape

    num_tokens = hidden_states.shape[0]
    if num_tokens == 0:
        return torch.empty_like(hidden_states)
    top_k = topk_ids.shape[1]
    n1 = w1.shape[1]
    n2 = w2.shape[1]
    chunk_size = min(int(os.getenv("LMSLIM_FUSED_MOE_CHUNK_SIZE", "32768")), num_tokens)
    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16
    output = torch.empty_like(hidden_states)

    for begin in range(0, num_tokens, chunk_size):
        end = min(begin + chunk_size, num_tokens)
        token_count = end - begin
        current_x = hidden_states[begin:end]
        current_ids = topk_ids[begin:end]
        current_weights = topk_weights[begin:end]
        cache1 = cache13[: token_count * top_k * n1].view(token_count, top_k, n1)
        cache3 = cache13[: token_count * top_k * n2].view(token_count, top_k, n2)

        config1, config2 = w4a8_triton.get_w8a8moe_json(
            token_count, w1.shape[0], n1, n2, n1 // 2
        )
        sorted_ids, expert_ids, padded_count = w4a8_triton.moe_align_block_size(
            current_ids, config1["BLOCK_SIZE_M"], global_num_experts, expert_map
        )
        qx, x_scale = per_token_quant_int8(current_x)
        w4a8_triton.invoke_fused_moe_kernel_w4a8(
            qx,
            w1,
            cache1,
            x_scale,
            w1_scale,
            None,
            current_weights,
            sorted_ids,
            expert_ids,
            padded_count,
            apply_router_weight_on_input,
            top_k,
            config1,
            compute_type=compute_type,
        )

        gate, up = cache1.chunk(2, dim=-1)
        if activation == "silu":
            activated = F.silu(gate) * up
        elif activation == "gelu":
            activated = F.gelu(gate) * up
        else:
            raise ValueError(f"Unsupported FusedMoE activation: {activation}")
        qactivated, activated_scale = per_token_quant_int8(
            activated.reshape(token_count * top_k, n1 // 2)
        )
        w4a8_triton.invoke_fused_moe_kernel_w4a8(
            qactivated,
            w2,
            cache3,
            activated_scale,
            w2_scale,
            None,
            current_weights,
            sorted_ids,
            expert_ids,
            padded_count,
            not apply_router_weight_on_input,
            1,
            config2,
            compute_type=compute_type,
        )
        reduced = cache3.sum(dim=1).mul_(routed_scaling_factor)
        if shared_output is not None:
            reduced.add_(shared_output[begin:end])
        output[begin:end].copy_(reduced)

    return output


class SlimQuantW4A8Int8Config(QuantizationConfig):
    """Config class for W8A8 Int8 Quantization.

    - Weight: static, per-channel, symmetric
    - Activation: dynamic, per-token, symmetric
    """

    def __init__(self):
        pass

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_name(self) -> str:
        return "slimquant_w4a8"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SlimQuantW4A8Int8Config":
        return cls()

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        from sglang.srt.layers.moe.fused_moe_triton import (
            FusedMoE,
        )

        if isinstance(layer, LinearBase):
            return SlimQuantW4A8Int8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return SlimQuantW4A8Int8MoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class SlimQuantW4A8Int8LinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: SlimQuantW4A8Int8Config):
        self.quantization_config = quantization_config
        self.tritonsingleton = W8a8GetCacheJSON()
        self.w8a8_strategy = int(os.getenv("W8A8_SUPPORT_METHODS", "1"))

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        n = layer.weight.shape[0]
        k = layer.weight.shape[1]

        if self.w8a8_strategy == 1:
            if {n, k} not in self.tritonsingleton.weight_shapes:
                self.tritonsingleton.weight_shapes.append({n, k})
                json_file = self.tritonsingleton.get_w8a8json_name(n, k)
                configs_dict = self.tritonsingleton.get_triton_cache(json_file, n, k)

                if configs_dict:
                    self.tritonsingleton.triton_json_dict.update(configs_dict)

                    for key, value in configs_dict.items():
                        m = int(key.split("_")[0])
                        quant_tools.triton_int8_gemm_helper(
                            m=m,
                            n=n,
                            k=k,
                            per_token_act_quant=True,
                            per_out_channel_weight_quant=True,
                            use_bias=False,
                            device=layer.weight.device,
                            best_config=value,
                        )
        elif self.w8a8_strategy == 3:
            layer.weight.data = layer.weight.data.T
        else:
            weight_data = layer.weight.data
            _weight = weight_data.T.contiguous().reshape(n, -1)
            layer.weight.data = _weight

        layer.weight = Parameter(layer.weight.t(), requires_grad=False)
        layer.weight_scale = Parameter(layer.weight_scale.data, requires_grad=False)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        weight_loader = extra_weight_attrs.get("weight_loader")
        self.logical_widths = output_partition_sizes

        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes), input_size_per_partition, dtype=torch.int8
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        input_quant_args: Optional[list[torch.Tensor]] = None,
        silu_quant_args: Optional[list[torch.Tensor]] = None,
    ):
        if _use_fused_rms_quant and input_quant_args is not None:
            assert len(input_quant_args) == 2
            x_q, x_scale = input_quant_args
        elif _use_fused_silu_mul_quant and silu_quant_args is not None:
            x_q, x_scale = silu_quant_args
        else:
            x_q, x_scale = per_token_quant_int8(x)

        if self.w8a8_strategy == 1:
            m = x_q.shape[0]
            k = x_q.shape[1]
            n = layer.weight.shape[1]

            if len(W8A8_TRITONJSON.triton_json_dict) == 0:
                best_config = None

            elif f"1_{n}_{k}" in W8A8_TRITONJSON.triton_json_dict:
                if m <= 16:
                    m_ = m
                elif m <= 64:
                    m_ = (m + 3) & -4  # 取值到最近的4的倍数
                elif m <= 160:
                    m_ = (m + 7) & -8

                elif m < 200:  # 256
                    m_ = 160
                elif m < 480:  # 512
                    m_ = 256
                elif m < 960:  # 1024
                    m_ = 512
                elif m < 2048:
                    m_ = 1024
                elif m < 4096:
                    m_ = 2048
                elif m < 6000:
                    m_ = 4096
                else:
                    m_ = 8192

                best_config = W8A8_TRITONJSON.triton_json_dict[f"{m_}_{n}_{k}"]

            else:
                best_config = None

            # if best_config==None:
            #    print("m:{},n:{},k:{}".format(m,n,k))
            #    print("config not found!")

            output = quant_ops.triton_scaled_mm(
                x_q,
                layer.weight,
                scale_a=x_scale,
                scale_b=layer.weight_scale,
                out_dtype=x.dtype,
                bias=bias,
                best_config=best_config,
            )
            return output
        elif self.w8a8_strategy == 2:
            return quant_ops.cutlass_scaled_mm(
                x_q,
                layer.weight,
                scale_a=x_scale,
                scale_b=layer.weight_scale,
                out_dtype=x.dtype,
                bias=bias,
            )
        elif self.w8a8_strategy == 3:
            return quant_ops.blaslt_scaled_mm(
                x_q,
                layer.weight,
                scale_a=x_scale,
                scale_b=layer.weight_scale,
                out_dtype=x.dtype,
                bias=None,
            )
        else:
            return quant_ops.rocblas_scaled_mm(
                x_q,
                layer.weight,
                scale_a=x_scale,
                scale_b=layer.weight_scale,
                out_dtype=x.dtype,
                bias=bias,
            )


class SlimQuantW4A8Int8MoEMethod:
    """MoE method for W4A8INT8.
    Supports loading INT8 checkpoints with static weight scale and
    dynamic/static activation scale.
    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.
    Args:
        quant_config: The quantization config.
    """

    def __new__(cls, *args, **kwargs):

        if not hasattr(cls, "_initialized"):
            original_init = cls.__init__
            new_cls = type(
                cls.__name__,
                (FusedMoEMethodBase,),
                {
                    "__init__": original_init,
                    **{k: v for k, v in cls.__dict__.items() if k != "__dict__"},
                },
            )
            obj = super(new_cls, new_cls).__new__(new_cls)
            obj.__init__(*args, **kwargs)
            return obj
        return super().__new__(cls)

    def __init__(self, quant_config):
        self.quant_config = quant_config
        self.tritonsingleton = W8a8GetCacheJSON()

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import (
            FusedMoeWeightScaleSupported,
        )

        tp_size = get_tensor_model_parallel_world_size()
        intermediate_size = intermediate_size_per_partition

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size, hidden_size // 2, dtype=torch.int8
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, hidden_size, intermediate_size // 2, dtype=torch.int8
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, 2 * intermediate_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w13_input_scale = None
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = None
        layer.register_parameter("w2_input_scale", w2_input_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        E = layer.w13_weight.shape[0]
        N1 = layer.w13_weight.shape[1]
        N2 = layer.w2_weight.shape[1]
        K = N1 // 2
        if [E, N1, N2, K] not in self.tritonsingleton.moe_weight_shapes:
            self.tritonsingleton.moe_weight_shapes.append([E, N1, N2, K])

        TOPK = self.tritonsingleton.topk

        json_file = self.tritonsingleton.get_moeint8json_name(
            E, N1, N2, K, TOPK, use_int4_w4a8=True
        )
        configs_dict = self.tritonsingleton.get_moeint8_triton_cache(
            json_file, E, N1, N2, K, TOPK
        )

        # warmup
        if configs_dict:
            self.tritonsingleton.triton_moejson_dict.update(configs_dict)

        layer.w13_weight = Parameter(layer.w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(layer.w2_weight, requires_grad=False)
        layer.w13_weight_scale = Parameter(
            layer.w13_weight_scale.data, requires_grad=False
        )
        layer.w2_weight_scale = Parameter(
            layer.w2_weight_scale.data, requires_grad=False
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def _apply_triton(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_output,
        activation: str,
        shared_output: Optional[torch.Tensor],
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        x, topk_weights = apply_topk_weights_cpu(
            self.moe_runner_config.apply_router_weight_on_input, topk_weights, x
        )
        cache13 = get_moe_cache(
            topk_ids.shape[1],
            layer.w13_weight.shape[1],
            layer.w2_weight.shape[1],
            device=x.device,
            dtype=x.dtype,
        )
        routed_scaling_factor = (
            self.moe_runner_config.routed_scaling_factor
            if self.moe_runner_config.routed_scaling_factor is not None
            else 1.0
        )
        output = fused_experts_impl_w4a8_triton(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            cache13,
            activation=activation,
            apply_router_weight_on_input=self.moe_runner_config.apply_router_weight_on_input,
            global_num_experts=self.moe_runner_config.num_experts,
            expert_map=getattr(layer, "expert_map", None),
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            routed_scaling_factor=routed_scaling_factor,
            shared_output=shared_output,
        )
        return output

    @torch._dynamo.disable()
    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output,
        i_q: Optional[torch.Tensor] = None,
        i_s: Optional[torch.Tensor] = None,
    ):
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

        output = self._apply_triton(
            layer,
            dispatch_output.hidden_states,
            dispatch_output.topk_output,
            layer.moe_runner_config.activation,
            shared_output=None,
        )
        return StandardCombineInput(hidden_states=output)

    @torch._dynamo.disable()
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
        return self._apply_triton(
            layer, x, topk_output, activation, shared_output=shared_output
        )
