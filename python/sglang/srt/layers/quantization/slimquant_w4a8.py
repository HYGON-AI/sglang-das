import os
from typing import Any, Callable, Dict, List, Optional

import torch
from lmslim import quant_tools
from lmslim.layers.gemm.int8_utils import per_token_quant_int8
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
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

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

            return quant_ops.triton_scaled_mm(
                x_q,
                layer.weight,
                scale_a=x_scale,
                scale_b=layer.weight_scale,
                out_dtype=x.dtype,
                bias=bias,
                best_config=best_config,
            )
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
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        tp_size = get_tensor_model_parallel_world_size()

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

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        use_nn_moe: Optional[bool] = False,
        routed_scaling_factor: Optional[float] = None,
        use_fused_gate: Optional[bool] = False,
        **_,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts

        if enable_eplb:
            raise NotImplementedError(
                "EPLB not supported for `SlimQuantW4A8Int8MoEMethod` yet."
            )
        # Expert selection
        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            routed_scaling_factor=routed_scaling_factor,
            use_fused_gate=use_fused_gate,
        )

        return fused_experts(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            use_int4_w4a8=True,
            per_channel_quant=True,
            activation=activation,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            w1_scale=(layer.w13_weight_scale),
            w2_scale=(layer.w2_weight_scale),
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            use_nn_moe=use_nn_moe,
        )
