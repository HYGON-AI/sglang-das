from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.moe import (
    MoeRunner,
    MoeRunnerBackend,
    MoeRunnerConfig,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.layers.parameter import ChannelQuantScaleParameter, ModelWeightParameter
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8_kernel import (
    fp8_dtype,
    is_fp8_fnuz,
    per_token_group_quant_fp8,
)
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    cutlass_fp8_supported,
    input_to_float8,
    normalize_e4m3fn_to_e4m3fnuz,
)
from sglang.srt.utils import get_bool_env_var, is_dcu, set_weight_attrs
if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

_is_fp8_fnuz = is_fp8_fnuz()
_is_dcu = is_dcu()
_use_fp8_w8a8_moe = get_bool_env_var("SGLANG_USE_FP8_W8A8_MOE")

try:
    from lmslim.layers.fused_moe.fuse_moe_w4a8_marlin import fused_experts_impl_w4a8_marlin
except Exception:
    print("INFO: Please install lmslim if you want to infer the quantitative model of moe.\n")

class W8A8Fp8Config(QuantizationConfig):
    """Config class for W8A8 FP8 Quantization.

    Weight Quantization:
    - Method: Static quantization
    - Granularity: Per-channel
    - Type: Symmetric

    Activation Quantization:
    - Method: Dynamic quantization
    - Granularity: Per-token
    - Type: Symmetric

    Note:
    - For models without offline quantization, weights will be quantized during model loading
    - If CUTLASS is supported: Per-channel weight quantization is used
    - If CUTLASS is not supported: Falls back to per-tensor weight quantization
    """

    def __init__(self, is_checkpoint_fp8_serialized: bool = False):
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    @classmethod
    def get_name(self) -> str:
        return "w8a8_fp8"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> W8A8Fp8Config:
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = (
            "compressed-tensors" in quant_method or "w8a8_fp8" in quant_method
        )
        return cls(is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            return W8A8Fp8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return W8A8FP8MoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class W8A8Fp8LinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: W8A8Fp8Config):
        self.cutlass_fp8_supported = cutlass_fp8_supported()
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight

        if self.quantization_config.is_checkpoint_fp8_serialized:
            weight_scale = layer.weight_scale.detach()
            # If checkpoint offline quantized with w8a8_fp8, load the weight and weight_scale directly.
            if _is_fp8_fnuz:
                weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                    weight=weight, weight_scale=weight_scale
                )

            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
        else:
            # If checkpoint not offline quantized, quantize the weights with per-channel quantization.
            if self.cutlass_fp8_supported:
                # if cutlass supported, we use cutlass_scaled_mm
                # which requires per-channel quantization on weight
                qweight, weight_scale = per_token_group_quant_fp8(
                    layer.weight, layer.weight.shape[-1]
                )
                weight_scale = weight_scale.t().contiguous()
            else:
                # if cutlass not supported, we fall back to use torch._scaled_mm
                # which requires per tensor quantization on weight
                qweight, weight_scale = input_to_float8(layer.weight, dtype=fp8_dtype)

            # Update the layer with the new values.
            layer.weight = Parameter(qweight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            layer.input_scale = None

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
        weight_dtype = (
            torch.float8_e4m3fn
            if self.quantization_config.is_checkpoint_fp8_serialized
            else params_dtype
        )

        weight_loader = extra_weight_attrs.get("weight_loader")
        self.logical_widths = output_partition_sizes

        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=weight_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        if self.quantization_config.is_checkpoint_fp8_serialized:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_scale", weight_scale)
        else:
            layer.weight_scale = None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        return apply_fp8_linear(
            x,
            layer.weight,
            layer.weight_scale,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_fp8_supported,
        )


class W8A8FP8MoEMethod(FusedMoEMethodBase):
    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.
    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.
    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: W8A8Fp8Config):
        self.quant_config = quant_config
        self.use_deepep = get_moe_a2a_backend().is_deepep()

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=fp8_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=fp8_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
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
        if _is_dcu and _use_fp8_w8a8_moe:
            w1 = layer.w13_weight
            w2 = layer.w2_weight
            w1_shape = w1.shape
            w2_shape = w2.shape
            if (w1.is_cuda and w2.is_cuda):
                if w1.dim() != 3 or w2.dim() != 3 or w1.size(0) != w2.size(
                    0):
                    raise RuntimeError("Unexpected MoE weight shapes")
                twoN, K = w1.size(1), w1.size(2)
                if w2.size(1) != K:
                    raise RuntimeError("Unexpected MoE w2 layout")
                N = w2.size(2)
                if twoN != 2 * N:
                    raise RuntimeError("Unexpected MoE hidden dims")
                if (K % 16 != 0 or K % 32 != 0 or N % 16 != 0
                    or twoN % 32 != 0):
                    raise RuntimeError("Marlin packing requires alignment")

                from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
                    w8a8_2_marlin_weight, weight8bit_nt_kpack2_marlin,weight8bit_nt_kpack2_marlin1)

                def _pack_per_expert(weight: torch.Tensor) -> torch.Tensor:
                    num_experts = weight.shape[0]
                    for i in range(num_experts):
                        new_expert = w8a8_2_marlin_weight(
                            weight[i]).contiguous()
                        weight.data[i].view(-1).copy_(new_expert.view(-1))
                    weight = weight.reshape((-1,) + new_expert.shape)

                    return weight

                def _pack_per_expert_deepep(weight: torch.Tensor) -> torch.Tensor:
                    num_experts = weight.shape[0]
                    for i in range(num_experts):
                        if is_moe_prefill() :
                            new_expert = weight8bit_nt_kpack2_marlin(
                                weight[i]).contiguous()
                        else:
                            new_expert = weight8bit_nt_kpack2_marlin1(
                                weight[i]).contiguous()
                        weight.data[i].view(-1).copy_(new_expert.view(-1))
                    weight = weight.reshape((-1,) + new_expert.shape)
                    return weight

                with torch.no_grad():
                    if self.use_deepep:
                        w1_packed = _pack_per_expert_deepep(w1)
                        w2_packed = _pack_per_expert_deepep(w2)
                    else:
                        w1_packed = _pack_per_expert(w1)
                        w2_packed = _pack_per_expert(w2)

                    new_w1 = Parameter(w1_packed, requires_grad=False)
                    new_w2 = Parameter(w2_packed, requires_grad=False)

                    if hasattr(w1, "__dict__"):
                        for k, v in w1.__dict__.items():
                            setattr(new_w1, k, v)
                    if hasattr(w2, "__dict__"):
                        for k, v in w2.__dict__.items():
                            setattr(new_w2, k, v)

                    setattr(new_w1, "_w8a8_fp8_packed", True)
                    setattr(new_w1, "w1_shape", w1_shape)
                    setattr(new_w2, "_w8a8_fp8_packed", True)
                    setattr(new_w2, "w2_shape", w2_shape)

                    layer.w13_weight = new_w1
                    layer.w2_weight = new_w2
                    layer._w8a8_fp8_packed = True
        else:
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
        moe_runner_backend = get_moe_runner_backend()
        if moe_runner_backend.is_auto():
            moe_runner_backend = MoeRunnerBackend.TRITON

        if moe_runner_backend.is_aiter() and _is_dcu:
            self.runner = MoeRunner(MoeRunnerBackend.AITER, moe_runner_config)
        elif moe_runner_backend.is_triton():
            self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)
        else:
            raise ValueError(f"Unsupported MoE runner backend: {moe_runner_backend}")

    def get_triton_quant_info(self, layer: torch.nn.Module) -> TritonMoeQuantInfo:
        return TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            use_fp8_w8a8=True,
            per_channel_quant=True,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
        bias: Optional[torch.Tensor] = None,
        i_q: Optional[torch.Tensor] = None,
        i_s: Optional[torch.Tensor] = None,
    ) -> CombineInput:
        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        moe_runner_config = self.moe_runner_config

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        if _is_dcu and self.runner.runner_backend.is_aiter():
            from sglang.srt.layers.moe.moe_runner.aiter import (
                get_aiter_w8a8_fp8_quant_info,
            )

            quant_info = get_aiter_w8a8_fp8_quant_info(layer)
            combine_input = self.runner.run(dispatch_output, quant_info)
            if bias is not None:
                return StandardCombineInput(
                    hidden_states=combine_input.hidden_states + bias
                )
            return combine_input

        if _is_dcu and _use_fp8_w8a8_moe:
            if (getattr(layer.w13_weight, "_w8a8_fp8_packed", False)
                or getattr(layer.w2_weight, "_w8a8_fp8_packed", False)):
                topk_weights, topk_ids, _ = topk_output
                use_prequant_input = i_q is not None and i_s is not None
                if moe_runner_config.apply_router_weight_on_input:
                    assert topk_weights.dim() == 2, "`topk_weights` should be (num_tokens, topk)"
                    _, tk = topk_weights.shape
                    assert tk == 1, "DCU marlin path: apply_router_weight_on_input requires topk=1"
                    x = x * topk_weights.to(x.dtype)
                    topk_weights = torch.ones_like(topk_weights, dtype=torch.float32)
                    # Router-weighted input no longer matches precomputed rms-quant activations.
                    use_prequant_input = False
                from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe_fp8_w8a8
                origin_w1_shape = getattr(layer.w13_weight, "w1_shape", None)
                origin_w2_shape = getattr(layer.w2_weight, "w2_shape", None)
                output = fused_moe_fp8_w8a8(
                    hidden_states=x,
                    w1=layer.w13_weight,
                    w2=layer.w2_weight,
                    w1_scale=layer.w13_weight_scale,
                    w2_scale=layer.w2_weight_scale,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    global_num_experts=self.moe_runner_config.num_experts,
                    inplace=True,
                    origin_w1_shape=origin_w1_shape,
                    origin_w2_shape=origin_w2_shape,
                    routed_scaling_factor=moe_runner_config.routed_scaling_factor,
                    bias=bias,
                    hidden_states_fp8_input=i_q if use_prequant_input else None,
                    hidden_states_scale_fp8_input=i_s if use_prequant_input else None,
                )
                return StandardCombineInput(hidden_states=output)
        quant_info = TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            use_fp8_w8a8=True,
            per_channel_quant=True,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
        )
        return self.runner.run(dispatch_output, quant_info)
