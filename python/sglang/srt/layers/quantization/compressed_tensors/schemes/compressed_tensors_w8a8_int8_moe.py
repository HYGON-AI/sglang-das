from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
from compressed_tensors.quantization import QuantizationStrategy

from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    NPUW8A8Int8DynamicMoEMethod,
)
from sglang.srt.layers.moe import (
    MoeRunner,
    MoeRunnerBackend,
    MoeRunnerConfig,
)
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.utils import get_bool_env_var, is_dcu, is_hip, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

__all__ = [
    "CompressedTensorsW8A8Int8MoE",
    "NPUCompressedTensorsW8A8Int8DynamicMoE",
]

logger = logging.getLogger(__name__)

_is_hip = is_hip()
_is_dcu = is_dcu()
_use_aiter_moe = get_bool_env_var("SGLANG_ROCM_USE_AITER_MOE", default="true")


class NPUCompressedTensorsW8A8Int8DynamicMoE(CompressedTensorsMoEScheme):

    def __init__(self, weight_quant, input_quant):
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.kernel = NPUW8A8Int8DynamicMoEMethod()

        self.static_input_scales = not self.input_quant.dynamic
        per_channel = (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN
        )
        if not per_channel:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found "
                f"{self.weight_quant}, {self.input_quant}"
            )

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found static input scales."
            )

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

        params_dtype = torch.int8

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
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
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        assert self.weight_quant.strategy == QuantizationStrategy.CHANNEL
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        assert not self.static_input_scales
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        return self.kernel.apply(layer, dispatch_output)


class CompressedTensorsW8A8Int8MoE(CompressedTensorsMoEScheme):
    """INT8 W8A8 MoE scheme for GPU/DCU (non-NPU).

    Supports channelwise dynamic per-token quantization for MoE layers.
    Uses aiter MoE when available (SGLANG_ROCM_USE_AITER_MOE=true),
    with a Triton fallback via MoeRunner.
    """

    def __init__(self, weight_quant, input_quant):
        self.weight_quant = weight_quant
        self.input_quant = input_quant

        per_channel = (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN
        )
        if not per_channel:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found "
                f"{self.weight_quant}, {self.input_quant}"
            )

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found static input scales."
            )

    @classmethod
    def get_min_capability(cls) -> int:
        # ampere and up
        return 80

    @staticmethod
    def _shuffle_w8a8_gemm1(weight_data):
        from aiter.ops.shuffle import moe_layout_shuffle_gemm1

        w_i8 = weight_data.to(torch.int8)
        return moe_layout_shuffle_gemm1(w_i8)

    @staticmethod
    def _shuffle_w8a8_gemm2(weight_data):
        from aiter.ops.shuffle import moe_layout_shuffle_gemm2

        w_i8 = weight_data.to(torch.int8)
        return moe_layout_shuffle_gemm2(w_i8)

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

        params_dtype = torch.int8

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
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
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        assert self.weight_quant.strategy == QuantizationStrategy.CHANNEL
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        assert not self.static_input_scales
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight_scale = torch.nn.Parameter(
            layer.w13_weight_scale.data, requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            layer.w2_weight_scale.data, requires_grad=False
        )
        if not _use_aiter_moe:
            return
        shuffled_w13 = self._shuffle_w8a8_gemm1(layer.w13_weight)
        layer.w13_weight = torch.nn.Parameter(
            shuffled_w13.view(*layer.w13_weight.shape), requires_grad=False
        )
        shuffled_w2 = self._shuffle_w8a8_gemm2(layer.w2_weight)
        layer.w2_weight = torch.nn.Parameter(
            shuffled_w2.view(*layer.w2_weight.shape), requires_grad=False
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
        bias: Optional[torch.Tensor] = None,
        i_q: Optional[torch.Tensor] = None,
        i_s: Optional[torch.Tensor] = None,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_weights, topk_ids, router_logits = dispatch_output.topk_output

        if _use_aiter_moe:
            from aiter.moe import get_aiter_moe_config, aiter_moe, MoeQuantType

            E = layer.w13_weight.size(0)
            K = x.size(-1)
            N1 = layer.w13_weight.size(1)
            topk = topk_ids.size(1)
            w1_input = layer.w13_weight.view(E, N1, K)
            w2_input = layer.w2_weight.view(E, K, N1 // 2)

            status, moe_cfg = get_aiter_moe_config(
                M=x.shape[0],
                E=E,
                N1=N1,
                N2=N1 // 2,
                K=K,
                top_k=topk,
                block_size=None,
                dtype=x.dtype,
                quant_type=MoeQuantType.W8A8,
            )
            if not status:
                raise RuntimeError(
                    "aiter backend did not find a valid w8a8 moe config for "
                    f"M={x.shape[0]}, E={E}, N1={N1}, N2={N1 // 2}, K={K}, topk={topk}, "
                    f"dtype={x.dtype}"
                )
            output = aiter_moe(
                hidden_states=x,
                w1=w1_input,
                w2=w2_input,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                moe_config=moe_cfg,
                activation=getattr(layer, "activation", "silu"),
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=getattr(layer, "w13_input_scale", None),
                a2_scale=getattr(layer, "w2_input_scale", None),
                global_num_experts=E,
                expert_map=getattr(layer, "expert_map", None),
            )
            return StandardCombineInput(hidden_states=output)

        # Triton fallback: route through MoeRunner with INT8 W8A8 quant info
        quant_info = TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            use_int8_w8a8=True,
            per_channel_quant=True,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
        )
        return self.runner.run(dispatch_output, quant_info)
