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

import logging
from typing import TYPE_CHECKING, Optional

import torch
from compressed_tensors.quantization import QuantizationStrategy

from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUW8A8Int8MoEMethod,
)
from sglang.srt.layers.moe.moe_runner import MoeRunner, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.utils import (
    MoeRunnerBackend,
    get_moe_a2a_backend,
    get_moe_runner_backend,
    will_use_aiter_moe,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.utils import get_bool_env_var, is_hcu, is_hip, set_weight_attrs

if TYPE_CHECKING:
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
_is_hcu = is_hcu()
_use_aiter_moe = _is_hip and get_bool_env_var(
    "SGLANG_ROCM_USE_AITER_MOE", default="true"
)


class NPUCompressedTensorsW8A8Int8DynamicMoE(CompressedTensorsMoEScheme):

    def __init__(self, weight_quant, input_quant):
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.w13_kernel = NPUW8A8Int8MoEMethod()
        self.w2_kernel = NPUW8A8Int8MoEMethod()

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
        self.w13_kernel.process_weights_after_loading(layer, "w13")
        self.w2_kernel.process_weights_after_loading(layer, "w2")

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        layer.w13_kernel = self.w13_kernel
        layer.w2_kernel = self.w2_kernel
        moe_runner_config.layer = layer
        self.moe_runner_config = moe_runner_config
        backend = get_moe_runner_backend()
        if backend.is_auto():
            backend = MoeRunnerBackend.ASCEND
        self.runner = MoeRunner(backend, moe_runner_config)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.moe_runner.ascend import AscendQuantInfo

        quant_info = AscendQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            w13_weight_scale=layer.w13_weight_scale,
            w2_weight_scale=layer.w2_weight_scale,
            w13_weight_offset=layer.w13_weight_offset,
            w2_weight_offset=layer.w2_weight_offset,
            w13_weight_bias=getattr(layer, "w13_weight_bias", None),
            w2_weight_bias=getattr(layer, "w2_weight_bias", None),
            w13_scale_bias=getattr(layer, "w13_scale_bias", None),
            w2_scale_bias=getattr(layer, "w2_scale_bias", None),
        )
        return self.runner.run(dispatch_output, quant_info)


class CompressedTensorsW8A8Int8MoE(CompressedTensorsMoEScheme):
    """INT8 W8A8 MoE scheme for GPU/HCU (non-NPU).

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

    @staticmethod
    def _normalize_weight_scales(layer: torch.nn.Module) -> None:
        """Qwen3.8-Flash-Next INT8 checkpoints store scales as bf16."""
        for name in ("w13_weight_scale", "w2_weight_scale"):
            scale = getattr(layer, name)
            if scale.dtype != torch.float32:
                setattr(
                    layer,
                    name,
                    torch.nn.Parameter(scale.to(torch.float32), requires_grad=False),
                )
            else:
                setattr(
                    layer,
                    name,
                    torch.nn.Parameter(scale.data, requires_grad=False),
                )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._normalize_weight_scales(layer)
        if _is_hcu and will_use_aiter_moe():
            from sglang.srt.layers.moe.moe_runner.aiter import (
                process_weights_after_loading_aiter_w8a8_int8,
            )

            process_weights_after_loading_aiter_w8a8_int8(layer)
            return
        if not _use_aiter_moe or _is_hcu:
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
        moe_runner_backend = get_moe_runner_backend()
        if moe_runner_backend.is_auto():
            if will_use_aiter_moe() and get_moe_a2a_backend().supports_aiter():
                moe_runner_backend = MoeRunnerBackend.AITER
            else:
                moe_runner_backend = MoeRunnerBackend.TRITON
        if moe_runner_backend.is_aiter() or moe_runner_backend.is_triton():
            self.runner = MoeRunner(moe_runner_backend, moe_runner_config)
        else:
            raise ValueError(
                f"CompressedTensorsW8A8Int8MoE does not support "
                f"moe_runner_backend={moe_runner_backend}."
            )

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

        if _is_hcu and self.runner.runner_backend.is_aiter():
            from sglang.srt.layers.moe.moe_runner.aiter import (
                get_aiter_w8a8_int8_quant_info,
            )

            quant_info = get_aiter_w8a8_int8_quant_info(layer)
            combine_input = self.runner.run(dispatch_output, quant_info)
            if bias is not None:
                return StandardCombineInput(
                    hidden_states=combine_input.hidden_states + bias
                )
            return combine_input

        if _use_aiter_moe and not _is_hcu:
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