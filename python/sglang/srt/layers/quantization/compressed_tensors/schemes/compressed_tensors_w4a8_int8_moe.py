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
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUW4A8Int8MoEMethod,
)
from sglang.srt.layers.moe.moe_runner import MoeRunner, MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeRunnerBackend, get_moe_runner_backend
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.utils import is_hcu, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

__all__ = [
    "HCUCompressedTensorsW4A8Int8DynamicMoE",
    "NPUCompressedTensorsW4A8Int8DynamicMoE",
]


logger = logging.getLogger(__name__)
_is_hcu = is_hcu()


class NPUCompressedTensorsW4A8Int8DynamicMoE(CompressedTensorsMoEScheme):
    ### TODO: Get rid of code duplication with python/sglang/srt/modelslim/modelslim_moe.py @OrangeRedeng @TamirBaydasov
    def __init__(self, quantization_config) -> None:
        self.group_size = 0
        self.is_per_channel_weight = self.group_size == 0
        self.tp_size = 1
        self.activation_use_clip = (
            quantization_config.get("config_groups", {})
            .get("group_1", {})
            .get("activation_use_clip", False)
        )
        self.w13_kernel = NPUW4A8Int8MoEMethod(
            is_per_channel_weight=self.is_per_channel_weight,
            activation_use_clip=self.activation_use_clip,
        )
        self.w2_kernel = NPUW4A8Int8MoEMethod(
            is_per_channel_weight=self.is_per_channel_weight,
            activation_use_clip=self.activation_use_clip,
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        self.num_experts = num_experts
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        # >> weight
        w13_output_size = intermediate_size_per_partition
        w2_output_size = hidden_size // 2
        w13_weight = torch.nn.Parameter(
            torch.empty(num_experts, w13_output_size, hidden_size, dtype=torch.int8),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w2_output_size,
                intermediate_size_per_partition,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # >> scale
        weight_scale_dtype = torch.int64 if self.activation_use_clip else torch.float32
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                1,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, 1, dtype=weight_scale_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # >> offset
        w13_weight_offset = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_offset", w13_weight_offset)
        set_weight_attrs(w13_weight_offset, extra_weight_attrs)

        w2_weight_offset = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_offset", w2_weight_offset)
        set_weight_attrs(w2_weight_offset, extra_weight_attrs)

        # >>> special param for w4a8
        if self.activation_use_clip:
            self._init_activation_clip_params(
                layer,
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                extra_weight_attrs,
            )
        else:
            self._init_extra_scale_params(
                layer,
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                extra_weight_attrs,
            )

    def _init_activation_clip_params(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        extra_weight_attrs: dict,
    ) -> None:
        """
        Initializes bias and alpha parameters for quantization schemes that use activation clipping.

        This helper registers `w13_bias`, `w2_bias`, and `w2_alpha`, which are required to
        shift and scale the activations or outputs to compensate for the precision loss
        introduced by clamping activations.
        """
        w13_bias = torch.nn.Parameter(
            torch.ones(
                num_experts, 2 * intermediate_size_per_partition, dtype=torch.float
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_bias", w13_bias)
        set_weight_attrs(w13_bias, extra_weight_attrs)

        w2_bias = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, dtype=torch.float),
            requires_grad=False,
        )
        layer.register_parameter("w2_bias", w2_bias)
        set_weight_attrs(w2_bias, extra_weight_attrs)

        w2_alpha = torch.nn.Parameter(
            torch.ones(num_experts, dtype=torch.float), requires_grad=False
        )
        layer.register_parameter("w2_alpha", w2_alpha)
        set_weight_attrs(w2_alpha, extra_weight_attrs)

    def _init_extra_scale_params(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        extra_weight_attrs: dict,
    ) -> None:
        """
        Initializes additional scaling, offset, and bias parameters for quantization schemes without activation clipping.

        This method registers the following parameters:
        1. Scale Biases: `w13_scale_bias` and `w2_scale_bias`.
        2. Secondary Quantization Params (initialized only for grouped quantization):
            `w13_weight_scale_second`, `w13_weight_offset_second`,
            `w2_weight_scale_second`, and `w2_weight_offset_second`.
        """
        if not self.is_per_channel_weight:
            w13_weight_scale_second = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    hidden_size // self.group_size,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale_second", w13_weight_scale_second)
            set_weight_attrs(w13_weight_scale_second, extra_weight_attrs)

            w13_weight_offset_second = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    hidden_size // self.group_size,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter(
                "w13_weight_offset_second", w13_weight_offset_second
            )
            set_weight_attrs(w13_weight_offset_second, extra_weight_attrs)

            w2_weight_scale_second = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition // self.group_size,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_scale_second", w2_weight_scale_second)
            set_weight_attrs(w2_weight_scale_second, extra_weight_attrs)

            w2_weight_offset_second = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition // self.group_size,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_offset_second", w2_weight_offset_second)
            set_weight_attrs(w2_weight_offset_second, extra_weight_attrs)

        w13_scale_bias = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scale_bias", w13_scale_bias)
        set_weight_attrs(w13_scale_bias, extra_weight_attrs)

        w2_scale_bias = torch.nn.Parameter(
            torch.empty(
                num_experts, hidden_size, 16 // self.tp_size, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_scale_bias", w2_scale_bias)
        set_weight_attrs(w2_scale_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "is_hcu_w4a8_deep_gemm_converted", False):
            return

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
            w13_scale_bias=layer.w13_scale_bias,
            w2_scale_bias=layer.w2_scale_bias,
            w13_weight_bias=getattr(layer, "w13_weight_bias", None),
            w2_weight_bias=getattr(layer, "w2_weight_bias", None),
        )
        return self.runner.run(dispatch_output, quant_info)


class HCUCompressedTensorsW4A8Int8DynamicMoE(CompressedTensorsMoEScheme):
    """HCU DeepGEMM path for dynamic-activation compressed-tensors W4A8 MoE."""

    def __init__(self, *args, **kwargs) -> None:
        if not _is_hcu:
            raise RuntimeError(
                "HCUCompressedTensorsW4A8Int8DynamicMoE is only available on HCU"
            )
        from .compressed_tensors_wNa16_moe import (
            CompressedTensorsWNA16MoE,
        )

        CompressedTensorsWNA16MoE.__init__(self, *args, **kwargs)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from .compressed_tensors_wNa16_moe import (
            CompressedTensorsWNA16MoE,
        )

        CompressedTensorsWNA16MoE.create_weights(
            self,
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    @staticmethod
    def _convert_packed_weight(weight: torch.Tensor) -> torch.Tensor:
        weight = weight.view(torch.uint8)
        high_nibble = weight >> 4
        # compressed-tensors stores (q + 8) low-nibble first. LightOp reads
        # signed two's-complement INT4 high-nibble first.
        weight <<= 4
        weight |= high_nibble
        weight ^= 0x88
        return weight.view(torch.int8)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ) -> None:
        backend = get_moe_runner_backend()
        if backend.is_auto():
            backend = MoeRunnerBackend.DEEP_GEMM
        if not backend.is_deep_gemm():
            raise ValueError(
                "HCU compressed-tensors W4A8 MoE currently supports only the "
                f"deep_gemm runner, got {backend.value!r}"
            )
        self.moe_runner_config = moe_runner_config

    @staticmethod
    def _pad_deep_gemm_w2(
        w2: torch.Tensor,
        intermediate_size: int,
        padded_intermediate_size: int,
    ) -> torch.Tensor:
        if padded_intermediate_size == intermediate_size:
            return w2
        return F.pad(w2, (0, (padded_intermediate_size - intermediate_size) // 2))

    @staticmethod
    def _guard_deep_gemm_scale_storage(scale: torch.Tensor) -> torch.Tensor:
        # Keep the small channel-scale tensor away from the end of a 2 MiB
        # device mapping because the HCU kernel reads scales vectorially.
        guard_elements = (2 * 1024 * 1024) // scale.element_size()
        storage = torch.empty(
            scale.numel() + guard_elements,
            dtype=scale.dtype,
            device=scale.device,
        )
        storage[: scale.numel()].copy_(scale.reshape(-1))
        return storage[: scale.numel()].view_as(scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Convert the checkpoint layout [E, K/8, N] int32 into the byte-packed
        # [E, N, K/2] layout expected by HCU DeepGEMM.
        w13 = (
            layer.w13_weight_packed.data.transpose(1, 2)
            .contiguous()
            .view(torch.uint8)
        )
        w2 = layer.w2_weight_packed.data.transpose(1, 2).contiguous().view(torch.uint8)
        w13 = self._convert_packed_weight(w13)
        w2 = self._convert_packed_weight(w2)
        # Canonicalize singleton-dimension strides because LightOp consumes
        # that stride directly for per-channel scales.
        w13_scale = (
            layer.w13_weight_scale.data.transpose(1, 2)
            .squeeze(-1)
            .contiguous()
            .float()
            .unsqueeze(-1)
        )
        w2_scale = (
            layer.w2_weight_scale.data.transpose(1, 2)
            .squeeze(-1)
            .contiguous()
            .float()
            .unsqueeze(-1)
        )

        from sglang.srt.layers.quantization.w4a8_utils import (
            w4a8_weight_repack_impl,
        )

        intermediate_size = w2.shape[2] * 2
        padded_intermediate_size = ((intermediate_size + 127) // 128) * 128
        w2 = self._pad_deep_gemm_w2(
            w2, intermediate_size, padded_intermediate_size
        )
        layer.w4a8_intermediate_size = intermediate_size
        layer.w4a8_padded_intermediate_size = padded_intermediate_size
        w13 = w4a8_weight_repack_impl(w13, use_deepep=True)
        w2 = w4a8_weight_repack_impl(w2, use_deepep=True)

        # LightOp expands each signed INT4 nibble into the high half of int8.
        w13_scale.div_(16)
        w2_scale.div_(16)
        w13_scale = self._guard_deep_gemm_scale_storage(w13_scale)
        w2_scale = self._guard_deep_gemm_scale_storage(w2_scale)

        layer.w13_weight_packed = torch.nn.Parameter(w13, requires_grad=False)
        layer.w2_weight_packed = torch.nn.Parameter(w2, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(w13_scale, requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)
        if hasattr(layer, "dispatcher"):
            layer.dispatcher.set_quant_config(
                {
                    "normal_dispatcher_output_dtype": "bf16",
                    "normal_expert_alignment": 256,
                }
            )
        layer.is_hcu_w4a8_deep_gemm_converted = True

    def _run_deep_gemm_masked(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        masked_m: torch.Tensor,
        expected_m: int,
    ) -> torch.Tensor:
        from lightop.quant import per_token_quant_int8

        if self.moe_runner_config.activation != "silu":
            raise ValueError("HCU W4A8 DeepGEMM currently supports only SiLU")

        q_a1, q_a1_scale = per_token_quant_int8(hidden_states)
        gate_up = torch.empty(
            (
                hidden_states.shape[0],
                hidden_states.shape[1],
                layer.w13_weight_scale.shape[1],
            ),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.sglang.m_grouped_w4a8_gemm_nt_masked(
            q_a1,
            q_a1_scale,
            layer.w13_weight_packed,
            layer.w13_weight_scale,
            gate_up,
            masked_m,
            expected_m,
        )

        gate, up = gate_up.chunk(2, dim=-1)
        swiglu_limit = self.moe_runner_config.swiglu_limit
        if swiglu_limit is not None:
            gate.clamp_(max=swiglu_limit)
            up.clamp_(min=-swiglu_limit, max=swiglu_limit)
        activated = F.silu(gate) * up
        padded_intermediate_size = layer.w4a8_padded_intermediate_size
        if activated.shape[-1] != padded_intermediate_size:
            # GEMM2 K must be 128-aligned. This is the only remaining layout
            # copy; eliminating it requires a kernel that accepts K=288.
            activated = F.pad(
                activated, (0, padded_intermediate_size - activated.shape[-1])
            )
        q_a2, q_a2_scale = per_token_quant_int8(activated)

        output = torch.empty(
            (
                hidden_states.shape[0],
                hidden_states.shape[1],
                layer.w2_weight_scale.shape[1],
            ),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.sglang.m_grouped_w4a8_gemm_nt_masked(
            q_a2,
            q_a2_scale,
            layer.w2_weight_packed,
            layer.w2_weight_scale,
            output,
            masked_m,
            expected_m,
        )
        return output

    def _apply_deepep_normal_deep_gemm(
        self, layer: torch.nn.Module, dispatch_output
    ):
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            ep_gather,
            ep_scatter_no_scale,
        )
        from sglang.srt.layers.moe.token_dispatcher.deepep import (
            DeepEPNormalCombineInput,
        )

        x = dispatch_output.hidden_states
        topk_ids = dispatch_output.topk_ids
        topk_weights = dispatch_output.topk_weights
        counts = dispatch_output.num_recv_tokens_per_expert
        all_tokens = sum(counts)

        if x.dtype != torch.bfloat16 or dispatch_output.hidden_states_scale is not None:
            raise RuntimeError(
                "HCU W4A8 DeepGEMM requires unquantized BF16 DeepEP activations"
            )
        if all_tokens == 0:
            return DeepEPNormalCombineInput(
                hidden_states=torch.zeros_like(x),
                topk_ids=topk_ids,
                topk_weights=topk_weights,
            )

        num_local_experts = layer.w13_weight_scale.shape[0]
        if len(counts) != num_local_experts:
            raise RuntimeError(
                "DeepEP expert counts do not match local W4A8 experts: "
                f"{len(counts)} != {num_local_experts}"
            )

        valid_topk_ids = topk_ids[topk_ids >= 0].to(torch.int64)
        masked_m = torch.bincount(
            valid_topk_ids, minlength=num_local_experts
        ).to(torch.int32)
        expected_m = max(counts)
        padded_m = ((expected_m + 255) // 256) * 256

        # Scatter directly into the padded per-expert layout consumed by
        # masked DeepGEMM. output_index records these padded offsets, so the
        # gather can also read the GEMM result directly. This removes both
        # per-expert copy loops and both compact intermediate buffers.
        masked_x = torch.zeros(
            (num_local_experts, padded_m, x.shape[1]),
            dtype=x.dtype,
            device=x.device,
        )
        expert_start_loc = torch.arange(
            0,
            num_local_experts * padded_m,
            padded_m,
            dtype=torch.int32,
            device=x.device,
        )
        output_index = torch.full(
            topk_ids.shape, -1, dtype=torch.int32, device=x.device
        )
        ep_scatter_no_scale(
            x,
            topk_ids,
            masked_m,
            expert_start_loc,
            masked_x.view(-1, x.shape[1]),
            masked_m,
            output_index,
            hcu_use_preinitialized_expert_offsets=True,
        )

        masked_output = self._run_deep_gemm_masked(
            layer, masked_x, masked_m, expected_m
        )
        output = torch.empty_like(x)
        ep_gather(
            masked_output.view(-1, masked_output.shape[-1]),
            topk_ids,
            topk_weights,
            output_index,
            output,
        )
        return DeepEPNormalCombineInput(
            hidden_states=output,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    def apply_weights(self, layer: torch.nn.Module, dispatch_output):
        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        if not DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            raise ValueError(
                "HCU compressed-tensors W4A8 DeepGEMM requires DeepEP normal dispatch"
            )
        return self._apply_deepep_normal_deep_gemm(layer, dispatch_output)
