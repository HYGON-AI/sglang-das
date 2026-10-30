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

import torch

from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_scheme import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.utils import set_weight_attrs

_PACK_FACTOR = 8  # eight INT4 values per INT32


class HCUCompressedTensorsW4A8Int8DynamicMoE(CompressedTensorsMoEScheme):
    """Compressed-tensors channelwise INT4 weights x dynamic per-token INT8.

    On-disk layout (pack-quantized, not GPTQ-transposed):
      weight_packed: int32 [out, in // 8], offset-binary (q + 8), low nibble first
      weight_scale:  [out, 1] per-output-channel
    Loaded as [E, out, in // 8] so the int32 -> uint8 view directly yields the
    LightOp W4A8 layout [E, N, K // 2].
    """

    def __init__(self, quant_config, weight_quant, num_gpu_experts: int = -1):
        if (
            weight_quant.num_bits != 4
            or weight_quant.strategy != "channel"
            or not weight_quant.symmetric
            or weight_quant.actorder is not None
        ):
            raise ValueError(
                "HCU W4A8 MoE requires symmetric channelwise INT4 without actorder, "
                f"got {weight_quant}"
            )
        self.quant_config = quant_config
        self.weight_quant = weight_quant

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

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

        inter = intermediate_size_per_partition
        assert hidden_size % _PACK_FACTOR == 0 and inter % _PACK_FACTOR == 0, (
            f"hidden={hidden_size}, inter_per_partition={inter} must be multiples of 8"
        )
        # [E, out, in // 8]: default FusedMoE loader shards w13 on dim 0 (out)
        # and w2 on dim 1 (in // 8); no GPTQ transpose (is_transposed unset).
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * inter,
                hidden_size // _PACK_FACTOR,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                inter // _PACK_FACTOR,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Checkpoint carries weight_shape tensors; keep a sink for them.
        w13_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )
        layer.register_parameter("w13_weight_shape", w13_weight_shape)
        set_weight_attrs(w13_weight_shape, extra_weight_attrs)
        w2_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )
        layer.register_parameter("w2_weight_shape", w2_weight_shape)
        set_weight_attrs(w2_weight_shape, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, 2 * inter, 1, dtype=torch.float32),
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

        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    @staticmethod
    def _convert_packed_weight(weight: torch.Tensor) -> torch.Tensor:
        """Convert compressed-tensors INT4 bytes to the LightOp layout."""
        weight = weight.view(torch.uint8)
        high_nibble = weight >> 4
        # compressed-tensors stores (q + 8) low-nibble first.  LightOp reads
        # signed two's-complement INT4 high-nibble first.
        weight <<= 4
        weight |= high_nibble
        weight ^= 0x88
        return weight.view(torch.int8)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "is_w4a8_converted", False):
            return

        # [E, N, K // 8] int32 -> [E, N, K // 2] bytes (little-endian, so the
        # byte order matches the on-disk element order), then fix nibble order
        # and offset-binary -> two's complement for LightOp.
        for name in ("w13_weight_packed", "w2_weight_packed"):
            packed = getattr(layer, name).data.contiguous().view(torch.uint8)
            setattr(
                layer,
                name,
                torch.nn.Parameter(
                    self._convert_packed_weight(packed), requires_grad=False
                ),
            )

        # LightOp expands each signed INT4 nibble into the high half of int8,
        # multiplying its integer value by 16.  Compensate in the scale.
        # Keep [E, N, 1] with canonical (N, 1, 1) strides.
        for name in ("w13_weight_scale", "w2_weight_scale"):
            scale = getattr(layer, name).data.float().contiguous().div_(16)
            setattr(layer, name, torch.nn.Parameter(scale, requires_grad=False))

        hidden = layer.w2_weight_packed.shape[1]
        assert layer.w13_weight_packed.shape[2] * 2 == hidden, (
            f"w13 {tuple(layer.w13_weight_packed.shape)} vs hidden {hidden}"
        )
        assert (
            layer.w2_weight_packed.shape[2] * 2 * 2 == layer.w13_weight_packed.shape[1]
        ), (
            f"w2 {tuple(layer.w2_weight_packed.shape)} vs "
            f"w13 {tuple(layer.w13_weight_packed.shape)}"
        )
        layer.is_w4a8_converted = True

    def apply_weights(self, layer: torch.nn.Module, dispatch_output):
        from lightop._lmslim_native.vllm_compat.fused_moe_cache import get_moe_cache

        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker
        from sglang.srt.layers.moe.topk import apply_topk_weights_cpu
        from sglang.srt.layers.quantization.slimquant_w4a8 import (
            fused_experts_impl_w4a8_triton,
        )

        x = dispatch_output.hidden_states
        if DispatchOutputChecker.format_is_standard(dispatch_output):
            from sglang.srt.layers.moe.token_dispatcher.standard import (
                StandardCombineInput,
            )

            topk_weights = dispatch_output.topk_output.topk_weights
            topk_ids = dispatch_output.topk_output.topk_ids
            combine_input_cls = StandardCombineInput
        elif DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            from sglang.srt.layers.moe.token_dispatcher.deepep import (
                DeepEPNormalCombineInput,
            )

            topk_weights = dispatch_output.topk_weights
            topk_ids = dispatch_output.topk_ids
            combine_input_cls = DeepEPNormalCombineInput
        else:
            raise ValueError(
                f"Unsupported W4A8 dispatch format: {dispatch_output.format}"
            )
        x, topk_weights = apply_topk_weights_cpu(
            self.moe_runner_config.apply_router_weight_on_input, topk_weights, x
        )
        cache13 = get_moe_cache(
            topk_ids.shape[1],
            layer.w13_weight_packed.shape[1],
            layer.w2_weight_packed.shape[1],
            device=x.device,
            dtype=x.dtype,
        )
        output = fused_experts_impl_w4a8_triton(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            topk_weights,
            topk_ids,
            cache13,
            activation=self.moe_runner_config.activation,
            apply_router_weight_on_input=self.moe_runner_config.apply_router_weight_on_input,
            global_num_experts=self.moe_runner_config.num_experts,
            expert_map=getattr(layer, "expert_map", None),
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            routed_scaling_factor=(
                self.moe_runner_config.routed_scaling_factor
                if self.moe_runner_config.routed_scaling_factor is not None
                else 1.0
            ),
            shared_output=None,
            swiglu_limit=self.moe_runner_config.swiglu_limit,
        )
        if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            return combine_input_cls(
                hidden_states=output,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
            )
        return combine_input_cls(hidden_states=output)
