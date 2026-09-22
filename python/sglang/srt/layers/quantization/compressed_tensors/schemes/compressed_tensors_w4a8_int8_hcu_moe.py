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

from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16_moe import (
    CompressedTensorsWNA16TritonMoE,
)


class HCUCompressedTensorsW4A8Int8DynamicMoE(CompressedTensorsWNA16TritonMoE):
    """Compressed-tensors INT4 weights with dynamic per-token INT8 activations."""

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
        super().process_weights_after_loading(layer)

        # The parent conversion produces [E, N, K / 2], but its packed value
        # convention differs from the one consumed by LightOp.
        layer.w13_weight_packed = torch.nn.Parameter(
            self._convert_packed_weight(layer.w13_weight_packed),
            requires_grad=False,
        )
        layer.w2_weight_packed = torch.nn.Parameter(
            self._convert_packed_weight(layer.w2_weight_packed),
            requires_grad=False,
        )
        # A transposed [E, 1, N] channel scale is considered contiguous by
        # PyTorch even though its [E, N, 1] stride is (N, 1, N).  LightOp uses
        # the singleton-dimension stride explicitly, so canonicalize it to
        # (N, 1, 1) to avoid out-of-bounds scale reads.
        # LightOp expands each signed INT4 nibble into the high half of int8,
        # multiplying its integer value by 16.  Compensate in the scale.
        layer.w13_weight_scale = torch.nn.Parameter(
            layer.w13_weight_scale.squeeze(-1)
            .contiguous()
            .float()
            .unsqueeze(-1)
            .div_(16),
            requires_grad=False,
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            layer.w2_weight_scale.squeeze(-1)
            .contiguous()
            .float()
            .unsqueeze(-1)
            .div_(16),
            requires_grad=False,
        )

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
