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
_HCU_AITER_INT8_NOSHUFFLE_ATTR = "_sglang_hcu_aiter_int8_noshuffle"
_hcu_aiter_int8_noshuffle_logged: set[tuple[int, ...]] = set()


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
        """Cast non-fp32 scales to fp32. Kernels require fp32; some INT8
        checkpoints (Qwen3.8-Flash-Next) store bf16. fp32 checkpoints are a no-op.
        """
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

    def _int8_moe_lookup_shape(
        self, layer: torch.nn.Module
    ) -> Optional[tuple[int, int, int, int, int]]:
        w13 = getattr(layer, "w13_weight", None)
        w2 = getattr(layer, "w2_weight", None)
        if w13 is None or w2 is None or w13.ndim != 3 or w2.ndim != 3:
            return None
        E, N1, K = (int(dim) for dim in w13.shape)
        E2, N2, n = (int(dim) for dim in w2.shape)
        if E != E2 or N2 != K or N1 != 2 * n:
            return None
        top_k = getattr(layer, "top_k", None)
        if top_k is None:
            cfg = getattr(layer, "moe_runner_config", None) or getattr(
                self, "moe_runner_config", None
            )
            top_k = getattr(cfg, "top_k", None) if cfg is not None else None
        if not top_k:
            return None
        return E, N1, N2, K, int(top_k)

    def _probe_hcu_aiter_int8_noshuffle(self, layer: torch.nn.Module) -> bool:
        """Return True only for HCU + AITER + a matching no-shuffle INT8 ASM table.

        This is the Qwen3.8-Flash-Next path. Other INT8 MoE shapes (for example
        GLM-5.2) keep the legacy shuffle + aiter_moe / Triton behavior.
        """
        if not _is_hcu:
            return False
        if not will_use_aiter_moe():
            return False
        if not get_moe_a2a_backend().supports_aiter():
            return False
        if self.weight_quant.strategy != QuantizationStrategy.CHANNEL:
            return False
        if self.input_quant.strategy != QuantizationStrategy.TOKEN:
            return False
        if getattr(self, "static_input_scales", False):
            return False

        shape = self._int8_moe_lookup_shape(layer)
        if shape is None:
            return False
        E, N1, N2, K, top_k = shape

        try:
            from aiter.moe import MoeQuantType, MoeSolutionType, get_aiter_moe_config

            kwargs = dict(
                M=1,
                E=E,
                N1=N1,
                N2=N2,
                K=K,
                top_k=top_k,
                block_size=0,
                dtype=torch.bfloat16,
                quant_type=MoeQuantType.W8A8,
                activation="silu",
                spec_sol_type=MoeSolutionType.ASM,
                use_shuffle=0,
            )
            try:
                status, config = get_aiter_moe_config(**kwargs)
            except TypeError:
                kwargs.pop("spec_sol_type", None)
                try:
                    status, config = get_aiter_moe_config(**kwargs)
                except TypeError:
                    kwargs.pop("use_shuffle", None)
                    kwargs.pop("activation", None)
                    status, config = get_aiter_moe_config(**kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "HCU AITER INT8 no-shuffle probe failed for "
                "E=%s N1=%s N2=%s K=%s topk=%s (%s); using legacy INT8 MoE path",
                E,
                N1,
                N2,
                K,
                top_k,
                exc,
            )
            return False

        matched = bool(status and config is not None)
        if matched:
            need_shuffle = bool(getattr(config, "need_shuffle", False))
            solution_type = getattr(config, "solution_type", None)
            if need_shuffle:
                matched = False
            elif solution_type is not None:
                from aiter.moe import MoeSolutionType

                if solution_type != MoeSolutionType.ASM:
                    matched = False

        if shape not in _hcu_aiter_int8_noshuffle_logged:
            _hcu_aiter_int8_noshuffle_logged.add(shape)
            if matched:
                logger.info(
                    "Using HCU AITER INT8 no-shuffle MoE: "
                    "E=%s N1=%s N2=%s K=%s topk=%s",
                    E,
                    N1,
                    N2,
                    K,
                    top_k,
                )
            else:
                logger.info(
                    "No matching HCU AITER INT8 no-shuffle ASM config for "
                    "E=%s N1=%s N2=%s K=%s topk=%s; using legacy INT8 MoE path",
                    E,
                    N1,
                    N2,
                    K,
                    top_k,
                )
        return matched

    def _should_use_hcu_aiter_int8_noshuffle(self, layer: torch.nn.Module) -> bool:
        cached = getattr(layer, _HCU_AITER_INT8_NOSHUFFLE_ATTR, None)
        if cached is True:
            return True
        matched = self._probe_hcu_aiter_int8_noshuffle(layer)
        if matched:
            setattr(layer, _HCU_AITER_INT8_NOSHUFFLE_ATTR, True)
            return True
        # Only cache a negative result when the MoE shape is complete, so an
        # early call without top_k can still probe later.
        if self._int8_moe_lookup_shape(layer) is not None:
            setattr(layer, _HCU_AITER_INT8_NOSHUFFLE_ATTR, False)
        return False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._normalize_weight_scales(layer)
        if self._should_use_hcu_aiter_int8_noshuffle(layer):
            from sglang.srt.layers.moe.moe_runner.aiter import (
                process_weights_after_loading_aiter_w8a8_int8,
            )

            process_weights_after_loading_aiter_w8a8_int8(layer)
            return
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
        moe_runner_backend = get_moe_runner_backend()
        use_noshuffle = self._should_use_hcu_aiter_int8_noshuffle(layer)
        if moe_runner_backend.is_auto():
            moe_runner_backend = (
                MoeRunnerBackend.AITER if use_noshuffle else MoeRunnerBackend.TRITON
            )
        elif moe_runner_backend.is_aiter() and not use_noshuffle:
            # Explicit aiter without a matching no-shuffle table: keep the
            # legacy apply path (aiter_moe + shuffled weights), which uses a
            # Triton MoeRunner placeholder like the pre-Flash-Next INT8 code.
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

        if (
            self._should_use_hcu_aiter_int8_noshuffle(layer)
            and self.runner.runner_backend.is_aiter()
        ):
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