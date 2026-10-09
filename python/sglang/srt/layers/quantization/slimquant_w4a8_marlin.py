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

import logging
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe import (
    MoeRunner,
    MoeRunnerBackend,
    MoeRunnerConfig,
    get_deepep_mode,
    get_moe_a2a_backend,
)
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.slimquant_w4a8 import SlimQuantW4A8Int8LinearMethod
from sglang.srt.layers.quantization.w4a8_utils import w4a8_weight_repack_impl
from sglang.srt.utils import set_weight_attrs

# from sglang.srt.layers.moe.token_dispatcher.base import CombineInput


logger = logging.getLogger(__name__)

W4A8_TPMOE_BACKEND_ENV = "SGLANG_W4A8_TPMOE_BACKEND"
W4A8_TPMOE_BACKEND_AUTO = "auto"
W4A8_TPMOE_BACKEND_LIGHTOP = "lightop"
W4A8_TPMOE_BACKEND_AITER = "aiter"
_requested_backend = (
    os.getenv(W4A8_TPMOE_BACKEND_ENV, W4A8_TPMOE_BACKEND_AUTO).strip().lower()
)


_lmslim_w4a8_marlin_available = False
_aiter_w4a8_marlin_available = False

if _requested_backend in {W4A8_TPMOE_BACKEND_AUTO, W4A8_TPMOE_BACKEND_LIGHTOP}:
    try:
        from lightop.moe import (
            fused_experts_impl_w4a8_marlin,
        )

        _lmslim_w4a8_marlin_available = True
    except Exception:
        logger.info(
            "INFO: Please install lightop if you want to infer the quantitative model of moe.\n"
        )

if _requested_backend in {W4A8_TPMOE_BACKEND_AUTO, W4A8_TPMOE_BACKEND_AITER}:
    try:
        from aiter.moe import MoeQuantType, aiter_moe, get_aiter_moe_config
        from aiter.ops.shuffle import w4a8_moe_layout_shuffle_gemm2

        _aiter_w4a8_marlin_available = True
    except Exception:
        pass

if _requested_backend not in {
    W4A8_TPMOE_BACKEND_AUTO,
    W4A8_TPMOE_BACKEND_LIGHTOP,
    W4A8_TPMOE_BACKEND_AITER,
}:
    raise ValueError(
        f"Unsupported {W4A8_TPMOE_BACKEND_ENV}={_requested_backend!r}. "
        f"Supported values: {W4A8_TPMOE_BACKEND_AUTO!r}, "
        f"{W4A8_TPMOE_BACKEND_LIGHTOP!r}, {W4A8_TPMOE_BACKEND_AITER!r}."
    )

if _requested_backend == W4A8_TPMOE_BACKEND_AUTO:
    if _lmslim_w4a8_marlin_available:
        _resolved_backend = W4A8_TPMOE_BACKEND_LIGHTOP
    elif _aiter_w4a8_marlin_available:
        _resolved_backend = W4A8_TPMOE_BACKEND_AITER
    else:
        raise RuntimeError(
            "Neither lightop nor aiter backend is available for w4a8 tpmoe."
        )
elif _requested_backend == W4A8_TPMOE_BACKEND_LIGHTOP:
    if not _lmslim_w4a8_marlin_available:
        raise RuntimeError(
            "lightop backend is selected for w4a8 tpmoe, but lightop is not available."
        )
    _resolved_backend = W4A8_TPMOE_BACKEND_LIGHTOP
else:
    if not _aiter_w4a8_marlin_available:
        raise RuntimeError(
            "aiter backend is selected for w4a8 tpmoe, but aiter is not available."
        )
    _resolved_backend = W4A8_TPMOE_BACKEND_AITER

logger.info(
    "[slimquant_w4a8_marlin] "
    f"requested_backend={_requested_backend}, "
    f"resolved_backend={_resolved_backend}"
)


class MarlinMoeWorkspace:
    """
    Singleton manager for device-specific workspace buffers used by w4a8 Marlin-MoE.
    global_reduce_buffer will take 1.5MB * cus (about 120MB for BW200) memory in each device
    """

    _instances = {}

    def __new__(cls, device):
        if device not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[device] = instance
        return cls._instances[device]

    def __init__(self, device):
        if self._initialized:
            return
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.workspace = torch.zeros(
            500, dtype=torch.int, device=device, requires_grad=False
        )
        self.global_reduce_buffer = torch.zeros(
            sms * 6 * 128 * 512, dtype=torch.int, device=device, requires_grad=False
        )
        self._initialized = True

    def get_buffers(self):
        return self.workspace, self.global_reduce_buffer


def repack_and_shuffle_w4a8(weight_data, E):
    """
    逐 expert 处理 [n, k_half]
    处理完直接写回 weight_data[i]
    """
    # 原始 shape: [E, n, k_half]
    for i in range(E):
        # 1. 取当前 expert [n, k_half]
        expert = weight_data[i]
        n, k_half = expert.shape

        # 2. repack 逻辑（连续 → blocked）
        w_u8 = expert.to(torch.uint8)

        # 解包 1byte → 2个4bit
        w_unpacked = torch.stack([(w_u8 >> 4) & 0x0F, w_u8 & 0x0F], dim=-1).view(n, -1)

        # 8个4bit分块重排
        blocks = w_unpacked.view(n, -1, 8)
        w_low = blocks[..., :4]
        w_high = blocks[..., 4:]
        packed = (w_low << 4) | w_high
        packed = packed.view(n, k_half)

        # 3. shuffle
        w_marlin_in = w4a8_moe_layout_shuffle_gemm2(packed)
        w_marlin_in = w_marlin_in.reshape(n, k_half)
        # 4. 直接写回
        weight_data[i] = w_marlin_in

    return weight_data


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


class SlimQuantW4A8Int8MarlinConfig(QuantizationConfig):
    """Config class for W4A8 Int8 Quantization.
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
        return "slimquant_w4a8_marlin"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, any]) -> "SlimQuantW4A8Int8MarlinConfig":
        return cls()

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        if hf_quant_cfg.get("quant_method") == "slimquant_w4a8" and user_quant in (
            "slimquant_w4a8_marlin",
            "slimquant_marlin",
        ):
            return cls.get_name()
        return None

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
            if _resolved_backend == W4A8_TPMOE_BACKEND_AITER:
                return SlimQuantW4A8Int8AiterMoEMethod(self)
            return SlimQuantW4A8Int8MarlinMoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class HCUW4A8Int8DeepEPMoEMethod(FusedMoEMethodBase):
    # DeepEPMoE selects this method independently of the global CT config,
    # which still resolves mixed W8A8 Linear layers.
    hcu_w4a8_hipc = True

    def __init__(self, quant_config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        # Retain checkpoint names and CT loader partitioning until post-load.
        # No actorder indices or zero points are needed for this qualified path.
        extra_weight_attrs.update(is_transposed=True, quant_method="channel")
        shapes = {
            "w13_weight_packed": (
                num_experts,
                hidden_size // 8,
                2 * intermediate_size_per_partition,
            ),
            "w2_weight_packed": (
                num_experts,
                intermediate_size_per_partition // 8,
                hidden_size,
            ),
            "w13_weight_scale": (num_experts, 1, 2 * intermediate_size_per_partition),
            "w2_weight_scale": (num_experts, 1, hidden_size),
            "w13_weight_shape": (num_experts, 2),
            "w2_weight_shape": (num_experts, 2),
        }
        for name, shape in shapes.items():
            dtype = params_dtype if name.endswith("scale") else torch.int32
            param = torch.nn.Parameter(
                torch.empty(shape, dtype=dtype), requires_grad=False
            )
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra_weight_attrs)

    def create_moe_runner(self, layer, moe_runner_config):
        if moe_runner_config.activation != "silu" or not moe_runner_config.is_gated:
            raise ValueError("HCU W4A8 HIPC supports gated SiLU only")
        if moe_runner_config.apply_router_weight_on_input:
            raise ValueError("HCU W4A8 HIPC applies router weights on output only")
        self.moe_runner_config = moe_runner_config
        self.runner = None  # DeepEPMoE owns contiguous/masked compute.

    @staticmethod
    def _convert_packed_weight(weight):
        weight = weight.transpose(1, 2).contiguous().view(torch.uint8)
        # CT stores offset-8, low nibble first; HIPC reads signed, high first.
        high = weight >> 4
        weight <<= 4
        weight |= high
        weight ^= 0x88
        return weight.view(torch.int8)

    @staticmethod
    def _channel_scale(scale):
        scale = scale.transpose(1, 2).squeeze(-1).contiguous().float().unsqueeze(-1)
        # CT has true scales; the reference SlimQuant checkpoint has scale/16.
        scale.div_(16)
        # Preserve the tested HCU vector-read mapping guard.
        storage = torch.empty(
            scale.numel() + (2 * 1024 * 1024) // 4,
            dtype=torch.float32,
            device=scale.device,
        )
        storage[: scale.numel()].copy_(scale.reshape(-1))
        return storage[: scale.numel()].view_as(scale)

    def process_weights_after_loading(self, layer):
        import deepgemm
        from sglang.srt.layers.moe.ep_moe.layer import (
            _use_w4a8_contiguous_hipc,
            _use_w4a8_masked_hipc,
        )

        mode = get_deepep_mode()
        normal_layout = "hipc" if _use_w4a8_contiguous_hipc else "marlin"
        ll_layout = "hipc" if _use_w4a8_masked_hipc else "marlin_masked"
        layout = normal_layout if mode.enable_normal() else ll_layout
        layouts = {layout}
        if mode.enable_low_latency():
            layouts.add(ll_layout)
        if mode.enable_normal() and normal_layout == "marlin":
            if self.moe_runner_config.swiglu_limit is not None:
                raise RuntimeError(
                    "LightOp Marlin apply_ep does not support V4.1 swiglu_limit; "
                    "set SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1"
                )
        for name in (() if "hipc" not in layouts else (
            "pack_w4a8_moe_hipc_weight",
            "m_grouped_w4a8_gemm_nt_contiguous_hipc",
            "m_grouped_w4a8_gemm_nt_masked_hipc",
        )):
            if not hasattr(deepgemm, name):
                raise RuntimeError(f"HCU W4A8 DeepEP requires deepgemm.{name}")
        w13 = self._convert_packed_weight(layer.w13_weight_packed.data)
        w2 = self._convert_packed_weight(layer.w2_weight_packed.data)
        intermediate_size = w2.shape[-1] * 2
        padded_size = ((intermediate_size + 127) // 128) * 128
        layer.w4a8_intermediate_size = intermediate_size
        layer.w4a8_padded_intermediate_size = (
            padded_size if layout == "hipc" else intermediate_size
        )
        layer.w4a8_ll_padded_intermediate_size = (
            padded_size if ll_layout == "hipc" else intermediate_size
        )
        for prefix, weight in (("w13", w13), ("w2", w2)):
            packed_layouts = {}
            for target_layout in layouts:
                # HIPC packing mutates its input. Clone only when auto mode
                # genuinely requires different normal and low-latency layouts.
                current = weight.clone() if len(layouts) > 1 else weight
                if target_layout == "hipc":
                    if prefix == "w2" and padded_size != intermediate_size:
                        current = F.pad(current, (0, (padded_size - intermediate_size) // 2))
                    packed = deepgemm.pack_w4a8_moe_hipc_weight(current)
                else:
                    packed = w4a8_weight_repack_impl(
                        current, use_deepep=target_layout == "marlin_masked"
                    )
                packed_layouts[target_layout] = packed
            layer.register_parameter(
                prefix + "_weight",
                torch.nn.Parameter(packed_layouts[layout], requires_grad=False),
            )
            if mode.is_auto() and ll_layout != layout:
                layer.register_parameter(
                    prefix + "_weight_low_latency",
                    torch.nn.Parameter(packed_layouts[ll_layout], requires_grad=False),
                )
            delattr(layer, prefix + "_weight_packed")
            scale_name = prefix + "_weight_scale"
            scale = self._channel_scale(getattr(layer, scale_name).data)
            setattr(layer, scale_name, torch.nn.Parameter(scale, requires_grad=False))
        layer.dispatcher.set_quant_config(
            {
                "normal_dispatcher_output_dtype": "int8",
                "low_latency_dispatcher_output_dtype": "int8",
                "normal_expert_alignment": 256,
                "hcu_w4a8_int8_dispatch": True,
                "hcu_w4a8_deepep_ll_producer_event": True,
            }
        )
        logger.info_once(
            "HCU W4A8 reference DeepEP enabled: normal=INT8, low_latency=INT8; "
            f"contiguous_hipc={_use_w4a8_contiguous_hipc}, masked_hipc={_use_w4a8_masked_hipc}"
        )

    def apply(self, layer, dispatch_output):
        return layer.run_moe_core(dispatch_output)

    def apply_ep(self, **kwargs):
        # Keep the reference Marlin fallback without recreating the CT scheme.
        if self.moe_runner_config.swiglu_limit is not None:
            raise RuntimeError(
                "LightOp Marlin apply_ep does not support V4.1 swiglu_limit; "
                "set SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1"
            )
        return SlimQuantW4A8Int8MarlinMoEMethod.apply_ep(self, **kwargs)


class SlimQuantW4A8Int8MarlinMoEMethod:
    """MoE method for W4A8INT8 Marlin.
    Supports loading INT8 checkpoints with static weight scale and
    dynamic/static activation scale.
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
        layer.w13_weight = Parameter(
            w4a8_weight_repack_impl(layer.w13_weight, use_deepep=self.use_deepep),
            requires_grad=False,
        )
        layer.w2_weight = Parameter(
            w4a8_weight_repack_impl(layer.w2_weight, use_deepep=self.use_deepep),
            requires_grad=False,
        )
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

    @torch._dynamo.disable()  # TODO: 性能优化需lmslim/lightop配合
    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output,
        i_q: Optional[torch.Tensor] = None,
        i_s: Optional[torch.Tensor] = None,
        # local_expert_mapping,
    ):
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

        topk_weights, topk_ids, _ = topk_output
        x, topk_weights = apply_topk_weights_cpu(
            self.moe_runner_config.apply_router_weight_on_input, topk_weights, x
        )
        workspace, global_reduce_buffer = MarlinMoeWorkspace(x.device).get_buffers()
        routed_scaling_factor = (
            self.moe_runner_config.routed_scaling_factor
            if self.moe_runner_config.routed_scaling_factor is not None
            else 1.0
        )
        output = fused_experts_impl_w4a8_marlin(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            workspace=workspace,
            global_reduce_buffer=global_reduce_buffer,
            inplace=True,
            use_int4_w4a8=True,
            per_channel_quant=True,
            activation=layer.moe_runner_config.activation,
            expert_map=getattr(layer, "expert_map", None),
            apply_router_weight_on_input=self.moe_runner_config.apply_router_weight_on_input,
            global_num_experts=layer.moe_runner_config.num_experts,
            w1_scale=(layer.w13_weight_scale),
            w2_scale=(layer.w2_weight_scale),
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            use_nn_moe=False,
            routed_scaling_factor=routed_scaling_factor,
        )
        return StandardCombineInput(hidden_states=output)

    @torch._dynamo.disable()  # TODO: 性能优化需lmslim/lightop配合
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
        from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        x, topk_weights = apply_topk_weights_cpu(
            self.moe_runner_config.apply_router_weight_on_input, topk_weights, x
        )
        workspace, global_reduce_buffer = MarlinMoeWorkspace(x.device).get_buffers()
        routed_scaling_factor = (
            self.moe_runner_config.routed_scaling_factor
            if self.moe_runner_config.routed_scaling_factor is not None
            else 1.0
        )
        return fused_experts_impl_w4a8_marlin(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            workspace=workspace,
            global_reduce_buffer=global_reduce_buffer,
            inplace=True,
            use_int4_w4a8=True,
            per_channel_quant=True,
            activation=activation,
            expert_map=getattr(layer, "expert_map", None),
            apply_router_weight_on_input=self.moe_runner_config.apply_router_weight_on_input,
            global_num_experts=layer.moe_runner_config.num_experts,
            w1_scale=(layer.w13_weight_scale),
            w2_scale=(layer.w2_weight_scale),
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            use_nn_moe=False,
            routed_scaling_factor=routed_scaling_factor,
            shared_output=shared_output,
            i_q=i_q,
            i_s=i_s,
        )

    # def _apply(
    #     self,
    #     layer: torch.nn.Module,
    #     x: torch.Tensor,
    #     router_logits: torch.Tensor,
    #     top_k: int,
    #     #renormalize: bool,
    #     #use_grouped_topk: bool = False,
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
    #     **_
    # ) -> torch.Tensor:
    #     from sglang.srt.layers.moe.fused_moe_triton import (FusedMoE, FusedMoeWeightScaleSupported)
    #     from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
    #     if enable_eplb:
    #         raise NotImplementedError(
    #             "EPLB not supported for `SlimQuantW4A8Int8MarlinMoEMethod` yet.")
    #     # Expert selection
    #     topk_weights, topk_ids = FusedMoE.select_experts(
    #         hidden_states=x,
    #         router_logits=router_logits,
    #         #use_grouped_topk=use_grouped_topk,
    #         top_k=top_k,
    #         #renormalize=renormalize,
    #         topk_group=topk_group,
    #         num_expert_group=num_expert_group,
    #         custom_routing_function=custom_routing_function,
    #         scoring_func=scoring_func,
    #         e_score_correction_bias=e_score_correction_bias,
    #         routed_scaling_factor=routed_scaling_factor,
    #         use_fused_gate=use_fused_gate
    #     )
    #     workspace, global_reduce_buffer = MarlinMoeWorkspace(x.device).get_buffers()
    #     return fused_experts_impl_w4a8_marlin(
    #         x,
    #         layer.w13_weight,
    #         layer.w2_weight,
    #         topk_weights=topk_weights,
    #         topk_ids=topk_ids,
    #         workspace=workspace,
    #         global_reduce_buffer=global_reduce_buffer,
    #         inplace=True,
    #         use_int4_w4a8=True,
    #         per_channel_quant=True,
    #         activation=activation,
    #         expert_map=expert_map,
    #         apply_router_weight_on_input=apply_router_weight_on_input,
    #         global_num_experts=global_num_experts,
    #         w1_scale=(layer.w13_weight_scale),
    #         w2_scale=(layer.w2_weight_scale),
    #         a1_scale=layer.w13_input_scale,
    #         a2_scale=layer.w2_input_scale,
    #         use_nn_moe=use_nn_moe,
    #     )

    def apply_ep(
        self,
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        use_nn_moe: Optional[bool] = False,
        num_local_tokens: Optional[torch.Tensor] = None,
        # config_select_bs: Optional[int] = None,
        routed_scaling_factor: Optional[float] = 1.0,
        shared_output: Optional[torch.Tensor] = None,
        # scales: Optional[torch.Tensor] = None,
        num_recv_tokens_per_expert: List = None,
        **_,
    ):
        workspace, global_reduce_buffer = MarlinMoeWorkspace(x.device).get_buffers()
        routed_scaling_factor = (
            1.0 if routed_scaling_factor is None else routed_scaling_factor
        )
        return fused_experts_impl_w4a8_marlin(
            x,
            w1,
            w2,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            workspace=workspace,
            global_reduce_buffer=global_reduce_buffer,
            inplace=True,
            use_int4_w4a8=True,
            per_channel_quant=True,
            activation=activation,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            use_nn_moe=use_nn_moe,
            shared_output=shared_output,
            routed_scaling_factor=float(routed_scaling_factor),
            # num_local_tokens=num_local_tokens,
            # config_select_bs=config_select_bs,
            # q_scales=scales
        )


class SlimQuantW4A8Int8AiterMoEMethod:
    """MoE method for W4A8INT8 AITER."""

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
        layer.w13_weight = Parameter(
            repack_and_shuffle_w4a8(layer.w13_weight.data, E), requires_grad=False
        )
        layer.w2_weight = Parameter(
            repack_and_shuffle_w4a8(layer.w2_weight.data, E), requires_grad=False
        )

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

    @torch._dynamo.disable()
    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output,
    ):
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_weights, topk_ids, _ = dispatch_output.topk_output
        # x, topk_weights = apply_topk_weights_cpu(
        #     self.moe_runner_config.apply_router_weight_on_input, topk_weights, x
        # )
        if x.shape[0] == 0:
            return StandardCombineInput(hidden_states=x)

        e = layer.w13_weight.size(0)
        k = x.size(-1)
        n1 = layer.w13_weight.size(1)
        n2 = n1 // 2
        topk = topk_ids.size(1)

        if x.dim() == 2:
            m = x.size(0)
        else:
            assert x.dim() == 3
            assert x.size(0) == e, f"{x.size(0)} == {e}"
            m = x.size(1)

        status, moe_config = get_aiter_moe_config(
            M=m,
            E=e,
            N1=n1,
            N2=n2,
            K=k,
            top_k=topk,
            block_size=None,
            dtype=x.dtype,
            quant_type=MoeQuantType.W4A8,
        )
        if not status:
            raise RuntimeError(
                "aiter backend did not find a valid w4a8 tpmoe config for "
                f"M={m}, E={e}, N1={n1}, N2={n2}, K={k}, topk={topk}, "
                f"dtype={x.dtype}."
            )

        output = aiter_moe(
            x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            moe_config=moe_config,
            activation="silu",
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            global_num_experts=e,
            expert_map=None,
            routed_scaling_factor=self.moe_runner_config.routed_scaling_factor,
        )
        return StandardCombineInput(hidden_states=output)
