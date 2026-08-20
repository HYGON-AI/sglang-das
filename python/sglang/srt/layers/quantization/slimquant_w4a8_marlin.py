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
import triton.language as tl
from torch.nn.parameter import Parameter

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe import (
    MoeRunner,
    MoeRunnerBackend,
    MoeRunnerConfig,
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
W4A8_TPMOE_BACKEND_TRITON = "triton"
_requested_backend = (
    os.getenv(W4A8_TPMOE_BACKEND_ENV, W4A8_TPMOE_BACKEND_AUTO).strip().lower()
)


_lmslim_w4a8_marlin_available = False
_lmslim_w4a8_triton_available = False
_aiter_w4a8_marlin_available = False
MoeQuantType = None
aiter_moe = None
get_aiter_moe_config = None
w4a8_moe_layout_shuffle_gemm2 = None


def _ensure_aiter_w4a8_marlin_available() -> None:
    global _aiter_w4a8_marlin_available
    global MoeQuantType, aiter_moe, get_aiter_moe_config
    global w4a8_moe_layout_shuffle_gemm2

    if _aiter_w4a8_marlin_available:
        return
    try:
        from aiter.moe import MoeQuantType as _MoeQuantType
        from aiter.moe import aiter_moe as _aiter_moe
        from aiter.moe import get_aiter_moe_config as _get_aiter_moe_config
        from aiter.ops.shuffle import (
            w4a8_moe_layout_shuffle_gemm2 as _w4a8_moe_layout_shuffle_gemm2,
        )
    except Exception as e:
        raise RuntimeError(
            "SGLANG_DSPARK_FORCE_W4A8_TPMOE_AITER=1 requires the AITER W4A8 "
            "MoE backend, but AITER could not be imported."
        ) from e

    MoeQuantType = _MoeQuantType
    aiter_moe = _aiter_moe
    get_aiter_moe_config = _get_aiter_moe_config
    w4a8_moe_layout_shuffle_gemm2 = _w4a8_moe_layout_shuffle_gemm2
    _aiter_w4a8_marlin_available = True

if _requested_backend in {
    W4A8_TPMOE_BACKEND_AUTO,
    W4A8_TPMOE_BACKEND_LIGHTOP,
    W4A8_TPMOE_BACKEND_TRITON,
}:
    try:
        from lightop.moe import (
            fused_experts_impl_w4a8_marlin,
        )

        _lmslim_w4a8_marlin_available = True
    except Exception:
        logger.info(
            "INFO: Please install lightop if you want to infer the quantitative model of moe.\n"
        )

    try:
        from lightop._lmslim_native.layers.fused_moe import w4a8 as w4a8_triton
        from lightop._lmslim_native.vllm_compat.fused_moe_cache import get_moe_cache
        from lightop.quant import per_token_quant_int8

        _lmslim_w4a8_triton_available = True
    except Exception:
        logger.info(
            "INFO: Please install lightop triton kernels if you want to use w4a8 triton tpmoe.\n"
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
    W4A8_TPMOE_BACKEND_TRITON,
}:
    raise ValueError(
        f"Unsupported {W4A8_TPMOE_BACKEND_ENV}={_requested_backend!r}. "
        f"Supported values: {W4A8_TPMOE_BACKEND_AUTO!r}, "
        f"{W4A8_TPMOE_BACKEND_LIGHTOP!r}, {W4A8_TPMOE_BACKEND_AITER!r}, "
        f"{W4A8_TPMOE_BACKEND_TRITON!r}."
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
elif _requested_backend == W4A8_TPMOE_BACKEND_TRITON:
    if not _lmslim_w4a8_triton_available:
        raise RuntimeError(
            "triton backend is selected for w4a8 tpmoe, but lightop triton kernels are not available."
        )
    _resolved_backend = W4A8_TPMOE_BACKEND_TRITON
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


def _get_w4a8_triton_chunk_size(
    cache13: torch.Tensor,
    *,
    top_k: int,
    n1: int,
    n2: int,
    num_tokens: int,
) -> int:
    requested_chunk_size = int(os.getenv("LMSLIM_FUSED_MOE_CHUNK_SIZE", "32768"))
    if requested_chunk_size <= 0:
        raise ValueError(
            "LMSLIM_FUSED_MOE_CHUNK_SIZE must be positive, "
            f"got {requested_chunk_size}."
        )
    cache_token_capacity = cache13.numel() // (top_k * max(n1, n2))
    if cache_token_capacity <= 0:
        raise RuntimeError(
            "W4A8 Triton MoE cache is too small: "
            f"cache_numel={cache13.numel()}, top_k={top_k}, n1={n1}, n2={n2}."
        )
    return min(requested_chunk_size, num_tokens, cache_token_capacity)


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
    shared_output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run SlimQuant W4A8 Triton MoE GEMMs without Marlin repack."""
    assert hidden_states.ndim == 2 and hidden_states.is_contiguous()
    assert hidden_states.shape[1] == w1.shape[2] * 2
    assert topk_weights.shape == topk_ids.shape

    num_tokens = hidden_states.shape[0]
    if num_tokens == 0:
        return torch.empty_like(hidden_states)

    top_k = topk_ids.shape[1]
    n1 = w1.shape[1]
    n2 = w2.shape[1]
    chunk_size = _get_w4a8_triton_chunk_size(
        cache13,
        top_k=top_k,
        n1=n1,
        n2=n2,
        num_tokens=num_tokens,
    )
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
            from sglang.srt.layers.moe.utils import (
                should_force_dspark_w4a8_tpmoe_aiter,
            )

            force_aiter = should_force_dspark_w4a8_tpmoe_aiter()
            if force_aiter:
                _ensure_aiter_w4a8_marlin_available()
            use_aiter = force_aiter or (
                _resolved_backend == W4A8_TPMOE_BACKEND_AITER
            )
            selected_method = (
                SlimQuantW4A8Int8AiterMoEMethod
                if use_aiter
                else SlimQuantW4A8Int8MarlinMoEMethod
            )
            logger.info(
                "[slimquant_w4a8_marlin] selected_moe_method=%s "
                "force_dspark_aiter=%s resolved_backend=%s",
                selected_method.__name__,
                force_aiter,
                _resolved_backend,
            )
            return selected_method(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


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
        self.use_triton = _resolved_backend == W4A8_TPMOE_BACKEND_TRITON

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
        if self.use_triton:
            layer.w13_weight = Parameter(layer.w13_weight, requires_grad=False)
            layer.w2_weight = Parameter(layer.w2_weight, requires_grad=False)
        else:
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


    def _apply_triton(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        shared_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        return fused_experts_impl_w4a8_triton(
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
        if self.use_triton:
            output = self._apply_triton(
                layer,
                x,
                topk_weights,
                topk_ids,
                layer.moe_runner_config.activation,
                shared_output=None,
            )
            return StandardCombineInput(hidden_states=output)

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
        if self.use_triton:
            if i_q is not None or i_s is not None:
                raise NotImplementedError(
                    "pre-quantized activation input is not supported by the Triton W4A8 MoE path yet."
                )
            return self._apply_triton(
                layer,
                x,
                topk_weights,
                topk_ids,
                activation,
                shared_output=shared_output,
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
        if self.use_triton:
            if shared_output is not None:
                raise NotImplementedError(
                    "shared_output is not supported by apply_ep Triton W4A8 MoE path yet."
                )
            cache13 = get_moe_cache(
                topk_ids.shape[1],
                w1.shape[1],
                w2.shape[1],
                device=x.device,
                dtype=x.dtype,
            )
            return fused_experts_impl_w4a8_triton(
                x,
                w1,
                w2,
                topk_weights,
                topk_ids,
                cache13,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                routed_scaling_factor=float(routed_scaling_factor),
                shared_output=None,
            )

        workspace, global_reduce_buffer = MarlinMoeWorkspace(x.device).get_buffers()
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
