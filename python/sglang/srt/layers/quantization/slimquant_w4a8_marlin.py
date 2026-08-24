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
from sglang.srt.environ import envs
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe import (
    MoeRunner,
    MoeRunnerBackend,
    MoeRunnerConfig,
    get_moe_a2a_backend,
)
from sglang.srt.layers.moe.utils import (
    W4A8_TPMOE_BACKEND_AITER,
    W4A8_TPMOE_BACKEND_AUTO,
    W4A8_TPMOE_BACKEND_LIGHTOP,
    W4A8_TPMOE_BACKEND_TRITON,
    get_dspark_w4a8_tpmoe_backend_override,
    normalize_w4a8_tpmoe_backend,
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

_lmslim_w4a8_marlin_available = False
_lmslim_w4a8_triton_available = False
_aiter_w4a8_marlin_available = False
fused_experts_impl_w4a8_marlin = None
w4a8_triton = None
moe_align_block_size_lightop = None
get_moe_cache = None
per_token_quant_int8 = None
MoeQuantType = None
aiter_moe = None
get_aiter_moe_config = None
w4a8_moe_layout_shuffle_gemm2 = None


def _ensure_lightop_w4a8_marlin_available() -> None:
    global _lmslim_w4a8_marlin_available
    global fused_experts_impl_w4a8_marlin

    if _lmslim_w4a8_marlin_available:
        return
    try:
        from lightop.moe import (
            fused_experts_impl_w4a8_marlin as _fused_experts_impl_w4a8_marlin,
        )
        from lightop._lmslim_native.layers.fused_moe import (
            w4a8_marlin as _lightop_w4a8_marlin,
        )
    except Exception as e:
        raise RuntimeError(
            "lightop backend is selected for w4a8 tpmoe, but lightop is not available."
        ) from e

    _lightop_w4a8_marlin.moe_align_block_size_lightop = (
        _safe_lightop_ep_moe_align_block_size
    )
    fused_experts_impl_w4a8_marlin = _fused_experts_impl_w4a8_marlin
    _lmslim_w4a8_marlin_available = True


def _ensure_lightop_w4a8_triton_available() -> None:
    global _lmslim_w4a8_triton_available
    global w4a8_triton, moe_align_block_size_lightop, get_moe_cache, per_token_quant_int8

    if _lmslim_w4a8_triton_available:
        return
    try:
        from lightop._lmslim_native.layers.fused_moe import w4a8 as _w4a8_triton
        from lightop._lmslim_native.layers.fused_moe.w4a8_marlin import (
            moe_align_block_size_lightop as _moe_align_block_size_lightop,
        )
        from lightop._lmslim_native.vllm_compat.fused_moe_cache import (
            get_moe_cache as _get_moe_cache,
        )
        from lightop.quant import per_token_quant_int8 as _per_token_quant_int8
    except Exception as e:
        raise RuntimeError(
            "triton backend is selected for w4a8 tpmoe, but lightop triton "
            "kernels are not available."
        ) from e

    w4a8_triton = _w4a8_triton
    moe_align_block_size_lightop = _moe_align_block_size_lightop
    get_moe_cache = _get_moe_cache
    per_token_quant_int8 = _per_token_quant_int8
    _lmslim_w4a8_triton_available = True


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
            "aiter backend is selected for w4a8 tpmoe, but aiter is not available."
        ) from e

    MoeQuantType = _MoeQuantType
    aiter_moe = _aiter_moe
    get_aiter_moe_config = _get_aiter_moe_config
    w4a8_moe_layout_shuffle_gemm2 = _w4a8_moe_layout_shuffle_gemm2
    _aiter_w4a8_marlin_available = True


def _try_ensure_backend_available(ensure_backend) -> bool:
    try:
        ensure_backend()
        return True
    except RuntimeError as e:
        logger.info(str(e))
        return False


def _resolve_w4a8_tpmoe_backend(
    requested_backend: str, *, env_name: str = W4A8_TPMOE_BACKEND_ENV
) -> str:
    backend = normalize_w4a8_tpmoe_backend(requested_backend, env_name=env_name)
    if backend == W4A8_TPMOE_BACKEND_AUTO:
        if _try_ensure_backend_available(_ensure_lightop_w4a8_marlin_available):
            return W4A8_TPMOE_BACKEND_LIGHTOP
        if _try_ensure_backend_available(_ensure_aiter_w4a8_marlin_available):
            return W4A8_TPMOE_BACKEND_AITER
        raise RuntimeError(
            "Neither lightop nor aiter backend is available for w4a8 tpmoe."
        )
    if backend == W4A8_TPMOE_BACKEND_LIGHTOP:
        _ensure_lightop_w4a8_marlin_available()
    elif backend == W4A8_TPMOE_BACKEND_TRITON:
        _ensure_lightop_w4a8_triton_available()
    else:
        _ensure_aiter_w4a8_marlin_available()
    return backend


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


def _safe_lightop_ep_moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    expert_mask: Optional[torch.Tensor] = None,
    num_local_tokens: Optional[torch.Tensor] = None,
    pad_sorted_ids: bool = False,
    ep_size: int = 8,
    num_token: Optional[int] = None,
):
    if expert_map is None:
        raise ValueError("The safe LightOp EP align requires expert_map.")
    local_num_experts = num_experts // ep_size
    valid = (topk_ids >= 0) & (topk_ids < expert_map.numel())
    safe_ids = topk_ids.clamp(0, expert_map.numel() - 1).to(torch.long)
    local_ids = expert_map[safe_ids]
    local_ids = torch.where(
        valid & (local_ids >= 0) & (local_ids < local_num_experts),
        local_ids,
        torch.full_like(local_ids, -1),
    )
    if expert_mask is not None:
        local_ids = local_ids.masked_fill(expert_mask, -1)
    if num_local_tokens is not None:
        rows = torch.arange(topk_ids.shape[0], device=topk_ids.device)
        local_ids = local_ids.masked_fill(rows[:, None] >= num_local_tokens, -1)

    from sglang.srt.layers.moe.fused_moe_triton import (
        moe_align_block_size as sglang_moe_align_block_size,
    )

    return sglang_moe_align_block_size(local_ids, block_size, local_num_experts)


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
    num_local_tokens: Optional[torch.Tensor] = None,
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
    if global_num_experts == -1:
        global_num_experts = w1.shape[0]
    expert_mask = None
    if expert_map is not None:
        # Mirrors the lightop marlin EP path: the align op expects an
        # (num_tokens, top_k) validity mask; all received slots are valid.
        expert_mask = torch.zeros(
            (num_tokens, top_k), dtype=torch.bool, device=hidden_states.device
        )
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
        if expert_map is None:
            # StandardDispatcher apply path: topk_ids are LOCAL (0..E-1) with a
            # -1 sentinel for experts owned by other EP ranks. lightop's
            # vllm_compat.moe_align_block_size treats -1 as a large expert id and
            # faults with num_experts=256 or misbuckets the -1 entries with
            # num_experts=E. Route through sglang's align, which maps -1 to a
            # sentinel expert on HCU and maps the sentinel blocks back to -1 for
            # the GEMM kernel's filter-expert path.
            from sglang.srt.layers.moe.fused_moe_triton import (
                moe_align_block_size as sglang_moe_align_block_size,
            )

            sorted_ids, expert_ids, padded_count = sglang_moe_align_block_size(
                current_ids, config1["BLOCK_SIZE_M"], w1.shape[0]
            )
        else:
            # DeepEPMoE apply_ep path: topk_ids are GLOBAL and expert_map maps
            # global experts to local ids (-1 for experts on other ranks).
            # lightop's vllm_compat align drops expert_map in the lightop op
            # path, so use the lightop EP align from the marlin backend, which
            # passes the map to the op and aligns only this rank's tokens.
            sorted_ids, expert_ids, padded_count = moe_align_block_size_lightop(
                current_ids,
                config1["BLOCK_SIZE_M"],
                global_num_experts,
                expert_map=expert_map,
                expert_mask=expert_mask[begin:end],
                num_local_tokens=num_local_tokens,
                ep_size=global_num_experts // w1.shape[0],
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
        if expert_map is not None:
            # The EP align only schedules this rank's slots, so the second
            # GEMM never writes the rows of slots routed to other ranks.
            # cache1 is dead by now and cache3 overlaps its prefix, so it is
            # safe to zero cache3 here; the reduce below then sums only the
            # local contributions, matching the marlin backend's masked
            # op.moe_sum reduce.
            cache3.zero_()
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
            dspark_backend_override = get_dspark_w4a8_tpmoe_backend_override()
            if dspark_backend_override is None:
                requested_backend = envs.SGLANG_W4A8_TPMOE_BACKEND.get()
                env_name = W4A8_TPMOE_BACKEND_ENV
            else:
                requested_backend = dspark_backend_override
                env_name = "SGLANG_DSPARK_FORCE_W4A8_TPMOE_BACKEND"
            resolved_backend = _resolve_w4a8_tpmoe_backend(
                requested_backend,
                env_name=env_name,
            )
            if resolved_backend == W4A8_TPMOE_BACKEND_AITER:
                selected_method = SlimQuantW4A8Int8AiterMoEMethod
            elif resolved_backend == W4A8_TPMOE_BACKEND_TRITON:
                selected_method = SlimQuantW4A8Int8TritonMoEMethod
            else:
                selected_method = SlimQuantW4A8Int8MarlinMoEMethod
            quant_method = selected_method(self)
            logger.info(
                "[slimquant_w4a8_marlin] selected_moe_method=%s "
                "requested_backend=%s resolved_backend=%s dspark_backend_override=%s",
                selected_method.__name__,
                requested_backend,
                resolved_backend,
                dspark_backend_override,
            )
            return quant_method
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


class SlimQuantW4A8Int8TritonMoEMethod:
    """MoE method for W4A8INT8 Triton.

    Supports loading INT8 checkpoints with static weight scale and
    dynamic/static activation scale. Weights are used without Marlin repack;
    the GEMMs run through the lightop triton w4a8 fused-MoE kernels.
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
        output = self._apply_triton(
            layer,
            x,
            topk_weights,
            topk_ids,
            layer.moe_runner_config.activation,
            shared_output=None,
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
        routed_scaling_factor = (
            1.0 if routed_scaling_factor is None else routed_scaling_factor
        )
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
            num_local_tokens=num_local_tokens,
            shared_output=None,
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
