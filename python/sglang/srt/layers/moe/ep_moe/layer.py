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
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch
import torch.nn.functional as F

from sglang.kernels.ops.moe.ep_moe_kernels import (
    build_m_indices_triton,
    ep_gather,
    ep_scatter,
    ep_scatter_no_scale,
    tma_align_input_scale,
)
from sglang.kernels.ops.moe.rocm_moe_utils import upscale, upscale_mxfp4
from sglang.kernels.ops.quantization.fp8_kernel import (
    is_fp8_fnuz,
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.batch_overlap.single_batch_overlap import DownGemmOverlapArgs
from sglang.srt.distributed import (
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
)
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.utils import FusedMoEMode, npu_format_cast
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.dp_attention import (
    get_is_extend_in_batch,
    set_is_extend_in_batch,
)
from sglang.srt.layers.moe import (  # should_use_flashinfer_trtllm_moe, # 找不到
    get_deepep_mode,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import (
    FusedMoE,
    moe_forward_piecewise_cuda_graph_impl,
)
from sglang.srt.layers.moe.moe_runner.deep_gemm import copy_list_to_gpu_no_ce
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPNormalCombineInput,
)
from sglang.srt.layers.moe.token_dispatcher.moriep import (
    MoriEPLLCombineInput,
    MoriEPNormalCombineInput,
)
from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
    TopKOutput,
    TopKOutputChecker,
)
from sglang.srt.layers.moe.utils import _get_deepgemm_shuffle_unique
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsFusedMoEMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors_marlin import (
    SlimQuantCompressedTensorsMarlinConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    NPUCompressedTensorsW4A16Int4DynamicMoE,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.layers.quantization.quark.schemes import QuarkW4A4MXFp4MoE
from sglang.srt.layers.quantization.slimquant_w4a8_marlin import (
    SlimQuantW4A8Int8MarlinConfig,
)
from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config, W4AFp8MoEMethod
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.utils import (
    ceil_div,
    direct_register_custom_op,
    dispose_tensor,
    get_bool_env_var,
    get_int_env_var,
    is_hcu,
    is_hip,
    is_npu,
)
from sglang.srt.utils.offloader import get_offloader

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        DeepEPLLDispatchOutput,
        DeepEPNormalDispatchOutput,
        DispatchOutput,
    )

from deepgemm import (
    m_grouped_bf16_gemm_nt_contiguous,
    m_grouped_bf16_gemm_nt_masked,
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_i8_gemm_nt_contiguous,
    m_grouped_i8_gemm_nt_masked,
    m_grouped_w4a8_gemm_nt_masked,
)
from deepgemm.m_group_gemm import grouped_gemm_w4a16_nt_masked_entry
from lightop import moe as lightop_op
from lightop.activation import (
    fuse_silu_and_mul,
    fuse_silu_mul_fp8_quant,
    fuse_silu_mul_fp8_quant_ep,
    fuse_silu_mul_quant,
    fuse_silu_mul_quant_ep,
)

_is_hip = is_hip()
_is_npu = is_npu()
_is_hcu = is_hcu()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_use_fp8_w8a8_moe = get_bool_env_var("SGLANG_USE_FP8_W8A8_MOE")
_use_marlin_w16a16_moe = get_bool_env_var("SGLANG_USE_MARLIN_W16A16_MOE")
_use_marlin_w4a16_moe = get_bool_env_var("SGLANG_USE_MARLIN_W4A16_MOE_OPT")
_use_lightop_ep_moe_align = get_bool_env_var("SGLANG_USE_LIGHTOP_EP_MOE_ALIGN", "true")
_use_lightop_ep_scatter = get_bool_env_var("SGLANG_USE_LIGHTOP_EP_SCATTER", "true")
_use_lightop_ep_gather = get_bool_env_var("SGLANG_USE_LIGHTOP_EP_GATHER", "true")

if _use_aiter and not _is_hcu:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
elif _is_npu:
    import torch_npu

logger = logging.getLogger(__name__)


def _can_use_lightop_ep_scatter(
    recv_x: torch.Tensor,
    recv_x_scale: Optional[torch.Tensor],
    recv_topk: torch.Tensor,
    output_tensor: torch.Tensor,
    output_tensor_scale: torch.Tensor,
    *,
    counts_are_aligned: bool,
    scale_ue8m0: bool = False,
) -> bool:
    # lightop's ep_scatter is a byte scatter: int8 and FP8 activations share
    # the same copy path as long as scales are per-token fp32.
    return (
        _use_lightop_ep_scatter
        and recv_x.element_size() == 1
        and output_tensor.element_size() == 1
        and recv_x_scale is not None
        and recv_x_scale.dtype == torch.float32
        and output_tensor_scale.dtype == torch.float32
        and (recv_x_scale.dim() == 1 or recv_x_scale.shape[-1] == 1)
        and (output_tensor_scale.dim() == 1 or output_tensor_scale.shape[-1] == 1)
        and recv_topk.dtype == torch.int64
        and counts_are_aligned
        and not scale_ue8m0
    )


def _can_use_lightop_ep_moe_align(
    topk_ids: torch.Tensor,
    device: torch.device,
    num_experts: int,
    total_elements: Optional[int],
    block_size: int,
    counts_are_aligned: bool,
) -> bool:
    return (
        _use_lightop_ep_moe_align
        and hasattr(lightop_op, "ep_build_m_indices")
        and counts_are_aligned
        and total_elements is not None
        and total_elements > 0
        and total_elements % block_size == 0
        and topk_ids.device == device
        and topk_ids.dtype == torch.int64
        and topk_ids.is_contiguous()
        and 0 < num_experts <= 1024
        and block_size > 0
    )


@torch.no_grad()
def _build_m_indices_with_optional_lightop(
    topk_ids: torch.Tensor,
    device: torch.device,
    num_experts: int,
    *,
    total_elements: Optional[int] = None,
    block_size: int = 256,
    counts_are_aligned: bool = False,
) -> torch.Tensor:
    if _can_use_lightop_ep_moe_align(
        topk_ids,
        device,
        num_experts,
        total_elements,
        block_size,
        counts_are_aligned,
    ):
        m_indices = torch.full((total_elements,), -1, device=device, dtype=torch.int32)
        lightop_op.ep_build_m_indices(topk_ids, m_indices, num_experts, block_size)
        return m_indices

    return build_m_indices_triton(topk_ids, device, num_experts)


def _can_use_lightop_ep_gather(
    input_tensor: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    input_index: torch.Tensor,
    output_tensor: torch.Tensor,
) -> bool:
    return (
        _use_lightop_ep_gather
        and hasattr(lightop_op, "ep_gather")
        and input_tensor.dim() == 2
        and output_tensor.dim() == 2
        and input_tensor.shape[1] == output_tensor.shape[1]
        and input_tensor.dtype == output_tensor.dtype
        and input_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and topk_ids.dtype == torch.int64
        and topk_weights.dtype == torch.float32
        and input_index.dtype == torch.int32
        and topk_ids.shape == topk_weights.shape
        and topk_ids.shape == input_index.shape
        and topk_ids.shape[1] <= 8
        and input_tensor.is_contiguous()
        and output_tensor.is_contiguous()
        and topk_ids.is_contiguous()
        and topk_weights.is_contiguous()
        and input_index.is_contiguous()
    )


@torch.no_grad()
def _ep_gather_with_optional_lightop(
    input_tensor: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    input_index: torch.Tensor,
    output_tensor: torch.Tensor,
) -> None:
    if _can_use_lightop_ep_gather(
        input_tensor, topk_ids, topk_weights, input_index, output_tensor
    ):
        lightop_op.ep_gather(
            input_tensor, topk_ids, topk_weights, input_index, None, output_tensor
        )
        return

    ep_gather(input_tensor, topk_ids, topk_weights, input_index, output_tensor)


@torch.no_grad()
def _ep_scatter_with_optional_lightop(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor,
    recv_topk: torch.Tensor,
    num_recv_tokens_per_expert: torch.Tensor,
    output_tensor: torch.Tensor,
    output_tensor_scale: torch.Tensor,
    all_tokens: int,
    *,
    counts_are_aligned: bool,
    scale_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_num_expert = num_recv_tokens_per_expert.shape[0]

    if _can_use_lightop_ep_scatter(
        recv_x,
        recv_x_scale,
        recv_topk,
        output_tensor,
        output_tensor_scale,
        counts_are_aligned=counts_are_aligned,
        scale_ue8m0=scale_ue8m0,
    ):
        output_index = torch.full(
            recv_topk.shape, -1, device=recv_topk.device, dtype=torch.int32
        )
        m_indices = torch.full(
            (all_tokens,), -1, device=recv_x.device, dtype=torch.int32
        )
        lightop_op.ep_scatter(
            recv_x,
            recv_x_scale,
            recv_topk,
            None,
            num_recv_tokens_per_expert,
            output_tensor,
            output_tensor_scale,
            m_indices,
            output_index,
            local_num_expert,
            256,
        )
        return m_indices, output_index

    output_index = torch.full(
        recv_topk.shape, -1, device=recv_topk.device, dtype=torch.int32
    )
    expert_start_loc = torch.zeros_like(num_recv_tokens_per_expert)
    m_indices = _build_m_indices_with_optional_lightop(
        recv_topk,
        recv_x.device,
        local_num_expert,
        total_elements=all_tokens if counts_are_aligned else None,
        block_size=256,
        counts_are_aligned=counts_are_aligned,
    )
    ep_scatter(
        recv_x,
        recv_x_scale,
        recv_topk,
        num_recv_tokens_per_expert,
        expert_start_loc,
        output_tensor,
        output_tensor_scale,
        m_indices,
        output_index,
        scale_ue8m0=scale_ue8m0,
    )
    return m_indices, output_index


# ------ custom op for lightop
def m_grouped_w4a8_gemm_nt_masked_wrapper(
    a0: torch.Tensor,
    a1: torch.Tensor,
    b0: torch.Tensor,
    b1: torch.Tensor,
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m_per_group: int,
) -> torch.Tensor:
    return m_grouped_w4a8_gemm_nt_masked(
        (a0, a1),
        (b0, b1),
        d,
        masked_m,
        expected_m_per_group,
    )


def m_grouped_w4a8_gemm_nt_masked_fake(
    a0: torch.Tensor,
    a1: torch.Tensor,
    b0: torch.Tensor,
    b1: torch.Tensor,
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m_per_group: int,
) -> torch.Tensor:
    return d


def m_grouped_i8_gemm_nt_masked_wrapper(
    a0: torch.Tensor,
    a1: torch.Tensor,
    b0: torch.Tensor,
    b1: torch.Tensor,
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m_per_group: int,
) -> torch.Tensor:
    shuffle_unique, _mode = _get_deepgemm_shuffle_unique()
    return m_grouped_i8_gemm_nt_masked(
        (a0, a1),
        (b0, b1),
        d,
        masked_m,
        expected_m_per_group,
        shuffle_unique=shuffle_unique,
    )


def m_grouped_i8_gemm_nt_masked_fake(
    a0: torch.Tensor,
    a1: torch.Tensor,
    b0: torch.Tensor,
    b1: torch.Tensor,
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m_per_group: int,
) -> torch.Tensor:
    return d


def fuse_silu_mul_quant_ep_wrapper(
    input: torch.Tensor,
    tokens_per_expert: Optional[torch.Tensor] = None,
    num_local_tokens_tensor: Optional[torch.Tensor] = None,
    topk: int = 1,
    expect_m: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    return fuse_silu_mul_quant_ep(
        input, tokens_per_expert, num_local_tokens_tensor, topk, expect_m
    )


def fuse_silu_mul_quant_ep_fake(
    input: torch.Tensor,
    tokens_per_expert: Optional[torch.Tensor] = None,
    num_local_tokens_tensor: Optional[torch.Tensor] = None,
    topk: int = 1,
    expect_m: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    E, T, H = input.shape
    d = H // 2
    output = torch.empty(E, T, d, dtype=torch.int8, device=input.device)
    scales = torch.empty((E, T, 1), device=input.device, dtype=torch.float32)
    return output, scales


direct_register_custom_op(
    op_name="m_grouped_w4a8_gemm_nt_masked",
    op_func=m_grouped_w4a8_gemm_nt_masked_wrapper,
    mutates_args=[],
    fake_impl=m_grouped_w4a8_gemm_nt_masked_fake,
)
direct_register_custom_op(
    op_name="m_grouped_i8_gemm_nt_masked",
    op_func=m_grouped_i8_gemm_nt_masked_wrapper,
    mutates_args=[],
    fake_impl=m_grouped_i8_gemm_nt_masked_fake,
)

direct_register_custom_op(
    op_name="fuse_silu_mul_quant_ep",
    op_func=fuse_silu_mul_quant_ep_wrapper,
    mutates_args=[],
    fake_impl=fuse_silu_mul_quant_ep_fake,
)


# TODO(kaixih@nvidia): ideally we should merge this logic into
# `fill_gateup_input_triton_kernel` to directly generate e8m0 scale.
@torch.compile
def _cast_to_e8m0_with_rounding_up(x: torch.Tensor) -> torch.Tensor:
    temp = x.to(torch.float32).view(torch.int32)
    exp = torch.bitwise_right_shift(temp, 23)
    mant = torch.bitwise_and(temp, 0x7FFFFF)
    is_ru = torch.logical_and(
        torch.logical_and((mant > 0), (exp != 0xFE)),
        ~torch.logical_and((exp == 0), (mant <= 0x400000)),
    )
    exp = torch.where(is_ru, exp + 1, exp)
    new_x = exp.to(torch.uint8).view(torch.int)
    return new_x.transpose(1, 2).contiguous().transpose(1, 2)


class DeepEPMoE(FusedMoE):
    """
    MoE Expert Parallel Impl based on DeepEP (https://github.com/deepseek-ai/DeepEP/tree/main)
    Mooncake EP shares the same class, as they expose the same interface.
    """

    _has_printed = False

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
            **kwargs,
        )
        is_humming = (
            get_moe_runner_backend().is_humming()
            or get_moe_runner_backend().is_auto()
            and quant_config is not None
            and quant_config.get_name() == "humming"
        )
        if is_humming:
            self.deprecate_flag = True
        elif _is_hcu and _use_aiter:
            self.deprecate_flag = False
        elif _use_aiter:
            self.deprecate_flag = True
        elif _is_npu:
            self.deprecate_flag = True
        elif deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and isinstance(
            quant_config, Fp8Config
        ):
            self.deprecate_flag = True
        elif (
            deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
        ):
            self.deprecate_flag = True
        elif (
            get_moe_runner_backend().is_flashinfer_cutedsl()
            and quant_config is not None
            and quant_config.get_name() in ("modelopt_fp4", "modelopt_mixed")
        ):
            self.deprecate_flag = True
        elif (
            deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and get_moe_runner_backend().is_deep_gemm()
            and quant_config is not None
            and quant_config.get_name() == "mxfp4"
        ):
            # MXFP4 experts (e.g. Kimi K3) on the DeepGEMM fp8_fp4 W4A8 path:
            # route through the modern FusedMoE runner (Mxfp4MoEMethod.apply).
            self.deprecate_flag = True
        elif (
            quant_config is None
            and self.w13_weight.dtype == torch.bfloat16
            and get_moe_runner_backend().is_deep_gemm()
            and (
                (
                    get_moe_a2a_backend().is_deepep()
                    and get_deepep_mode().enable_low_latency()
                )
                or get_moe_a2a_backend().is_pplx()
            )
            and not _is_npu
            and not _is_hip
        ):
            assert (
                deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            ), "Unquantized DeepEP low-latency MoE requires DeepGEMM BF16"
            self.deprecate_flag = True
        else:
            self.deprecate_flag = False

        if self.deprecate_flag:
            return

        self.use_w4a16_marlin = False
        if isinstance(quant_config, Fp8Config):
            self.use_block_quant = getattr(self.quant_method, "block_quant", False)
            self.use_fp8_w8a8 = True
            self.fp8_dtype = torch.float8_e4m3fn
            self.use_w4afp8 = False
            self.use_w4a8_marlin = False
            self.use_w8a8_marlin = False
            self.use_bf16_marlin = False
        elif isinstance(quant_config, W4AFp8Config):
            self.use_w4afp8 = True
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
            self.use_w4a8_marlin = False
            self.use_w8a8_marlin = False
            self.use_bf16_marlin = False
        elif isinstance(quant_config, SlimQuantW4A8Int8MarlinConfig):
            self.use_block_quant = getattr(self.quant_method, "block_quant", False)
            self.block_shape = (
                self.quant_method.quant_config.weight_block_size
                if self.use_block_quant
                else None
            )
            self.use_w4afp8 = False
            self.use_fp8_w8a8 = False
            self.activation_scheme = None
            self.use_w4a8_marlin = True
            self.use_w8a8_marlin = False
            self.use_bf16_marlin = False
        elif isinstance(quant_config, SlimQuantCompressedTensorsMarlinConfig):
            self.use_block_quant = getattr(self.quant_method, "block_quant", False)
            self.block_shape = (
                self.quant_method.quant_config.weight_block_size
                if self.use_block_quant
                else None
            )
            self.use_w4afp8 = False
            self.use_fp8_w8a8 = False
            self.activation_scheme = None
            self.use_w4a8_marlin = False
            self.use_w8a8_marlin = True
            self.use_bf16_marlin = False
        elif _use_fp8_w8a8_moe and _is_hcu:
            self.use_w4afp8 = False
            self.use_fp8_w8a8 = True
            self.use_block_quant = False
            self.use_w4afp8 = False
            self.use_w4a8_marlin = False
            self.use_w8a8_marlin = False
            self.use_bf16_marlin = False
        elif _use_marlin_w16a16_moe and _is_hcu:
            self.use_w4afp8 = False
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
            self.use_w4afp8 = False
            self.use_w4a8_marlin = False
            self.use_w8a8_marlin = False
            self.use_bf16_marlin = True
        elif _use_marlin_w4a16_moe and _is_hcu:
            self.use_w4afp8 = False
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
            self.use_w4a8_marlin = False
            self.use_w8a8_marlin = False
            self.use_bf16_marlin = False
            self.use_w4a16_marlin = True
        else:
            self.use_w4afp8 = False
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
            self.use_w4afp8 = False
            self.use_w4a8_marlin = False
            self.use_w8a8_marlin = False
            self.use_bf16_marlin = False

        self.deepep_mode = get_deepep_mode()

        if quant_config is None and hasattr(self.dispatcher, "set_quant_config"):
            self.dispatcher.set_quant_config({"bf16_dispatch": True})
        # if (
        #     self.deepep_mode.enable_low_latency()
        #     and not _is_npu
        #     and not _is_hip
        #     and not (
        #         get_moe_runner_backend().is_flashinfer_cutedsl()
        #         and self.quant_config.get_name() == "modelopt_fp4"
        #     )
        # ):
        #     # AMD HIP, NPU supports low_latency deepep without deepgemm
        #     # NV FP4 quantization with flashinfer_cutedsl also supports low_latency deepep without deepgemm
        #     assert (
        #         deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
        #     ), f"DeepEP {self.deepep_mode} mode requires deep_gemm"
        if _use_aiter:
            # expert_mask is of size (self.num_local_experts + 1),
            # the extra 1 is for invalid rank_id (in original deepep, the invalid rank_id is -1, but aiter does not allow -1, we use a mask to make those ids invalid)
            # for instance, if we have 4 experts on this rank, we would have a expert_mask like:
            #     self.expert_mask = [1, 1, 1, 1, 0]
            # idx from 0-3 is valid and will be processed, while idx == 4 will be masked out
            self.expert_mask = torch.zeros(
                (self.num_local_experts + 1),
                device=torch.cuda.current_device(),
                dtype=torch.int,
            )
            # the last one is invalid rank_id
            self.expert_mask[:-1] = 1

    def _a2a_forward_with_output_impl(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # eager run under breakable cuda graph
        saved_is_extend_in_batch = get_is_extend_in_batch()
        set_is_extend_in_batch(True)
        try:
            output.copy_(
                self.forward_impl(
                    hidden_states,
                    StandardTopKOutput(topk_weights, topk_ids, router_logits),
                )
            )
        finally:
            set_is_extend_in_batch(saved_is_extend_in_batch)

    def _a2a_forward_capture_stub(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # Capture pass only: record the buffer address, skip the
        # rank-coupled a2a. Warmup and replay run the real body.
        output.zero_()

    a2a_forward_with_output = eager_on_graph(
        True, capture_stub=_a2a_forward_capture_stub
    )(_a2a_forward_with_output_impl)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        i_q: Optional[torch.Tensor] = None,
        i_s: Optional[torch.Tensor] = None,
    ):
        # DeepEP NORMAL mode is not capturable; run it as an eager node.
        if is_in_breakable_cuda_graph():
            assert TopKOutputChecker.format_is_standard(
                topk_output
            ), "Only standard topk output is supported for breakable cuda graph"
            output = torch.empty_like(hidden_states)
            self.a2a_forward_with_output(
                hidden_states,
                topk_output.topk_weights,
                topk_output.topk_ids,
                topk_output.router_logits,
                output,
            )
            return output
        if is_in_tc_piecewise_cuda_graph():
            assert TopKOutputChecker.format_is_standard(
                topk_output
            ), "Only standard topk output is supported for piecewise cuda graph"
            return moe_forward_piecewise_cuda_graph_impl(
                hidden_states,
                topk_output.topk_weights,
                topk_output.topk_ids,
                topk_output.router_logits,
                self.layer_id,
            )
        else:
            return self.forward_impl(hidden_states, topk_output)

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):

        if self.deprecate_flag:
            return super().forward_impl(
                hidden_states,
                topk_output,
            )

        # TODO: can we call super().forward here?
        dispatch_output = self.dispatcher.dispatch(
            hidden_states=hidden_states, topk_output=topk_output
        )
        combine_input = self.run_moe_core(dispatch_output)
        hidden_states = self.dispatcher.combine(
            combine_input=combine_input,
        )

        return hidden_states

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        return self.dispatcher.dispatch(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

    def run_moe_core(
        self,
        dispatch_output: DispatchOutput,
    ):

        if self.deprecate_flag:
            return super().run_moe_core(
                dispatch_output,
            )

        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            # assert deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and self.use_fp8_w8a8
            if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and self.use_fp8_w8a8:
                output = self.forward_deepgemm_contiguous(dispatch_output)
            elif self.use_w4a8_marlin:
                output = self.forward_deepgemm_w4a8_marlin_contiguous(dispatch_output)
            elif self.use_w8a8_marlin:
                output = self.forward_groupgemm_w8a8_marlin_contiguous(dispatch_output)
            elif self.use_fp8_w8a8:
                output = self.forward_groupgemm_w8a8_fp8_contiguous(dispatch_output)
            elif self.use_bf16_marlin:
                output = self.forward_groupgemm_bf16_contiguous(dispatch_output)
            elif self.use_w4afp8:
                output = self.forward_cutlass_w4afp8(dispatch_output)
            else:
                raise ValueError(f"Dispatch output is not supported")
        elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
            if self.quant_config is None:
                output = self.forward_unquantized_deepep_ll(dispatch_output)
            elif (
                get_moe_runner_backend().is_flashinfer_cutedsl()
                and self.quant_config is not None
                and self.quant_config.get_name() == "modelopt_fp4"
            ):
                output = self.forward_flashinfer_cutedsl(dispatch_output)
            elif self.use_w4afp8:
                output = self.forward_cutlass_w4afp8_masked(dispatch_output)
            elif self.use_w4a8_marlin:
                output = self.forward_groupgemm_w4a8_marlin_masked(dispatch_output)
            elif self.use_w8a8_marlin:
                output = self.forward_groupgemm_w8a8_marlin_masked(dispatch_output)
            elif self.use_fp8_w8a8:
                output = self.forward_groupgemm_w8a8_fp8_masked(dispatch_output)
            elif self.use_bf16_marlin:
                output = self.forward_groupgemm_bf16_masked(dispatch_output)
            elif self.use_w4a16_marlin:
                output = self.forward_groupgemm_w4a16_marlin_masked(dispatch_output)
            else:
                assert False, "forward_deepgemm_masked is deprecated"
        elif _use_aiter:
            assert DispatchOutputChecker.format_is_deepep(dispatch_output)
            # in forward_aiter, we skip token permutation and unpermutation, which have been fused inside aiter kernel
            output = self.forward_aiter(dispatch_output)
        elif _is_npu:
            assert DispatchOutputChecker.format_is_deepep(dispatch_output)
            output = self.forward_npu(dispatch_output)

        combine_input_wrapper = (
            DeepEPNormalCombineInput
            if DispatchOutputChecker.format_is_deepep_normal(dispatch_output)
            else DeepEPLLCombineInput
        )

        return combine_input_wrapper(
            hidden_states=output,
            topk_ids=dispatch_output.topk_ids,
            topk_weights=dispatch_output.topk_weights,
        )

    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        overlap_args: Optional[Dict[str, Any]] = None,
    ):
        return self.dispatcher.combine(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            overlap_args=overlap_args,
        )

    def forward_aiter(
        self,
        dispatch_output: Union[DeepEPNormalDispatchOutput, DeepEPLLDispatchOutput],
    ):
        hidden_states, topk_ids, topk_weights = (
            dispatch_output.hidden_states,
            dispatch_output.topk_ids,
            dispatch_output.topk_weights,
        )

        if hidden_states.shape[0] == 0:
            return hidden_states

        # in original deepep, idx == -1 meaning invalid and will not be processed.
        # aiter does not accept -1, we use a expert mask to make these idx invalid
        # (idx == num_local_experts) meaning not used in aiter fused_moe
        topk_ids_copy = topk_ids.to(torch.int32)
        topk_ids_copy[topk_ids_copy == -1] = self.num_local_experts

        return fused_moe(
            hidden_states,
            self.w13_weight,
            self.w2_weight,
            topk_weights,
            topk_ids_copy,
            w1_scale=self.w13_weight_scale_inv,
            w2_scale=self.w2_weight_scale_inv,
            quant_type=QuantType.per_128x128,
            activation=(
                ActivationType.Silu
                if self.moe_runner_config.activation == "silu"
                else ActivationType.Gelu
            ),
            expert_mask=self.expert_mask,
        )

    def forward_unquantized_deepep_ll(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):
        hidden_states, hidden_states_scale, _, _, masked_m, _ = dispatch_output
        assert hidden_states_scale is None
        assert self.moe_runner_config.activation == "silu"
        assert self.moe_runner_config.is_gated
        assert hidden_states.dim() == 3

        num_experts, max_tokens, _ = hidden_states.shape
        token_offsets = torch.arange(max_tokens, device=hidden_states.device)
        valid_mask = (
            token_offsets.unsqueeze(0) < masked_m[:num_experts].unsqueeze(1)
        ).unsqueeze(-1)
        hidden_states = hidden_states.masked_fill(~valid_mask, 0)

        gate_up = torch.bmm(hidden_states, self.w13_weight.transpose(1, 2))
        w13_bias = getattr(self, "w13_weight_bias", None)
        if w13_bias is not None:
            gate_up = gate_up + w13_bias.unsqueeze(1)

        gate, up = gate_up.chunk(2, dim=-1)
        hidden_states = F.silu(gate) * up

        output = torch.bmm(hidden_states, self.w2_weight.transpose(1, 2))
        w2_bias = getattr(self, "w2_weight_bias", None)
        if w2_bias is not None:
            output = output + w2_bias.unsqueeze(1)
        return output.masked_fill(~valid_mask, 0)

    def forward_deepgemm_w4a8_marlin_contiguous(
        self,
        dispatch_output: DeepEPNormalDispatchOutput,
    ):
        (
            hidden_states,
            hidden_states_scale,
            topk_idx,
            topk_weights,
            num_recv_tokens_per_expert,
        ) = dispatch_output
        # hidden_states_int8, hidden_states_scale = hidden_states_int8
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"
        all_tokens = sum(num_recv_tokens_per_expert)

        if all_tokens <= 0:
            return hidden_states.bfloat16()
        rank_expert_offset = get_moe_expert_parallel_rank() * (
            self.num_experts // get_moe_expert_parallel_world_size()
        )
        topk_idx = torch.where(
            topk_idx == -1,
            self.num_experts - 1 if rank_expert_offset == 0 else 0,
            topk_idx + rank_expert_offset,
        )
        expert_output = self.quant_method.apply_ep(
            x=hidden_states,
            w1=self.w13_weight,
            w2=self.w2_weight,
            topk_ids=topk_idx,
            topk_weights=topk_weights,
            global_num_experts=self.moe_runner_config.num_experts,
            expert_map=self.expert_map,
            activation=self.moe_runner_config.activation,
            apply_router_weight_on_input=self.moe_runner_config.apply_router_weight_on_input,
            use_nn_moe=False,
            w1_scale=self.w13_weight_scale,
            w2_scale=self.w2_weight_scale,
            a1_scale=hidden_states_scale,
            routed_scaling_factor=self.moe_runner_config.routed_scaling_factor,
        )
        return expert_output

    # def forward_groupgemm_w8a8_marlin_contiguous(
    #     self,
    #     dispatch_output: DeepEPNormalOutput,
    # ):
    #     hidden_states, hidden_states_scale, topk_idx, topk_weights, num_recv_tokens_per_expert = dispatch_output
    #
    #     assert self.quant_method is not None
    #     assert self.moe_runner_config.activation == "silu"
    #     all_tokens = sum(num_recv_tokens_per_expert)
    #     if all_tokens <= 0:
    #         return hidden_states.bfloat16()
    #
    #     device = hidden_states.device
    #     M = hidden_states.shape[0]
    #     K = hidden_states.shape[1]
    #     topk = topk_idx.shape[1]
    #
    #     active_experts = set()
    #     token_expert_pos = [None] * M
    #     for t in range(M):
    #         lst = []
    #         for pos in range(topk):
    #             e = int(topk_idx[t, pos].item())
    #             if e >= 0:
    #                 lst.append((e, pos))
    #                 active_experts.add(e)
    #         token_expert_pos[t] = lst
    #
    #     if not active_experts:
    #         return hidden_states.bfloat16()
    #     active_experts = sorted(list(active_experts))
    #
    #     counts = defaultdict(int)
    #     for t in range(M):
    #         for (e, pos) in token_expert_pos[t]:
    #             counts[e] += 1
    #
    #     per_expert_block = {}
    #     for e in active_experts:
    #         cnt = counts[e]
    #         needed = ((cnt + 255) // 256) * 256  # same as ceil(cnt/256)*256
    #         per_expert_block[e] = max(256, needed)
    #
    #     expert_slot_offset = {}
    #     offset = 0
    #     for e in active_experts:
    #         expert_slot_offset[e] = offset
    #         offset += per_expert_block[e]
    #     pad_M = offset
    #
    #     hidden_states_packed = torch.empty((pad_M, K), device=device, dtype=hidden_states.dtype)
    #     hidden_states_scale_packed = torch.empty((pad_M,), device=device, dtype=hidden_states_scale.dtype)
    #     m_indices = torch.full((pad_M,), -1, device=device, dtype=torch.int32)
    #
    #     slot_counters = {e: 0 for e in active_experts}
    #     token_row_weight_list = {t: [] for t in range(M)}
    #
    #     for t in range(M):
    #         for (e, pos) in token_expert_pos[t]:
    #             start = expert_slot_offset[e]
    #             slot = slot_counters[e]
    #             row = start + slot
    #             hidden_states_packed[row] = hidden_states[t]
    #             hidden_states_scale_packed[row] = hidden_states_scale[t]
    #             m_indices[row] = e
    #             slot_counters[e] += 1
    #
    #             # record weight (as float32 on device)
    #             w = topk_weights[t, pos]
    #             w_f = w.float() if w.dtype != torch.float32 else w
    #             token_row_weight_list[t].append((row, w_f))
    #
    #     # q_a1_all, q_a1_scale = per_token_quant_int8(hidden_states_packed)
    #     N = self.w13_weight.size(1)
    #     gateup_output = torch.empty((pad_M, N * 16), device=device, dtype=torch.bfloat16)
    #     m_grouped_w8a8_gemm_nt_contig_asm(
    #         (hidden_states_packed, hidden_states_scale_packed),
    #         (self.w13_weight, self.w13_weight_scale),
    #         gateup_output,
    #         m_indices,
    #     )
    #     del hidden_states_packed, hidden_states_scale_packed
    #     q_a2_all, q_a2_scale = fuse_silu_mul_quant(gateup_output)
    #     down_output = torch.empty((pad_M, K), device=device, dtype=torch.bfloat16)
    #     down_output = m_grouped_w8a8_gemm_nt_contig_asm(
    #         (q_a2_all, q_a2_scale),
    #         (self.w2_weight, self.w2_weight_scale),
    #         down_output,
    #         m_indices,
    #     )
    #     result = torch.zeros((M, K), device=device, dtype=down_output.dtype)
    #     for t in range(M):
    #         for (row, w) in token_row_weight_list[t]:
    #             result[t].addcmul_(down_output[row].float(), w)
    #
    #     return result.to(down_output.dtype)

    def forward_groupgemm_w8a8_fp8_contiguous(
        self,
        dispatch_output: DeepEPNormalDispatchOutput,
    ):
        (
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            num_recv_tokens_per_expert,
        ) = dispatch_output
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"
        if num_recv_tokens_per_expert is None:
            return hidden_states.bfloat16()

        all_tokens = sum(num_recv_tokens_per_expert)
        if all_tokens <= 0:
            return hidden_states.bfloat16()

        M, K = hidden_states.size()
        w13_shape = getattr(self, "_dsv4_w13_weight_shape", None)
        if w13_shape is not None:
            N = w13_shape[1]
        else:
            N = self.w13_weight.size(1)
        # from deepgemm.m_group_gemm import pack_int8_weight_enk_to_w6_low_latency
        # w13_repacked = pack_int8_weight_enk_to_w6_low_latency(self.w13_weight)
        # w2_repacked = pack_int8_weight_enk_to_w6_low_latency(self.w2_weight)
        w13_weight_fp8 = (
            self.w13_weight_deepgemm,
            # self.w13_weight,
            self.w13_weight_scale,
        )

        w2_weight_fp8 = (
            self.w2_weight_deepgemm,
            # self.w2_weight,
            self.w2_weight_scale,
        )

        hidden_states_shape = hidden_states.shape
        hidden_states_device = hidden_states.device
        input_tensor = [
            torch.empty(
                (all_tokens, K),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            ),
            (
                torch.empty(
                    (all_tokens, hidden_states_scale.shape[-1]),
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
            ),
        ]
        if get_offloader().forbid_copy_engine_usage:
            num_recv_tokens_per_expert_gpu = copy_list_to_gpu_no_ce(
                num_recv_tokens_per_expert
            )
        else:
            num_recv_tokens_per_expert_gpu = torch.tensor(
                num_recv_tokens_per_expert,
                dtype=torch.int32,
                pin_memory=True,
                device="cpu",
            ).cuda(non_blocking=True)
        counts_are_aligned = all(
            count % 256 == 0 for count in num_recv_tokens_per_expert
        )
        m_indices, output_index = _ep_scatter_with_optional_lightop(
            hidden_states,
            hidden_states_scale,
            topk_ids,
            num_recv_tokens_per_expert_gpu,
            input_tensor[0],
            input_tensor[1],
            all_tokens,
            counts_are_aligned=counts_are_aligned,
        )

        gateup_output_factory = torch.empty if counts_are_aligned else torch.zeros
        gateup_output = gateup_output_factory(
            (all_tokens, N),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )

        m_grouped_fp8_gemm_nt_contiguous(
            input_tensor,
            w13_weight_fp8,
            gateup_output,
            m_indices,
        )
        del input_tensor

        swiglu_limit = self.moe_runner_config.swiglu_limit
        if swiglu_limit is None:
            q_a2_all, q_a2_scale = fuse_silu_mul_fp8_quant(
                gateup_output,
                fp8type=0,
            )
        else:
            q_a2_all, q_a2_scale = fuse_silu_mul_fp8_quant(
                gateup_output,
                fp8type=0,
                limit=swiglu_limit,
            )
        del gateup_output

        down_output = torch.empty(
            (all_tokens, K),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )

        m_grouped_fp8_gemm_nt_contiguous(
            (q_a2_all, q_a2_scale),
            w2_weight_fp8,
            down_output,
            m_indices,
        )

        gather_out = torch.empty(
            hidden_states_shape,
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )

        _ep_gather_with_optional_lightop(
            down_output, topk_ids, topk_weights, output_index, gather_out
        )
        del down_output

        return gather_out

    def forward_groupgemm_bf16_contiguous(
        self,
        dispatch_output: DeepEPNormalDispatchOutput,
    ):
        (
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            num_recv_tokens_per_expert,
        ) = dispatch_output
        assert self.moe_runner_config.activation == "silu"
        if num_recv_tokens_per_expert is None:
            return hidden_states.bfloat16()

        all_tokens = sum(num_recv_tokens_per_expert)
        if all_tokens <= 0:
            return hidden_states.bfloat16()

        M, K = hidden_states.size()
        N = self.w13_weight.size(1)

        hidden_states_shape = hidden_states.shape
        hidden_states_device = hidden_states.device

        input_tensor = torch.empty(
            (all_tokens, K), device=hidden_states.device, dtype=hidden_states.dtype
        )
        if get_offloader().forbid_copy_engine_usage:
            num_recv_tokens_per_expert_gpu = copy_list_to_gpu_no_ce(
                num_recv_tokens_per_expert
            )
        else:
            num_recv_tokens_per_expert_gpu = torch.tensor(
                num_recv_tokens_per_expert,
                dtype=torch.int32,
                pin_memory=True,
                device="cpu",
            ).cuda(non_blocking=True)
        expert_start_loc = torch.zeros_like(num_recv_tokens_per_expert_gpu)
        local_num_expert = num_recv_tokens_per_expert_gpu.shape[0]
        counts_are_aligned = all(
            count % 256 == 0 for count in num_recv_tokens_per_expert
        )
        m_indices = _build_m_indices_with_optional_lightop(
            topk_ids,
            hidden_states.device,
            local_num_expert,
            total_elements=all_tokens if counts_are_aligned else None,
            block_size=256,
            counts_are_aligned=counts_are_aligned,
        )
        output_index = torch.full(
            topk_ids.shape, -1, device=topk_ids.device, dtype=torch.int32
        )

        ep_scatter_no_scale(
            hidden_states,
            topk_ids,
            num_recv_tokens_per_expert_gpu,
            expert_start_loc,
            input_tensor,
            m_indices,
            output_index,
        )

        gateup_output = torch.zeros(
            (all_tokens, N),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )

        m_grouped_bf16_gemm_nt_contiguous(
            input_tensor,
            self.w13_weight,
            gateup_output,
            m_indices,
        )
        q_a2_all = torch.empty(
            (all_tokens, N // 2), device=hidden_states.device, dtype=torch.bfloat16
        )
        fuse_silu_and_mul(input=gateup_output, output=q_a2_all)
        del gateup_output

        down_output = torch.empty(
            (all_tokens, K),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )

        m_grouped_bf16_gemm_nt_contiguous(
            q_a2_all,
            self.w2_weight,
            down_output,
            m_indices,
        )

        gather_out = torch.empty(
            hidden_states_shape,
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )

        _ep_gather_with_optional_lightop(
            down_output, topk_ids, topk_weights, output_index, gather_out
        )
        del down_output

        return gather_out

    def forward_groupgemm_w8a8_marlin_contiguous(
        self,
        dispatch_output: DeepEPNormalDispatchOutput,
    ):
        (
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            num_recv_tokens_per_expert,
        ) = dispatch_output

        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"
        if num_recv_tokens_per_expert is None:
            return hidden_states.bfloat16()

        all_tokens = sum(num_recv_tokens_per_expert)
        if all_tokens <= 0:
            return hidden_states.bfloat16()

        M, K = hidden_states.size()
        w13_shape = getattr(self, "_dsv4_w13_weight_shape", None)
        if w13_shape is not None:
            N = w13_shape[1]
        else:
            N = self.w13_weight.size(1)
        w13_weight = getattr(self, "w13_weight_deepgemm", None)
        w2_weight = getattr(self, "w2_weight_deepgemm", None)
        if w13_weight is None:
            w13_weight = self.w13_weight
        if w2_weight is None:
            w2_weight = self.w2_weight
        w13_weight_int8 = (
            w13_weight,
            (self.w13_weight_scale),
        )
        w2_weight_int8 = (
            w2_weight,
            (self.w2_weight_scale),
        )

        hidden_states_shape = hidden_states.shape
        hidden_states_device = hidden_states.device
        a_int8 = torch.empty(
            (all_tokens, K),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        a_scale = torch.empty(
            (all_tokens, hidden_states_scale.shape[-1]),
            device=hidden_states.device,
            dtype=torch.float32,
        )
        if get_offloader().forbid_copy_engine_usage:
            num_recv_tokens_per_expert_gpu = copy_list_to_gpu_no_ce(
                num_recv_tokens_per_expert
            )
        else:
            num_recv_tokens_per_expert_gpu = torch.tensor(
                num_recv_tokens_per_expert,
                dtype=torch.int32,
                pin_memory=True,
                device="cpu",
            ).cuda(non_blocking=True)
        m_indices, output_index = _ep_scatter_with_optional_lightop(
            hidden_states,
            hidden_states_scale,
            topk_ids,
            num_recv_tokens_per_expert_gpu,
            a_int8,
            a_scale,
            all_tokens,
            counts_are_aligned=all(
                count % 256 == 0 for count in num_recv_tokens_per_expert
            ),
        )

        gateup_output_factory = (
            torch.empty
            if all(count % 256 == 0 for count in num_recv_tokens_per_expert)
            else torch.zeros
        )
        gateup_output = gateup_output_factory(
            (all_tokens, N),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        shuffle_unique, _mode = _get_deepgemm_shuffle_unique()
        m_grouped_i8_gemm_nt_contiguous(
            (a_int8, a_scale),
            w13_weight_int8,
            gateup_output,
            m_indices,
            shuffle_unique=shuffle_unique,
        )

        q_a2_all, q_a2_scale = fuse_silu_mul_quant(gateup_output)
        del gateup_output

        down_output = torch.empty(
            (all_tokens, K),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )

        m_grouped_i8_gemm_nt_contiguous(
            (q_a2_all, q_a2_scale),
            w2_weight_int8,
            down_output,
            m_indices,
            shuffle_unique=shuffle_unique,
        )

        gather_out = torch.zeros(
            hidden_states_shape,
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )

        _ep_gather_with_optional_lightop(
            down_output, topk_ids, topk_weights, output_index, gather_out
        )
        del down_output

        return gather_out

    def forward_deepgemm_contiguous(
        self,
        dispatch_output: DeepEPNormalDispatchOutput,
    ):
        (
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            num_recv_tokens_per_expert,
        ) = dispatch_output
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"
        if num_recv_tokens_per_expert is None:
            return hidden_states.bfloat16()
        all_tokens = sum(num_recv_tokens_per_expert)
        if all_tokens <= 0:
            return hidden_states.bfloat16()
        M, K = hidden_states.size()
        N = self.w13_weight.size(1)
        scale_block_size = 128

        w13_weight_fp8 = (
            self.w13_weight,
            (
                self.w13_weight_scale_inv
                if self.use_block_quant
                else self.w13_weight_scale
            ),
        )
        w2_weight_fp8 = (
            self.w2_weight,
            (
                self.w2_weight_scale_inv
                if self.use_block_quant
                else self.w2_weight_scale
            ),
        )

        hidden_states_shape = hidden_states.shape
        hidden_states_device = hidden_states.device
        scale_hidden_size = K // 128

        input_tensor = [
            torch.empty(
                (all_tokens, K),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            ),
            (
                # TODO check whether need `zeros`
                torch.zeros(
                    (ceil_div(K // 128, 4), all_tokens),
                    device=hidden_states.device,
                    dtype=torch.int,
                ).transpose(0, 1)
                if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
                else torch.empty(
                    (all_tokens, scale_hidden_size),
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
            ),
        ]
        m_indices = torch.empty(
            all_tokens, device=hidden_states.device, dtype=torch.int32
        )

        if get_offloader().forbid_copy_engine_usage:
            num_recv_tokens_per_expert_gpu = copy_list_to_gpu_no_ce(
                num_recv_tokens_per_expert
            )
        else:
            num_recv_tokens_per_expert_gpu = torch.tensor(
                num_recv_tokens_per_expert,
                dtype=torch.int32,
                pin_memory=True,
                device="cpu",
            ).cuda(non_blocking=True)
        output_index = torch.empty(
            topk_ids.shape, device=topk_ids.device, dtype=torch.int32
        )
        expert_start_loc = torch.empty_like(num_recv_tokens_per_expert_gpu)
        ep_scatter(
            hidden_states,
            hidden_states_scale,
            topk_ids,
            num_recv_tokens_per_expert_gpu,
            expert_start_loc,
            input_tensor[0],
            input_tensor[1],
            m_indices,
            output_index,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        )
        dispose_tensor(hidden_states)

        gateup_output = torch.empty(
            (all_tokens, N),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        if not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            input_tensor[1] = tma_align_input_scale(input_tensor[1])
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            input_tensor, w13_weight_fp8, gateup_output, m_indices
        )
        del input_tensor
        down_input = torch.empty(
            (
                all_tokens,
                N // 2,
            ),
            device=gateup_output.device,
            dtype=torch.bfloat16,
        )
        fuse_silu_and_mul(input=gateup_output.view(-1, N), output=down_input)
        del gateup_output
        down_output = torch.empty(
            (all_tokens, K),
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        down_input_fp8, down_input_scale = sglang_per_token_group_quant_fp8(
            down_input,
            scale_block_size,
            column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        )
        del down_input
        if not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            down_input_scale = tma_align_input_scale(down_input_scale)
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            (down_input_fp8, down_input_scale),
            w2_weight_fp8,
            down_output,
            m_indices,
        )
        del down_input_fp8, down_input_scale

        gather_out = torch.empty(
            hidden_states_shape,
            device=hidden_states_device,
            dtype=torch.bfloat16,
        )
        _ep_gather_with_optional_lightop(
            down_output, topk_ids, topk_weights, output_index, gather_out
        )

        return gather_out

    def forward_flashinfer_cutedsl(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):
        hidden_states, hidden_states_scale, _, _, masked_m, _ = dispatch_output
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"

        output = self.quant_method.apply_without_routing_weights(
            layer=self,
            x=(hidden_states, hidden_states_scale),
            masked_m=masked_m,
            moe_runner_config=self.moe_runner_config,
        )
        return output

    def forward_cutlass_w4afp8(
        self,
        dispatch_output: DeepEPNormalDispatchOutput,
    ):
        assert self.moe_runner_config.activation in ("silu", "situ")
        assert isinstance(self.quant_method, W4AFp8MoEMethod)
        return self.quant_method.apply_deepep_normal(
            layer=self,
            dispatch_output=dispatch_output,
        )

    def forward_groupgemm_w4a8_marlin_masked(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):

        hidden_states, hidden_states_scale, _, _, masked_m, expected_m = dispatch_output
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"

        # base shapes
        num_groups, m, k = hidden_states.size()
        expected_m = min(m, expected_m)

        # ---- first quant: ensure float input for quantizer ----
        # q_a1_all, q_a1_scale = per_token_quant_int8_triton_opt(hidden_states, masked_m)
        # ---- weights & scales ----
        w13_weight = self.w13_weight
        w13_scales = self.w13_weight_scale
        w2_weight = self.w2_weight
        w2_scales = self.w2_weight_scale

        n1 = w13_scales.size(1)
        gateup_output = torch.empty(
            (num_groups, m, n1), device=hidden_states.device, dtype=torch.bfloat16
        )

        # ---- first GEMM ----
        torch.ops.sglang.m_grouped_w4a8_gemm_nt_masked(
            hidden_states,
            hidden_states_scale,
            w13_weight,
            w13_scales,
            gateup_output,
            masked_m,
            expected_m,
        )

        q_a2_all, q_a2_scale = torch.ops.sglang.fuse_silu_mul_quant_ep(
            gateup_output, masked_m
        )
        # The first-stage BF16 activation is no longer needed after quantization.
        # Releasing it here lowers peak memory during low-latency graph capture.
        del gateup_output

        # ---- second GEMM ----
        n2 = w2_scales.size(1)
        down_output = torch.empty(
            (num_groups, m, n2), device=q_a2_all.device, dtype=torch.bfloat16
        )

        torch.ops.sglang.m_grouped_w4a8_gemm_nt_masked(
            q_a2_all,
            q_a2_scale,
            w2_weight,
            w2_scales,
            down_output,
            masked_m,
            expected_m,
        )

        return down_output

    def forward_groupgemm_w8a8_marlin_masked(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):

        hidden_states, hidden_states_scale, topk_ids, _, masked_m, expected_m = (
            dispatch_output
        )
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"
        # base shapes
        num_groups, m, k = hidden_states.size()
        expected_m = min(m, expected_m)

        # ---- first quant: ensure float input for quantizer ----
        # q_a1_all, q_a1_scale = per_token_quant_int8_triton_opt(hidden_states, masked_m)

        # ---- weights & scales ----
        w13_weight = self.w13_weight
        w13_scales = self.w13_weight_scale
        w2_weight = self.w2_weight
        w2_scales = self.w2_weight_scale

        n1 = w13_scales.size(1)
        gateup_output = torch.empty(
            (num_groups, m, n1), device=hidden_states.device, dtype=torch.bfloat16
        )

        # ---- first GEMM ----
        torch.ops.sglang.m_grouped_i8_gemm_nt_masked(
            hidden_states,
            hidden_states_scale,
            w13_weight,
            w13_scales,
            gateup_output,
            masked_m,
            expected_m,
        )

        q_a2_all, q_a2_scale = torch.ops.sglang.fuse_silu_mul_quant_ep(
            gateup_output, masked_m
        )
        # The first-stage BF16 activation is no longer needed after quantization.
        # Releasing it here lowers peak memory during low-latency graph capture.
        del gateup_output

        # ---- second GEMM ----
        n2 = w2_scales.size(1)
        down_output = torch.empty(
            (num_groups, m, n2), device=q_a2_all.device, dtype=torch.bfloat16
        )

        torch.ops.sglang.m_grouped_i8_gemm_nt_masked(
            q_a2_all,
            q_a2_scale,
            w2_weight,
            w2_scales,
            down_output,
            masked_m,
            expected_m,
        )

        return down_output

    def forward_groupgemm_w8a8_fp8_masked(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):

        hidden_states, hidden_states_scale, topk_ids, _, masked_m, expected_m = (
            dispatch_output
        )
        # This HCU low-latency DeepGEMM path predates the MoeRunner refactor.
        # Its overlap state is therefore stored directly on DeepEPMoE by
        # FusedMoE.set_overlap_args(), and no ``self.runner`` is constructed.
        down_gemm_overlap_args: Optional[DownGemmOverlapArgs] = getattr(
            self, "down_gemm_overlap_args", None
        )
        meta_overlap_args: Optional[dict] = getattr(
            self, "meta_overlap_args", None
        )
        assert self.moe_runner_config.activation == "silu"
        # base shapes
        num_groups, m, k = hidden_states.size()
        expected_m = min(m, expected_m)

        # from deepgemm.m_group_gemm import pack_int8_weight_enk_to_w6_low_latency
        # w13_repacked = pack_int8_weight_enk_to_w6_low_latency(self.w13_weight)
        # w2_repacked = pack_int8_weight_enk_to_w6_low_latency(self.w2_weight)

        # ---- weights & scales ----
        # w13_weight = self.w13_weight
        w13_weight = self.w13_weight_deepgemm
        w13_scales = self.w13_weight_scale
        # w2_weight = self.w2_weight
        w2_weight = self.w2_weight_deepgemm
        w2_scales = self.w2_weight_scale

        n1 = w13_scales.size(1)
        gateup_output = torch.empty(
            (num_groups, m, n1), device=hidden_states.device, dtype=torch.bfloat16
        )

        from deepgemm.m_group_gemm import m_grouped_fp8_gemm_nt_masked_ll

        m_grouped_fp8_gemm_nt_masked_ll(
            (hidden_states, hidden_states_scale),
            (w13_weight, w13_scales),
            gateup_output,
            masked_m,
            expected_m,
        )

        q_a2_all, q_a2_scale = fuse_silu_mul_fp8_quant_ep(
            input=gateup_output, fp8type=0, tokens_per_expert=masked_m
        )
        # The first-stage BF16 activation is no longer needed after quantization.
        # Releasing it here lowers peak memory during low-latency graph capture.
        del gateup_output

        # ---- second GEMM ----
        n2 = w2_scales.size(1)
        down_output = torch.empty(
            (num_groups, m, n2), device=q_a2_all.device, dtype=torch.bfloat16
        )

        enable_overlap = down_gemm_overlap_args is not None

        if enable_overlap:
            down_gemm_overlap_args.start_event.record()

        m_grouped_fp8_gemm_nt_masked_ll(
            (q_a2_all, q_a2_scale),
            (w2_weight, w2_scales),
            down_output,
            masked_m,
            expected_m,
            enable_overlap,
            down_gemm_overlap_args.signal if enable_overlap else None,
        )

        if meta_overlap_args is not None:
            meta_overlap_args["block_m"] = 64
            meta_overlap_args["threshold"] = 32

        return down_output

    def forward_groupgemm_bf16_masked(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):

        hidden_states, hidden_states_scale, topk_ids, _, masked_m, expected_m = (
            dispatch_output
        )
        assert self.moe_runner_config.activation == "silu"
        # base shapes
        num_groups, m, k = hidden_states.size()
        expected_m = min(m, expected_m)

        # ---- weights ----
        w13_weight = self.w13_weight
        w2_weight = self.w2_weight

        n1 = w13_weight.size(1)
        gateup_output = torch.empty(
            (num_groups, m, n1), device=hidden_states.device, dtype=torch.bfloat16
        )
        # ---- first GEMM ----
        m_grouped_bf16_gemm_nt_masked(
            hidden_states,
            w13_weight,
            gateup_output,
            masked_m,
            expected_m,
        )

        q_a2_all = torch.empty(
            (num_groups, m, n1 // 2), device=hidden_states.device, dtype=torch.bfloat16
        )
        fuse_silu_and_mul(input=gateup_output, output=q_a2_all)
        # The first-stage BF16 activation is no longer needed after SiLU*mul.
        # Releasing it here lowers peak memory during low-latency graph capture.
        del gateup_output
        # ---- second GEMM ----
        n2 = w2_weight.size(1)
        down_output = torch.empty(
            (num_groups, m, n2), device=q_a2_all.device, dtype=torch.bfloat16
        )

        m_grouped_bf16_gemm_nt_masked(
            q_a2_all,
            w2_weight,
            down_output,
            masked_m,
            expected_m,
        )

        return down_output

    def forward_cutlass_w4afp8_masked(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):
        assert self.moe_runner_config.activation in ("silu", "situ")
        assert isinstance(self.quant_method, W4AFp8MoEMethod)
        return self.quant_method.apply_deepep_ll(
            layer=self,
            dispatch_output=dispatch_output,
        )

    def forward_npu(
        self,
        dispatch_output: Union[DeepEPNormalDispatchOutput, DeepEPLLDispatchOutput],
    ):
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"

        from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
            npu_fused_moe_without_routing_weights_bf16,
        )
        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        # NOTE: Ascend's Dispatch & Combine does not support FP16
        output_dtype = torch.bfloat16
        group_list_type = 1

        if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            if TYPE_CHECKING:
                assert isinstance(dispatch_output, DeepEPNormalDispatchOutput)
            hidden_states, hidden_states_scale, _, _, num_recv_tokens_per_expert = (
                dispatch_output
            )

            group_list = torch.tensor(
                num_recv_tokens_per_expert,
                dtype=torch.int64,
                device=hidden_states.device,
            )

            if self.w13_weight.dtype == torch.bfloat16:
                hidden_states = npu_fused_moe_without_routing_weights_bf16(
                    self, hidden_states, group_list_type, group_list, output_dtype
                )
            else:
                input_quant = get_bool_env_var("DEEP_NORMAL_MODE_USE_INT8_QUANT")
                if not input_quant and not isinstance(
                    self.quant_method,
                    (
                        NPUCompressedTensorsW4A16Int4DynamicMoE,
                        CompressedTensorsFusedMoEMethod,
                    ),
                ):
                    hidden_states, hidden_states_scale = torch_npu.npu_dynamic_quant(
                        hidden_states
                    )
                hidden_states = self.quant_method.apply_without_routing_weights(
                    self,
                    hidden_states,
                    hidden_states_scale,
                    group_list_type,
                    group_list,
                    output_dtype,
                )
        elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
            if TYPE_CHECKING:
                assert isinstance(dispatch_output, DeepEPLLDispatchOutput)
            (
                hidden_states,
                hidden_states_scale,
                topk_ids,
                topk_weights,
                group_list,
                _,
            ) = dispatch_output

            group_list = group_list.to(torch.int64)

            if self.w13_weight.dtype == torch.bfloat16:
                hidden_states = npu_fused_moe_without_routing_weights_bf16(
                    self, hidden_states, group_list_type, group_list, output_dtype
                )
            else:
                hidden_states = self.quant_method.apply_without_routing_weights(
                    self,
                    hidden_states,
                    hidden_states_scale,
                    group_list_type,
                    group_list,
                    output_dtype,
                )
        else:
            raise ValueError(f"Not Supported DeepEP format {dispatch_output.format}")

        return hidden_states

    def forward_groupgemm_w4a16_marlin_masked(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):
        hidden_states, _, _, _, masked_m, expected_m = dispatch_output
        assert self.quant_method is not None
        assert self.moe_runner_config.activation == "silu"

        num_groups, m, _ = hidden_states.size()
        expected_m = min(m, expected_m)
        w13_weight = self.w13_weight_packed
        w13_scales = self.w13_weight_scale
        w2_weight = self.w2_weight_packed
        w2_scales = self.w2_weight_scale

        n1 = w13_scales.size(1)
        gateup_output = torch.empty(
            (num_groups, m, n1),
            device=hidden_states.device,
            dtype=torch.bfloat16,
        )
        grouped_gemm_w4a16_nt_masked_entry(
            hidden_states,
            w13_weight,
            w13_scales,
            gateup_output,
            masked_m,
            expected_m,
        )

        q_a2_all = torch.empty(
            (num_groups, m, n1 // 2),
            device=hidden_states.device,
            dtype=torch.bfloat16,
        )
        fuse_silu_and_mul(
            input=gateup_output,
            output=q_a2_all,
            mask_m=masked_m,
            expect_m=expected_m,
        )
        del gateup_output

        n2 = w2_scales.size(1)
        down_output = torch.empty(
            (num_groups, m, n2),
            device=q_a2_all.device,
            dtype=torch.bfloat16,
        )
        grouped_gemm_w4a16_nt_masked_entry(
            q_a2_all,
            w2_weight,
            w2_scales,
            down_output,
            masked_m,
            expected_m,
        )
        return down_output


class NpuFuseEPMoE(DeepEPMoE):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
            **kwargs,
        )

        self.quant_method.process_weights_after_loading = (
            self._process_weights_after_loading
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        forward_shared_experts=None,
        alt_stream=None,
        disable_sbo=False,
    ):
        return self.dispatcher.dispatch(
            hidden_states=hidden_states,
            topk_output=topk_output,
            gmm1_permuted_weight=self.w13_weight,
            gmm1_permuted_weight_scale=self.w13_weight_scale,
            gmm2_weight=self.w2_weight,
            gmm2_weight_scale=self.w2_weight_scale,
        ).hidden_state

    def permute_w13_weight_scale(self, w: torch.Tensor, tile_n: int):
        if tile_n % 2 != 0:
            raise ValueError(f"tile_n must be even, got {tile_n}")

        *dims, n = w.shape
        if n % tile_n != 0:
            raise ValueError(f"Last dimension {n} must be divisible by tile_n {tile_n}")

        w_reshaped = w.reshape(*dims, 2, n // tile_n, tile_n // 2)

        # Permute the last two dimensions.
        perm_order = list(range(len(dims))) + [-2, -3, -1]
        w_permuted = w_reshaped.permute(perm_order)

        return w_permuted.reshape(*dims, n)

    def reshape_w13_weight(self, weight: torch.Tensor, dim: int, chunk_size: int = 64):
        # Achieving greater computing power through reshape on Ascend.
        original_shape = weight.shape
        if dim < 0:
            dim += len(original_shape)

        if original_shape[dim] % (2 * chunk_size) != 0:
            raise ValueError(
                f"Dimension {dim} size {original_shape[dim]} must be divisible by {2 * chunk_size}"
            )

        new_shape = (
            *original_shape[:dim],
            2,
            original_shape[dim] // (2 * chunk_size),
            chunk_size,
            *original_shape[dim + 1 :],
        )

        weight = weight.view(new_shape)
        weight = weight.transpose(dim, dim + 1).contiguous()

        return weight.view(*original_shape[:dim], -1, *original_shape[dim + 1 :])

    def release_weight_cache(self, weight: torch.Tensor):
        # .contiguous() introduces additional memory overhead and needs to be released using resize_(0)
        origin_weight = weight.data.transpose(1, 2)
        new_weight = origin_weight.contiguous()
        origin_weight.untyped_storage().resize_(0)
        return new_weight

    def scale_from_float_to_int64(self, scale):
        import numpy as np

        scale = torch.from_numpy(
            np.frombuffer(
                scale.cpu().to(torch.float32).numpy().tobytes(), dtype=np.int32
            ).astype(np.int64)
        ).to(scale.device)
        return torch.nn.Parameter(scale, requires_grad=False)

    def _process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if (
            envs.SGLANG_NPU_FUSED_MOE_MODE.get()
            == FusedMoEMode.DISPATCH_FFN_COMBINE.value
        ):
            w13_weight = self.release_weight_cache(layer.w13_weight)
            layer.w13_weight.data = npu_format_cast(w13_weight)
            w2_weight = self.release_weight_cache(layer.w2_weight)
            layer.w2_weight.data = npu_format_cast(w2_weight)

            layer.w13_weight_scale.data = layer.w13_weight_scale.data.view(
                layer.w13_weight_scale.data.shape[0], -1
            )
            w2_scale = layer.w2_weight_scale.data.squeeze(-1).contiguous()
            layer.w2_weight_scale = torch.nn.Parameter(
                w2_scale.to(torch.float32), requires_grad=False
            )

            layer.w13_weight_scale = self.scale_from_float_to_int64(
                layer.w13_weight_scale.data
            )
            layer.w2_weight_scale = self.scale_from_float_to_int64(
                layer.w2_weight_scale.data
            )
        else:
            cpu_w13 = layer.w13_weight.data.transpose(1, 2).cpu()
            layer.w13_weight.data = self.reshape_w13_weight(cpu_w13, -1).npu()
            w13_scale = layer.w13_weight_scale.data.squeeze(-1).contiguous()
            w13_scale = self.permute_w13_weight_scale(w13_scale, 128)
            layer.w13_weight_scale = torch.nn.Parameter(
                w13_scale.to(torch.float32), requires_grad=False
            )
            layer.w13_weight.data = npu_format_cast(layer.w13_weight.data)
            layer.w2_weight.data = npu_format_cast(layer.w2_weight.data)

            w2_scale = layer.w2_weight_scale.data.squeeze(-1).contiguous()
            layer.w2_weight_scale = torch.nn.Parameter(
                w2_scale.to(torch.float32), requires_grad=False
            )

        if hasattr(layer, "w13_weight_offset"):
            layer.w13_weight_offset = torch.nn.Parameter(
                layer.w13_weight_offset.data.squeeze(-1).contiguous(),
                requires_grad=False,
            )
        if hasattr(layer, "w2_weight_offset"):
            layer.w2_weight_offset = torch.nn.Parameter(
                layer.w2_weight_offset.data.squeeze(-1).contiguous(),
                requires_grad=False,
            )


class MoriEPMoE(DeepEPMoE):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
            **kwargs,
        )

        assert _use_aiter, "Mori need to be used together with aiter as of now"
        self.expert_mask = torch.zeros(
            (self.num_experts),
            device=torch.cuda.current_device(),
            dtype=torch.int32,
        )
        expert_start_idx = self.moe_ep_rank * self.num_local_experts
        expert_end_idx = expert_start_idx + self.num_local_experts
        self.expert_mask[expert_start_idx:expert_end_idx] = 1

        self.mori_moe_max_input_tokens = get_int_env_var(
            "SGLANG_MORI_MOE_MAX_INPUT_TOKENS", 0
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        num_token = hidden_states.shape[0]
        dispatch_output = self.dispatcher.dispatch(
            hidden_states=hidden_states, topk_output=topk_output
        )
        combine_input = self.run_moe_core(dispatch_output)
        hidden_states = self.dispatcher.combine(
            combine_input=combine_input,
        )

        return hidden_states[:num_token]

    def run_moe_core(
        self,
        dispatch_output: DispatchOutput,
    ):
        is_fp8_quant = isinstance(self.quant_method, Fp8MoEMethod)
        is_quark_w4a4 = hasattr(self, "scheme") and isinstance(
            self.scheme, QuarkW4A4MXFp4MoE
        )

        (
            dispatch_a1,
            dispatch_scale,
            dispatch_ids,
            dispatch_weights,
            dispatch_recv_token_num,
            _origin_topk_ids,
            _origin_topk_weights,
            output_dtype,
        ) = (
            dispatch_output.hidden_states,
            dispatch_output.hidden_states_scale,
            dispatch_output.topk_ids,
            dispatch_output.topk_weights,
            dispatch_output.num_recv_tokens_per_expert,
            dispatch_output.origin_topk_ids,
            dispatch_output.origin_topk_weights,
            dispatch_output.out_dtype,
        )

        # Truncate dispatch tensors to reduce MoE computation on padding rows.
        # dispatch_a1 has shape (M, hidden_size) where M is the full buffer size,
        # but only the first dispatch_recv_token_num rows are valid.
        # mori combine only reads [0, totalRecvTokenNum), so the truncated
        # output can be passed directly without padding back.
        if self.mori_moe_max_input_tokens > 0:
            limit = self.mori_moe_max_input_tokens
            dispatch_a1 = dispatch_a1[:limit]
            if dispatch_scale is not None:
                dispatch_scale = dispatch_scale[:limit]
            dispatch_ids = dispatch_ids[:limit]
            dispatch_weights = dispatch_weights[:limit]

        w13_weight = self.w13_weight
        w2_weight = self.w2_weight

        w13_scale = None
        w2_scale = None

        quant_type = QuantType.No

        if (
            not is_fp8_quant
            and dispatch_scale is not None
            and dispatch_a1.dtype != torch.float4_e2m1fn_x2
        ):
            if is_quark_w4a4:
                # W4A4 model with FP8 dispatch: must dequant FP8->BF16 first,
                # because the FP4 per_1x32 quantization path needs BF16 input
                dispatch_a1 = upscale(
                    dispatch_a1, dispatch_scale, dispatch_recv_token_num, output_dtype
                )
                dispatch_scale = None
            else:
                # Non-W4A4 model with FP8 dispatch: pass FP8 hidden_states + scale
                # directly to fused_moe, avoiding unnecessary dequant->requant round-trip
                quant_type = QuantType.per_128x128

        if dispatch_a1.dtype == torch.float4_e2m1fn_x2 and dispatch_scale is not None:
            if is_fp8_quant:
                # FP8 weights + FP4 dispatch is not supported by fused_moe kernels
                # (no kernel for q_dtype_a=fp4x2, q_dtype_w=fp8).
                # Must dequant FP4->BF16 first; fused_moe will re-quant to FP8 internally.
                dispatch_a1 = upscale_mxfp4(
                    dispatch_a1, dispatch_scale, dispatch_recv_token_num, output_dtype
                )
                dispatch_scale = None
            elif quant_type == QuantType.No:
                # Skip upscale_mxfp4: pass FP4 hidden_states + scale directly to fused_moe
                # fused_moe with QuantType.per_1x32 can accept pre-quantized fp4x2 input
                quant_type = QuantType.per_1x32

        if is_quark_w4a4:
            if hasattr(torch, "float4_e2m1fn_x2"):
                w13_weight = self.w13_weight.view(torch.float4_e2m1fn_x2)
                w2_weight = self.w2_weight.view(torch.float4_e2m1fn_x2)

            w13_scale = self.w13_weight_scale
            w2_scale = self.w2_weight_scale
            quant_type = QuantType.per_1x32

            if hasattr(self.w13_weight, "is_shuffled"):
                w13_weight.is_shuffled = True
                w2_weight.is_shuffled = True
        elif is_fp8_quant:
            if hasattr(self, "w13_weight_scale_inv"):
                w13_scale = self.w13_weight_scale_inv
            if hasattr(self, "w2_weight_scale_inv"):
                w2_scale = self.w2_weight_scale_inv

            # Only set per_128x128 if quant_type was not already set by
            # a prior dispatch path (e.g. FP4 dispatch sets per_1x32)
            if quant_type == QuantType.No:
                quant_type = QuantType.per_128x128

        # [KK TODO] should to call the apply of quant method to handle fused moe
        hidden_states = fused_moe(
            hidden_states=dispatch_a1,
            w1=w13_weight,
            w2=w2_weight,
            w1_scale=w13_scale,
            w2_scale=w2_scale,
            a1_scale=dispatch_scale,
            topk_weight=dispatch_weights,
            topk_ids=dispatch_ids,
            quant_type=quant_type,
            activation=(
                ActivationType.Silu
                if self.moe_runner_config.activation == "silu"
                else ActivationType.Gelu
            ),
            expert_mask=self.expert_mask,
            num_local_tokens=dispatch_recv_token_num,
            dtype=output_dtype,
        )

        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        combine_input_wrapper = (
            MoriEPNormalCombineInput
            if DispatchOutputChecker.format_is_deepep_normal(dispatch_output)
            else MoriEPLLCombineInput
        )

        return combine_input_wrapper(
            hidden_states=hidden_states,
            topk_ids=dispatch_output.origin_topk_ids,
            topk_weights=dispatch_output.origin_topk_weights,
        )


def get_moe_impl_class(quant_config: Optional[QuantizationConfig]):
    # [TODO] kk, temporary solution
    if get_moe_a2a_backend().is_mori():
        return MoriEPMoE
    if (
        get_moe_a2a_backend().is_deepep()
        or get_moe_a2a_backend().is_mooncake()
        or get_moe_a2a_backend().is_nixl()
        or get_moe_a2a_backend().is_pplx()
    ):
        return DeepEPMoE
    if get_moe_a2a_backend().is_ascend_fuseep():
        return NpuFuseEPMoE

    return FusedMoE
