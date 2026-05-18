# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2023-2024 SGLang Team
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
# ==============================================================================

# Adapted from:
# https://github.com/vllm-project/vllm/blob/fb6af8bc086328ca6659e72d11ffd4309ce4de22/vllm/model_executor/models/deepseek_v2.py
"""Inference-only DeepseekV2 model."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.jit_kernel.deepseek_v4 import (
    silu_and_mul_clamp,
    silu_and_mul_contig_post_quant,
)
from sglang.srt.batch_overlap.single_batch_overlap import SboFlags, compute_overlap_args
from sglang.srt.batch_overlap.two_batch_overlap import (
    MaybeTboDeepEPDispatcher,
    model_forward_maybe_tbo,
)
from sglang.srt.configs.model_config import (
    compute_mla_mscale_scaling,
    get_nsa_index_head_dim,
    get_nsa_index_n_heads,
    get_nsa_index_topk,
    is_deepseek_nsa,
)
from sglang.srt.distributed import (
    divide,
    get_moe_expert_parallel_world_size,
    get_pp_group,
    parallel_state,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
    parallel_state,
)
from sglang.srt.environ import envs
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.amx_utils import PackWeightMethod
from sglang.srt.layers.attention.nsa.nsa_indexer import Indexer
from sglang.srt.layers.attention.nsa.utils import (
    can_nsa_cp_split,
    is_nsa_enable_prefill_cp,
    nsa_use_prefill_cp,
)
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    enable_moe_dense_fully_dp,
    get_attn_tp_context,
)
from sglang.srt.layers.communicator_nsa_cp import NSACPLayerCommunicator
from sglang.srt.layers.dp_attention import (
    get_attention_cp_rank,
    get_attention_cp_size,
    get_attention_tp_group,
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    get_moe_runner_backend,
    should_skip_post_experts_all_reduce,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.hash_topk import HashTopK
from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    DispatchOutput,
)
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.moe.utils import (
    RoutingMethodType,
    filter_moe_weight_param_global_expert,
    is_deepep_class_backend,
    is_sbo_enabled,
    is_tbo_enabled,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
)
from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
    maybe_fuse_routed_scale_and_shared_add,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.utils.cp_utils import (
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    cp_split_and_rebuild_position,
    prepare_context_parallel_metadata,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.deepseek_common.attention_backend_handler import (
    AttentionBackendRegistry,
)
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    AttnForwardMethod,
    DeepseekMHAForwardMixin,
    DeepseekMLACpuForwardMixin,
    DeepseekMLAForwardMixin,
    DeepseekMLARocmForwardMixin,
)
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.models.deepseek_common.utils import (
    _device_sm,
    _get_llama_4_scaling,
    _is_cpu,
    _is_cpu_amx_available,
    _is_cuda,
    _is_gfx95_supported,
    _is_hip,
    _is_musa,
    _is_npu,
    _is_xpu,
    _use_aiter,
    _use_aiter_gfx95,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    BumpAllocator,
    LazyValue,
    add_prefix,
    get_bool_env_var,
    get_int_env_var,
    get_device_sm,
    is_cpu,
    is_cuda,
    is_dcu,
    is_npu,
    is_gfx95_supported,
    is_hip,
    is_non_idle_and_non_empty,
    log_info_on_rank0,
    make_layers,
    use_intel_amx_backend,
    cpu_has_amx_support,
)
from sglang.srt.utils.custom_op import register_custom_op

if _use_aiter:
    from sglang.srt.layers.rocm_linear_utils import aiter_dsv3_router_gemm
from sglang.srt.layers.attention.lightop_concat import concat_decode_opt
from sglang.srt.layers.moe.utils import is_sbo_enabled
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed import (
    parallel_state,
)
from enum import IntEnum, auto
import tqdm
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
_is_dcu = is_dcu()
_is_hip = is_hip()
_is_cuda = is_cuda()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_device_sm = get_device_sm()
_is_gfx95_supported = is_gfx95_supported()
_is_sbo_enabled = is_sbo_enabled()
_use_lightop_moe_sum_mul_add = get_bool_env_var("SGLANG_USE_LIGHTOP_MOE_SUM_MUL_ADD")
_use_fused_silu_mul_quant = get_bool_env_var("SGLANG_USE_FUSED_SILU_MUL_QUANT")
_use_aiter_gfx95 = _use_aiter and _is_gfx95_supported
_use_opt_cat_decode = get_bool_env_var("SGLANG_USE_OPT_CAT")
_use_fused_mla_cat = get_bool_env_var("SGLANG_USE_FUSED_MLA_CAT")
_use_fused_rmsnorm_rope = get_bool_env_var("SGLANG_USE_FUSED_RMSNORM_ROPE")
_use_fused_rms_quant = get_bool_env_var("SGLANG_USE_FUSED_RMS_QUANT")
_rms_quant_path = get_int_env_var('SGLANG_USE_RMS_QUANT_PATH')
if _use_fused_rmsnorm_rope:
    from lightop import fused_rms_norm_rope_contiguous
    fused_rms_norm_rope_contiguous = torch._dynamo.disable(fused_rms_norm_rope_contiguous)
    # 此处注册自定义算子反而导致多了很多triton kernel影响性能
    # def fused_rms_norm_rope_wrapper(
    #     positions: torch.Tensor,
    #     q_pe: torch.Tensor,
    #     k_pe: torch.Tensor,
    #     kv_c: torch.Tensor,
    #     kv_c_normed: torch.Tensor,
    #     weight: torch.Tensor,
    #     cos_sin_cache: torch.Tensor,
    #     slot_mapping: torch.Tensor,
    #     kv_cache: torch.Tensor,
    #     kv_cache_dtype_str: str,
    #     kv_cache_scale: float,
    #     is_neox_style: bool = False,
    #     epsilon: float = 1e-5,
    # ) -> None:
    #     return fused_rms_norm_rope_contiguous(
    #         positions=positions,
    #         q_pe=q_pe,
    #         k_pe=k_pe,
    #         kv_c=kv_c,
    #         kv_c_normed=kv_c_normed,
    #         weight=weight,
    #         cos_sin_cache=cos_sin_cache,
    #         slot_mapping=slot_mapping,
    #         kv_cache=kv_cache,
    #         kv_cache_dtype_str=kv_cache_dtype_str,
    #         kv_cache_scale=kv_cache_scale,
    #         is_neox_style=is_neox_style,
    #         epsilon=epsilon,
    #     )
    # def fused_rms_norm_rope_fake(
    #     positions: torch.Tensor,
    #     q_pe: torch.Tensor,
    #     k_pe: torch.Tensor,
    #     kv_c: torch.Tensor,
    #     kv_c_normed: torch.Tensor,
    #     weight: torch.Tensor,
    #     cos_sin_cache: torch.Tensor,
    #     slot_mapping: torch.Tensor,
    #     kv_cache: torch.Tensor,
    #     kv_cache_dtype_str: str,
    #     kv_cache_scale: float,
    #     is_neox_style: bool = False,
    #     epsilon: float = 1e-5,
    #     ) -> None:
    #         return q_pe, k_pe, kv_c_normed, kv_cache
    # direct_register_custom_op(
    #     op_name="fused_rms_norm_rope_contiguous",
    #     op_func=fused_rms_norm_rope_wrapper,
    #     mutates_args=['q_pe', 'k_pe', 'kv_c_normed', 'kv_cache'],
    #     fake_impl=fused_rms_norm_rope_fake,
    # )


if _use_aiter_gfx95:
    from sglang.srt.layers.rocm_linear_utils import (
        get_dsv3_gemm_output_zero_allocator_size,
    )

if _use_aiter:
    pass

if _is_cuda:
    from flashinfer.gemm import mm_M1_16_K7168_N256 as _raw_dsv3_router_gemm
    from sgl_kernel import dsv3_fused_a_gemm, dsv3_router_gemm
elif _is_hip:
    from sglang.srt.layers.attention.triton_ops.rocm_mla_decode_rope import (
        decode_attention_fwd_grouped_rope,
    )
    from sglang.srt.layers.quantization.awq.awq_triton import (
        awq_dequantize_triton as awq_dequantize,
    )
    from sgl_kernel import merge_state_v2
elif _is_npu:
    from sglang.srt.hardware_backend.npu.modules.deepseek_v2_attention_mla_npu import (
        forward_dsa_core_npu,
        forward_dsa_prepare_npu,
        forward_mha_core_npu,
        forward_mha_prepare_npu,
        forward_mla_core_npu,
        forward_mla_prepare_npu,
    )
elif _is_musa:
    from sgl_kernel import dsv3_fused_a_gemm, dsv3_router_gemm
else:
    pass

logger = logging.getLogger(__name__)

# 暂时先放这
def ds_bmm_wrapper(q: torch.Tensor, w: torch.Tensor, scale: float, dtype: torch.dtype):
    # # scale=1时去掉elementwise数乘
    if abs(scale - 1) < 1e-6:
        q_out = torch.bmm(q.to(dtype).transpose(0, 1), w.to(dtype))
    else:
        q_out = torch.bmm(q.to(dtype).transpose(0, 1), w.to(dtype) * scale)
    return q_out


FORWARD_ABSORB_CORE_ATTENTION_BACKENDS = [
    "fa3",
    "nsa",
    "flashinfer",
    "cutlass_mla",
    "trtllm_mla",
    "ascend",
]
dtype_mapping = {
    torch.float8_e4m3fn: "fp8_e4m3",
    torch.float8_e4m3fnuz: "fp8_e4m3",
    torch.float8_e5m2: "fp8_e5m2",
    torch.float8_e5m2fnuz: "fp8_e5m2",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}

def add_forward_absorb_core_attention_backend(backend_name):
    if backend_name not in FORWARD_ABSORB_CORE_ATTENTION_BACKENDS:
        FORWARD_ABSORB_CORE_ATTENTION_BACKENDS.append(backend_name)
        logger.info(f"Added {backend_name} to FORWARD_ABSORB_CORE_ATTENTION_BACKENDS.")


class AttnForwardMethod(IntEnum):
    # Use multi-head attention
    MHA = auto()

    # Use absorbed multi-latent attention
    MLA = auto()

    # Use multi-head attention, but with KV cache chunked.
    # This method can avoid OOM when prefix lengths are long.
    MHA_CHUNKED_KV = auto()

    # Use multi-head attention, execute the MHA for prefix and extended kv in one shot
    # when the sequence lengths are below the threshold.
    MHA_ONE_SHOT = auto()

    # Use MLA but with fused RoPE
    MLA_FUSED_ROPE = auto()

    # Use MLA with fused RoPE kernel for CPU
    MLA_FUSED_ROPE_CPU = auto()

    # Use multi-head attention for NPU
    MHA_NPU = auto()

    # Use absorbed multi-latent attention for NPU
    MLA_NPU = auto()

    # Use Deepseek V3.2 sparse multi-latent attention for NPU
    DSA_NPU = auto()


def _dispatch_mla_subtype(attn, forward_batch):
    if _is_hip:
        if attn.rocm_fused_decode_mla and forward_batch.forward_mode.is_decode():
            return AttnForwardMethod.MLA_FUSED_ROPE
        else:
            return AttnForwardMethod.MLA
    else:
        if hasattr(attn, "fused_qkv_a_proj_with_mqa") and use_intel_amx_backend(attn):
            return AttnForwardMethod.MLA_FUSED_ROPE_CPU
        else:
            return AttnForwardMethod.MLA


class AttentionBackendRegistry:
    _handlers = {}

    @classmethod
    def register(cls, backend_name, handler_func):
        cls._handlers[backend_name] = handler_func

    @classmethod
    def get_handler(cls, backend_name):
        return cls._handlers.get(backend_name, cls._handlers.get("triton"))


def handle_attention_ascend(attn, forward_batch):
    if (
        forward_batch.forward_mode.is_extend()
        and not forward_batch.forward_mode.is_target_verify()
        and not forward_batch.forward_mode.is_draft_extend()
        and not forward_batch.forward_mode.is_draft_extend_v2()
    ):
        if hasattr(attn, "indexer"):
            return AttnForwardMethod.DSA_NPU
        else:
            return AttnForwardMethod.MHA_NPU
    else:
        if hasattr(attn, "indexer"):
            return AttnForwardMethod.DSA_NPU
        else:
            return AttnForwardMethod.MLA_NPU


def _get_sum_extend_prefix_lens(forward_batch):
    return (
        sum(forward_batch.extend_prefix_lens_cpu)
        if forward_batch.extend_prefix_lens_cpu is not None
        else 0
    )


def _support_mha_one_shot(attn: DeepseekV2AttentionMLA, forward_batch, backend_name):
    attn_supported = backend_name in ["fa3", "flashinfer", "flashmla"]
    sum_seq_lens = (
        sum(forward_batch.seq_lens_cpu) if forward_batch.seq_lens_cpu is not None else 0
    )
    return attn_supported and sum_seq_lens <= forward_batch.get_max_chunk_capacity()


def _handle_attention_backend(
    attn: DeepseekV2AttentionMLA, forward_batch, backend_name
):
    if is_in_piecewise_cuda_graph():
        return AttnForwardMethod.MLA

    sum_extend_prefix_lens = _get_sum_extend_prefix_lens(forward_batch)
    disable_ragged = (
        backend_name in ["flashinfer", "flashmla"]
    ) and attn.flashinfer_mla_disable_ragged

    if (
        not disable_ragged
        and forward_batch.forward_mode.is_extend_without_speculative()
        and (
            (
                sum_extend_prefix_lens >= attn.chunked_prefix_cache_threshold
                and not attn.disable_chunked_prefix_cache
            )
            or sum_extend_prefix_lens == 0
        )
    ):
        if _support_mha_one_shot(attn, forward_batch, backend_name):
            return AttnForwardMethod.MHA_ONE_SHOT
        return AttnForwardMethod.MHA_CHUNKED_KV
    else:
        return _dispatch_mla_subtype(attn, forward_batch)


def handle_attention_flashinfer(attn, forward_batch):
    return _handle_attention_backend(attn, forward_batch, "flashinfer")


def handle_attention_fa3(attn, forward_batch):
    # when deterministic inference is enabled, use MLA
    if get_global_server_args().enable_deterministic_inference:
        return _dispatch_mla_subtype(attn, forward_batch)
    else:
        return _handle_attention_backend(attn, forward_batch, "fa3")


def handle_attention_flashmla(attn, forward_batch):
    return _handle_attention_backend(attn, forward_batch, "flashmla")


def handle_attention_dcu_mla(attn, forward_batch):
    return _handle_attention_backend(attn, forward_batch, "dcu_mla")


def handle_attention_cutlass_mla(attn, forward_batch):
    return _handle_attention_backend(attn, forward_batch, "cutlass_mla")


def handle_attention_fa4(attn, forward_batch):
    # TODO(cicirori): use FA4 MHA for DeepSeekV3 for now
    return AttnForwardMethod.MHA_CHUNKED_KV


def handle_attention_trtllm_mla(attn, forward_batch):
    if is_in_piecewise_cuda_graph():
        return AttnForwardMethod.MLA

    sum_extend_prefix_lens = _get_sum_extend_prefix_lens(forward_batch)
    if forward_batch.forward_mode.is_extend_without_speculative() and (
        not attn.disable_chunked_prefix_cache or sum_extend_prefix_lens == 0
    ):
        return AttnForwardMethod.MHA_CHUNKED_KV
    else:
        return _dispatch_mla_subtype(attn, forward_batch)


def handle_attention_aiter(attn, forward_batch):
    if forward_batch.forward_mode.is_extend_without_speculative():
        return AttnForwardMethod.MHA
    else:
        return AttnForwardMethod.MLA


def handle_attention_nsa(attn, forward_batch):
    """
    Dispatch logic is centralized in NativeSparseAttnBackend.set_nsa_prefill_impl and executed
    in init_forward_metadata. Read the decision from backend.use_mha.
    """

    backend = forward_batch.attn_backend
    if isinstance(backend, TboAttnBackend):  # if enable tbo, get primary backend
        backend = backend.primary
    if hasattr(backend, "use_mha") and backend.use_mha:
        return AttnForwardMethod.MHA_ONE_SHOT
    return AttnForwardMethod.MLA


def handle_attention_triton(attn, forward_batch):
    if is_in_piecewise_cuda_graph():
        return AttnForwardMethod.MLA

    # when deterministic inference is enabled, use MLA
    if get_global_server_args().enable_deterministic_inference:
        return _dispatch_mla_subtype(attn, forward_batch)

    if (
        forward_batch.forward_mode.is_extend_without_speculative()
        and sum(forward_batch.extend_prefix_lens_cpu) == 0
    ):
        return AttnForwardMethod.MHA
    else:
        return _dispatch_mla_subtype(attn, forward_batch)


class DeepseekV2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        swiglu_limit: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size
        self.swiglu_limit = swiglu_limit

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if not hasattr(self.gate_up_proj, "weight") and hasattr(
            self.gate_up_proj, "weight_packed"
        ):
            self.gate_up_proj.weight = self.gate_up_proj.weight_packed
        if not hasattr(self.down_proj, "weight") and hasattr(
            self.down_proj, "weight_packed"
        ):
            self.down_proj.weight = self.down_proj.weight_packed
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        forward_batch=None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: BumpAllocator = None,
        rms_weight: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        update_hd: Optional[bool] = False,
    ):
        if (self.tp_size == 1) and x.shape[0] == 0:
            if _use_fused_rms_quant and rms_weight is not None and residual is not None:
                return x, None, None, None
            return x

        if (
            gemm_output_zero_allocator is not None
            and x.shape[0] <= 256
            and self.gate_up_proj.weight.dtype == torch.uint8
        ):
            y = gemm_output_zero_allocator.allocate(
                x.shape[0] * self.gate_up_proj.output_size_per_partition
            ).view(x.shape[0], self.gate_up_proj.output_size_per_partition)
            x = (x, None, y)

        gate_up, _ = self.gate_up_proj(x)
        # Fast path: fused silu+clamp+fp8_quant+deepgemm when conditions met.
        # Only valid when down_proj does NOT need an all-reduce and its weights
        # are fp8 (uint8 storage with weight_scale_inv).
        if (
            self.swiglu_limit is not None
            and not self.down_proj.reduce_results
            and self.down_proj.weight.dtype == torch.uint8
            and hasattr(self.down_proj, "weight_scale_inv")
        ):
            M, N = gate_up.shape
            down_input_fp8 = gate_up.new_empty((M, N // 2), dtype=torch.float8_e4m3fn)
            scale_block_size = 128
            down_input_scale = create_per_token_group_quant_fp8_output_scale(
                x_shape=(M, N // 2),
                device=gate_up.device,
                group_size=scale_block_size,
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )
            silu_and_mul_contig_post_quant(
                input=gate_up,
                output=down_input_fp8,
                output_scale=down_input_scale,
                quant_group_size=scale_block_size,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                transposed=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                swiglu_limit=float(self.swiglu_limit),
            )
            down_output = gate_up.new_empty(
                (M, self.down_proj.output_size), dtype=torch.bfloat16
            )
            deep_gemm_wrapper.gemm_nt_f8f8bf16(
                (down_input_fp8, down_input_scale),
                (self.down_proj.weight, self.down_proj.weight_scale_inv),
                down_output,
            )
            return down_output
        # Fallback: fused silu+clamp kernel (still faster than unfused)
        if self.swiglu_limit is not None:
            M, N = gate_up.shape
            x = gate_up.new_empty((M, N // 2))
            silu_and_mul_clamp(gate_up, x, float(self.swiglu_limit))
        else:
            x = self.act_fn(gate_up)
        x, _ = self.down_proj(
            x,
            skip_all_reduce=should_allreduce_fusion or use_reduce_scatter,
        )
        return x


class MoEGate(nn.Module):
    def __init__(
        self,
        config,
        quant_config,
        prefix: str = "",
        is_nextn: bool = False,
        is_hash_moe: bool = False,
        is_deepseek_v4: bool = False,
    ):
        super().__init__()
        self.is_nextn = is_nextn
        self.is_deepseek_v4 = is_deepseek_v4
        self.weight = nn.Parameter(
            torch.empty((config.n_routed_experts, config.hidden_size))
        )

        if config.topk_method == "noaux_tc" and not is_hash_moe:
            correction_bias_dtype = torch.float32
            if quant_config is not None:
                if (
                    quant_config.get_name() == "modelopt_fp4"
                    and get_moe_runner_backend().is_flashinfer_trtllm()
                ):
                    correction_bias_dtype = torch.bfloat16
                elif _use_aiter and quant_config.get_name() in (
                    "fp8",
                    "compressed_tensors",
                    "quark",
                ):
                    correction_bias_dtype = torch.bfloat16
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((config.n_routed_experts), dtype=correction_bias_dtype)
            )
        else:
            self.e_score_correction_bias = None
        if _is_cpu and _is_cpu_amx_available:
            self.quant_method = PackWeightMethod(weight_names=["weight"])
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()

    def forward(
        self,
        hidden_states,
        gemm_output_zero_allocator: BumpAllocator = None,
        forward_batch: ForwardBatch = None,
    ):
        if use_intel_amx_backend(self):
            return torch.ops.sgl_kernel.weight_packed_linear(
                hidden_states,
                self.weight,
                None,  # bias
                True,  # is_vnni
            )

        if get_global_server_args().enable_deterministic_inference:
            return F.linear(hidden_states, self.weight, None)

        if (
            not self.is_deepseek_v4
            and forward_batch is not None
            and nsa_use_prefill_cp(forward_batch)
        ):
            logits = F.linear(hidden_states, self.weight, None)
        else:
            # NOTE: For some unknown reason, router_gemm seems degrade accept length.
            if (
                _is_cuda
                and hidden_states.shape[0] <= 16
                and hidden_states.shape[1] == 7168
                and (self.weight.shape[0] == 256 or self.weight.shape[0] == 384)
                and _device_sm >= 90
            ):
                if _device_sm in [100, 103] and self.weight.shape[0] == 256:
                    # router gemm output float32
                    logits = torch.empty(
                        hidden_states.shape[0],
                        self.weight.shape[0],
                        device=hidden_states.device,
                        dtype=torch.float32,
                    )
                    flashinfer_dsv3_router_gemm(logits, hidden_states, self.weight)
                else:
                    logits = dsv3_router_gemm(
                        hidden_states, self.weight, out_dtype=torch.float32
                    )

            elif _use_aiter:
                logits = aiter_dsv3_router_gemm(hidden_states, self.weight)
            else:
                if self.is_deepseek_v4:
                    from sglang.jit_kernel.deepseek_v4 import linear_bf16_fp32

                    logits = linear_bf16_fp32(hidden_states, self.weight)
                else:
                    # After testing, we may use the faster code in `if deepseek v4` branch
                    logits = F.linear(hidden_states, self.weight, None)

        return logits


class DeepseekV2MoE(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
        is_deepseek_v4: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.moe_ep_size = get_moe_expert_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts

        n_shared_experts = (
            0 if config.n_shared_experts is None else int(config.n_shared_experts)
        )
        _fusion_disabled = get_global_server_args().disable_shared_experts_fusion

        # num_fused_shared_experts drives weight remapping in deepseek_weight_loader:
        # mlp.shared_experts → mlp.experts.256 when > 0.
        self.num_fused_shared_experts = 0 if _fusion_disabled else n_shared_experts

        # DeepEP shared expert fusion: shared expert is fused into the same MoE kernel
        # as a local expert at the home EP rank. Expert layout is expanded from 256
        # routed to 256+EP_size (e.g. 272 for EP=16). TopK handles interleaving.
        _is_deepep_fusion = (
            is_deepep_class_backend() and self.num_fused_shared_experts > 0
        )

        if _is_deepep_fusion:
            # 256 routed + EP_size shared slots = 272 experts total (for EP=16)
            num_experts_for_moe = config.n_routed_experts + self.moe_ep_size
            top_k_for_moe = config.num_experts_per_tok + 1  # 8 routed + 1 shared
            # Interleaving for DeepEP dispatch is handled by TopK internally.
        else:
            num_experts_for_moe = (
                config.n_routed_experts + self.num_fused_shared_experts
            )
            top_k_for_moe = config.num_experts_per_tok + self.num_fused_shared_experts

        self.config = config
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.is_nextn = is_nextn

        n_hash_layers = getattr(config, "num_hash_layers", 0)
        self.is_hash = layer_id < n_hash_layers and not (is_deepseek_v4 and is_nextn)

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.gate = MoEGate(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("gate", prefix),
            is_nextn=is_nextn,
            is_hash_moe=self.is_hash,
            is_deepseek_v4=is_deepseek_v4,
        )

        # scaling factor for fused shared experts on AMD-platform.
        # DeepEP doesn't need this: shared expert is only computed on home rank
        # (not all-reduced), so no 1/ep_size correction is needed.
        fused_shared_experts_scaling_factor = None
        if (
            self.moe_ep_size > 1
            and self.num_fused_shared_experts > 0
            and not _is_deepep_fusion
        ):
            # if enable_ep_moe tp_szie == ep_size, every gpu get shared experts gemm output
            # so we scale with 1 / self.moe_ep_size in ep mode which will make it equalation as in tp mode
            # with fused_shared_experts
            fused_shared_experts_scaling_factor = 1.0 / float(self.moe_ep_size)

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=num_experts_for_moe
            + get_global_server_args().ep_num_redundant_experts,
            num_fused_shared_experts=self.num_fused_shared_experts,
            top_k=top_k_for_moe,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=self.layer_id,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            routing_method_type=getattr(
                config, "routing_method_type", RoutingMethodType.DeepSeekV3
            ),
            swiglu_limit=getattr(config, "swiglu_limit", None),
            prefix=add_prefix("experts", prefix),
        )

        self.num_expert_group = getattr(config, "n_group", None)
        self.topk_group = getattr(config, "topk_group", None)
        self.use_grouped_topk = (
            self.num_expert_group is not None
            and self.topk_group is not None
            and self.num_expert_group > self.topk_group
        )

        if self.is_hash and not (is_nextn and is_deepseek_v4):
            self.topk = HashTopK(
                topk=config.num_experts_per_tok + self.num_fused_shared_experts,
                num_experts=config.n_routed_experts,
                num_fused_shared_experts=self.num_fused_shared_experts,
                vocab_size=config.vocab_size,
                scoring_func=config.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
            )
        else:
            # Default: grouped noaux_tc top-k. Covers V3/V3.2/GLM-5/Glm4MoeLite.
            topk_kwargs = dict(
                top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
                layer_id=self.layer_id,
                renormalize=config.norm_topk_prob,
                use_grouped_topk=True,
                num_expert_group=config.n_group,
                num_fused_shared_experts=self.num_fused_shared_experts,
                topk_group=config.topk_group,
                correction_bias=self.gate.e_score_correction_bias,
                quant_config=quant_config,
                routed_scaling_factor=self.routed_scaling_factor,
                apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
                fused_shared_experts_scaling_factor=fused_shared_experts_scaling_factor,
                # Some Fp4 MoE backends require the output format to be bypassed but the MTP layers are unquantized
                # and requires the output format to be standard (except trtllm). We use quant_config to determine the output format.
                output_format=(
                    TopKOutputFormat.STANDARD
                    if (quant_config is None)
                    and (not get_moe_runner_backend().is_flashinfer_trtllm())
                    else None
                ),
            )
            # DSV4 override: ungrouped sqrtsoftplus + fp4 expert layout flag.
            if is_deepseek_v4:
                topk_kwargs.update(
                    use_grouped_topk=False,
                    scoring_func=config.scoring_func,
                    is_fp4_experts=getattr(quant_config, "is_fp4_experts", False),
                )
            self.topk = TopK(**topk_kwargs)

        self.shared_experts_is_int8 = False
        self.shared_experts_is_fp8 = False
        self.shared_experts_weight_block_size = None
        self._shared_expert_tp1 = False
        # Shared experts: skip when fused into MoE kernel (self.num_fused_shared_experts > 0)
        # or when DeepEP fusion is enabled (shared expert is local slot 16 in FusedMoE, no separate MLP).
        if (
            config.n_shared_experts is not None
            and config.n_shared_experts > 0
            and self.num_fused_shared_experts == 0
            and not _is_deepep_fusion
        ):
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            # Disable TP for shared experts for A2A/FP4 allgather paths, or when
            # explicitly requested for DSV4 checkpoints whose shared scales are
            # not divisible by the global TP size.
            _shared_expert_use_tp1 = (
                get_moe_a2a_backend().is_deepep()
                or get_moe_a2a_backend().is_mooncake()
                or get_moe_a2a_backend().is_nixl()
                or get_moe_a2a_backend().is_mori()
                or get_moe_a2a_backend().is_ascend_fuseep()
                or get_moe_a2a_backend().is_flashinfer()
                or get_moe_a2a_backend().is_megamoe()
                or should_use_flashinfer_cutlass_moe_fp4_allgather()
                or envs.SGLANG_SHARED_EXPERT_TP1.get()
            )
            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                swiglu_limit=getattr(config, "swiglu_limit", None),
                prefix=add_prefix("shared_experts", prefix),
                **(dict(tp_rank=0, tp_size=1) if _shared_expert_use_tp1 else {}),
            )
            self._shared_expert_tp1 = _shared_expert_use_tp1
            is_packed_weight = hasattr(
                self.shared_experts.gate_up_proj.quant_method, "quant_config"
            ) and self.shared_experts.gate_up_proj.quant_method.quant_config.get_name() in {
                "awq",
                "awq_marlin",
                "moe_wna16",
            }
            self.shared_experts_is_int8 = (
                not is_packed_weight
                and self.shared_experts.gate_up_proj.weight.dtype == torch.int8
            )
            self.shared_experts_is_fp8 = (
                not is_packed_weight
                and self.shared_experts.gate_up_proj.weight.dtype == torch.float8_e4m3fn
            )
            # if self.shared_experts_is_fp8:
            #     if (
            #         _use_aiter
            #         and config.quantization_config.get("quant_method")
            #         == "compressed-tensors"
            #     ):
            #         # For compressed-tensors ptpc model, don't need to check the weight_block_size
            #         pass
            #     else:
            #         assert (
            #             self.shared_experts.gate_up_proj.quant_method.quant_config.weight_block_size
            #             == self.shared_experts.down_proj.quant_method.quant_config.weight_block_size
            #         )
            #         self.shared_experts_weight_block_size = (
            #             self.shared_experts.gate_up_proj.quant_method.quant_config.weight_block_size
            #         )

        self.top_k = config.num_experts_per_tok

        if (
            get_moe_a2a_backend().is_deepep()
            or get_moe_a2a_backend().is_mooncake()
            or get_moe_a2a_backend().is_nixl()
            or get_moe_a2a_backend().is_mori()
            or get_moe_a2a_backend().is_ascend_fuseep()
        ):
            # TODO: we will support tp < ep in the future
            self.ep_size = get_moe_expert_parallel_world_size()
            self.num_experts = (
                config.n_routed_experts
                + get_global_server_args().ep_num_redundant_experts
            )
            self.renormalize = config.norm_topk_prob
            self.correction_bias = (
                self.gate.e_score_correction_bias.data
                if self.gate.e_score_correction_bias is not None
                else None
            )

        self._enable_a2a_moe = (
            get_moe_a2a_backend().is_deepep()
            or get_moe_a2a_backend().is_mooncake()
            or get_moe_a2a_backend().is_nixl()
            or get_moe_a2a_backend().is_mori()
            or get_moe_a2a_backend().is_ascend_fuseep()
            or get_moe_a2a_backend().is_flashinfer()
        )
        self._fuse_shared_experts_inside_sbo = SboFlags.fuse_shared_experts_inside_sbo()

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
            and filter_moe_weight_param_global_expert(
                name, x, self.experts.num_local_experts
            )
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: BumpAllocator = None,
        input_ids: Optional[torch.Tensor] = None,
        input_ids_global: Optional[torch.Tensor] = None,
        rms_weight: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        update_hd: Optional[bool] = False,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.mega_moe import forward_mega_moe, should_use_mega_moe

        if should_use_mega_moe(self, hidden_states):
            return forward_mega_moe(
                self,
                hidden_states,
                forward_batch,
                input_ids_global=input_ids_global,
            )

        if not self._enable_a2a_moe:
            if (
                self.alt_stream is not None
                and self.num_fused_shared_experts == 0
                and hidden_states.shape[0] > 0
                and get_is_capture_mode()
                and not (
                    get_global_server_args().enable_torch_compile
                    and hidden_states.shape[0]
                    <= get_global_server_args().torch_compile_max_bs
                    * (get_global_server_args().speculative_num_draft_tokens or 1)
                )
            ):
                return self.forward_normal_dual_stream(
                    hidden_states,
                    should_allreduce_fusion,
                    use_reduce_scatter,
                    gemm_output_zero_allocator,
                    input_ids,
                    input_ids_global=input_ids_global,
                    rms_weight = rms_weight,
                    residual = residual,
                )
            else:
                return self.forward_normal(
                    hidden_states,
                    should_allreduce_fusion,
                    use_reduce_scatter,
                    gemm_output_zero_allocator,
                    input_ids,
                    input_ids_global=input_ids_global,
                    rms_weight = rms_weight,
                    residual = residual,
                )
        else:
            return self.forward_deepep(hidden_states, forward_batch, rms_weight=rms_weight, residual=residual,)

    def forward_normal_dual_stream(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: BumpAllocator = None,
        input_ids: Optional[torch.Tensor] = None,
        input_ids_global: Optional[torch.Tensor] = None,
        rms_weight: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        # shared_output = self._forward_shared_experts(
        #     hidden_states, gemm_output_zero_allocator
        # )
        i_q = None
        i_s = None
        if _use_fused_rms_quant:
            shared_output, new_resi, i_q, i_s = self._forward_shared_experts(
                hidden_states, gemm_output_zero_allocator,
                rms_weight = rms_weight,
                residual = residual,
            )
        else:
            shared_output = self._forward_shared_experts(
                hidden_states, gemm_output_zero_allocator,
            )
        server_args = get_global_server_args()
        dispatch_info = (
            ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id)
            if server_args.enable_eplb
            else None
        )
        with torch.cuda.stream(self.alt_stream):
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states, gemm_output_zero_allocator)
            topk_kwargs = (
                {"input_ids": input_ids_global}
                if getattr(self, "is_hash", False)
                else {}
            )
            topk_output = self.topk(
                hidden_states,
                router_logits,
                expert_location_dispatch_info=dispatch_info,
                **topk_kwargs,
            )
            final_hidden_states = self.experts(hidden_states, topk_output)
            if not (_is_cuda or _is_musa) or isinstance(
                self.experts.quant_method, KTEPWrapperMethod
            ):
                final_hidden_states *= self.routed_scaling_factor

        current_stream.wait_stream(self.alt_stream)

        final_hidden_states = maybe_fuse_routed_scale_and_shared_add(
            self.experts,
            final_hidden_states,
            None if self._shared_expert_tp1 else shared_output,
            self.routed_scaling_factor,
        )

        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True,
            use_reduce_scatter=use_reduce_scatter,
            should_allreduce_fusion=should_allreduce_fusion,
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        # TP1 shared experts are replicated, so add them after all-reduce to
        # avoid summing the same shared output once per TP rank.
        if self._shared_expert_tp1:
            final_hidden_states += shared_output
        return final_hidden_states

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: BumpAllocator = None,
        input_ids: Optional[torch.Tensor] = None,
        input_ids_global: Optional[torch.Tensor] = None,
        rms_weight: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hasattr(self, "shared_experts") and use_intel_amx_backend(
            self.shared_experts.gate_up_proj
        ):
            return self.forward_cpu(hidden_states, should_allreduce_fusion)
        server_args = get_global_server_args()
        dispatch_info = (
            ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id)
            if server_args.enable_eplb
            else None
        )
        defer_shared = not self.experts.moe_runner_config.inplace
        if hidden_states.shape[0] > 0:
            if not defer_shared and not self._fuse_shared_experts_inside_sbo:
                shared_output = self._forward_shared_experts(
                    hidden_states, gemm_output_zero_allocator,
                )
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states, gemm_output_zero_allocator)
            topk_kwargs = (
                {"input_ids": input_ids_global}
                if getattr(self, "is_hash", False)
                else {}
            )
            topk_output = self.topk(
                hidden_states,
                router_logits,
                expert_location_dispatch_info=dispatch_info,
                **topk_kwargs,
            )
        else:
            shared_output = None
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        if _use_lightop_moe_sum_mul_add:
            if _use_fused_rms_quant and i_q is not None and i_s is not None:  # maybe needn't
                final_hidden_states = self.experts(hidden_states, topk_output, shared_output=shared_output, i_q=i_q, i_s=i_s)
            else:
                final_hidden_states = self.experts(hidden_states, topk_output, shared_output=shared_output)
        else:
            if self._fuse_shared_experts_inside_sbo:
                shared_output = None

                def _pre_combine_hook(
                    dispatcher: BaseDispatcher, combine_input: CombineInput
                ):

                    nonlocal shared_output
                    self.alt_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(self.alt_stream):
                        shared_output = self._forward_shared_experts(
                            hidden_states, gemm_output_zero_allocator
                        )

                    pre_combine_hook_handle.remove()

                def _post_combine_hook(
                    dispatcher: BaseDispatcher, hidden_states: torch.Tensor
                ):
                    nonlocal shared_output
                    torch.cuda.current_stream().wait_stream(self.alt_stream)
                    post_combine_hook_handle.remove()

                pre_combine_hook_handle = self.experts.dispatcher.register_pre_combine_hook(
                    _pre_combine_hook
                )
                post_combine_hook_handle = (
                    self.experts.dispatcher.register_post_combine_hook(_post_combine_hook)
                )
            final_hidden_states = self.experts(
            hidden_states,
            topk_output,
        )
        if (
            not _is_cuda
            and not _is_musa
            and not _is_dcu
            and not _is_xpu
            and not _use_aiter
            or isinstance(self.experts.quant_method, KTEPWrapperMethod)
        ):
            # fused in biased_grouped_topk so we can skip here
            final_hidden_states *= self.routed_scaling_factor

        if (
            defer_shared
            and hidden_states.shape[0] > 0
            and not self._fuse_shared_experts_inside_sbo
        ):
            shared_output = self._forward_shared_experts(
                hidden_states, gemm_output_zero_allocator
            )

        final_hidden_states = maybe_fuse_routed_scale_and_shared_add(
            self.experts,
            final_hidden_states,
            None if self._shared_expert_tp1 else shared_output,
            self.routed_scaling_factor,
        )

        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True,
            use_reduce_scatter=use_reduce_scatter,
            should_allreduce_fusion=should_allreduce_fusion,
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        # TP1 shared experts are replicated, so add them after all-reduce to
        # avoid summing the same shared output once per TP rank.
        if shared_output is not None and self._shared_expert_tp1:
            final_hidden_states += shared_output
        return final_hidden_states

    def forward_cpu(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
    ) -> torch.Tensor:
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        fused_experts_out = self.experts(
            hidden_states=hidden_states, topk_output=topk_output
        )

        assert use_intel_amx_backend(
            self.shared_experts.gate_up_proj
        ) == use_intel_amx_backend(self.shared_experts.down_proj)
        # [Note] inplace should be False in fused_experts.
        # If inplace is True in fused_experts (self.experts), hidden_states will be changed after fused_experts
        # While hidden_states is still needed in shared_expert.
        final_hidden_states = torch.ops.sgl_kernel.shared_expert_cpu(
            hidden_states,
            self.shared_experts.gate_up_proj.weight,
            self.shared_experts.down_proj.weight,
            fused_experts_out,
            self.routed_scaling_factor,
            True,  # inplace
            self.shared_experts_is_int8,  # use_int8_w8a8
            self.shared_experts_is_fp8,  # use_fp8_w8a16
            (
                self.shared_experts.gate_up_proj.weight_scale
                if self.shared_experts_is_int8
                else (
                    self.shared_experts.gate_up_proj.weight_scale_inv
                    if self.shared_experts_is_fp8
                    else None
                )
            ),  # w1_scale
            (
                self.shared_experts.down_proj.weight_scale
                if self.shared_experts_is_int8
                else (
                    self.shared_experts.down_proj.weight_scale_inv
                    if self.shared_experts_is_fp8
                    else None
                )
            ),  # w2_scale
            (
                self.shared_experts_weight_block_size
                if self.shared_experts_is_fp8
                else None
            ),  # block_size
            True,  # is_vnni
        )
        if self.tp_size > 1 and not should_allreduce_fusion:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states

    def forward_deepep(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        input_ids_global: Optional[torch.Tensor] = None,
        rms_weight: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shared_output = None
        sbo_enabled_flag = self._fuse_shared_experts_inside_sbo and not self.is_nextn
        sbo_overlap_dispatch_flag = (
            sbo_enabled_flag and SboFlags.enable_dispatch_shared_one_stream_overlap()
        )
        sbo_overlap_combine_flag = (
            sbo_enabled_flag and SboFlags.enable_combine_shared_two_stream_overlap()
        )

        if hidden_states.shape[0] > 0:
            i_q = None
            i_s = None
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states, forward_batch=forward_batch)
            router_logits = self.gate(hidden_states, forward_batch=forward_batch)
            if not sbo_enabled_flag and self.num_fused_shared_experts == 0:
                if self.alt_stream is not None:
                    self.alt_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(self.alt_stream):
                        if _use_fused_rms_quant:
                            shared_output, new_resi, i_q, i_s = self._forward_shared_experts(hidden_states, rms_weight=rms_weight, residual=residual)
                        else:
                            shared_output = self._forward_shared_experts(hidden_states)
                        shared_output.record_stream(self.alt_stream)
                        shared_event = self.alt_stream.record_event()
                else:
                    if _use_fused_rms_quant:
                        shared_output, new_resi, i_q, i_s = self._forward_shared_experts(hidden_states, rms_weight=rms_weight, residual=residual)
                    else:
                        shared_output = self._forward_shared_experts(hidden_states)
            topk_kwargs = (
                {"input_ids": input_ids_global}
                if getattr(self, "is_hash", False)
                else {}
            )
            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
                **topk_kwargs,
            )
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        if sbo_overlap_dispatch_flag:
            shared_output = None

            def _deepep_dispatch_hook(dispatcher: BaseDispatcher):
                nonlocal shared_output
                shared_output = self._forward_shared_experts(hidden_states)
                for handle in deepep_dispatch_hook_handle:
                    handle.remove()

            def _post_dispatch_hook(
                dispatcher: BaseDispatcher, dispatch_output: DispatchOutput
            ):
                combine_overlap_args, down_gemm_overlap_args, meta_overlap_args = (
                    compute_overlap_args(dispatch_output, self.alt_stream)
                )
                dispatcher.set_overlap_args(
                    combine_overlap_args=combine_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )
                self.experts.set_overlap_args(
                    down_gemm_overlap_args=down_gemm_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )
                post_dispatch_hook_handle.remove()

            def _post_combine_hook(
                dispatcher: BaseDispatcher, hidden_states: torch.Tensor
            ):
                dispatcher.clear_overlap_args()
                self.experts.clear_overlap_args()
                post_combine_hook_handle.remove()

            assert isinstance(self.experts.dispatcher, MaybeTboDeepEPDispatcher)
            deepep_dispatch_hook_handle = (
                self.experts.dispatcher.register_deepep_dispatch_hook(
                    _deepep_dispatch_hook
                )
            )
            post_dispatch_hook_handle = (
                self.experts.dispatcher.register_post_dispatch_hook(_post_dispatch_hook)
            )
            post_combine_hook_handle = (
                self.experts.dispatcher.register_post_combine_hook(_post_combine_hook)
            )

        elif sbo_overlap_combine_flag:
            shared_output = None

            def _post_dispatch_hook(
                dispatcher: BaseDispatcher, dispatch_output: DispatchOutput
            ):

                combine_overlap_args, down_gemm_overlap_args, meta_overlap_args = (
                    compute_overlap_args(dispatch_output, self.alt_stream)
                )
                dispatcher.set_overlap_args(
                    combine_overlap_args=combine_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )
                self.experts.set_overlap_args(
                    down_gemm_overlap_args=down_gemm_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )

                post_dispatch_hook_handle.remove()

            # NOTE: sbo not compatable with rms_quant
            def _pre_combine_hook(
                dispatcher: BaseDispatcher, combine_input: CombineInput
            ):

                nonlocal shared_output

                if (
                    e := dispatcher.meta_overlap_args.get("record_event_after_down")
                ) is not None:
                    e.record()

                # TODO reduce sm for non-deepgemm
                # with deep_gemm_wrapper.configure_deep_gemm_num_sms(
                #     dispatcher.meta_overlap_args["compute_num_sms"]
                # ):
                shared_output = self._forward_shared_experts(hidden_states)

                pre_combine_hook_handle.remove()

            def _post_combine_hook(
                dispatcher: BaseDispatcher, hidden_states: torch.Tensor
            ):
                dispatcher.clear_overlap_args()
                self.experts.clear_overlap_args()
                post_combine_hook_handle.remove()

            post_dispatch_hook_handle = (
                self.experts.dispatcher.register_post_dispatch_hook(_post_dispatch_hook)
            )
            pre_combine_hook_handle = self.experts.dispatcher.register_pre_combine_hook(
                _pre_combine_hook
            )
            post_combine_hook_handle = (
                self.experts.dispatcher.register_post_combine_hook(_post_combine_hook)
            )
        elif envs.SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO.get():
            # On GB200: Shared experts overlapped on alt_stream, down gemm overlapped with DeepEP Combine

            def _post_dispatch_hook(
                dispatcher: BaseDispatcher, dispatch_output: DispatchOutput
            ):

                combine_overlap_args, down_gemm_overlap_args, meta_overlap_args = (
                    compute_overlap_args(dispatch_output, self.alt_stream)
                )
                dispatcher.set_overlap_args(
                    combine_overlap_args=combine_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )
                self.experts.set_overlap_args(
                    down_gemm_overlap_args=down_gemm_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )

                post_dispatch_hook_handle.remove()

            def _pre_combine_hook(
                dispatcher: BaseDispatcher, combine_input: CombineInput
            ):
                if (
                    e := dispatcher.meta_overlap_args.get("record_event_after_down")
                ) is not None:
                    e.record()
                pre_combine_hook_handle.remove()

            def _post_combine_hook(
                dispatcher: BaseDispatcher, hidden_states: torch.Tensor
            ):
                dispatcher.clear_overlap_args()
                self.experts.clear_overlap_args()
                post_combine_hook_handle.remove()

            post_dispatch_hook_handle = (
                self.experts.dispatcher.register_post_dispatch_hook(_post_dispatch_hook)
            )
            pre_combine_hook_handle = self.experts.dispatcher.register_pre_combine_hook(
                _pre_combine_hook
            )
            post_combine_hook_handle = (
                self.experts.dispatcher.register_post_combine_hook(_post_combine_hook)
            )

        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

        if (
            hidden_states.shape[0] > 0
            and not sbo_enabled_flag
            and self.num_fused_shared_experts == 0
            and self.alt_stream is not None
        ):
            torch.cuda.current_stream().wait_event(shared_event)

        if shared_output is not None:
            x = shared_output
            # aiter moe call will handle routed_scaling_factor in the function
            # so add _use_aiter condition to eliminate to use self.routed_scaling_factor in add_ call
            if self.experts.should_fuse_routed_scaling_factor_in_topk or _use_aiter:
                x.add_(final_hidden_states)
            else:
                x.add_(final_hidden_states, alpha=self.routed_scaling_factor)
            final_hidden_states = x
        else:
            if not (
                self.experts.should_fuse_routed_scaling_factor_in_topk or _use_aiter
            ):
                final_hidden_states *= self.routed_scaling_factor

        return final_hidden_states

    def _forward_shared_experts(
        self,
        hidden_states,
        gemm_output_zero_allocator: BumpAllocator = None,
        rms_weight: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        update_hd: Optional[bool] = True,
    ):
        if (hidden_states.shape[0] > 0) and (self.num_fused_shared_experts == 0):
            return self.shared_experts(
                hidden_states,
                gemm_output_zero_allocator=gemm_output_zero_allocator,
                rms_weight = rms_weight,
                residual = residual,
                update_hd = update_hd,
            )
        else:
            if _use_fused_rms_quant and rms_weight is not None and residual is not None:
                return None, None, None, None
            return None

    def op_gate(self, state):
        if is_non_idle_and_non_empty(
            state.forward_batch.forward_mode, state.hidden_states_mlp_input
        ):
            # router_logits: (num_tokens, n_experts)
            state.router_logits = self.gate(state.hidden_states_mlp_input)
        else:
            state.router_logits = None

    def op_shared_experts(self, state):
        hidden_states_mlp_input = state.pop("hidden_states_mlp_input")
        if (self.num_fused_shared_experts == 0) and is_non_idle_and_non_empty(
            state.forward_batch.forward_mode, hidden_states_mlp_input
        ):
            state.shared_output = self.shared_experts(hidden_states_mlp_input)
        else:
            state.shared_output = None

    def op_select_experts(self, state):
        router_logits = state.pop("router_logits")
        hidden_states = state.hidden_states_mlp_input
        if router_logits is not None:
            with get_global_expert_distribution_recorder().with_current_layer(
                self.layer_id
            ):
                state.topk_output = self.topk(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    num_token_non_padded=state.forward_batch.num_token_non_padded,
                    expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                        layer_id=self.layer_id,
                    ),
                )
        else:
            state.topk_output = self.topk.empty_topk_output(hidden_states.device)

    def op_dispatch_a(self, state):
        if self.ep_size > 1:
            self.experts.dispatcher.dispatch_a(
                hidden_states=state.hidden_states_mlp_input,
                topk_output=state.pop("topk_output"),
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_dispatch_b(self, state):
        if self.ep_size > 1:
            with get_global_expert_distribution_recorder().with_current_layer(
                self.layer_id
            ):
                state.dispatch_output = self.experts.dispatcher.dispatch_b(
                    tbo_subbatch_index=state.get("tbo_subbatch_index"),
                )

    def op_experts(self, state):
        state.combine_input = self.experts.run_moe_core(
            dispatch_output=state.dispatch_output,
        )

    def op_combine_a(self, state):
        if self.ep_size > 1:
            self.experts.dispatcher.combine_a(
                combine_input=state.pop("combine_input"),
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )
            state.pop("dispatch_output")

    def op_combine_b(self, state):
        if self.ep_size > 1:
            state.hidden_states_after_combine = self.experts.dispatcher.combine_b(
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_output(self, state):
        final_hidden_states = state.pop("hidden_states_after_combine")

        if get_moe_a2a_backend().is_mori():
            num_tokens = state.pop("num_tokens")
            final_hidden_states = final_hidden_states[:num_tokens]

        if (shared_output := state.pop("shared_output")) is not None:
            x = shared_output
            if _use_aiter:
                x.add_(final_hidden_states)
            else:
                x.add_(final_hidden_states, alpha=self.routed_scaling_factor)
            final_hidden_states = x
        elif _use_aiter:
            # fused in aiter_biased_grouped_topk so we can skip here
            pass
        else:
            final_hidden_states *= self.routed_scaling_factor

        state.hidden_states_mlp_output = final_hidden_states


class DeepseekV2AttentionMLA(
    nn.Module,
    DeepseekMHAForwardMixin,
    DeepseekMLAForwardMixin,
    DeepseekMLARocmForwardMixin,
    DeepseekMLACpuForwardMixin,
):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        layer_id: int = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        skip_rope: bool = False,
        is_nextn: bool = False,
        input_layernorm: Optional[torch.nn.Module] = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.quant_config = quant_config
        self.input_layernorm = input_layernorm  # this is from other place
        # self.input_layernorm_weight = input_layernorm_weight
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        self.use_nsa = is_deepseek_nsa(config)
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            assert self.use_nsa, "CP currently only supports deepseek v3.2 model"
        # cp reuse the attn_tp comm group but need to duplicate the weights
        if self.nsa_enable_prefill_cp and self.use_nsa:
            self.cp_size = get_attention_cp_size()
        self.num_heads = num_heads
        assert num_heads % attn_tp_size == 0
        self.num_local_heads = num_heads // attn_tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.kv_cache_dtype = get_global_server_args().kv_cache_dtype

        # NOTE modification to rope_scaling must be done early enough, b/c e.g. Indexer needs it
        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"

        # For tensor parallel attention
        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("fused_qkv_a_proj_with_mqa", prefix),
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=self._get_q_b_proj_quant_config(quant_config),
                eps=config.rms_norm_eps,
                prefix=add_prefix("q_b_proj", prefix),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_proj", prefix),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("kv_a_proj_with_mqa", prefix),
            )

        self.skip_topk = None
        self.next_skip_topk = None
        if self.use_nsa:
            is_neox_style = not getattr(config, "indexer_rope_interleave", False)
            self.indexer = Indexer(
                hidden_size=hidden_size,
                index_n_heads=get_nsa_index_n_heads(config),
                index_head_dim=get_nsa_index_head_dim(config),
                rope_head_dim=qk_rope_head_dim,
                index_topk=get_nsa_index_topk(config),
                q_lora_rank=q_lora_rank,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                scale_fmt="ue8m0",
                block_size=128,
                rope_scaling=rope_scaling,
                is_neox_style=is_neox_style,
                prefix=add_prefix("indexer", prefix),
                quant_config=quant_config,
                layer_id=layer_id,
                alt_stream=alt_stream,
            )
            # Refer: https://arxiv.org/abs/2603.12201 for more details.
            # skip_topk: when True, this layer will skip computation and reuse previous layer's topk indices.
            # next_skip_topk: when True, the next layer will skip computation and reuse this layer's topk indices.
            if is_nextn:
                self.skip_topk = False
                self.next_skip_topk = False
            else:
                self.index_topk_freq = getattr(config, "index_topk_freq", 1)
                self.index_topk_pattern = getattr(config, "index_topk_pattern", None)
                if self.index_topk_pattern is None:
                    self.skip_topk = max(layer_id - 1, 0) % self.index_topk_freq != 0
                    self.next_skip_topk = layer_id % self.index_topk_freq != 0
                else:
                    self.skip_topk = self.index_topk_pattern[layer_id] == "S"
                    if layer_id < len(self.index_topk_pattern) - 1:
                        self.next_skip_topk = (
                            self.index_topk_pattern[layer_id + 1] == "S"
                        )
                    else:
                        self.next_skip_topk = False

        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_b_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        # O projection.
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)

        if not skip_rope:
            is_neox_style = not getattr(config, "rope_interleave", True)
            self.rotary_emb = get_rope_wrapper(
                qk_rope_head_dim,
                rotary_dim=qk_rope_head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
                is_neox_style=is_neox_style,
                device=get_global_server_args().device,
            )

            if rope_scaling and rope_scaling.get("apply_yarn_scaling", True):
                self.scaling = compute_mla_mscale_scaling(rope_scaling, self.scaling)
        else:
            self.rotary_emb = None
        self.use_deepseek_yarn_rope = rope_scaling is not None

        self.attn_mqa = RadixAttention(
            self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=self.kv_lora_rank,
            quant_config=quant_config,
            prefix=add_prefix("attn_mqa", prefix),
        )

        self.attn_mha = RadixAttention(
            self.num_local_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
            quant_config=quant_config,
            prefix=add_prefix("attn_mha", prefix),
        )

        self.alt_stream = alt_stream
        self.attn_mha.kv_b_proj = None

        self.w_kc = None
        self.w_vc = None
        self.w_scale = 1.0

        self.w_scale_k = None
        self.w_scale_v = None
        self.use_deep_gemm_bmm = False

        self.current_attention_backend = (
            None  # Attention backend used by current forward batch
        )

        self.has_fused_proj = hasattr(self, "fused_qkv_a_proj_with_mqa")
        self.is_packed_weight = (
            self.has_fused_proj
            and hasattr(self.fused_qkv_a_proj_with_mqa.quant_method, "quant_config")
            and self.fused_qkv_a_proj_with_mqa.quant_method.quant_config.get_name()
            in {"awq", "awq_marlin", "moe_wna16"}
        )
        self.use_min_latency_fused_a_gemm = (
            self.has_fused_proj
            and not self.is_packed_weight
            and self.fused_qkv_a_proj_with_mqa.weight.dtype == torch.bfloat16
            and self.fused_qkv_a_proj_with_mqa.weight.shape[0] == 2112
            and self.fused_qkv_a_proj_with_mqa.weight.shape[1] == 7168
            and _is_cuda
            and 90 <= _device_sm < 120
        )

        self.init_mha_forward()
        self.init_mla_forward()
        self.init_mla_fused_rope_rocm_forward()
        self.init_mla_fused_rope_cpu_forward()

    def dispatch_attn_forward_method(
        self, forward_batch: ForwardBatch
    ) -> AttnForwardMethod:
        # Determine attention backend used by current forward batch
        if forward_batch.forward_mode.is_decode_or_idle():
            attention_backend = get_global_server_args().decode_attention_backend
        elif (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend()
        ):
            # Use the specified backend for speculative operations (both verify and draft extend)
            if get_global_server_args().speculative_attention_mode == "decode":
                attention_backend = get_global_server_args().decode_attention_backend
            else:  # default to prefill
                attention_backend = get_global_server_args().prefill_attention_backend
        else:
            attention_backend = get_global_server_args().prefill_attention_backend
        self.current_attention_backend = attention_backend

        handler = AttentionBackendRegistry.get_handler(attention_backend)
        return handler(self, forward_batch)

    def op_prepare(self, state):
        state.attn_intermediate_state = self.forward_prepare(
            positions=state.positions,
            hidden_states=state.pop("hidden_states_after_comm_pre_attn"),
            forward_batch=state.forward_batch,
            zero_allocator=state.zero_allocator,
        )

    def op_core(self, state):
        result = self.forward_core(state.pop("attn_intermediate_state"))
        # forward_core may return (hidden_states, topk_indices) for NSA models
        # with index cache enabled. In the TBO path, topk_indices is not
        # propagated between layers, so we discard it here.
        if isinstance(result, tuple):
            state.hidden_states_after_attn = result[0]
        else:
            state.hidden_states_after_attn = result

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        layer_scatter_modes: LayerScatterModes = None,
        llama_4_scaling: Optional[torch.Tensor] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ):
        s = self.forward_prepare(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
            layer_scatter_modes=layer_scatter_modes,
            llama_4_scaling=llama_4_scaling,
            prev_topk_indices=prev_topk_indices,
        )
        return self.forward_core(s)

    def forward_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        layer_scatter_modes: LayerScatterModes = None,
        llama_4_scaling: Optional[torch.Tensor] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ):
        if self.attn_mha.kv_b_proj is None:
            self.attn_mha.kv_b_proj = self.kv_b_proj

        # when hidden_states is a tuple of tensors, the tuple will include quantized weight and scale tensor
        if isinstance(hidden_states, tuple):
            if (
                not get_attn_tp_context().input_scattered
                and hidden_states[0].shape[0] == 0
            ):
                assert (
                    not self.o_proj.reduce_results
                ), "short-circuiting allreduce will lead to hangs"
                return hidden_states[0]
        else:
            if (
                not get_attn_tp_context().input_scattered
                and hidden_states.shape[0] == 0
            ):
                assert (
                    not self.o_proj.reduce_results
                ), "short-circuiting allreduce will lead to hangs"
                return hidden_states, None, forward_batch, None
        
        attn_forward_method = self.dispatch_attn_forward_method(forward_batch)
        if attn_forward_method == AttnForwardMethod.MHA:
            inner_state = self.forward_normal_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MHA_CHUNKED_KV:
            inner_state = self.forward_normal_chunked_kv_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MHA_ONE_SHOT:
            inner_state = self.forward_normal_one_shot_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MLA:
            inner_state = self.forward_absorb_prepare(
                positions,
                hidden_states,
                forward_batch,
                zero_allocator,
                llama_4_scaling,
                prev_topk_indices,
            )
        elif attn_forward_method == AttnForwardMethod.MLA_FUSED_ROPE_ROCM:
            inner_state = self.forward_absorb_fused_mla_rope_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MLA_FUSED_ROPE_CPU:
            inner_state = self.forward_absorb_fused_mla_rope_cpu_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MHA_NPU:
            inner_state = forward_mha_prepare_npu(
                self,
                positions,
                hidden_states,
                forward_batch,
                zero_allocator,
                layer_scatter_modes,
            )
        elif attn_forward_method == AttnForwardMethod.MLA_NPU:
            inner_state = forward_mla_prepare_npu(
                self,
                positions,
                hidden_states,
                forward_batch,
                zero_allocator,
                layer_scatter_modes,
            )
        elif attn_forward_method == AttnForwardMethod.DSA_NPU:
            inner_state = forward_dsa_prepare_npu(
                self,
                positions,
                hidden_states,
                forward_batch,
                zero_allocator,
                layer_scatter_modes,
                prev_topk_indices,
            )
        else:
            raise NotImplementedError
        return None, attn_forward_method, forward_batch, inner_state

    def forward_core(self, intermediate_state):
        hidden_states, attn_forward_method, forward_batch, inner_state = (
            intermediate_state
        )
        if inner_state is None:
            return hidden_states
        if attn_forward_method == AttnForwardMethod.MHA:
            return self.forward_normal_core(*inner_state)
        elif attn_forward_method == AttnForwardMethod.MHA_CHUNKED_KV:
            return self.forward_normal_chunked_kv_core(*inner_state)
        elif attn_forward_method == AttnForwardMethod.MHA_ONE_SHOT:
            return self.forward_normal_one_shot_core(*inner_state)
        elif attn_forward_method == AttnForwardMethod.MLA:
            return self.forward_absorb_core(*inner_state)
        elif attn_forward_method == AttnForwardMethod.MLA_FUSED_ROPE_ROCM:
            return self.forward_absorb_fused_mla_rope_core(*inner_state)
        elif attn_forward_method == AttnForwardMethod.MLA_FUSED_ROPE_CPU:
            return self.forward_absorb_fused_mla_rope_cpu_core(*inner_state)
        elif attn_forward_method == AttnForwardMethod.MHA_NPU:
            return forward_mha_core_npu(self, *inner_state)
        elif attn_forward_method == AttnForwardMethod.MLA_NPU:
            return forward_mla_core_npu(self, *inner_state)
        elif attn_forward_method == AttnForwardMethod.DSA_NPU:
            return forward_dsa_core_npu(self, *inner_state)
        else:
            raise NotImplementedError

    def prepare_qkv_latent(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ):
        assert self.q_lora_rank is not None
        # When the module is wrapped with LoRA, the fused GEMM fast-path would
        # bypass the adapter because it reads weight.T directly.
        lora_active = getattr(self.fused_qkv_a_proj_with_mqa, "set_lora", False)
        if (
            (not isinstance(hidden_states, tuple))
            and hidden_states.shape[0] >= 1
            and hidden_states.shape[0] <= 16
            and self.use_min_latency_fused_a_gemm
            and not lora_active
        ):
            qkv_latent = dsv3_fused_a_gemm(
                hidden_states, self.fused_qkv_a_proj_with_mqa.weight.T
            )
        else:
            if _use_fused_rms_quant and self.input_layernorm is not None:
                # NOTE: suspected
                if _rms_quant_path == 1:
                # path 1
                    if forward_batch.residual_rms_per_quant_int8 is None:
                        qkv_latent, _ = self.fused_qkv_a_proj_with_mqa(
                            hidden_states, 
                            rms_weight=self.input_layernorm.weight.data,
                            residual=None,
                            update_hd=False)
                        forward_batch.residual_rms_per_quant_int8 = hidden_states
                    else:
                        qkv_latent, _ = self.fused_qkv_a_proj_with_mqa(
                            hidden_states, 
                            rms_weight=self.input_layernorm.weight.data,
                            residual=forward_batch.residual_rms_per_quant_int8,
                            update_hd=False)
                else:
                # path 2
                    if forward_batch.residual_rms_per_quant_int8 is None: 
                        _normed = self.input_layernorm(hidden_states)
                        forward_batch.residual_rms_per_quant_int8 = hidden_states
                    else:
                        _normed, _ = self.input_layernorm(hidden_states, forward_batch.residual_rms_per_quant_int8)
                    qkv_latent = self.fused_qkv_a_proj_with_mqa(_normed)[0]
                
            else:
                qkv_latent = self.fused_qkv_a_proj_with_mqa(hidden_states)[0]
        return qkv_latent

    def rebuild_cp_kv_cache(self, latent_cache, forward_batch, k_nope, k_pe):
        # support allgather+rerrange
        latent_cache[..., : self.kv_lora_rank] = k_nope.squeeze(1)
        latent_cache[..., self.kv_lora_rank :] = k_pe.squeeze(1)
        latent_cache_output = cp_all_gather_rerange_output(
            latent_cache.contiguous(),
            self.cp_size,
            forward_batch,
            torch.cuda.current_stream(),
        )
        k_nope = latent_cache_output[..., : self.kv_lora_rank].unsqueeze(1)
        k_pe = latent_cache_output[..., self.kv_lora_rank :].unsqueeze(1)
        return k_nope, k_pe

    def forward_absorb_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ):
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
        global _use_fused_rmsnorm_rope

        q_lora = None
        topk_indices = None
        if self.q_lora_rank is not None:
            q, latent_cache = (
                get_attn_tp_context()
                .fetch_qkv_latent()
                .split(
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1,
                )
            )
            k_nope = latent_cache[..., : self.kv_lora_rank]
            # overlap qk norm
            if self.alt_stream is not None and get_is_capture_mode() and not _use_fused_rmsnorm_rope and not _use_fused_rms_quant:
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                q = self.q_a_layernorm(q)
                with torch.cuda.stream(self.alt_stream):
                    k_nope = self.kv_a_layernorm(k_nope)
                current_stream.wait_stream(self.alt_stream)
            else:
                if _use_aiter_gfx95 and self.q_b_proj.weight.dtype == torch.uint8:
                    q, _, k_nope, *_ = fused_rms_mxfp4_quant(
                        q,
                        self.q_a_layernorm.weight,
                        self.q_a_layernorm.variance_epsilon,
                        k_nope,
                        self.kv_a_layernorm.weight,
                        self.kv_a_layernorm.variance_epsilon,
                    )
                else:
                    q_lora = None
                    if (
                        _use_aiter_gfx95
                        and self.q_b_proj.weight.dtype == torch.float8_e4m3fn
                    ):
                        if self.use_nsa:
                            q_quanted, q_lora, k_nope, _ = fused_rms_fp8_group_quant(
                                q,
                                self.q_a_layernorm.weight,
                                self.q_a_layernorm.variance_epsilon,
                                k_nope,
                                self.kv_a_layernorm.weight,
                                self.kv_a_layernorm.variance_epsilon,
                                group_size=128,
                                dtype_quant=torch.float8_e4m3fn,
                                res1=None,
                                output_unquantized_inp1=True,
                            )
                            q = q_quanted
                        else:
                            q, _, k_nope, _ = fused_rms_fp8_group_quant(
                                q,
                                self.q_a_layernorm.weight,
                                self.q_a_layernorm.variance_epsilon,
                                k_nope,
                                self.kv_a_layernorm.weight,
                                self.kv_a_layernorm.variance_epsilon,
                                group_size=128,
                                dtype_quant=torch.float8_e4m3fn,
                                res1=None,
                                output_unquantized_inp1=False,
                            )

                    else:
                        if not _use_fused_rms_quant:
                            q = self.q_a_layernorm(q)
                        if not _use_fused_rmsnorm_rope:
                            k_nope = self.kv_a_layernorm(k_nope)
            # q_lora needed by indexer
            if self.use_nsa:
                if q_lora is None:
                    q_lora = q
            # overlap q_b_proj and indexer during decode
            if (
                self.alt_stream is not None
                and get_is_capture_mode()
                and forward_batch.forward_mode.is_decode_or_idle()
                and q_lora is not None
            ):
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                with torch.cuda.stream(self.alt_stream):
                    k_nope = k_nope.unsqueeze(1)
                    q = self.q_b_proj(q)[0].view(
                        -1, self.num_local_heads, self.qk_head_dim
                    )
                topk_indices = self.indexer(
                    x=hidden_states,
                    q_lora=q_lora,
                    positions=positions,
                    forward_batch=forward_batch,
                    layer_id=self.layer_id,
                )
                current_stream.wait_stream(self.alt_stream)
            else:
                if not _use_fused_rmsnorm_rope:
                    k_nope = k_nope.unsqueeze(1)
                if not _use_fused_rms_quant:
                    q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
                else:
                    q = self.q_b_proj(q, rms_weight=self.q_a_layernorm.weight.data, residual=None,
                                    update_hd=False)[0].view(-1, self.num_local_heads, self.qk_head_dim)
                if q_lora is not None:
                    topk_indices = self.indexer(
                        x=hidden_states,
                        q_lora=q_lora,
                        positions=positions,
                        forward_batch=forward_batch,
                        layer_id=self.layer_id,
                    )
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            k_nope = latent_cache[..., : self.kv_lora_rank]
            if not _use_fused_rmsnorm_rope:
                k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)
        if _use_fused_rmsnorm_rope:
            k_nope_normed = torch.empty(k_nope.shape, dtype=k_nope.dtype, device=k_nope.device)
            weight = self.kv_a_layernorm.weight
            variance_epsilon = self.kv_a_layernorm.variance_epsilon
            cos_sin_cache = self.rotary_emb.cos_sin_cache
            kv_pool = forward_batch.token_to_kv_pool
            # 暂不注册自定义算子，多很多kernel影响性能
            fused_rms_norm_rope_contiguous(positions, q_pe, k_pe.squeeze(1), k_nope,
                                          k_nope_normed, weight, cos_sin_cache,
                                          forward_batch.out_cache_loc,
                                          kv_pool.kv_buffer[self.layer_id - kv_pool.start_layer],
                                          dtype_mapping.get(kv_pool.dtype),1.0,False,variance_epsilon)
            k_nope = k_nope.unsqueeze(1)
        if self.use_deep_gemm_bmm:
            q_nope_val, q_nope_scale, masked_m, expected_m, aligned_m = (
                per_token_group_quant_mla_deep_gemm_masked_fp8(q_nope.transpose(0, 1))
            )
            q_nope_out = q_nope.new_empty(
                (self.num_local_heads, aligned_m, self.kv_lora_rank)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (q_nope_val, q_nope_scale),
                (self.w_kc, self.w_scale_k),
                q_nope_out,
                masked_m,
                expected_m,
            )
            q_nope_out = q_nope_out[:, :expected_m, :]
        elif _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            if _use_aiter_gfx95 and self.w_kc.dtype == torch.uint8:
                x = q_nope.transpose(0, 1)
                q_nope_out = torch.empty(
                    x.shape[0],
                    x.shape[1],
                    self.w_kc.shape[2],
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                batched_gemm_afp4wfp4_pre_quant(
                    x,
                    self.w_kc.transpose(-2, -1),
                    self.w_scale_k.transpose(-2, -1),
                    torch.bfloat16,
                    q_nope_out,
                )
            else:
                # q_nope_out = ds_bmm_wrapper(q_nope, self.w_kc, self.w_scale, torch.bfloat16)
                 q_nope_out = torch.bmm(
                    q_nope.to(torch.bfloat16).transpose(0, 1),
                    self.w_kc.to(torch.bfloat16) * self.w_scale,
                )
        elif self.w_kc.dtype == torch.float8_e4m3fn and not _is_dcu:
            # fix bmm_fp8 error under cublas12.9 caused by bumpallocator, detail in pr#11612
            q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
                q_nope.transpose(0, 1),
                (
                    torch.zeros((1,), dtype=torch.float32, device=q_nope.device)
                    if _is_cublas_ge_129
                    else zero_allocator.allocate(1)
                ),
            )
            q_nope_out = bmm_fp8(
                q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
            )
        else:
            # q_t =q_nope.transpose(0, 1).to(torch.bfloat16)
            #     w_vc_t = self.w_vc.contiguous().to(torch.float32).to(torch.bfloat16)
            #     torch.bmm(
            #         q_t,
            #         w_vc_t
            #     )
            q_nope_out = torch.bmm(q_nope.transpose(0, 1).to(torch.bfloat16), self.w_kc.to(torch.bfloat16) * self.w_scale)
        q_nope_out = q_nope_out.transpose(0, 1)
        
        if (
            self.rotary_emb is not None
            and (not self._fuse_rope_for_trtllm_mla(forward_batch))
            and (not _use_aiter or not _is_gfx95_supported or self.use_nsa)
            and not _use_fused_rmsnorm_rope
        ):
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        if nsa_use_prefill_cp(forward_batch):
            # support allgather+rerrange
            k_nope, k_pe = self.rebuild_cp_kv_cache(
                latent_cache, forward_batch, k_nope, k_pe
            )
        return (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            forward_batch,
            zero_allocator,
            positions,
            topk_indices,
            llama_4_scaling,
            # k if k is not None else None,
        )

    def forward_absorb_core(
        self,
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
        llama_4_scaling,
        k=None,
    ):
        save_kv_cache = True
        if self.current_attention_backend in FORWARD_ABSORB_CORE_ATTENTION_BACKENDS:
            extra_args = {}
            if self._fuse_rope_for_trtllm_mla(forward_batch):
                extra_args = {
                    "cos_sin_cache": self.rotary_emb.cos_sin_cache,
                    "is_neox": self.rotary_emb.is_neox_style,
                    "llama_4_scaling": llama_4_scaling,
                }

            attn_output = self.attn_mqa(
                q_nope_out,
                k_nope,
                k_nope,
                forward_batch,
                q_rope=q_pe,
                k_rope=k_pe,
                **extra_args,
                **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
            )
        else:
            if _use_aiter_gfx95:
                cos = self.rotary_emb.cos_cache
                sin = self.rotary_emb.sin_cache

                kv_cache_dtype = (
                    fp8_dtype if self.kv_cache_dtype == "fp8_e4m3" else q_nope_out.dtype
                )

                q, _, _, k = fused_qk_rope_cat_and_cache_mla(
                    q_nope_out,
                    q_pe,
                    k_nope,
                    k_pe,
                    forward_batch.token_to_kv_pool.get_key_buffer(
                        self.attn_mqa.layer_id
                    ),
                    forward_batch.out_cache_loc,
                    positions,
                    cos,
                    sin,
                    self.attn_mqa.k_scale,
                    self.rotary_emb.is_neox_style,
                    q_out_dtype=kv_cache_dtype,
                )

                save_kv_cache = False
                attn_output = self.attn_mqa( 
                    q,
                    k,
                    k_nope,
                    forward_batch,
                    **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
                )
            elif _use_fused_mla_cat:
                save_kv_cache = False
                attn_output = self.attn_mqa( 
                    q_nope_out,
                    k_nope,
                    k_nope,
                    forward_batch,
                    q_rope=q_pe,
                    k_rope=k_pe,
                    **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
                )
            else:
                if _use_opt_cat_decode and q_nope_out.shape[0] < 1024:
                    q = concat_decode_opt(q_nope_out, q_pe, dim=2)
                    # k = torch.cat([k_nope, k_pe], dim=-1)  # TODO: remove when use fused rms rope set kv

                    attn_output = self.attn_mqa(
                        q,
                        k_nope,
                        k_nope,
                        forward_batch,
                        k_rope=k_pe,
                        **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
                    )

                else:
                    q = torch.cat([q_nope_out, q_pe], dim=-1)
                    k = torch.cat([k_nope, k_pe], dim=-1)

                    attn_output = self.attn_mqa(
                        q,
                        k,
                        k_nope,
                        forward_batch,
                        **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
                    )

            # Apply llama 4 scaling if provided
            if llama_4_scaling is not None:
                q *= llama_4_scaling

                attn_output = self.attn_mqa(
                    q,
                    k,
                    k_nope,
                    forward_batch,
                    save_kv_cache=save_kv_cache,
                    **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
                )
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)
        if self.use_deep_gemm_bmm:
            attn_output_val, attn_output_scale, masked_m, expected_m, aligned_m = (
                per_token_group_quant_mla_deep_gemm_masked_fp8(
                    attn_output.transpose(0, 1)
                )
            )
            attn_bmm_output = attn_output.new_empty(
                (self.num_local_heads, aligned_m, self.v_head_dim)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (attn_output_val, attn_output_scale),
                (self.w_vc, self.w_scale_v),
                attn_bmm_output,
                masked_m,
                expected_m,
            )
            attn_bmm_output = (
                attn_bmm_output[:, :expected_m, :].transpose(0, 1).flatten(1, 2)
            )
        elif _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            if _use_aiter_gfx95 and self.w_vc.dtype == torch.uint8:
                x = attn_output.transpose(0, 1)
                attn_bmm_output = torch.empty(
                    x.shape[0],
                    x.shape[1],
                    self.w_vc.shape[2],
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                batched_gemm_afp4wfp4_pre_quant(
                    x,
                    self.w_vc.transpose(-2, -1),
                    self.w_scale_v.transpose(-2, -1),
                    torch.bfloat16,
                    attn_bmm_output,
                )
            else:
                # attn_bmm_output = ds_bmm_wrapper(attn_output, self.w_vc, self.w_scale, torch.bfloat16)
                attn_bmm_output = torch.bmm(
                    attn_output.to(torch.bfloat16).transpose(0, 1),
                    self.w_vc.to(torch.bfloat16) * self.w_scale,
                )

            if self.o_proj.weight.dtype == torch.uint8:
                attn_bmm_output = attn_bmm_output.transpose(0, 1)
                attn_bmm_output = fused_flatten_mxfp4_quant(attn_bmm_output)
            # elif self.o_proj.weight.dtype == torch.float8_e4m3fn:
            #     attn_bmm_output = attn_bmm_output.transpose(0, 1)
            #     attn_bmm_output = fused_flatten_fp8_group_quant(
            #         attn_bmm_output, group_size=128, dtype_quant=torch.float8_e4m3fn
            #     )
            else:
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)

        # elif self.w_vc.dtype == torch.float8_e4m3fn:
        #     attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
        #         attn_output.transpose(0, 1),
        #         (
        #             torch.zeros((1,), dtype=torch.float32, device=attn_output.device)
        #             if _is_cublas_ge_129
        #             else zero_allocator.allocate(1)
        #         ),
        #     )
        #     attn_bmm_output = bmm_fp8(
        #         attn_output_val,
        #         self.w_vc,
        #         attn_output_scale,
        #         self.w_scale,
        #         torch.bfloat16,
        #     )
        #     attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        else:
            if is_in_piecewise_cuda_graph():
                # torch dynamo requires out= op was called where output tensor was non-contiguous
                attn_bmm_output = (
                    torch.bmm(attn_output.transpose(0, 1), self.w_vc)
                    .transpose(0, 1)
                    .flatten(1, 2)
                )
            else:
                attn_bmm_output = torch.empty(
                    (attn_output.shape[0], self.num_local_heads * self.v_head_dim),
                    dtype=attn_output.dtype,
                    device=attn_output.device,
                )
                q_t = attn_output.transpose(0, 1).to(torch.bfloat16)
                w_vc_t = self.w_vc.contiguous().to(torch.float32).to(torch.bfloat16)
                torch.bmm(
                    q_t,
                    w_vc_t,
                    out=attn_bmm_output.view(
                        -1, self.num_local_heads, self.v_head_dim
                    ).transpose(0, 1),
                )
                # torch.bmm(
                #     attn_output.transpose(0, 1).to(torch.bfloat16),
                #     self.w_vc.to(torch.bfloat16),
                #     out=attn_bmm_output.view(
                #         -1, self.num_local_heads, self.v_head_dim
                #     ).transpose(0, 1),
                # )
        output, _ = self.o_proj(attn_bmm_output)
        # 如果第一维是1才 squeeze
        return output

    def forward_absorb_fused_mla_rope_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        enable_rope_fusion = (
            os.getenv("SGLANG_FUSED_MLA_ENABLE_ROPE_FUSION", "1") == "1"
        )
        # NOTE: hidden_states can be a tuple for some quantization paths.
        # For shape/device/dtype, use the first tensor; still pass the original
        # hidden_states through linear ops which may accept tuple inputs.
        hidden_states_tensor = (
            hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
        )

        q_len = hidden_states_tensor.shape[0]
        q_input = hidden_states_tensor.new_empty(
            q_len, self.num_local_heads, self.kv_lora_rank + self.qk_rope_head_dim
        )
        if self.q_lora_rank is not None:
            q, latent_cache = self.fused_qkv_a_proj_with_mqa(hidden_states)[0].split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        if _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            q_nope_out = torch.bmm(
                q_nope.to(torch.bfloat16).transpose(0, 1),
                self.w_kc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_kc.dtype == torch.float8_e4m3fn:
            q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
                q_nope.transpose(0, 1),
                zero_allocator.allocate(1),
                dtype=torch.float8_e4m3fn,
            )
            q_nope_out = bmm_fp8(
                q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
            )
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)
        q_input[..., : self.kv_lora_rank] = q_nope_out.transpose(0, 1)
        v_input = latent_cache[..., : self.kv_lora_rank]
        v_input = self.kv_a_layernorm(v_input.contiguous()).unsqueeze(1)
        k_input = latent_cache.unsqueeze(1)
        k_input[..., : self.kv_lora_rank] = v_input

        if not enable_rope_fusion:
            k_pe = k_input[..., self.kv_lora_rank :]
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
            q_input[..., self.kv_lora_rank :] = q_pe
            k_input[..., self.kv_lora_rank :] = k_pe
            k_pe_output = None
        else:
            k_pe_output = torch.empty_like(k_input[..., self.kv_lora_rank :])

        q_input[..., self.kv_lora_rank :] = q_pe

        # attn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)
        # Use Fused ROPE with use_rope=OFF.
        attn_output = torch.empty(
            (q_len, self.num_local_heads, self.kv_lora_rank),
            dtype=q.dtype,
            device=q.device,
        )
        attn_logits, _, kv_indptr, kv_indices, _, _, _ = (
            forward_batch.attn_backend.forward_metadata
        )
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        num_kv_split = forward_batch.attn_backend.num_kv_splits
        sm_scale = self.attn_mqa.scaling
        if attn_logits is None:
            attn_logits = torch.empty(
                (
                    forward_batch.batch_size,
                    self.num_local_heads,
                    num_kv_split,
                    self.kv_lora_rank + 1,
                ),
                dtype=torch.float32,
                device=q.device,
            )

        # save current latent cache.
        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.attn_mqa, forward_batch.out_cache_loc, k_input, None
        )
        key_cache_buf = forward_batch.token_to_kv_pool.get_key_buffer(
            self.attn_mqa.layer_id
        )
        val_cache_buf = key_cache_buf[..., : self.kv_lora_rank]

        return (
            q_input,
            key_cache_buf,
            val_cache_buf,
            attn_output,
            kv_indptr,
            kv_indices,
            k_pe_output,
            cos_sin_cache,
            positions,
            attn_logits,
            num_kv_split,
            sm_scale,
            enable_rope_fusion,
            k_input,
            forward_batch,
            zero_allocator,
        )

    def forward_absorb_fused_mla_rope_cpu_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        assert self.q_lora_rank is not None and use_intel_amx_backend(
            self
        ), "forward_absorb_fused_mla_rope_cpu_prepare requires q_lora_rank is not None and use_intel_amx_backend"

        q_input, k_input, v_input = (
            torch.ops.sgl_kernel.qkv_proj_with_rope_fused_weight(
                hidden_states,
                self.fused_qkv_a_proj_with_mqa.weight,
                self.q_b_proj.weight,
                self.w_kc,
                self.q_a_layernorm.weight,
                self.kv_a_layernorm.weight,
                positions,
                self.rotary_emb.cos_sin_cache,
                self.kv_a_layernorm.variance_epsilon,
                self.qkv_proj_with_rope_is_int8,
                self.qkv_proj_with_rope_is_fp8,
                (
                    self.fused_qkv_a_proj_with_mqa.weight_scale
                    if self.qkv_proj_with_rope_is_int8
                    else (
                        self.fused_qkv_a_proj_with_mqa.weight_scale_inv
                        if self.qkv_proj_with_rope_is_fp8
                        else None
                    )
                ),
                (
                    self.q_b_proj.weight_scale
                    if self.qkv_proj_with_rope_is_int8
                    else (
                        self.q_b_proj.weight_scale_inv
                        if self.qkv_proj_with_rope_is_fp8
                        else None
                    )
                ),
                True,  # is_vnni
                self.weight_block_size,
                self.q_lora_rank,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
            )
        )
        return (q_input, k_input, v_input, forward_batch, zero_allocator)

    def forward_absorb_fused_mla_rope_core(
        self,
        q_input,
        key_cache_buf,
        val_cache_buf,
        attn_output,
        kv_indptr,
        kv_indices,
        k_pe_output,
        cos_sin_cache,
        positions,
        attn_logits,
        num_kv_split,
        sm_scale,
        enable_rope_fusion,
        k_input,
        forward_batch,
        zero_allocator,
    ):
        decode_attention_fwd_grouped_rope(
            q_input,
            key_cache_buf,
            val_cache_buf,
            attn_output,
            kv_indptr,
            kv_indices,
            k_pe_output,
            self.kv_lora_rank,
            self.rotary_emb.rotary_dim,
            cos_sin_cache,
            positions,
            attn_logits,
            num_kv_split,
            sm_scale,
            logit_cap=self.attn_mqa.logit_cap,
            use_rope=enable_rope_fusion,
            is_neox_style=self.rotary_emb.is_neox_style,
        )

        if enable_rope_fusion:
            k_input[..., self.kv_lora_rank :] = k_pe_output
            forward_batch.token_to_kv_pool.set_kv_buffer(
                self.attn_mqa, forward_batch.out_cache_loc, k_input, None
            )

        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        if _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            attn_bmm_output = torch.bmm(
                attn_output.to(torch.bfloat16).transpose(0, 1),
                self.w_vc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_vc.dtype == torch.float8_e4m3fn:
            attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                attn_output.transpose(0, 1),
                zero_allocator.allocate(1),
                dtype=torch.float8_e4m3fn,
            )
            attn_bmm_output = bmm_fp8(
                attn_output_val,
                self.w_vc,
                attn_output_scale,
                self.w_scale,
                torch.bfloat16,
            )
        else:
            attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)
        attn_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        output, _ = self.o_proj(attn_output)

        return output

    def forward_absorb_fused_mla_rope_cpu_core(
        self, q_input, k_input, v_input, forward_batch, zero_allocator
    ):
        assert self.q_lora_rank is not None and use_intel_amx_backend(
            self
        ), "forward_absorb_fused_mla_rope_cpu_core requires q_lora_rank is not None and use_intel_amx_backend"

        attn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        # [Note] Align shapes of bmm inputs.
        # Shapes of inputs:
        #   q_nope: [M, B, K]
        #   original self.w_kc: [B, K, N]
        #   current self.w_kc (which has been converted in PackWeightMethod): [B, N, K]

        # Shapes of inputs to sgl_kernel.cpu.bmm:
        #   out: [B, M, N]
        #   mat1: [B, M, K]
        #   mat2: [B, N, K]
        B = self.w_vc.size(0)
        N = self.w_vc.size(1)
        M = attn_output.size(0)
        output = torch.empty([M, int(B * N)], dtype=attn_output.dtype)
        attn_bmm_output = output.view([M, B, N]).transpose_(0, 1)
        torch.ops.sgl_kernel.bmm_cpu(
            attn_bmm_output,
            attn_output.transpose(0, 1),
            self.w_vc,
            True,  # is_vnni
            None,  # scale
        )
        attn_output = output
        output, _ = self.o_proj(attn_output)

        return output

    def _chunked_prefix_attn_mha(
        self,
        q: torch.Tensor,
        accum_output: torch.Tensor,
        accum_lse: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:

        assert forward_batch.num_prefix_chunks is not None
        for i in range(forward_batch.num_prefix_chunks):
            forward_batch.set_prefix_chunk_idx(i)

            kv_indices = forward_batch.prefix_chunk_kv_indices[i]
            # Fetch latent cache from memory pool with precomputed chunked kv indices
            latent_cache_buf, dtype = forward_batch.token_to_kv_pool.get_key_buffer_DeepSeekV2(
                self.attn_mha.layer_id
            )
            latent_cache = (
                latent_cache_buf[forward_batch.prefix_chunk_kv_indices[i]]
                .contiguous()
                .view(dtype)
                .to(q.dtype)
            )

            kv_a_normed, k_pe = latent_cache.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_a_normed = kv_a_normed.squeeze(1).contiguous()
            kv = self.kv_b_proj(kv_a_normed)[0]
            kv = kv.view(
                -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            v = kv[..., self.qk_nope_head_dim :]
            k_nope = kv[..., : self.qk_nope_head_dim]

            k = torch.empty(
                (
                    k_nope.shape[0],
                    self.num_local_heads,
                    self.qk_nope_head_dim + self.qk_rope_head_dim,
                ),
                dtype=v.dtype,
                device=v.device,
            )
            k[..., : self.qk_nope_head_dim] = k_nope
            k[..., self.qk_nope_head_dim :] = k_pe

            output, lse = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
            tmp_output = torch.empty_like(accum_output)
            tmp_lse = torch.empty_like(accum_lse)
            merge_state_v2(output, lse, accum_output, accum_lse, tmp_output, tmp_lse)
            accum_output, accum_lse = tmp_output, tmp_lse
            del kv, k, v, output, lse, tmp_output, tmp_lse

        return accum_output

    def forward_normal_chunked_kv_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        # In normal mha, the k and v tensors will become overly large when the prefix length is long.
        # To avoid this, we split the kv cache into chunks and process them one after another.
        # Since mha is compute friendly, the for loop induced here will not introduce significant overhead.
        # The top comments in https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/common.py
        # will be helpful for understanding the purpose of this function.

        # First do normal mha forward to get output for extended part
        return self.forward_normal_prepare(
            positions, hidden_states, forward_batch, zero_allocator
        )

    def forward_normal_chunked_kv_core(self, q, k, v, forward_batch):
        has_extend_prefix = forward_batch.extend_prefix_lens_cpu is not None and any(
            forward_batch.extend_prefix_lens_cpu
        )
        # Only initialize the info once
        if has_extend_prefix and forward_batch.num_prefix_chunks is None:
            forward_batch.prepare_chunked_prefix_cache_info(q.device)
            if hasattr(forward_batch.attn_backend, "init_mha_chunk_metadata"):
                forward_batch.attn_backend.init_mha_chunk_metadata(forward_batch)

        forward_batch.mha_return_lse = has_extend_prefix
        # Do mha for extended part without prefix
        forward_batch.set_attn_attend_prefix_cache(False)
        attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
        
        # Do mha attention with chunked prefix cache if there are any sequence with prefix
        if has_extend_prefix:
            attn_output, lse = attn_output
            forward_batch.set_attn_attend_prefix_cache(True)
            attn_output = self._chunked_prefix_attn_mha(
                q=q,
                accum_output=attn_output,
                accum_lse=lse,
                forward_batch=forward_batch,
            )

        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output

    def forward_normal_one_shot_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        forward_batch.mha_one_shot = True
        return self.forward_normal_prepare(
            positions, hidden_states, forward_batch, zero_allocator
        )

    def forward_normal_one_shot_core(self, q, k, v, forward_batch):
        has_extend_prefix = any(forward_batch.extend_prefix_lens_cpu)
        # Only initialize the info once
        if has_extend_prefix and forward_batch.num_prefix_chunks is None:
            forward_batch.num_prefix_chunks = 0
            if hasattr(forward_batch.attn_backend, "init_mha_chunk_metadata"):
                forward_batch.attn_backend.init_mha_chunk_metadata(forward_batch)
        forward_batch.mha_return_lse = False
        # Do mha for extended part without prefix
        forward_batch.set_attn_attend_prefix_cache(False)
        return self.forward_normal_core(q, k, v, forward_batch)

    def _set_mla_kv_buffer(
        self,
        latent_cache: torch.Tensor,
        kv_a: torch.Tensor,
        k_pe: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if _is_cuda or _use_aiter_gfx95:
            # Save latent cache
            forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                self.attn_mha, forward_batch.out_cache_loc, kv_a.unsqueeze(1), k_pe
            )
        elif _is_npu:
            # To reduce a time-costing split operation
            forward_batch.token_to_kv_pool.set_kv_buffer(
                self.attn_mha, forward_batch.out_cache_loc, kv_a.unsqueeze(1), k_pe
            )
        else:
            latent_cache[:, :, : self.kv_lora_rank] = kv_a.unsqueeze(1)
            latent_cache[:, :, self.kv_lora_rank :] = k_pe

            # Save latent cache
            forward_batch.token_to_kv_pool.set_kv_buffer(
                self.attn_mha, forward_batch.out_cache_loc, latent_cache, None
            )

    def _get_mla_kv_buffer(
        self,
        kv_indices: torch.Tensor,
        dst_dtype: torch.dtype,
        forward_batch: ForwardBatch,
    ):
        if _is_cuda or _use_aiter_gfx95:
            kv_a, k_pe = forward_batch.token_to_kv_pool.get_mla_kv_buffer(
                self.attn_mha, kv_indices, dst_dtype
            )
            kv_a = kv_a.squeeze(1)
        else:
            latent_cache_buf = forward_batch.token_to_kv_pool.get_key_buffer(
                self.attn_mha.layer_id
            )
            latent_cache = latent_cache_buf[kv_indices].contiguous().to(dst_dtype)

            kv_a, k_pe = latent_cache.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_a = kv_a.squeeze(1).contiguous()
        return kv_a, k_pe

    def _get_mla_kv_buffer_from_fp8(
        self,
        forward_batch: ForwardBatch,
    ):
        """
        Dequantize FP8 KV cache to BF16 for MLA attention (NSA-specific format).

        Returns: (kv_a, k_pe) both in BF16
        """
        backend = forward_batch.attn_backend
        if isinstance(backend, TboAttnBackend):  # if enable tbo, get primary backend
            backend = backend.primary
        kv_indices = backend.forward_metadata.page_table_1_flattened
        assert (
            kv_indices is not None
        ), "page_table_1_flattened should have been generated for FP8 MHA path"

        kv_cache_fp8 = forward_batch.token_to_kv_pool.get_key_buffer(
            self.attn_mha.layer_id
        )

        kv_latent_bf16 = dequantize_k_cache_paged(kv_cache_fp8, kv_indices)

        kv_a = kv_latent_bf16[:, :, : self.kv_lora_rank].squeeze(1).contiguous()
        k_pe = kv_latent_bf16[:, :, self.kv_lora_rank :]

        return kv_a, k_pe

    def _concat_and_cast_mha_k(self, k_nope, k_pe, forward_batch):
        # Temporary for DeepSeek V3/R1 only, but can generalize if needed
        k_shape = (k_nope.shape[0], self.num_local_heads, self.qk_head_dim)
        if (
            _is_cuda
            and (self.num_local_heads == 128)
            and (self.qk_nope_head_dim == 128)
            and (self.qk_rope_head_dim == 64)
        ):
            k = k_nope.new_empty(*k_shape)
            concat_mla_k(k=k, k_nope=k_nope, k_rope=k_pe)
        elif _is_cuda:
            # fa3 mha support fp8 inputs
            if (
                self.current_attention_backend == "fa3"
                and self.kv_cache_dtype != "auto"
            ):
                attn_dtype = forward_batch.token_to_kv_pool.dtype
            else:
                attn_dtype = k_nope.dtype
            k = k_nope.new_empty(*k_shape, dtype=attn_dtype)
            concat_and_cast_mha_k_triton(k, k_nope, k_pe)
        elif _is_hip and self.current_attention_backend == "aiter":
            k = k_nope.new_empty(*k_shape)
            concat_and_cast_mha_k_triton(k, k_nope, k_pe)
        else:
            k = k_nope.new_empty(*k_shape)
            k[..., : self.qk_nope_head_dim] = k_nope
            k[..., self.qk_nope_head_dim :] = k_pe
        return k

    @staticmethod
    def _get_q_b_proj_quant_config(quant_config):
        if envs.SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN.get():
            # refer to real DeepSeek V3 quant config
            return Fp8Config(
                is_checkpoint_fp8_serialized=True,
                weight_block_size=[128, 128],
            )
        else:
            return quant_config


class DeepseekV2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        moe_quant_config_override: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        if hasattr(config, "rope_parameters"):
            rope_theta = config.rope_parameters["rope_theta"]
            assert rope_theta is not None, f"rope_theta not found in config: {config}"
            rope_type = config.rope_parameters.get("rope_type")
            rope_scaling = config.rope_parameters if rope_type != "default" else None
        else:
            rope_theta = config.rope_theta
            rope_scaling = config.rope_scaling
        max_position_embeddings = config.max_position_embeddings
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            get_global_server_args().speculative_algorithm
        )
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        self.layer_id = layer_id
        self.is_nextn = is_nextn
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DeepseekV2AttentionMLA(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=(
                config.q_lora_rank if hasattr(config, "q_lora_rank") else None
            ),
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            layer_id=layer_id,
            reduce_results=False,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
            is_nextn=is_nextn,
            input_layernorm=self.input_layernorm,
        )
        if not hasattr(config, "q_lora_rank") and envs.SGLANG_USE_AG_AFTER_QLORA.get():
            raise ValueError(
                "SGLANG_USE_AG_AFTER_QLORA only supports the model with q_lora_rank"
            )

        self.is_layer_sparse = self._is_layer_sparse(layer_id, is_nextn=is_nextn)
        is_previous_layer_sparse = self._is_layer_sparse(layer_id - 1, is_nextn=False)
        is_next_layer_sparse = self._is_layer_sparse(layer_id + 1, is_nextn=False)

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=1 if is_nextn else config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        if self.is_layer_sparse:
            self.mlp = DeepseekV2MoE(
                config=config,
                quant_config=moe_quant_config_override or quant_config,
                prefix=add_prefix("mlp", prefix),
                layer_id=self.layer_id,
                alt_stream=alt_stream,
                is_nextn=is_nextn,
            )
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
                swiglu_limit=getattr(config, "swiglu_limit", None),
            )

        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self._gfx95_quant_format = self._detect_gfx95_quant_format()

        if self.nsa_enable_prefill_cp:
            self.layer_communicator = NSACPLayerCommunicator(
                layer_scatter_modes=self.layer_scatter_modes,
                input_layernorm=self.input_layernorm,
                post_attention_layernorm=self.post_attention_layernorm,
                allow_reduce_scatter=True,
                is_last_layer=(
                    is_nextn or (self.layer_id == self.config.num_hidden_layers - 1)
                ),
                qkv_latent_func=self.self_attn.prepare_qkv_latent,
            )
        else:
            self.layer_communicator = LayerCommunicator(
                layer_scatter_modes=self.layer_scatter_modes,
                input_layernorm=self.input_layernorm,
                post_attention_layernorm=self.post_attention_layernorm,
                allow_reduce_scatter=True,
                is_last_layer=(
                    is_nextn or (self.layer_id == self.config.num_hidden_layers - 1)
                ),
                qkv_latent_func=self.self_attn.prepare_qkv_latent,
            )

    def _detect_gfx95_quant_format(self) -> str:
        if not _is_gfx95_supported:
            return ""
        weight = getattr(
            getattr(self.self_attn, "fused_qkv_a_proj_with_mqa", None), "weight", None
        )
        if weight is None:
            return ""
        if weight.dtype == torch.uint8:
            return "mxfp4"
        if weight.dtype == getattr(torch, "float8_e4m3fn", None):
            return "fp8"
        return ""

    def _is_layer_sparse(self, layer_id: int, is_nextn: bool) -> bool:
        return is_nextn or (
            self.config.n_routed_experts is not None
            and layer_id >= self.config.first_k_dense_replace
            and layer_id % self.config.moe_layer_freq == 0
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
        gemm_output_zero_allocator: BumpAllocator = None,
        llama_4_scaling: Optional[torch.Tensor] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        quant_format = (
            "mxfp4"
            if (
                _is_gfx95_supported
                and getattr(self.self_attn, "fused_qkv_a_proj_with_mqa", None)
                is not None
                and getattr(self.self_attn.fused_qkv_a_proj_with_mqa, "weight", None)
                is not None
                and self.self_attn.fused_qkv_a_proj_with_mqa.weight.dtype == torch.uint8
            )
            else (
                "fp8"
                if (
                    _is_gfx95_supported
                    and getattr(self.self_attn, "fused_qkv_a_proj_with_mqa", None)
                    is not None
                    and getattr(
                        self.self_attn.fused_qkv_a_proj_with_mqa, "weight", None
                    )
                    is not None
                    and self.self_attn.fused_qkv_a_proj_with_mqa.weight.dtype
                    == getattr(torch, "float8_e4m3fn", None)
                )
                else ""
            )
        )
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states,
            residual,
            forward_batch,
            getattr(self, "_gfx95_quant_format", ""),
        )
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
            llama_4_scaling=llama_4_scaling,
            layer_scatter_modes=self.layer_scatter_modes,
            prev_topk_indices=prev_topk_indices,
        )
        if isinstance(hidden_states, tuple):
            hidden_states, topk_indices = hidden_states
        else:
            topk_indices = None

        # residual = forward_batch.residual_rms_per_quant_int8  # residual在attn中没有变化，不用这句
        # 判断sbo+moe，如果是则不跳过norm
        forward_batch.rms_quant_flag = not (self.is_layer_sparse and _is_sbo_enabled)  # NOTE: if _is_sbo_enabled reliable?
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )

        # For DP with padding, reduce scatter can be used instead of all-reduce.
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )
        
        if isinstance(self.mlp, DeepseekV2MLP):
            gemm_output_zero_allocator = None
        # 前三层dense，开融合时返回值为4个
        if _use_fused_rms_quant and residual is not None and self.post_attention_layernorm.weight.data is not None and isinstance(self.mlp, DeepseekV2MLP):
            hidden_states, _, _, _ = self.mlp(
                hidden_states,
                forward_batch,
                should_allreduce_fusion,
                use_reduce_scatter,
                gemm_output_zero_allocator,
                rms_weight=self.post_attention_layernorm.weight.data,
                residual=residual,
            )
        else:  # 不管开不开融合，结果是一个就行
            hidden_states = self.mlp(
                hidden_states,
                forward_batch,
                should_allreduce_fusion,
                use_reduce_scatter,
                gemm_output_zero_allocator,
                rms_weight=self.post_attention_layernorm.weight.data,
                residual=residual,
            )

        if not self.nsa_enable_prefill_cp and should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True

        if not should_allreduce_fusion:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        return hidden_states, residual, topk_indices

    def op_comm_prepare_attn(
        self,
        state,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
        tbo_subbatch_index: Optional[int] = None,
    ):
        state.hidden_states_after_comm_pre_attn, state.residual_after_input_ln = (
            self.layer_communicator.prepare_attn(hidden_states, residual, forward_batch)
        )
        if get_moe_a2a_backend().is_mori():
            state.num_tokens = hidden_states.shape[0]
        state.update(
            dict(
                forward_batch=forward_batch,
                positions=positions,
                zero_allocator=zero_allocator,
                tbo_subbatch_index=tbo_subbatch_index,
            )
        )

    def op_comm_prepare_mlp(self, state):
        state.hidden_states_mlp_input, state.residual_after_comm_pre_mlp = (
            self.layer_communicator.prepare_mlp(
                state.pop("hidden_states_after_attn"),
                state.pop("residual_after_input_ln"),
                state.forward_batch,
            )
        )

    def op_mlp(self, state):
        hidden_states = state.pop("hidden_states_mlp_input")
        if not (
            enable_moe_dense_fully_dp()
            and (not self.is_layer_sparse)
            and hidden_states.shape[0] == 0
        ):
            state.hidden_states_mlp_output = self.mlp(
                hidden_states, state.forward_batch
            )
        else:
            state.hidden_states_mlp_output = hidden_states

    def op_comm_postprocess_layer(self, state):
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            state.pop("hidden_states_mlp_output"),
            state.pop("residual_after_comm_pre_mlp"),
            state.forward_batch,
        )

        output = dict(
            positions=state.positions,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=state.forward_batch,
            zero_allocator=state.zero_allocator,
            tbo_subbatch_index=state.tbo_subbatch_index,
        )

        state.clear(
            expect_keys={
                "positions",
                "forward_batch",
                "zero_allocator",
                "tbo_subbatch_index",
            }
        )
        return output


class DeepseekV2Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.first_k_dense_replace = config.first_k_dense_replace
        self.pp_group = get_pp_group()
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_size = get_attention_cp_size()
        else:
            self.cp_size = None

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                use_attn_tp_group=is_dp_attention_enabled(),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.alt_stream = (
            torch.cuda.Stream()
            if (
                _is_cuda
                or _is_musa
                or envs.SGLANG_NPU_USE_MULTI_STREAM.get()
                or envs.SGLANG_ROCM_USE_MULTI_STREAM.get()
            )
            else None
        )

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: DeepseekV2DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=self.alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
            offloader_kwargs=dict(
                submodule_accessor=lambda layer: (
                    layer.mlp.experts
                    if isinstance(layer.mlp, DeepseekV2MoE)
                    else layer.mlp
                ),
                whitelist_param_names_creator=lambda module: (
                    [
                        "w13_weight",
                        "w2_weight",
                        # only for nvfp4
                        *(
                            [
                                "w13_blockscale_swizzled",
                                "w2_blockscale_swizzled",
                            ]
                            if hasattr(module, "w13_blockscale_swizzled")
                            else []
                        ),
                    ]
                    if isinstance(module, FusedMoE)
                    else []
                ),
            ),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        self.gemm_output_zero_allocator_size = 0
        if (
            _use_aiter_gfx95
            and config.n_routed_experts == 256
            and self.embed_tokens.embedding_dim == 7168
        ):
            num_moe_layers = sum(
                [
                    1
                    for i in range(len(self.layers))
                    if isinstance(self.layers[i].mlp, DeepseekV2MoE)
                ]
            )

            allocate_size = 0
            for i in range(len(self.layers)):
                if isinstance(self.layers[i].mlp, DeepseekV2MoE):
                    # tp_size = get_tensor_model_parallel_world_size()
                    is_a2a_moe = is_deepep_class_backend()
                    tp_size = (
                        1 if is_a2a_moe else get_tensor_model_parallel_world_size()
                    )
                    intermediate_size = (
                        config.moe_intermediate_size * config.n_shared_experts
                    )
                    share_expert_output_size_per_partition = divide(
                        intermediate_size * 2, tp_size
                    )
                    allocate_size = share_expert_output_size_per_partition
                    break

            self.gemm_output_zero_allocator_size = (
                get_dsv3_gemm_output_zero_allocator_size(
                    config.n_routed_experts,
                    num_moe_layers,
                    allocate_size,
                    self.embed_tokens.embedding_dim,
                )
            )
        self.layers_to_capture = []
        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
            self.enable_a2a_moe = True
        else:
            self.enable_a2a_moe = False

        # llama_4_scaling: for supporting Mistral-Large-3 model
        self.llama_4_scaling_config = getattr(config, "llama_4_scaling", None)

    def get_input_embeddings(self) -> torch.Tensor:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        total_num_layers = self.end_layer - self.start_layer
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]
        device = hidden_states.device
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2 * (2 if forward_batch.can_run_tbo else 1),
            dtype=torch.float32,
            device=device,
        )

        has_gemm_output_zero_allocator = hasattr(
            self, "gemm_output_zero_allocator_size"
        )

        gemm_output_zero_allocator = (
            BumpAllocator(
                buffer_size=self.gemm_output_zero_allocator_size,
                dtype=torch.float32,
                device=device,
            )
            if has_gemm_output_zero_allocator
            and self.gemm_output_zero_allocator_size > 0
            else None
        )

        if nsa_use_prefill_cp(forward_batch):
            if self.pp_group.is_first_rank:
                hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
            positions = cp_split_and_rebuild_position(forward_batch, positions)
        # llama_4_scaling: for supporting Mistral-Large-3 model
        # Compute llama 4 scaling once per forward pass if enabled
        llama_4_scaling: Optional[torch.Tensor] = None
        if self.llama_4_scaling_config is not None:
            llama_4_scaling = _get_llama_4_scaling(
                original_max_position_embeddings=self.llama_4_scaling_config[
                    "original_max_position_embeddings"
                ],
                scaling_beta=self.llama_4_scaling_config["beta"],
                positions=positions,
            )

        normal_start_layer = self.start_layer
        normal_end_layer = self.end_layer
        if forward_batch.can_run_tbo:
            if (
                self.first_k_dense_replace > normal_start_layer
                and self.first_k_dense_replace < normal_end_layer
            ):
                normal_end_layer = self.first_k_dense_replace
            elif self.first_k_dense_replace < normal_start_layer:
                normal_end_layer = normal_start_layer = 0
        aux_hidden_states = []
        topk_indices = None
        for i in range(normal_start_layer, normal_end_layer):
            # NOTE: torch dynamo does not support graph break in context manager
            ctx = (
                nullcontext()
                if not get_global_server_args().disable_piecewise_cuda_graph
                else get_global_expert_distribution_recorder().with_current_layer(i)
            )
            with ctx:
                if i in self.layers_to_capture:
                    if self.enable_a2a_moe and i > self.first_k_dense_replace:
                        aux_hidden_state = get_attention_tp_group().all_gather(
                            hidden_states + residual, dim=0
                        )
                        aux_hidden_states.append(aux_hidden_state)
                    else:
                        aux_hidden_states.append(hidden_states + residual)
                layer = self.layers[i]
                hidden_states, residual, topk_indices = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                    zero_allocator,
                    gemm_output_zero_allocator,
                    llama_4_scaling,
                    prev_topk_indices=topk_indices,
                )
        if normal_end_layer != self.end_layer:
            hidden_states, residual = model_forward_maybe_tbo(
                layers=self.layers[normal_end_layer : self.end_layer],
                enable_tbo=True,
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                residual=residual,
                input_data_scatter_mode=self.layers[
                    normal_end_layer - 1
                ].layer_scatter_modes.layer_output_mode,
                zero_allocator=zero_allocator,
            )
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if not forward_batch.forward_mode.is_idle():
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)
        if self.pp_group.is_last_rank and nsa_use_prefill_cp(forward_batch):
            # allgather + rerrange
            hidden_states = cp_all_gather_rerange_output(
                hidden_states,
                self.cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )
        if len(aux_hidden_states) == 0:
            return hidden_states
        return hidden_states, aux_hidden_states


class DeepseekV2ForCausalLM(nn.Module, DeepseekV2WeightLoaderMixin):
    # for quark model load
    packed_modules_mapping = {}

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # for quark model load
        # Fuse q_a_proj and kv_a_proj_with_mqa along output dimension when q_lora_rank is not None
        self.fuse_qkv_a_proj = (
            hasattr(config, "q_lora_rank") and config.q_lora_rank is not None
        )
        if self.fuse_qkv_a_proj:
            self.packed_modules_mapping["fused_qkv_a_proj_with_mqa"] = [
                "q_a_proj",
                "kv_a_proj_with_mqa",
            ]

        self.pp_group = get_pp_group()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.determine_num_fused_shared_experts()
        self.use_nsa = is_deepseek_nsa(config)
        self.model = DeepseekV2Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )

        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                )
        else:
            # ranks other than the last rank will have a placeholder layer
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config)

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, DeepseekV2MoE)
            }
        )
        self.capture_aux_hidden_states = False

        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_rank = get_attention_cp_rank()
            self.cp_size = get_attention_cp_size()
        else:
            self.cp_rank = self.cp_size = None

        q_lora_rank = config.q_lora_rank if hasattr(config, "q_lora_rank") else None
        get_attn_tp_context().init_context(q_lora_rank, is_deepseek_nsa(config))

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    def determine_num_fused_shared_experts(
        self, architecture: str = "DeepseekV3ForCausalLM"
    ):
        self.num_fused_shared_experts = 0
        server_args = get_global_server_args()

        if server_args.disable_shared_experts_fusion:
            return

        disable_reason = None
        if server_args.enforce_shared_experts_fusion:
            pass
        elif is_sbo_enabled() or is_tbo_enabled():
            disable_reason = "SBO/TBO enabled: incompatible with fusing shared expert into MoE kernel."
        elif is_deepep_class_backend():
            disable_reason = "DeepEP: fusion off by default (use --enforce-shared-experts-fusion to enable)."
        elif (
            self.config.architectures[0] != architecture
            or self.config.n_routed_experts != 256
            or self.config.n_shared_experts != 1
        ):
            disable_reason = "Config does not support fused shared expert(s)."
        elif (
            (not _is_cuda or torch.cuda.get_device_capability("cuda") < (8, 0))
            and (not _is_hip or torch.cuda.get_device_capability("cuda") < (9, 4))
            and (not _is_musa or torch.musa.get_device_capability("musa") < (3, 1))
        ):
            disable_reason = (
                "Only Deepseek V3/R1 on NV-platform with capability >= 80 "
                "or AMD-platform with capability >= gfx942(MI30x) can use shared experts fusion optimization."
                "or MT-platform with capability >= 31 can use shared experts fusion optimization."
            )
        elif get_moe_expert_parallel_world_size() > 1 and (
            not _is_hip or torch.cuda.get_device_capability("cuda") < (9, 4)
        ):
            disable_reason = (
                "Only Deepseek V3/R1 on AMD-platform with capability >= gfx942(MI30x) "
                "can use shared experts fusion optimization under expert parallelism."
            )
        elif self.quant_config and self.quant_config.get_name() == "w4afp8":
            disable_reason = "Deepseek V3/R1 W4AFP8 model uses different quant method for routed experts and shared experts."

        if disable_reason is not None:
            server_args.disable_shared_experts_fusion = True
            self.num_fused_shared_experts = 0
            log_info_on_rank0(
                logger,
                f"{disable_reason} Shared experts fusion optimization is disabled.",
            )
            return

        self.num_fused_shared_experts = self.config.n_shared_experts

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if self.nsa_enable_prefill_cp:
            if can_nsa_cp_split(
                len(input_ids), self.cp_size, self.use_nsa, forward_batch
            ):
                forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(
                    len(input_ids),
                    self.cp_rank,
                    self.cp_size,
                    forward_batch.seq_lens_cpu.tolist(),
                )

        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden_states = self.model(
                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        else:
            return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        self.do_load_weights(weights, is_nextn)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=config.n_group,
        )

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        if layer_ids is None:
            self.capture_aux_hidden_states = True
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]
        else:
            self.capture_aux_hidden_states = True
            # we plus 1 here because in sglang, for the ith layer, it takes the output
            # of the (i-1)th layer as aux hidden state
            self.model.layers_to_capture = [val + 1 for val in layer_ids]

    def set_dflash_layers_to_capture(self, layer_ids: List[int]):
        if not self.pp_group.is_last_rank:
            return

        if layer_ids is None:
            raise ValueError(
                "DFLASH requires explicit layer_ids for aux hidden capture."
            )

        self.capture_aux_hidden_states = True
        self.model.layers_to_capture = [val + 1 for val in layer_ids]


class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    pass


class DeepseekV32ForCausalLM(DeepseekV2ForCausalLM):
    pass


@register_custom_op(
    op_name="flashinfer_dsv3_router_gemm",
    mutates_args=[],
    fake_impl=lambda logits, hidden_states, weight: None,
)
def flashinfer_dsv3_router_gemm(
    logits: torch.Tensor,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    _raw_dsv3_router_gemm(
        hidden_states,
        weight.t(),
        logits,
        launch_with_pdl=True,
    )


EntryClass = [DeepseekV2ForCausalLM, DeepseekV3ForCausalLM, DeepseekV32ForCausalLM]
