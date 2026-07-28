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

# Adapted from DeepSeek and Mixtral implementation
"""Inference-only MiniMax M2 model compatible with HuggingFace weights."""

import os
import logging
from contextlib import nullcontext
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn
from transformers import PretrainedConfig

from sglang.kernel_api_logging import debug_kernel_api
from sglang.kernels.ops.communication.all_reduce import (
    fused_parallel_qknorm,
    get_fused_parallel_qknorm_max_occupancy,
)
from sglang.srt.batch_overlap.two_batch_overlap import model_forward_maybe_tbo
from sglang.srt.distributed import (
    get_pp_group,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)
from sglang.srt.layers.dp_attention import (
    attn_tp_all_reduce,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    should_skip_post_experts_all_reduce,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_int8_marlin_for_minimax_m2 import (
    fused_experts_impl_int8_marlin_minimax_m2,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
    narrow_padded_param_and_loaded_weight,
)
from sglang.srt.runtime_context import get_forward, get_parallel, get_server_args

# get_bool_env_var is defined in sglang.srt.utils.common, not sglang.srt.distributed.
# Importing from the wrong module causes this file to fail import, which prevents the
# native MiniMaxM2ForCausalLM from registering in ModelRegistry. The fallback to the
# transformers wrapper then crashes on config.rope_parameters (transformers v5 issue).
# Other files (custom_all_reduce.py, hf_transformers_utils.py) also use sglang.srt.utils.
from sglang.srt.utils import (
    BumpAllocator,
    add_prefix,
    cpu_has_amx_support,
    get_bool_env_var,
    get_compiler_backend,
    is_cpu,
    is_cuda,
    is_hcu,
    is_non_idle_and_non_empty,
    is_npu,
    make_layers,
)
from sglang.srt.utils.custom_op import register_custom_op
from sglang.srt.utils.hf_transformers_utils import get_rope_config

logger = logging.getLogger(__name__)
_is_cpu = is_cpu()
_is_amx_available = cpu_has_amx_support()
_is_cuda = is_cuda()
_is_hcu = is_hcu()
_is_npu = is_npu()

_use_fused_rms_quant = get_bool_env_var("SGLANG_USE_FUSED_RMS_QUANT")

from lmslim.layers.gemm.int8_utils import per_token_quant_int8

if _is_hcu:
    from lightop.rmsnorm import fused_rms_norm_contiguous

if _is_npu:
    from sgl_kernel_npu.norm.split_qkv_tp_rmsnorm_rope import split_qkv_tp_rmsnorm_rope

_FP32GEMM_GATE = None


def _gate_fp32gemm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    global _FP32GEMM_GATE
    if _FP32GEMM_GATE is None:
        if os.environ.get("SGLANG_USE_FP32GEMM_GATE_CUSTOM", "0") == "1":
            try:
                import fp32gemm

                _FP32GEMM_GATE = fp32gemm
            except ImportError:
                _FP32GEMM_GATE = False
        else:
            _FP32GEMM_GATE = False

    if _FP32GEMM_GATE is not False and hidden_states.dtype == torch.bfloat16:
        return _FP32GEMM_GATE.gemm_abt(hidden_states, weight)

    return F.linear(hidden_states.float(), weight, bias=None)


@triton.jit
def rmsnorm_sumsq_kernel_serial(
    x1_ptr,  # T* [B, D]
    x2_ptr,  # T* [B, D]
    stride_x1,  # int
    stride_x2,  # int
    sum_sq_ptr,  # float* [B]
    B,  # int
    D1,  # int
    D2,  # int
    BLOCK_SIZE1: tl.constexpr,
    BLOCK_SIZE2: tl.constexpr,
):
    row_id = tl.program_id(0)
    x1_row = x1_ptr + row_id * stride_x1
    x2_row = x2_ptr + row_id * stride_x2

    offsets1 = tl.arange(0, BLOCK_SIZE1)
    mask1 = offsets1 < D1
    offsets2 = tl.arange(0, BLOCK_SIZE2)
    mask2 = offsets2 < D2

    x1 = tl.load(x1_row + offsets1, mask=mask1, other=0.0)
    x2 = tl.load(x2_row + offsets2, mask=mask2, other=0.0)

    x1_f32 = x1.to(tl.float32)
    sum_sq1 = tl.sum(x1_f32 * x1_f32, axis=0)

    x2_f32 = x2.to(tl.float32)
    sum_sq2 = tl.sum(x2_f32 * x2_f32, axis=0)

    tl.store(sum_sq_ptr + row_id, sum_sq1)
    tl.store(sum_sq_ptr + row_id + B, sum_sq2)


@triton.jit
def rmsnorm_apply_kernel_serial(
    x1_ptr,  # T* [B, D]
    x2_ptr,  # T* [B, D]
    w1_ptr,  # T* [D]
    w2_ptr,  # T* [D]
    sum_sq_ptr,  # float* [B]
    out1_ptr,  # T* [B, D]
    out2_ptr,  # T* [B, D]
    B,  # int
    D1,  # int
    D2,  # int
    stride_x1,  # int
    stride_x2,  # int
    tp_world,  # int
    eps,  # float
    BLOCK_SIZE1: tl.constexpr,
    BLOCK_SIZE2: tl.constexpr,
):
    row_id = tl.program_id(0)
    x1_row = x1_ptr + row_id * stride_x1
    x2_row = x2_ptr + row_id * stride_x2
    out1_row = out1_ptr + row_id * stride_x1
    out2_row = out2_ptr + row_id * stride_x2

    sum_sq1 = tl.load(sum_sq_ptr + row_id)
    sum_sq2 = tl.load(sum_sq_ptr + row_id + B)
    inv_rms1 = tl.rsqrt(sum_sq1 / D1 / tp_world + eps)
    inv_rms2 = tl.rsqrt(sum_sq2 / D2 / tp_world + eps)

    offsets1 = tl.arange(0, BLOCK_SIZE1)
    offsets2 = tl.arange(0, BLOCK_SIZE2)

    mask1 = offsets1 < D1
    mask2 = offsets2 < D2

    x1 = tl.load(x1_row + offsets1, mask=mask1, other=0.0)
    w1 = tl.load(w1_ptr + offsets1, mask=mask1, other=1.0)
    x2 = tl.load(x2_row + offsets2, mask=mask2, other=0.0)
    w2 = tl.load(w2_ptr + offsets2, mask=mask2, other=1.0)

    out1 = (x1.to(tl.float32) * inv_rms1 * w1.to(tl.float32)).to(x1.dtype)
    out2 = (x2.to(tl.float32) * inv_rms2 * w2.to(tl.float32)).to(x2.dtype)
    tl.store(out1_row + offsets1, out1, mask=mask1)
    tl.store(out2_row + offsets2, out2, mask=mask2)


@debug_kernel_api
def rms_sumsq_serial(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    assert x1.is_cuda and x2.is_cuda
    B, D1 = x1.shape
    B2, D2 = x2.shape
    assert B == B2

    stride_x1 = x1.stride(0)
    stride_x2 = x2.stride(0)

    # We found that custom all-reduce `sglang::cross_device_reduce_1stage`
    # is much faster than the nccl all-reduce in torch.
    # However, `should_custom_ar` checks if the reduced buffer is 16-byte aligned.
    # RMSNormTP reduces a [B, 2] fp32 tensor, so we pad the total element count to
    # satisfy the alignment requirement.
    B_padded = (B + B2 + 3) // 4 * 4

    sum_sq = torch.empty(B_padded, device=x1.device, dtype=torch.float32)

    BLOCK_SIZE1 = triton.next_power_of_2(D1)
    BLOCK_SIZE2 = triton.next_power_of_2(D2)

    grid = (B,)

    rmsnorm_sumsq_kernel_serial[grid](
        x1,
        x2,
        stride_x1,
        stride_x2,
        sum_sq,
        B,
        D1,
        D2,
        BLOCK_SIZE1,
        BLOCK_SIZE2,
    )
    return sum_sq


@debug_kernel_api
def rms_apply_serial(
    x1: torch.Tensor,
    x2: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    sum_sq: torch.Tensor,
    tp_world: int = 1,
    eps: float = 1e-5,
) -> torch.Tensor:
    assert x1.is_cuda and x2.is_cuda and w1.is_cuda and w2.is_cuda and sum_sq.is_cuda
    B, D1 = x1.shape
    B2, D2 = x2.shape
    assert B == B2

    stride_x1 = x1.stride(0)
    stride_x2 = x2.stride(0)
    out1 = torch.empty(B, D1, device=x1.device, dtype=x1.dtype)
    out2 = torch.empty(B, D2, device=x2.device, dtype=x2.dtype)

    BLOCK_SIZE1 = triton.next_power_of_2(D1)
    BLOCK_SIZE2 = triton.next_power_of_2(D2)

    grid = (B,)

    rmsnorm_apply_kernel_serial[grid](
        x1,
        x2,
        w1,
        w2,
        sum_sq,
        out1,
        out2,
        B,
        D1,
        D2,
        stride_x1,
        stride_x2,
        tp_world,
        eps,
        BLOCK_SIZE1,
        BLOCK_SIZE2,
    )
    return out1, out2


class MiniMaxM2RMSNormTP(nn.Module):
    """RMSNorm with Tensor Parallel support for QK normalization."""

    def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.minimax_opt = get_server_args().minimax_opt
        self.attn_tp_size = get_parallel().attn_tp_size
        self.attn_tp_rank = get_parallel().attn_tp_rank

        # Align with QKVParallelLinear pattern
        if self.minimax_opt:
            self.num_heads = num_heads
            self.num_head_replicas = 1
        elif self.attn_tp_size >= num_heads:
            assert self.attn_tp_size % num_heads == 0, (
                f"attn_tp_size ({self.attn_tp_size}) must be divisible by num_heads ({num_heads})"
            )
            self.num_heads = 1
            self.num_head_replicas = self.attn_tp_size // num_heads
        else:
            assert num_heads % self.attn_tp_size == 0, (
                f"num_heads ({num_heads}) must be divisible by attn_tp_size ({self.attn_tp_size})"
            )
            self.num_heads = num_heads // self.attn_tp_size
            self.num_head_replicas = 1

        self.head_dim = hidden_size // num_heads

        # Weight parameter is sharded across TP ranks
        self.weight = nn.Parameter(torch.ones(self.num_heads * self.head_dim))
        self.weight.weight_loader = self.weight_loader
        self.variance_epsilon = eps

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        """Custom weight loader that handles TP sharding."""
        if self.minimax_opt:
            param.data.copy_(loaded_weight)
            return

        shard_id = self.attn_tp_rank // self.num_head_replicas
        shard_size = param.data.shape[0]

        if _is_cpu and _is_amx_available:
            # Handle uneven TP sharding on CPU
            param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                param.data,
                loaded_weight,
                0,  # param_data_start
                shard_id * shard_size,  # weight_start
                0,  # shard_axis
                shard_size,
            )
            param_data.copy_(loaded_weight)
            return

        shard_end = (shard_id + 1) * shard_size
        assert shard_end <= loaded_weight.shape[0], (
            f"Weight shard out of bounds: shard [{shard_id * shard_size}:{shard_end}] "
            f"exceeds loaded_weight size {loaded_weight.shape[0]} "
            f"(attn_tp_rank={self.attn_tp_rank}, num_head_replicas={self.num_head_replicas})"
        )
        shard = slice(shard_id * shard_size, shard_end)
        param.data.copy_(loaded_weight[shard])

    @torch.compile(dynamic=True, backend=get_compiler_backend())
    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with TP-aware variance computation."""
        assert residual is None, "RMSNormTP does not support residual connection."

        orig_dtype = x.dtype
        x = x.to(torch.float32)

        # Compute variance across the full dimension (not just local shard)
        variance = x.pow(2).mean(dim=-1, keepdim=True, dtype=torch.float32)

        if self.attn_tp_size > 1:
            # All-reduce variance across TP ranks to get global variance
            variance = attn_tp_all_reduce(variance) / self.attn_tp_size

        # Normalize and apply local weight shard
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = (x * self.weight).to(orig_dtype)

        return x


@register_custom_op(mutates_args=["q", "k"])
def fused_tp_qknorm(
    counter: int,
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
) -> None:
    return fused_parallel_qknorm(
        MiniMaxM2QKRMSNorm.COMM_MAP[counter].obj,
        q,
        k,
        q_weight,
        k_weight,
        eps=eps,
    )


class MiniMaxM2QKRMSNorm:
    COUNTER = 0
    COMM_MAP: Dict[int, Any] = {}

    def __init__(
        self,
        q_norm: MiniMaxM2RMSNormTP,
        k_norm: MiniMaxM2RMSNormTP,
    ) -> None:
        assert q_norm.variance_epsilon == k_norm.variance_epsilon
        self._q_norm = q_norm
        self._k_norm = k_norm
        self._world_size = self._q_norm.attn_tp_size
        self._eps = q_norm.variance_epsilon
        use_fused_norm = get_bool_env_var("SGLANG_USE_FUSED_PARALLEL_QKNORM")

        self._forward_impl = self._forward_naive
        if (
            not self._q_norm.minimax_opt
            and self._world_size > 1
            and _is_cuda
            and use_fused_norm
        ):
            occupancy = get_fused_parallel_qknorm_max_occupancy(
                q_norm.weight.dtype,
                self._world_size,
                # NOTE: we need full dimension
                q_dim=q_norm.weight.shape[0] * self._world_size,
                k_dim=k_norm.weight.shape[0] * self._world_size,
            )
            counter = MiniMaxM2QKRMSNorm._get_comm(q_norm.weight.device, occupancy)
            if counter is not None:
                self._counter = counter
                self._forward_impl = self._forward_fused
        elif _is_cpu and _is_amx_available:
            self._forward_impl = self._forward_cpu

    @lru_cache
    @staticmethod
    def _get_comm(device: torch.device, occupancy: int):
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            CustomAllReduceV2,
        )

        props = torch.cuda.get_device_properties(device)
        # probe the maximum tokens for one prefill
        server_args = get_server_args()
        max_tokens = server_args.chunked_prefill_size
        if max_tokens is None:
            max_tokens = server_args.model_config.context_len
        max_tokens = max(max_tokens, server_args.max_prefill_tokens)
        logger.info(f"[AR] Using CustomAllReduceV2 for MiniMaxM2 with {max_tokens = }")
        ALIGN = 512
        # typically, this should not exceed 1M, since max_tokens is usually less than 16384
        max_size = ((8 * max_tokens + ALIGN - 1) // ALIGN) * ALIGN
        comm = CustomAllReduceV2(
            group=get_parallel().attn_tp_group.cpu_group,
            device=device,
            max_pull_size=0,
            max_pull_blocks=0,
            max_push_size=max_size,
            max_push_blocks=props.multi_processor_count * occupancy,
        )
        counter = MiniMaxM2QKRMSNorm.COUNTER
        MiniMaxM2QKRMSNorm.COUNTER += 1
        MiniMaxM2QKRMSNorm.COMM_MAP[counter] = comm
        return counter if not comm.disabled else None

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        return self._forward_impl(q, k)

    def forward_opt(self, q: torch.Tensor, k: torch.Tensor):
        q, k = q.contiguous(), k.contiguous()
        if _is_hcu:
            q = fused_rms_norm_contiguous(
                torch.empty_like(q), q, self._q_norm.weight, self._eps
            )
            k = fused_rms_norm_contiguous(
                torch.empty_like(k), k, self._k_norm.weight, self._eps
            )
            return q, k

        orig_dtype = q.dtype
        q_f = q.float()
        k_f = k.float()
        q = (
            q_f
            * torch.rsqrt(q_f.pow(2).mean(dim=-1, keepdim=True) + self._eps)
            * self._q_norm.weight
        ).to(orig_dtype)
        k = (
            k_f
            * torch.rsqrt(k_f.pow(2).mean(dim=-1, keepdim=True) + self._eps)
            * self._k_norm.weight
        ).to(orig_dtype)
        return q, k

    def _forward_naive(self, q: torch.Tensor, k: torch.Tensor):
        q, k = q.contiguous(), k.contiguous()
        sum_sq = rms_sumsq_serial(q, k)
        if self._world_size > 1:
            sum_sq = attn_tp_all_reduce(sum_sq)
        return rms_apply_serial(
            q,
            k,
            self._q_norm.weight,
            self._k_norm.weight,
            sum_sq,
            self._world_size,
            self._eps,
        )

    def _forward_fused(self, q: torch.Tensor, k: torch.Tensor):
        fused_tp_qknorm(
            self._counter,
            q,
            k,
            self._q_norm.weight,
            self._k_norm.weight,
            self._eps,
        )
        return q, k

    def _forward_cpu(self, q: torch.Tensor, k: torch.Tensor):
        # TODO: add c++ kernel for cpu
        q = self._q_norm(q.contiguous())
        k = self._k_norm(k.contiguous())
        return q, k


class MiniMaxM2MoE(nn.Module):
    """MiniMax MoE implementation using DeepEP for Expert Parallel support."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_parallel().tp_size
        if self.tp_size > config.num_local_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_local_experts}."
            )
        self.use_routing_bias = getattr(config, "use_routing_bias", False)
        if self.use_routing_bias:
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(config.num_local_experts, dtype=torch.float32)
            )
            self.e_score_correction_bias.weight_loader = (
                MiniMaxM2MoE.ebias_weight_loader
            )
        else:
            self.e_score_correction_bias = None

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.num_local_experts
            + get_server_args().ep_num_redundant_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
        )
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=True,
            scoring_func=config.scoring_func,
            correction_bias=self.e_score_correction_bias,
            routed_scaling_factor=1.0,
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )

        self.layer_id = layer_id

        if get_moe_a2a_backend().is_deepep():
            self.ep_size = get_parallel().moe_ep_size
            self.top_k = config.num_experts_per_tok

    @staticmethod
    def ebias_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight.to(torch.float32))

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> torch.Tensor:
        if (
            not get_moe_a2a_backend().is_deepep()
            and not get_moe_a2a_backend().is_ascend_fuseep()
        ):
            if get_server_args().minimax_opt:
                return self.forward_normal_opt(hidden_states)
            return self.forward_normal(hidden_states)
        else:
            return self.forward_deepep(hidden_states, forward_batch)

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if hidden_states.shape[0] > 0:
            # router_logits: (num_tokens, n_experts)
            router_logits = _gate_fp32gemm(hidden_states, self.gate.weight)
            topk_output = self.topk(hidden_states, router_logits)
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        final_hidden_states = self.experts(hidden_states, topk_output)
        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)

    def forward_normal_opt(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        local_tokens, hidden_dim = hidden_states.shape
        num_tokens = local_tokens * self.tp_size
        adtype = hidden_states.dtype
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = _gate_fp32gemm(hidden_states, self.gate.weight)
        topk_output = self.topk(hidden_states, router_logits)
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids

        tp_group = get_tp_group()
        i_q, i_s = per_token_quant_int8(hidden_states)

        i_q = tp_group.all_gather(i_q, dim=0)
        i_s = tp_group.all_gather(i_s, dim=0)
        topk_weights = tp_group.all_gather(topk_weights, dim=0)
        topk_ids = tp_group.all_gather(topk_ids, dim=0)

        final_hidden_states = fused_experts_impl_int8_marlin_minimax_m2(
            num_tokens,
            hidden_dim,
            hidden_states.device,
            adtype,
            self.experts.w13_weight,
            self.experts.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            activation="silu",
            apply_router_weight_on_input=False,
            use_int8_w8a8=True,
            per_channel_quant=True,
            global_num_experts=self.experts.num_experts,
            w1_scale=self.experts.w13_weight_scale,
            w2_scale=self.experts.w2_weight_scale,
            a1_scale=self.experts.w13_input_scale,
            a2_scale=self.experts.w2_input_scale,
            i_q=i_q,
            i_s=i_s,
            routed_scaling_factor=1.0,
        )
        if self.tp_size > 1:
            chunks = final_hidden_states.chunk(self.tp_size)
            reduced = torch.empty(
                num_tokens // self.tp_size,
                hidden_dim,
                device=final_hidden_states.device,
                dtype=final_hidden_states.dtype,
            )
            tp_group.reduce_scatter(reduced, list(chunks))
            return reduced

        return final_hidden_states.view(num_tokens, hidden_dim)

    def forward_deepep(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        if hidden_states.shape[0] > 0:
            # router_logits: (num_tokens, n_experts)
            router_logits = _gate_fp32gemm(hidden_states, self.gate.weight)
            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
        else:
            topk_output = self.topk.empty_topk_output(device=hidden_states.device)
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

        return final_hidden_states

    # TBO Operations for MiniMax MoE
    def op_gate(self, state):
        """Gate operation for TBO - compute router logits"""
        if is_non_idle_and_non_empty(
            state.forward_batch.forward_mode, state.hidden_states_mlp_input
        ):  # router_logits: (num_tokens, num_experts)
            state.router_logits, _ = self.gate(state.hidden_states_mlp_input)
            # state.router_logits = _gate_fp32gemm(
            #     state.hidden_states_mlp_input, self.gate.weight
            # )
        else:
            state.router_logits = None

    def op_select_experts(self, state):
        """Expert selection operation for TBO"""
        router_logits = state.pop("router_logits")
        hidden_states = state.hidden_states_mlp_input

        if router_logits is not None:
            ctx = (
                nullcontext()
                if check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
                else get_global_expert_distribution_recorder().with_current_layer(
                    self.layer_id
                )
            )
            with ctx:
                state.topk_weights_local, state.topk_idx_local, _ = self.topk(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    num_token_non_padded=state.forward_batch.num_token_non_padded,
                    expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                        layer_id=self.layer_id,
                    ),
                )
        else:
            state.topk_idx_local = torch.full(
                (0, self.top_k), -1, dtype=torch.int, device=hidden_states.device
            )
            state.topk_weights_local = torch.empty(
                (0, self.top_k), dtype=torch.float32, device=hidden_states.device
            )

    def op_dispatch_a(self, state):
        """Dispatch A operation for TBO - start async dispatch"""
        if self.ep_size > 1:
            self.experts.deepep_dispatcher.dispatch_a(
                hidden_states=state.pop("hidden_states_mlp_input"),
                topk_idx=state.pop("topk_idx_local"),
                topk_weights=state.pop("topk_weights_local"),
                forward_batch=state.forward_batch,
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_dispatch_b(self, state):
        """Dispatch B operation for TBO - complete async dispatch"""
        if self.ep_size > 1:
            ctx = (
                nullcontext()
                if check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
                else get_global_expert_distribution_recorder().with_current_layer(
                    self.layer_id
                )
            )
            with ctx:
                state.dispatch_output = self.experts.deepep_dispatcher.dispatch_b(
                    tbo_subbatch_index=state.get("tbo_subbatch_index"),
                )

    def op_experts(self, state):
        """Expert computation for TBO"""
        state.hidden_states_experts_output = self.experts.moe_impl(
            dispatch_output=state.dispatch_output,
        )

    def op_combine_a(self, state):
        """Combine A operation for TBO - start async combine"""
        if self.ep_size > 1:
            self.experts.deepep_dispatcher.combine_a(
                hidden_states=state.pop("hidden_states_experts_output"),
                topk_idx=state.dispatch_output.topk_idx,
                topk_weights=state.dispatch_output.topk_weights,
                forward_batch=state.forward_batch,
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )
            state.pop("dispatch_output")

    def op_combine_b(self, state):
        """Combine B operation for TBO - complete async combine"""
        if self.ep_size > 1:
            state.hidden_states_after_combine = (
                self.experts.deepep_dispatcher.combine_b(
                    tbo_subbatch_index=state.get("tbo_subbatch_index"),
                )
            )

    def op_output(self, state):
        """Output operation for TBO - final MLP output"""
        final_hidden_states = state.pop("hidden_states_after_combine")
        # MiniMax doesn't have shared experts like DeepSeek, so no need to add them
        state.hidden_states_mlp_output = final_hidden_states


class MiniMaxM2Attention(nn.Module):
    """MiniMax Attention implementation with QK normalization and partial RoPE."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        # Use attention TP rank/size for dp-attention support
        attn_tp_rank = get_parallel().attn_tp_rank
        attn_tp_size = get_parallel().attn_tp_size

        # Get dimensions from config
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = config.num_key_value_heads

        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)

        # Use head_dim from config if available, otherwise calculate
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.q_size_full = self.total_num_heads * self.head_dim
        self.kv_size_full = self.total_num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # RoPE settings - support partial RoPE
        # FIXME: minimax_m2 config use external config that not compatible with transformers v5
        self.rope_theta, self.rope_scaling = get_rope_config(config)
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.rotary_dim = getattr(
            config, "rotary_dim", self.head_dim
        )  # MiniMax uses rotary_dim=64

        # QK Normalization settings
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        self.qk_norm_type = getattr(config, "qk_norm_type", "per_layer")

        if get_server_args().minimax_opt:
            self.qkv_proj = QKVParallelLinear(
                self.hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=False,
                quant_config=quant_config,
                tp_rank=0,
                tp_size=1,
                prefix=add_prefix("qkv_proj", prefix),
            )
            self.o_proj = ReplicatedLinear(
                self.total_num_heads * self.head_dim,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("o_proj", prefix),
            )
        else:
            self.qkv_proj = QKVParallelLinear(
                self.hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=False,
                quant_config=quant_config,
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
                prefix=add_prefix("qkv_proj", prefix),
            )
            self.o_proj = RowParallelLinear(
                self.total_num_heads * self.head_dim,
                self.hidden_size,
                bias=False,
                reduce_results=False,
                quant_config=quant_config,
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
                prefix=add_prefix("o_proj", prefix),
            )

        # Setup RoPE with partial rotary dimension
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,  # Use partial rotary dimension
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=self.rope_scaling,
        )

        # QK Normalization layers
        if self.use_qk_norm:
            if self.qk_norm_type == "per_layer":
                # Use RMSNormTP for proper tensor parallel support
                # Use total dimensions (before TP sharding) for correct normalization
                self.q_norm = MiniMaxM2RMSNormTP(
                    self.total_num_heads * self.head_dim,
                    num_heads=self.total_num_heads,
                    eps=config.rms_norm_eps,
                )
                self.k_norm = MiniMaxM2RMSNormTP(
                    self.total_num_kv_heads * self.head_dim,
                    num_heads=self.total_num_kv_heads,
                    eps=config.rms_norm_eps,
                )
                self.qk_norm_impl = MiniMaxM2QKRMSNorm(self.q_norm, self.k_norm)
            else:
                raise ValueError(f"Unsupported qk_norm_type: {self.qk_norm_type}")

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        rms_weight: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
    ):
        if hidden_states.shape[0] == 0:
            assert not self.o_proj.reduce_results, (
                "short-circuiting allreduce will lead to hangs"
            )
            return hidden_states, forward_batch, None
        qkv, _ = self.qkv_proj(hidden_states, rms_weight=rms_weight, residual=residual)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.use_qk_norm:
            q, k = self.qk_norm_impl.forward(q, k)
        q, k = self.rotary_emb(positions, q, k)
        inner_state = q, k, v, forward_batch
        return None, forward_batch, inner_state

    def forward_prepare_opt(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        rms_weight: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
    ):
        if hidden_states.shape[0] == 0:
            return hidden_states, forward_batch, None

        attn_tp_group = get_parallel().attn_tp_group
        attn_tp_size = attn_tp_group.world_size
        tokens_per_rank = hidden_states.shape[0]

        qkv, _ = self.qkv_proj(hidden_states, rms_weight=rms_weight, residual=residual)
        q, k, v = qkv.split(
            [self.q_size_full, self.kv_size_full, self.kv_size_full], dim=-1
        )
        if self.use_qk_norm:
            q, k = q.contiguous(), k.contiguous()
            q, k = self.qk_norm_impl.forward_opt(q, k)
        q, k = self.rotary_emb.forward_native(positions, q, k)

        q = q.view(tokens_per_rank, attn_tp_size, self.q_size)
        if self.total_num_kv_heads >= attn_tp_size:
            k = k.view(tokens_per_rank, attn_tp_size, self.kv_size)
            v = v.view(tokens_per_rank, attn_tp_size, self.kv_size)
        else:
            kv_replicas = attn_tp_size // self.total_num_kv_heads
            k = k.view(tokens_per_rank, self.total_num_kv_heads, self.head_dim)
            v = v.view(tokens_per_rank, self.total_num_kv_heads, self.head_dim)
            k = k.repeat_interleave(kv_replicas, dim=1)
            v = v.repeat_interleave(kv_replicas, dim=1)
        qkv_by_head_rank = torch.cat([q, k, v], dim=-1).permute(1, 0, 2).contiguous()

        qkv_restored = torch.empty_like(qkv_by_head_rank)
        attn_tp_group.all_to_all_single(qkv_restored, qkv_by_head_rank)
        qkv_restored = qkv_restored.view(
            tokens_per_rank * attn_tp_size, self.q_size + 2 * self.kv_size
        )
        q, k, v = qkv_restored.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        inner_state = q, k, v, forward_batch
        return None, forward_batch, inner_state

    def forward_prepare_npu(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if hidden_states.shape[0] == 0:
            assert not self.o_proj.reduce_results, (
                "short-circuiting allreduce will lead to hangs"
            )
            return hidden_states, forward_batch, None
        qkv, _ = self.qkv_proj(hidden_states)
        if self.use_qk_norm:
            cos_sin = self.rotary_emb.cos_sin_cache.index_select(0, positions.flatten())
            cos, sin = cos_sin.chunk(2, dim=-1)
            q, k, v = split_qkv_tp_rmsnorm_rope(
                input=qkv,
                cos=cos,
                sin=sin,
                q_weight=self.q_norm.weight,
                k_weight=self.k_norm.weight,
                q_hidden_size=self.q_size,
                kv_hidden_size=self.kv_size,
                head_dim=self.head_dim,
                rotary_dim=self.rotary_dim,
                eps=self.q_norm.variance_epsilon,
                tp_world=self.q_norm.attn_tp_size,
                tp_group=get_parallel().attn_tp_group.device_group,
            )
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = q.contiguous(), k.contiguous()
            q, k = self.rotary_emb(positions, q, k)

        inner_state = q, k, v, forward_batch
        return None, forward_batch, inner_state

    def forward_core(self, intermediate_state):
        hidden_states, forward_batch, inner_state = intermediate_state
        if inner_state is None:
            return hidden_states
        attn_output = self.attn(*inner_state)
        output, _ = self.o_proj(attn_output)
        return output

    def forward_core_opt(self, intermediate_state):
        hidden_states, forward_batch, inner_state = intermediate_state
        if inner_state is None:
            return hidden_states

        attn_output = self.attn(*inner_state)
        attn_tp_group = get_parallel().attn_tp_group
        attn_tp_size = attn_tp_group.world_size
        tokens_per_rank = attn_output.shape[0] // attn_tp_size

        attn_output = attn_output.view(attn_tp_size, tokens_per_rank, self.q_size)
        attn_output_restored = torch.empty_like(attn_output)
        attn_tp_group.all_to_all_single(attn_output_restored, attn_output)
        attn_output_restored = (
            attn_output_restored.permute(1, 0, 2)
            .contiguous()
            .view(tokens_per_rank, attn_tp_size * self.q_size)
        )

        output, _ = self.o_proj(attn_output_restored)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        rms_weight: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if get_server_args().minimax_opt:
            s = self.forward_prepare_opt(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                rms_weight=rms_weight,
                residual=residual,
            )
            return self.forward_core_opt(s)

        if not _is_npu:
            s = self.forward_prepare(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                rms_weight=rms_weight,
                residual=residual,
            )
        else:
            s = self.forward_prepare_npu(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        return self.forward_core(s)

    def op_prepare(self, state):
        if get_server_args().minimax_opt:
            state.attn_intermediate_state = self.forward_prepare_opt(
                positions=state.positions,
                hidden_states=state.pop("hidden_states_after_comm_pre_attn"),
                forward_batch=state.forward_batch,
            )
        else:
            state.attn_intermediate_state = self.forward_prepare(
                positions=state.positions,
                hidden_states=state.pop("hidden_states_after_comm_pre_attn"),
                forward_batch=state.forward_batch,
            )

    def op_core(self, state):
        state.hidden_states_after_attn = self.forward_core(
            state.pop("attn_intermediate_state")
        )


class MiniMaxM2DecoderLayer(nn.Module):
    """MiniMax Decoder Layer implementation with MoE support."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id

        # TBO support: All MiniMax layers are sparse (MoE)
        self.is_layer_sparse = True

        self.self_attn = MiniMaxM2Attention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        self.block_sparse_moe = MiniMaxM2MoE(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("block_sparse_moe", prefix),
        )

        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6)
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6)
        )

        is_previous_layer_sparse = True
        is_next_layer_sparse = True
        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(layer_id == config.num_hidden_layers - 1),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        captured_last_layer_outputs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if get_server_args().minimax_opt:
            return self.forward_opt(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
            )

        if _use_fused_rms_quant:
            pre_norm = hidden_states.clone() if residual is None else residual
            hidden_states, _ = self.layer_communicator.prepare_attn(
                hidden_states, residual, forward_batch, skip_layernorm=True
            )
            if residual is not None:
                assert residual.shape[0] == hidden_states.shape[0], (
                    f"Fused RMS quant requires hidden_states ({hidden_states.shape[0]}) "
                    f"and residual ({residual.shape[0]}) to have the same token count. "
                    f"Set SGLANG_USE_FUSED_RMS_QUANT=0 or disable --moe-a2a-backend."
                )
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                rms_weight=self.input_layernorm.weight,
                residual=residual,
            )
            hidden_states, residual = self.layer_communicator.prepare_mlp(
                hidden_states, pre_norm, forward_batch
            )
        else:
            # Self Attention
            hidden_states, residual = (
                self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                    hidden_states,
                    residual,
                    forward_batch,
                    captured_last_layer_outputs=captured_last_layer_outputs,
                )
            )
            if not forward_batch.forward_mode.is_idle():
                hidden_states = self.self_attn(
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                )

            # Fully Connected (MLP or MoE)

            hidden_states, residual = self.layer_communicator.prepare_mlp(
                hidden_states, residual, forward_batch
            )

        fuse_mlp_allreduce = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )

        mlp_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        with get_forward().scoped(
            fuse_mlp_allreduce=fuse_mlp_allreduce,
            mlp_reduce_scatter=mlp_reduce_scatter,
        ):
            hidden_states = self.block_sparse_moe(hidden_states, forward_batch)

        if fuse_mlp_allreduce:
            hidden_states._sglang_needs_allreduce_fusion = True
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        return hidden_states, residual

    def forward_opt(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states, hidden_states

        if _use_fused_rms_quant:
            pre_norm = hidden_states.clone() if residual is None else residual
            if residual is not None:
                assert residual.shape[0] == hidden_states.shape[0], (
                    f"Fused RMS quant requires hidden_states ({hidden_states.shape[0]}) "
                    f"and residual ({residual.shape[0]}) to have the same token count."
                )
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                rms_weight=self.input_layernorm.weight,
                residual=residual,
            )
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, pre_norm
            )
        else:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(
                    hidden_states, residual, None
                )

            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
        hidden_states = self.block_sparse_moe(hidden_states, forward_batch)
        return hidden_states, residual

    # TBO Operations for MiniMax Decoder Layer
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
        """Communication prepare for attention - TBO operation"""
        state.hidden_states_after_comm_pre_attn, state.residual_after_input_ln = (
            self.layer_communicator.prepare_attn(hidden_states, residual, forward_batch)
        )
        state.update(
            dict(
                forward_batch=forward_batch,
                positions=positions,
                zero_allocator=zero_allocator,
                tbo_subbatch_index=tbo_subbatch_index,
            )
        )

    def op_comm_prepare_mlp(self, state):
        """Communication prepare for MLP - TBO operation"""
        state.hidden_states_mlp_input, state.residual_after_comm_pre_mlp = (
            self.layer_communicator.prepare_mlp(
                state.pop("hidden_states_after_attn"),
                state.pop("residual_after_input_ln"),
                state.forward_batch,
            )
        )

    def op_comm_postprocess_layer(self, state):
        """Communication postprocess for layer - TBO operation"""
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
        return output


class MiniMaxM2Model(nn.Module):
    """MiniMax Model implementation."""

    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.padding_idx = getattr(config, "pad_token_id", 0)
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                use_attn_tp_group=is_dp_attention_enabled(),
            )

        def layer_fn(idx, prefix: str) -> nn.Module:
            return MiniMaxM2DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            layer_fn,
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        # For EAGLE3 support
        self.layers_to_capture = []

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors, Tuple[torch.Tensor, list[torch.Tensor]]]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.get_input_embeddings(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        use_minimax_sp = get_server_args().minimax_opt
        if use_minimax_sp:
            attn_tp_group = get_parallel().attn_tp_group
            attn_tp_size = attn_tp_group.world_size
            attn_tp_rank = attn_tp_group.rank_in_group
            num_tokens = hidden_states.shape[0]
            tokens_per_rank = num_tokens // attn_tp_size
            start = attn_tp_rank * tokens_per_rank
            hidden_states = hidden_states[start : start + tokens_per_rank]
            positions = positions[start : start + tokens_per_rank]
            if residual is not None:
                residual = residual[start : start + tokens_per_rank]

        aux_hidden_states = []
        if forward_batch.can_run_tbo and not get_server_args().minimax_opt:
            hidden_states, residual = model_forward_maybe_tbo(
                layers=self.layers,
                enable_tbo=True,
                input_data_scatter_mode=ScatterMode.model_input_output(),
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                residual=residual,
            )
        else:
            for i in range(self.start_layer, self.end_layer):
                ctx = (
                    nullcontext()
                    if check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
                    else get_global_expert_distribution_recorder().with_current_layer(i)
                )
                with ctx:
                    layer = self.layers[i]
                    hidden_states, residual = layer(
                        positions=positions,
                        forward_batch=forward_batch,
                        hidden_states=hidden_states,
                        residual=residual,
                        captured_last_layer_outputs=(
                            aux_hidden_states if i in self.layers_to_capture else None
                        ),
                    )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if hidden_states.shape[0] != 0:
            if residual is not None:
                hidden_states, _ = self.norm(hidden_states, residual)
            else:
                hidden_states = self.norm(hidden_states)

        if use_minimax_sp:
            attn_tp_group = get_parallel().attn_tp_group
            attn_tp_size = attn_tp_group.world_size
            hidden_states_gathered = torch.empty(
                hidden_states.shape[0] * attn_tp_size,
                hidden_states.shape[1],
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            attn_tp_group.all_gather_into_tensor(hidden_states_gathered, hidden_states)
            hidden_states = hidden_states_gathered

        if len(aux_hidden_states) == 0:
            return hidden_states
        return hidden_states, aux_hidden_states


class MiniMaxM2ForCausalLM(nn.Module):
    """MiniMax M2 model for causal language modeling."""

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        self.model = MiniMaxM2Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )

        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=None,
                prefix=add_prefix("lm_head", prefix),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config)

        # For EAGLE3
        self.capture_aux_hidden_states = False

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    @property
    def start_layer(self) -> int:
        return self.model.start_layer

    @property
    def end_layer(self) -> int:
        return self.model.end_layer

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[list[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load model weights with proper mapping for MiniMax architecture."""

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model

            _is_kv_scale = name.endswith(".k_scale") or name.endswith(".v_scale")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # Skip kv cache scales - maybe_remap_kv_scale_name expects the
                # original checkpoint name (e.g. self_attn.k_proj.k_scale) to
                # remap it to self_attn.attn.k_scale. Renaming k_proj -> qkv_proj
                # here would break that pattern match.
                if _is_kv_scale:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name not in params_dict:
                    continue

                if name.endswith(".bias"):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation

        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_local_experts,
            num_groups=None,
        )


def get_spec_layer_idx_from_weight_name(
    config: PretrainedConfig, weight_name: str
) -> Optional[int]:
    if hasattr(config, "num_mtp_modules") and (config.num_mtp_modules > 0):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_mtp_modules):
            if weight_name.startswith(f"model.layers.{layer_idx + i}."):
                return layer_idx + i
    return None


# Entry class for model registration
EntryClass = MiniMaxM2ForCausalLM
