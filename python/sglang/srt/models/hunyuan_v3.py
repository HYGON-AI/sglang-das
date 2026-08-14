# coding=utf-8
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

# Copyright 2026 The HunYuan team.
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

from typing import Iterable, Optional, Tuple

import torch
from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    get_moe_tensor_parallel_world_size,
    get_pp_group,
    get_pp_indices,
    get_tensor_model_parallel_world_size,
    moe_expert_parallel_all_reduce,
    moe_tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    enable_moe_dense_fully_dp,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
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
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import filter_moe_weight_param_global_expert
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import ForwardBatch
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import get_bool_env_var, is_cuda, is_hcu, is_hip, make_layers
from sglang.srt.utils.common import LazyValue
from sglang.srt.utils.hf_transformers_utils import get_rope_config
from torch import nn
from transformers import PretrainedConfig

_is_hcu = is_hcu()
_use_fused_hunyuan_rotary = get_bool_env_var("SGLANG_USE_FUSED_RMS_ROTARY")
# Same env as DeepSeek: aiter MoE applies routed_scaling_factor internally.
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()

if _is_hcu:
    from lightop.attention import rms_rotary_embedding_fuse_with_kv_store


def _pack_hy3_sp_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tp_size: int,
    q_size: int,
    kv_size: int,
    total_num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Pack sequence-sharded full QKV into equal head shards for all-to-all."""
    tokens_per_rank = q.shape[0]
    q = q.view(tokens_per_rank, tp_size, q_size)
    if total_num_kv_heads >= tp_size:
        k = k.view(tokens_per_rank, tp_size, kv_size)
        v = v.view(tokens_per_rank, tp_size, kv_size)
    else:
        kv_replicas = tp_size // total_num_kv_heads
        k = k.view(tokens_per_rank, total_num_kv_heads, head_dim)
        v = v.view(tokens_per_rank, total_num_kv_heads, head_dim)
        k = k.repeat_interleave(kv_replicas, dim=1)
        v = v.repeat_interleave(kv_replicas, dim=1)
    return torch.cat([q, k, v], dim=-1).permute(1, 0, 2).contiguous()


def _merge_hy3_sp_attention_output(
    attn_output_by_source_rank: torch.Tensor,
) -> torch.Tensor:
    """Merge all-to-all output back into a local sequence shard with all Q heads."""
    return (
        attn_output_by_source_rank.permute(1, 0, 2)
        .contiguous()
        .flatten(start_dim=1)
    )


def _apply_hy3_qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply HY3's shared per-head Q/K RMSNorm and preserve packed widths."""
    q_shape = q.shape
    k_shape = k.shape
    q = q_norm(q.reshape(-1, head_dim)).view(q_shape)
    k = k_norm(k.reshape(-1, head_dim)).view(k_shape)
    return q, k


def _reshape_hy3_sp_attention_output(
    attn_output: torch.Tensor,
    tp_size: int,
    q_size: int,
) -> torch.Tensor:
    """Split global-sequence attention output by its source sequence rank."""
    if attn_output.ndim != 2 or attn_output.shape[1] != q_size:
        raise ValueError(
            "HY3 SP attention output must have shape [num_tokens, q_size], "
            f"got {tuple(attn_output.shape)} with q_size={q_size}."
        )
    if attn_output.shape[0] % tp_size != 0:
        raise ValueError(
            "HY3 SP attention output token count must be divisible by the "
            f"attention TP size, got {attn_output.shape[0]} and {tp_size}."
        )
    tokens_per_rank = attn_output.shape[0] // tp_size
    return attn_output.view(tp_size, tokens_per_rank, q_size)


class HYV3FeedForward(nn.Module):
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
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ):
        skip_all_reduce = should_allreduce_fusion or use_reduce_scatter
        gate_up, _ = self.gate_up_proj(x)
        out = self.act_fn(gate_up)
        out, _ = self.down_proj(out, skip_all_reduce=skip_all_reduce)
        return out


class HYV3MoEFused(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.tp_size = get_moe_tensor_parallel_world_size()
        self.dense_tp_size = get_tensor_model_parallel_world_size()
        self.ep_size = get_moe_expert_parallel_world_size()
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.n_routed_experts = config.num_experts
        top_k = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size

        self.expert_bias = nn.Parameter(
            torch.empty(config.num_experts, dtype=torch.float32)
        )
        self.expert_bias.weight_loader = HYV3MoEFused.ebias_weight_loader
        scoring_func = "sigmoid"
        self.e_score_correction_bias = self.expert_bias
        self.router_scaling_factor = getattr(config, "router_scaling_factor", 1.0)
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            params_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        # self.experts = FusedMoE(
        #     num_experts=self.n_routed_experts,
        #     top_k=top_k,
        #     hidden_size=config.hidden_size,
        #     intermediate_size=intermediate_size,
        #     reduce_results=False,
        #     layer_id=layer_id,
        #     quant_config=quant_config,
        #     prefix=f"{prefix}.experts",
        # )

        experts_cls = get_moe_impl_class(quant_config)
        self.experts = experts_cls(
            num_experts=self.n_routed_experts
            + get_global_server_args().ep_num_redundant_experts,
            top_k=top_k,
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            reduce_results=False,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            use_grouped_topk=True,
            num_expert_group=1,
            topk_group=1,
            renormalize=config.route_norm,
            scoring_func=scoring_func,
            correction_bias=self.e_score_correction_bias,
            routed_scaling_factor=self.router_scaling_factor,
            apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
        )

        if getattr(config, "num_shared_experts", 0) > 0:
            self.shared_mlp = HYV3FeedForward(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size
                * config.num_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_mlp",
                reduce_results=False,
                **(
                    dict(tp_rank=0, tp_size=1)
                    if get_moe_a2a_backend().is_deepep()
                    else {}
                ),
            )
        else:
            self.shared_mlp = None

    @staticmethod
    def ebias_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight.to(torch.float32))

    def get_moe_weights(self):
        num_local_experts = self.experts.num_local_experts
        result = []
        for name, x in self.experts.named_parameters():
            if name == "correction_bias":
                continue
            if filter_moe_weight_param_global_expert(name, x, num_local_experts):
                result.append(x.data)
        for name, x in self.experts.named_buffers():
            if (
                x.ndim > 0
                and x.shape[0] == num_local_experts
                and not getattr(x, "_sglang_require_global_experts", False)
            ):
                result.append(x)
        return result

    def _combine_routed_and_shared(
        self,
        final_hidden_states: torch.Tensor,
        shared_output: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Combine routed MoE output with shared MLP; apply router_scaling_factor once.

        When ``should_fuse_routed_scaling_factor_in_topk`` / TopK
        ``apply_routed_scaling_factor_on_output`` is False, TopK (including
        lightop after 049f5e) leaves renormalized probabilities and scaling
        must happen here — same contract as DeepSeek/Bailing. Skip only when
        aiter MoE already applies rsf internally.
        """
        # Do NOT skip on SGLANG_USE_LIGHTOP: after topk.py passes
        # apply_routed_scaling_factor_on_output through to moe_fused_gate,
        # lightop no longer bakes rsf when apply is False (Hy3 W8A8+triton).
        scale_already_applied = (
            self.experts.should_fuse_routed_scaling_factor_in_topk or _use_aiter
        )
        if scale_already_applied:
            if shared_output is not None:
                return final_hidden_states + shared_output
            return final_hidden_states

        rsf = self.router_scaling_factor
        if shared_output is not None:
            return shared_output + final_hidden_states * rsf
        return final_hidden_states * rsf

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        if get_moe_a2a_backend().is_deepep():
            return self._forward_deepep(hidden_states, forward_batch)

        return self.forward_normal(
            hidden_states,
            should_allreduce_fusion,
            use_reduce_scatter,
        )

    def _forward_deepep(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch],
    ) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        shared_output = None
        with get_global_expert_distribution_recorder().with_current_layer(
            self.layer_id
        ):
            if hidden_states.shape[0] > 0:
                router_logits, _ = self.gate(hidden_states.to(dtype=torch.float32))
                topk_output = self.topk(
                    hidden_states,
                    router_logits,
                    num_token_non_padded=(
                        forward_batch.num_token_non_padded
                        if forward_batch is not None
                        else None
                    ),
                    expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                        layer_id=self.layer_id,
                    ),
                )
                if self.shared_mlp is not None:
                    shared_output = self.shared_mlp(hidden_states)
            else:
                topk_output = self.topk.empty_topk_output(hidden_states.device)

            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                topk_output=topk_output,
            )

        final_hidden_states = self._combine_routed_and_shared(
            final_hidden_states, shared_output
        )

        return final_hidden_states.view(orig_shape)

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        if (
            self.alt_stream is not None
            and self.shared_mlp is not None
            and hidden_states.shape[0] > 0
            and get_is_capture_mode()
        ):
            return self._forward_dual_stream(
                hidden_states,
                should_allreduce_fusion,
                use_reduce_scatter,
            )
        return self._forward_single_stream(
            hidden_states,
            should_allreduce_fusion,
            use_reduce_scatter,
        )

    def _forward_single_stream(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits, _ = self.gate(hidden_states.to(dtype=torch.float32))
        with get_global_expert_distribution_recorder().with_current_layer(
            self.layer_id
        ):
            topk_output = self.topk(
                hidden_states,
                router_logits,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
            if self.shared_mlp is not None:
                shared_output = self.shared_mlp(hidden_states)
                final_hidden_states = self.experts(
                    hidden_states=hidden_states, topk_output=topk_output
                )
                final_hidden_states = self._combine_routed_and_shared(
                    final_hidden_states, shared_output
                )
            else:
                final_hidden_states = self.experts(
                    hidden_states=hidden_states, topk_output=topk_output
                )
                final_hidden_states = self._combine_routed_and_shared(
                    final_hidden_states, None
                )

        if self.ep_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=False,
            use_reduce_scatter=use_reduce_scatter,
            should_allreduce_fusion=should_allreduce_fusion,
        ):
            final_hidden_states = moe_expert_parallel_all_reduce(final_hidden_states)

        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True,
            use_reduce_scatter=use_reduce_scatter,
            should_allreduce_fusion=should_allreduce_fusion,
        ):
            final_hidden_states = moe_tensor_model_parallel_all_reduce(
                final_hidden_states
            )

        return final_hidden_states.view(orig_shape)

    def _forward_dual_stream(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        """Shared experts on main stream, routed experts on alt stream."""
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)

        shared_output = self.shared_mlp(hidden_states)

        with torch.cuda.stream(self.alt_stream):
            router_logits, _ = self.gate(hidden_states.to(dtype=torch.float32))
            with get_global_expert_distribution_recorder().with_current_layer(
                self.layer_id
            ):
                topk_output = self.topk(
                    hidden_states,
                    router_logits,
                    expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                        layer_id=self.layer_id,
                    ),
                )
                final_hidden_states = self.experts(
                    hidden_states=hidden_states, topk_output=topk_output
                )

        current_stream.wait_stream(self.alt_stream)
        if get_moe_a2a_backend().is_deepep():
            if self.dense_tp_size > 1:
                shared_output = tensor_model_parallel_all_reduce(shared_output)
            return self._combine_routed_and_shared(
                final_hidden_states, shared_output
            ).view(orig_shape)
        final_hidden_states = self._combine_routed_and_shared(
            final_hidden_states, shared_output
        )

        if self.ep_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=False,
            use_reduce_scatter=use_reduce_scatter,
            should_allreduce_fusion=should_allreduce_fusion,
        ):
            final_hidden_states = moe_expert_parallel_all_reduce(final_hidden_states)

        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True,
            use_reduce_scatter=use_reduce_scatter,
            should_allreduce_fusion=should_allreduce_fusion,
        ):
            final_hidden_states = moe_tensor_model_parallel_all_reduce(
                final_hidden_states
            )

        return final_hidden_states.view(orig_shape)


class HYV3Attention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)

        self.head_dim = getattr(config, "head_dim", hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.q_size_full = self.total_num_heads * self.head_dim
        self.kv_size_full = self.total_num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.use_qk_norm = getattr(
            config, "use_qk_norm", getattr(config, "qk_norm", False)
        )
        self.hy3_sp = get_global_server_args().hy3_sp

        if self.hy3_sp:
            self.qkv_proj = QKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=False,
                quant_config=quant_config,
                tp_rank=0,
                tp_size=1,
                prefix=f"{prefix}.qkv_proj",
            )
            self.o_proj = ReplicatedLinear(
                self.total_num_heads * self.head_dim,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.o_proj",
            )
        else:
            self.qkv_proj = QKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=False,
                quant_config=quant_config,
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
                prefix=f"{prefix}.qkv_proj",
            )
            self.o_proj = RowParallelLinear(
                self.total_num_heads * self.head_dim,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
                reduce_results=False,
                prefix=f"{prefix}.o_proj",
            )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=True,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=f"{prefix}.attn",
        )
        if self.use_qk_norm:
            rms_norm_eps = getattr(config, "rms_norm_eps", 1e-5)
            # HY3 stores one [head_dim] Q/K norm weight shared by all heads.
            # MiniMax's per-layer Q/K norm instead stores weights spanning all
            # heads, so its TP norm implementation is not interchangeable here.
            self.q_norm = RMSNorm(self.head_dim, rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, rms_norm_eps)
        self.page_size = 64
        if get_global_server_args().kv_cache_dtype == "fp8_e4m3":
            self.kv_cache_dtype = torch.float8_e4m3fn
        elif get_global_server_args().kv_cache_dtype == "fp8_e5m2":
            self.kv_cache_dtype = torch.float8_e5m2
        elif get_global_server_args().kv_cache_dtype in ("bf16", "bfloat16"):
            self.kv_cache_dtype = torch.bfloat16
        else:
            self.kv_cache_dtype = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if self.hy3_sp:
            return self.forward_sp(positions, hidden_states, forward_batch)

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        used_fused_hunyuan_rotary_kv_store = False
        if (
            _is_hcu
            and _use_fused_hunyuan_rotary
            and self.use_qk_norm
            and self.head_dim == self.rotary_emb.rotary_dim
        ):
            cos_sin_cache = self.rotary_emb.cos_sin_cache
            if cos_sin_cache.device != q.device or cos_sin_cache.dtype != q.dtype:
                cos_sin_cache = cos_sin_cache.to(
                    q.device, dtype=q.dtype, non_blocking=True
                )
                self.rotary_emb.cos_sin_cache = cos_sin_cache

            k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(
                self.attn.layer_id
            )
            kv_cache_dtype = self.kv_cache_dtype or k_buffer.dtype
            q, k, v = rms_rotary_embedding_fuse_with_kv_store(
                positions,
                q,
                k,
                v,
                cos_sin_cache,
                self.head_dim,
                self.page_size,
                k_buffer,
                v_buffer,
                forward_batch.out_cache_loc,
                is_neox=True,
                weight_q=self.q_norm.weight,
                weight_k=self.k_norm.weight,
                output_dtype=kv_cache_dtype,
                residual_q=None,
                residual_k=None,
                k_scale=None,
                v_scale=None,
                epsilon=self.q_norm.variance_epsilon,
            )
            used_fused_hunyuan_rotary_kv_store = True
        else:
            if self.use_qk_norm:
                q, k = _apply_hy3_qk_norm(
                    q, k, self.q_norm, self.k_norm, self.head_dim
                )
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(
            q,
            k,
            v,
            forward_batch,
            save_kv_cache=not used_fused_hunyuan_rotary_kv_store,
        )
        output, _ = self.o_proj(attn_output)
        return output

    def forward_sp(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states

        attn_tp_group = get_attention_tp_group()
        attn_tp_size = attn_tp_group.world_size
        tokens_per_rank = hidden_states.shape[0]

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.q_size_full, self.kv_size_full, self.kv_size_full], dim=-1
        )
        if self.use_qk_norm:
            q, k = _apply_hy3_qk_norm(
                q, k, self.q_norm, self.k_norm, self.head_dim
            )
        q, k = self.rotary_emb.forward_native(positions, q, k)

        qkv_by_head_rank = _pack_hy3_sp_qkv(
            q,
            k,
            v,
            attn_tp_size,
            self.q_size,
            self.kv_size,
            self.total_num_kv_heads,
            self.head_dim,
        )
        qkv_by_source_rank = torch.empty_like(qkv_by_head_rank)
        attn_tp_group.all_to_all_single(qkv_by_source_rank, qkv_by_head_rank)
        qkv_by_source_rank = qkv_by_source_rank.view(
            tokens_per_rank * attn_tp_size, self.q_size + 2 * self.kv_size
        )
        q, k, v = qkv_by_source_rank.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        attn_output = self.attn(q, k, v, forward_batch)
        attn_output_by_head_rank = _reshape_hy3_sp_attention_output(
            attn_output, attn_tp_size, self.q_size
        )
        attn_output_by_source_rank = torch.empty_like(attn_output_by_head_rank)
        attn_tp_group.all_to_all_single(
            attn_output_by_source_rank, attn_output_by_head_rank
        )
        attn_output = _merge_hy3_sp_attention_output(
            attn_output_by_source_rank
        )
        output, _ = self.o_proj(attn_output)
        return output


class HYV3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        rope_theta, _ = get_rope_config(config)
        self.self_attn = HYV3Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        first_k_dense_replace = getattr(config, "first_k_dense_replace", 0)
        if layer_id < first_k_dense_replace:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = HYV3FeedForward(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )
            self.block_type = "feedforward"
            is_layer_sparse = False
            is_previous_layer_sparse = False
            is_next_layer_sparse = layer_id + 1 >= first_k_dense_replace
        else:
            self.mlp = HYV3MoEFused(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                alt_stream=alt_stream,
            )
            self.block_type = "moe"
            is_layer_sparse = True
            is_previous_layer_sparse = (
                layer_id > 0 and layer_id - 1 >= first_k_dense_replace
            )
            is_next_layer_sparse = (
                layer_id != config.num_hidden_layers - 1
                and layer_id + 1 >= first_k_dense_replace
            )

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=is_layer_sparse,
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if get_global_server_args().hy3_sp:
            return self.forward_sp(
                positions, hidden_states, forward_batch, residual
            )

        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states,
            residual,
            forward_batch,
        )

        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states,
            residual,
            forward_batch,
        )

        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        if self.block_type == "moe":
            hidden_states = self.mlp(
                hidden_states,
                forward_batch,
                should_allreduce_fusion,
                use_reduce_scatter,
            )
        else:
            hidden_states = self.mlp(
                hidden_states,
                should_allreduce_fusion=should_allreduce_fusion,
                use_reduce_scatter=use_reduce_scatter,
            )

        if should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states,
                residual,
                forward_batch,
            )

        return hidden_states, residual

    def forward_sp(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.shape[0] == 0:
            return hidden_states, hidden_states

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        if self.block_type == "moe":
            hidden_states = self.mlp(hidden_states, forward_batch)
        else:
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class HYV3Model(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                enable_tp=not is_dp_attention_enabled(),
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.alt_stream = torch.cuda.Stream() if is_cuda() else None

        self.start_layer, self.end_layer = get_pp_indices(
            config.num_hidden_layers,
            self.pp_group.rank_in_group,
            self.pp_group.world_size,
        )
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: (
                HYV3DecoderLayer(
                    config=config,
                    layer_id=idx,
                    quant_config=quant_config,
                    prefix=prefix,
                    alt_stream=self.alt_stream,
                )
            ),
            prefix=f"{prefix}.layers",
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
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

        use_hy3_sp = get_global_server_args().hy3_sp
        if use_hy3_sp:
            attn_tp_group = get_attention_tp_group()
            attn_tp_size = attn_tp_group.world_size
            attn_tp_rank = attn_tp_group.rank_in_group
            if hidden_states.shape[0] % attn_tp_size != 0:
                raise ValueError(
                    "HY3 SP requires the token count to be divisible by the attention "
                    f"TP size, got {hidden_states.shape[0]} and {attn_tp_size}."
                )
            tokens_per_rank = hidden_states.shape[0] // attn_tp_size
            token_start = attn_tp_rank * tokens_per_rank
            token_end = token_start + tokens_per_rank
            hidden_states = hidden_states[token_start:token_end]
            positions = positions[token_start:token_end]
            if residual is not None:
                residual = residual[token_start:token_end]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )

        if not forward_batch.forward_mode.is_idle():
            hidden_states, _ = self.norm(hidden_states, residual)

        if use_hy3_sp:
            attn_tp_group = get_attention_tp_group()
            hidden_states_gathered = hidden_states.new_empty(
                hidden_states.shape[0] * attn_tp_group.world_size,
                hidden_states.shape[1],
            )
            attn_tp_group.all_gather_into_tensor(
                hidden_states_gathered, hidden_states
            )
            hidden_states = hidden_states_gathered

        return hidden_states


class HYV3ForCausalLM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        self.model = HYV3Model(config, quant_config, prefix=f"{prefix}.model")
        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.lm_head",
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
            )
        else:
            self.lm_head = PPMissingLayer()

        if getattr(self.config, "tie_word_embeddings", False):
            if not (self.pp_group.is_first_rank and self.pp_group.is_last_rank):
                raise ValueError(
                    "Pipeline parallelism for Hunyuan3 with tied word embeddings "
                    "is not supported because embed_tokens and lm_head live on "
                    "different pipeline stages."
                )
            self.lm_head.weight = self.model.embed_tokens.weight
        if self.pp_group.is_last_rank:
            self.logits_processor = LogitsProcessor(config)
        else:
            self.logits_processor = PPMissingLayer()
        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, HYV3MoEFused)
            }
        )

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

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
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )
        if not self.pp_group.is_last_rank:
            return hidden_states
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        # expert_params_mapping = FusedMoE.make_expert_params_mapping(
        moe_impl_class = get_moe_impl_class(self.quant_config)
        expert_params_mapping = moe_impl_class.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts
            + get_global_server_args().ep_num_redundant_experts,
        )

        params_dict = dict(self.named_parameters())
        num_nextn_layers = getattr(self.config, "num_nextn_predict_layers", 0)

        for name, loaded_weight in weights:
            if "lm_head.weight" in name and getattr(
                self.config, "tie_word_embeddings", False
            ):
                continue

            if "rotary_emb.inv_freq" in name:
                continue

            layer_id = get_layer_id(name)
            if layer_id is not None and (
                layer_id < self.model.start_layer or layer_id >= self.model.end_layer
            ):
                continue

            if num_nextn_layers > 0 and name.startswith("model.layers."):
                parts = name.split(".")
                if len(parts) >= 3 and int(parts[2]) >= self.config.num_hidden_layers:
                    continue

            is_found = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                is_found = True
                break
            if is_found:
                continue

            # Handle expert weights (including fp8 weight_scale, input_scale)
            is_expert_weight = False
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue
                is_expert_weight = True
                name_mapped = name.replace(weight_name, param_name)
                if name_mapped not in params_dict:
                    continue
                param = params_dict[name_mapped]
                weight_loader = param.weight_loader
                weight_loader(
                    param,
                    loaded_weight,
                    name_mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                break
            if is_expert_weight:
                continue

            if "router.gate." in name:
                name = name.replace("router.", "")
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation

        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None,
        )


EntryClass = [HYV3ForCausalLM]
