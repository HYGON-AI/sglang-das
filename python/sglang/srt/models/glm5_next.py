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
import logging
import re
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers.models.glm4v.configuration_glm4v import Glm4vVisionConfig

from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated
from sglang.kernels.ops.layernorm.mhc import hc_contract
from sglang.kernels.ops.layernorm.mhc import hc_post as _hc_post_fn
from sglang.kernels.ops.layernorm.mhc import hc_pre as _hc_pre_fn
from sglang.srt.configs.glm5_next import Glm5NextConfig as ModelNextConfig
from sglang.srt.configs.model_config import is_deepseek_dsa
from sglang.srt.distributed.parallel_state import (
    get_moe_expert_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.distributed.utils import divide
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.attention import vision_utils
from sglang.srt.layers.attention.dsa.utils import can_dsa_cp_split as can_cp_split
from sglang.srt.layers.attention.dsa.utils import (
    dsa_use_prefill_cp,
    is_dsa_enable_prefill_cp,
)
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    enable_moe_dense_fully_dp,
    get_attn_tp_context,
)
from sglang.srt.layers.communicator_glm5_next_cp import (
    Glm5NextCPLayerCommunicator,
    maybe_prefetch_full_attention_kv,
)
from sglang.srt.layers.communicator_glm5_next_mhc import MHCLayerCommunicator
from sglang.srt.layers.communicator_glm5_next_mhc_cp import (
    Glm5NextMHCCPLayerCommunicator,
)
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.fused_rms_quant import (
    is_lightop_sglang_rms_quant_available,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelBatchedLinear,
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    MergedColumnParallelRepeatedLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.layers.utils.common import PPMissingLayer
from sglang.srt.layers.utils.cp_utils import (
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_position,
)
from sglang.srt.layers.utils.glm5_next_cp import (
    cp_plain_all_gather,
    cp_plain_reduce_scatter,
    cp_plain_split,
    cp_plain_to_scattered,
    cp_scattered_to_plain,
    prepare_glm5_next_context_parallel_metadata,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    PPProxyTensors,
)
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
)
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.models.deepseek_common.utils import (
    _device_sm,
    _is_cuda,
    _is_gfx95_supported,
    _is_hcu,
    _use_aiter_gfx95,
)
from sglang.srt.runtime_context import get_forward, get_parallel

if _is_hcu:
    from sglang.kernels.ops.attention.fla.hcu.fused_norm_gate import FusedRMSNormGated

from sglang.srt.layers.attention.glm5_next.runtime import (
    get_glm5_next_runtime_args as get_global_server_args,
)
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2AttentionMLA as ModelNextMLAAttention,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2MLP as ModelNextMLP
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE as ModelNextMoe
from sglang.srt.models.glm5_next_vision import Glm5NextVisionModel
from sglang.srt.models.glm_visual import GlmVisualEncoderMixin
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.common import (
    BumpAllocator,
    LazyValue,
    add_prefix,
    log_info_on_rank0,
    make_layers,
    set_weight_attrs,
)

if _use_aiter_gfx95:
    from sglang.srt.layers.rocm_linear_utils import (
        get_dsv3_gemm_output_zero_allocator_size,
    )

logger = logging.getLogger(__name__)


KDA_SAFE_GATE_LOWER_BOUND = -5.0
KDA_NEG_EIGVAL_BETA_SCALE = 2.0
KDA_DEFAULT_BETA_SCALE = 1.0


def _linear_supports_prequantized_input(linear: nn.Module) -> bool:
    return bool(
        getattr(
            getattr(linear, "quant_method", None),
            "supports_prequantized_input",
            False,
        )
    )


def _apply_linear_with_optional_quant(linear, x, input_quant_args):
    if input_quant_args is None or not _linear_supports_prequantized_input(linear):
        return linear(x)
    return linear(x, input_quant_args=input_quant_args)


def _get_config_dtype(config: ModelNextConfig):
    dtype = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
    if isinstance(dtype, str):
        return getattr(torch, dtype, torch.bfloat16)
    return dtype if dtype is not None else torch.bfloat16


class ModelNextLinearAttention(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        config: ModelNextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = 1e-5,
        prefix: str = "",
        reduce_results: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if is_dsa_enable_prefill_cp():
            head_shard_size = get_parallel().attn_cp_size
            head_shard_rank = get_parallel().attn_cp_rank
        else:
            head_shard_size = get_parallel().attn_tp_size
            head_shard_rank = get_parallel().attn_tp_rank

        def head_sharded_weight_loader(shard_axis: int):
            def loader(param: torch.Tensor, loaded_weight: torch.Tensor):
                shard_size = param.data.shape[shard_axis]
                loaded_weight = loaded_weight.narrow(
                    shard_axis, head_shard_rank * shard_size, shard_size
                )
                return default_weight_loader(param, loaded_weight)

            return loader

        self.hidden_size = hidden_size
        self.config = config
        self.head_dim = config.linear_attn_config["head_dim"]
        self.num_heads = config.linear_attn_config["num_heads"]
        self.num_k_heads = config.linear_attn_config["num_heads"]
        self.num_v_heads = config.linear_attn_config["num_heads"]
        self.head_k_dim = config.linear_attn_config["head_dim"]
        self.head_v_dim = config.linear_attn_config["head_dim"]
        self.layer_idx = layer_idx
        self.prefix = prefix
        assert self.num_heads % head_shard_size == 0
        self.local_num_heads = divide(self.num_heads, head_shard_size)
        self.nsa_enable_prefill_cp = is_dsa_enable_prefill_cp()

        projection_size = self.head_dim * self.num_heads
        self.conv_size = config.linear_attn_config["short_conv_kernel_size"]
        self.allow_neg_eigval = config.linear_allow_neg_eigval
        _cfg_lower_bound = config.linear_attn_config.get("gate_lower_bound", None)
        _cfg_safe_gate = config.linear_attn_config.get("safe_gate", False)
        self.safe_gate = _cfg_lower_bound is not None or _cfg_safe_gate
        self._resolved_gate_lower_bound = (
            _cfg_lower_bound
            if _cfg_lower_bound is not None
            else KDA_SAFE_GATE_LOWER_BOUND
            if _cfg_safe_gate
            else None
        )

        # Optional experimental fusion for the KDA projections.
        self.do_fuse_qkvbfg = envs.SGLANG_GLM5_NEXT_FUSE_QKVBFG.get()
        if self.do_fuse_qkvbfg:
            # Fuse q/k/v/beta (column-parallel) + f_a/g_a (replicated) into one
            # projection, and f_b/g_b into one batched bmm.
            self.qkvb_sizes = [
                projection_size,
                projection_size,
                projection_size,
                self.num_heads,
            ]
            self.fg_sizes = [self.head_dim, self.head_dim]

            self.fused_qkvbfg_a_proj = MergedColumnParallelRepeatedLinear(
                self.hidden_size,
                self.qkvb_sizes,
                self.fg_sizes,
                quant_config=None,  # quant_config,
                prefix=f"{prefix}.fused_qkvbfg_a_proj",
                tp_rank=head_shard_rank,
                tp_size=head_shard_size,
            )
            self.split_sizes = [
                3 * projection_size // head_shard_size,  # qkv
                self.num_heads // head_shard_size,  # beta
                2 * self.head_dim,  # f_a, g_a (replicated, no sharding)
            ]
            self.fused_fg_b_proj = ColumnParallelBatchedLinear(
                2,
                self.head_dim,
                projection_size,
                dtype=_get_config_dtype(config),
                tp_rank=head_shard_rank,
                tp_size=head_shard_size,
            )
        else:
            self.qkv_proj = QKVParallelLinear(
                self.hidden_size,
                self.head_dim,
                self.num_heads,
                self.num_k_heads,
                bias=False,
                quant_config=quant_config,
                tp_rank=head_shard_rank,
                tp_size=head_shard_size,
                prefix=f"{prefix}.qkv_proj",
            )

            self.f_a_proj = ReplicatedLinear(
                self.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.f_a_proj",
            )

            self.f_b_proj = ColumnParallelLinear(
                self.head_dim,
                projection_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.f_b_proj",
                tp_rank=head_shard_rank,
                tp_size=head_shard_size,
            )

            self.b_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.b_proj",
                tp_rank=head_shard_rank,
                tp_size=head_shard_size,
            )

            self.g_a_proj = ReplicatedLinear(
                self.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_a_proj",
            )
            self.g_b_proj = ColumnParallelLinear(
                self.head_dim,
                projection_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_b_proj",
                tp_rank=head_shard_rank,
                tp_size=head_shard_size,
            )

        self.dt_bias = nn.Parameter(
            torch.empty(divide(projection_size, head_shard_size), dtype=torch.float32)
        )
        set_weight_attrs(
            self.dt_bias,
            {"weight_loader": head_sharded_weight_loader(0)},
        )

        self.qkv_conv1d = MergedColumnParallelLinear(
            input_size=self.conv_size,
            output_sizes=[projection_size, projection_size, projection_size],
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.qkv_conv1d",
            tp_rank=head_shard_rank,
            tp_size=head_shard_size,
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.qkv_conv1d.weight.data = self.qkv_conv1d.weight.data.unsqueeze(1)

        self.A_log = nn.Parameter(
            torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32)
        )

        def a_log_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor):
            if loaded_weight.dim() == 1:
                loaded_weight = loaded_weight.view([1, 1, -1, 1])
            return head_sharded_weight_loader(2)(param, loaded_weight)

        set_weight_attrs(self.A_log, {"weight_loader": a_log_weight_loader})

        self.o_norm = FusedRMSNormGated(
            self.head_dim, eps=rms_norm_eps, activation="sigmoid"
        )
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            reduce_results=reduce_results,
            tp_rank=head_shard_rank,
            tp_size=head_shard_size,
        )

        conv_weights = self.qkv_conv1d.weight.squeeze(1)
        bias = self.qkv_conv1d.bias

        self.attn = RadixLinearAttention(
            layer_id=self.layer_idx,
            num_q_heads=self.local_num_heads,
            num_k_heads=self.local_num_heads,
            num_v_heads=self.local_num_heads,
            head_q_dim=self.head_k_dim,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_weights=conv_weights,
            bias=bias,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )

        self.attn.beta_scale = (
            KDA_NEG_EIGVAL_BETA_SCALE
            if self.allow_neg_eigval
            else KDA_DEFAULT_BETA_SCALE
        )
        self.attn.safe_gate = self.safe_gate
        self.attn.safe_gate_lower_bound = KDA_SAFE_GATE_LOWER_BOUND
        self.attn.lower_bound = self._resolved_gate_lower_bound

        self._cp_fuse_symm_mem = envs.SGLANG_DSA_CP_FUSE_SYMM_MEM.get()

    def forward_qkvbfg(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        input_quant_args=None,
    ):
        cp_prefill = dsa_use_prefill_cp(forward_batch)
        if cp_prefill and self._cp_fuse_symm_mem:
            from torch.distributed._symmetric_memory import (
                _fused_all_gather_matmul,
            )

            _, [qkv, beta, fa_out, ga_out] = _fused_all_gather_matmul(
                hidden_states.contiguous(),
                [
                    p.weight.t()
                    for p in (
                        self.qkv_proj,
                        self.b_proj,
                        self.f_a_proj,
                        self.g_a_proj,
                    )
                ],
                gather_dim=0,
                group_name=get_parallel().attn_cp_group.device_group.group_name,
                return_A=False,
            )
        else:
            if cp_prefill:
                hidden_states = cp_plain_all_gather(
                    hidden_states, get_parallel().attn_cp_size, forward_batch
                )
                # The gathered tensor has a different row layout from the
                # local prequantized input.  This mode is excluded by the
                # caller, but retain a safe fallback if that changes.
                input_quant_args = None
            qkv = _apply_linear_with_optional_quant(
                self.qkv_proj, hidden_states, input_quant_args
            )[0]
            beta = _apply_linear_with_optional_quant(
                self.b_proj, hidden_states, input_quant_args
            )[0]
            fa_out = _apply_linear_with_optional_quant(
                self.f_a_proj, hidden_states, input_quant_args
            )[0]
            ga_out = _apply_linear_with_optional_quant(
                self.g_a_proj, hidden_states, input_quant_args
            )[0]

        forget_gate = self.f_b_proj(fa_out)[0]
        g_proj_states = self.g_b_proj(ga_out)[0]
        return qkv, beta, forget_gate, g_proj_states

    def forward_qkvbfg_fused(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ):
        cp_prefill = dsa_use_prefill_cp(forward_batch)
        if cp_prefill and self._cp_fuse_symm_mem:
            from torch.distributed._symmetric_memory import (
                _fused_all_gather_matmul,
            )

            _, [fused_states] = _fused_all_gather_matmul(
                hidden_states.contiguous(),
                [self.fused_qkvbfg_a_proj.weight.t()],
                gather_dim=0,
                group_name=get_parallel().attn_cp_group.device_group.group_name,
                return_A=False,
            )
        else:
            if cp_prefill:
                hidden_states = cp_plain_all_gather(
                    hidden_states, get_parallel().attn_cp_size, forward_batch
                )
            fused_states = self.fused_qkvbfg_a_proj(hidden_states)

        qkv, beta, fg_a_states = torch.split(fused_states, self.split_sizes, dim=-1)
        forget_gate, g_proj_states = self.fused_fg_b_proj(
            fg_a_states.view(-1, 2, self.head_dim).transpose(0, 1)
        )
        return qkv, beta, forget_gate, g_proj_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        **kwargs,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            return hidden_states

        if self.do_fuse_qkvbfg:
            mixed_qkv, beta, forget_gate, g_proj_states = self.forward_qkvbfg_fused(
                hidden_states, forward_batch
            )
        else:
            input_quant_args = kwargs.get("input_quant_args")
            if input_quant_args is None:
                mixed_qkv, beta, forget_gate, g_proj_states = self.forward_qkvbfg(
                    hidden_states, forward_batch
                )
            else:
                mixed_qkv, beta, forget_gate, g_proj_states = self.forward_qkvbfg(
                    hidden_states,
                    forward_batch,
                    input_quant_args=input_quant_args,
                )

        # For prefill, chunk_kda expects raw gate as [B, T, H, K], while beta is
        # already sigmoid-activated by the caller. Decode and target verification
        # keep raw gate/beta so the recurrent kernel can update state in one
        # fused pass.
        if not (
            forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_target_verify()
        ):
            forget_gate = forget_gate.unflatten(-1, (-1, self.head_dim))
            beta = self.attn.beta_scale * beta.float().sigmoid()
            forget_gate = forget_gate.unsqueeze(0)
        beta = beta.unsqueeze(0)

        core_attn_out = self.attn(
            forward_batch,
            mixed_qkv=mixed_qkv,
            a=forget_gate,
            b=beta,
        )

        norm_gate = g_proj_states.unflatten(
            -1, (-1, self.head_dim)
        )  # ... (h d) -> ... h d
        core_attn_out = self.o_norm(core_attn_out, norm_gate)
        core_attn_out = core_attn_out.squeeze(0).flatten(-2)  # 1 n h d -> n (h d)

        cp_prefill = dsa_use_prefill_cp(forward_batch)
        if cp_prefill and self._cp_fuse_symm_mem:
            from torch.distributed._symmetric_memory import (
                _fused_matmul_reduce_scatter,
            )

            return _fused_matmul_reduce_scatter(
                core_attn_out.contiguous(),
                self.o_proj.weight.t(),
                reduce_op="sum",
                scatter_dim=0,
                group_name=get_parallel().attn_cp_group.device_group.group_name,
            )
        output = self.o_proj(core_attn_out)[0]
        if cp_prefill:
            output = cp_plain_reduce_scatter(output, get_parallel().attn_cp_size)
        elif self.nsa_enable_prefill_cp:
            # KDA heads are statically CP-sharded (head_shard_size = cp_size
            # at init), so o_proj is always a per-rank partial sum;
            # attn_tp=1 means TP reduce won't cover it. Under CP-extend the
            # reduce is fused with o_proj above (or done as a plain
            # reduce_scatter), but decode still needs an explicit CP
            # all_reduce here. Must rebind output because the group's
            # all_reduce returns a NEW tensor when an out-of-place path is
            # picked (custom AR, mscclpp, symm-mem, piecewise CUDA graph).
            output = get_parallel().attn_cp_group.all_reduce(output)
        return output


class ModelNextDecoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelNextConfig,
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
        rope_theta = config.rope_theta
        rope_scaling = config.rope_scaling
        max_position_embeddings = config.max_position_embeddings
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            get_global_server_args().speculative_algorithm
        )
        rms_norm_eps = config.rms_norm_eps

        self.nsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        self.layer_id = layer_id
        self.is_nextn = is_nextn
        self.is_linear_attn = config.is_kda_layer(layer_id)
        self.is_layer_sparse = self._is_layer_sparse(layer_id, is_nextn=is_nextn)

        if self.is_linear_attn:
            self.self_attn = ModelNextLinearAttention(
                layer_idx=layer_id,
                hidden_size=config.hidden_size,
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                rms_norm_eps=rms_norm_eps,
                reduce_results=False,
            )
        else:
            self.self_attn = ModelNextMLAAttention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
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
                skip_rope=(
                    config.qk_rope_head_dim == 0 or getattr(config, "mla_nope", False)
                ),
            )

        if not hasattr(config, "q_lora_rank") and envs.SGLANG_USE_AG_AFTER_QLORA.get():
            raise ValueError(
                "SGLANG_USE_AG_AFTER_QLORA only supports the model with q_lora_rank"
            )

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
            self.mlp = ModelNextMoe(
                config=config,
                quant_config=moe_quant_config_override or quant_config,
                prefix=add_prefix("mlp", prefix),
                layer_id=self.layer_id,
                alt_stream=alt_stream,
                is_nextn=is_nextn,
            )
        else:
            if enable_moe_dense_fully_dp() or self.nsa_enable_prefill_cp:
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = ModelNextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
                swiglu_limit=config.swiglu_limit,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.is_first_layer = self.layer_id == 0
        self.is_last_layer = is_nextn or (
            self.layer_id == self.config.num_hidden_layers - 1
        )

        if self.config.mhc:
            hc_mult = config.hc_mult
            mix_hc = (2 + hc_mult) * hc_mult
            hc_dim = hc_mult * config.hidden_size

            # mHC params live directly on the decoder layer so their names
            # (hc_{attn,ffn}_{base,scale,fn}) match the ckpt verbatim and
            # default_weight_loader hits them without any rename. The
            # communicator reads them at runtime via MHCState(layer=self).
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
            self.hc_attn_fn = nn.Parameter(
                torch.empty(mix_hc, hc_dim, dtype=torch.float32)
            )

            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
            self.hc_ffn_fn = nn.Parameter(
                torch.empty(mix_hc, hc_dim, dtype=torch.float32)
            )

        shared_kwargs: Dict[str, Any] = dict(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=self.is_last_layer,
            qkv_latent_func=(
                self.self_attn.prepare_qkv_latent if not self.is_linear_attn else None
            ),
        )

        if self.config.mhc and self.nsa_enable_prefill_cp:
            self.layer_communicator = Glm5NextMHCCPLayerCommunicator(
                **shared_kwargs,
                is_first_layer=self.is_first_layer,
                hc_mult=config.hc_mult,
                hc_attn_pre=self.hc_attn_pre,
                hc_ffn_pre=self.hc_ffn_pre,
                hc_post=self.hc_post,
                is_layer_sparse=self.is_layer_sparse,
            )
        elif self.config.mhc:
            self.layer_communicator = MHCLayerCommunicator(
                **shared_kwargs,
                is_first_layer=self.is_first_layer,
                hc_mult=config.hc_mult,
                hc_attn_pre=self.hc_attn_pre,
                hc_ffn_pre=self.hc_ffn_pre,
                hc_post=self.hc_post,
                is_layer_sparse=self.is_layer_sparse,
            )
        elif self.nsa_enable_prefill_cp:
            self.layer_communicator = Glm5NextCPLayerCommunicator(**shared_kwargs)
        else:
            self.layer_communicator = LayerCommunicator(**shared_kwargs)

        if self.is_linear_attn:
            if self.self_attn.do_fuse_qkvbfg:
                attn_prequantized_projection = None
            else:
                attn_prequantized_projection = self.self_attn.qkv_proj
        else:
            attn_prequantized_projection = getattr(
                self.self_attn,
                "fused_qkv_a_proj_with_mqa",
                None,
            )

        use_mhc_rms_quant = (
            envs.SGLANG_USE_FUSED_RMS_QUANT.get()
            and is_lightop_sglang_rms_quant_available()
            and isinstance(self.layer_communicator, MHCLayerCommunicator)
        )
        self._can_fuse_attn_rms_quant = (
            use_mhc_rms_quant
            and (not self.is_linear_attn or not self.self_attn.do_fuse_qkvbfg)
            and attn_prequantized_projection is not None
            and _linear_supports_prequantized_input(attn_prequantized_projection)
        )
        mlp_with_gate_up = (
            self.mlp.shared_experts
            if isinstance(self.mlp, ModelNextMoe)
            and self.mlp.num_fused_shared_experts == 0
            and hasattr(self.mlp, "shared_experts")
            else self.mlp
        )
        mlp_gate_up_proj = getattr(mlp_with_gate_up, "gate_up_proj", None)
        self._can_fuse_mlp_rms_quant = (
            use_mhc_rms_quant
            and mlp_gate_up_proj is not None
            and _linear_supports_prequantized_input(mlp_gate_up_proj)
        )

    def hc_attn_pre(self, hidden_states, out_norm_weight, out_norm_eps):
        """mHC pre-stage for the attention sub-layer (reads hc_attn_* params)."""
        assert self.config.mhc, "hc_attn_pre is only valid when config.mhc=True"
        return _hc_pre_fn(
            x=hidden_states,
            hc_fn=self.hc_attn_fn,
            hc_scale=self.hc_attn_scale,
            hc_base=self.hc_attn_base,
            hc_mult=self.config.hc_mult,
            rms_eps=self.config.rms_norm_eps,
            hc_eps=self.config.hc_eps,
            sinkhorn_iters=self.config.hc_sinkhorn_iters,
            post_mult_value=self.config.hc_post_mult_value,
            hc_norm_weight=None,
            out_norm_weight=out_norm_weight,
            out_norm_eps=out_norm_eps,
        )

    def hc_ffn_pre(self, hidden_states, out_norm_weight, out_norm_eps):
        """mHC pre-stage for the FFN sub-layer (reads hc_ffn_* params)."""
        assert self.config.mhc, "hc_ffn_pre is only valid when config.mhc=True"
        return _hc_pre_fn(
            x=hidden_states,
            hc_fn=self.hc_ffn_fn,
            hc_scale=self.hc_ffn_scale,
            hc_base=self.hc_ffn_base,
            hc_mult=self.config.hc_mult,
            rms_eps=self.config.rms_norm_eps,
            hc_eps=self.config.hc_eps,
            sinkhorn_iters=self.config.hc_sinkhorn_iters,
            post_mult_value=self.config.hc_post_mult_value,
            hc_norm_weight=None,
            out_norm_weight=out_norm_weight,
            out_norm_eps=out_norm_eps,
        )

    def hc_post(self, hidden_states, residual, h_res, h_post):
        """mHC post-stage (parameter-free, scalar hc_mult only)."""
        assert self.config.mhc, "hc_post is only valid when config.mhc=True"
        return _hc_post_fn(
            x=hidden_states,
            residual=residual,
            h_post=h_post,
            h_res=h_res,
            hc_mult=self.config.hc_mult,
        )

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
        prev_topk_indices: Optional[torch.Tensor] = None,
        next_full_attention_layer_id: Optional[int] = None,
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

        fuse_attn_rms_quant = self._can_fuse_attn_rms_quant and not dsa_use_prefill_cp(
            forward_batch, self.nsa_enable_prefill_cp
        )
        prepare_attn_kwargs = {"fuse_rms_quant": True} if fuse_attn_rms_quant else {}
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states,
            residual,
            forward_batch,
            quant_format,
            **prepare_attn_kwargs,
        )
        attn_input_quant_args = (
            self.layer_communicator.take_attn_input_quant_args()
            if fuse_attn_rms_quant
            else None
        )

        # MLA's CP attention consumes the scattered (round-robin/zigzag)
        # layout while the cross-layer contract is plain (block-contiguous,
        # see ModelNextModel.forward). KDA handles its own CP gather/scatter
        # inside ModelNextLinearAttention, so only MLA layers need this wrap.
        # NOTE: prepare_attn already stored an AttentionInputs referencing the
        # plain hidden_states for fetch_qkv_latent(); rebind that ref to the
        # scattered tensor so q/kv latent and positions stay token-aligned.
        mla_cp_wrap = not self.is_linear_attn and dsa_use_prefill_cp(
            forward_batch, self.nsa_enable_prefill_cp
        )
        if mla_cp_wrap:
            hidden_states = cp_plain_to_scattered(
                hidden_states, forward_batch, get_parallel().attn_cp_size
            )
            get_attn_tp_context().set_hidden_states_local(hidden_states)

        attn_quant_kwargs = (
            {"input_quant_args": attn_input_quant_args}
            if self.is_linear_attn and attn_input_quant_args is not None
            else {}
        )
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
            layer_scatter_modes=self.layer_scatter_modes,
            prev_topk_indices=prev_topk_indices,
            **attn_quant_kwargs,
        )
        if isinstance(hidden_states, tuple):
            hidden_states, topk_indices = hidden_states
        else:
            topk_indices = None

        if mla_cp_wrap:
            hidden_states = cp_scattered_to_plain(
                hidden_states, forward_batch, get_parallel().attn_cp_size
            )

        maybe_prefetch = getattr(
            self.layer_communicator, "maybe_prefetch_next_full_attention_kv", None
        )
        if maybe_prefetch is not None:
            maybe_prefetch(forward_batch, next_full_attention_layer_id)

        prepare_mlp_kwargs = (
            {"fuse_rms_quant": True} if self._can_fuse_mlp_rms_quant else {}
        )
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states,
            residual,
            forward_batch,
            **prepare_mlp_kwargs,
        )
        mlp_input_quant_args = (
            self.layer_communicator.take_mlp_input_quant_args()
            if self._can_fuse_mlp_rms_quant
            else None
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

        if isinstance(self.mlp, ModelNextMLP):
            gemm_output_zero_allocator = None

        mlp_quant_kwargs = (
            {"input_quant_args": mlp_input_quant_args}
            if mlp_input_quant_args is not None and isinstance(self.mlp, ModelNextMLP)
            else {}
        )
        prequantized_shared_expert = mlp_input_quant_args is not None and isinstance(
            self.mlp, ModelNextMoe
        )
        if prequantized_shared_expert:
            self.mlp.set_shared_expert_input_quant_args(mlp_input_quant_args)
        try:
            with get_forward().scoped(
                fuse_mlp_allreduce=should_allreduce_fusion,
                mlp_reduce_scatter=use_reduce_scatter,
            ):
                hidden_states = self.mlp(
                    hidden_states,
                    forward_batch=forward_batch,
                    gemm_output_zero_allocator=gemm_output_zero_allocator,
                    **mlp_quant_kwargs,
                )
        finally:
            if prequantized_shared_expert:
                self.mlp.clear_shared_expert_input_quant_args()

        if not self.nsa_enable_prefill_cp and should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True

        if not should_allreduce_fusion:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states,
                residual,
                forward_batch,
            )

        return hidden_states, residual, topk_indices


class ModelNextModel(nn.Module):
    def __init__(
        self,
        config: ModelNextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        if (
            envs.SGLANG_USE_FUSED_RMS_QUANT.get()
            and not is_lightop_sglang_rms_quant_available()
        ):
            log_info_on_rank0(
                logger,
                "SGLANG_USE_FUSED_RMS_QUANT was requested, but the rebuilt "
                "LightOp SGLang entry point is unavailable; using native RMSNorm.",
            )
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.first_k_dense_replace = config.first_k_dense_replace
        self.consumed_train_tokens = None
        self.consumed_train_samples = None
        self.pp_group = get_pp_group()
        self.nsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_size = get_parallel().attn_cp_size
        else:
            self.cp_size = None

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                enable_tp=not is_dp_attention_enabled(),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.alt_stream = (
            torch.cuda.Stream()
            if _is_cuda or envs.SGLANG_NPU_USE_MULTI_STREAM.get()
            else None
        )

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: ModelNextDecoderLayer(
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
                    if isinstance(layer.mlp, ModelNextMoe)
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
        local_full_attention_layer_ids = [
            layer_id
            for layer_id in config.full_attention_layer_ids
            if self.start_layer <= layer_id < self.end_layer
        ]
        self.first_full_attention_layer_id = (
            local_full_attention_layer_ids[0]
            if local_full_attention_layer_ids
            else None
        )
        self.next_full_attention_layer_id = dict(
            zip(
                local_full_attention_layer_ids,
                local_full_attention_layer_ids[1:],
            )
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
                    if isinstance(self.layers[i].mlp, ModelNextMoe)
                ]
            )

            allocate_size = 0
            for i in range(len(self.layers)):
                if isinstance(self.layers[i].mlp, ModelNextMoe):
                    # tp_size = get_tensor_model_parallel_world_size()
                    a2a_backend = get_moe_a2a_backend()
                    is_a2a_moe = (
                        a2a_backend.is_deepep()
                        or a2a_backend.is_mori()
                        or a2a_backend.is_mooncake()
                    )
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
        device = input_embeds.device if input_embeds is not None else input_ids.device
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

        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = None if self.config.mhc else pp_proxy_tensors["residual"]

        if dsa_use_prefill_cp(forward_batch, self.nsa_enable_prefill_cp):
            _check_rank_consistency = (
                envs.SGLANG_DEBUG_HACK_CP_CHECK_RANK_CONSISTENCY.get()
                and self.pp_group.is_first_rank
            )
            if _check_rank_consistency:
                _pre_split_hidden_states = hidden_states.clone()
                _pre_split_positions = positions.clone()

            if self.pp_group.is_first_rank:
                # Plain cross-layer contract: scatter hidden_states as
                # block-contiguous (rank i holds [i*K, (i+1)*K)) so KDA
                # layers can use vanilla all_gather without rerange.
                # `positions` stays in the scattered (round-robin/zigzag)
                # layout below because MLA's CP attention -- the only
                # consumer of positions -- aligns positions with its own
                # scattered hidden_states after MLA prepare_attn converts
                # h from plain to scattered.
                hidden_states = cp_plain_split(hidden_states)
            positions = cp_split_and_rebuild_position(forward_batch, positions)

            if _check_rank_consistency:
                _gathered_hidden = cp_plain_all_gather(hidden_states, self.cp_size)
                assert torch.equal(_gathered_hidden, _pre_split_hidden_states), (
                    "SGLANG_DEBUG_HACK_CP_CHECK_RANK_CONSISTENCY: "
                    "cp_plain_split after cp_plain_all_gather is not identity "
                    "on hidden_states."
                )
                _gathered_positions = cp_all_gather_rerange_output(
                    positions.unsqueeze(-1),
                    self.cp_size,
                    forward_batch,
                    torch.cuda.current_stream(),
                ).squeeze(-1)
                assert torch.equal(_gathered_positions, _pre_split_positions), (
                    "SGLANG_DEBUG_HACK_CP_CHECK_RANK_CONSISTENCY: "
                    "cp_split_and_rebuild_position after "
                    "cp_all_gather_rerange_output is not identity on positions."
                )

            if _is_hcu:
                maybe_prefetch_full_attention_kv(
                    forward_batch, self.first_full_attention_layer_id
                )

        aux_hidden_states = []
        topk_indices = None
        for i in range(self.start_layer, self.end_layer):
            # NOTE: torch dynamo does not support graph break in context manager
            ctx = (
                nullcontext()
                if check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
                else get_global_expert_distribution_recorder().with_current_layer(i)
            )
            with ctx:
                if i in self.layers_to_capture:
                    if self.config.mhc:
                        aux_hidden_states.append(
                            hidden_states
                            if hidden_states.shape[-1] == self.config.hidden_size
                            else hc_contract(hidden_states, self.config.hc_mult)
                        )
                    elif self.enable_a2a_moe and i > self.first_k_dense_replace:
                        aux_hidden_state = get_parallel().attn_tp_group.all_gather(
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
                    prev_topk_indices=topk_indices,
                    next_full_attention_layer_id=(
                        self.next_full_attention_layer_id.get(i)
                    ),
                )

        if not self.pp_group.is_last_rank:
            pp_tensors = {"hidden_states": hidden_states}
            if not self.config.mhc:
                pp_tensors["residual"] = residual
            return PPProxyTensors(pp_tensors)
        else:
            if not forward_batch.forward_mode.is_idle():
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)

        if self.pp_group.is_last_rank and dsa_use_prefill_cp(
            forward_batch, self.nsa_enable_prefill_cp
        ):
            # Plain contract: rank-major all_gather output is already in
            # natural sequential order, no rerange needed.
            hidden_states = cp_plain_all_gather(
                hidden_states, self.cp_size, forward_batch
            )
        if len(aux_hidden_states) == 0:
            return hidden_states
        return hidden_states, aux_hidden_states


class ModelNextForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False
    # Fused module -> checkpoint shard names.  The compressed-tensors ignore
    # routing (should_ignore_layer) uses this to expand a fused name such as
    # qkv_proj back to q_proj/k_proj/v_proj before matching the checkpoint's
    # ignore rules.  Without these entries, the KDA QKV projections and the
    # dense MLP gate/up fusion are never recognized as ignored, so the loader
    # wrongly materializes FP8 weights and a finfo(min) scale for layers the
    # checkpoint keeps in BF16.
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "qkv_conv1d": ["q_conv1d", "k_conv1d", "v_conv1d"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    _STACKED_PARAMS_MAPPING = [
        # Fused KDA "a" projections (used when do_fuse_qkvbfg=True).
        # Listed first so .q_proj on a KDA layer routes to fused_qkvbfg_a_proj;
        # the loader falls through to qkv_proj when the fused param is absent.
        ("fused_qkvbfg_a_proj", "q_proj", 0),
        ("fused_qkvbfg_a_proj", "k_proj", 1),
        ("fused_qkvbfg_a_proj", "v_proj", 2),
        ("fused_qkvbfg_a_proj", "b_proj", 3),
        ("fused_qkvbfg_a_proj", "f_a_proj", 4),
        ("fused_qkvbfg_a_proj", "g_a_proj", 5),
        # Fused KDA "b" projections (used when do_fuse_qkvbfg=True).
        ("fused_fg_b_proj", "f_b_proj", 0),
        ("fused_fg_b_proj", "g_b_proj", 1),
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("qkv_conv1d", "q_conv1d", 0),
        ("qkv_conv1d", "k_conv1d", 1),
        ("qkv_conv1d", "v_conv1d", 2),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    _NEXTN_SPEC_NAMES = ("shared_head.norm", "eh_proj", "enorm", "hnorm")
    _EAGLE_IGNORE_NAMES = ("eagle_draft_tokens_map", "eagle_lm_head.weight")
    _SHARED_EXPERTS_PATTERN = re.compile(
        r"^model\.layers\.(\d+)\.mlp\.shared_experts\.(.+)$"
    )
    _AWQ_LIKE_QUANT_METHOD = {"awq", "awq_marlin", "moe_wna16"}

    def __init__(
        self,
        config: ModelNextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.fuse_qkv_a_proj = config.q_lora_rank is not None
        if self.fuse_qkv_a_proj:
            self.packed_modules_mapping["fused_qkv_a_proj_with_mqa"] = [
                "q_a_proj",
                "kv_a_proj_with_mqa",
            ]

        self.pp_group = get_pp_group()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        if quant_config is not None:
            quant_config.update_packed_modules_mapping(self.packed_modules_mapping)
        self.quant_config = quant_config
        self.determine_num_fused_shared_experts()
        self.use_dsa = is_deepseek_dsa(config)
        self.model = ModelNextModel(
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
                if isinstance(layer.mlp, ModelNextMoe)
            }
        )
        self.capture_aux_hidden_states = False

        self.nsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_rank = get_parallel().attn_cp_rank
            self.cp_size = get_parallel().attn_cp_size
        else:
            self.cp_rank = self.cp_size = None

        get_attn_tp_context().init_context(config.q_lora_rank, self.use_dsa, config.mhc)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

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
        if self.nsa_enable_prefill_cp:
            if can_cp_split(len(input_ids), self.cp_size, self.use_dsa, forward_batch):
                forward_batch.attn_cp_metadata = (
                    prepare_glm5_next_context_parallel_metadata(
                        len(input_ids),
                        self.cp_rank,
                        self.cp_size,
                        forward_batch,
                    )
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

    @classmethod
    def shared_experts_fusion_disable_reason(cls, hf_config, quant_config):
        # Main installs this decision before constructing any MoE layer.
        config = getattr(hf_config, "text_config", hf_config)
        if not getattr(config, "n_shared_experts", None):
            return "No shared experts are defined in the config."
        if not _is_cuda:
            return "GLM shared experts fusion requires CUDA devices."
        if _device_sm is not None and _device_sm < 80:
            return "GLM shared experts fusion requires SM80 or newer GPUs."
        if get_moe_expert_parallel_world_size() > 1:
            return "GLM shared experts fusion does not support expert parallelism."
        if get_moe_a2a_backend().is_deepep():
            return "GLM shared experts fusion does not support DeepEP."
        return None

    def determine_num_fused_shared_experts(self):
        from sglang.srt.layers.moe.utils import is_shared_experts_fusion_disabled

        self.num_fused_shared_experts = (
            0 if is_shared_experts_fusion_disabled() else self.config.n_shared_experts
        )
        if self.num_fused_shared_experts:
            assert self.num_fused_shared_experts == 1

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        is_nextn: bool = False,
        params_dict: Optional[Dict[str, torch.nn.Parameter]] = None,
        is_eagle: bool = False,
    ):
        # NextN reuses this loader without inheriting ModelNextForCausalLM.
        loader_cls = ModelNextForCausalLM
        config = self.config
        n_routed = config.n_routed_experts
        num_fused_shared = self.num_fused_shared_experts

        nextn_prefix: Optional[str] = None
        if is_nextn:
            if not hasattr(config, "num_nextn_predict_layers"):
                raise ValueError("num_nextn_predict_layers is not in the config")
            assert config.num_nextn_predict_layers == 1, (
                "Only 1 nextn layer is supported"
            )
            layer_id = 0 if config.num_hidden_layers == 1 else config.num_hidden_layers
            nextn_prefix = f"model.layers.{layer_id}"

        if num_fused_shared > 0:

            def iter_weights_with_fused_shared_experts(
                weights: Iterable[Tuple[str, torch.Tensor]],
            ) -> Iterable[Tuple[str, torch.Tensor]]:
                for name, weight in weights:
                    match = loader_cls._SHARED_EXPERTS_PATTERN.match(name)
                    if match:
                        layer_id = int(match.group(1))
                        suffix = match.group(2)
                        name = f"model.layers.{layer_id}.mlp.experts.{self.config.n_routed_experts}.{suffix}"
                    yield name, weight

            weights = iter_weights_with_fused_shared_experts(weights)

        expert_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=n_routed + num_fused_shared,
        )

        fuse_qkv_a_proj = getattr(config, "q_lora_rank", None) is not None
        cached_a_proj: Dict[str, torch.Tensor] = {}

        if params_dict is None:
            params_dict = dict(self.named_parameters())

        qc = self.quant_config
        if (
            qc is not None
            and qc.get_name() in loader_cls._AWQ_LIKE_QUANT_METHOD
            and not any("attn" in m for m in qc.modules_to_not_convert)
        ):
            fused_cat_dim = 1
        else:
            fused_cat_dim = 0

        num_nextn_skip = getattr(config, "num_nextn_predict_layers", 0) or 0
        weight_names = []

        for name, loaded_weight in weights:
            weight_names.append(name)

            if is_nextn or is_eagle:
                if nextn_prefix and not name.startswith(nextn_prefix):
                    continue
                if nextn_prefix is not None:
                    if "shared_head.head" in name or "embed_tokens" in name:
                        continue
                    if any(w in name for w in loader_cls._NEXTN_SPEC_NAMES):
                        name = name.replace(nextn_prefix, "model")
                    else:
                        name = name.replace(nextn_prefix, "model.decoder")
            else:
                if num_nextn_skip > 0 and name.startswith("model.layers"):
                    parts = name.split(".")
                    if len(parts) >= 3 and int(parts[2]) >= config.num_hidden_layers:
                        continue

            # Skip keys outside this PP rank's layer partition (noise otherwise)
            if name.startswith("model.layers") and hasattr(self, "start_layer"):
                parts = name.split(".")
                if len(parts) >= 3:
                    layer_id = int(parts[2])
                    if not (self.start_layer <= layer_id < self.end_layer):
                        continue

            if "rotary_emb.inv_freq" in name:
                continue

            # ---- for stacked params ----
            matched = False
            for param_name, weight_name, shard_id in loader_cls._STACKED_PARAMS_MAPPING:
                if weight_name not in name or "mlp.experts" in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                matched = True
                break
            if matched:
                continue

            # ---- for expert params ----
            is_expert_weight = False
            for param_name, weight_name, expert_id, shard_id in expert_mapping:
                if weight_name not in name:
                    continue
                is_expert_weight = True
                mapped = name.replace(weight_name, param_name)
                if mapped in params_dict:
                    param = params_dict[mapped]
                    param.weight_loader(
                        param,
                        loaded_weight,
                        mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                break
            if is_expert_weight:
                continue

            # ---- for other params ----
            if name.endswith(".bias") and name not in params_dict:
                continue
            if is_eagle and name in loader_cls._EAGLE_IGNORE_NAMES:
                continue

            if fuse_qkv_a_proj and ("q_a_proj" in name or "kv_a_proj_with_mqa" in name):
                cached_a_proj[name] = loaded_weight
                is_q = "q_a_proj" in name
                q_name = (
                    name if is_q else name.replace("kv_a_proj_with_mqa", "q_a_proj")
                )
                kv_name = (
                    name.replace("q_a_proj", "kv_a_proj_with_mqa") if is_q else name
                )
                if q_name not in cached_a_proj or kv_name not in cached_a_proj:
                    continue

                fused = torch.cat(
                    [cached_a_proj.pop(q_name), cached_a_proj.pop(kv_name)],
                    dim=fused_cat_dim,
                )
                target = name.replace(
                    "q_a_proj" if is_q else "kv_a_proj_with_mqa",
                    "fused_qkv_a_proj_with_mqa",
                )
                if target not in params_dict:
                    continue
                param = params_dict[target]
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, fused)
                continue

            if ("k_scale" in name or "v_scale" in name) and name not in params_dict:
                name = name.replace("_proj", "attn_mqa")

            if name not in params_dict:
                logger.warning(f"Parameter {name} not found in params_dict")
                continue

            param = params_dict[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, loaded_weight)

        if getattr(config, "mla", False):
            self.post_load_weights(is_nextn=is_nextn, weight_names=weight_names)

    def post_load_weights(self, is_nextn: bool = False, weight_names=None):
        if not is_nextn:
            local_full_layers = {
                layer_id
                for layer_id in self.config.full_attention_layer_ids
                if self.model.start_layer <= layer_id < self.model.end_layer
            }
            if weight_names is None:
                weight_names = [
                    f"model.layers.{layer_id}.self_attn.kv_b_proj.weight"
                    for layer_id in local_full_layers
                ]
            else:
                weight_names = [
                    name
                    for name in weight_names
                    if "kv_b_proj" in name
                    and int(name.split(".")[2]) in local_full_layers
                ]
        DeepseekV2WeightLoaderMixin.post_load_weights(
            self, is_nextn=is_nextn, weight_names=weight_names
        )

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        if callable(getattr(self.model, "load_kv_cache_scales", None)):
            self.model.load_kv_cache_scales(quantization_param_path)
        else:
            logger.warning(
                f"{self.model.__class__} does not support loading scaling factors."
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
            self.model.layers_to_capture = [val + 1 for val in layer_ids]


class Glm5NextForCausalLM(ModelNextForCausalLM):
    pass


class Glm5NextForConditionalGeneration(GlmVisualEncoderMixin, ModelNextForCausalLM):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: ModelNextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        config.encoder_only = getattr(config, "encoder_only", False)
        config.language_only = getattr(config, "language_only", False)

        if config.encoder_only:
            nn.Module.__init__(self)
            self.fuse_qkv_a_proj = config.q_lora_rank is not None
            if self.fuse_qkv_a_proj:
                self.packed_modules_mapping["fused_qkv_a_proj_with_mqa"] = [
                    "q_a_proj",
                    "kv_a_proj_with_mqa",
                ]
            self.pp_group = get_pp_group()
            self.config = config
            self.tp_size = get_tensor_model_parallel_world_size()
            if quant_config is not None:
                quant_config.update_packed_modules_mapping(self.packed_modules_mapping)
            self.quant_config = quant_config
            self.num_fused_shared_experts = 0
            self.use_dsa = is_deepseek_dsa(config)
            self.model = None
            self.lm_head = PPMissingLayer()
            self.logits_processor = LogitsProcessor(config)
            self._routed_experts_weights_of_layer = LazyValue(lambda: {})
            self.capture_aux_hidden_states = False
            self.nsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
            if self.nsa_enable_prefill_cp:
                self.cp_rank = get_parallel().attn_cp_rank
                self.cp_size = get_parallel().attn_cp_size
            else:
                self.cp_rank = self.cp_size = None
            get_attn_tp_context().init_context(
                config.q_lora_rank, self.use_dsa, config.mhc
            )
        else:
            super().__init__(config=config, quant_config=quant_config, prefix=prefix)
            self.config.encoder_only = config.encoder_only
            self.config.language_only = config.language_only

        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder
        vision_config = getattr(config, "vision_config", None)
        if isinstance(vision_config, dict):
            vision_config = Glm4vVisionConfig(**vision_config)
            self.config.vision_config = vision_config

        if (
            not self.config.language_only
            and vision_config is not None
            and isinstance(vision_config, Glm4vVisionConfig)
        ):
            vision_utils.update_vit_attn_dummy_heads_config(self.config)
            vision_quant_config = quant_config
            quant_ignore = getattr(quant_config, "ignore", ())
            if (
                quant_config is not None
                and any(
                    name in quant_ignore for name in ("visual", "model.visual")
                )
            ):
                # GLM checkpoints can exclude the complete vision subtree.
                # Leaf-name matching misses packed qkv/gate-up projections and
                # would cast their floating-point weights into quantized
                # parameters. This applies to both INT8 and compressed FP8.
                vision_quant_config = None
            self.visual = Glm5NextVisionModel(
                vision_config,
                quant_config=vision_quant_config,
                prefix=add_prefix("visual", prefix),
                use_data_parallel=self.use_data_parallel,
                text_config=getattr(config, "text_config", None),
                swiglu_limit=getattr(config, "swiglu_limit", None),
            )
        else:
            self.visual = None

        self.is_mrope_enabled = "mrope_section" in (self.config.rope_scaling or {})

    @property
    def start_layer(self):
        return getattr(getattr(self, "model", None), "start_layer", 0)

    @property
    def end_layer(self):
        model = getattr(self, "model", None)
        end_layer = getattr(model, "end_layer", None)
        if end_layer is not None:
            return end_layer
        cfg = getattr(model, "config", None) or self.config
        return int(getattr(cfg, "num_hidden_layers", 0))

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("encoder_only GLM5 Next VLM cannot run language forward")

        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        if self.nsa_enable_prefill_cp:
            if can_cp_split(len(input_ids), self.cp_size, self.use_dsa, forward_batch):
                forward_batch.attn_cp_metadata = (
                    prepare_glm5_next_context_parallel_metadata(
                        len(input_ids),
                        self.cp_rank,
                        self.cp_size,
                        forward_batch,
                    )
                )

        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden_states = general_mm_embed_routine(
                input_ids=input_ids,
                forward_batch=forward_batch,
                language_model=self.model,
                multimodal_model=self,
                positions=positions,
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

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        is_nextn: bool = False,
        params_dict: Optional[Dict[str, torch.nn.Parameter]] = None,
        is_eagle: bool = False,
    ):
        def iter_normalized_weights():
            for name, loaded_weight in weights:
                if "language_model." in name:
                    name = name.replace("language_model.", "")
                if "model.visual." in name:
                    name = name.replace("model.visual.", "visual.")
                if self.config.language_only and "visual" in name:
                    continue
                if "visual" in name:
                    name = name.replace("attn.qkv.", "attn.qkv_proj.")
                    if getattr(self.config, "vision_config", None) is not None:
                        loaded_weight = vision_utils.pad_vit_attn_dummy_heads(
                            self.config, name, loaded_weight
                        )
                yield name, loaded_weight

        if self.config.encoder_only:
            return super().load_weights(
                (
                    (name, loaded_weight)
                    for name, loaded_weight in iter_normalized_weights()
                    if "visual" in name
                ),
                is_nextn=is_nextn,
                params_dict=params_dict,
                is_eagle=is_eagle,
            )

        return super().load_weights(
            iter_normalized_weights(),
            is_nextn=is_nextn,
            params_dict=params_dict,
            is_eagle=is_eagle,
        )

    def post_load_weights(self, is_nextn: bool = False, weight_names=None):
        if self.model is None:
            return
        return super().post_load_weights(is_nextn=is_nextn, weight_names=weight_names)


EntryClass = [Glm5NextForCausalLM, Glm5NextForConditionalGeneration]
