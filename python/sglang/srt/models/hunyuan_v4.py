# Hunyuan-V4 (Hy4-preview) target model.
#
# Ported from an internal fork's ``models/hunyuan_v4.py`` (itself adapted
# from upstream sgl-project/sglang PR #36805 "Support Hy4-preview"), adjusted
# to this fork's API surface:
#   * fork-specific fused iHC backends are dropped; iHC runs the eager torch path.
#   * a custom vocab embedding -> upstream ``VocabParallelEmbedding`` with
#     ``get_embedding_tp_kwargs()``.
#   * The CP helpers live in ``layers/attention/dsa/utils`` and
#     ``layers/utils/cp_utils`` here, and the CP metadata attribute is
#     ``attn_cp_metadata`` (not ``nsa_cp_metadata``).
#   * The DSA top-k carry between shared-indexer layers (and to/from the MTP
#     draft) goes through ``IndexTopKShareState``.
#   * attn-tp rank/size/cp rank/size come from ``get_parallel()``.
#
# Architecture notes carried over verbatim from the port:
#   * iHC replaces the flat (T, H) residual with a (T, hc_mult, H) stream, so
#     the model never uses LayerCommunicator; DP attention works because the
#     stream is per-token and only ever *split*, never gathered.
#   * HYV4Attention is a sparse MLA with an elementwise output gate
#     (gated MLA) and learnable attention sinks.

import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_attn_tp_group, get_pp_group
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.utils import (
    can_dsa_cp_split,
    dsa_use_prefill_cp,
    is_dsa_enable_prefill_cp,
    is_dsa_prefill_cp_round_robin_split,
)
from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
from sglang.srt.layers.communicator import (
    AttentionInputs,
    enable_moe_dense_fully_dp,
    get_attn_tp_context,
)
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather_into_tensor,
    attn_tp_reduce_scatter_tensor,
    get_local_dp_buffer,
    is_dp_attention_enabled,
)
from sglang.srt.layers.hy4_ihc_tilelang import (
    try_tilelang_ihc_head,
    try_tilelang_ihc_post,
    try_tilelang_ihc_pre,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.utils.cp_utils import (
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    cp_split_and_rebuild_position,
    prepare_context_parallel_metadata,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    get_embedding_tp_kwargs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    AttnForwardMethod,
)
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2AttentionMLA,
    DeepseekV2MLP,
    DeepseekV2MoE,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import BumpAllocator, add_prefix, is_cuda

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()


def hyv4_dp_attn_scattered() -> bool:
    """Does this layout run the layer stack in ``ScatterMode.SCATTERED``?

    HYV4 does not use LayerCommunicator: the iHC residual is a
    ``(T, hc_mult, H)`` stream that no ScatterMode can describe. What makes DP
    attention work anyway is that the stream is per-token, so it can simply be
    *split* along the token dim and never needs to be communicated. With
    ``attn_tp_size > 1`` the whole layer stack therefore runs on this rank's
    ``1 / attn_tp_size`` slice of the DP rank's tokens (SCATTERED), and only the
    2D hidden state is gathered to TP_ATTN_FULL for attention and scattered back
    afterwards -- the same two collectives per layer the DeepSeek path uses.

    With ``attn_tp_size == 1`` (dp_size == tp_size) SCATTERED and TP_ATTN_FULL
    are the same layout and no communication is needed at all.
    """
    return is_dp_attention_enabled() and get_parallel().attn_tp_size > 1


def hyv4_attn_tp_split(tensor: torch.Tensor) -> torch.Tensor:
    """TP_ATTN_FULL -> SCATTERED along the token dim. A view, no communication.

    Mirrors ``CommunicateSummableTensorPairFn._scatter``. Works on the 2D hidden
    state and on the 3D iHC stream alike, which is the whole trick.
    """
    attn_tp_size = get_parallel().attn_tp_size
    return tensor.tensor_split(attn_tp_size)[get_parallel().attn_tp_rank]


def hyv4_attn_tp_gather(hidden_states: torch.Tensor) -> torch.Tensor:
    """SCATTERED -> TP_ATTN_FULL.

    Mirrors ``CommunicateSimpleFn._scattered_to_tp_attn_full``. Each DP rank's
    token count is pre-padded to a multiple of ``attn_tp_size``, so the local
    DP buffer is exactly ``attn_tp_size`` times the input.
    """
    output = get_local_dp_buffer(get_attn_tp_group())
    attn_tp_all_gather_into_tensor(output, hidden_states)
    return output


def hyv4_attn_tp_reduce_scatter(hidden_states: torch.Tensor) -> torch.Tensor:
    """Partial TP_ATTN_FULL -> summed SCATTERED, in a single collective.

    Mirrors ``CommunicateWithAllReduceAndLayerNormFn._scatter_hidden_states_and_residual``:
    the attention output is left unreduced (``reduce_results=False``) so that one
    reduce_scatter does both the attn-tp sum and the token-dim split. The sum has
    to complete before the iHC post gate, which is non-linear -- that is why
    HYV4 cannot instead fold the reduction into a later dp_gather.
    """
    output = hyv4_attn_tp_split(hidden_states)
    attn_tp_reduce_scatter_tensor(output, hidden_states)
    return output


def permute_hyv4_indexer_weight(name, loaded_weight, config):
    """Move the indexer's rope slice to the front of each head group.

    The HYV4 checkpoint stores the indexer head layout as ``[nope | rope]``,
    while the DSA indexer in this repo consumes ``[rope | nope]``.
    """
    if ".self_attn.indexer.wq_b." in name:
        group_count = config.index_n_heads
    elif any(
        key in name
        for key in (
            ".self_attn.indexer.wk.",
            ".self_attn.indexer.k_norm.",
        )
    ):
        group_count = 1
    else:
        return loaded_weight

    shape = loaded_weight.shape
    loaded_weight = loaded_weight.reshape(
        group_count,
        config.index_head_dim,
        *shape[1:],
    )
    rope_dim = config.qk_rope_head_dim
    return torch.cat(
        (
            loaded_weight[:, -rope_dim:],
            loaded_weight[:, :-rope_dim],
        ),
        dim=1,
    ).reshape(shape)


def hyv4_linear_scale_suffix(model: nn.Module) -> str:
    """Suffix to append to a checkpoint ``*.weight_scale`` name.

    The HYV4 checkpoints always name linear weight scales ``weight_scale``, but
    fp8 block quantization registers them as ``weight_scale_inv`` while the
    per-channel int4/int8 methods keep ``weight_scale``.
    """
    for name, _ in model.named_parameters():
        if name.endswith(".weight_scale_inv"):
            return "_inv"
    return ""


class HYV4HCPreLayer(nn.Module):
    """Produce the iHC pre/post gates and reduce the (T, hc_mult, H) stream.

    Runs the eager torch path: the gate projection is a float32
    ``[2 * hc_mult, hc_mult * hidden]`` GEMM over the flattened stream with an
    RMS scale, and the reduce is a per-token weighted sum. (The internal fork fused
    both into MHC kernels; a similar fusion could reuse this fork's DeepSeek-V4
    MHC kernels with an identity comb later.)
    """

    def __init__(self, config: PretrainedConfig, prefix: str):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.magnitude = config.hc_magnitude
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_fn = ReplicatedLinear(
            config.hidden_size * config.hc_mult,
            2 * config.hc_mult,
            bias=False,
            params_dtype=torch.float32,
            prefix=add_prefix("hc_fn", prefix),
        )
        self.hc_scale = nn.Parameter(torch.empty(2, dtype=torch.float32))
        self.hc_base = nn.Parameter(
            torch.empty(2 * config.hc_mult, dtype=torch.float32)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rms_weight: Optional[torch.Tensor] = None,
        rms_eps: float = 0.0,
    ):
        shape = hidden_states.shape
        use_tilelang = envs.SGLANG_OPT_HY4_IHC_TILELANG.get()
        fused = None
        if use_tilelang:
            fused = try_tilelang_ihc_pre(
                hidden_states, self.hc_fn.weight, self.hc_scale, self.hc_base,
                self.rms_norm_eps, self.hc_eps, self.magnitude,
            )
        if fused is not None:
            reduced, post = fused
        else:
            flat = hidden_states.flatten(1).float()
            scale = torch.rsqrt(
                flat.square().mean(-1, keepdim=True) + self.rms_norm_eps
            )
            gates = self.hc_fn(flat)[0] * scale
            pre = (
                torch.sigmoid(
                    gates[..., : self.hc_mult] * self.hc_scale[0]
                    + self.hc_base[: self.hc_mult]
                )
                + self.hc_eps
            )
            post = (
                self.magnitude
                * torch.sigmoid(
                    gates[..., self.hc_mult :] * self.hc_scale[1]
                    + self.hc_base[self.hc_mult :]
                )
                + self.hc_eps
            )
            reduced = torch.sum(pre.unsqueeze(-1) * hidden_states.reshape(shape), dim=1)
            reduced = reduced.to(hidden_states.dtype)
        if rms_weight is not None:
            reduced_float = reduced.float()
            reduced = (
                reduced_float
                * torch.rsqrt(
                    reduced_float.square().mean(dim=-1, keepdim=True) + rms_eps
                )
                * rms_weight.float()
            ).to(hidden_states.dtype)
        return reduced, post


class HYV4HCLayer(nn.Module):
    """One interleaved-hyper-connection (iHC) junction of a decoder layer."""

    def __init__(self, config: PretrainedConfig, prefix: str):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_pre = HYV4HCPreLayer(config, add_prefix("hc_pre", prefix))

    def prepare_input(self, hidden_states: torch.Tensor):
        if hidden_states.ndim == 3:
            return hidden_states
        if hidden_states.ndim != 2:
            raise RuntimeError(
                f"iHC expects a 2D or 3D tensor, got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[-1] == self.hidden_size:
            return hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        if hidden_states.shape[-1] == self.hidden_size * self.hc_mult:
            return hidden_states.reshape(-1, self.hc_mult, self.hidden_size)
        raise RuntimeError(
            "iHC input width must equal hidden_size or hc_mult * hidden_size"
        )

    def pre(self, hidden_states: torch.Tensor, norm: Optional[RMSNorm] = None):
        if hidden_states.shape[0] == 0:
            # DP attention hands idle ranks a zero-token batch, and the layer
            # still has to run for the MoE all-to-all. Every GEMM/RMSNorm below
            # is a no-op on zero rows, so short-circuit while keeping the shape
            # contract: a 2D hidden state, a (T, hc_mult) fp32 gate, the 3D
            # stream.
            return (
                hidden_states[:, 0],
                hidden_states.new_zeros((0, self.hc_mult), dtype=torch.float32),
                hidden_states,
            )
        reduced, post = self.hc_pre(hidden_states)
        if norm is not None:
            reduced = norm(reduced)
        return reduced, post, hidden_states

    def post(self, output, residual, post):
        fused = try_tilelang_ihc_post(output, residual, post)
        if fused is not None:
            return fused
        result = post.float().unsqueeze(-1) * output.float().unsqueeze(1)
        return (result + residual.float()).to(output.dtype)

    def post_pre(self, output, residual, post, next_layer, norm):
        next_residual = self.post(output, residual, post)
        next_residual = next_layer.prepare_input(next_residual)
        return next_layer.pre(next_residual, norm)


class HYV4HCHeadLayer(nn.Module):
    """Collapse the hc_mult residual streams back to one hidden state."""

    def __init__(self, config: PretrainedConfig, prefix: str):
        super().__init__()
        self.config = config
        self.hc_head_fn = ReplicatedLinear(
            config.hc_mult * config.hidden_size,
            config.hc_mult,
            bias=False,
            params_dtype=torch.float32,
            prefix=add_prefix("hc_head_fn", prefix),
        )
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(
            torch.empty(config.hc_mult, dtype=torch.float32)
        )

    def forward(self, hidden_states: torch.Tensor, norm: Optional[RMSNorm] = None):
        fused = try_tilelang_ihc_head(
            hidden_states,
            self.hc_head_fn.weight,
            self.hc_head_scale,
            self.hc_head_base,
            self.config.rms_norm_eps,
            self.config.hc_eps,
        )
        if fused is not None:
            return fused if norm is None else norm(fused)
        shape = hidden_states.shape
        flat = hidden_states.flatten(1).float()
        scale = torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + self.config.rms_norm_eps
        )
        gates = self.hc_head_fn(flat)[0] * scale
        gates = (
            torch.sigmoid(gates * self.hc_head_scale + self.hc_head_base)
            + self.config.hc_eps
        )
        output = torch.sum(gates.unsqueeze(-1) * flat.reshape(shape), dim=1)
        output = output.to(hidden_states.dtype)
        return output if norm is None else norm(output)


class HYV4Attention(DeepseekV2AttentionMLA):
    """Sparse MLA with an elementwise output gate and learnable attention sinks."""

    def __init__(
        self,
        config,
        layer_id,
        quant_config=None,
        prefix="",
        alt_stream=None,
        is_nextn=False,
    ):
        rope_parameters = config.rope_parameters
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_parameters["rope_theta"],
            rope_scaling=(
                None
                if rope_parameters.get("rope_type") == "default"
                else rope_parameters
            ),
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            # Plain TP (no DP attention): attn_tp_size == tp_size, so o_proj's
            # own all-reduce over the global TP group is the right one and HYV4
            # needs it, having no LayerCommunicator to do it later.
            # DP attention with attn_tp_size > 1: leave the output partial so
            # ``hyv4_attn_tp_reduce_scatter`` can fuse the attn-tp sum with the
            # split back to SCATTERED.
            # DP attention with attn_tp_size == 1: o_proj has tp_size == 1 and
            # skips the reduction either way.
            reduce_results=not is_dp_attention_enabled(),
            layer_id=layer_id,
            prefix=prefix,
            alt_stream=alt_stream,
            is_nextn=is_nextn,
        )
        attn_tp_rank = get_parallel().attn_tp_rank
        attn_tp_size = get_parallel().attn_tp_size
        self.linear_gate = ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * config.v_head_dim,
            bias=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("linear_gate", prefix),
        )
        self.local_gate_width = self.num_local_heads * config.v_head_dim
        if self.linear_gate.output_size_per_partition != self.local_gate_width:
            raise ValueError(
                "HYV4 attention gate shard width must match the local attention "
                f"output width: {self.linear_gate.output_size_per_partition} != "
                f"{self.local_gate_width}"
            )
        self.learnable_sink_param = nn.Parameter(
            torch.empty(self.num_local_heads, dtype=torch.float32)
        )
        self.learnable_sink_param.weight_loader = self._sink_weight_loader

    @staticmethod
    def _sink_weight_loader(param, loaded_weight):
        attn_tp_size = get_parallel().attn_tp_size
        heads = loaded_weight.shape[0] // attn_tp_size
        start = get_parallel().attn_tp_rank * heads
        param.data.copy_(loaded_weight[start : start + heads].float())

    def prepare_attention_output_gate(self, hidden_states):
        return self.linear_gate(hidden_states)[0]

    def apply_attention_output_gate(self, attn_out, gate):
        if gate.shape != attn_out.shape:
            raise ValueError(
                "HYV4 projected attention gate shape must match the local attention "
                f"output shape: {tuple(gate.shape)} != {tuple(attn_out.shape)}"
            )
        return attn_out * torch.sigmoid(gate)

    def dispatch_attn_forward_method(self, forward_batch: ForwardBatch):
        # The DSA handler picks MHA_ONE_SHOT vs MLA off ``use_mha`` on the live
        # backend. HYV4 only has the sparse MLA path, so pin it.
        backend = get_attn_backend()
        backend = getattr(backend, "primary", backend)
        if getattr(backend, "use_mha", False) is not False:
            backend.use_mha = False
        method = super().dispatch_attn_forward_method(forward_batch)
        # if method != AttnForwardMethod.MLA:
        if method != AttnForwardMethod.MLA and method != AttnForwardMethod.MLA_ROCM:
            raise RuntimeError(
                f"HYV4 requires the sparse MLA attention path, got {method}"
            )
        return method


class HYV4DecoderLayer(nn.Module):
    def __init__(self, config, layer_id, quant_config=None, prefix="", alt_stream=None):
        super().__init__()
        self.self_attn = HYV4Attention(
            config,
            layer_id,
            quant_config,
            add_prefix("self_attn", prefix),
            alt_stream,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if config.mlp_layer_types[layer_id] == "dense":
            # With DP attention the dense layer must stay fully data parallel:
            # a TP-sharded MLP would all-reduce over the whole TP group, which
            # spans DP ranks holding different tokens.
            mlp_tp_rank, mlp_tp_size = (
                (0, 1) if enable_moe_dense_fully_dp() else (None, None)
            )
            self.mlp = DeepseekV2MLP(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )
        else:
            self.mlp = DeepseekV2MoE(
                config,
                layer_id,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                alt_stream=alt_stream,
            )
            if hasattr(self.mlp, "shared_experts"):
                # The SwiGLU limit applies to routed experts only. The shared
                # expert bakes it into ``act_fn`` at construction time and
                # ``forward`` only ever consults ``act_fn.limit``, so clearing
                # the bookkeeping attribute on its own leaves the clamp active.
                self.mlp.shared_experts.swiglu_limit = None
                self.mlp.shared_experts.act_fn.limit = None
        self.hc_attn_layer = HYV4HCLayer(config, add_prefix("hc_attn_layer", prefix))
        self.hc_mlp_layer = HYV4HCLayer(config, add_prefix("hc_mlp_layer", prefix))
        self.dp_attn_scattered = hyv4_dp_attn_scattered()

    def forward(
        self,
        positions,
        hidden_states,
        forward_batch,
        zero_allocator,
        prev_topk_indices=None,
    ):
        # The iHC stream (and therefore ``residual``) stays in whatever layout
        # the layer received: SCATTERED when attn_tp_size > 1, otherwise the DP
        # rank's full local tokens. It is never communicated.
        hidden_states = self.hc_attn_layer.prepare_input(hidden_states)
        hidden_states, post, residual = self.hc_attn_layer.pre(
            hidden_states, self.input_layernorm
        )
        if self.dp_attn_scattered:
            # Attention needs every token of the DP rank (TP_ATTN_FULL).
            hidden_states = hyv4_attn_tp_gather(hidden_states)
        get_attn_tp_context().set_attn_inputs(
            AttentionInputs(
                hidden_states, forward_batch, self.self_attn.prepare_qkv_latent
            )
        )
        hidden_states = self.self_attn(
            positions,
            hidden_states,
            forward_batch,
            zero_allocator,
            prev_topk_indices=prev_topk_indices,
        )
        if isinstance(hidden_states, tuple):
            hidden_states, topk_indices = hidden_states
        else:
            topk_indices = None
        if self.dp_attn_scattered:
            # o_proj left the output partial; sum over the attn-tp group and go
            # back to SCATTERED so it lines up with ``residual`` again.
            hidden_states = hyv4_attn_tp_reduce_scatter(hidden_states)
        hidden_states, post, residual = self.hc_attn_layer.post_pre(
            hidden_states,
            residual,
            post,
            self.hc_mlp_layer,
            self.post_attention_layernorm,
        )
        if isinstance(self.mlp, DeepseekV2MoE):
            hidden_states = self.mlp(hidden_states, forward_batch)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = self.hc_mlp_layer.post(hidden_states, residual, post)
        return hidden_states, topk_indices


class HYV4Model(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        if get_pp_group().world_size != 1:
            raise ValueError("HYV4 pipeline parallelism is not supported")
        if is_dp_attention_enabled():
            # iHC replaces the flat (T, H) residual with a (T, hc_mult, H)
            # stream, which LayerCommunicator's scatter/gather modes cannot
            # describe. That stream is purely per-token though, and it never
            # crosses a communication boundary: every collective lives inside
            # ``self_attn`` (the o_proj all-reduce) or inside the MoE. So DP
            # attention works as long as the stream only ever has to be *split*
            # along the token dim, never gathered:
            #   * attn_tp_size == 1 -- SCATTERED == TP_ATTN_FULL, nothing to do.
            #   * attn_tp_size > 1  -- the layer stack runs SCATTERED and only
            #     the 2D hidden state crosses the attn-tp group, see
            #     ``hyv4_dp_attn_scattered``.
            # ScatterMode.FULL is what cannot be expressed, so the MoE has to
            # dispatch tokens itself and the dense layer has to be fully DP.
            if get_moe_a2a_backend().is_none():
                raise ValueError(
                    "HYV4 DP attention requires an all-to-all MoE backend "
                    "(--moe-a2a-backend deepep): with the plain EP/TP MoE the "
                    "sparse layers run in ScatterMode.FULL, which needs a "
                    "dp_gather/dp_scatter around the iHC residual."
                )
            if not enable_moe_dense_fully_dp():
                raise ValueError(
                    "HYV4 DP attention requires --moe-dense-tp-size 1: a "
                    "TP-sharded dense MLP would all-reduce over the whole TP "
                    "group and mix tokens across DP ranks."
                )
            if get_global_server_args().enable_two_batch_overlap:
                raise ValueError(
                    "HYV4 does not support two-batch overlap: TBO slices the "
                    "batch through LayerScatterModes.middle_residual_mode, "
                    "which the iHC stream has no equivalent of."
                )
        self.dp_attn_scattered = hyv4_dp_attn_scattered()
        self.config = config
        self.start_layer = 0
        self.end_layer = config.num_hidden_layers
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
            params_dtype=(
                torch.float16 if get_global_server_args().dtype == "float16" else None
            ),
            **get_embedding_tp_kwargs(),
        )
        self.alt_stream = (
            torch.cuda.Stream()
            if _is_cuda or envs.SGLANG_NPU_USE_MULTI_STREAM.get()
            else None
        )
        self.layers = nn.ModuleList(
            [
                HYV4DecoderLayer(
                    config,
                    i,
                    quant_config,
                    add_prefix(f"layers.{i}", prefix),
                    self.alt_stream,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.hc_head = HYV4HCHeadLayer(config, add_prefix("hc_head", prefix))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        self.cp_rank = get_parallel().attn_cp_rank
        self.cp_size = get_parallel().attn_cp_size

    def _maybe_prepare_prefill_cp(self, input_ids, forward_batch):
        """Build the DSA CP metadata for this batch, mirroring DeepseekV4Model.

        The metadata has no producer outside the model: every CP-capable model
        sets it itself before its layer loop, and ``dsa_use_prefill_cp``
        returns False while it is None. Without this the CP split below is
        silently skipped.
        """
        if not (
            self.dsa_enable_prefill_cp
            and forward_batch.extend_seq_lens_cpu is not None
        ):
            return False
        if not can_dsa_cp_split(len(input_ids), self.cp_size, True, forward_batch):
            return False
        forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(
            len(input_ids),
            self.cp_rank,
            self.cp_size,
            forward_batch.seq_lens_cpu.tolist(),
            extend_seqs_len=forward_batch.extend_seq_lens_cpu,
        )
        if is_dsa_prefill_cp_round_robin_split():
            # In round-robin-split mode the CP metadata decides the local token
            # order, so the attention/indexer metadata built before
            # model.forward() must be rebuilt to match.
            attn_backend = get_attn_backend()
            metadata = attn_backend.forward_metadata
            core_meta = metadata.core_attn_metadata
            core_meta.apply_cp_reindex()
            core_meta.init_flashmla_related(is_prefill=True)
            if metadata.indexer_metadata is not None:
                metadata.indexer_metadata = (
                    attn_backend.init_forward_metadata_indexer(core_meta)
                )
        return True

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        # Prefill CP boundary sits at the model level on purpose. iHC carries a
        # 3D (T, hc_mult, H) residual between layers, which the 2D CP helpers
        # cannot describe, but both the embedding output and the hc_head output
        # are flat (T, H). Splitting here and gathering after hc_head keeps the
        # iHC stream entirely CP-local, so no per-layer CP metadata is needed.
        self._maybe_prepare_prefill_cp(input_ids, forward_batch)
        use_cp = self.dsa_enable_prefill_cp and dsa_use_prefill_cp(forward_batch)
        if use_cp:
            if not getattr(self, "_cp_logged", False):
                self._cp_logged = True
                logger.info(
                    "HYV4 prefill CP engaged: tokens=%d cp_size=%d",
                    hidden_states.shape[0],
                    self.cp_size,
                )
            hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
            positions = cp_split_and_rebuild_position(forward_batch, positions)
        if self.dp_attn_scattered:
            # ScatterMode.model_input_output() is TP_ATTN_FULL, but the layer
            # stack runs SCATTERED. This split is a view, not a collective.
            hidden_states = hyv4_attn_tp_split(hidden_states)
        zero_allocator = BumpAllocator(
            buffer_size=2 * len(self.layers),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        # Shared-indexer layers reuse the previous layer's topk, so carry it
        # through the loop the same way DeepseekV2Model.forward does.
        index_topk_share = IndexTopKShareState(forward_batch, None)
        for layer in self.layers:
            hidden_states, topk_indices = layer(
                positions,
                hidden_states,
                forward_batch,
                zero_allocator,
                prev_topk_indices=index_topk_share.topk_indices,
            )
            index_topk_share.update(topk_indices)
        index_topk_share.publish()
        if forward_batch.forward_mode.is_idle():
            # Collapse the 3D iHC stream to the 2D model output without running
            # hc_head / norm on a zero-token batch. Zero tokens means the two
            # layouts coincide, so no gather is needed either.
            return hidden_states[:, 0]
        hidden_states = self.hc_head(hidden_states, self.norm)
        if self.dp_attn_scattered:
            # Back to TP_ATTN_FULL for the logits processor / lm_head.
            hidden_states = hyv4_attn_tp_gather(hidden_states)
        if use_cp:
            hidden_states = cp_all_gather_rerange_output(
                hidden_states.contiguous(),
                self.cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )
        return hidden_states


class HYV4ForCausalLM(nn.Module, DeepseekV2WeightLoaderMixin):
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()
        self.model = HYV4Model(config, quant_config, add_prefix("model", prefix))
        self.num_fused_shared_experts = max(
            (
                layer.mlp.num_fused_shared_experts
                for layer in self.model.layers
                if isinstance(layer.mlp, DeepseekV2MoE)
            ),
            default=0,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            # Checkpoint weights stay in bf16; LogitsProcessor emits fp32 logits
            # because config.enable_lm_head_fp32 is set.
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds=input_embeds
        )
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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        scale_suffix = hyv4_linear_scale_suffix(self)

        def mapped_weights():
            for name, loaded_weight in weights:
                if name.startswith("model.mtp_layers."):
                    continue
                loaded_weight = permute_hyv4_indexer_weight(
                    name, loaded_weight, self.config
                )
                if name.endswith((".hc_fn", ".hc_head_fn")):
                    name += ".weight"
                if name.endswith(".weight_scale"):
                    name += scale_suffix
                yield name, loaded_weight

        self.do_load_weights(mapped_weights())


EntryClass = [HYV4ForCausalLM]
