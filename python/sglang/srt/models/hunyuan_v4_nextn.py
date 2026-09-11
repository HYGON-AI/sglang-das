# Hunyuan-V4 (Hy4-preview) MTP/NEXTN draft model.
#
# Ported from an internal fork's ``models/hunyuan_v4_nextn.py`` (itself
# adapted from upstream sgl-project/sglang PR #36805), adjusted to this fork's
# API surface (see hunyuan_v4.py for the full list). The MTP top-k carry goes
# through ``IndexTopKShareState`` here instead of the fork-local
# ``reuse_mtp_topk_indices`` flag.

import copy
from typing import Iterable, Tuple

import torch
from torch import nn

from sglang.srt.distributed import get_pp_group
from sglang.srt.environ import envs
from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
from sglang.srt.layers.communicator import AttentionInputs, get_attn_tp_context
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    get_embedding_tp_kwargs,
)
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.srt.models.hunyuan_v4 import (
    HYV4Attention,
    hyv4_attn_tp_gather,
    hyv4_attn_tp_reduce_scatter,
    hyv4_attn_tp_split,
    hyv4_dp_attn_scattered,
    hyv4_linear_scale_suffix,
    permute_hyv4_indexer_weight,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import BumpAllocator, add_prefix, is_cuda

_is_cuda = is_cuda()


def _mtp_quant_config(quant_config):
    """Re-point the checkpoint's MTP ignore entries at the draft decoder."""
    if quant_config is None:
        return None

    quant_config = copy.deepcopy(quant_config)
    decoder_prefix = "model.decoder"

    def normalize_name(name):
        for mtp_prefix in (
            "model.mtp.layers.0",
            "model.mtp_layers.0",
            "mtp.layers.0",
            "mtp_layers.0",
        ):
            name = name.replace(mtp_prefix, decoder_prefix)
        return name

    ignored_layers = getattr(quant_config, "ignored_layers", None)
    if ignored_layers is not None:
        quant_config.ignored_layers = list(
            dict.fromkeys(normalize_name(name) for name in ignored_layers)
        )

    ignored_modules = getattr(quant_config, "ignore", None)
    if ignored_modules is not None:
        quant_config.ignore = list(
            dict.fromkeys(normalize_name(name) for name in ignored_modules)
        )

    # Compressed-tensors applies this override before consulting its ignore list.
    if hasattr(quant_config, "linear_fp8_config"):
        quant_config.linear_fp8_config = None

    return quant_config


class HYV4MTPDecoderLayer(nn.Module):
    """The draft layer is a plain pre-norm layer: the checkpoint carries no iHC."""

    def __init__(self, config, quant_config=None, prefix="", alt_stream=None):
        super().__init__()
        self.self_attn = HYV4Attention(
            config,
            0,
            quant_config,
            add_prefix("self_attn", prefix),
            alt_stream,
            is_nextn=True,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = DeepseekV2MoE(
            config,
            0,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
            alt_stream=alt_stream,
            is_nextn=True,
        )
        if hasattr(self.mlp, "shared_experts"):
            # Same as the trunk layers: ``forward`` reads ``act_fn.limit``, so the
            # bookkeeping attribute alone would leave the clamp active.
            self.mlp.shared_experts.swiglu_limit = None
            self.mlp.shared_experts.act_fn.limit = None
        self.dp_attn_scattered = hyv4_dp_attn_scattered()

    def forward(
        self,
        positions,
        hidden_states,
        forward_batch,
        zero_allocator,
        prev_topk_indices=None,
    ):
        # The draft layer carries no iHC, so this is the plain DeepSeek shape:
        # enter in TP_ATTN_FULL (that is what eh_proj produced), run attention
        # there, then drop to SCATTERED for the MoE the way
        # DeepseekV2DecoderLayer does when mlp_mode == SCATTERED.
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
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
            hidden_states = hyv4_attn_tp_reduce_scatter(hidden_states)
            residual = hyv4_attn_tp_split(residual)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states, forward_batch)
        return hidden_states, residual, topk_indices


class HYV4ModelNextN(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
            params_dtype=(
                torch.float16 if get_global_server_args().dtype == "float16" else None
            ),
            **get_embedding_tp_kwargs(),
        )
        self.enorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.alt_stream = (
            torch.cuda.Stream()
            if _is_cuda or envs.SGLANG_NPU_USE_MULTI_STREAM.get()
            else None
        )
        self.decoder = HYV4MTPDecoderLayer(
            config,
            quant_config,
            add_prefix("decoder", prefix),
            self.alt_stream,
        )
        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.dp_attn_scattered = hyv4_dp_attn_scattered()

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        if hidden_states.shape[0] > 0:
            hidden_states = self.eh_proj(
                torch.cat(
                    (
                        self.enorm(hidden_states),
                        self.hnorm(forward_batch.spec_info.hidden_states),
                    ),
                    dim=-1,
                )
            )
        zero_allocator = BumpAllocator(
            buffer_size=2,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        index_topk_share = IndexTopKShareState.from_mtp_carry(forward_batch)
        hidden_states, residual, topk_indices = self.decoder(
            positions,
            hidden_states,
            forward_batch,
            zero_allocator,
            prev_topk_indices=index_topk_share.topk_indices,
        )
        index_topk_share.update(topk_indices)
        index_topk_share.publish()
        # Join alt_stream before returning: leftover work there can create a
        # cross-stream dependency that deadlocks the next target-model graph
        # replay. Same guard as DeepseekModelNextN.
        if self.alt_stream is not None:
            torch.cuda.current_stream().wait_stream(self.alt_stream)
        if forward_batch.forward_mode.is_idle():
            return hidden_states
        hidden_states, _ = self.shared_head.norm(hidden_states, residual)
        if self.dp_attn_scattered:
            # SCATTERED after the MoE; the logits processor wants TP_ATTN_FULL.
            hidden_states = hyv4_attn_tp_gather(hidden_states)
        return hidden_states


class HYV4ForCausalLMNextN(nn.Module, DeepseekV2WeightLoaderMixin):
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()
        self.model = HYV4ModelNextN(
            config, _mtp_quant_config(quant_config), prefix=add_prefix("model", prefix)
        )
        self.num_fused_shared_experts = self.model.decoder.mlp.num_fused_shared_experts
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch):
        hidden_states = self.model(input_ids, positions, forward_batch)
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
        # The HYV4 checkpoint names the draft layer "model.mtp_layers.0", while
        # do_load_weights maps a "model.layers.<nextn_layer_id>" prefix onto
        # "model.decoder". Ask the mixin for that prefix instead of recomputing
        # it.
        layer_prefix = self._initialize_nextn_conf(True).nextn_layer_prefix
        scale_suffix = hyv4_linear_scale_suffix(self)

        def mapped_weights():
            for name, loaded_weight in weights:
                if not name.startswith("model.mtp_layers.0."):
                    continue
                name = name.replace("model.mtp_layers.0", layer_prefix)
                if name.endswith(".final_layernorm.weight"):
                    name = name.replace(
                        ".final_layernorm.weight", ".shared_head.norm.weight"
                    )
                loaded_weight = permute_hyv4_indexer_weight(
                    name, loaded_weight, self.config
                )
                if name.endswith(".weight_scale"):
                    name += scale_suffix
                yield name, loaded_weight

        self.do_load_weights(mapped_weights(), is_nextn=True)

    def post_load_weights(self, is_nextn=True, weight_names=None):
        super().post_load_weights(is_nextn=True, weight_names=weight_names)


EntryClass = [HYV4ForCausalLMNextN]
