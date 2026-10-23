"""Inference-only WeLMV4 NextN (MTP) speculative draft model.

Ported from the official WeLM SGLang tree. The draft shares the target's
token embedding, over-encoding (OE) embeddings and LM head through
``set_embed_and_head`` instead of allocating its own copies (the OE tables
alone are ~15 GiB per TP rank).
"""

import logging
import os
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.welmv4 import (
    Qwen2MoeDecoderLayer,
    WelmV4FusedRMSNorm,
    WeLMV4MoeForCausalLM,
    hash_input_ids_vectorized,
    welm_mtp_mirror_kv,
    welm_spec_n_gram_input_ids,
    _empty_welm_kv_mirror_states,
    _get_welm_kv_mirror_states,
    _set_welm_kv_mirror_states,
    _can_fuse_embed_ar,
    welm_embed_with_oe,
)
from sglang.srt.utils import add_prefix, is_cuda

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()


def _welm_trace_draft_logits(forward_batch, input_ids, positions, logits_output):
    """Optional low-volume draft-logit/state trace for MTP diagnosis."""
    path = os.getenv("WELM_TRACE_DRAFT_LOGITS")
    if not path:
        return
    try:
        states = getattr(forward_batch, "model_specific_states", None) or {}
        mirror = states.get("welm_kv_mirror_states", {})
        trace = {
            "mode": str(forward_batch.forward_mode),
            "input_shape": tuple(input_ids.shape),
            "position_shape": tuple(positions.shape),
            "input_head": input_ids[:8].detach().cpu().tolist(),
            "position_head": positions[:8].detach().cpu().tolist(),
            "state_keys": sorted(str(k) for k in mirror),
            "state_shapes": {
                str(k): (tuple(v[0].shape), tuple(v[1].shape))
                for k, v in mirror.items()
            },
        }
        logits = getattr(logits_output, "next_token_logits", None)
        if isinstance(logits, torch.Tensor) and logits.numel() > 0:
            rows = [0] if logits.shape[0] == 1 else [0, logits.shape[0] - 1]
            trace["logit_rows"] = []
            for row in rows:
                values, ids = torch.topk(logits[row].float(), k=5)
                trace["logit_rows"].append(
                    {
                        "row": row,
                        "ids": ids.detach().cpu().tolist(),
                        "values": values.detach().cpu().tolist(),
                    }
                )
        with open(path, "a", encoding="utf-8") as f:
            import json
            f.write(json.dumps(trace, sort_keys=True) + "\n")
    except Exception:
        logger.exception("WeLM draft trace failed")


class WeLMV4ModelNextN(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        draft_model_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        # Shared with the target model via set_embed_and_head.
        self.embed_tokens = None
        self.oe_embed = None
        self.oe_gate_up_proj = None
        self.oe_grams = config.oe_grams
        self.oe_vocab_sizes = config.oe_vocab_sizes
        self.scale_seq_times = getattr(config, "scale_seq_times", 0)

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=True)

        # Each checkpoint MTP layer is a complete module (eh_proj, norms,
        # block, head).  Multi-layer EAGLE builds one draft model per step and
        # passes ``draft_model_idx`` so that step runs only its own layer.
        mtp_layer_ids = (
            [draft_model_idx]
            if draft_model_idx is not None
            else range(config.num_nextn_predict_layers)
        )
        self.alt_stream = torch.cuda.Stream() if _is_cuda else None
        self.decoder_layers = nn.ModuleList(
            [
                Qwen2MoeDecoderLayer(
                    config,
                    i,
                    quant_config=quant_config,
                    is_nextn=True,
                    prefix=add_prefix("decoder", prefix),
                    alt_stream=self.alt_stream,
                )
                for i in mtp_layer_ids
            ]
        )
        # Multi-layer EAGLE step k feeds inputs shifted by ``1 + k`` tokens
        # relative to their positions (each step rotates the window once more).
        self.draft_token_shift = 1 + (draft_model_idx or 0)

        self.shared_head = nn.Module()
        self.shared_head.norm = WelmV4FusedRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def _oe_embedding(self, input_ids, forward_batch, hidden_states):
        return self._oe_embedding_from(
            input_ids, forward_batch.n_gram_input_ids, hidden_states
        )

    def _oe_embedding_from(self, input_ids, n_gram_input_ids, hidden_states):
        max_n = max(self.oe_grams)
        grams = []
        for n in range(2, max_n + 1):
            gram = n_gram_input_ids.get_gram(n) if n_gram_input_ids is not None else None
            grams.append(gram if gram is not None else torch.zeros_like(input_ids))
        input_ids_ngram = []
        input_ids_ngram_tmp = input_ids
        for g in range(1, max_n):
            input_ids_ngram_tmp = input_ids_ngram_tmp + grams[g - 1] * (
                self.vocab_size**g
            )
            input_ids_ngram.append(hash_input_ids_vectorized(input_ids_ngram_tmp))
        emb_ngram = []
        for i, vs in enumerate(self.oe_vocab_sizes):
            emb_ngram.append(self.oe_embed[i](input_ids_ngram[self.oe_grams[i] - 2] % vs))
        emb_new, _ = self.oe_gate_up_proj(torch.cat(emb_ngram, dim=-1))
        return (hidden_states + emb_new) / 2.0

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        if len(self.oe_grams) > 0:
            forward_batch.n_gram_input_ids = welm_spec_n_gram_input_ids(
                input_ids, positions, forward_batch, is_draft=True,
                draft_shift=self.draft_token_shift,
            )
        if (
            input_embeds is None
            and len(self.oe_grams) > 0
            and input_ids.shape[0] > 0
            and _can_fuse_embed_ar(self.embed_tokens, self.oe_embed)
        ):
            hidden_states = welm_embed_with_oe(
                self.embed_tokens, self.oe_embed, self.oe_gate_up_proj, input_ids,
                forward_batch.n_gram_input_ids, self.vocab_size, self.oe_grams,
                self.oe_vocab_sizes,
            )
        else:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            if len(self.oe_grams) > 0:
                hidden_states = self._oe_embedding(input_ids, forward_batch, hidden_states)

        if len(self.oe_grams) > 0:
            mode = forward_batch.forward_mode
            # Target verify already publishes the mirror K/V tensors needed by
            # draft-extend.  Reconstructing them from the n-gram ring here
            # loses the accepted-tree row mapping after variable rejection and
            # can overwrite the target state with rows for a different path.
            # Ordinary draft decode has no target forward in the same call, so
            # it still needs the ring-based reconstruction fallback.
            if (
                os.environ.get("WELM_MTP_DECODE_MIRROR", "1") == "1"
                and not mode.is_idle()
                and not mode.is_extend_without_speculative()
                and input_ids.shape[0] > 0
                and (
                    not mode.is_draft_extend_v2()
                    or not _get_welm_kv_mirror_states(forward_batch)
                )
            ):
                welm_mtp_mirror_kv(
                    input_ids,
                    positions,
                    forward_batch,
                    self.embed_tokens,
                    self._oe_embedding_from,
                )

        if (
            forward_batch.enable_kv_mirror
            and forward_batch.forward_mode.is_extend_without_speculative()
        ):
            # The KV-mirror target prefill only returns the last hidden state
            # of each request; keep the matching embedding rows.
            main_hidden_states = forward_batch.spec_info.hidden_states
            if (
                main_hidden_states is not None
                and hidden_states.shape[0] != main_hidden_states.shape[0]
            ):
                if forward_batch.custom_last_index is None:
                    forward_batch.custom_last_index = (
                        torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
                    )
                hidden_states = hidden_states[forward_batch.custom_last_index]

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

        residual = None
        final_experts_output = None
        final_shared_output = None
        kv_mirror_states = _get_welm_kv_mirror_states(forward_batch)
        with get_global_expert_distribution_recorder().disable_this_region():
            for layer in self.decoder_layers:
                hidden_states, residual, kv_mirror_states = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                    kv_mirror_states,
                )
                final_experts_output = getattr(layer, "final_mlp_experts_output", None)
                final_shared_output = getattr(layer, "final_mlp_shared_output", None)
        _set_welm_kv_mirror_states(forward_batch, kv_mirror_states)

        pre_norm_hidden_states = hidden_states
        if not forward_batch.forward_mode.is_idle():
            if residual is not None:
                if final_experts_output is not None:
                    hidden_states = final_experts_output.float() + residual.float()
                    if final_shared_output is not None:
                        hidden_states = hidden_states + final_shared_output.float()
                else:
                    hidden_states = hidden_states.float() + residual.float()
                hidden_states = hidden_states.to(self.shared_head.norm.weight.dtype)
            pre_norm_hidden_states = hidden_states
            hidden_states, _ = self.shared_head.norm(hidden_states)
        return hidden_states, pre_norm_hidden_states


class WeLMV4MoeForCausalLMNextN(WeLMV4MoeForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        draft_model_idx: Optional[int] = None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.pp_group = get_pp_group()
        self.draft_model_idx = draft_model_idx

        self.model = WeLMV4ModelNextN(
            config,
            quant_config,
            prefix=add_prefix("model", prefix),
            draft_model_idx=draft_model_idx,
        )
        self.lm_head = None
        # Some checkpoints train a dedicated MTP output head
        # (``shared_head.head``) that differs from the target lm_head; drafting
        # with the target head then lowers acceptance.  Load it and decide in
        # set_embed_and_head whether it is needed.
        self.mtp_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("mtp_head", prefix),
        )
        self._mtp_head_loaded = False
        self.logits_processor = LogitsProcessor(config)
        super().post_init_after_load_weights(is_nextn=True)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        **kwargs,
    ):
        hidden_states, pre_norm_hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds
        )
        # Like the target model, hand the next draft step the hidden state
        # before the final norm: that is what hnorm was trained on.
        aux_hidden_states = None
        if (
            forward_batch.capture_hidden_mode.need_capture()
            and os.environ.get("WELM_MTP_DRAFT_PRENORM", "1") == "1"
        ):
            aux_hidden_states = [pre_norm_hidden_states]
        logits_output = self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )
        _welm_trace_draft_logits(forward_batch, input_ids, positions, logits_output)
        logits_output.model_specific_states = forward_batch.model_specific_states
        return logits_output

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        if self.draft_model_idx is not None:
            weights = select_mtp_layer_weights(
                self.config, weights, self.draft_model_idx
            )
        super().load_weights(self._take_mtp_head(weights), is_nextn=True)

    def _take_mtp_head(self, weights):
        # After select_mtp_layer_weights the active MTP layer is always the
        # first one; the base NextN loader skips shared_head.head.
        head_name = (
            f"model.layers.{self.config.num_target_hidden_layers}"
            ".shared_head.head.weight"
        )
        for name, tensor in weights:
            if name == head_name:
                param = self.mtp_head.weight
                getattr(param, "weight_loader", default_weight_loader)(param, tensor)
                self._mtp_head_loaded = True
                continue
            yield name, tensor

    def set_embed_and_head(self, embed, head):
        super().set_embed_and_head(embed, head)
        if self.mtp_head is None:
            return
        if self._mtp_head_loaded and not torch.equal(
            self.mtp_head.weight, head.weight
        ):
            self.lm_head = self.mtp_head
        else:
            # Identical to (or absent in favour of) the target head: share it
            # and release the duplicate.
            self.mtp_head = None
            torch.cuda.empty_cache()


def select_mtp_layer_weights(config, weights, draft_model_idx: int):
    """Keep only MTP layer ``num_target + idx`` and present it as the first
    MTP layer, which the base NextN loader maps onto ``decoder_layers.0`` and
    the model-level eh_proj / enorm / hnorm / shared_head."""
    num_target = config.num_target_hidden_layers
    first_prefix = f"model.layers.{num_target}."
    keep_prefix = f"model.layers.{num_target + draft_model_idx}."
    mtp_prefixes = tuple(
        f"model.layers.{num_target + i}."
        for i in range(config.num_nextn_predict_layers)
    )
    for name, tensor in weights:
        if name.startswith(keep_prefix):
            yield first_prefix + name[len(keep_prefix):], tensor
        elif not name.startswith(mtp_prefixes):
            yield name, tensor


EntryClass = WeLMV4MoeForCausalLMNextN
