"""Inference-only Qwen4-Exp MTP speculative decoding."""

import os
import copy
import logging
from contextlib import ExitStack
from typing import Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.hyperconnection import GroupedGemmaRMSNorm
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP, _mtp_quant_config
from sglang.srt.models.qwen4_exp import Qwen4ExpModel
from sglang.srt.runtime_context import get_model, get_parallel
from sglang.srt.utils import add_prefix, is_npu

logger = logging.getLogger(__name__)


class Qwen4ExpForCausalLMMTP(Qwen3_5ForCausalLMMTP):
    # The draft's quant config takes packed_modules_mapping from this class
    # (model_loader._get_quantization_config). Without it the fused qkv_proj /
    # shared_expert.gate_up_proj never match the checkpoint's per-shard
    # w8a8_include globs, so their int8 weights load as BF16 and the scales
    # are dropped.
    packed_modules_mapping = Qwen3_5ForCausalLM.packed_modules_mapping

    # The Qwen4Exp MTP layer is a full decoder layer whose routed experts are
    # SlimQuant-packed exactly like the backbone, so the checkpoint stores
    # ``mlp.experts.gate_up_proj(.weight_scale)`` / ``mlp.experts.down_proj(
    # .weight_scale)`` rather than the BF16 ``w13_weight`` layout the base
    # class assumes.  Mirror ``Qwen4ExpModel``'s table: scale entries first.
    fused_expert_params_mapping = [
        (
            "experts.w13_weight_scale",
            "experts.gate_up_proj.weight_scale",
            0,
            "w1",
        ),
        ("experts.w2_weight_scale", "experts.down_proj.weight_scale", 0, "w2"),
        ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
        ("experts.w2_weight", "experts.down_proj", 0, "w2"),
    ]

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)

        self.is_multimodal = hasattr(config, "text_config")
        if self.is_multimodal:
            config = config.text_config

        # Deepcopy so MTP-only mutations below don't leak into the main model.
        config = copy.deepcopy(config)
        config.num_hidden_layers = 1
        config.layer_types = ["full_attention"]
        config.full_attention_interval = 1
        config.ple_layer_ids = []
        # The bundled MTP layer ships a QSA indexer, but on HIP the draft's QSA
        # extend indices are not numerically reliable (the same limitation that
        # forces ``Skip HIP Qwen4Exp MTP draft CUDA graphs``), and every draft
        # ends up rejected.  Hiding ``indexer_n_heads`` makes this single layer
        # build native dense attention instead; the drafter only ever sees
        # ``num_draft_tokens`` rows, so dense attention costs nothing
        # measurable, while the indexer weights are simply skipped by the
        # loader.
        if os.environ.get("SGLANG_MTP_DENSE_ATTN", "0") == "1":
            if getattr(config, "indexer_n_heads", None) is not None:
                config.indexer_n_heads = None

        quant_config = _mtp_quant_config(quant_config)

        self.config = config
        self.tp_size = get_parallel().tp_size
        self.quant_config = quant_config
        self.pp_group = get_pp_group()
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        self._mtp_input_fusion = self._init_mtp_input_fusion(config, prefix)

        self.model = Qwen4ExpModel(
            config,
            quant_config,
            prefix=add_prefix("mtp", prefix),
            is_nextn=True,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("model.shared_head.head", prefix),
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

    def _init_pre_fc_norms(self, config: PretrainedConfig) -> None:
        self.pre_fc_norm_embedding = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # mHC residual is hc_count independent streams. Checkpoint
        # pre_fc_norm_hidden.weight is stored flat (10240,) but the four
        # hidden_size slices have disjoint statistics; RMS over the concatenated
        # 10240-d vector mixes branches and collapses MTP accept rate.
        if self.hc_count > 1 and os.environ.get("SGLANG_MTP_PLAIN_PFC_NORM", "0") == "1":
            self.pre_fc_norm_hidden = GemmaRMSNorm(
                self.hc_count * config.hidden_size, eps=config.rms_norm_eps
            )
        elif self.hc_count > 1:
            self.pre_fc_norm_hidden = GroupedGemmaRMSNorm(
                self.hc_count * config.hidden_size,
                eps=config.rms_norm_eps,
                group_size=config.hidden_size,
            )
        else:
            self.pre_fc_norm_hidden = GemmaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

    def _init_linear_projections(
        self, config: PretrainedConfig, prefix: str
    ) -> None:
        # These projections are quantized in INT8-w8a8-ngram checkpoints.
        # Building plain nn.Linear modules silently copies the raw int8 values
        # into BF16 weights and drops their weight_scale tensors in load_weights.
        self.fc_embedding = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=add_prefix("mtp.fc_embedding", prefix),
        )
        self.fc_hidden = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=add_prefix("mtp.fc_hidden", prefix),
        )

    def _init_standard_fusion(self, config: PretrainedConfig, prefix: str):
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self._init_pre_fc_norms(config)
        return self._fuse_standard

    def _init_mtp_input_fusion(self, config: PretrainedConfig, prefix: str):
        if self.hc_count <= 1:
            return self._init_standard_fusion(config, prefix)

        self._init_linear_projections(config, prefix)
        self._init_pre_fc_norms(config)
        return self._fuse_residual_linear_shared

    def _fuse_residual_linear_shared(
        self, input_embeds: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        input_embeds, _ = self.fc_embedding(
            self.pre_fc_norm_embedding(input_embeds)
        )
        orig_shape = hidden_states.shape
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        decoder_view = hidden_states.view(
            *hidden_states.shape[:-1], self.hc_count, self.hidden_size
        )
        # Quantized Linear backends are not required to preserve arbitrary
        # leading dimensions.  In particular, the HCU hipBLASLt W8A8 wrapper
        # takes ``shape[-2]`` as GEMM M, so passing [tokens, hc_count, hidden]
        # would only project one hc_count-wide slice and then broadcast it over
        # tokens.  Flatten every logical row explicitly and restore the HC
        # layout after the projection.
        encoder_inputs, _ = self.fc_hidden(
            decoder_view.reshape(-1, self.hidden_size)
        )
        encoder_inputs = encoder_inputs.view_as(decoder_view)
        return (input_embeds.unsqueeze(-2) + encoder_inputs).view(orig_shape)

    def _fuse_standard(
        self, input_embeds: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        input_embeds = self.pre_fc_norm_embedding(input_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        return self.fc(torch.cat((input_embeds, hidden_states), dim=-1))

    def _npu_quant_context(self):
        exit_stack = ExitStack()
        if (
            is_npu()
            and self.quant_config is None
            and get_model().quantization is not None
        ):
            exit_stack.enter_context(envs.SGLANG_DEEPEP_BF16_DISPATCH.override(True))
            exit_stack.enter_context(
                envs.DEEP_NORMAL_MODE_USE_INT8_QUANT.override(False)
            )
        return exit_stack

    def _prepare_input_embeds(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert input_embeds is None
        input_embeds = forward_batch.mm_input_embeds
        if (
            forward_batch.forward_mode.is_extend()
            and forward_batch.contains_mm_inputs()
            and not forward_batch.forward_mode.is_draft_extend_v2()
        ):
            assert input_embeds is not None
            last_indices = (
                forward_batch.extend_start_loc + forward_batch.extend_seq_lens - 1
            ).long()
            input_embeds[last_indices] = self.model.embed_tokens(
                input_ids[last_indices]
            )
        if input_embeds is None:
            input_embeds = self.model.embed_tokens(input_ids)
        return input_embeds

    def _set_hc_logits_hidden_states(
        self,
        logits_output,
        hc_hidden_states: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> None:
        if hc_hidden_states is None:
            return

        # EAGLE v2 stores one hidden state per request in the future map.
        # When draft extend emits a token-shaped HC tensor, keep only the
        # last token per request so the overlap cache sees [bs, hidden].
        if (
            not forward_batch.forward_mode.is_draft_extend_v2()
            and forward_batch.extend_seq_lens is not None
            and hc_hidden_states.shape[0] != forward_batch.extend_seq_lens.shape[0]
        ):
            last_index = (
                torch.cumsum(forward_batch.extend_seq_lens.to(torch.int64), dim=0) - 1
            )
            hc_hidden_states = hc_hidden_states[last_index]

        assert hc_hidden_states.shape[-1] == self.hc_count * self.hidden_size
        logits_output.hidden_states = hc_hidden_states

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        with self._npu_quant_context():
            input_embeds = self._prepare_input_embeds(
                input_ids, forward_batch, input_embeds
            )
            hidden_states = forward_batch.spec_info.hidden_states
            if not forward_batch.forward_mode.is_idle():
                hidden_states = self._mtp_input_fusion(input_embeds, hidden_states)

            with get_global_expert_distribution_recorder().disable_this_region():
                model_output = self.model(
                    input_ids,
                    positions,
                    forward_batch,
                    hidden_states,
                )

            hc_hidden_states = None
            if isinstance(model_output, tuple):
                hidden_states, hc_hidden_states = model_output
            else:
                hidden_states = model_output

        logits_output = self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )
        self._set_hc_logits_hidden_states(logits_output, hc_hidden_states, forward_batch)
        return logits_output


# Same HIP-graph issue as the target decoder: fusion GEMMs + QSA sitting in
# one captured graph replay as token-0 / garbage drafts. Breakable capture
# eager-runs the whole MTP forward; the graph remains the outer decode/extend
# scaffolding.
Qwen4ExpForCausalLMMTP.forward = eager_on_graph(True)(Qwen4ExpForCausalLMMTP.forward)


EntryClass = [Qwen4ExpForCausalLMMTP]
