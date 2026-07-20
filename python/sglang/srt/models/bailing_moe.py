# coding=utf-8
# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""SGLang BailingMoE model."""

import logging
from typing import Iterable, List, Optional, Tuple, Union

from sglang.srt.batch_overlap.single_batch_overlap import SboFlags, compute_overlap_args
from sglang.srt.batch_overlap.two_batch_overlap import MaybeTboDeepEPDispatcher
from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher, CombineInput, DispatchOutput
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_pp_group,
    parallel_state,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    enable_moe_dense_fully_dp,
)
from sglang.srt.layers.dp_attention import (
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_deepep_mode,
    get_moe_a2a_backend,
    should_skip_post_experts_all_reduce,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.token_dispatcher import DeepEPDispatcher
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import filter_moe_weight_param_global_expert, is_sbo_enabled
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import (
    apply_qk_norm,
    create_fused_set_kv_buffer_arg,
    enable_fused_set_kv_buffer,
)
from sglang.srt.runtime_context import (
    get_forward,
    get_parallel,
    get_server_args,
    get_stream,
)
from sglang.srt.utils import (
    add_prefix,
    get_bool_env_var,
    is_cuda,
    is_hcu,
    is_non_idle_and_non_empty,
    make_layers,
)

LoraConfig = None
logger = logging.getLogger(__name__)
_is_cuda = is_cuda()
_is_hcu = is_hcu()

_use_fused_bailing_silu_mul_fp8_quant = get_bool_env_var("SGLANG_USE_FUSED_BAILING_SILU_MUL_FP8_QUANT")
_use_fused_bailing_rms_rotary = get_bool_env_var("SGLANG_USE_FUSED_RMS_ROTARY")
_use_fused_bailing_rms_quant = get_bool_env_var("SGLANG_USE_FUSED_BAILING_RMS_QUANT")
_use_fused_bailing_moe_sum_add = get_bool_env_var("SGLANG_USE_FUSED_BAILING_MOE_SUM_ADD", "true")

if _is_hcu:
    from lightop import rms_rotary_embedding_fuse_with_kv_store


class BailingMoEMLP(nn.Module):
    def __init__(
        self,
        intermediate_size: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: Optional[bool] = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size

        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=config.use_bias,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=config.use_bias,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

        if config.hidden_act != "silu":
            raise ValueError("Unsupported activation. Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> torch.Tensor:
        hidden_states_tensor = (
            hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
        )
        if (self.tp_size == 1) and hidden_states_tensor.shape[0] == 0:
            return hidden_states

        gate_up, _ = self.gate_up_proj(hidden_states)
        if _use_fused_bailing_silu_mul_fp8_quant:
            hidden_states, _ = self.down_proj(
                gate_up,
                forward_batch=forward_batch,
                use_fused_silu_mul_fp8_quant=True,
            )
        else:
            hidden_states = self.act_fn(gate_up)
            hidden_states, _ = self.down_proj(
                hidden_states,
                forward_batch=forward_batch,
            )
        return hidden_states


class BailingMoEGate(nn.Module):
    def __init__(
        self,
        config,
        params_dtype: Optional[torch.dtype] = None,
        prefix: str = "",
    ):
        super().__init__()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        if _is_hcu and self.params_dtype == torch.float32:
            self.weight = nn.Parameter(
                torch.empty(
                    (config.num_experts, config.hidden_size),
                    dtype=torch.bfloat16,
                ),
            )
        else:
            self.weight = nn.Parameter(
                torch.empty(
                    (config.num_experts, config.hidden_size),
                    dtype=self.params_dtype,
                ),
            )

        if getattr(config, "moe_router_enable_expert_bias", False):
            self.expert_bias = nn.Parameter(
                torch.empty((config.num_experts,), dtype=torch.float32),
            )
        else:
            self.expert_bias = None

    def forward(self, hidden_states):
        logits = F.linear(hidden_states.to(self.weight.dtype), self.weight, None).to(
            hidden_states.dtype
        )
        return logits


class BailingMoESparseMoeBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.tp_size = get_parallel().tp_size
        self.moe_ep_size = get_parallel().moe_ep_size
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size
        self.num_shared_experts = config.num_shared_experts
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.score_function = getattr(config, "score_function", None)
        self.num_fused_shared_experts = (
            0
            if get_server_args().disable_shared_experts_fusion or get_moe_a2a_backend().is_deepep()
            else config.num_shared_experts
        )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        # Gate always runs at half / full precision for now.
        router_dtype = getattr(config, "router_dtype", None)
        if router_dtype is None:
            self.router_dtype = None
        elif router_dtype == "fp32":
            self.router_dtype = torch.float32
        else:
            self.router_dtype = torch.bfloat16

        # TODO global_server_args.ep_num_redundant_experts is used for eplb, not supported now
        assert get_server_args().ep_num_redundant_experts == 0
        # check group topk
        self.num_expert_group = getattr(config, "n_group", 0)
        self.topk_group = getattr(config, "topk_group", 0)
        if self.num_expert_group > 0 or self.topk_group > 0:
            assert (
                self.num_expert_group > 0
                and 0 < self.topk_group <= self.num_expert_group
            )
            self.use_grouped_topk = True
        else:
            self.num_expert_group = self.topk_group = None
            self.use_grouped_topk = False

        self.num_experts = (
            config.num_experts + get_server_args().ep_num_redundant_experts
        )

        self.gate = BailingMoEGate(
            config=config,
            params_dtype=self.router_dtype,
            prefix=add_prefix("gate", prefix),
        )
        self.correction_bias = (
            self.gate.expert_bias.data if self.gate.expert_bias is not None else None
        )

        if self.score_function is not None:
            assert (
                self.score_function == "softmax" and self.correction_bias is None
            ) or (
                self.score_function == "sigmoid" and self.correction_bias is not None
            ), "score_function and correction_bias should be in 2 combination (softmax, None) or (sigmoid, not None)"

        # scaling factor for fused shared experts on AMD-platform.
        fused_shared_experts_scaling_factor = None
        if self.moe_ep_size > 1 and self.num_fused_shared_experts > 0:
            # if enable_ep_moe tp_szie == ep_size, every gpu get shared experts gemm output
            # so we scale with 1 / self.moe_ep_size in ep mode which will make it equalation as in tp mode
            # with fused_shared_experts
            fused_shared_experts_scaling_factor = 1.0 / float(self.moe_ep_size)

        self.topk = TopK(
            top_k=self.top_k + self.num_fused_shared_experts,
            layer_id=self.layer_id,
            renormalize=self.norm_topk_prob,
            use_grouped_topk=self.use_grouped_topk,
            num_expert_group=self.num_expert_group,
            num_fused_shared_experts=self.num_fused_shared_experts,
            topk_group=self.topk_group,
            correction_bias=self.correction_bias,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            fused_shared_experts_scaling_factor=fused_shared_experts_scaling_factor,
        )

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.num_experts
            + self.num_fused_shared_experts
            + get_server_args().ep_num_redundant_experts,
            num_fused_shared_experts=self.num_fused_shared_experts,
            top_k=self.top_k + self.num_fused_shared_experts,
            layer_id=self.layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            prefix=add_prefix("experts", prefix),
        )
        # shared expert
        if config.num_shared_experts is not None and self.num_fused_shared_experts == 0:
            if hasattr(config, "moe_shared_expert_intermediate_size"):
                intermediate_size = config.moe_shared_expert_intermediate_size
            else:
                intermediate_size = config.moe_intermediate_size
            intermediate_size *= config.num_shared_experts
            # disable tp for shared experts when enable deepep moe
            self.shared_experts = BailingMoEMLP(
                intermediate_size=intermediate_size,
                config=config,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
                **(
                    dict(tp_rank=0, tp_size=1)
                    if get_moe_a2a_backend().is_deepep()
                    else {}
                ),
            )
        # dispatcher
        if get_moe_a2a_backend().is_deepep():
            # TODO: we will support tp < ep in the future
            self.ep_size = get_parallel().tp_size

            self.deepep_dispatcher = DeepEPDispatcher(
                group=parallel_state.get_tp_group().device_group,
                router_topk=self.top_k,
                permute_fusion=True,
                num_experts=self.num_experts,
                num_local_experts=config.num_experts // self.tp_size,
                hidden_size=config.hidden_size,
                params_dtype=config.torch_dtype,
                deepep_mode=get_deepep_mode(),
                async_finish=True,  # TODO
                return_recv_hook=True,
            )
        self._fuse_shared_experts_inside_sbo = SboFlags.fuse_shared_experts_inside_sbo()

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        moe_i_q: Optional[torch.Tensor] = None,
        moe_i_s: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not get_moe_a2a_backend().is_deepep():
            return self.forward_normal(
                hidden_states,
                moe_i_q=moe_i_q,
                moe_i_s=moe_i_s,
            )
        else:
            return self.forward_deepep(
                hidden_states, forward_batch, moe_i_q=moe_i_q, moe_i_s=moe_i_s
            )

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
            and filter_moe_weight_param_global_expert(
                name, x, self.experts.num_local_experts
            )
        ]

    def _forward_shared_experts(
        self,
        hidden_states: torch.Tensor,
        moe_i_q: Optional[torch.Tensor] = None,
        moe_i_s: Optional[torch.Tensor] = None,
    ):
        shared_output = None
        if (self.num_shared_experts > 0) and (self.num_fused_shared_experts == 0):
            shared_input = (
                (moe_i_q, moe_i_s)
                if (moe_i_q is not None and moe_i_s is not None)
                else hidden_states
            )
            shared_output = self.shared_experts(shared_input)
        return shared_output

    def _forward_router_experts(
        self,
        hidden_states: torch.Tensor,
        shared_output: Optional[torch.Tensor] = None,
        moe_i_q: Optional[torch.Tensor] = None,
        moe_i_s: Optional[torch.Tensor] = None,
    ):
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        return self.experts(
            hidden_states,
            topk_output,
            shared_output=shared_output,
            i_q=moe_i_q,
            i_s=moe_i_s,
        )

    def forward_normal_dual_stream(
        self,
        hidden_states: torch.Tensor,
        moe_i_q: Optional[torch.Tensor] = None,
        moe_i_s: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        if moe_i_q is not None and moe_i_s is not None:
            shared_output = self._forward_shared_experts(
                hidden_states,
                moe_i_q=moe_i_q.clone(),
                moe_i_s=moe_i_s.clone(),
            )
        else:
            shared_output = self._forward_shared_experts(hidden_states.clone())

        with torch.cuda.stream(self.alt_stream):
            router_output = self._forward_router_experts(
                hidden_states, moe_i_q=moe_i_q, moe_i_s=moe_i_s
            )
        current_stream.wait_stream(self.alt_stream)

        return router_output, shared_output

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        moe_i_q: Optional[torch.Tensor] = None,
        moe_i_s: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)

        if (
            self.alt_stream is not None
            and hidden_states.shape[0] > 0
            and self.num_fused_shared_experts == 0
            and get_is_capture_mode()
        ):
            final_hidden_states, shared_output = self.forward_normal_dual_stream(
                hidden_states, moe_i_q=moe_i_q, moe_i_s=moe_i_s
            )
        else:
            if self.num_fused_shared_experts > 0:
                final_hidden_states = self._forward_router_experts(
                    hidden_states,
                    moe_i_q=moe_i_q,
                    moe_i_s=moe_i_s,
                )
            else:
                shared_output = self._forward_shared_experts(
                    hidden_states,
                    moe_i_q=moe_i_q,
                    moe_i_s=moe_i_s,
                )

                if not _use_fused_bailing_moe_sum_add:
                    final_hidden_states = self._forward_router_experts(
                        hidden_states,
                        moe_i_q=moe_i_q,
                        moe_i_s=moe_i_s,
                    )
                    if self.num_shared_experts > 0:
                        final_hidden_states = final_hidden_states + shared_output
                else:
                    final_hidden_states = self._forward_router_experts(
                        hidden_states,
                        shared_output,
                        moe_i_q=moe_i_q,
                        moe_i_s=moe_i_s,
                    )

        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True,
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states.view(num_tokens, hidden_size)

    def forward_deepep(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        moe_i_q: Optional[torch.Tensor] = None,
        moe_i_s: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shared_output = None
        forward_mode = forward_batch.forward_mode
        sbo_enabled_flag = self._fuse_shared_experts_inside_sbo
        sbo_overlap_dispatch_flag = (
            sbo_enabled_flag and SboFlags.enable_dispatch_shared_one_stream_overlap()
        )
        sbo_overlap_combine_flag = (
            sbo_enabled_flag and SboFlags.enable_combine_shared_two_stream_overlap()
        )

        if is_non_idle_and_non_empty(forward_mode, hidden_states):
            if not sbo_enabled_flag:
                if self.alt_stream is not None:
                    self.alt_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(self.alt_stream):
                        shared_input = (
                            (moe_i_q, moe_i_s)
                            if (moe_i_q is not None and moe_i_s is not None)
                            else hidden_states
                        )
                        shared_output = self.shared_experts(shared_input)
                        shared_output.record_stream(self.alt_stream)
                        shared_event = self.alt_stream.record_event()
                else:
                    shared_input = (
                        (moe_i_q, moe_i_s)
                        if (moe_i_q is not None and moe_i_s is not None)
                        else hidden_states
                    )
                    shared_output = self.shared_experts(shared_input)
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states)
            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        if sbo_overlap_dispatch_flag:
            shared_output = None

            def _deepep_dispatch_hook(dispatcher: BaseDispatcher):
                nonlocal shared_output
                shared_input = (
                    (moe_i_q, moe_i_s)
                    if (moe_i_q is not None and moe_i_s is not None)
                    else hidden_states
                )
                shared_output = self.shared_experts(shared_input)
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
                shared_input = (
                    (moe_i_q, moe_i_s)
                    if (moe_i_q is not None and moe_i_s is not None)
                    else hidden_states
                )
                shared_output = self.shared_experts(shared_input)

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
            i_q=moe_i_q,
            i_s=moe_i_s,
        )

        if (
            hidden_states.shape[0] > 0
            and not sbo_enabled_flag
            and self.alt_stream is not None
        ):
            torch.cuda.current_stream().wait_event(shared_event)
        if shared_output is not None:
            x = shared_output
            if self.experts.should_fuse_routed_scaling_factor_in_topk:
                x.add_(final_hidden_states)
            else:
                x.add_(final_hidden_states, alpha=self.routed_scaling_factor)
            final_hidden_states = x
        else:
            if not self.experts.should_fuse_routed_scaling_factor_in_topk:
                final_hidden_states *= self.routed_scaling_factor

        return final_hidden_states


class BailingMoEAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_kv_heads = config.num_key_value_heads
        self.dp_size = get_parallel().attn_dp_size
        attn_tp_rank = get_parallel().attn_tp_rank
        attn_tp_size = get_parallel().attn_tp_size

        assert self.total_num_heads % attn_tp_size == 0
        if self.total_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_kv_heads == 0
        assert self.total_num_heads >= self.total_kv_heads

        self.num_heads = self.total_num_heads // attn_tp_size
        self.head_dim = config.head_dim or (self.hidden_size // self.total_num_heads)
        self.q_size = self.head_dim * self.num_heads

        self.num_kv_heads = max(1, self.total_kv_heads // attn_tp_size)
        self.kv_size = max(1, self.num_kv_heads * self.head_dim)

        self.scale = self.head_dim**-0.5

        self.use_qk_norm = getattr(config, "use_qk_norm", False)

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_kv_heads,
            bias=(config.use_bias or config.use_qkv_bias),
            quant_config=quant_config,
            prefix=add_prefix("query_key_value", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        if self.use_qk_norm:
            self.query_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.dense = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=config.use_bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("dense", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        if hasattr(config, "partial_rotary_factor"):
            self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        elif hasattr(config, "rotary_dim"):
            self.rotary_dim = config.rotary_dim
        else:
            self.rotary_dim = self.head_dim
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_parameters["rope_theta"],
            rope_scaling=config.rope_parameters,
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scale,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

        self.alt_stream = alt_stream
        self.page_size = 64
        self.layer_id = layer_id
        if get_server_args().kv_cache_dtype == "fp8_e4m3":
            self.kv_cache_dtype = torch.float8_e4m3fn
        elif get_server_args().kv_cache_dtype == "fp8_e5m2":
            self.kv_cache_dtype = torch.float8_e5m2
        elif get_server_args().kv_cache_dtype in ("bf16", "bfloat16"):
            self.kv_cache_dtype = torch.bfloat16

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if isinstance(hidden_states, tuple):
            if hidden_states[0].shape[0] == 0:
                return hidden_states
        else:
            if hidden_states.shape[0] == 0:
                return hidden_states
        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        can_fuse_set_kv = (
            self.head_dim == self.rotary_emb.rotary_dim
            and enable_fused_set_kv_buffer(forward_batch)
        )
        if self.use_qk_norm and not _use_fused_bailing_rms_rotary:
            q, k = apply_qk_norm(
                q=q,
                k=k,
                q_norm=self.query_layernorm,
                k_norm=self.key_layernorm,
                head_dim=self.head_dim,
                alt_stream=self.alt_stream,
            )
        if self.use_qk_norm and _is_hcu and _use_fused_bailing_rms_rotary:
            # Fused RMSNorm + RoPE + kv_store path through custom op.
            cos_sin_cache = self.rotary_emb.cos_sin_cache
            if (cos_sin_cache.device != q.device
                    or cos_sin_cache.dtype != q.dtype):
                cos_sin_cache = cos_sin_cache.to(q.device,
                                                 dtype=q.dtype,
                                                 non_blocking=True)
                # Persist the converted cache so we don't re-copy/re-allocate
                # on every forward when the original buffer starts on CPU.
                self.rotary_emb.cos_sin_cache = cos_sin_cache
            k_buffer, v_buffer = get_token_to_kv_pool().get_kv_buffer(self.layer_id)

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
                weight_q=self.query_layernorm.weight,
                weight_k=self.key_layernorm.weight,
                output_dtype=self.kv_cache_dtype,
                residual_q=None,
                residual_k=None,
                k_scale=None,
                v_scale=None,
                epsilon=self.query_layernorm.variance_epsilon,
            )
        else:
            q, k = self.rotary_emb(
                positions,
                q,
                k,
                fused_set_kv_buffer_arg=(
                    create_fused_set_kv_buffer_arg(
                        value=v,
                        layer=self.attn,
                        forward_batch=forward_batch,
                    )
                    if can_fuse_set_kv
                    else None
                ),
            )
        context_layer = self.attn(
            q,
            k,
            v,
            forward_batch,
            save_kv_cache=not can_fuse_set_kv,
        )
        attn_output, _ = self.dense(context_layer)
        return attn_output


class BailingMoEBlock(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size

        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.dp_size = get_parallel().attn_dp_size
        self.attention = BailingMoEAttention(
            config,
            layer_id,
            quant_config,
            reduce_results=False,
            prefix=add_prefix("attention", prefix),
            alt_stream=alt_stream,
        )
        self.layer_id = layer_id
        self.attn_tp_size = get_parallel().attn_tp_size
        self.attn_tp_rank = get_parallel().attn_tp_rank

        self.is_layer_sparse = self._is_layer_sparse(
            config, layer_id=layer_id, is_nextn=False
        )
        is_previous_layer_sparse = self._is_layer_sparse(
            config, layer_id=layer_id - 1, is_nextn=False
        )
        is_next_layer_sparse = self._is_layer_sparse(
            config, layer_id=layer_id + 1, is_nextn=False
        )

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        self.is_last_layer = self.layer_id == config.num_hidden_layers - 1
        self.use_deepep = get_moe_a2a_backend().is_deepep()
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        if self.is_layer_sparse:
            self.mlp = BailingMoESparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                alt_stream=alt_stream,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = BailingMoEMLP(
                intermediate_size=config.intermediate_size,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(self.layer_id == self.config.num_hidden_layers - 1),
        )

    def _is_layer_sparse(
        self, config: PretrainedConfig, layer_id: int, is_nextn: bool
    ) -> bool:
        return is_nextn or (
            config.num_experts is not None and layer_id >= config.first_k_dense_replace
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        captured_last_layer_outputs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        hidden_states, residual = (
            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                hidden_states,
                residual,
                forward_batch,
                captured_last_layer_outputs=captured_last_layer_outputs,
            )
        )
        if isinstance(hidden_states, tuple):
            if hidden_states[0].shape[0] != 0:
                hidden_states = self.attention(
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                )
        else:
            if hidden_states.shape[0] != 0:
                hidden_states = self.attention(
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                )

        # Keep this fusion path minimal: only first 4 dense layers.
        dense_rms_quant_fusion = (
            _is_hcu
            and _use_fused_bailing_rms_quant
            and (not self.is_layer_sparse)
            and (not is_dp_attention_enabled() or get_server_args().ep_size > 1)
        )
        sparse_rms_quant_fusion = (
            _is_hcu
            and _use_fused_bailing_rms_quant
            and self.is_layer_sparse
            and (not is_dp_attention_enabled() or get_server_args().ep_size > 1)
        )
        rms_quant_fusion = dense_rms_quant_fusion or sparse_rms_quant_fusion
        prev_rms_quant_flag = forward_batch.rms_quant_flag
        prev_sparse_rms_quant_fusion = getattr(
            forward_batch, "bailing_sparse_rms_quant_fusion", False
        )
        prev_sparse_norm_hidden_states = getattr(
            forward_batch, "bailing_sparse_norm_hidden_states", None
        )
        forward_batch.rms_quant_flag = rms_quant_fusion
        forward_batch.bailing_sparse_rms_quant_fusion = sparse_rms_quant_fusion
        forward_batch.bailing_sparse_norm_hidden_states = None
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
        )
        sparse_moe_i_q = None
        sparse_moe_i_s = None
        if sparse_rms_quant_fusion:
            sparse_norm_hidden_states = getattr(
                forward_batch, "bailing_sparse_norm_hidden_states", None
            )
            if sparse_norm_hidden_states is None or not isinstance(hidden_states, tuple):
                raise RuntimeError(
                    "sparse rms+quant fusion expects normalized hidden_states and quant outputs"
                )
            sparse_moe_i_q, sparse_moe_i_s = hidden_states
            hidden_states = sparse_norm_hidden_states
        forward_batch.rms_quant_flag = prev_rms_quant_flag
        forward_batch.bailing_sparse_rms_quant_fusion = prev_sparse_rms_quant_fusion
        forward_batch.bailing_sparse_norm_hidden_states = prev_sparse_norm_hidden_states

        fuse_mlp_allreduce = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )

        # For DP with padding, reduce scatter can be used instead of all-reduce.
        mlp_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        with get_forward().scoped(
            fuse_mlp_allreduce=fuse_mlp_allreduce,
            mlp_reduce_scatter=mlp_reduce_scatter,
        ):
            if self.is_layer_sparse:
                hidden_states = self.mlp(
                    hidden_states,
                    forward_batch,
                    moe_i_q=sparse_moe_i_q,
                    moe_i_s=sparse_moe_i_s,
                )
            else:
                hidden_states = self.mlp(hidden_states, forward_batch)

        if fuse_mlp_allreduce:
            hidden_states._sglang_needs_allreduce_fusion = True
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        return hidden_states, residual


class BailingMoEModel(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.hidden_size
        if self.pp_group.is_first_rank:
            self.word_embeddings = VocabParallelEmbedding(
                self.vocab_size,
                self.embed_dim,
                quant_config=quant_config,
                prefix=add_prefix("word_embeddings", prefix),
                use_attn_tp_group=is_dp_attention_enabled(),
            )
        else:
            self.word_embeddings = PPMissingLayer()

        self.embedding_dropout = torch.nn.Dropout(config.embedding_dropout)

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: BailingMoEBlock(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(self.embed_dim, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        self.layers_to_capture = []

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.word_embeddings(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        aux_hidden_states = []
        for i in range(self.start_layer, self.end_layer):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                if i in self.layers_to_capture:
                    aux_hidden_states.append(
                        hidden_states if residual is None else hidden_states + residual
                    )
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                    captured_last_layer_outputs=(
                        aux_hidden_states
                        if getattr(layer, "_is_layer_to_capture", False)
                        else None
                    ),
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

        if len(aux_hidden_states) == 0:
            return hidden_states
        return hidden_states, aux_hidden_states


class BailingMoEForCausalLM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.pp_group = get_pp_group()

        self.config = config
        self.quant_config = quant_config
        alt_stream = get_stream("alt") if _is_cuda or is_sbo_enabled() else None

        self.model = BailingMoEModel(
            config,
            quant_config,
            alt_stream=alt_stream,
            prefix=add_prefix("model", ""),
        )

        # tie_word_embeddings为true，复用tie_word_embeddings，反之是独立的
        if config.tie_word_embeddings:
            self.lm_head = self.model.word_embeddings
        else:
            # TODO something wrong with ParallelLMHead with DP attention enabled
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_server_args().enable_dp_lm_head,
            )
        self.logits_processor = LogitsProcessor(config)
        self.num_fused_shared_experts = (
            0
            if get_server_args().disable_shared_experts_fusion or get_moe_a2a_backend().is_deepep()
            else config.num_shared_experts
        )

        self.capture_aux_hidden_states = False

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def get_embed_and_head(self):
        """Used by the eagle_worker."""
        return self.model.word_embeddings.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        """Used by the eagle_worker."""
        del self.model.word_embeddings.weight
        del self.lm_head.weight
        self.model.word_embeddings.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
                # compatible with old design
                nextn_layer_id = (
                    0
                    if self.config.num_hidden_layers == 1
                    else self.config.num_hidden_layers
                )
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        if is_nextn:
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
            nextn_spec_weight_names = [
                "final_layernorm",
                "eh_proj",
                "enorm",
                "hnorm",
            ]
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts + self.num_fused_shared_experts,
        )

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if (
                ("v_head" in name)
                or ("inv_freq" in name)
                or (self.config.tie_word_embeddings and "lm_head" in name)
            ):
                continue

            if (
                hasattr(self.config, "norm_head")
                and self.config.norm_head
                and "lm_head.weight" in name
            ):
                import torch.nn.functional as F

                loaded_weight = F.normalize(loaded_weight, dim=0, p=2, eps=1e-7)
            
            if self.num_fused_shared_experts > 0 and "mlp.shared_experts" in name:
                name = name.replace(
                    "mlp.shared_experts",
                    f"mlp.experts.{self.config.num_experts}",
                )

            if is_nextn:
                if not name.startswith(nextn_layer_prefix):
                    continue

                # Use shared head and embed weights from target model
                if "shared_head.head" in name or "embed_tokens" in name:
                    continue

                is_decoder = True
                # For nextn specific weights
                for weight_name in nextn_spec_weight_names:
                    if weight_name in name:
                        name = name.replace(nextn_layer_prefix, "model")
                        is_decoder = False
                        break
                # For decoder layer weights
                if is_decoder:
                    name = name.replace(nextn_layer_prefix, "model.decoder")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
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
                    if name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

        if not is_nextn:
            self.routed_experts_weights_of_layer = {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if not isinstance(layer, PPMissingLayer)
                and isinstance(layer.mlp, BailingMoESparseMoeBlock)
            }

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        num_groups = getattr(config, "n_group", 0)
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None if num_groups == 0 else num_groups,
        )

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]
        else:
            # Add +1 because in SGLang, for the i-th layer, the auxiliary hidden state
            # corresponds to the output of layer (i - 1).
            self.model.layers_to_capture = [val + 1 for val in layer_ids]


class BailingMoeForCausalLM(BailingMoEForCausalLM):
    pass


class BailingMoeV2ForCausalLM(BailingMoEForCausalLM):
    pass


EntryClass = [BailingMoEForCausalLM, BailingMoeForCausalLM, BailingMoeV2ForCausalLM]
