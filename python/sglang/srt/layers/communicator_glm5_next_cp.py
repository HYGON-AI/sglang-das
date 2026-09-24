from functools import partial
from typing import Callable, Optional

import torch

from sglang.srt.layers.attention.dsa.utils import (
    dsa_prefill_has_history,
    dsa_use_prefill_cp,
    is_dsa_enable_prefill_cp,
)
from sglang.srt.layers.communicator import (
    CommunicateContext,
    CommunicateSimpleFn,
    CommunicateSummableTensorPairFn,
    CommunicateWithAllReduceAndLayerNormFn,
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)
from sglang.srt.layers.dp_attention import (
    attn_cp_all_gather_into_tensor,
    attn_cp_reduce_scatter_tensor,
    get_local_dp_buffer,
)
from sglang.srt.mem_cache.layer_split import build_main_kv_page_plan
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    get_req_to_token_pool,
    get_token_to_kv_pool,
)
from sglang.srt.runtime_context import get_parallel

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


def nsa_enable_prefill_cp():
    # After using cp, the communication mode of this part changes.
    # The three parts of prepare_attn, prepare_mlp, and postprocess_layer
    # no longer require additional communication for reduce, scatter, etc.
    return is_dsa_enable_prefill_cp()


def maybe_prefetch_full_attention_kv(
    forward_batch: ForwardBatch,
    full_attention_layer_id: Optional[int],
) -> None:
    if full_attention_layer_id is None or not dsa_use_prefill_cp(forward_batch):
        return

    prefetch_mla_kv = getattr(get_token_to_kv_pool(), "prefetch_mla_kv_buffer", None)
    if prefetch_mla_kv is not None:
        configure_page_plan = getattr(
            get_token_to_kv_pool(),
            "configure_main_kv_page_plan",
            None,
        )
        if configure_page_plan is not None:
            page_plan = None
            can_compact_main_kv = (
                getattr(
                    get_token_to_kv_pool(),
                    "layer_shard_enabled",
                    False,
                )
                and forward_batch.forward_mode.is_extend_without_speculative()
                and getattr(forward_batch, "hisparse_coordinator", None) is None
                and forward_batch.extend_prefix_lens_cpu is not None
                and forward_batch.out_cache_loc is not None
            )
            if can_compact_main_kv:
                if forward_batch.dsa_layer_split_main_kv_page_plan is None:
                    req_to_token = get_req_to_token_pool().req_to_token
                    forward_batch.dsa_layer_split_main_kv_page_plan = (
                        build_main_kv_page_plan(
                            req_to_token=req_to_token,
                            req_pool_indices=forward_batch.req_pool_indices,
                            prefix_lens=forward_batch.extend_prefix_lens_cpu,
                            current_locs=forward_batch.out_cache_loc,
                            page_size=get_token_to_kv_pool().page_size,
                        )
                    )
                page_plan = forward_batch.dsa_layer_split_main_kv_page_plan
            # An explicit None disables a compact layout left by an earlier
            # ForwardBatch before the legacy full-pool path is entered.
            configure_page_plan(page_plan, forward_batch)
        prefetch_mla_kv(
            full_attention_layer_id,
            has_history=dsa_prefill_has_history(forward_batch),
        )


def maybe_prefetch_next_full_attention_kv(
    forward_batch: ForwardBatch,
    next_full_attention_layer_id: Optional[int],
) -> None:
    maybe_prefetch_full_attention_kv(forward_batch, next_full_attention_layer_id)


class Glm5NextCPLayerCommunicator(LayerCommunicator):
    def __init__(
        self,
        layer_scatter_modes: LayerScatterModes,
        input_layernorm: torch.nn.Module,
        post_attention_layernorm: torch.nn.Module,
        # Reduce scatter requires skipping all-reduce in model code after MoE/MLP, so only enable for models which have that implemented. Remove flag once done for all models that use LayerCommunicator.
        allow_reduce_scatter: bool = False,
        is_last_layer: bool = False,
        qkv_latent_func: Optional[Callable] = None,
    ):
        super().__init__(
            layer_scatter_modes=layer_scatter_modes,
            input_layernorm=input_layernorm,
            post_attention_layernorm=post_attention_layernorm,
            allow_reduce_scatter=allow_reduce_scatter,
            is_last_layer=is_last_layer,
            qkv_latent_func=qkv_latent_func,
        )

    def _post_init_communicate(self):
        # SCATTERED in attn tp is different from SCATTERED in global tp when dp_size > 1
        if self.layer_scatter_modes.mlp_mode != ScatterMode.SCATTERED:
            assert self._context.attn_dp_size == 1, (
                f"dp_size should be 1 when moe_runner_backend is none"
            )
        self._communicate_simple_fn = NSACPCommunicateSimpleFn.get_fn(
            input_mode=ScatterMode.SCATTERED,
            output_mode=ScatterMode.SCATTERED,
            context=self._context,
        )
        self._communicate_with_all_reduce_and_layer_norm_fn = NSACPCommunicateWithAllReduceAndLayerNormFn.get_fn(
            hidden_states_input_mode=ScatterMode.SCATTERED,
            residual_input_mode=ScatterMode.SCATTERED,
            hidden_states_output_mode=self.layer_scatter_modes.mlp_mode,  # SCATTERED, FULL
            residual_output_mode=ScatterMode.SCATTERED,
            context=self._context,
        )
        self._communicate_summable_tensor_pair_fn = NSACPCommunicateSummableTensorPairFn.get_fn(
            hidden_states_input_mode=self.layer_scatter_modes.mlp_mode,  # SCATTERED, FULL
            residual_input_mode=ScatterMode.SCATTERED,
            output_mode=ScatterMode.SCATTERED,
            context=self._context,
        )

    def maybe_prefetch_next_full_attention_kv(
        self,
        forward_batch: ForwardBatch,
        next_full_attention_layer_id: Optional[int],
    ) -> None:
        maybe_prefetch_next_full_attention_kv(
            forward_batch, next_full_attention_layer_id
        )


class NSACPCommunicateSimpleFn(CommunicateSimpleFn):
    @staticmethod
    def get_fn(
        input_mode: ScatterMode,
        output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        if context.is_same_group_size(input_mode, output_mode):
            return NSACPCommunicateSimpleFn._trivial

        raise NotImplementedError(f"{input_mode=} {output_mode=}")


class NSACPCommunicateWithAllReduceAndLayerNormFn(
    CommunicateWithAllReduceAndLayerNormFn
):
    """Besides communication, needs to
    1. All reduce in tp_attn_group on hidden_states
    2. Apply layer norm
    """

    @staticmethod
    def get_fn(
        hidden_states_input_mode: ScatterMode,
        residual_input_mode: ScatterMode,
        hidden_states_output_mode: ScatterMode,
        residual_output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        assert hidden_states_input_mode == ScatterMode.SCATTERED
        assert residual_input_mode == ScatterMode.SCATTERED
        assert residual_output_mode == ScatterMode.SCATTERED
        if hidden_states_output_mode == ScatterMode.SCATTERED:
            return NSACPCommunicateWithAllReduceAndLayerNormFn._simple

        if hidden_states_output_mode == ScatterMode.FULL:
            return partial(
                NSACPCommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual,
                residual_input_mode=residual_input_mode,
            )

        raise NotImplementedError(
            f"{hidden_states_input_mode=} {residual_input_mode=} {hidden_states_output_mode=} {residual_output_mode=}"
        )

    @staticmethod
    def _gather_hidden_states_and_residual(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
        *,
        residual_input_mode,
    ):
        if hidden_states.shape[0] != 0:
            hidden_states, residual = layernorm(hidden_states, residual)
        # for prefill: attn tp scattered -> full
        # for decode: attn tp full -> full
        if dsa_use_prefill_cp(forward_batch):
            assert context.attn_dp_size == 1
            hidden_states, local_hidden_states = (
                get_local_dp_buffer(get_parallel().attn_cp_group),
                hidden_states,
            )
            attn_cp_all_gather_into_tensor(
                hidden_states,
                local_hidden_states,
            )
        return hidden_states, residual


class NSACPCommunicateSummableTensorPairFn(CommunicateSummableTensorPairFn):
    """It is allowed to make (hidden_states, residual) := (hidden_states + residual, None) if needed."""

    @staticmethod
    def get_fn(
        hidden_states_input_mode: ScatterMode,
        residual_input_mode: ScatterMode,
        output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        # Check exact enum match first: even if group sizes happen to be equal
        # (e.g. tp_size == attn_cp_size makes FULL and SCATTERED both size 1),
        # FULL and SCATTERED have different data layouts under CP and require
        # an explicit scatter operation.
        if (
            (hidden_states_input_mode == ScatterMode.FULL)
            and (residual_input_mode == ScatterMode.SCATTERED)
            and (output_mode == ScatterMode.SCATTERED)
        ):
            return NSACPCommunicateSummableTensorPairFn._scatter_hidden_states

        if context.is_same_group_size(
            hidden_states_input_mode, output_mode
        ) and context.is_same_group_size(residual_input_mode, output_mode):
            return NSACPCommunicateSummableTensorPairFn._trivial

        raise NotImplementedError(
            f"{hidden_states_input_mode=} {residual_input_mode=} {output_mode=}"
        )

    @staticmethod
    def _scatter_hidden_states(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        allow_reduce_scatter: bool = False,
    ):
        # for prefill: full -> attn tp scattered
        # for decode: full -> attn tp full
        if dsa_use_prefill_cp(forward_batch):
            assert context.attn_dp_size == 1
            input_hidden_states = hidden_states
            hidden_states = hidden_states.tensor_split(context.attn_cp_size)[
                context.attn_cp_rank
            ]
            attn_cp_reduce_scatter_tensor(hidden_states, input_hidden_states)
        return hidden_states, residual
