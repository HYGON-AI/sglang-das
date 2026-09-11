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


from functools import partial
from typing import Callable, Optional

import torch

from sglang.srt.layers.attention.dsa.utils import (
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
from sglang.srt.layers.utils.cp_utils import mla_use_prefill_cp
from sglang.srt.mem_cache.dsa_cache_layer_split import build_main_kv_page_plan
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    get_req_to_token_pool,
    get_token_to_kv_pool,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_hcu


def dsa_enable_prefill_cp():
    # After using cp, the communication mode of this part changes.
    # The three parts of prepare_attn, prepare_mlp, and postprocess_layer
    # no longer require additional communication for reduce, scatter, etc.
    return is_dsa_enable_prefill_cp()


def _dsa_prefill_has_history(forward_batch: ForwardBatch) -> bool:
    prefix_lens = forward_batch.extend_prefix_lens_cpu
    if prefix_lens is None:
        return True
    return any(int(prefix_len) > 0 for prefix_len in prefix_lens)


def maybe_configure_main_kv_page_plan(forward_batch: ForwardBatch) -> None:
    """Install this batch's compact Main-KV mapping when LayerSplit is active."""

    token_to_kv_pool = get_token_to_kv_pool()
    use_prefill_cp = dsa_use_prefill_cp(forward_batch)
    configure_page_plan = getattr(token_to_kv_pool, "configure_main_kv_page_plan", None)
    if configure_page_plan is not None:
        page_plan = None
        can_compact_main_kv = (
            is_hcu()
            and getattr(token_to_kv_pool, "layer_shard_enabled", False)
            and use_prefill_cp
            and forward_batch.forward_mode.is_extend_without_speculative()
            # Compact mapping is batch-specific; retain the official full-
            # scratch path while TBO alternates between two child batches.
            and not forward_batch.can_run_tbo
            and forward_batch.tbo_parent_token_range is None
            and forward_batch.extend_prefix_lens_cpu is not None
            and forward_batch.out_cache_loc is not None
        )
        if can_compact_main_kv:
            if forward_batch.dsa_layer_split_main_kv_page_plan is None:
                forward_batch.dsa_layer_split_main_kv_page_plan = (
                    build_main_kv_page_plan(
                        req_to_token=get_req_to_token_pool().req_to_token,
                        req_pool_indices=forward_batch.req_pool_indices,
                        prefix_lens=forward_batch.extend_prefix_lens_cpu,
                        current_locs=forward_batch.out_cache_loc,
                        page_size=token_to_kv_pool.page_size,
                    )
                )
            page_plan = forward_batch.dsa_layer_split_main_kv_page_plan
        # Explicitly clear a compact layout left by an earlier ForwardBatch
        # before entering the legacy full-pool path.
        configure_page_plan(page_plan, forward_batch)


def maybe_prefetch_full_attention_kv(
    forward_batch: ForwardBatch,
    full_attention_layer_id: Optional[int],
) -> None:
    """Configure the batch plan and prefetch one DSA layer's caches."""

    maybe_configure_main_kv_page_plan(forward_batch)

    if full_attention_layer_id is None or not dsa_use_prefill_cp(forward_batch):
        return

    token_to_kv_pool = get_token_to_kv_pool()
    has_history = _dsa_prefill_has_history(forward_batch)
    # Index-K and Main-KV share one communicator. Every rank must enqueue
    # collectives in this order; skip-topk layers make the Index-K call a no-op.
    prefetch_index_buffer = getattr(token_to_kv_pool, "prefetch_index_buffer", None)
    if is_hcu() and prefetch_index_buffer is not None:
        prefetch_index_buffer(
            full_attention_layer_id,
            has_history=has_history,
        )
    prefetch_kv_buffer = getattr(token_to_kv_pool, "prefetch_kv_buffer", None)
    if prefetch_kv_buffer is not None:
        prefetch_kv_buffer(
            full_attention_layer_id,
            has_history=has_history,
        )


def maybe_prefetch_next_full_attention_kv(
    forward_batch: ForwardBatch,
    next_full_attention_layer_id: Optional[int],
) -> None:
    """Prefetch the next layer while the current layer's MLP runs."""

    maybe_prefetch_full_attention_kv(forward_batch, next_full_attention_layer_id)


def dsa_cp_gather_hidden_states(hidden_states: torch.Tensor):
    attn_dp_size = get_parallel().attn_dp_size
    attn_tp_size = get_parallel().attn_tp_size
    assert attn_dp_size == 1 and attn_tp_size == 1
    hidden_states, local_hidden_states = (
        get_local_dp_buffer(get_parallel().attn_cp_group),
        hidden_states,
    )
    attn_cp_all_gather_into_tensor(hidden_states, local_hidden_states)
    return hidden_states


def dsa_cp_reduce_scatter_hidden_states(hidden_states: torch.Tensor):
    attn_dp_size = get_parallel().attn_dp_size
    attn_tp_size = get_parallel().attn_tp_size
    assert attn_dp_size == 1 and attn_tp_size == 1
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    input_hidden_states = hidden_states
    hidden_states = hidden_states.tensor_split(cp_size)[cp_rank]
    attn_cp_reduce_scatter_tensor(hidden_states, input_hidden_states)
    return hidden_states


class DSACPLayerCommunicator(LayerCommunicator):
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
            layer_scatter_modes,
            input_layernorm,
            post_attention_layernorm,
            allow_reduce_scatter,
            is_last_layer,
            qkv_latent_func,
        )

    def _post_init_communicate(self):
        # SCATTERED in attn tp is different from SCATTERED in global tp when dp_size > 1
        if self.layer_scatter_modes.mlp_mode != ScatterMode.SCATTERED:
            assert (
                self._context.attn_dp_size == 1
            ), f"dp_size should be 1 when moe_runner_backend is none"
        self._communicate_simple_fn = DSACPCommunicateSimpleFn.get_fn(
            input_mode=ScatterMode.SCATTERED,
            output_mode=ScatterMode.SCATTERED,
            context=self._context,
        )
        self._communicate_with_all_reduce_and_layer_norm_fn = DSACPCommunicateWithAllReduceAndLayerNormFn.get_fn(
            hidden_states_input_mode=ScatterMode.SCATTERED,
            residual_input_mode=ScatterMode.SCATTERED,
            hidden_states_output_mode=self.layer_scatter_modes.mlp_mode,  # SCATTERED, FULL
            residual_output_mode=ScatterMode.SCATTERED,
            context=self._context,
        )
        self._communicate_summable_tensor_pair_fn = DSACPCommunicateSummableTensorPairFn.get_fn(
            hidden_states_input_mode=self.layer_scatter_modes.mlp_mode,  # SCATTERED, FULL
            residual_input_mode=ScatterMode.SCATTERED,
            output_mode=ScatterMode.SCATTERED,
            context=self._context,
        )


class DSACPCommunicateSimpleFn(CommunicateSimpleFn):
    @staticmethod
    def get_fn(
        input_mode: ScatterMode,
        output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        if context.is_same_group_size(input_mode, output_mode):
            return DSACPCommunicateSimpleFn._trivial

        raise NotImplementedError(f"{input_mode=} {output_mode=}")


class DSACPCommunicateWithAllReduceAndLayerNormFn(
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
            return DSACPCommunicateWithAllReduceAndLayerNormFn._simple

        if hidden_states_output_mode == ScatterMode.FULL:
            return partial(
                DSACPCommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual,
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
        if dsa_use_prefill_cp(forward_batch) or mla_use_prefill_cp(forward_batch):
            hidden_states = dsa_cp_gather_hidden_states(hidden_states)
        return hidden_states, residual


class DSACPCommunicateSummableTensorPairFn(CommunicateSummableTensorPairFn):
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
            return DSACPCommunicateSummableTensorPairFn._scatter_hidden_states

        if context.is_same_group_size(
            hidden_states_input_mode, output_mode
        ) and context.is_same_group_size(residual_input_mode, output_mode):
            return DSACPCommunicateSummableTensorPairFn._trivial

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
        if dsa_use_prefill_cp(forward_batch) or mla_use_prefill_cp(forward_batch):
            hidden_states = dsa_cp_reduce_scatter_hidden_states(hidden_states)
        return hidden_states, residual
