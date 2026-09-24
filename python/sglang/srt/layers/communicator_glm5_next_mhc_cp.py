from typing import Optional

from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from sglang.srt.layers.communicator_glm5_next_cp import (
    maybe_prefetch_next_full_attention_kv,
)
from sglang.srt.layers.communicator_glm5_next_mhc import MHCLayerCommunicator
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.layers.utils.glm5_next_cp import (
    cp_plain_all_gather,
    cp_plain_reduce_scatter,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class Glm5NextMHCCPLayerCommunicator(MHCLayerCommunicator):
    """Communication for ModelNext's KDA/NSA hybrid CP layout.

    Cross-layer mHC state is block-contiguous ("plain"). KDA consumes that
    layout directly and performs its own CP gather/reduce-scatter. NSA consumes
    the round-robin CP layout, so only NSA inputs/outputs are converted here.
    """

    def _use_cp_plain_moe_tp_fallback(
        self,
        forward_batch: ForwardBatch,
        is_sparse_layer: bool,
    ) -> bool:
        return (
            is_sparse_layer
            and dsa_use_prefill_cp(forward_batch)
            and get_moe_a2a_backend().is_none()
        )

    def prepare_mlp_input(
        self,
        hidden_states,
        forward_batch: ForwardBatch,
        is_sparse_layer: bool,
    ):
        self._mlp_comm_kind = None
        self._a2a_scatter_chunks = None
        if self._use_cp_plain_moe_tp_fallback(forward_batch, is_sparse_layer):
            self._mlp_comm_kind = "cp_plain_gather"
            return cp_plain_all_gather(hidden_states, forward_batch=forward_batch)
        return hidden_states

    def restore_mlp_output(
        self,
        hidden_states,
        forward_batch: ForwardBatch,
    ):
        del forward_batch
        if self._mlp_comm_kind == "cp_plain_gather":
            hidden_states = cp_plain_reduce_scatter(hidden_states)

        self._mlp_comm_kind = None
        self._a2a_scatter_chunks = None
        return hidden_states

    def should_use_reduce_scatter(self, forward_batch: ForwardBatch):
        return self._use_cp_plain_moe_tp_fallback(
            forward_batch,
            self.is_layer_sparse,
        )

    def maybe_prefetch_next_full_attention_kv(
        self,
        forward_batch: ForwardBatch,
        next_full_attention_layer_id: Optional[int],
    ) -> None:
        maybe_prefetch_next_full_attention_kv(
            forward_batch, next_full_attention_layer_id
        )
