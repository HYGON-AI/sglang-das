# Copyright 2023-2026 SGLang Team
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

"""Public import facade and runtime helpers for context parallel strategies."""

from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, Optional, Tuple

from sglang.srt.layers.cp.base import (
    BaseContextParallelMetadata,
    ContextParallelStrategy,
    ContextParallelStrategyKind,
    CPAttentionBackendKind,
    get_cp_strategy,
)
from sglang.srt.layers.cp.interleave import (
    InterleaveContextParallelMetadata,
    InterleaveCPStrategy,
)
from sglang.srt.layers.cp.padding import pad_logical_token_to_physical
from sglang.srt.layers.cp.zigzag import (
    ContextParallelMetadata,
    ZigzagContextParallelMetadata,
    ZigzagCPStrategy,
    compute_zigzag_cp_physical_token_count,
)
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

CP_V2_DEFAULT_MODEL_CLASSES = frozenset(
    {
        "DeepseekV32ForCausalLM",
        "GlmMoeDsaForCausalLM",
        "GptOssForCausalLM",
        "MiMoV2FlashForCausalLM",
        "MiMoV2ForCausalLM",
        "Qwen3MoeForCausalLM",
        "DeepseekV3ForCausalLM",
    }
)


def is_glm_dsa_cache_layer_split_enabled(model_runner: "ModelRunner") -> bool:
    """Whether DSA GPU KV/indexer cache layers are sharded across CP ranks.

    Layer split is a prefill-CP-only optimization for DSA (DeepSeek Sparse
    Attention) MLA models (e.g. GLM-5.2). Single-layer NextN drafts can opt in
    after the EAGLE worker supplies a compatible target Main-KV scratch pool.
    """
    from sglang.srt.configs.model_config import is_deepseek_dsa

    return (
        (
            not model_runner.is_draft_worker
            or getattr(model_runner, "dsa_layer_split_scratch_source", None) is not None
        )
        and model_runner.server_args.enable_dsa_cache_layer_split
        and model_runner.use_mla_backend
        and is_deepseek_dsa(model_runner.model_config.hf_config)
    )


def get_glm_dsa_cp_layer_shard_info(
    model_runner: "ModelRunner",
) -> Tuple[Optional[int], int]:
    """Return ``(layer_shard_rank, layer_shard_size)`` for the DSA KV pool.

    ``(None, 1)`` disables sharding (feature off or only one CP rank).
    """
    if not is_glm_dsa_cache_layer_split_enabled(model_runner):
        return None, 1
    shard_size = get_parallel().attn_cp_size
    if shard_size <= 1:
        return None, 1
    return get_parallel().attn_cp_rank, shard_size


def get_glm_dsa_layer_split_effective_num_layers(
    model_runner: "ModelRunner", num_layers: int
) -> int:
    """Per-rank owned layer count used when sizing the DSA KV cell.

    Under layer split each CP rank only stores ``ceil(num_layers / shard_size)``
    layers, plus one extra layer for the remote scratch buffer used when reading
    a layer owned by another CP rank.
    """
    if not is_glm_dsa_cache_layer_split_enabled(model_runner):
        return num_layers
    shard_size = get_parallel().attn_cp_size
    if shard_size <= 1:
        return num_layers
    owned_layers_upper_bound = (num_layers + shard_size - 1) // shard_size
    if getattr(model_runner, "dsa_layer_split_scratch_source", None) is not None:
        # The draft aliases target Main-KV scratch, so count only its owner
        # storage here. Index-K has its own independent scratch budget.
        return owned_layers_upper_bound
    return max(1, owned_layers_upper_bound + 1)


def get_layer_shard_range(
    rank: int, shard_size: int, total_layers: int
) -> Tuple[int, int]:
    """Contiguous ``[start, end)`` local-layer range owned by ``rank``.

    Layers are split as evenly as possible; the first ``total_layers %
    shard_size`` ranks own one extra layer.
    """
    base = total_layers // shard_size
    rem = total_layers % shard_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


def get_layer_owner(local_layer_idx: int, shard_size: int, total_layers: int) -> int:
    """CP rank that owns ``local_layer_idx`` under the contiguous split."""
    for rank in range(shard_size):
        start, end = get_layer_shard_range(rank, shard_size, total_layers)
        if start <= local_layer_idx < end:
            return rank
    raise ValueError(
        f"Invalid local_layer_idx={local_layer_idx} for "
        f"shard_size={shard_size}, total_layers={total_layers}"
    )


def enable_cp_v2() -> bool:
    """Return whether the CP-v2 path is enabled for this process."""
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_ENABLE_CP_V2.get())


def is_cp_v2_active(forward_batch) -> bool:
    """Return whether the current forward batch is running through CP-v2."""
    if not enable_cp_v2():
        return False
    forward_mode = getattr(forward_batch, "forward_mode", None)
    if forward_mode is None or not forward_mode.is_context_parallel_extend():
        return False

    strategy = get_cp_strategy()
    if strategy is None:
        return False

    input_ids = getattr(forward_batch, "input_ids", None)
    if input_ids is None:
        return False

    # DP attention synchronizes the CP layout before ForwardBatch reaches the
    # model.  An all-zero list means that another non-idle DP replica cannot
    # enter CP for this global forward (for example, a 1-token batch paired
    # with a 16K Prefill).  Every replica must then use the ordinary full-TP
    # layout; allowing only the locally eligible replica to shard would make
    # the model-body collectives disagree.
    global_cp_tokens = getattr(forward_batch, "global_cp_num_tokens_cpu", None)
    if global_cp_tokens is not None and not any(global_cp_tokens):
        return False

    return strategy.can_apply(len(input_ids), forward_batch)


def normalize_dp_cp_token_counts(
    global_num_tokens: list[int], global_cp_num_tokens: list[int]
) -> list[int]:
    """Select one collective layout for a synchronized attention-DP forward.

    CP may coexist with an idle DP replica, which participates with zero local
    rows.  It may not coexist with a non-idle replica that is ineligible for CP:
    that replica needs the full-TP layout.  In the latter case disable CP for
    the whole global forward so every rank executes matching collectives.
    """
    if len(global_num_tokens) != len(global_cp_num_tokens):
        raise ValueError(
            "DP/CP token-count width mismatch: "
            f"num_tokens={global_num_tokens}, cp_tokens={global_cp_num_tokens}"
        )
    has_cp_batch = any(int(tokens) > 0 for tokens in global_cp_num_tokens)
    has_non_cp_work = any(
        int(tokens) > 0 and int(cp_tokens) == 0
        for tokens, cp_tokens in zip(global_num_tokens, global_cp_num_tokens)
    )
    if has_cp_batch and has_non_cp_work:
        return [0] * len(global_cp_num_tokens)
    return [int(tokens) for tokens in global_cp_num_tokens]


def get_cp_v2_physical_token_count(
    *, num_tokens: int, extend_seq_lens, cp_size: int
) -> int:
    """Compute this DP replica's CP-local padded rows in the scheduler.

    Zero means that CP-v2 is inactive for the local batch.  This function is
    deliberately CPU-only because it runs before the scheduler's existing DP
    metadata all-gather.
    """
    if not enable_cp_v2() or cp_size <= 1 or extend_seq_lens is None:
        return 0
    strategy = get_cp_strategy()
    if not isinstance(strategy, ZigzagCPStrategy):
        return 0

    from types import SimpleNamespace

    extend_seq_lens = [int(length) for length in extend_seq_lens]
    probe = SimpleNamespace(
        forward_mode=None,
        extend_seq_lens_cpu=extend_seq_lens,
    )
    if not strategy.can_apply(int(num_tokens), probe):
        return 0
    return compute_zigzag_cp_physical_token_count(extend_seq_lens, cp_size)


def prepare_cp_forward(forward_batch) -> None:
    """Build CP-v2 metadata for an active context-parallel prefill batch."""
    assert is_cp_v2_active(forward_batch)
    strategy = get_cp_strategy()
    assert strategy is not None

    seq_lens_cpu = _to_int_list(getattr(forward_batch, "seq_lens_cpu", None))
    extend_lens_cpu = _to_int_list(getattr(forward_batch, "extend_seq_lens_cpu", None))
    num_tokens = (
        sum(extend_lens_cpu)
        if extend_lens_cpu is not None
        else len(forward_batch.input_ids)
    )
    if forward_batch.attn_cp_metadata is None:
        forward_batch.attn_cp_metadata = strategy.build_metadata(
            num_tokens=num_tokens,
            seqs_len=seq_lens_cpu,
            extend_seqs_len=extend_lens_cpu,
        )
        pad_logical_token_to_physical(forward_batch.attn_cp_metadata)

    global_cp_tokens = getattr(forward_batch, "global_cp_num_tokens_cpu", None)
    if global_cp_tokens is not None:
        dp_rank = get_parallel().attn_dp_rank
        expected = int(global_cp_tokens[dp_rank])
        actual = int(
            forward_batch.attn_cp_metadata.per_rank_actual_token[
                get_parallel().attn_cp_rank
            ]
        )
        if actual != expected:
            raise RuntimeError(
                "CP-v2 scheduler/model physical-token mismatch: "
                f"dp_rank={dp_rank}, cp_rank={get_parallel().attn_cp_rank}, "
                f"metadata={actual}, scheduled={expected}, "
                f"global_cp_tokens={global_cp_tokens}"
            )

    if getattr(forward_batch, "out_cache_loc", None) is not None:
        forward_batch.out_cache_loc = forward_batch.out_cache_loc[:num_tokens]


def cp_split_before_forward(
    complete_hidden_states: Any,
    complete_position_ids: Any,
    forward_batch,
) -> Tuple[Optional[Any], Optional[Any]]:
    """Shard embeddings and positions for CP-v2 model-runner forwarding."""
    assert is_cp_v2_active(forward_batch)
    assert complete_hidden_states is not None
    assert getattr(forward_batch, "attn_cp_metadata", None) is not None
    return (
        cp_shard_hidden_states(complete_hidden_states, forward_batch),
        cp_shard_position_ids(complete_position_ids, forward_batch),
    )


def cp_shard_hidden_states(complete_hidden_states: Any, forward_batch):
    assert is_cp_v2_active(forward_batch)
    strategy = get_cp_strategy()
    assert strategy is not None
    assert complete_hidden_states is not None
    assert getattr(forward_batch, "attn_cp_metadata", None) is not None
    return strategy.shard_hidden_states(complete_hidden_states, forward_batch)


def cp_shard_position_ids(complete_position_ids: Any, forward_batch):
    assert is_cp_v2_active(forward_batch)
    strategy = get_cp_strategy()
    assert strategy is not None
    assert complete_position_ids is not None
    assert getattr(forward_batch, "attn_cp_metadata", None) is not None
    return strategy.shard_position_ids(complete_position_ids, forward_batch)


def cp_round_robin_input_ids_v2(input_ids: Any, forward_batch):
    assert is_cp_v2_active(forward_batch)
    if not get_moe_a2a_backend().is_none():
        return cp_shard_hidden_states(input_ids, forward_batch)

    physical_tokens = sum(forward_batch.attn_cp_metadata.per_rank_actual_token)
    padded_input_ids = input_ids.new_zeros(physical_tokens)
    padded_input_ids[: input_ids.shape[0]] = input_ids
    return padded_input_ids.view(-1, get_parallel().attn_cp_size).T.flatten()


def cp_gather_after_forward(x: Any, forward_batch, stream: Optional[Any] = None):
    """Gather CP-v2 hidden states at the model boundary when this batch is active."""
    assert is_cp_v2_active(forward_batch)
    strategy = get_cp_strategy()
    assert strategy is not None

    if isinstance(x, tuple):
        gathered = tuple(
            (
                strategy.gather_hidden_states(item, forward_batch, stream)
                if item is not None
                else None
            )
            for item in x
        )
        # MiMo's text-only body returns (hidden_states, None); logits expects a tensor.
        if len(gathered) == 2 and gathered[1] is None:
            return gathered[0]
        return gathered

    return strategy.gather_hidden_states(x, forward_batch, stream)


def cp_materialize_global_token_order(
    x: Any, forward_batch, stream: Optional[Any] = None
):
    """Materialize a CP tensor in the global logical token order."""
    if is_cp_v2_active(forward_batch):
        strategy = get_cp_strategy()
        assert strategy is not None
        return strategy.gather_kv_cache(x, forward_batch, stream)

    # TODO(hzh0425): Keep the legacy gather temporarily for CP-v1 compatibility. Remove it
    # with the follow-up CP-v1 cleanup.
    from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_output

    return cp_all_gather_rerange_output(
        x, get_parallel().attn_cp_size, forward_batch, stream
    )


@contextmanager
def cp_shard_model_inputs(
    complete_hidden_states: Any,
    complete_position_ids: Any,
    forward_batch,
):
    """Restore the shared batch so logits processing keeps full-batch metadata."""
    assert is_cp_v2_active(forward_batch)
    sharded_hidden_states = cp_shard_hidden_states(
        complete_hidden_states, forward_batch
    )
    sharded_positions = cp_shard_position_ids(complete_position_ids, forward_batch)

    spec_info = getattr(forward_batch, "spec_info", None)
    spec_hidden_states = getattr(spec_info, "hidden_states", None)
    spec_hidden_states_backup = None
    if (
        spec_hidden_states is not None
        and spec_hidden_states.shape[0] == complete_hidden_states.shape[0]
    ):
        spec_hidden_states_backup = spec_hidden_states
        spec_info.hidden_states = cp_shard_hidden_states(
            spec_hidden_states, forward_batch
        )

    # ``global_cp_num_tokens_cpu`` is populated only when attention-DP peers
    # must agree on a CP-local model-body buffer.  Plain DP1/CP runs retain
    # the legacy full-TP collective contract and do not need that remapping.
    cp_dp_context = (
        cp_local_dp_state(forward_batch)
        if getattr(forward_batch, "global_cp_num_tokens_cpu", None) is not None
        else nullcontext()
    )
    try:
        with cp_dp_context:
            yield sharded_hidden_states, sharded_positions
    finally:
        if spec_hidden_states_backup is not None:
            spec_info.hidden_states = spec_hidden_states_backup


@contextmanager
def cp_local_dp_state(forward_batch):
    """Use the per-CP DP layout while a model body participates in CP prefill.

    Active ranks enter through ``cp_shard_model_inputs``.  Idle DP peers must
    enter the same state explicitly so their MLP collectives use the identical
    per-CP communication group and buffer layout.
    """
    state = _enter_cp_local_dp_state(forward_batch)
    try:
        yield
    finally:
        _restore_cp_local_dp_state(forward_batch, state)


def _enter_cp_local_dp_state(forward_batch):
    """Expose CP-local token slots to DP-attention during the model body.

    Each CP rank gathers only the corresponding shard across attention-DP
    replicas.  MoE then gathers those DP-complete shards across CP.  After the
    model body returns, logits processing sees the original full-sequence
    per-DP layout again.
    """
    global_cp_tokens = getattr(forward_batch, "global_cp_num_tokens_cpu", None)
    if global_cp_tokens is None:
        raise RuntimeError(
            "CP-v2 with DP attention requires scheduler-provided "
            "global_cp_num_tokens_cpu"
        )

    import torch

    from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len

    parallel = get_parallel()
    cp_local_tokens = [int(tokens) for tokens in global_cp_tokens]
    local_index = parallel.attn_dp_rank if len(cp_local_tokens) > 1 else 0
    cp_local_gpu = torch.tensor(
        cp_local_tokens,
        dtype=forward_batch.global_num_tokens_gpu.dtype,
        device=forward_batch.global_num_tokens_gpu.device,
    )
    state = (
        forward_batch.global_num_tokens_cpu,
        forward_batch.global_num_tokens_gpu,
        forward_batch.global_dp_buffer_len,
        forward_batch.dp_padding_mode,
        forward_batch.dp_local_start_pos,
        forward_batch.dp_local_num_tokens,
        getattr(forward_batch, "dp_local_token_index", None),
        getattr(forward_batch, "cp_local_dp_layout", False),
    )

    forward_batch.global_num_tokens_cpu = cp_local_tokens
    forward_batch.global_num_tokens_gpu = cp_local_gpu
    forward_batch.global_dp_buffer_len = sum(cp_local_tokens)
    # Variable CP shards require SUM_LEN.  The full-TP all-reduce gathers the
    # distinct slots; MAX_LEN reduce-scatter assumes one slot per attention DP.
    forward_batch.dp_padding_mode = DpPaddingMode.SUM_LEN
    forward_batch.dp_local_start_pos = None
    forward_batch.dp_local_num_tokens = None
    forward_batch.dp_local_token_index = local_index
    forward_batch.cp_local_dp_layout = True
    set_dp_buffer_len(
        forward_batch.global_dp_buffer_len,
        cp_local_tokens[local_index],
        False,
        cp_local_tokens,
        cp_local_gpu,
    )
    return state


def _restore_cp_local_dp_state(forward_batch, state) -> None:
    from sglang.srt.layers.dp_attention import set_dp_buffer_len

    (
        global_num_tokens_cpu,
        global_num_tokens_gpu,
        global_dp_buffer_len,
        dp_padding_mode,
        dp_local_start_pos,
        dp_local_num_tokens,
        dp_local_token_index,
        cp_local_dp_layout,
    ) = state
    forward_batch.global_num_tokens_cpu = global_num_tokens_cpu
    forward_batch.global_num_tokens_gpu = global_num_tokens_gpu
    forward_batch.global_dp_buffer_len = global_dp_buffer_len
    forward_batch.dp_padding_mode = dp_padding_mode
    forward_batch.dp_local_start_pos = dp_local_start_pos
    forward_batch.dp_local_num_tokens = dp_local_num_tokens
    forward_batch.dp_local_token_index = dp_local_token_index
    forward_batch.cp_local_dp_layout = cp_local_dp_layout
    parallel = get_parallel()
    local_index = parallel.attn_dp_rank if len(global_num_tokens_cpu) > 1 else 0
    set_dp_buffer_len(
        global_dp_buffer_len,
        int(global_num_tokens_cpu[local_index]),
        dp_padding_mode.is_max_len(),
        global_num_tokens_cpu,
        global_num_tokens_gpu,
    )


def _to_int_list(values) -> Optional[list[int]]:
    if values is None:
        return None
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(x) for x in values]


__all__ = [
    "BaseContextParallelMetadata",
    "CPAttentionBackendKind",
    "ContextParallelMetadata",
    "ContextParallelStrategy",
    "ContextParallelStrategyKind",
    "InterleaveCPStrategy",
    "InterleaveContextParallelMetadata",
    "ZigzagCPStrategy",
    "ZigzagContextParallelMetadata",
    "CP_V2_DEFAULT_MODEL_CLASSES",
    "enable_cp_v2",
    "get_cp_v2_physical_token_count",
    "normalize_dp_cp_token_counts",
    "get_cp_strategy",
    "is_cp_v2_active",
    "cp_gather_after_forward",
    "cp_materialize_global_token_order",
    "cp_round_robin_input_ids_v2",
    "cp_shard_hidden_states",
    "cp_shard_model_inputs",
    "cp_shard_position_ids",
    "cp_split_before_forward",
    "prepare_cp_forward",
    "is_glm_dsa_cache_layer_split_enabled",
    "get_glm_dsa_cp_layer_shard_info",
    "get_glm_dsa_layer_split_effective_num_layers",
    "get_layer_shard_range",
    "get_layer_owner",
]
