import logging
from functools import lru_cache
from typing import TYPE_CHECKING, List, Tuple, Union, Optional

import torch
import triton

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.forward_batch_utils import (
    effective_forward_mode,
)
from sglang.srt.layers.dp_attention import DpPaddingMode, dp_slot_in
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_memory,
    get_parallel,
    process_model_config,
)
from sglang.srt.utils import get_bool_env_var, is_cuda, is_hcu, is_hip, is_musa
from sglang.srt.utils.common import ceil_align, ceil_div

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def aiter_can_use_preshuffle_paged_mqa() -> bool:
    """Whether aiter's preshuffle paged MQA / cache kernels can be used on this runtime.

    aiter's ``deepgemm_fp8_paged_mqa_logits`` only supports ``KVBlockSize > 1`` and
    ``Preshuffle=True`` on its gluon kernel path. The gluon path is enabled when
    Triton >= 3.5.0, OR when ``AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1`` is set
    (which additionally requires that the AOT gluon kernel artifacts ship inside
    the aiter wheel/image). Otherwise aiter asserts ``KVBlockSize == 1`` and
    refuses ``Preshuffle=True``.

    sglang's DSA indexer uses this single decision to pick:
      * ``page_size``: 64 (preshuffle) vs 1 (legacy) on ROCm
      * ``Preshuffle`` / ``preshuffle`` flags on the aiter MQA + cache kernels
      * ``get_page_table_64`` vs ``get_page_table_1`` on the metadata
      * whether ``GetKAndS.execute`` uses the aiter or the triton implementation

    The result is cached so the cost is paid once per process.

    Set ``SGLANG_DSA_HIP_DISABLE_PRESHUFFLE=1`` to force the legacy path even when
    the gluon kernel would otherwise be available (useful for CI bisection).
    """
    if not is_hip():
        return False
    if not get_bool_env_var("SGLANG_USE_AITER"):
        return False
    if envs.SGLANG_DSA_HIP_DISABLE_PRESHUFFLE.get():
        return False
    if get_bool_env_var("AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS"):
        return True
    try:
        from packaging.version import Version

        return Version(Version(triton.__version__).base_version) >= Version("3.5.0")
    except Exception:
        return False


@lru_cache(maxsize=1)
def gfx950_fused_indexer_runtime_ok() -> bool:
    """Whether this runtime can serve the gfx950 fused indexer: aiter with
    preshuffled paged MQA, and kernels that build.

    Reached only on gfx950, since fused_decode.supported_hardware() is evaluated
    first. Every decline here is therefore a configuration or toolchain error;
    it is logged, as a warning when the path was asked for by name."""
    from sglang.srt.runtime_context import get_exec

    requested = get_exec().kernel.enable_dsa_fused_indexer
    if requested is False:
        return False  # asked for the standard path; not worth a line per server

    # Every decline logs its reason. Without this the path is invisible: a run
    # with the switch on and one with it off produce identical logs, and telling
    # the two apart cost a day of bisecting benchmark results.
    def _refuse(reason: str) -> bool:
        # Asked for by name: warn, but still start on the standard path.
        log = logger.warning if requested is True else logger.info
        log("gfx950 fused DSA indexer disabled: %s", reason)
        return False

    # No hardware term here: fused_decode.supported_hardware() is the hardware
    # half of the gate and runs first, so anything reaching this point is
    # already on gfx950. What is left is what a deployment can get wrong.
    if not get_bool_env_var("SGLANG_USE_AITER"):
        return _refuse("SGLANG_USE_AITER is not set")
    if not aiter_can_use_preshuffle_paged_mqa():
        return _refuse("aiter cannot use preshuffled paged MQA logits")
    from sglang.kernels.ops.attention.dsa.hip_gfx950 import loader

    # modules_or_none logged the build error; do not repeat the compiler output.
    if loader.modules_or_none() is None:
        return _refuse("the kernels failed to build, see the warning above")
    logger.info("gfx950 fused DSA indexer enabled")
    return True


def gfx950_model_shape_supported(**kwargs) -> bool:
    """Static per-model half of the gate: shapes and dtypes that cannot change
    after load."""
    from sglang.kernels.ops.attention.dsa.hip_gfx950 import model_shape_supported

    return model_shape_supported(**kwargs)


def hadamard_preserved(indexer) -> bool:
    """Whether Indexer._maybe_rotate still applies the Hadamard the fused kernels
    fold in. If not, the fused path must stay off, or prefill and decode would
    write different index-K formats."""
    device = indexer.k_norm.weight.device
    probe = torch.zeros(1, indexer.head_dim, dtype=torch.bfloat16, device=device)
    probe[0, 0] = 1.0
    rotated = indexer._maybe_rotate(probe)
    # A 128-point Hadamard sends e_0 to a vector whose every entry is 128**-0.5;
    # the identity leaves 127 zeros. Check the magnitude too, so a transform that
    # is merely dense does not pass for the rotation the kernels assume.
    expected = float(indexer.head_dim) ** -0.5
    if not bool(
        (rotated != 0).all()
        and torch.allclose(
            rotated.float().abs(),
            torch.full_like(rotated.float(), expected),
            rtol=0.05,
            atol=0.0,
        )
    ):
        logger.warning(
            "gfx950 fused DSA indexer disabled: Indexer._maybe_rotate does not "
            "apply the Hadamard rotation the fused kernels assume"
        )
        return False
    return True


# Tile size for the indexer FP8 K-cache preshuffle layout. Store and gather
# kernels reorganize each page into (tile x tile) blocks so the aiter preshuffle
# paged-MQA gather can consume the cache directly.
INDEXER_K_CACHE_PRESHUFFLE_TILE = 16


if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def compute_dsa_seqlens(original_seq_lens, dsa_index_topk: int, index_kpool: int = 1):
    if index_kpool <= 1:
        return original_seq_lens.clamp(max=dsa_index_topk)

    # Clamp only complete pools; the unfinished tail must remain selectable
    # outside the pooled top-k budget.
    full_pool_tokens = (
        torch.div(original_seq_lens, index_kpool, rounding_mode="floor") * index_kpool
    )
    selected_history_tokens = full_pool_tokens.clamp(max=dsa_index_topk)
    tail_tokens = original_seq_lens - full_pool_tokens
    return selected_history_tokens + tail_tokens


def should_remap_pd_dsa_seed_to_local_slots() -> bool:
    """Whether a PD seed should enter the allocator-local fused TopK domain."""
    from sglang.srt.layers.attention.glm5_next import is_glm5_next_hcu
    from sglang.srt.utils import is_hcu

    if is_hcu():
        config = process_model_config()
        if is_glm5_next_hcu(config.hf_config):
            from sglang.srt.layers.attention.glm5_next.runtime import (
                get_glm5_next_runtime_args,
            )
            from sglang.srt.layers.attention.glm5_next.utils import (
                should_remap_pd_dsa_seed_to_local_slots as hcu_should_remap,
            )

            return hcu_should_remap(get_glm5_next_runtime_args(), config)
    return (
        (is_cuda() or is_hip())
        and envs.SGLANG_DSA_FUSE_TOPK.get()
        and get_disagg().disaggregation_mode == "decode"
        and not get_memory().enable_hisparse
        and not get_parallel().dcp_enabled
    )


def should_use_dsa_fused_topk(seed_dsa_topk_from_draft_extend: bool) -> bool:
    """Select fused TopK for PD IndexShare.

    PD Prefill worker:
    - Target prefill: fused TopK enabled.
    - Draft extend: fused TopK disabled.

    PD Decode worker:
    - Draft decode / target verify / draft extend: fused TopK enabled.
    """
    pd_index_share_seed = (
        get_disagg().disaggregation_mode != "null" and seed_dsa_topk_from_draft_extend
    )
    return envs.SGLANG_DSA_FUSE_TOPK.get() and (
        not pd_index_share_seed or should_remap_pd_dsa_seed_to_local_slots()
    )


def dsa_prefill_has_history(forward_batch: "ForwardBatch") -> bool:
    prefix_lens = forward_batch.extend_prefix_lens_cpu
    return prefix_lens is None or any(int(length) > 0 for length in prefix_lens)


def is_dsa_enable_prefill_cp():
    if is_hcu():
        return get_parallel().enable_dsa_prefill_context_parallel
    if is_musa():
        return False

    # Generic prefill CP derives activation from the runtime topology and model
    # architecture.
    if get_parallel().attn_cp_size <= 1:
        return False
    from sglang.srt.configs.model_config import is_deepseek_dsa, is_deepseek_v4

    hf_config = process_model_config().hf_config
    if is_hip():
        return is_deepseek_v4(hf_config)
    return is_deepseek_dsa(hf_config) or is_deepseek_v4(hf_config)


def is_dsa_prefill_cp_interleave():
    return is_dsa_enable_prefill_cp() and get_parallel().cp_strategy == "interleave"


def is_dsa_prefill_cp_round_robin_split():
    if is_hcu():
        return (
            is_dsa_enable_prefill_cp()
            and get_parallel().dsa_prefill_cp_mode == "round-robin-split"
        )
    return is_dsa_prefill_cp_interleave()


# Structural surface where the graph DSA split-op dispatch (DSA indexer) and the
# MLA BMM-into-attention fusion apply: a non-speculative extend (prefill) running
# inside a piecewise/breakable CUDA graph. Both fusions are now on by default on
# this surface (no feature flag); each adds its own extra carve-outs at its call
# site (e.g. the indexer also excludes DSA prefill context parallelism).
def is_graph_dsa_split_op_surface(forward_batch: "ForwardBatch") -> bool:
    return (
        is_cuda()
        and (is_in_tc_piecewise_cuda_graph() or is_in_breakable_cuda_graph())
        and effective_forward_mode(forward_batch).is_extend_without_speculative()
    )


def can_dsa_prefill_cp_interleave(forward_batch: "ForwardBatch"):
    if not forward_batch.forward_mode.is_context_parallel_extend():
        return False
    cp_size = get_parallel().attn_cp_size
    seq_len = sum(forward_batch.extend_seq_lens_cpu)
    return (
        is_dsa_prefill_cp_interleave()
        and seq_len > 0
        and seq_len >= cp_size
        and cp_size > 1
    )


def can_dsa_prefill_cp_round_robin_split(forward_batch: "ForwardBatch"):
    if not effective_forward_mode(forward_batch).is_context_parallel_extend():
        return False
    cp_size = get_parallel().attn_cp_size
    seq_len = sum(forward_batch.extend_seq_lens_cpu)
    return (
        is_dsa_prefill_cp_round_robin_split()
        and seq_len > 0
        and seq_len >= cp_size
        and cp_size > 1
    )


def dsa_cp_round_robin_split_data(input_: Union[torch.Tensor, List]):
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    if isinstance(input_, (tuple, list)):
        indices = range(cp_rank, len(input_), cp_size)
        return input_[indices]

    tokens = len(input_)
    if tokens % cp_size != 0:
        cur_len = tokens // cp_size + (tokens % cp_size > cp_rank)
        if cur_len == 0:
            return input_.new_empty(0, *input_.shape[1:])
        indices = torch.arange(cp_rank, tokens, cp_size, device=input_.device)
        return input_[indices]

    shard = input_.view(-1, cp_size, *input_.shape[1:])[:, cp_rank]
    if shard.stride(0) != input_.stride(0):
        shard = shard.clone(memory_format=torch.contiguous_format)
    return shard


def cal_padded_tokens(forward_batch: "ForwardBatch"):
    # Consistent with the padding calculation logic in ForwardBatch.prepare_mlp_sync_batch,
    # calculate the actual token length after padding when attn_tp_size > 1 or in the MAX_LEN padding mode.
    from sglang.srt.layers.cp.padding import get_cp_padding_align_size
    from sglang.srt.layers.cp.utils import enable_cp_v2, is_cp_active

    # CP-v2 already pads each rank-local shard to its physical size
    if is_cp_active(forward_batch):
        return forward_batch.attn_cp_metadata.per_rank_actual_token[
            get_parallel().attn_cp_rank
        ]

    global_num_tokens = forward_batch.global_num_tokens_cpu.copy()
    sync_group_size = len(global_num_tokens)
    attn_cp_size = get_parallel().attn_cp_size
    if not enable_cp_v2():
        cp_align_size = get_cp_padding_align_size()
        for i in range(sync_group_size):
            global_num_tokens[i] = ceil_align(global_num_tokens[i], cp_align_size)
    # Reuse the mode selected when the DP buffer was prepared.
    dp_padding_mode = forward_batch.dp_padding_mode
    if dp_padding_mode is None:
        dp_padding_mode = DpPaddingMode.get_dp_padding_mode(
            forward_batch.is_extend_in_batch, global_num_tokens
        )
    if dp_padding_mode.is_max_len():
        tokens = max(global_num_tokens)
    else:
        tokens = global_num_tokens[dp_slot_in(global_num_tokens)]
    if can_dsa_prefill_cp_round_robin_split(forward_batch):
        tokens = ceil_div(tokens, attn_cp_size)
    return tokens


def pad_dsa_cache_seqlens(forward_batch: "ForwardBatch", dsa_cache_seqlens):
    attn_cp_size = get_parallel().attn_cp_size
    needs_cp_pad = attn_cp_size > 1 and can_dsa_prefill_cp_round_robin_split(
        forward_batch
    )
    needs_dp_pad = forward_batch.global_num_tokens_cpu is not None
    if not needs_cp_pad and not needs_dp_pad:
        return dsa_cache_seqlens
    tokens = cal_padded_tokens(forward_batch)
    pad_len = tokens - dsa_cache_seqlens.shape[0]
    if pad_len > 0:
        dsa_cache_seqlens = torch.cat(
            [
                dsa_cache_seqlens,
                dsa_cache_seqlens.new_zeros(pad_len, *dsa_cache_seqlens.shape[1:]),
            ]
        )
    return dsa_cache_seqlens


def can_dsa_cp_split(seq_len: int, cp_size: int, use_dsa: bool, forward_batch):
    if (
        cp_size <= 1
        or not use_dsa
        or not effective_forward_mode(forward_batch).is_context_parallel_extend()
        or not is_dsa_enable_prefill_cp()
        or sum(forward_batch.extend_seq_lens_cpu) < cp_size
    ):
        return False
    if is_dsa_prefill_cp_round_robin_split():
        cur_cp_seq_len = seq_len // cp_size
        assert seq_len % cp_size == 0, (
            f"Expect seq len can divided by cp size, but got seq len {seq_len}, cp size {cp_size}"
        )
    else:
        cur_cp_seq_len = seq_len // (cp_size * 2)
    return cur_cp_seq_len != 0


from sglang.kernels.ops.attention.dsa.cp_split import (
    dsa_cp_interleave_q_seqs_kernel as dsa_cp_round_robin_split_q_seqs_kernel,
)


def dsa_cp_round_robin_split_q_seqs_cpu(extend_seqs):
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    extra_seq = 0
    q_seqs = []
    for cur_len in extend_seqs:
        cur_len += extra_seq
        cur_seq = cur_len // cp_size + int(cur_len % cp_size > cp_rank)
        q_seqs.append(cur_seq)
        extra_seq = cur_len - cur_seq * cp_size
    bs_idx = [i for i, q_len in enumerate(q_seqs) if q_len > 0]
    q_seqs = [q_len for q_len in q_seqs if q_len > 0]
    return q_seqs, bs_idx


def dsa_cp_round_robin_split_q_seqs(
    extend_seqs_cpu, extend_seqs
) -> Tuple[List, torch.Tensor, List, torch.Tensor]:
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    ret_q_lens_cpu, bs_idx_cpu = dsa_cp_round_robin_split_q_seqs_cpu(
        extend_seqs_cpu
    )
    ret_q_lens = torch.empty(
        (len(bs_idx_cpu),), device=extend_seqs.device, dtype=extend_seqs.dtype
    )
    bs_idx = torch.empty(
        (len(bs_idx_cpu),), device=extend_seqs.device, dtype=torch.int32
    )
    dsa_cp_round_robin_split_q_seqs_kernel[(1,)](
        extend_seqs, ret_q_lens, bs_idx, len(extend_seqs), cp_size, cp_rank
    )
    return ret_q_lens_cpu, ret_q_lens, bs_idx_cpu, bs_idx


def dsa_use_prefill_cp(forward_batch, dsa_enable_prefill_cp=None):
    if dsa_enable_prefill_cp is None:
        dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
    if (
        forward_batch.attn_cp_metadata is not None
        and dsa_enable_prefill_cp
        and effective_forward_mode(forward_batch).is_context_parallel_extend()
    ):
        return True
    else:
        return False


def _dsa_prefill_has_history(forward_batch: "ForwardBatch") -> bool:
    prefix_lens = forward_batch.extend_prefix_lens_cpu
    if prefix_lens is None:
        return True
    return any(int(prefix_len) > 0 for prefix_len in prefix_lens)


def maybe_configure_main_kv_page_plan(forward_batch: "ForwardBatch") -> None:
    """Install this batch's compact Main-KV mapping when LayerSplit is active."""

    from sglang.srt.mem_cache.dsa_cache_layer_split import build_main_kv_page_plan
    from sglang.srt.model_executor.forward_context import get_req_to_token_pool

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
    forward_batch: "ForwardBatch",
    full_attention_layer_id: Optional[int],
) -> None:
    """Configure the batch plan and prefetch one DSA layer's caches."""

    maybe_configure_main_kv_page_plan(forward_batch)

    if full_attention_layer_id is None or not dsa_use_prefill_cp(forward_batch):
        return

    token_to_kv_pool = get_token_to_kv_pool()
    has_history = _dsa_prefill_has_history(forward_batch)
    prefetch_mla = getattr(token_to_kv_pool, "prefetch_mla_kv_buffer", None)
    if prefetch_mla is not None:
        prefetch_mla(full_attention_layer_id, has_history=has_history)
    else:
        # Index-K and Main-KV share a communicator, so all ranks enqueue them
        # in this order. A skip-topk layer makes the Index-K call a no-op.
        prefetch_index = getattr(token_to_kv_pool, "prefetch_index_buffer", None)
        if is_hcu() and prefetch_index is not None:
            prefetch_index(full_attention_layer_id, has_history=has_history)
        prefetch_kv = getattr(token_to_kv_pool, "prefetch_kv_buffer", None)
        if prefetch_kv is not None:
            prefetch_kv(full_attention_layer_id, has_history=has_history)


def maybe_prefetch_next_full_attention_kv(
    forward_batch: "ForwardBatch",
    next_full_attention_layer_id: Optional[int],
) -> None:
    """Prefetch the next layer while the current layer's MLP runs."""

    maybe_prefetch_full_attention_kv(forward_batch, next_full_attention_layer_id)



def fp8_mqa_logits_ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def fp8_mqa_logits_make_fused_kv(
    kv_fp8: torch.Tensor,
    kv_scales: torch.Tensor,
    block_kv: int,
    head_dim: int,
) -> torch.Tensor:
    num_phys_blocks = kv_fp8.shape[0]
    per_token_size = head_dim + 4
    block_bytes = block_kv * per_token_size
    scale_offset = block_kv * head_dim

    fused = torch.zeros(
        num_phys_blocks, block_bytes, dtype=torch.uint8, device=kv_fp8.device
    )
    for blk in range(num_phys_blocks):
        fused[blk, :scale_offset] = kv_fp8[blk].view(torch.uint8).reshape(-1)
        fused[blk, scale_offset:] = (
            kv_scales[blk].float().contiguous().view(torch.uint8).reshape(-1)
        )
    return fused.view(num_phys_blocks, block_kv, 1, per_token_size)
