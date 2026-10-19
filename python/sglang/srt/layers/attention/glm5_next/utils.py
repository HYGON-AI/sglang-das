from functools import lru_cache
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from sglang.srt.configs.model_config import get_dsa_index_kpool
from sglang.srt.environ import envs
from sglang.srt.layers.attention.glm5_next.runtime import (
    get_glm5_next_runtime_args as get_global_server_args,
)
from sglang.srt.layers.dp_attention import DpPaddingMode, get_attention_dp_rank
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import (
    get_bool_env_var,
    is_cuda,
    is_hcu,
    is_hip,
    is_pin_memory_available,
)
from sglang.srt.utils.common import ceil_align, ceil_div


def should_remap_pd_dsa_seed_to_local_slots(
    server_args: "ServerArgs", model_config=None
) -> bool:
    """Whether a PD IndexShare seed can be localized for fused TopK.

    Prefill transfers request-relative positions.  A decode worker can safely
    turn those positions into its own allocator slots only when it owns the
    complete, non-CP page table and does not need a HiSparse page swap.
    """
    from sglang.srt.layers.attention.glm5_next import is_glm5_next_hcu

    hf_config = getattr(model_config, "hf_config", None)
    hcu_glm5_no_kpool = (
        is_glm5_next_hcu(hf_config) and get_dsa_index_kpool(hf_config) <= 1
    )
    return (
        (is_cuda() or is_hcu())
        and envs.SGLANG_DSA_FUSE_TOPK.get()
        and server_args.disaggregation_mode == "decode"
        and not server_args.enable_hisparse
        and server_args.attn_cp_size == 1
        and not hcu_glm5_no_kpool
    )


def should_use_dsa_fused_topk(
    server_args: "ServerArgs",
    seed_dsa_topk_from_draft_extend: bool,
    model_config=None,
) -> bool:
    """Choose the index domain emitted by an DSA backend.

    The draft-extend backend on a PD prefill worker must emit logical
    request-relative positions for the wire.  On decode, those positions are
    remapped once at batch assembly, so the draft and target backends may use
    physical fused-TopK indices thereafter.
    """
    pd_index_share_seed = (
        seed_dsa_topk_from_draft_extend and server_args.disaggregation_mode != "null"
    )
    return envs.SGLANG_DSA_FUSE_TOPK.get() and (
        not pd_index_share_seed
        or should_remap_pd_dsa_seed_to_local_slots(server_args, model_config)
    )


def remap_pd_dsa_seed_to_local_slots(
    relative_positions: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Map request-relative PD seeds to decode-local physical KV slots.

    ``-1`` is the wire sentinel.  Any malformed active row invalidates the
    complete batch seed so callers can fall back to recomputing TopK instead
    of accidentally consuming allocator padding slot 0.
    """
    if relative_positions.ndim != 2 or req_pool_indices.ndim != 1:
        return None
    if relative_positions.shape[0] != req_pool_indices.shape[0]:
        return None
    if seq_lens.shape[0] != req_pool_indices.shape[0]:
        return None
    table_width = req_to_token.shape[1]
    if table_width == 0:
        return None

    # Reuse the existing batched Triton page-table transform so PD remap and
    # the regular unfused attention path share exactly the same logical→slot
    # semantics (including the int32 output contract).
    valid_positions = relative_positions >= 0
    gather_positions = relative_positions.clamp(min=0, max=table_width - 1).to(
        torch.int64
    )
    from sglang.srt.layers.attention.glm5_next.transform_index import (
        transform_index_page_table_decode,
    )

    local_page_table = req_to_token.index_select(
        0, req_pool_indices.to(dtype=torch.int64)
    )
    local_slots = transform_index_page_table_decode(
        page_table=local_page_table,
        topk_indices=gather_positions.to(dtype=torch.int32),
        page_size=1,
    )
    invalid_rows = torch.any(
        (relative_positions < -1)
        | (relative_positions >= seq_lens[:, None])
        | (relative_positions >= table_width)
        | (valid_positions & (local_slots <= 0)),
        dim=1,
    )
    local_slots.masked_fill_(~valid_positions, -1)
    local_slots.masked_fill_(invalid_rows[:, None], -1)
    if torch.any(torch.all(local_slots < 0, dim=1)).item():
        return None
    return local_slots


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
    if get_bool_env_var("SGLANG_DSA_HIP_DISABLE_PRESHUFFLE"):
        return False
    if get_bool_env_var("AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS"):
        return True
    try:
        from packaging.version import Version

        return Version(Version(triton.__version__).base_version) >= Version("3.5.0")
    except Exception:
        return False


if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.server_args import ServerArgs


def compute_dsa_seqlens(original_seq_lens, dsa_index_topk: int, index_kpool: int = 1):
    if index_kpool <= 1:
        return original_seq_lens.clamp(max=dsa_index_topk)

    full_pool_tokens = (
        torch.div(original_seq_lens, index_kpool, rounding_mode="floor") * index_kpool
    )
    selected_history_tokens = full_pool_tokens.clamp(max=dsa_index_topk)
    tail_tokens = original_seq_lens - full_pool_tokens
    return selected_history_tokens + tail_tokens


def is_dsa_enable_prefill_cp():
    return get_global_server_args().enable_dsa_prefill_context_parallel


def is_dsa_prefill_cp_in_seq_split():
    return (
        is_dsa_enable_prefill_cp()
        and get_global_server_args().dsa_prefill_cp_mode == "in-seq-split"
    )


def is_dsa_prefill_cp_round_robin_split():
    return (
        is_dsa_enable_prefill_cp()
        and get_global_server_args().dsa_prefill_cp_mode == "round-robin-split"
    )


def effective_forward_mode(forward_batch: "ForwardBatch"):
    """Return the pre-DP-padding forward mode.

    ``ForwardBatch.prepare_mlp_sync_batch`` temporarily rewrites
    ``forward_mode`` from DECODE/TARGET_VERIFY/DRAFT_EXTEND to EXTEND so the
    padded MLP sync path can be reused. Code that decides *what algorithm to
    run* (prefill vs. verify vs. decode kernels, CP scatter/gather, mamba
    metadata layout, etc.) must consult the ORIGINAL mode, not the remap.
    Reading raw ``forward_mode`` in those places causes the KDA/DSA kernels to
    interpret verify tree metadata as chunked prefill and step off the end of
    ``query_start_loc`` / ``extend_seq_lens``.
    """

    original_mode = forward_batch._original_forward_mode
    return original_mode if original_mode is not None else forward_batch.forward_mode


# Legacy alias kept for existing internal call sites.
_effective_forward_mode = effective_forward_mode


def dsa_prefill_has_history(forward_batch: "ForwardBatch") -> bool:
    """Whether this prefill batch needs previously cached DSA K/V state."""
    prefix_lens = forward_batch.extend_prefix_lens_cpu
    if prefix_lens is None:
        # Be conservative for modes that do not expose CPU prefix lengths.
        return True
    return any(int(prefix_len) > 0 for prefix_len in prefix_lens)


def can_dsa_prefill_cp_round_robin_split(forward_batch: "ForwardBatch"):
    if not _effective_forward_mode(forward_batch).is_context_parallel_extend():
        return False
    cp_size = get_parallel().attn_cp_size
    seq_len = sum(forward_batch.extend_seq_lens_cpu)
    return (
        is_dsa_prefill_cp_round_robin_split()
        and seq_len > 0
        and seq_len >= cp_size
        and cp_size > 1
    )


def cp_all_gather_rerange_fused(sources, cp_size, forward_batch, safe=False):
    # Fused multimem rerange is optional; return None to use cp_utils fallback.
    return None


def dsa_cp_round_robin_split_data(input_: Union[torch.Tensor, List]):
    """
    # for round-robin-split, split the tokens evenly according to the rule of token_idx % cp_size.
    |   +-----------before split------------+|
    | token0, token1, token2, token3, token4, token5, token6, token7, ...
    |
    |   +--------------result-------------------+
    | dp_atten_tp0: token0, token4, token8, token12, token16, ... |
    | dp_atten_tp1: token1, token5, token9, token13, token17, ... |
    | dp_atten_tp2: token2, token6, token10, token14, token18, ... |
    | dp_atten_tp3: token3, token7, token11, token15, token19, ... |
    |   +-------------------------+
    """
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

    # for torch device tensor
    return input_.view(-1, cp_size, *input_.shape[1:])[:, cp_rank].contiguous()


def cal_padded_tokens(forward_batch: "ForwardBatch"):
    # Consistent with the padding calculation logic in ForwardBatch.prepare_mlp_sync_batch,
    # calculate the actual token length after padding when attn_tp_size > 1 or in the MAX_LEN padding mode.
    global_num_tokens = forward_batch.global_num_tokens_cpu.copy()
    sync_group_size = len(global_num_tokens)
    attn_cp_size = get_parallel().attn_cp_size
    for i in range(sync_group_size):
        global_num_tokens[i] = ceil_align(global_num_tokens[i], attn_cp_size)
    dp_padding_mode = DpPaddingMode.get_dp_padding_mode(
        forward_batch.is_extend_in_batch, global_num_tokens
    )
    if dp_padding_mode.is_max_len():
        tokens = max(global_num_tokens)
    elif len(global_num_tokens) > 1:
        tokens = global_num_tokens[get_attention_dp_rank()]
    else:
        tokens = global_num_tokens[0]
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
    if is_dsa_prefill_cp_round_robin_split():
        cur_cp_seq_len = ceil_div(seq_len, cp_size)
    else:
        # TODO current just support prefill batch=1 and len(input_ids) > self.cp_size * 2
        # Note: (self.cp_size * 2) To achieve load balancing for seq computation,
        # the seq data needs to be divided and recombined at twice the size of cp_size.
        cur_cp_seq_len = seq_len // (cp_size * 2)
    if (
        cur_cp_seq_len != 0
        and cp_size > 1
        and use_dsa
        and _effective_forward_mode(forward_batch).is_context_parallel_extend()
        and is_dsa_enable_prefill_cp()
        and sum(forward_batch.extend_seq_lens_cpu) >= cp_size
    ):
        return True
    else:
        return False


# GLM NOTE: tokens is the per-batch sequence count; as a constexpr it baked a
# new kernel variant per batch size, and in_seqs is a view with batch-dependent
# alignment. Each variant is a full HCU JIT recompile on every rank (native
# heap residue per compile), so keep both out of the specialization key.
@triton.jit(
    do_not_specialize=["tokens"],
    do_not_specialize_on_alignment=["in_seqs_ptr"],
)
def dsa_cp_round_robin_split_q_seqs_kernel(
    in_seqs_ptr,
    out_seqs_ptr,
    bs_idx_ptr,
    tokens,
    cp_size: tl.constexpr,
    cp_rank: tl.constexpr,
):
    extra_seq = 0
    bs_idx = 0
    for bs in range(tokens):
        cur_len = tl.load(in_seqs_ptr + bs)
        cur_len += extra_seq
        cur_seq = cur_len // cp_size + (cur_len % cp_size > cp_rank)
        if cur_seq > 0:
            tl.store(bs_idx_ptr + bs_idx, bs)
            tl.store(out_seqs_ptr + bs_idx, cur_seq)
            bs_idx += 1
        extra_seq = cur_len - cur_seq * cp_size


def dsa_cp_round_robin_split_q_seqs_cpu(extend_seqs):
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    extra_seq = 0
    q_seqs = []
    for bs, cur_len in enumerate(extend_seqs):
        cur_len += extra_seq
        cur_seq = cur_len // cp_size + int(cur_len % cp_size > cp_rank)
        q_seqs.append(cur_seq)
        extra_seq = cur_len - cur_seq * cp_size
    bs_idx = list([i for i, x in enumerate(q_seqs) if x > 0])
    q_seqs = [q_len for q_len in q_seqs if q_len > 0]
    return q_seqs, bs_idx


def dsa_cp_round_robin_split_q_seqs(
    extend_seqs_cpu, extend_seqs
) -> Tuple[List, torch.Tensor, List, torch.Tensor]:
    """
    round-robin-split distributes tokens across ranks based on token_idx % cp_size.

    Return:
    ret_q_lens_cpu(List) and ret_q_lens(torch.Tensor): the partitioned length (excluding zeros) on the current cp rank
        for each sequence after distribution across cp ranks.
    bs_idx_cpu(List) and bs_idx(torch.Tensor): marks which sequences are ultimately selected,
        i.e., those with a partitioned length greater than zero.
    """
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    # len(ret_q_lens_cpu) == len(bs_idx_cpu)
    ret_q_lens_cpu, bs_idx_cpu = dsa_cp_round_robin_split_q_seqs_cpu(extend_seqs_cpu)
    ret_q_lens = torch.empty(
        (len(bs_idx_cpu),), device=extend_seqs.device, dtype=extend_seqs.dtype
    )
    bs_idx = torch.empty(
        (len(bs_idx_cpu),), device=extend_seqs.device, dtype=torch.int32
    )
    grid = (1,)
    dsa_cp_round_robin_split_q_seqs_kernel[grid](
        extend_seqs, ret_q_lens, bs_idx, len(extend_seqs), cp_size, cp_rank
    )
    return ret_q_lens_cpu, ret_q_lens, bs_idx_cpu, bs_idx


def dsa_use_prefill_cp(forward_batch, dsa_enable_prefill_cp=None):
    if dsa_enable_prefill_cp is None:
        dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
    # DP max-length padding temporarily maps decode/target-verify/speculative
    # modes to EXTEND so they can share the padded MLP synchronization path.
    # That mapping must not enable prefill CP: its token layout and CP metadata
    # still describe the original mode, and running the round-robin gather on
    # the padded speculative tensor can overrun the collective output buffer.
    effective_forward_mode = _effective_forward_mode(forward_batch)
    if (
        forward_batch.attn_cp_metadata is not None
        and dsa_enable_prefill_cp
        and effective_forward_mode.is_context_parallel_extend()
    ):
        return True
    else:
        return False


def copy_cpu_values_to_device(
    values,
    device: torch.device,
    *,
    dtype: Optional[torch.dtype] = None,
    dst: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Copy CPU values to a device tensor without forcing a sync when possible."""
    device = torch.device(device)
    if dst is not None:
        if dtype is None:
            dtype = dst.dtype
        elif dst.dtype != dtype:
            raise ValueError(f"dst dtype {dst.dtype} does not match dtype {dtype}.")

    src = torch.as_tensor(values, dtype=dtype)
    if src.device.type != "cpu":
        src = src.cpu()

    if dst is None:
        dst = torch.empty(src.shape, dtype=src.dtype, device=device)
    elif dst.shape != src.shape:
        raise ValueError(
            f"dst shape {dst.shape} does not match source shape {src.shape}."
        )

    use_pinned = is_pin_memory_available(device)
    cpu_values = torch.empty(src.shape, dtype=src.dtype, pin_memory=use_pinned)
    cpu_values.copy_(src)
    dst.copy_(cpu_values, non_blocking=use_pinned)
    return dst


def uses_logical_mtp_topk_indices(forward_batch) -> bool:
    from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState

    return IndexTopKShareState.from_mtp_carry(forward_batch).should_publish
