# Copyright 2025 XunhaoLai. All rights reserved.

import os
from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from ..common.utils import (
    check_sparse_kv_fp8,
    get_cu_seqblocks,
    robust_allocator,
    sparse_out_dtype,
    unit_scale,
)


@triton.jit
def _scatter_query_group_union_kernel(
    topk_ptr,
    membership_ptr,
    cu_seqlens_q,
    cu_seqblocks_q,
    num_heads,
    stride_th,
    stride_tn,
    stride_tk,
    stride_mh,
    stride_mg,
    stride_mb,
    max_num_blocks,
    TOPK: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    FLAT_SIZE: tl.constexpr,
    BUILD_QUERY_MASK: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_block_start = tl.load(cu_seqblocks_q + pid_b)
    q_block_len = tl.load(cu_seqblocks_q + pid_b + 1) - q_block_start
    if pid_q >= q_block_len:
        return

    off = tl.arange(0, FLAT_SIZE)
    row = off // TOPK
    col = off % TOPK
    valid = (row < BLOCK_SIZE_Q) & (pid_q * BLOCK_SIZE_Q + row < q_len)
    values = tl.load(
        topk_ptr
        + pid_h * stride_th
        + (q_start + pid_q * BLOCK_SIZE_Q + row) * stride_tn
        + col * stride_tk,
        mask=valid,
        other=0x7FFFFFFF,
    ).to(tl.int32)
    valid = valid & (values >= 0) & (values < max_num_blocks)
    membership = (
        membership_ptr
        + pid_h * stride_mh
        + (q_block_start + pid_q) * stride_mg
        + values * stride_mb
    )
    if BUILD_QUERY_MASK:
        tl.atomic_or(membership, 1 << row, mask=valid)
    else:
        # A single program owns the complete (head, query-group) row.
        # Duplicate lanes only write the same value.
        tl.store(membership, 1, mask=valid)


@triton.jit
def _compact_query_group_union_kernel(
    membership_ptr,
    union_ptr,
    query_mask_ptr,
    overflow_ptr,
    num_groups,
    max_num_blocks,
    stride_mh,
    stride_mg,
    stride_mb,
    stride_uh,
    stride_ug,
    stride_uk,
    stride_qmh,
    stride_qmg,
    stride_qmu,
    BLOCK_MAX: tl.constexpr,
    MAX_UNION: tl.constexpr,
    BUILD_QUERY_MASK: tl.constexpr,
):
    pid_g = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_g >= num_groups:
        return

    block_id = tl.arange(0, BLOCK_MAX)
    membership = tl.load(
        membership_ptr + pid_h * stride_mh + pid_g * stride_mg + block_id * stride_mb,
        mask=block_id < max_num_blocks,
        other=0,
    )
    present = membership != 0
    union_pos = tl.cumsum(present.to(tl.int32), axis=0) - 1
    tl.store(
        union_ptr + pid_h * stride_uh + pid_g * stride_ug + union_pos * stride_uk,
        block_id,
        mask=present & (union_pos < MAX_UNION),
    )
    if BUILD_QUERY_MASK:
        tl.store(
            query_mask_ptr
            + pid_h * stride_qmh
            + pid_g * stride_qmg
            + union_pos * stride_qmu,
            membership,
            mask=present & (union_pos < MAX_UNION),
        )
    unique_count = tl.sum(present.to(tl.int32), axis=0)
    # Some HCU Triton builds may issue masked vector stores with colliding
    # destination offsets.  Re-establish the padding contract explicitly.
    off_u = tl.arange(0, MAX_UNION)
    tl.store(
        union_ptr + pid_h * stride_uh + pid_g * stride_ug + off_u * stride_uk,
        -1,
        mask=off_u >= unique_count,
    )
    tl.store(overflow_ptr + pid_h * num_groups + pid_g, unique_count > MAX_UNION)


@triton.jit
def _fused_query_group_topk_union_kernel(
    topk_ptr,
    membership_ptr,
    union_ptr,
    query_mask_ptr,
    overflow_ptr,
    cu_seqlens_q,
    cu_seqblocks_q,
    num_heads,
    stride_th,
    stride_tn,
    stride_tk,
    stride_mh,
    stride_mg,
    stride_mb,
    stride_uh,
    stride_ug,
    stride_uk,
    stride_qmh,
    stride_qmg,
    stride_qmu,
    num_groups,
    max_num_blocks,
    TOPK: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    FLAT_SIZE: tl.constexpr,
    BLOCK_MAX: tl.constexpr,
    MAX_UNION: tl.constexpr,
    BUILD_QUERY_MASK: tl.constexpr,
):
    """Build one grouped Top-K union without intermediate kernel launches.

    A single program owns one (batch, KV head, query group), so it can clear
    its scratch row, scatter all per-query Top-K ids, and compact the row after
    workgroup barriers.  In particular, the input ids do not need to be sorted:
    compaction walks block ids in increasing order and therefore preserves the
    sparse-attention consumer's sorted-union contract.
    """

    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_block_start = tl.load(cu_seqblocks_q + pid_b)
    q_block_len = tl.load(cu_seqblocks_q + pid_b + 1) - q_block_start
    group = q_block_start + pid_q

    block_id = tl.arange(0, BLOCK_MAX)
    scratch = (
        membership_ptr + pid_h * stride_mh + group * stride_mg + block_id * stride_mb
    )
    valid_group = pid_q < q_block_len
    tl.store(scratch, 0, mask=valid_group & (block_id < max_num_blocks))
    tl.debug_barrier()

    off = tl.arange(0, FLAT_SIZE)
    row = off // TOPK
    col = off % TOPK
    valid = valid_group & (row < BLOCK_SIZE_Q) & (pid_q * BLOCK_SIZE_Q + row < q_len)
    values = tl.load(
        topk_ptr
        + pid_h * stride_th
        + (q_start + pid_q * BLOCK_SIZE_Q + row) * stride_tn
        + col * stride_tk,
        mask=valid,
        other=-1,
    ).to(tl.int32)
    # AITER returns score-ranked ids, whereas the established sparse-main
    # numerical contract consumes every query row in ascending block-id order.
    # Sort all rows owned by this query-group program and write them back in
    # place.  This replaces the former standalone sort launch while preserving
    # the exact Q1 accumulation order used by accuracy baselines.
    values_2d = tl.reshape(values, (BLOCK_SIZE_Q, TOPK))
    values_2d = tl.sort(values_2d, dim=1, descending=False)
    values = tl.reshape(values_2d, (FLAT_SIZE,))
    tl.store(
        topk_ptr
        + pid_h * stride_th
        + (q_start + pid_q * BLOCK_SIZE_Q + row) * stride_tn
        + col * stride_tk,
        values,
        mask=valid,
    )
    valid = valid & (values >= 0) & (values < max_num_blocks)
    value_scratch = (
        membership_ptr + pid_h * stride_mh + group * stride_mg + values * stride_mb
    )
    if BUILD_QUERY_MASK:
        tl.atomic_or(value_scratch, 1 << row, mask=valid)
    else:
        tl.atomic_or(value_scratch, 1, mask=valid)
    tl.debug_barrier()

    membership = tl.load(
        scratch,
        mask=valid_group & (block_id < max_num_blocks),
        other=0,
    )
    present = membership != 0
    union_pos = tl.cumsum(present.to(tl.int32), axis=0) - 1
    tl.store(
        union_ptr + pid_h * stride_uh + group * stride_ug + union_pos * stride_uk,
        block_id,
        mask=valid_group & present & (union_pos < MAX_UNION),
    )
    if BUILD_QUERY_MASK:
        tl.store(
            query_mask_ptr
            + pid_h * stride_qmh
            + group * stride_qmg
            + union_pos * stride_qmu,
            membership,
            mask=valid_group & present & (union_pos < MAX_UNION),
        )
    unique_count = tl.sum(present.to(tl.int32), axis=0)
    off_u = tl.arange(0, MAX_UNION)
    tl.store(
        union_ptr + pid_h * stride_uh + group * stride_ug + off_u * stride_uk,
        -1,
        mask=valid_group & (off_u >= unique_count),
    )
    if BUILD_QUERY_MASK:
        tl.store(
            query_mask_ptr
            + pid_h * stride_qmh
            + group * stride_qmg
            + off_u * stride_qmu,
            0,
            mask=valid_group & (off_u >= unique_count),
        )
    tl.store(
        overflow_ptr + pid_h * num_groups + group,
        unique_count > MAX_UNION,
        mask=valid_group,
    )


def build_query_group_topk_union(
    topk_idx: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqblocks_q: torch.Tensor,
    block_size_q: int,
    all_groups: int,
    max_num_blocks: int,
    max_union: int = 256,
    return_query_mask: bool = False,
    workspace: Optional[dict[str, torch.Tensor]] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Compact per-query Top-K rows into one sorted union per query group."""

    if block_size_q not in (2, 4, 8, 16, 32, 64):
        raise ValueError(f"unsupported grouped-union block_size_q={block_size_q}")
    num_heads, _, topk = topk_idx.shape
    flat_size = triton.next_power_of_2(block_size_q * topk)
    if flat_size > 1024:
        raise ValueError(f"grouped union input is too large: {block_size_q}*{topk}")
    if max_num_blocks <= 0:
        raise ValueError(f"max_num_blocks must be positive, got {max_num_blocks}")
    # The grouped consumer is exact only when every per-query selected block
    # fits.  Never make correctness depend on a model-dependent empirical
    # union-size observation; the theoretical bound is small (16*16=256 for
    # MiniMax-M3) and costs <1% versus a 64-entry buffer on BW1100.
    if max_union < block_size_q * topk:
        raise ValueError(
            f"max_union={max_union} is smaller than the exact bound "
            f"block_size_q*topk={block_size_q * topk}"
        )
    if max_union & (max_union - 1):
        raise ValueError(f"max_union must be a power of two, got {max_union}")
    if return_query_mask and block_size_q > 31:
        raise ValueError("query-membership bitmask supports block_size_q <= 31")
    fused_union = os.environ.get("SGLANG_MINIMAX_PREFILL_FUSED_TOPK_UNION", "0") == "1"
    if fused_union and flat_size != block_size_q * topk:
        raise ValueError(
            "fused grouped union requires a power-of-two Top-K so each query "
            f"row can be sorted in-kernel, got topk={topk}"
        )
    if workspace is None:
        membership = torch.empty(
            (num_heads, all_groups, max_num_blocks),
            dtype=torch.int32,
            device=topk_idx.device,
        )
        union = torch.empty(
            (num_heads, all_groups, max_union),
            dtype=torch.int32,
            device=topk_idx.device,
        )
        overflow = torch.empty(
            (num_heads, all_groups), dtype=torch.bool, device=topk_idx.device
        )
        query_mask = (
            torch.empty(
                (num_heads, all_groups, max_union),
                dtype=torch.int32,
                device=topk_idx.device,
            )
            if return_query_mask
            else None
        )
    else:
        expected = {
            "membership": ((num_heads, all_groups, max_num_blocks), torch.int32),
            "union": ((num_heads, all_groups, max_union), torch.int32),
            "overflow": ((num_heads, all_groups), torch.bool),
        }
        for name, (shape, dtype) in expected.items():
            value = workspace.get(name)
            if (
                value is None
                or value.shape != shape
                or value.dtype != dtype
                or value.device != topk_idx.device
                or not value.is_contiguous()
            ):
                raise ValueError(
                    f"invalid grouped-union workspace[{name!r}]: "
                    f"shape={getattr(value, 'shape', None)} "
                    f"dtype={getattr(value, 'dtype', None)}"
                )
        membership = workspace["membership"]
        union = workspace["union"]
        overflow = workspace["overflow"]
        if return_query_mask:
            query_mask = workspace.get("query_mask")
            if (
                query_mask is None
                or query_mask.shape != (num_heads, all_groups, max_union)
                or query_mask.dtype != torch.int32
                or query_mask.device != topk_idx.device
                or not query_mask.is_contiguous()
            ):
                raise ValueError("invalid grouped-union workspace['query_mask']")
        else:
            query_mask = None
    batch_size = cu_seqlens_q.shape[0] - 1
    # all_groups is the authoritative number of q-block groups. Using
    # ceil(total_q / block_size_q) is wrong for non-power-of-two group sizes
    # such as MTP G=3 (block_size_q=4).
    max_groups = max(1, int(all_groups))
    if fused_union:
        _fused_query_group_topk_union_kernel[
            (
                max_groups,
                batch_size * num_heads,
            )
        ](
            topk_idx,
            membership,
            union,
            query_mask,
            overflow,
            cu_seqlens_q,
            cu_seqblocks_q,
            num_heads,
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_idx.stride(2),
            membership.stride(0),
            membership.stride(1),
            membership.stride(2),
            union.stride(0),
            union.stride(1),
            union.stride(2),
            query_mask.stride(0) if query_mask is not None else 0,
            query_mask.stride(1) if query_mask is not None else 0,
            query_mask.stride(2) if query_mask is not None else 0,
            all_groups,
            max_num_blocks,
            TOPK=topk,
            BLOCK_SIZE_Q=block_size_q,
            FLAT_SIZE=flat_size,
            BLOCK_MAX=triton.next_power_of_2(max_num_blocks),
            MAX_UNION=max_union,
            BUILD_QUERY_MASK=return_query_mask,
            num_warps=8,
            num_stages=1,
        )
    else:
        membership.zero_()
        union.fill_(-1)
        overflow.zero_()
        if query_mask is not None:
            query_mask.zero_()
        _scatter_query_group_union_kernel[(max_groups, batch_size * num_heads)](
            topk_idx,
            membership,
            cu_seqlens_q,
            cu_seqblocks_q,
            num_heads,
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_idx.stride(2),
            membership.stride(0),
            membership.stride(1),
            membership.stride(2),
            max_num_blocks,
            TOPK=topk,
            BLOCK_SIZE_Q=block_size_q,
            FLAT_SIZE=flat_size,
            BUILD_QUERY_MASK=return_query_mask,
            num_warps=4,
            num_stages=2,
        )
        _compact_query_group_union_kernel[(all_groups, num_heads)](
            membership,
            union,
            query_mask,
            overflow,
            all_groups,
            max_num_blocks,
            membership.stride(0),
            membership.stride(1),
            membership.stride(2),
            union.stride(0),
            union.stride(1),
            union.stride(2),
            query_mask.stride(0) if query_mask is not None else 0,
            query_mask.stride(1) if query_mask is not None else 0,
            query_mask.stride(2) if query_mask is not None else 0,
            BLOCK_MAX=triton.next_power_of_2(max_num_blocks),
            MAX_UNION=max_union,
            BUILD_QUERY_MASK=return_query_mask,
            num_warps=8,
            num_stages=2,
        )
    if os.environ.get("SGLANG_MINIMAX_PREFILL_UNION_STRICT_CHECK", "0") == "1":
        if bool(overflow.any().item()):
            raise RuntimeError(
                f"MiniMax grouped Top-K union exceeded max_union={max_union}"
            )
    if query_mask is not None:
        return union, query_mask
    return union


@triton.heuristics(
    {
        "BLOCK_SIZE_KD": lambda args: triton.next_power_of_2(args["qk_head_dim"]),
        "BLOCK_SIZE_VD": lambda args: triton.next_power_of_2(args["v_head_dim"]),
        "BLOCK_SIZE_H": lambda args: triton.next_power_of_2(
            max(
                16 // args["BLOCK_SIZE_Q"],
                triton.next_power_of_2(args["gqa_group_size"]),
            )
        ),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
        "BLOCK_SIZE_QH": lambda args: args["BLOCK_SIZE_Q"] * args["BLOCK_SIZE_H"],
        "HAS_SINK": lambda args: args["sink_ptr"] is not None,
    }
)
@triton.autotune(
    # Configs that fail to compile on the target arch are skipped, so widening
    # the num_warps x num_stages grid only adds candidates, never a bad kernel.
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in (2, 4, 8)
        for ns in (2, 3, 4)
    ],
    key=[
        "BLOCK_SIZE_Q",
        "BLOCK_SIZE_K",
        "qk_head_dim",
        "v_head_dim",
        "gqa_group_size",
        "max_topk",
        "autotune_q_bucket",
    ],
)
@triton.jit
def _gqa_share_sparse_fwd_kernel(
    q_ptr,  # Q: n x h x d
    k_cache_ptr,  # K paged: max_slots x kh x d
    v_cache_ptr,  # V paged: max_slots x kh x d
    sink_ptr,  # Sink: h x d
    t_ptr,  # topk_idx: kh x n x k
    exact_t_ptr,  # optional per-query topk: kh x total_q x exact_topk
    query_group_mask_ptr,  # optional bitmask: kh x query_group x union
    o_ptr,  # O: n x h x d
    req_to_token_ptr,  # req_to_token: max_reqs x max_kv_len
    # seqlens
    cu_seqlens_q,
    cu_seqblocks_q,
    seq_lens,
    prefix_lens,
    slot_ids,
    # shape
    max_slots,
    num_kv_heads,
    gqa_group_size,
    qk_head_dim,
    v_head_dim,
    max_topk,
    autotune_q_bucket,
    # q loop num
    num_q_loop,
    # sm_scale
    sm_scale,
    # per-tensor KV dequant scales (1.0 when the cache is unit-scaled)
    k_scale,
    v_scale,
    # Scale softmax probabilities into the useful e4m3 range before FP8 PV.
    p_scale,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_sh,
    stride_sd,
    stride_th,
    stride_tn,
    stride_tk,
    stride_eth,
    stride_etn,
    stride_etk,
    stride_mh,
    stride_mg,
    stride_mu,
    stride_on,
    stride_oh,
    stride_od,
    stride_r2t_b,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_KD: tl.constexpr,
    BLOCK_SIZE_VD: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
    # has sink
    HAS_SINK: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_PV: tl.constexpr,
    PER_QUERY_MASK: tl.constexpr,
    PRECOMPUTED_QUERY_MASK: tl.constexpr,
    EXACT_TOPK: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    # get batch id and head id
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_h = pid_kh * gqa_group_size
    # get q k start and len after rmpad
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_block_start = tl.load(cu_seqblocks_q + pid_b)
    q_block_len = tl.load(cu_seqblocks_q + pid_b + 1) - q_block_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    sid = (
        tl.load(slot_ids + pid_b).to(tl.int64) + max_slots
    ) % max_slots  # safety against negative
    if pid_q * num_q_loop >= q_block_len:
        return
    real_q_loop = min(num_q_loop, q_block_len - pid_q * num_q_loop)
    if HAS_SINK:
        sink_ptrs = tl.make_block_ptr(
            base=sink_ptr + pid_h * stride_sh,
            shape=(gqa_group_size, qk_head_dim),
            strides=(stride_sh, stride_sd),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_KD),
            order=(1, 0),
        )
        sink = tl.load(sink_ptrs, boundary_check=(0, 1), padding_option="zero").to(
            tl.float32
        )
    # offsets for paged K/V load
    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_kd = tl.arange(0, BLOCK_SIZE_KD)
    off_vd = tl.arange(0, BLOCK_SIZE_VD)
    kd_mask = off_kd < qk_head_dim
    vd_mask = off_vd < v_head_dim
    for j in range(real_q_loop):
        pid_q_j = pid_q * num_q_loop + j
        # init topk idx pointer
        t_ptr_j = t_ptr + (q_block_start + pid_q_j) * stride_tn + pid_kh * stride_th
        # we assume that the topk_idx is right padded with -1
        off_t = tl.arange(0, BLOCK_SIZE_T)
        topk_idx = tl.load(t_ptr_j + off_t * stride_tk, mask=off_t < max_topk, other=-1)
        valid_idx = tl.where(topk_idx >= 0, off_t, -1)
        real_topk = tl.sum(valid_idx != -1, axis=0)
        # init qkv pointer
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
            shape=(q_len, gqa_group_size, qk_head_dim),
            strides=(stride_qn, stride_qh, stride_qd),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_KD),
            order=(2, 1, 0),
        )
        # load q, shape: [BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D] -> [BLOCK_SIZE_QH, BLOCK_SIZE_D]
        q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
        # init statistics
        off_q_k = (
            tl.arange(0, BLOCK_SIZE_Q)[:, None]
            + pid_q_j * BLOCK_SIZE_Q
            + prefix_len
            - tl.arange(0, BLOCK_SIZE_K)[None, :]
        )
        if HAS_SINK:
            m_i = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H), dtype=tl.float32)
            lse_i = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H), dtype=tl.float32)
            qsink = (
                tl.sum(q.to(tl.float32) * sink[None, :, :], axis=2) * sm_scale_log2e
            )  # (BLOCK_SIZE_Q, BLOCK_SIZE_H)
            m_i += qsink
            lse_i += qsink
            m_i = tl.reshape(m_i, BLOCK_SIZE_QH)
            lse_i = tl.reshape(lse_i, BLOCK_SIZE_QH)
        else:
            m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
            lse_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
        acc_o = tl.full((BLOCK_SIZE_QH, BLOCK_SIZE_VD), 0, dtype=tl.float32)
        q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_KD)
        if PER_QUERY_MASK and not PRECOMPUTED_QUERY_MASK:
            exact_q = tl.arange(0, BLOCK_SIZE_Q)
            exact_t = tl.arange(0, EXACT_TOPK)
            exact_ids = tl.load(
                exact_t_ptr
                + pid_kh * stride_eth
                + (q_start + pid_q_j * BLOCK_SIZE_Q + exact_q[:, None]) * stride_etn
                + exact_t[None, :] * stride_etk,
                mask=(pid_q_j * BLOCK_SIZE_Q + exact_q[:, None] < q_len),
                other=-1,
            )
        # sparse attention
        for i in range(real_topk):
            # get current block start index (absolute K position)
            block_id = tl.load(t_ptr_j).to(tl.int32)
            c = block_id * BLOCK_SIZE_K
            t_ptr_j = t_ptr_j + stride_tk
            # paged load K via req_to_token: pos -> slot -> k_cache
            pos = c + off_n
            pos_mask = pos < seq_len
            slots = tl.load(
                req_to_token_ptr + sid * stride_r2t_b + pos,
                mask=pos_mask,
                other=0,
            ).to(tl.int64)
            slots = (slots + max_slots) % max_slots  # safety against negative
            # k shape: [BLOCK_SIZE_KD, BLOCK_SIZE_K] (transposed for tl.dot)
            k = tl.load(
                k_cache_ptr
                + slots[None, :] * stride_ks
                + pid_kh * stride_kh
                + off_kd[:, None] * stride_kd,
                mask=kd_mask[:, None] & pos_mask[None, :],
                other=0.0,
            )
            if IS_FP8:
                # fp8 main K cache: widening cast with bf16/fp16 Q (unit-scaled
                # cache -> exact inverse dequant; k_scale covers calibrated
                # caches), no-op with fp8 Q (fp8 attn-GEMM mode) so tl.dot runs
                # fp8x8. Compiled out when the cache is bf16.
                k = k.to(q.dtype)
            # compute qk
            qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
            # causal mask
            qk += tl.where(off_q_k[:, None, :] >= c, 0, float("-inf"))
            if PER_QUERY_MASK:
                if PRECOMPUTED_QUERY_MASK:
                    membership_bits = tl.load(
                        query_group_mask_ptr
                        + pid_kh * stride_mh
                        + (q_block_start + pid_q_j) * stride_mg
                        + i * stride_mu
                    )
                    member = (membership_bits & (1 << tl.arange(0, BLOCK_SIZE_Q))) != 0
                else:
                    member = tl.sum(exact_ids == block_id, axis=1) > 0
                qk += tl.where(member[:, None, None], 0, float("-inf"))
            qk = tl.reshape(qk, BLOCK_SIZE_QH, BLOCK_SIZE_K)
            # [BLOCK_SIZE_QH, qk_head_dim] @ [qk_head_dim, BLOCK_SIZE_K]
            #   -> [BLOCK_SIZE_QH, BLOCK_SIZE_K]
            qk += tl.dot(q, k) * (sm_scale_log2e * k_scale)
            # K boundary mask: positions beyond seq_len contribute -inf
            qk += tl.where(pos_mask[None, :], 0, float("-inf"))
            # Compute the online-softmax update.  A grouped-union row contains
            # blocks selected by any query in the group.  For a query that did
            # not select the current block, PER_QUERY_MASK makes the complete
            # qk row -inf.  Such a row must preserve its state bit-for-bit:
            # running it through exp2/log2 is mathematically an identity, but
            # the round trip perturbs lse_i and can change model tokens after
            # many layers.
            block_m = tl.max(qk, axis=1)
            has_current_value = block_m > float("-inf")
            m_ij = tl.maximum(m_i, block_m)
            if PER_QUERY_MASK:
                has_any_value = m_ij > float("-inf")
                safe_m_ij = tl.where(has_any_value, m_ij, 0.0)
                p = tl.where(qk > float("-inf"), tl.exp2(qk - safe_m_ij[:, None]), 0.0)
                acc_o_scale = tl.where(
                    m_i > float("-inf"), tl.exp2(m_i - safe_m_ij), 0.0
                )
            else:
                p = tl.exp2(qk - m_ij[:, None])
                acc_o_scale = tl.exp2(m_i - m_ij)
            l_ij = tl.sum(p, axis=1)
            # scale acc_o
            updated_acc_o = acc_o * acc_o_scale[:, None]
            # paged load V
            v = tl.load(
                v_cache_ptr
                + slots[:, None] * stride_vs
                + pid_kh * stride_vh
                + off_vd[None, :] * stride_vd,
                mask=pos_mask[:, None] & vd_mask[None, :],
                other=0.0,
            )
            if IS_FP8:
                # Cast V to the compute dtype: widening with bf16/fp16 Q (so
                # `p.to(v.dtype)` keeps P in the compute dtype), no-op with fp8
                # Q where P is quantized to e4m3 for the fp8 PV MMA — the same
                # accuracy contract as fmha_sm100's fp8 kernel.
                if FP8_PV:
                    v = v.to(q.dtype)
                else:
                    v = v.to(tl.bfloat16)
            p = (p * p_scale).to(v.dtype)
            updated_acc_o += tl.dot(p, v) * (v_scale / p_scale)
            # update statistics
            if PER_QUERY_MASK:
                lse_sum = (
                    tl.where(lse_i > float("-inf"), tl.exp2(lse_i - safe_m_ij), 0.0)
                    + l_ij
                )
                updated_lse_i = safe_m_ij + tl.log2(lse_sum)
                # For an inactive row acc_o_scale is exactly one and p is
                # exactly zero, so these two assignments already preserve the
                # accumulator and running maximum.  Avoid a BLOCK_SIZE_QH x D
                # select on the hot accumulator: only the log2/exp2 round trip
                # in lse_i needs an explicit bit-preserving guard.
                acc_o = updated_acc_o
                m_i = m_ij
                lse_i = tl.where(has_current_value, updated_lse_i, lse_i)
            else:
                acc_o = updated_acc_o
                m_i = m_ij
                lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
        # final scale
        has_value = lse_i > float("-inf")
        scale = tl.where(has_value, tl.exp2(m_i - lse_i), 0.0)
        acc_o = acc_o * scale[:, None]
        # save output
        acc_o = tl.reshape(acc_o, BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_VD)
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + q_start * stride_on + pid_h * stride_oh,
            shape=(q_len, gqa_group_size, v_head_dim),
            strides=(stride_on, stride_oh, stride_od),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_VD),
            order=(2, 1, 0),
        )
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))


@torch.no_grad()
def flash_prefill_with_gqa_share_sparse(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sink: Optional[torch.Tensor],
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    topk_idx: torch.Tensor,
    block_size_q: int,
    block_size_k: int,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_q: int,
    per_query_topk_idx: Optional[torch.Tensor] = None,
    query_group_mask: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    use_tma: bool = True,
    cu_seqblocks_q: Optional[torch.Tensor] = None,
    max_seqblock_q: Optional[int] = None,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
) -> torch.Tensor:
    triton.set_allocator(robust_allocator)
    is_fp8 = check_sparse_kv_fp8(q, k_cache, v_cache, label="prefill")
    k_scale = unit_scale(k_scale)
    v_scale = unit_scale(v_scale)
    fp8_pv = q.dtype == torch.float8_e4m3fn
    p_scale = (
        float(os.environ.get("SGLANG_M3_TRITON_FP8_P_SCALE", "448")) if fp8_pv else 1.0
    )
    if fp8_pv and not (0.0 < p_scale <= 448.0):
        raise ValueError(
            "SGLANG_M3_TRITON_FP8_P_SCALE must be in (0, 448] for e4m3fn, "
            f"got {p_scale}"
        )
    assert block_size_q in {1, 2, 4, 8, 16, 32, 64}
    assert block_size_k in {16, 32, 64, 128}
    # shape
    total_q, num_q_heads, qk_head_dim = q.shape
    max_slots, num_k_heads, _ = k_cache.shape
    _, num_v_heads, v_head_dim = v_cache.shape
    batch_size = cu_seqlens.shape[0] - 1
    topk = topk_idx.shape[-1]
    assert topk_idx.shape[0] == num_k_heads
    if per_query_topk_idx is not None:
        assert per_query_topk_idx.shape[0] == num_k_heads
        assert per_query_topk_idx.shape[1] == total_q
    if query_group_mask is not None:
        assert query_group_mask.shape == topk_idx.shape
    # gqa
    assert num_k_heads == num_v_heads
    assert num_q_heads % num_k_heads == 0
    gqa_group_size = num_q_heads // num_k_heads
    main_max_qh = int(os.environ.get("SGLANG_MINIMAX_PREFILL_MAIN_MAX_QH", "128"))
    if main_max_qh not in (128, 256):
        raise ValueError(
            "SGLANG_MINIMAX_PREFILL_MAIN_MAX_QH must be 128 or 256, "
            f"got {main_max_qh}"
        )
    assert gqa_group_size * block_size_q <= main_max_qh
    if sm_scale is None:
        sm_scale = qk_head_dim**-0.5
    # q_scale multiplies every Q-side logit (QK dot and sink), so it folds into
    # sm_scale; k_scale must not touch the sink term and stays a kernel arg.
    sm_scale = sm_scale * unit_scale(q_scale)
    if cu_seqblocks_q is None or max_seqblock_q is None:
        cu_seqblocks_q, max_seqblock_q, _, _, _, _ = get_cu_seqblocks(
            cu_seqlens, max_seqlen_q, block_size_q, block_size_k
        )
    # output tensor
    o = torch.empty(
        total_q, num_q_heads, v_head_dim, device=q.device, dtype=sparse_out_dtype(q)
    )
    # launch kernel
    num_q_loop = (
        max_seqblock_q // 131072 + 1
    )  # calculate multiple queries in one kernel if seqlence length is too long
    BLOCK_SIZE_Q = triton.next_power_of_2(block_size_q)
    BLOCK_SIZE_K = triton.next_power_of_2(block_size_k)
    grid = (
        triton.cdiv(triton.cdiv(max_seqlen_q, block_size_q), num_q_loop),
        num_k_heads,
        batch_size,
    )
    _gqa_share_sparse_fwd_kernel[grid](
        q,
        k_cache,
        v_cache,
        sink,
        topk_idx,
        per_query_topk_idx,
        query_group_mask,
        o,
        req_to_token,
        cu_seqlens,
        cu_seqblocks_q,
        seq_lens,
        prefix_lens,
        slot_ids,
        max_slots,
        num_k_heads,
        gqa_group_size,
        qk_head_dim,
        v_head_dim,
        topk,
        triton.next_power_of_2(max_seqlen_q),
        num_q_loop,
        sm_scale,
        k_scale,
        v_scale,
        p_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        sink.stride(0) if sink is not None else 0,
        sink.stride(1) if sink is not None else 0,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        per_query_topk_idx.stride(0) if per_query_topk_idx is not None else 0,
        per_query_topk_idx.stride(1) if per_query_topk_idx is not None else 0,
        per_query_topk_idx.stride(2) if per_query_topk_idx is not None else 0,
        query_group_mask.stride(0) if query_group_mask is not None else 0,
        query_group_mask.stride(1) if query_group_mask is not None else 0,
        query_group_mask.stride(2) if query_group_mask is not None else 0,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        req_to_token.stride(0),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        USE_TMA=use_tma,
        IS_FP8=is_fp8,
        FP8_PV=fp8_pv,
        PER_QUERY_MASK=(per_query_topk_idx is not None or query_group_mask is not None),
        PRECOMPUTED_QUERY_MASK=query_group_mask is not None,
        EXACT_TOPK=(
            per_query_topk_idx.shape[-1] if per_query_topk_idx is not None else 1
        ),
    )
    return o
