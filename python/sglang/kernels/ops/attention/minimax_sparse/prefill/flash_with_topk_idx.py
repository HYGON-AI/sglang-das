# Copyright 2025 XunhaoLai. All rights reserved.

import os
from typing import Optional

import torch
import triton
import triton.language as tl

from ..common.utils import (
    _bitonic_merge,
    _sort_ids_ascending,
    check_sparse_kv_fp8,
    get_cu_seqblocks,
    robust_allocator,
    sparse_out_dtype,
    unit_scale,
)


def _prune_prefill_score_configs(configs, named_args, **kwargs):
    """Pin the long-context exact-Q1 score tile when explicitly requested.

    Triton's isolated autotune does not model the two concurrent CP zigzag
    streams.  The override is therefore applied only to exact-Q1 index scoring
    at or above a configurable K bucket; short/ragged and grouped-score shapes
    keep the regular autotune search.
    """

    raw = os.environ.get("SGLANG_MINIMAX_PREFILL_SCORE_CONFIG", "").strip()
    if not raw:
        return configs
    try:
        requested = tuple(int(part.strip()) for part in raw.split(","))
    except ValueError as exc:
        raise ValueError(
            "SGLANG_MINIMAX_PREFILL_SCORE_CONFIG must be "
            "BLOCK_Q,BLOCK_K,NUM_WARPS,NUM_STAGES"
        ) from exc
    if len(requested) != 4:
        raise ValueError(
            "SGLANG_MINIMAX_PREFILL_SCORE_CONFIG must contain exactly four "
            "comma-separated integers"
        )
    selected = [
        config
        for config in configs
        if (
            config.kwargs.get("BLOCK_SIZE_Q"),
            config.kwargs.get("BLOCK_SIZE_K"),
            config.num_warps,
            config.num_stages,
        )
        == requested
    ]
    if not selected:
        raise ValueError(
            "SGLANG_MINIMAX_PREFILL_SCORE_CONFIG does not match an available "
            f"candidate: {raw!r}"
        )
    min_k_bucket = int(
        os.environ.get("SGLANG_MINIMAX_PREFILL_SCORE_CONFIG_MIN_K", "131072")
    )
    if min_k_bucket < 0:
        raise ValueError(
            "SGLANG_MINIMAX_PREFILL_SCORE_CONFIG_MIN_K must be non-negative"
        )
    if (
        int(named_args.get("Q_SAMPLE_INTERVAL", 1)) != 1
        or int(named_args.get("autotune_k_bucket", 0)) < min_k_bucket
    ):
        return configs
    return selected


@triton.heuristics(
    {
        "BLOCK_SIZE_KD": lambda args: triton.next_power_of_2(args["qk_head_dim"]),
        "BLOCK_SIZE_VD": lambda args: triton.next_power_of_2(args["v_head_dim"]),
        "HAS_SINK": lambda args: args["sink_ptr"] is not None,
    }
)
@triton.autotune(
    configs=[
        # Small block (64x64): low shared mem, can use higher num_stages
        triton.Config(
            {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=4
        ),
        # gfx938 has 64-lane waves and benefits from wider wave coverage on
        # the score-only M3 indexer.  These fill gaps in the upstream search
        # grid; autotune keeps them only when the real long-context shape wins.
        triton.Config(
            {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 64}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 64}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 64}, num_warps=8, num_stages=3
        ),
        # Medium block (64x128, 128x64): moderate shared mem, ns=2,3
        triton.Config(
            {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=3
        ),
        # gfx938 score-only indexer candidates.  Q32 can improve occupancy for
        # short query shards; the autotune key below includes sequence lengths
        # so a 192-token server warmup cannot poison the 128K+16K deployment
        # choice.
        triton.Config(
            {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 32, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 32, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 32, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 32, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 32, "BLOCK_SIZE_K": 64}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 128, "BLOCK_SIZE_K": 64}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 128, "BLOCK_SIZE_K": 64}, num_warps=8, num_stages=3
        ),
        # Large block (128x128): high shared mem, ns=2,3 only, nw=8
        triton.Config(
            {"BLOCK_SIZE_Q": 128, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 128, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_Q": 128, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2
        ),
    ],
    key=[
        "qk_head_dim",
        "v_head_dim",
        "block_size",
        "autotune_q_bucket",
        "autotune_k_bucket",
        "use_gumbel_topk",
        "SCORE_TYPE",
        "DISABLE_INDEX_VALUE",
        "COMPACT_SCORE",
        "Q_SAMPLE_INTERVAL",
        "PREPARE_EXTERNAL_TOPK",
    ],
    prune_configs_by={"early_config_prune": _prune_prefill_score_configs},
)
@triton.jit
def _flash_attn_fwd_with_block_score_kernel(
    q_ptr,  # Q: n x h x d
    k_cache_ptr,  # K paged: max_slots x kh x d
    v_cache_ptr,  # V paged: max_slots x kh x d
    sink_ptr,  # Sink: h x d
    o_ptr,  # O: n x h x d
    score_ptr,  # Score: h x n x max_seqblock
    req_to_token_ptr,  # req_to_token: max_reqs x max_kv_len
    # seqlens
    cu_seqlens,
    cu_seqblocks_q,
    seq_lens,
    prefix_lens,
    slot_ids,
    # shape
    max_slots,
    num_heads,
    gqa_group_size,
    qk_head_dim,
    v_head_dim,
    autotune_q_bucket,
    autotune_k_bucket,
    block_size: tl.constexpr,
    # sm_scale
    sm_scale,
    # per-tensor KV dequant scales (1.0 when the cache is unit-scaled)
    k_scale,
    v_scale,
    # gumbel topk
    use_gumbel_topk: tl.constexpr,
    gumbel_seed,
    # stride
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_v_s,
    stride_v_h,
    stride_v_d,
    stride_sink_h,
    stride_sink_d,
    stride_o_n,
    stride_o_h,
    stride_o_d,
    stride_s_h,
    stride_s_q,
    stride_s_k,
    stride_r2t_b,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_KD: tl.constexpr,
    BLOCK_SIZE_VD: tl.constexpr,
    # has sink
    HAS_SINK: tl.constexpr,
    SCORE_TYPE: tl.constexpr,
    DISABLE_INDEX_VALUE: tl.constexpr,
    IS_FP8: tl.constexpr,
    COMPACT_SCORE: tl.constexpr,
    Q_SAMPLE_INTERVAL: tl.constexpr,
    Q_SAMPLE_OFFSET: tl.constexpr,
    PREPARE_EXTERNAL_TOPK: tl.constexpr,
    FINALIZE_EXTERNAL_TOPK: tl.constexpr,
    INIT_BLOCKS: tl.constexpr,
    LOCAL_BLOCKS: tl.constexpr,
    MAX_SEQBLOCK_K: tl.constexpr,
):
    tl.static_assert(SCORE_TYPE == "max" or SCORE_TYPE == "lse")
    sm_scale_log2e = sm_scale * 1.4426950409
    tl.static_assert(BLOCK_SIZE_K >= block_size)
    BLOCKS_PER_K_BLOCK: tl.constexpr = BLOCK_SIZE_K // block_size
    # get batch id and head id
    pid_q, pid_bh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads
    pid_kh = pid_h // gqa_group_size
    # get q k start and len after rmpad
    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    block_start = tl.load(cu_seqblocks_q + pid_b)
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    sid = (
        tl.load(slot_ids + pid_b).to(tl.int64) + max_slots
    ) % max_slots  # safety against negative
    if BLOCK_SIZE_Q * pid_q * Q_SAMPLE_INTERVAL + Q_SAMPLE_OFFSET >= q_len:
        return
    block_num = (seq_len + block_size - 1) // block_size
    # init qkv pointer
    q_rows = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q
    q_pos = q_rows * Q_SAMPLE_INTERVAL + Q_SAMPLE_OFFSET
    if COMPACT_SCORE:
        off_qd = tl.arange(0, BLOCK_SIZE_KD)
        q = tl.load(
            q_ptr
            + (seq_start + q_pos[:, None]) * stride_q_n
            + pid_h * stride_q_h
            + off_qd[None, :] * stride_q_d,
            mask=(q_pos[:, None] < q_len) & (off_qd[None, :] < qk_head_dim),
            other=0.0,
        )
    else:
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + seq_start * stride_q_n + pid_h * stride_q_h,
            shape=(q_len, qk_head_dim),
            strides=(stride_q_n, stride_q_d),
            offsets=(pid_q * BLOCK_SIZE_Q, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_KD),
            order=(1, 0),
        )
        s_ptrs = tl.make_block_ptr(
            base=score_ptr + seq_start * stride_s_q + pid_h * stride_s_h,
            shape=(q_len, block_num),
            strides=(stride_s_q, stride_s_k),
            offsets=(pid_q * BLOCK_SIZE_Q, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCKS_PER_K_BLOCK),
            order=(1, 0),
        )
        # load q
        q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    if HAS_SINK:
        off_d = tl.arange(0, BLOCK_SIZE_KD)
        sink = tl.load(
            sink_ptr + pid_h * stride_sink_h + off_d * stride_sink_d,
            mask=off_d < qk_head_dim,
            other=0,
        )
    # init statistics
    off_q = q_pos + prefix_len
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_kd = tl.arange(0, BLOCK_SIZE_KD)
    off_vd = tl.arange(0, BLOCK_SIZE_VD)
    off_bpk = tl.arange(0, BLOCKS_PER_K_BLOCK)
    kd_mask = off_kd < qk_head_dim
    vd_mask = off_vd < v_head_dim
    if HAS_SINK:
        m_i = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)
        lse_i = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)
        qsink = (
            tl.sum(q.to(tl.float32) * sink[None, :].to(tl.float32), axis=1)
            * sm_scale_log2e
        )  # (BLOCK_SIZE_Q,)
        m_i += qsink
        lse_i += qsink
    else:
        m_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
        lse_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    acc_o = tl.full((BLOCK_SIZE_Q, BLOCK_SIZE_VD), 0, dtype=tl.float32)
    # attention
    diag_start = (
        (prefix_len + pid_q * BLOCK_SIZE_Q * Q_SAMPLE_INTERVAL + Q_SAMPLE_OFFSET)
        // BLOCK_SIZE_K
        * BLOCK_SIZE_K
    )
    hi = min(
        seq_len,
        prefix_len + (pid_q + 1) * BLOCK_SIZE_Q * Q_SAMPLE_INTERVAL + Q_SAMPLE_OFFSET,
    )
    for i in tl.range(0, hi, BLOCK_SIZE_K):
        # paged load K via req_to_token: pos -> slot -> k_cache
        pos = i + off_k
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
            + slots[None, :] * stride_k_s
            + pid_kh * stride_k_h
            + off_kd[:, None] * stride_k_d,
            mask=kd_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )
        if IS_FP8:
            # fp8 index K cache: widening cast with bf16/fp16 Q, no-op with fp8
            # Q (fp8 attn-GEMM mode; tl.dot runs fp8x8). Compiled out for bf16.
            k = k.to(q.dtype)
        # compute qk
        qk = tl.dot(q, k) * (sm_scale_log2e * k_scale)
        if i >= diag_start:
            qk = tl.where(off_q[:, None] >= (i + off_k)[None, :], qk, float("-inf"))
        # K boundary mask: positions beyond seq_len contribute -inf
        qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        # save score
        score = tl.reshape(
            qk, (BLOCK_SIZE_Q, BLOCKS_PER_K_BLOCK, block_size), can_reorder=False
        )
        sub_max = tl.max(score, axis=2)
        if SCORE_TYPE == "max":
            score = sub_max
        else:  # "lse"
            # fully-masked sub-blocks produce NaN via -inf - (-inf); clamp
            # back to -inf so downstream bitonic sort sees a clean sentinel.
            score = sub_max + tl.log2(
                tl.sum(tl.exp2(score - sub_max[:, :, None]), axis=2)
            )
            score = tl.where(score != score, float("-inf"), score)
        if use_gumbel_topk:
            # generate non-conflicting offset for random generation
            # noise_offset shape: (BLOCK_SIZE_Q, BLOCKS_PER_K_BLOCK)
            # random seed include head id, batch id and gumbel seed
            # (Head low 7 bits | Batch middle 12 bits | Other high bits)
            local_seed = (pid_h | (pid_b << 7) | (gumbel_seed << 19)).to(tl.int32)
            # noise offset include q index and k block index
            # [31-13: Q (19bits)] | [12-0: K_Block (13bits)]
            noise_offset = (off_q << 13)[:, None] | (off_bpk + i // block_size)[None, :]
            # gumbel noise (scaled to log2 scale to match sm_scale_log2e)
            noise = tl.rand(local_seed, offset=noise_offset)
            noise = tl.clamp(noise, min=1e-9, max=1 - 1e-9)  # avoid log(0)
            noise = -tl.log(-tl.log(noise)) * 1.4426950409
            score += noise
        if PREPARE_EXTERNAL_TOPK:
            # AITER Top-K does not receive the ragged/forced-block metadata
            # understood by the native selector. Materialize that contract
            # while the score tile is still in registers instead of launching
            # a second kernel that reads and rewrites the whole FP32 workspace.
            score_block = i // block_size + off_bpk[None, :]
            valid_blocks = (prefix_len + q_pos[:, None] + block_size) // block_size
            valid_score = score_block < valid_blocks
            score = tl.where(valid_score, score, -1e30)
            score = tl.where(valid_score & (score_block < INIT_BLOCKS), 1e30, score)
            score = tl.where(
                valid_score
                & (score_block >= tl.maximum(0, valid_blocks - LOCAL_BLOCKS)),
                1e29,
                score,
            )
        if COMPACT_SCORE:
            score_ptrs = (
                score_ptr
                + pid_h * stride_s_h
                + (block_start + q_rows[:, None]) * stride_s_q
                + (i // block_size + off_bpk[None, :]) * stride_s_k
            )
            tl.store(
                score_ptrs,
                score.to(score_ptr.dtype.element_ty),
                mask=(q_pos[:, None] < q_len)
                & ((i // block_size + off_bpk[None, :]) < block_num),
            )
        else:
            tl.store(
                s_ptrs,
                score.to(score_ptr.dtype.element_ty),
                boundary_check=(0, 1),
            )
        if not DISABLE_INDEX_VALUE:
            # compute m_ij and l_ij
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            # scale acc_o
            acc_o_scale = tl.exp2(m_i - m_ij)
            acc_o = acc_o * acc_o_scale[:, None]
            # paged load V
            v = tl.load(
                v_cache_ptr
                + slots[:, None] * stride_v_s
                + pid_kh * stride_v_h
                + off_vd[None, :] * stride_v_d,
                mask=pos_mask[:, None] & vd_mask[None, :],
                other=0.0,
            )
            if IS_FP8:
                # Cast V to the compute dtype (widening for bf16/fp16 Q; no-op
                # for fp8 Q where P is quantized to e4m3 for the fp8 PV MMA).
                v = v.to(q.dtype)
            p = p.to(v.dtype)
            acc_o += tl.dot(p, v) * v_scale
            # update statistics
            m_i = m_ij
            lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
        # update ptrs
        if not COMPACT_SCORE:
            s_ptrs = tl.advance(s_ptrs, (0, BLOCKS_PER_K_BLOCK))
    if PREPARE_EXTERNAL_TOPK and FINALIZE_EXTERNAL_TOPK:
        # The producer loop stops at the largest causal K tile. AITER scans the
        # complete fixed-width row, so initialize only the unwritten tail. This
        # is write-only and stays in the producer launch; the former prepare
        # kernel loaded these undefined bytes before replacing them with -inf.
        for tail_block in tl.range(0, MAX_SEQBLOCK_K, BLOCKS_PER_K_BLOCK):
            tail_ids = tail_block + off_bpk[None, :]
            valid_blocks = (prefix_len + q_pos[:, None] + block_size) // block_size
            tail_mask = (
                (q_pos[:, None] < q_len)
                & (tail_ids >= valid_blocks)
                & (tail_ids < MAX_SEQBLOCK_K)
            )
            if COMPACT_SCORE:
                tail_ptrs = (
                    score_ptr
                    + pid_h * stride_s_h
                    + (block_start + q_rows[:, None]) * stride_s_q
                    + tail_ids * stride_s_k
                )
            else:
                tail_ptrs = (
                    score_ptr
                    + pid_h * stride_s_h
                    + (seq_start + q_rows[:, None]) * stride_s_q
                    + tail_ids * stride_s_k
                )
            tl.store(tail_ptrs, -1e30, mask=tail_mask)
    if not DISABLE_INDEX_VALUE:
        # final scale
        acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]
        # save output
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + seq_start * stride_o_n + pid_h * stride_o_h,
            shape=(q_len, v_head_dim),
            strides=(stride_o_n, stride_o_d),
            offsets=(pid_q * BLOCK_SIZE_Q, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_VD),
            order=(1, 0),
        )
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.autotune(
    configs=[
        # Large configs for H200/B200 to support larger topk
        triton.Config({"BLOCK_SIZE_K": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE_K": 64}, num_warps=2, num_stages=2),
    ],
    key=[
        "BLOCK_SIZE_T"
    ],  # use BLOCK_SIZE_T instead of topk to reduce autotune frequency
)
@triton.jit
def _topk_index_kernel(
    s_ptr,  # Score: h x n x max_seqblock
    ti_ptr,  # topk_idx: h x n x topk
    # size
    sample_interval: tl.constexpr,
    sample_offset: tl.constexpr,
    block_size: tl.constexpr,
    # seqlens
    cu_seqlens,
    cu_seqblocks_q,
    prefix_lens,
    # shape
    topk,  # not constexpr to avoid recompilation when topk changes
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    # stride
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_ti_h,
    stride_ti_n,
    stride_ti_t,
    # META parameters
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    MASK_INIT: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
    SCORES_COMPACT: tl.constexpr,
):
    tl.static_assert(
        BLOCK_SIZE_K > BLOCK_SIZE_T
    )  # use BLOCK_SIZE_T instead of topk (stricter but safe)
    # get batch id and head id
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    # get q k start and len after rmpad
    seq_start = tl.load(cu_seqlens + pid_b)
    block_start = tl.load(cu_seqblocks_q + pid_b)
    block_num = tl.load(cu_seqblocks_q + pid_b + 1) - block_start
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q >= block_num:
        return
    # offsets
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    # init qkv pointer
    if SCORES_COMPACT:
        score_row = block_start + pid_q
    else:
        score_row = seq_start + pid_q * sample_interval
    s_ptrs = s_ptr + score_row * stride_s_n + pid_h * stride_s_h + off_k * stride_s_k
    # init statistics
    topk_score = tl.full((BLOCK_SIZE_K,), -1e30, dtype=tl.float32)
    topk_idx = tl.full((BLOCK_SIZE_K,), 0, dtype=tl.int32)
    left_half_mask = tl.arange(0, BLOCK_SIZE_K) < BLOCK_SIZE_K // 2
    # compute topk
    valid_blocks = (
        prefix_len + pid_q * sample_interval + sample_offset + block_size
    ) // block_size
    for i in tl.range(0, valid_blocks, BLOCK_SIZE_K):
        # masks
        causal_mask = i + off_k < valid_blocks
        local_mask = i + off_k >= max(0, valid_blocks - local_blocks)
        init_mask = i + off_k < init_blocks
        # load score
        score = tl.load(s_ptrs, mask=causal_mask, other=-1e30).to(tl.float32)
        # handle NaN: NaN inputs cause bitonic sort to fail, resulting in invalid indices (-2)
        # appearing in the topk list. We replace NaN with -inf to maintain sort order.
        score = tl.where(score != score, -1e30, score)
        s_ptrs = s_ptrs + stride_s_k * BLOCK_SIZE_K
        # fill init and local part, make sure init part is always in topk
        # and at the first position. Note: must use causal_mask to protect
        # init_mask to avoid selecting blocks outside causal window
        if MASK_INIT:
            score = tl.where(causal_mask & init_mask, score - 1e29, score)
        else:
            score = tl.where(causal_mask & init_mask, 1e30, score)
        if MASK_LOCAL:
            score = tl.where(causal_mask & local_mask, score - 1e28, score)
        else:
            score = tl.where(causal_mask & local_mask, 1e29, score)
        # bitonic merge
        topk_score, last_topk_score = score, topk_score
        topk_idx, last_topk_idx = (tl.where(causal_mask, i + off_k + 1, 0), topk_idx)
        n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
        for j in tl.static_range(1, n_dims):
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), j, 2, n_dims
            )
        if i != 0:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, False, n_dims
            )
            topk_score_new = last_topk_score * left_half_mask + topk_score * (
                1 - left_half_mask
            )
            topk_idx_new = last_topk_idx * left_half_mask + topk_idx * (
                1 - left_half_mask
            )
            topk_score, topk_idx = _bitonic_merge(
                topk_score_new, topk_idx_new.to(tl.int32), n_dims, True, n_dims
            )
        else:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, True, n_dims
            )
    # get topk, shape: [BLOCK_SIZE_T,]
    topk_mask = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    topk_idx = tl.sum(
        topk_mask[:, None]
        * tl.reshape(topk_idx - 1, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )
    # Ascending by block id, -1 tail: the MSA fmha_sm100 consumer requires
    # sorted kv_block_indexes (the bitonic pass above orders by score).
    topk_idx = _sort_ids_ascending(topk_idx, min(topk, valid_blocks), BLOCK_SIZE_T)
    # save topk
    ti_ptrs = (
        ti_ptr
        + (block_start + pid_q) * stride_ti_n
        + pid_h * stride_ti_h
        + off_t * stride_ti_t
    )
    topk_mask = tl.arange(0, BLOCK_SIZE_T) < min(topk, valid_blocks)
    tl.store(ti_ptrs, topk_idx.to(ti_ptrs.dtype.element_ty), mask=topk_mask)


@triton.jit
def _prepare_score_for_external_topk_kernel(
    s_ptr,
    cu_seqlens,
    cu_seqblocks_q,
    prefix_lens,
    sample_interval: tl.constexpr,
    sample_offset: tl.constexpr,
    block_size: tl.constexpr,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    max_seqblock_k: tl.constexpr,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    NUM_HEADS: tl.constexpr,
    SCORES_COMPACT: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Apply the exact indexer masks before a vendor Top-K implementation.

    The score producer intentionally leaves the non-causal tail uninitialized.
    The native Triton Top-K masks that tail while loading it and also forces the
    model's init/local blocks.  External Top-K kernels cannot see the ragged
    sequence metadata, so materialize those same rules in-place first.
    """

    pid_k = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_bh = tl.program_id(2)

    # The launcher flattens batch and head into the third grid dimension.
    pid_b = pid_bh // NUM_HEADS
    pid_h = pid_bh % NUM_HEADS
    seq_start = tl.load(cu_seqlens + pid_b)
    block_start = tl.load(cu_seqblocks_q + pid_b)
    block_num = tl.load(cu_seqblocks_q + pid_b + 1) - block_start
    prefix_len = tl.load(prefix_lens + pid_b)

    off_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    valid_row = pid_q < block_num
    valid_blocks = (
        prefix_len + pid_q * sample_interval + sample_offset + block_size
    ) // block_size
    valid = valid_row & (off_k < valid_blocks) & (off_k < max_seqblock_k)
    if SCORES_COMPACT:
        score_row = block_start + pid_q
    else:
        score_row = seq_start + pid_q * sample_interval
    ptrs = s_ptr + pid_h * stride_s_h + score_row * stride_s_n + off_k * stride_s_k
    score = tl.load(ptrs, mask=valid, other=-1e30).to(tl.float32)
    score = tl.where(score != score, -1e30, score)
    score = tl.where(valid & (off_k < init_blocks), 1e30, score)
    score = tl.where(
        valid & (off_k >= tl.maximum(0, valid_blocks - local_blocks)),
        1e29,
        score,
    )
    tl.store(ptrs, score, mask=valid_row & (off_k < max_seqblock_k))


@triton.jit
def _sort_external_topk_ids_kernel(
    ti_ptr,
    total_rows,
    topk: tl.constexpr,
    stride_row,
    stride_t,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid = tl.program_id(0)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    ptrs = ti_ptr + pid * stride_row + off_t * stride_t
    ids = tl.load(ptrs, mask=(pid < total_rows) & (off_t < topk), other=-1)
    ids = _sort_ids_ascending(ids, topk, BLOCK_SIZE_T)
    tl.store(ptrs, ids, mask=(pid < total_rows) & (off_t < topk))


@torch.no_grad()
def flash_prefill_with_topk_index(
    q: torch.Tensor,
    k_cache: torch.Tensor,  # paged
    v_cache: Optional[torch.Tensor],  # paged; ignored when disable_index_value=True
    sink: Optional[torch.Tensor],
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size_q: int,
    block_size_k: int,
    topk: int,
    init_blocks: int = 1,
    local_blocks: int = 2,
    sm_scale: Optional[float] = None,
    use_tma: bool = False,
    score_type: str = "max",
    disable_index_value: bool = False,
    cu_seqblocks_q: Optional[torch.Tensor] = None,
    max_seqblock_q: Optional[int] = None,
    all_seqblock_q: Optional[int] = None,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    sort_topk_ids: bool = True,
):
    assert score_type in (
        "max",
        "lse",
    ), f"score_type must be 'max' or 'lse', got {score_type!r}"
    triton.set_allocator(robust_allocator)
    # dtype check (v_cache is None under disable_index_value)
    is_fp8 = check_sparse_kv_fp8(
        q, k_cache, None if disable_index_value else v_cache, label="prefill indexer"
    )
    k_scale = unit_scale(k_scale)
    v_scale = unit_scale(v_scale)
    assert cu_seqlens.dtype == torch.int32
    # shape
    total_q, num_heads, qk_head_dim = q.shape
    max_slots, num_kv_heads, _ = k_cache.shape
    if disable_index_value:
        # placeholder for BLOCK_SIZE_VD; V is never loaded
        v_head_dim = qk_head_dim
    else:
        assert v_cache is not None
        assert v_cache.shape[1] == k_cache.shape[1]
        v_head_dim = v_cache.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    batch_size = cu_seqlens.shape[0] - 1
    assert qk_head_dim <= 256 and v_head_dim <= 256, "head_dim must be less than 256"
    if sink is not None:
        assert sink.shape[0] == num_heads and sink.shape[1] == qk_head_dim
    assert (
        init_blocks + local_blocks <= topk
    ), "init_blocks + local_blocks must be less than topk"
    if sm_scale is None:
        sm_scale = qk_head_dim**-0.5
    # q_scale multiplies every Q-side logit (QK dot and sink), so it folds into
    # sm_scale; k_scale must not touch the sink term and stays a kernel arg.
    sm_scale = sm_scale * unit_scale(q_scale)
    if cu_seqblocks_q is None or max_seqblock_q is None or all_seqblock_q is None:
        cu_seqblocks_q, max_seqblock_q, all_seqblock_q, _, _, _ = get_cu_seqblocks(
            cu_seqlens, max_seqlen_q, block_size_q, block_size_k
        )
    max_seqblock_k = triton.cdiv(max_seqlen_k, block_size_k)
    if disable_index_value:
        o = None
    else:
        o = torch.empty(
            total_q, num_heads, v_head_dim, dtype=sparse_out_dtype(q), device=q.device
        )
    compact_score = (
        disable_index_value
        and block_size_q > 1
        and os.environ.get("SGLANG_MINIMAX_COMPACT_BLOCK_SCORE", "1") == "1"
    )
    sample_offset = 0
    coarse_topk = topk
    score_rows = all_seqblock_q if compact_score else total_q
    score_shape = (num_heads, score_rows, max_seqblock_k)
    topk_backend = os.environ.get(
        "SGLANG_MINIMAX_PREFILL_TOPK_BACKEND", "triton"
    ).lower()
    if topk_backend not in ("triton", "aiter"):
        raise ValueError(
            "SGLANG_MINIMAX_PREFILL_TOPK_BACKEND must be 'triton' or 'aiter', "
            f"got {topk_backend!r}"
        )
    fuse_aiter_prepare = (
        topk_backend == "aiter"
        and os.environ.get("SGLANG_MINIMAX_FUSE_AITER_TOPK_PREPARE", "1") == "1"
    )
    # The producer writes every score that _topk_index_kernel is allowed to
    # read: its per-row ``valid_blocks`` is no larger than the producer's
    # causal ``hi``.  Initializing the entire temporary to -inf therefore adds
    # a pure device-memory pass (about 18 MiB for M3's 16K/128K-hit Q16 case).
    # Keep a fallback switch for backend bring-up and differential tests.
    if os.environ.get("SGLANG_MINIMAX_EMPTY_BLOCK_SCORE", "1") == "1":
        score = torch.empty(score_shape, dtype=torch.float32, device=q.device)
    else:
        score = torch.full(
            score_shape,
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )

    # Launch the score producer. A gfx938 AOT HIP/MFMA experiment was removed
    # after the real winner shape measured 3x slower than this Triton path.
    def grid(META):
        q_extent = max_seqblock_q if compact_score else max_seqlen_q
        return (
            triton.cdiv(q_extent, META["BLOCK_SIZE_Q"]),
            batch_size * num_heads,
        )

    _flash_attn_fwd_with_block_score_kernel[grid](
        q,
        k_cache,
        v_cache,
        sink,
        o,
        score,
        req_to_token,
        cu_seqlens,
        cu_seqblocks_q,
        seq_lens,
        prefix_lens,
        slot_ids,
        max_slots,
        num_heads,
        gqa_group_size,
        qk_head_dim,
        v_head_dim,
        triton.next_power_of_2(max_seqlen_q),
        triton.next_power_of_2(max_seqlen_k),
        block_size_k,
        sm_scale,
        k_scale,
        v_scale,
        False,
        1,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0) if v_cache is not None else 0,
        v_cache.stride(1) if v_cache is not None else 0,
        v_cache.stride(2) if v_cache is not None else 0,
        sink.stride(0) if sink is not None else 0,
        sink.stride(1) if sink is not None else 0,
        o.stride(0) if o is not None else 0,
        o.stride(1) if o is not None else 0,
        o.stride(2) if o is not None else 0,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        req_to_token.stride(0),
        SCORE_TYPE=score_type,
        DISABLE_INDEX_VALUE=disable_index_value,
        IS_FP8=is_fp8,
        COMPACT_SCORE=compact_score,
        Q_SAMPLE_INTERVAL=block_size_q if compact_score else 1,
        Q_SAMPLE_OFFSET=sample_offset,
        PREPARE_EXTERNAL_TOPK=fuse_aiter_prepare,
        FINALIZE_EXTERNAL_TOPK=True,
        INIT_BLOCKS=init_blocks,
        LOCAL_BLOCKS=local_blocks,
        MAX_SEQBLOCK_K=max_seqblock_k,
    )

    # topk extraction kernel
    topk_idx = torch.empty(
        (num_heads, all_seqblock_q, coarse_topk),
        device=score.device,
        dtype=torch.int32,
    )
    if topk_backend == "aiter":
        # AITER's native HIP Top-K is substantially faster than the generic
        # bitonic selector on gfx938. The producer above has already applied
        # the exact ragged/forced-block rules and initialized its causal tail.
        if not fuse_aiter_prepare:
            # Compatibility fallback retaining the standalone prepare pass
            # for AITER builds that cannot consume the fused producer output.
            prepare_block_size = 256
            prepare_grid = (
                triton.cdiv(max_seqblock_k, prepare_block_size),
                max_seqblock_q,
                batch_size * num_heads,
            )
            _prepare_score_for_external_topk_kernel[prepare_grid](
                score,
                cu_seqlens,
                cu_seqblocks_q,
                prefix_lens,
                block_size_q,
                sample_offset,
                block_size_k,
                init_blocks,
                local_blocks,
                max_seqblock_k,
                score.stride(0),
                score.stride(1),
                score.stride(2),
                NUM_HEADS=num_heads,
                SCORES_COMPACT=compact_score,
                BLOCK_SIZE_K=prepare_block_size,
                num_warps=4,
                num_stages=1,
            )
        try:
            from aiter.ops.topk_plain import topk_plain as aiter_topk_plain
        except ImportError as exc:
            raise RuntimeError(
                "AITER Top-K was requested but aiter.ops.topk_plain is unavailable"
            ) from exc
        flat_score = score.reshape(-1, max_seqblock_k)
        flat_idx = topk_idx.reshape(-1, coarse_topk)
        topk_values = torch.empty_like(flat_idx, dtype=score.dtype)
        empty_row_offsets = torch.empty(0, device=score.device, dtype=torch.int32)
        aiter_topk_plain(
            flat_score,
            flat_idx,
            topk_values,
            coarse_topk,
            True,
            empty_row_offsets,
            empty_row_offsets,
            -1,
            1,
        )
        if sort_topk_ids:
            sort_block_size = triton.next_power_of_2(coarse_topk)
            _sort_external_topk_ids_kernel[(flat_idx.shape[0],)](
                flat_idx,
                flat_idx.shape[0],
                coarse_topk,
                flat_idx.stride(0),
                flat_idx.stride(1),
                BLOCK_SIZE_T=sort_block_size,
                num_warps=1,
                num_stages=1,
            )
    else:
        grid = (max_seqblock_q, batch_size, num_heads)
        _topk_index_kernel[grid](
            score,
            topk_idx,
            block_size_q,
            sample_offset,
            block_size_k,
            cu_seqlens,
            cu_seqblocks_q,
            prefix_lens,
            coarse_topk,
            init_blocks,
            local_blocks,
            score.stride(0),
            score.stride(1),
            score.stride(2),
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_idx.stride(2),
            MASK_INIT=False,
            MASK_LOCAL=False,
            SCORES_COMPACT=compact_score,
        )
    return o, topk_idx
