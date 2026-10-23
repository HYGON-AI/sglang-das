"""Low-latency WeLM v4 kernels for small-batch decode / speculative verify.

Every kernel here is a drop-in replacement for a sequence of ops in
``welmv4.py`` / ``welmv4_op.py`` and keeps the reference numerics (fp32
accumulation of exact bf16 products, same tie-breaking in the routing top-k).
"""

import os

import torch
import triton
import triton.language as tl

# Router GEMV + top-k: used up to this many tokens; above it the original
# persistent-matmul router is faster.
WELM_FAST_ROUTER_MAX_M = int(os.environ.get("WELM_FAST_ROUTER_MAX_M", str(1 << 30)))


# ---------------------------------------------------------------------------
# Router: logits = x @ W^T (bf16 x bf16 -> fp32), split over K so that a
# handful of tokens still spreads the 2 MiB weight read over every CU. The
# partial sums are reduced in a fixed order by the top-k kernel (deterministic).
# ---------------------------------------------------------------------------
@triton.jit
def _welm_router_splitk_kernel(
    x_ptr,
    w_ptr,
    part_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xm,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    K_PER_SPLIT: tl.constexpr = K // SPLIT_K
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_begin = pid_k * K_PER_SPLIT
    for kk in tl.static_range(0, K_PER_SPLIT, BLOCK_K):
        k_idx = k_begin + kk + offs_k
        a = tl.load(
            x_ptr + offs_m[:, None].to(tl.int64) * stride_xm + k_idx[None, :],
            mask=m_mask[:, None],
            other=0.0,
        )
        b = tl.load(w_ptr + offs_n[None, :] * K + k_idx[:, None])
        acc = tl.dot(a, b, acc)
    out = part_ptr + (pid_k * M + offs_m[:, None]).to(tl.int64) * N + offs_n[None, :]
    tl.store(out, acc, mask=m_mask[:, None])


@triton.jit
def _welm_router_topk_kernel(
    part_ptr,
    bias_ptr,
    scores_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    M,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    TOPK: tl.constexpr,
    WRITE_SCORES: tl.constexpr,
    SHARED_ID: tl.constexpr = -1,
):
    row = tl.program_id(0)
    OUT_K: tl.constexpr = TOPK + (1 if SHARED_ID >= 0 else 0)
    if SHARED_ID >= 0:
        tl.store(topk_weights_ptr + row * OUT_K + TOPK, 1.0)
        tl.store(
            topk_ids_ptr + row * OUT_K + TOPK,
            tl.full((), SHARED_ID, tl.int32).to(topk_ids_ptr.dtype.element_ty),
        )
    offs = tl.arange(0, N)
    logits = tl.zeros((N,), dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        logits += tl.load(part_ptr + (s * M + row).to(tl.int64) * N + offs)
    scores = 1.0 / (1.0 + tl.exp(-logits))
    if WRITE_SCORES:
        tl.store(scores_ptr + row.to(tl.int64) * N + offs, scores)
    scores = tl.where(scores == scores, scores, -float("inf"))
    routing = scores + tl.load(bias_ptr + offs)
    routing = routing + 0.0  # -0.0 -> +0.0 so the integer key orders like floats

    # Same tie-breaking as mmq_style_expert_bias_topk: among equal routing
    # scores pick the smallest "tie_rank" (MMQ's warp-lane layout).
    elems_per_copy: tl.constexpr = 4
    copy_stride: tl.constexpr = 32 * elems_per_copy
    lane_idx = (offs % copy_stride) // elems_per_copy
    local_idx = (offs // copy_stride) * elems_per_copy + (offs % elems_per_copy)
    tie_rank = lane_idx * (N // 32) + local_idx

    bits = routing.to(tl.int32, bitcast=True)
    ordered = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF).to(tl.int64)
    low = ((N - 1 - tie_rank) * N + offs).to(tl.int64)
    key = (ordered << 32) | low
    neg: tl.constexpr = -9223372036854775807 - 1
    # Select ids first; gather their weights (sigmoid of the summed partial
    # logits, recomputed for the TOPK ids only) after the loop.
    KPAD: tl.constexpr = 16 if TOPK <= 16 else 32
    kk = tl.arange(0, KPAD)
    ids = tl.zeros((KPAD,), dtype=tl.int32)
    for k in tl.static_range(TOPK):
        best = tl.max(key, axis=0)
        idx = (best & (N - 1)).to(tl.int32)
        ids = tl.where(kk == k, idx, ids)
        key = tl.where(offs == idx, neg, key)
    lg = tl.zeros((KPAD,), dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        lg += tl.load(part_ptr + (s * M + row).to(tl.int64) * N + ids)
    w = 1.0 / (1.0 + tl.exp(-lg))
    w = tl.where(w == w, w, -float("inf"))
    m = kk < TOPK
    tl.store(topk_weights_ptr + row * OUT_K + kk, w, mask=m)
    tl.store(topk_ids_ptr + row * OUT_K + kk, ids.to(topk_ids_ptr.dtype.element_ty), mask=m)


_WELM_ROUTER_BLAS_MIN_M = int(__import__("os").environ.get("WELM_ROUTER_BLAS_MIN_M", "1000000000"))


def welm_fast_router_topk(
    x: torch.Tensor,
    weight_bf16: torch.Tensor,
    expert_bias: torch.Tensor,
    topk: int,
    write_scores: bool = False,
    shared_expert_id: int = -1,
):
    """sigmoid(x @ W^T) -> biased top-k. Returns (topk_weights, topk_ids, scores).

    With shared_expert_id >= 0 an extra column (id=shared_expert_id, weight 1)
    is appended so a fused shared expert runs inside the routed MoE."""
    M, K = x.shape
    N = weight_bf16.shape[0]
    assert N & (N - 1) == 0 and N % 32 == 0
    # Tuned on BW1101 (K=2048, N=512).
    if M <= 16:
        BLOCK_M, BLOCK_N, BLOCK_K, split_k, nw = 16, 32, 128, 16, 1
    elif M <= 64:
        BLOCK_M, BLOCK_N, BLOCK_K, split_k, nw = 16, 64, 64, 8, 1
    elif M <= 512:
        BLOCK_M, BLOCK_N, BLOCK_K, split_k, nw = 64, 64, 128, 4, 2
    elif M <= 2048:
        BLOCK_M, BLOCK_N, BLOCK_K, split_k, nw = 64, 128, 64, 4, 2
    else:
        # bit-identical to the 64x128 tile (same K order), ~30% faster
        BLOCK_M, BLOCK_N, BLOCK_K, split_k, nw = 128, 128, 64, 1, 4
    split_k = min(split_k, K // BLOCK_K)
    assert K % (split_k * BLOCK_K) == 0
    if M >= _WELM_ROUTER_BLAS_MIN_M and x.stride(1) == 1:
        # Large prefill: hipBLASLt bf16 GEMM with fp32 output is ~3x faster
        # than the Triton split-K tiles here.
        split_k = 1
        part = torch.mm(x, weight_bf16.t(), out_dtype=torch.float32).unsqueeze(0)
    else:
        part = None
    if part is None:
        part = torch.empty((split_k, M, N), dtype=torch.float32, device=x.device)
        grid = (N // BLOCK_N, split_k, triton.cdiv(M, BLOCK_M))
        _welm_router_splitk_kernel[grid](
            x,
            weight_bf16,
            part,
            M,
            N,
            K,
            x.stride(0),
            SPLIT_K=split_k,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=nw,
        )
    out_k = topk + (1 if shared_expert_id >= 0 else 0)
    topk_weights = torch.empty((M, out_k), dtype=torch.float32, device=x.device)
    topk_ids = torch.empty((M, out_k), dtype=torch.int64, device=x.device)
    scores = (
        torch.empty((M, N), dtype=torch.float32, device=x.device)
        if write_scores
        else topk_weights
    )
    _welm_router_topk_kernel[(M,)](
        part,
        expert_bias,
        scores,
        topk_weights,
        topk_ids,
        M,
        N,
        SPLIT_K=split_k,
        TOPK=topk,
        WRITE_SCORES=write_scores,
        SHARED_ID=shared_expert_id,
        num_warps=1,
    )
    return topk_weights, topk_ids, (scores if write_scores else None)


# ---------------------------------------------------------------------------
# Fused attention prologue for one token per program:
#   k <- rope(rms_norm(k) * gamma)        (written to k_out, bf16 rounded
#                                           after the norm and after rope,
#                                           exactly like the unfused ops)
#   q <- rope(q) [* q_scale]              (in place; q_scale reproduces the
#                                           HCU paged-attention prescale
#                                           (q.float() * s).to(bf16))
# Rope is NeoX style on the last ROPE_DIM dims of every head.
# ---------------------------------------------------------------------------
@triton.jit
def _welm_qk_norm_rope_kernel(
    q_ptr,
    k_ptr,
    k_out_ptr,
    gamma_ptr,
    pos_ptr,
    cos_sin_ptr,
    T,
    q_stride,
    k_stride,
    eps,
    q_scale,
    QH: tl.constexpr,
    KH: tl.constexpr,
    QH_B: tl.constexpr,
    KH_B: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HAS_K_NORM: tl.constexpr,
    SCALE_Q: tl.constexpr,
    q_out_ptr=None,
    Q_OUT: tl.constexpr = False,
    v_ptr=None,
    v_stride=0,
    kc_ptr=None,
    vc_ptr=None,
    loc_ptr=None,
    PAGE: tl.constexpr = 1,
    STORE_KV: tl.constexpr = False,
):
    t = tl.program_id(0)
    HALF: tl.constexpr = ROPE_DIM // 2
    NOPE: tl.constexpr = HEAD_DIM - ROPE_DIM
    d = tl.arange(0, HEAD_DIM)
    dh = tl.arange(0, HALF)
    pos = tl.load(pos_ptr + t).to(tl.int64)
    cos = tl.load(cos_sin_ptr + pos * ROPE_DIM + dh)
    sin = tl.load(cos_sin_ptr + pos * ROPE_DIM + HALF + dh)
    out_ty = k_out_ptr.dtype.element_ty

    # ---- k ----
    kh = tl.arange(0, KH_B)
    kmask = kh[:, None] < KH
    kbase = k_ptr + t.to(tl.int64) * k_stride + kh[:, None] * HEAD_DIM
    obase = k_out_ptr + t.to(tl.int64) * (KH * HEAD_DIM) + kh[:, None] * HEAD_DIM
    kx = tl.load(kbase + d[None, :], mask=kmask, other=0.0).to(tl.float32)
    k1 = tl.load(kbase + NOPE + dh[None, :], mask=kmask, other=0.0).to(tl.float32)
    k2 = tl.load(kbase + NOPE + HALF + dh[None, :], mask=kmask, other=0.0).to(tl.float32)
    if HAS_K_NORM:
        inv = tl.math.rsqrt(tl.sum(kx * kx, axis=1) / HEAD_DIM + eps)
        g = tl.load(gamma_ptr + d)
        g1 = tl.load(gamma_ptr + NOPE + dh)
        g2 = tl.load(gamma_ptr + NOPE + HALF + dh)
        kx = (kx * inv[:, None] * g[None, :]).to(out_ty).to(tl.float32)
        k1 = (k1 * inv[:, None] * g1[None, :]).to(out_ty).to(tl.float32)
        k2 = (k2 * inv[:, None] * g2[None, :]).to(out_ty).to(tl.float32)
    r1 = k1 * cos[None, :] - k2 * sin[None, :]
    r2 = k1 * sin[None, :] + k2 * cos[None, :]
    if STORE_KV:
        # Write k straight into the paged cache (HCU layout K [pages, H, P, D],
        # V [pages, H, D, P]) instead of a k_out buffer + a separate store.
        loc = tl.load(loc_ptr + t).to(tl.int64)
        page = loc // PAGE
        off = loc % PAGE
        obase = kc_ptr + ((page * KH + kh[:, None]) * PAGE + off) * HEAD_DIM
        vsrc = v_ptr + t.to(tl.int64) * v_stride + kh[:, None] * HEAD_DIM + d[None, :]
        vv = tl.load(vsrc, mask=kmask, other=0.0)
        vdst = vc_ptr + ((page * KH + kh[:, None]) * HEAD_DIM + d[None, :]) * PAGE + off
        tl.store(vdst, vv, mask=kmask)
    tl.store(obase + d[None, :], kx.to(out_ty), mask=kmask & (d[None, :] < NOPE))
    tl.store(obase + NOPE + dh[None, :], r1.to(out_ty), mask=kmask)
    tl.store(obase + NOPE + HALF + dh[None, :], r2.to(out_ty), mask=kmask)

    # ---- q ----
    qh = tl.arange(0, QH_B)
    qmask = qh[:, None] < QH
    qbase = q_ptr + t.to(tl.int64) * q_stride + qh[:, None] * HEAD_DIM
    q1 = tl.load(qbase + NOPE + dh[None, :], mask=qmask, other=0.0).to(tl.float32)
    q2 = tl.load(qbase + NOPE + HALF + dh[None, :], mask=qmask, other=0.0).to(tl.float32)
    r1 = (q1 * cos[None, :] - q2 * sin[None, :]).to(out_ty)
    r2 = (q1 * sin[None, :] + q2 * cos[None, :]).to(out_ty)
    # Q_OUT: write the full q into a contiguous [T, QH*HEAD_DIM] buffer so the
    # attention call needs no strided->contiguous copy; values are identical.
    if Q_OUT:
        dst = q_out_ptr + t.to(tl.int64) * (QH * HEAD_DIM) + qh[:, None] * HEAD_DIM
    else:
        dst = qbase
    nmask = qmask & (d[None, :] < NOPE)
    if SCALE_Q:
        qn = tl.load(qbase + d[None, :], mask=nmask, other=0.0)
        tl.store(dst + d[None, :], (qn.to(tl.float32) * q_scale).to(out_ty), mask=nmask)
        r1 = (r1.to(tl.float32) * q_scale).to(out_ty)
        r2 = (r2.to(tl.float32) * q_scale).to(out_ty)
    elif Q_OUT:
        qn = tl.load(qbase + d[None, :], mask=nmask, other=0.0)
        tl.store(dst + d[None, :], qn, mask=nmask)
    tl.store(dst + NOPE + dh[None, :], r1, mask=qmask)
    tl.store(dst + NOPE + HALF + dh[None, :], r2, mask=qmask)


def welm_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_dim: int,
    rope_dim: int,
    k_gamma: torch.Tensor = None,
    eps: float = 1e-6,
    q_scale: float = 1.0,
    return_q: bool = False,
    kv_store=None,
):
    """q, k: (T, heads*head_dim) row views. Returns k_out (T, kv_heads*head_dim),
    contiguous; q is updated in place, or with ``return_q`` written to a new
    contiguous buffer and returned as ``(k_out, q_out)``.

    ``kv_store=(v, k_cache, v_cache, loc)`` writes k and v directly into the
    HCU paged KV cache (K [pages, H, P, D], V [pages, H, D, P]) at ``loc``;
    the returned k is then the unmodified input (the cache holds the result).
    """
    T = q.shape[0]
    qh = q.shape[1] // head_dim
    kh = k.shape[1] // head_dim
    assert q.stride(1) == 1 and k.stride(1) == 1
    if kv_store is not None:
        v, k_cache, v_cache, loc = kv_store
        assert v.stride(1) == 1 and k_cache.shape[1] == kh
    k_out = (
        torch.empty((T, kh * head_dim), dtype=k.dtype, device=k.device)
        if kv_store is None
        else k
    )
    q_out = (
        torch.empty((T, qh * head_dim), dtype=q.dtype, device=q.device)
        if return_q
        else None
    )
    if T == 0:
        return (k_out, q_out) if return_q else k_out
    _welm_qk_norm_rope_kernel[(T,)](
        q,
        k,
        k_out,
        k_gamma if k_gamma is not None else k,
        positions,
        cos_sin_cache,
        T,
        q.stride(0),
        k.stride(0),
        eps,
        q_scale,
        QH=qh,
        KH=kh,
        QH_B=triton.next_power_of_2(qh),
        KH_B=max(2, triton.next_power_of_2(kh)),
        HEAD_DIM=head_dim,
        ROPE_DIM=rope_dim,
        HAS_K_NORM=k_gamma is not None,
        SCALE_Q=q_scale != 1.0,
        q_out_ptr=q_out,
        Q_OUT=return_q,
        v_ptr=kv_store[0] if kv_store is not None else None,
        v_stride=kv_store[0].stride(0) if kv_store is not None else 0,
        kc_ptr=kv_store[1] if kv_store is not None else None,
        vc_ptr=kv_store[2] if kv_store is not None else None,
        loc_ptr=kv_store[3] if kv_store is not None else None,
        PAGE=kv_store[1].shape[2] if kv_store is not None else 1,
        STORE_KV=kv_store is not None,
        num_warps=4,
    )
    return (k_out, q_out) if return_q else k_out


# ---------------------------------------------------------------------------
# Paged KV store for the HCU FA layout: K [pages, H, P, D], V [pages, H, D, P].
# Replaces loc // P, loc % P and two advanced-index writes (4+ kernels).
# ---------------------------------------------------------------------------
@triton.jit
def _welm_hcu_store_kv_kernel(
    k_ptr,
    v_ptr,
    kc_ptr,
    vc_ptr,
    loc_ptr,
    k_stride_t,
    k_stride_h,
    v_stride_t,
    v_stride_h,
    H: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    loc = tl.load(loc_ptr + t).to(tl.int64)
    page = loc // P
    off = loc % P
    d = tl.arange(0, D)
    kv = tl.load(k_ptr + t.to(tl.int64) * k_stride_t + h * k_stride_h + d)
    tl.store(kc_ptr + ((page * H + h) * P + off) * D + d, kv)
    dv = tl.arange(0, DV)
    vv = tl.load(v_ptr + t.to(tl.int64) * v_stride_t + h * v_stride_h + dv)
    tl.store(vc_ptr + ((page * H + h) * DV + dv) * P + off, vv)


def welm_hcu_store_kv(cache_k, cache_v, k_buffer, v_buffer, loc):
    """cache_k: (T, H, D) / (T, H*D); k_buffer (pages, H, P, D); v_buffer (pages, H, DV, P)."""
    T = loc.shape[0]
    if T == 0:
        return
    _, H, P, D = k_buffer.shape
    DV = v_buffer.shape[2]
    ck = cache_k.view(cache_k.shape[0], H, D)
    cv = cache_v.view(cache_v.shape[0], H, DV)
    assert ck.stride(2) == 1 and cv.stride(2) == 1
    _welm_hcu_store_kv_kernel[(T, H)](
        ck,
        cv,
        k_buffer,
        v_buffer,
        loc,
        ck.stride(0),
        ck.stride(1),
        cv.stride(0),
        cv.stride(1),
        H=H,
        P=P,
        D=D,
        DV=DV,
        num_warps=1 if D <= 128 else 2,
    )


# ---------------------------------------------------------------------------
# Short uniform-query attention as per-token decode rows (see
# flashattention_interface): row b*n+i gets block_table[b] and cache length
# max(seqused_k[b] - (n - 1) + i, 1). One kernel instead of five.
# ---------------------------------------------------------------------------
@triton.jit
def _short_q_expand_kernel(
    seqused_ptr,
    bt_ptr,
    lens_ptr,
    out_bt_ptr,
    W,
    bt_stride,
    N_Q: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    b = r // N_Q
    i = r % N_Q
    if tl.program_id(1) == 0:
        L = tl.load(seqused_ptr + b) - (N_Q - 1) + i
        tl.store(lens_ptr + r, tl.maximum(L, 1).to(lens_ptr.dtype.element_ty))
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = offs < W
    row = tl.load(bt_ptr + b.to(tl.int64) * bt_stride + offs, mask=m)
    tl.store(out_bt_ptr + r.to(tl.int64) * W + offs, row, mask=m)


def short_q_expand(seqused_k: torch.Tensor, block_table: torch.Tensor, n: int):
    bs, W = block_table.shape
    lens = torch.empty((bs * n,), dtype=seqused_k.dtype, device=seqused_k.device)
    out_bt = torch.empty((bs * n, W), dtype=block_table.dtype, device=block_table.device)
    if bs == 0:
        return lens, out_bt
    BLOCK = 1024
    _short_q_expand_kernel[(bs * n, triton.cdiv(W, BLOCK))](
        seqused_k, block_table, lens, out_bt, W, block_table.stride(0),
        N_Q=n, BLOCK=BLOCK, num_warps=4,
    )
    return lens, out_bt


# ---------------------------------------------------------------------------
# Single-program MoE align for small batches (decode / verify). Same output
# contract as moe_align_block_size's HCU path (negative ids -> sentinel bucket
# reported as expert -1; padding slots hold numel), in one launch instead of
# fill + zeros + count + cumsum + scatter.
# ---------------------------------------------------------------------------
@triton.jit
def _moe_align_small_kernel(
    topk_ids_ptr,
    sorted_ids_ptr,
    expert_ids_ptr,
    num_post_pad_ptr,
    scratch_ptr,
    numel,
    max_padded,
    num_buckets: tl.constexpr,
    invalid_expert: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    e = tl.arange(0, BLOCK_E)
    emask = e < num_buckets
    counts_ptr = scratch_ptr
    fill_ptr = scratch_ptr + BLOCK_E
    tl.store(counts_ptr + e, tl.zeros((BLOCK_E,), tl.int32))
    tl.store(fill_ptr + e, tl.zeros((BLOCK_E,), tl.int32))
    for start in range(0, max_padded, BLOCK_T):
        o = start + tl.arange(0, BLOCK_T)
        tl.store(sorted_ids_ptr + o, tl.zeros((BLOCK_T,), tl.int32) + numel, mask=o < max_padded)
    tl.debug_barrier()
    for start in range(0, numel, BLOCK_T):
        o = start + tl.arange(0, BLOCK_T)
        m = o < numel
        ex = tl.load(topk_ids_ptr + o, mask=m, other=0).to(tl.int32)
        ex = tl.where(ex < 0, invalid_expert, ex)
        tl.atomic_add(counts_ptr + ex, 1, mask=m, sem="relaxed")
    tl.debug_barrier()
    counts = tl.load(counts_ptr + e, mask=emask, other=0, cache_modifier=".cg")
    nblk = (counts + block_size - 1) // block_size
    padded = nblk * block_size
    excl = tl.cumsum(padded, axis=0) - padded
    tl.store(num_post_pad_ptr, tl.sum(padded, axis=0))
    tl.store(counts_ptr + e, excl)  # reuse counts slot as cumsum
    expert_val = tl.where(e == invalid_expert, -1, e)
    max_nblk = tl.max(nblk, axis=0)
    for i in range(0, max_nblk):
        tl.store(expert_ids_ptr + excl // block_size + i, expert_val, mask=emask & (i < nblk))
    tl.debug_barrier()
    for start in range(0, numel, BLOCK_T):
        o = start + tl.arange(0, BLOCK_T)
        m = o < numel
        ex = tl.load(topk_ids_ptr + o, mask=m, other=0).to(tl.int32)
        ex = tl.where(ex < 0, invalid_expert, ex)
        slot = tl.atomic_add(fill_ptr + ex, 1, mask=m, sem="relaxed")
        base = tl.load(counts_ptr + ex, mask=m, other=0, cache_modifier=".cg")
        tl.store(sorted_ids_ptr + base + slot, o.to(tl.int32), mask=m)


def moe_align_small(topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_post_pad):
    numel = topk_ids.numel()
    num_buckets = num_experts + 1
    BLOCK_E = triton.next_power_of_2(num_buckets)
    scratch = torch.empty((2 * BLOCK_E,), dtype=torch.int32, device=topk_ids.device)
    _moe_align_small_kernel[(1,)](
        topk_ids,
        sorted_ids,
        expert_ids,
        num_post_pad,
        scratch,
        numel,
        sorted_ids.numel(),
        num_buckets,
        num_experts,
        block_size,
        BLOCK_E=BLOCK_E,
        BLOCK_T=1024,
        num_warps=16,
    )


# ---------------------------------------------------------------------------
# y[t, h, :] *= sigmoid(gate[t, h]) with a strided (row-sliced) gate, so the
# head-wise gate can come straight out of the merged qkv+gate GEMM.
# Matches sigmoid_mul_kernel: sigmoid in fp32, cast to y dtype, then multiply.
# ---------------------------------------------------------------------------
@triton.jit
def _sigmoid_mul_strided_kernel(
    g_ptr, y_ptr, T, g_stride, y_stride_t, H: tl.constexpr, H_B: tl.constexpr, D: tl.constexpr
):
    t = tl.program_id(0)
    h = tl.arange(0, H_B)
    d = tl.arange(0, D)
    hm = h < H
    g = tl.load(g_ptr + t.to(tl.int64) * g_stride + h, mask=hm, other=0.0).to(tl.float32)
    s = tl.sigmoid(g).to(y_ptr.dtype.element_ty)
    ptr = y_ptr + t.to(tl.int64) * y_stride_t + h[:, None] * D + d[None, :]
    y = tl.load(ptr, mask=hm[:, None])
    tl.store(ptr, s[:, None] * y, mask=hm[:, None])


def sigmoid_mul_strided_(gate: torch.Tensor, y: torch.Tensor, num_heads: int):
    """gate: (T, H) with gate.stride(1) == 1; y: (T, H*D) contiguous rows, in place."""
    T = y.shape[0]
    if T == 0:
        return y
    D = y.shape[-1] // num_heads
    assert gate.stride(-1) == 1 and y.stride(-1) == 1
    _sigmoid_mul_strided_kernel[(T,)](
        gate, y, T, gate.stride(0), y.stride(0), H=num_heads,
        H_B=triton.next_power_of_2(num_heads), D=D, num_warps=4,
    )
    return y


# ---------------------------------------------------------------------------
# Skinny GEMM y = x @ W^T for decode-sized M (bf16 in/out, fp32 accumulate).
# hipBLASLt picks poor kernels for M <= ~64 on these shapes (e.g. the 38912 x
# 2048 per-rank lm_head runs 3-10x slower than a plain weight-streaming tile).
# ---------------------------------------------------------------------------
@triton.jit
def _skinny_gemm_kernel(
    x_ptr, w_ptr, y_ptr, M, N, K: tl.constexpr, sxm, sym,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    nm = offs_n < N
    mm = offs_m < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in tl.static_range(0, K, BLOCK_K):
        a = tl.load(
            x_ptr + offs_m[:, None].to(tl.int64) * sxm + (k0 + offs_k)[None, :],
            mask=mm[:, None], other=0.0,
        )
        b = tl.load(
            w_ptr + offs_n[None, :].to(tl.int64) * K + (k0 + offs_k)[:, None],
            mask=nm[None, :], other=0.0,
        )
        acc = tl.dot(a, b, acc)
    tl.store(
        y_ptr + offs_m[:, None].to(tl.int64) * sym + offs_n[None, :],
        acc.to(y_ptr.dtype.element_ty),
        mask=mm[:, None] & nm[None, :],
    )


# (max_M, BLOCK_M, BLOCK_N, BLOCK_K, num_warps) per weight shape (N, K), tuned on BW1101.
_SKINNY_CFGS = {}


def register_skinny_cfg(n: int, k: int, table):
    _SKINNY_CFGS[(n, k)] = sorted(table)


def skinny_gemm_supported(x: torch.Tensor, w: torch.Tensor) -> bool:
    tbl = _SKINNY_CFGS.get((w.shape[0], w.shape[1]))
    return (
        tbl is not None
        and x.dim() == 2
        and 0 < x.shape[0] <= tbl[-1][0]
        and x.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and x.stride(1) == 1
        and w.is_contiguous()
    )


def skinny_gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    N = w.shape[0]
    for max_m, bm, bn, bk, nw in _SKINNY_CFGS[(N, K)]:
        if M <= max_m:
            break
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    _skinny_gemm_kernel[(triton.cdiv(N, bn), triton.cdiv(M, bm))](
        x, w, y, M, N, K, x.stride(0), y.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw, num_stages=1,
    )
    return y


# lm_head per TP4 rank (vocab 155648 / 4 = 38912, hidden 2048): hipBLASLt is
# 3-13x slower than this at every M measured (1..4096), with identical output.
register_skinny_cfg(
    38912,
    2048,
    [
        (1, 16, 32, 128, 1),
        (4, 16, 32, 256, 1),
        (16, 32, 128, 128, 2),
        (32, 32, 128, 128, 4),
        (48, 64, 128, 128, 4),
        (64, 64, 64, 128, 4),
        (128, 64, 64, 128, 2),
        (512, 128, 256, 64, 8),
        (1024, 128, 128, 128, 8),
        (1 << 30, 128, 128, 64, 8),
    ],
)
# WeLM qkv + head-wise gate projection (TP4: 1536 + 2 * 256 + 6 rows).
register_skinny_cfg(2054, 2048, [(1, 16, 16, 256, 1), (16, 16, 16, 512, 1)])


def warmup_skinny_gemm(w: torch.Tensor) -> None:
    """JIT-compile every config registered for w's shape (avoids first-request stalls)."""
    tbl = _SKINNY_CFGS.get((w.shape[0], w.shape[1]))
    if tbl is None or w.dtype != torch.bfloat16 or w.device.type != "cuda":
        return
    prev = 0
    for max_m, *_ in tbl:
        m = min(max_m, max(prev + 1, 1), 4096)
        x = torch.zeros((m, w.shape[1]), dtype=w.dtype, device=w.device)
        skinny_gemm(x, w)
        prev = max_m
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Split-KV attention for short queries over a long paged prefix (prefill with
# prefix-cache hits).  The vendor varlen prefill kernel parallelises only over
# query blocks, so a few dozen new tokens against a 9k-30k cached prefix runs
# on a handful of workgroups (1.7-5.4 ms per layer).  Here every split of the
# KV range is its own program, and all query heads sharing a KV head are
# folded into the M dimension so each page is read once per split.
# KV layout is HCU legacy_bhsd: k [blocks, kv_heads, PAGE, D],
# v [blocks, kv_heads, D, PAGE].  Lengths are read on device (graph-safe).
# ---------------------------------------------------------------------------
@triton.jit
def _welm_pfx_attn_partial_kernel(
    Q, K, V, BT, SEQK, CUQ, MID_O, MID_LSE, sm_scale,
    stride_qt, stride_qh,
    stride_kb, stride_kh, stride_kp,
    stride_vb, stride_vh, stride_vd,
    stride_bt,
    stride_mo_t, stride_mo_h, stride_mo_s,
    stride_ml_t, stride_ml_h,
    G: tl.constexpr, KVH: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
    BLOCK_M: tl.constexpr, NUM_SPLITS: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    split = tl.program_id(2)
    b = pid_bh // KVH
    kvh = pid_bh % KVH
    q_start = tl.load(CUQ + b)
    n_q = tl.load(CUQ + b + 1) - q_start
    rows = n_q * G
    if pid_m * BLOCK_M >= rows:
        return
    kv_len = tl.load(SEQK + b)
    prefix = kv_len - n_q
    split_len = tl.cdiv(tl.cdiv(kv_len, NUM_SPLITS), PAGE) * PAGE
    k_lo = split * split_len
    k_hi = tl.minimum(k_lo + split_len, kv_len)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < rows
    tok = offs_m // G
    head = kvh * G + offs_m % G
    offs_d = tl.arange(0, D)
    q = tl.load(
        Q + (q_start + tok)[:, None] * stride_qt + head[:, None] * stride_qh + offs_d[None, :],
        mask=m_mask[:, None], other=0.0,
    )
    q_pos = prefix + tok
    offs_p = tl.arange(0, PAGE)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    for start in range(k_lo, k_hi, PAGE):
        page = tl.load(BT + b * stride_bt + start // PAGE)
        k = tl.load(K + page * stride_kb + kvh * stride_kh + offs_p[:, None] * stride_kp + offs_d[None, :])
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        k_pos = start + offs_p
        valid = (k_pos[None, :] <= q_pos[:, None]) & (k_pos[None, :] < k_hi)
        qk = tl.where(valid, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(qk - m_safe[:, None])
        alpha = tl.exp(m_i - m_safe)
        v = tl.load(V + page * stride_vb + kvh * stride_vh + offs_d[:, None] * stride_vd + offs_p[None, :])
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), tl.trans(v))
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    has = l_i > 0
    o = acc / tl.where(has, l_i, 1.0)[:, None]
    lse = tl.where(has, m_i + tl.log(tl.where(has, l_i, 1.0)), float("-inf"))
    t_abs = q_start + tok
    tl.store(
        MID_O + t_abs[:, None] * stride_mo_t + head[:, None] * stride_mo_h + split * stride_mo_s + offs_d[None, :],
        o, mask=m_mask[:, None],
    )
    tl.store(MID_LSE + t_abs * stride_ml_t + head * stride_ml_h + split, lse, mask=m_mask)


@triton.jit
def _welm_pfx_attn_reduce_kernel(
    MID_O, MID_LSE, SINKS, OUT,
    stride_mo_t, stride_mo_h, stride_mo_s,
    stride_ml_t, stride_ml_h,
    stride_ot, stride_oh,
    D: tl.constexpr, NUM_SPLITS: tl.constexpr, HAS_SINK: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    offs_s = tl.arange(0, NUM_SPLITS)
    offs_d = tl.arange(0, D)
    lse = tl.load(MID_LSE + t * stride_ml_t + h * stride_ml_h + offs_s)
    m = tl.max(lse, 0)
    if HAS_SINK:
        sink = tl.load(SINKS + h).to(tl.float32)
        m = tl.maximum(m, sink)
    w = tl.exp(lse - m)
    denom = tl.sum(w, 0)
    if HAS_SINK:
        denom += tl.exp(sink - m)
    o = tl.load(MID_O + t * stride_mo_t + h * stride_mo_h + offs_s[:, None] * stride_mo_s + offs_d[None, :])
    out = tl.sum(o * w[:, None], 0) / denom
    tl.store(OUT + t * stride_ot + h * stride_oh + offs_d, out.to(OUT.dtype.element_ty))


def welm_prefix_attention(
    q: torch.Tensor,            # [total_q, H, D]
    k_cache: torch.Tensor,      # [blocks, KVH, PAGE, D]
    v_cache: torch.Tensor,      # [blocks, KVH, D, PAGE]
    block_table: torch.Tensor,  # [bs, W] int32
    seqused_k: torch.Tensor,    # [bs] total kv length incl. new tokens
    cu_seqlens_q: torch.Tensor, # [bs + 1]
    max_seqlen_q: int,
    softmax_scale: float,
    sinks: torch.Tensor = None, # [H] or None
    out: torch.Tensor = None,
    num_splits: int = 32,
    block_m: int = 32,
    num_warps: int = 4,
    num_stages: int = 1,
):
    total_q, H, D = q.shape
    KVH, PAGE = k_cache.shape[1], k_cache.shape[2]
    G = H // KVH
    bs = cu_seqlens_q.shape[0] - 1
    mid_o = torch.empty((total_q, H, num_splits, D), dtype=torch.float32, device=q.device)
    mid_lse = torch.empty((total_q, H, num_splits), dtype=torch.float32, device=q.device)
    if out is None:
        out = torch.empty_like(q)
    grid = (bs * KVH, triton.cdiv(max_seqlen_q * G, block_m), num_splits)
    _welm_pfx_attn_partial_kernel[grid](
        q, k_cache, v_cache, block_table, seqused_k, cu_seqlens_q, mid_o, mid_lse,
        softmax_scale,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        block_table.stride(0),
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
        mid_lse.stride(0), mid_lse.stride(1),
        G=G, KVH=KVH, D=D, PAGE=PAGE, BLOCK_M=block_m, NUM_SPLITS=num_splits,
        num_warps=num_warps, num_stages=num_stages,
    )
    _welm_pfx_attn_reduce_kernel[(total_q, H)](
        mid_o, mid_lse, sinks if sinks is not None else mid_lse, out,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
        mid_lse.stride(0), mid_lse.stride(1),
        out.stride(0), out.stride(1),
        D=D, NUM_SPLITS=num_splits, HAS_SINK=sinks is not None,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# Decode (q_len=1) split-KV paged attention, fixed-size splits.
# ---------------------------------------------------------------------------
@triton.jit
def _welm_dec_attn_partial_kernel(
    Q, K, V, BT, SEQK, MID_O, MID_LSE, sm_scale,
    stride_qb, stride_qh,
    stride_kb, stride_kh, stride_kp,
    stride_vb, stride_vh, stride_vd,
    stride_bt,
    stride_mo_b, stride_mo_h, stride_mo_s,
    stride_ml_b, stride_ml_h,
    G: tl.constexpr, KVH: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
    BLOCK_M: tl.constexpr, CHUNK_PAGES: tl.constexpr, WINDOW_LEFT: tl.constexpr = -1,
):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    slot = split
    b = pid // KVH
    kvh = pid % KVH
    kv_len = tl.load(SEQK + b)
    if WINDOW_LEFT >= 0:
        # Sliding window: keys [kv_len - 1 - WINDOW_LEFT, kv_len). The grid
        # covers only the window; split 0 is the chunk holding its start.
        w_lo = tl.maximum(kv_len - 1 - WINDOW_LEFT, 0)
        split = split + w_lo // (CHUNK_PAGES * PAGE)
    else:
        w_lo = 0
    lo = split * (CHUNK_PAGES * PAGE)
    if lo >= kv_len:
        return
    hi = tl.minimum(lo + CHUNK_PAGES * PAGE, kv_len)
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < G
    head = kvh * G + offs_m
    offs_d = tl.arange(0, D)
    q = tl.load(Q + b * stride_qb + head[:, None] * stride_qh + offs_d[None, :],
                mask=m_mask[:, None], other=0.0)
    offs_p = tl.arange(0, PAGE)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    for start in range(lo, hi, PAGE):
        page = tl.load(BT + b * stride_bt + start // PAGE).to(tl.int64)
        k = tl.load(K + page * stride_kb + kvh * stride_kh + offs_p[:, None] * stride_kp + offs_d[None, :])
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        kpos = start + offs_p
        qk = tl.where(((kpos < hi) & (kpos >= w_lo))[None, :], qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(qk - m_safe[:, None])
        alpha = tl.exp(m_i - m_safe)
        v = tl.load(V + page * stride_vb + kvh * stride_vh + offs_d[:, None] * stride_vd + offs_p[None, :])
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), tl.trans(v))
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new
    o = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    tl.store(MID_O + b * stride_mo_b + head[:, None] * stride_mo_h + slot * stride_mo_s + offs_d[None, :],
             o, mask=m_mask[:, None])
    tl.store(MID_LSE + b * stride_ml_b + head * stride_ml_h + slot, lse, mask=m_mask)


@triton.jit
def _welm_dec_attn_reduce_kernel(
    MID_O, MID_LSE, SEQK, SINKS, OUT,
    stride_mo_b, stride_mo_h, stride_mo_s,
    stride_ml_b, stride_ml_h,
    stride_ob, stride_oh,
    D: tl.constexpr, PAGE: tl.constexpr, CHUNK_PAGES: tl.constexpr,
    BLOCK_S: tl.constexpr, HAS_SINK: tl.constexpr, WINDOW_LEFT: tl.constexpr = -1,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    kv_len = tl.load(SEQK + b)
    n = tl.cdiv(kv_len, CHUNK_PAGES * PAGE)
    s_begin = 0
    if WINDOW_LEFT >= 0:
        s_begin = tl.maximum(kv_len - 1 - WINDOW_LEFT, 0) // (CHUNK_PAGES * PAGE)
    offs_s = tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, D)
    m = tl.full((), float("-inf"), tl.float32)
    if HAS_SINK:
        m = tl.load(SINKS + h).to(tl.float32)
    denom = tl.zeros((), tl.float32)
    if HAS_SINK:
        denom = tl.full((), 1.0, tl.float32)
    acc = tl.zeros([D], dtype=tl.float32)
    n = n - s_begin  # partials are stored relative to the window's first split
    for s0 in range(0, n, BLOCK_S):
        s = s0 + offs_s
        msk = s < n
        lse = tl.load(MID_LSE + b * stride_ml_b + h * stride_ml_h + s, mask=msk, other=float("-inf"))
        m_new = tl.maximum(m, tl.max(lse, 0))
        scale = tl.exp(m - m_new)
        w = tl.exp(lse - m_new)
        o = tl.load(MID_O + b * stride_mo_b + h * stride_mo_h + s[:, None] * stride_mo_s + offs_d[None, :],
                    mask=msk[:, None], other=0.0)
        acc = acc * scale + tl.sum(o * w[:, None], 0)
        denom = denom * scale + tl.sum(w, 0)
        m = m_new
    out = acc / tl.where(denom > 0, denom, 1.0)
    tl.store(OUT + b * stride_ob + h * stride_oh + offs_d, out.to(OUT.dtype.element_ty))


def welm_decode_attention(q, k_cache, v_cache, block_table, seq_lens, softmax_scale,
                          max_context_len, sinks=None, out=None, chunk_pages=None,
                          num_warps=None, num_stages=1, block_s=32, window_left=-1):
    """One query token per row: q [B, H, D] -> out [B, H, D].

    k_cache [blocks, KVH, PAGE, D], v_cache [blocks, KVH, D, PAGE] (HCU legacy
    layout). Splits are CHUNK_PAGES pages each; the grid covers
    max_context_len and splits past a row's length exit, so no host-side
    length is needed (CUDA-graph safe)."""
    if chunk_pages is None:
        chunk_pages = 2 if q.shape[0] <= 16 else 4
    if num_warps is None:
        num_warps = 4 if q.shape[0] <= 16 else 2
    B, H, D = q.shape
    KVH, PAGE = k_cache.shape[1], k_cache.shape[2]
    G = H // KVH
    max_splits = triton.cdiv(max_context_len, chunk_pages * PAGE)
    # window_left >= 0: the grid spans the window (+1 for alignment); the
    # partials keep absolute split indices.
    grid_splits = (
        min(max_splits, triton.cdiv(window_left + 1, chunk_pages * PAGE) + 1)
        if window_left >= 0
        else max_splits
    )
    mid_o = torch.empty((B, H, grid_splits, D), dtype=torch.float32, device=q.device)
    mid_lse = torch.empty((B, H, grid_splits), dtype=torch.float32, device=q.device)
    if out is None:
        out = torch.empty_like(q)
    _welm_dec_attn_partial_kernel[(B * KVH, grid_splits)](
        q, k_cache, v_cache, block_table, seq_lens, mid_o, mid_lse, softmax_scale,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        block_table.stride(0),
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
        mid_lse.stride(0), mid_lse.stride(1),
        G=G, KVH=KVH, D=D, PAGE=PAGE, BLOCK_M=max(16, triton.next_power_of_2(G)),
        CHUNK_PAGES=chunk_pages, WINDOW_LEFT=window_left,
        num_warps=num_warps, num_stages=num_stages,
    )
    _welm_dec_attn_reduce_kernel[(B, H)](
        mid_o, mid_lse, seq_lens, sinks if sinks is not None else mid_lse, out,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
        mid_lse.stride(0), mid_lse.stride(1),
        out.stride(0), out.stride(1),
        D=D, PAGE=PAGE, CHUNK_PAGES=chunk_pages, BLOCK_S=block_s, HAS_SINK=sinks is not None,
        WINDOW_LEFT=window_left, num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# Paged KV (HCU legacy layout K [pages, KVH, P, D], V [pages, KVH, D, P]) ->
# contiguous K/V [tokens, KVH, D] packed per request at cu_k offsets. Same
# result as flash_attn_cuda.pagedkv_to_contiguouskv, which has a ~100 us
# floor per call (0.1-0.25 TB/s) at prefill sizes.
# ---------------------------------------------------------------------------
@triton.jit
def _welm_pagedkv_to_contig_kernel(
    K, V, KT, VT, BT, SEQK, CUK,
    stride_kb, stride_kh, stride_kp,
    stride_vb, stride_vh, stride_vd,
    stride_bt, stride_kt_t, stride_kt_h, stride_vt_t, stride_vt_h,
    D: tl.constexpr, PAGE: tl.constexpr,
):
    b = tl.program_id(0)
    p = tl.program_id(1)
    h = tl.program_id(2)
    n = tl.load(SEQK + b)
    t0 = p * PAGE
    if t0 >= n:
        return
    base = tl.load(CUK + b).to(tl.int64)
    page = tl.load(BT + b * stride_bt + p).to(tl.int64)
    offs_t = tl.arange(0, PAGE)
    offs_d = tl.arange(0, D)
    m = (t0 + offs_t) < n
    k = tl.load(K + page * stride_kb + h * stride_kh + offs_t[:, None] * stride_kp + offs_d[None, :])
    tl.store(KT + (base + t0 + offs_t)[:, None] * stride_kt_t + h * stride_kt_h + offs_d[None, :], k, mask=m[:, None])
    v = tl.load(V + page * stride_vb + h * stride_vh + offs_d[:, None] * stride_vd + offs_t[None, :])  # [D, P]
    tl.store(VT + (base + t0 + offs_t)[None, :] * stride_vt_t + h * stride_vt_h + offs_d[:, None], v, mask=m[None, :])


def welm_pagedkv_to_contiguous(kt, vt, k_cache, v_cache, block_table, seqused_k, cu_k, max_seqlen_k):
    """Fills kt/vt [>= sum(seqused_k), KVH, D] and cu_k [bs + 1] (int32)."""
    bs = seqused_k.shape[0]
    cu_k[:1].zero_()
    torch.cumsum(seqused_k, 0, out=cu_k[1:])
    KVH, PAGE, D = k_cache.shape[1], k_cache.shape[2], k_cache.shape[3]
    max_pages = triton.cdiv(int(max_seqlen_k), PAGE)
    if bs == 0 or max_pages == 0:
        return
    _welm_pagedkv_to_contig_kernel[(bs, max_pages, KVH)](
        k_cache, v_cache, kt, vt, block_table, seqused_k, cu_k,
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        block_table.stride(0), kt.stride(0), kt.stride(1), vt.stride(0), vt.stride(1),
        D=D, PAGE=PAGE, num_warps=4,
    )
