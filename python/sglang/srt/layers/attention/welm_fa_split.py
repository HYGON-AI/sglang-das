"""Split-KV prefill for short queries over long cached prefixes (HCU, hdim 256).

The fa256 varlen kernel parallelises over query blocks x heads only. With a
long cached prefix and a short new segment (the common prefix-cache-hit
prefill), there are too few query blocks to fill the device and every block
walks the whole prefix serially. Here the prefix keys are cut into chunks that
run as separate non-causal varlen sequences (the query rows are replicated per
chunk), the new tokens attend to themselves in one causal call, and the
partial results are merged with their log-sum-exp.

Gaps between the key ranges of a call (another request's keys) are covered by
zero-length query sequences, so both calls index the contiguous K/V buffer
produced by ``pagedkv_to_contiguouskv`` directly.
"""

from typing import List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

SPLIT_CHUNK = 2048
# Split only when the query blocks alone cannot fill the device.
SPLIT_MAX_TOTAL_Q = 2560
SPLIT_MIN_PREFIX = 2048

_meta_cache: dict = {}


@triton.jit
def _welm_fa_split_merge_kernel(
    O_PRE, LSE_PRE, O_SELF, LSE_SELF, BASE, STRIDE, CNT, SINKS, OUT,
    stride_op_t, stride_op_h, stride_lp_h,
    stride_os_t, stride_os_h, stride_ls_h,
    stride_o_t, stride_o_h,
    D: tl.constexpr, HAS_SINK: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    offs_d = tl.arange(0, D)
    m = tl.load(LSE_SELF + h * stride_ls_h + t)
    acc = tl.load(O_SELF + t * stride_os_t + h * stride_os_h + offs_d).to(tl.float32)
    denom = tl.full((), 1.0, tl.float32)
    base = tl.load(BASE + t)
    stride = tl.load(STRIDE + t)
    cnt = tl.load(CNT + t)
    for c in range(cnt):
        row = base + c * stride
        l = tl.load(LSE_PRE + h * stride_lp_h + row)
        o = tl.load(O_PRE + row * stride_op_t + h * stride_op_h + offs_d).to(tl.float32)
        m_new = tl.maximum(m, l)
        a = tl.exp(m - m_new)
        w = tl.exp(l - m_new)
        acc = acc * a + o * w
        denom = denom * a + w
        m = m_new
    if HAS_SINK:
        s = tl.load(SINKS + h).to(tl.float32)
        m_new = tl.maximum(m, s)
        a = tl.exp(m - m_new)
        acc = acc * a
        denom = denom * a + tl.exp(s - m_new)
    tl.store(OUT + t * stride_o_t + h * stride_o_h + offs_d,
             (acc / denom).to(OUT.dtype.element_ty))


def should_split(q_lens: Sequence[int], k_lens: Sequence[int]) -> bool:
    if sum(q_lens) > SPLIT_MAX_TOTAL_Q:
        return False
    return any(k - n >= SPLIT_MIN_PREFIX for n, k in zip(q_lens, k_lens))


def _build_meta(q_lens: Tuple[int, ...], k_lens: Tuple[int, ...], device):
    key = (q_lens, k_lens, device)
    meta = _meta_cache.get(key)
    if meta is not None:
        return meta
    C = SPLIT_CHUNK
    rep_idx: List[int] = []
    cq_pre, ck_pre = [0], [0]
    cq_self, ck_self = [0], [0]
    base, stride, cnt = [], [], []
    q_off = k_off = 0
    rows = 0
    max_q_pre = max_k_pre = 1
    for n, k in zip(q_lens, k_lens):
        p = k - n
        # prefix call: gap (previous request's self keys) then prefix chunks
        if k_off > ck_pre[-1]:
            cq_pre.append(cq_pre[-1]); ck_pre.append(k_off)
        s = (p + C - 1) // C if n > 0 else 0
        for c in range(s):
            c1 = min((c + 1) * C, p)
            rep_idx.extend(range(q_off, q_off + n))
            cq_pre.append(cq_pre[-1] + n); ck_pre.append(k_off + c1)
            max_q_pre = max(max_q_pre, n); max_k_pre = max(max_k_pre, c1 - c * C)
        if s == 0 and p > 0:
            cq_pre.append(cq_pre[-1]); ck_pre.append(k_off + p)
        base.extend([rows + i for i in range(n)])
        stride.extend([n] * n)
        cnt.extend([s] * n)
        rows += s * n
        # self call: gap (this request's prefix) then its own keys, causal
        if p > 0:
            cq_self.append(cq_self[-1]); ck_self.append(k_off + p)
        cq_self.append(cq_self[-1] + n); ck_self.append(k_off + k)
        q_off += n
        k_off += k
    i32 = dict(dtype=torch.int32, device=device)
    meta = dict(
        rep_idx=torch.tensor(rep_idx, dtype=torch.int64, device=device),
        cq_pre=torch.tensor(cq_pre, **i32), ck_pre=torch.tensor(ck_pre, **i32),
        cq_self=torch.tensor(cq_self, **i32), ck_self=torch.tensor(ck_self, **i32),
        base=torch.tensor(base, **i32), stride=torch.tensor(stride, **i32),
        cnt=torch.tensor(cnt, **i32),
        max_q_pre=max_q_pre, max_k_pre=max_k_pre,
        max_q_self=max(q_lens), max_k_self=max(q_lens),
        n_rep=len(rep_idx),
    )
    if len(_meta_cache) > 64:
        _meta_cache.clear()
    _meta_cache[key] = meta
    return meta


def fa256_split_prefill(fa256, q, kt, vt, q_lens, k_lens, softmax_scale, sinks,
                        cfg, out: Optional[torch.Tensor] = None):
    """q [total_q, H, D]; kt/vt contiguous per request ([sum k_lens, KVH, D])."""
    meta = _build_meta(tuple(int(x) for x in q_lens), tuple(int(x) for x in k_lens), q.device)
    total_q, H, D = q.shape
    if meta["n_rep"] > 0:
        qrep = q.index_select(0, meta["rep_idx"])
        o_pre, lse_pre = fa256.varlen_fwd256(
            qrep, kt, vt, None, meta["cq_pre"], meta["ck_pre"],
            meta["max_q_pre"], meta["max_k_pre"], softmax_scale,
            False, -1, -1, None, cfg,
        )
    else:
        o_pre = q[:1]
        lse_pre = torch.zeros((H, 1), dtype=torch.float32, device=q.device)
    o_self, lse_self = fa256.varlen_fwd256(
        q, kt, vt, None, meta["cq_self"], meta["ck_self"],
        meta["max_q_self"], meta["max_k_self"], softmax_scale,
        True, -1, -1, None, cfg,
    )
    if out is None:
        out = torch.empty_like(q)
    _welm_fa_split_merge_kernel[(total_q, H)](
        o_pre, lse_pre, o_self, lse_self, meta["base"], meta["stride"], meta["cnt"],
        sinks if sinks is not None else lse_self, out,
        o_pre.stride(0), o_pre.stride(1), lse_pre.stride(0),
        o_self.stride(0), o_self.stride(1), lse_self.stride(0),
        out.stride(0), out.stride(1),
        D=D, HAS_SINK=sinks is not None, num_warps=4,
    )
    return out
