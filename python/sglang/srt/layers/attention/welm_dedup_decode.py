"""Prefix-deduplicated decode attention for WeLM on HCU (legacy KV layout).

Decode rows that share a cached prefix hold the same physical KV pages (radix
cache). Each row's keys are cut into chunks of CHUNK pages; a full chunk is
identified by the id of its last page (equal page at equal table position ==
equal prefix up to it). Rows holding the same chunk are grouped (up to MB per
group) into one virtual sequence whose MB query rows run as the vendor paged
kernel's multi-token (mtp) queries, so the chunk's KV is read once per group.
A partial-output build of the vendor kernel (welm_pa_part.so) writes per
partition (normalized out, max, exp_sum); a merge kernel combines each row's
chunks and the attention sink.

Everything is device-side with static shapes (CUDA-graph safe).
"""
import importlib.util
import os

import torch
import triton
import triton.language as tl

CHUNK = 8      # pages per chunk (512 tokens)
MB = 5         # rows per virtual sequence (MB * 6 q-heads <= 64)
MAX_B = 64     # dedup batch-size limit (grouping is O(B^2) per chunk)
_OUT_OFF = 10_000_000   # vendor tmp layout (paged_attention_hip.h)
_MAX_OFF = 5_000_000

_mod = None
_bufs = {}


class _VendorPA:
    @staticmethod
    def pa_partial(*args):
        from flash_attn.flash_attn_interface import flash_attn_cuda

        return flash_attn_cuda.paged_attention_partial(*args)


def _pa():
    global _mod
    if _mod is None and os.environ.get("WELM_VENDOR_FA", "auto").lower() in ("auto", "1", "true"):
        try:
            from flash_attn.flash_attn_interface import flash_attn_cuda

            if hasattr(flash_attn_cuda, "paged_attention_partial"):
                _mod = _VendorPA
        except Exception:
            pass
    if _mod is None:
        so = os.path.join(os.path.dirname(__file__), "welm_pa_part.so")
        spec = importlib.util.spec_from_file_location("welm_pa_part", so)
        _mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_mod)
    return _mod


def available() -> bool:
    try:
        _pa()
        return True
    except Exception:
        return False


@triton.jit
def _dedup_group_kernel(
    BT, SEQK, CNT, BT_V, SEQ_V, MEM_V, MAP, B,
    stride_bt, stride_map,
    CHUNK: tl.constexpr, PAGE: tl.constexpr, MB: tl.constexpr, B_PAD: tl.constexpr,
    MB_PAD: tl.constexpr,
):
    c = tl.program_id(0)
    CT: tl.constexpr = CHUNK * PAGE
    r = tl.arange(0, B_PAD)
    in_b = r < B
    lens = tl.load(SEQK + r, mask=in_b, other=0)
    lo = c * CT
    has = in_b & (lo < lens)
    full = has & (lo + CT <= lens)
    key = tl.load(BT + r * stride_bt + (c + 1) * CHUNK - 1, mask=full, other=0)
    key = tl.where(full, key, -(r + 1))
    # pairwise: E[a, b] = rows a, b hold the same chunk
    E = (key[:, None] == key[None, :]) & has[:, None] & has[None, :]
    rank = tl.sum((E & (r[None, :] < r[:, None])).to(tl.int32), axis=1)
    owner = has & ((rank % MB) == 0)
    vid = tl.atomic_add(CNT + r * 0, owner.to(tl.int32), mask=owner)
    vid = tl.where(owner, vid, -1)
    # virtual sequence = the owner's chunk pages
    pg = tl.arange(0, CHUNK)
    bt_rows = tl.load(BT + r[:, None] * stride_bt + c * CHUNK + pg[None, :], mask=owner[:, None], other=0)
    tl.store(BT_V + vid[:, None] * CHUNK + pg[None, :], bt_rows, mask=owner[:, None])
    tl.store(SEQ_V + vid, tl.minimum(lens - lo, CT), mask=owner)
    # members of owner a: rows b with E[a, b] and rank[b] - rank[a] in [0, MB)
    j = tl.arange(0, MB_PAD)
    rel = rank[None, :] - rank[:, None]                                  # [a, b]
    mem3 = E[:, :, None] & (rel[:, :, None] == j[None, None, :])          # [a, b, j]
    memrow = tl.sum(tl.where(mem3, r[None, :, None] + 1, 0), axis=1) - 1  # [a, j]
    tl.store(MEM_V + vid[:, None] * MB + j[None, :], memrow,
             mask=owner[:, None] & (j[None, :] < MB))
    # map[b, c] = vid(owner of b) * MB + rank[b] % MB
    own_of = E & owner[:, None] & (rank[:, None] == (rank[None, :] // MB) * MB)   # [a, b]
    vid_b = tl.sum(tl.where(own_of, vid[:, None] + 1, 0), axis=0) - 1
    mp = tl.where(has, vid_b * MB + rank % MB, -1)
    tl.store(MAP + r * stride_map + c, mp, mask=in_b)


@triton.jit
def _dedup_merge_kernel(
    BUF_F, BUF_H, MAP, SEQ_V, SEQK, SINKS, OUT,
    stride_map, stride_ob, stride_oh,
    H: tl.constexpr, D: tl.constexpr, CT: tl.constexpr, PS: tl.constexpr,
    MAXP: tl.constexpr, MB: tl.constexpr, NC_BLOCK: tl.constexpr, HAS_SINK: tl.constexpr,
    MAX_OFF_F: tl.constexpr, OUT_OFF: tl.constexpr, DB: tl.constexpr,
):
    r = tl.program_id(0)
    h = tl.program_id(1)
    NH: tl.constexpr = MB * H
    L = tl.load(SEQK + r)
    nch = tl.cdiv(L, CT)
    offs_d = tl.program_id(2) * DB + tl.arange(0, DB)
    e = tl.arange(0, NC_BLOCK * MAXP)
    m_run = tl.full((), float("-inf"), tl.float32)
    den = tl.zeros((), tl.float32)
    acc = tl.zeros([DB], tl.float32)
    for c0 in range(0, nch, NC_BLOCK):
        c = c0 + e // MAXP
        p = e % MAXP
        okc = c < nch
        idx = tl.load(MAP + r * stride_map + c, mask=okc, other=0)
        vid = idx // MB
        slot = idx % MB
        Lv = tl.load(SEQ_V + vid, mask=okc, other=0)
        ok = okc & (p < tl.cdiv(Lv, PS))
        off = (vid.to(tl.int64) * NH + slot * H + h) * MAXP + p
        mx = tl.load(BUF_F + MAX_OFF_F + off, mask=ok, other=float("-inf"))
        es = tl.load(BUF_F + off, mask=ok, other=0.0)
        m_new = tl.maximum(m_run, tl.max(mx, 0))
        a = tl.exp2(m_run - m_new)
        w = es * tl.exp2(mx - m_new)
        w = tl.where(ok, w, 0.0)
        o = tl.load(BUF_H + OUT_OFF + off[:, None] * D + offs_d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        acc = acc * a + tl.sum(o * w[:, None], 0)
        den = den * a + tl.sum(w, 0)
        m_run = m_new
    if HAS_SINK:
        s = tl.load(SINKS + h).to(tl.float32) * 1.4426950408889634
        m_new = tl.maximum(m_run, s)
        a = tl.exp2(m_run - m_new)
        acc = acc * a
        den = den * a + tl.exp2(s - m_new)
    tl.store(OUT + r * stride_ob + h * stride_oh + offs_d, (acc / den).to(OUT.dtype.element_ty))


_tmp = {}


def _shared_tmp(device, numel):
    """One partial-output workspace per device, sized for the largest batch
    (only one forward runs at a time; every batch size reuses it)."""
    t = _tmp.get(device)
    if t is None or t.numel() < numel:
        need = max(numel, _OUT_OFF + MAX_B * 64 * MB * 6 * 2 * 256)
        t = torch.empty(need, dtype=torch.bfloat16, device=device)
        _tmp[device] = t
    return t


def _get_bufs(device, B, nc):
    key = (device, B, nc)
    b = _bufs.get(key)
    if b is None:
        nv = B * nc
        NH = MB * 6
        PS = 256 if NH > 15 else 512
        maxp = (CHUNK * 64 + PS - 1) // PS
        b = dict(
            cnt=torch.zeros(1, dtype=torch.int32, device=device),
            bt_v=torch.zeros((nv, CHUNK), dtype=torch.int32, device=device),
            seq_v=torch.zeros(nv, dtype=torch.int32, device=device),
            mem_v=torch.full((nv, MB), -1, dtype=torch.int32, device=device),
            map=torch.zeros((B, nc), dtype=torch.int32, device=device),
            tmp=_shared_tmp(device, _OUT_OFF + nv * NH * maxp * 256),
            PS=PS, maxp=maxp, nv=nv,
        )
        _bufs[key] = b
    return b


def prepare(block_table, seq_lens, max_context_len, page_size, B):
    """Group the batch's chunks into virtual sequences (once per forward)."""
    CT = CHUNK * page_size
    nc = triton.cdiv(max_context_len, CT)
    b = _get_bufs(block_table.device, B, nc)
    b["cnt"].zero_()
    b["seq_v"].zero_()
    _dedup_group_kernel[(nc,)](
        block_table, seq_lens, b["cnt"], b["bt_v"], b["seq_v"], b["mem_v"], b["map"], B,
        block_table.stride(0), b["map"].stride(0),
        CHUNK=CHUNK, PAGE=page_size, MB=MB, B_PAD=max(16, triton.next_power_of_2(B)),
        MB_PAD=triton.next_power_of_2(MB), num_warps=4,
    )
    return b


def get_prepared(max_context_len, page_size, B, device):
    """Buffers grouped by prepare() earlier in this forward."""
    return _get_bufs(device, B, triton.cdiv(max_context_len, CHUNK * page_size))


def attend(q, k_cache, v_cache, seq_lens, b, sinks=None, out=None):
    B, H, D = q.shape
    CT = CHUNK * k_cache.shape[2]
    _pa().pa_partial(b["tmp"], q, k_cache, v_cache, b["bt_v"], b["seq_v"], MB, b["PS"], b["maxp"],
                     b["mem_v"])
    if out is None:
        out = torch.empty_like(q)
    DB = 64
    _dedup_merge_kernel[(B, H, D // DB)](
        b["tmp"].view(torch.float32), b["tmp"], b["map"], b["seq_v"], seq_lens,
        sinks if sinks is not None else seq_lens, out,
        b["map"].stride(0), out.stride(0), out.stride(1),
        H=H, D=D, CT=CT, PS=b["PS"], MAXP=b["maxp"], MB=MB, NC_BLOCK=16,
        HAS_SINK=sinks is not None, MAX_OFF_F=_MAX_OFF // 2, OUT_OFF=_OUT_OFF, DB=DB,
        num_warps=2,
    )
    return out


def welm_dedup_decode_attention(q, k_cache, v_cache, block_table, seq_lens, max_context_len,
                                sinks=None, out=None):
    """q [B, H=6, D=256] (pre-scaled to the default 1/sqrt(D)) -> out [B, H, D]."""
    b = prepare(block_table, seq_lens, max_context_len, k_cache.shape[2], q.shape[0])
    return attend(q, k_cache, v_cache, seq_lens, b, sinks=sinks, out=out)
