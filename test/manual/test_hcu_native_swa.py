"""HCU-only numerical and graph regression tests for the native SWA adapter."""

import unittest
import torch
from sglang.srt.layers.attention.hcu_native_swa import native_hcu_sliding_attention
from sglang.srt.utils import is_hcu


def make_case(lengths, qlen=8, window=4095):
    torch.manual_seed(7)
    bs = len(lengths)
    hq = 16
    hk = 4
    d = 128
    pages = (max(lengths) + 63) // 64
    k = (torch.randn(bs * pages, hk, 64, d, device="cuda") * 0.2).to(torch.float8_e5m2)
    v = (torch.randn(bs * pages, hk, d, 64, device="cuda") * 0.2).to(torch.float8_e5m2)
    q = torch.randn(bs * qlen, hq, d, device="cuda", dtype=torch.bfloat16) * 0.2
    table = torch.randperm(bs * pages, device="cuda").to(torch.int32).view(bs, pages)
    seq = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    cuq = torch.arange(bs + 1, device="cuda", dtype=torch.int32) * qlen
    kw = dict(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cuq,
        max_seqlen_q=qlen,
        seqused_k=seq,
        window_size=(window, window),
        block_table=table,
        softmax_scale=d**-0.5,
        causal=False,
        q_descale=torch.ones(1, device="cuda"),
        k_descale=torch.ones(1, device="cuda"),
        v_descale=torch.ones(1, device="cuda"),
    )
    return kw


def reference(kw):
    rows = []
    q = kw["q"]
    qlen = kw["max_seqlen_q"]
    for b, n in enumerate(kw["seqused_k"].tolist()):
        ix = kw["block_table"][b].long()
        k = (
            kw["k"][ix]
            .permute(0, 2, 1, 3)
            .reshape(-1, 4, 128)[:n]
            .float()
            .repeat_interleave(4, 1)
        )
        v = (
            kw["v"][ix]
            .permute(0, 3, 1, 2)
            .reshape(-1, 4, 128)[:n]
            .float()
            .repeat_interleave(4, 1)
        )

        def scale_row(name):
            scale = kw[name]
            if scale is None:
                return torch.ones(16, device=q.device)
            scale = scale.float()
            row = (
                scale[b] if scale.ndim == 2 and scale.shape[0] > 1 else scale.flatten()
            )
            return row.repeat_interleave(16 // row.numel())

        k *= scale_row("k_descale")[:, None]
        v *= scale_row("v_descale")[:, None]
        query = q[b * qlen : (b + 1) * qlen].float()
        query *= scale_row("q_descale")[:, None]
        logits = torch.einsum("qhd,khd->hqk", query, k) * kw["softmax_scale"]
        qpos = n - qlen + torch.arange(qlen, device=q.device)
        kpos = torch.arange(n, device=q.device)
        mask = kpos[None] >= qpos[:, None] - kw["window_size"][0]
        mask &= kpos[None] <= qpos[:, None] + kw["window_size"][1]
        if kw["causal"]:
            mask &= kpos[None] <= qpos[:, None]
        logits.masked_fill_(~mask[None], -float("inf"))
        rows.append(
            torch.einsum("hqk,khd->qhd", logits.softmax(-1).nan_to_num(), v).to(q.dtype)
        )
    return torch.cat(rows)


@unittest.skipUnless(is_hcu(), "Requires HCU FlashAttention")
class NativeSWATest(unittest.TestCase):
    def test_numerics(self):
        for lengths in ([1, 64], [128], [4128], [21021], [128, 21021]):
            for causal in (False, True):
                with self.subTest(lengths=lengths, causal=causal):
                    kw = make_case(lengths)
                    kw["causal"] = causal
                    ref = reference(kw)
                    out = torch.full_like(ref, 77)
                    got = native_hcu_sliding_attention(**kw, out=out)
                    self.assertEqual(got.data_ptr(), out.data_ptr())
                    torch.testing.assert_close(got, ref, atol=4e-4, rtol=0.03)

    def test_scales_and_query_sizes(self):
        for qlen in (1, 4, 16):
            with self.subTest(qlen=qlen):
                kw = make_case([128, 4160], qlen=qlen)
                kw["q_descale"] = torch.tensor(
                    [[0.5, 1, 1.5, 2], [2, 1.5, 1, 0.5]], device="cuda"
                )
                kw["k_descale"] = torch.tensor([0.5, 1, 1.5, 2], device="cuda")
                kw["v_descale"] = torch.tensor([1.5], device="cuda")
                torch.testing.assert_close(
                    native_hcu_sliding_attention(**kw),
                    reference(kw),
                    atol=5e-4,
                    rtol=0.03,
                )

    def test_graph_live_lengths(self):
        kw = make_case([128, 21021])
        out = torch.empty_like(kw["q"])
        for _ in range(3):
            native_hcu_sliding_attention(**kw, out=out)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            native_hcu_sliding_attention(**kw, out=out)
        for lengths in ([128, 21021], [64, 8192], [512, 4160]):
            kw["seqused_k"].copy_(
                torch.tensor(lengths, device="cuda", dtype=torch.int32)
            )
            g.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(out, reference(kw), atol=4e-4, rtol=0.03)


if __name__ == "__main__":
    unittest.main()
