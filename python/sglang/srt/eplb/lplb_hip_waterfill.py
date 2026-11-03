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
"""On-device LPLB replica split for HIP: water-filling in one Triton program.

The host solver reads the all-reduced counts back every MoE layer, and that
stream sync drains the launch queue: on DSV4.1 PD prefill the GPU then idles
~120 ms per 32K step, more than the MegaMoE time the split saves. This kernel
keeps the whole solve on the stream.

Same LP as LPLBSolver: single-copy experts stay on their rank; each replicated
expert's tokens are split over its replica ranks to minimise the peak rank
load. Gauss-Seidel water-filling over the replicated experts in a fixed order;
each update pours the expert into its ranks at the level
min_j (c + sum of loads <= l_j) / #(loads <= l_j), so no sort is needed.
4 passes land within ~0.2% of the LP optimum on recorded DSV4.1 routing.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _lplb_waterfill_kernel(
    counts_ptr,  # (num_logical,) float32, all-reduced
    single_log_ptr,  # (S,) int32
    single_rank_ptr,  # (S,) int32
    rep_log_ptr,  # (R,) int32
    rep_rank_ptr,  # (R, M) int32, -1 padded distinct replica ranks
    rep_copies_ptr,  # (R, M) float32, replicas on that rank (1 for padding)
    rep_colrank_ptr,  # (R, C) int32, rank slot of each log2phy column, -1 padded
    share_ptr,  # (R, M) float32 scratch
    out_ptr,  # (num_logical, out_stride) float32, pre-filled with the even split
    num_logical,
    S,
    R,
    out_stride,
    min_pairs,
    PASSES: tl.constexpr,
    G: tl.constexpr,
    M: tl.constexpr,
    C: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    offs_e = tl.arange(0, BLOCK_E)
    total = tl.sum(tl.load(counts_ptr + offs_e, mask=offs_e < num_logical, other=0.0))
    if total >= min_pairs:
        g = tl.arange(0, G)
        offs_s = tl.arange(0, BLOCK_S)
        ms = offs_s < S
        s_log = tl.load(single_log_ptr + offs_s, mask=ms, other=0)
        s_rank = tl.load(single_rank_ptr + offs_s, mask=ms, other=-1)
        s_cnt = tl.load(counts_ptr + s_log, mask=ms, other=0.0)
        load = tl.sum(tl.where(s_rank[:, None] == g[None, :], s_cnt[:, None], 0.0), axis=0)

        offs_m = tl.arange(0, M)
        for i in range(R):
            tl.store(share_ptr + i * M + offs_m, tl.zeros([M], dtype=tl.float32))
        for _ in tl.static_range(PASSES):
            for i in range(R):
                rk = tl.load(rep_rank_ptr + i * M + offs_m)
                valid = rk >= 0
                mass = tl.load(counts_ptr + tl.load(rep_log_ptr + i))
                sh = tl.load(share_ptr + i * M + offs_m)
                onehot = rk[:, None] == g[None, :]
                rest = tl.sum(tl.where(onehot, load[None, :], 0.0), axis=1) - sh
                rest_v = tl.where(valid, rest, float("inf"))
                below = (rest_v[None, :] <= rest_v[:, None]) & valid[None, :]
                num = mass + tl.sum(tl.where(below, rest_v[None, :], 0.0), axis=1)
                den = tl.sum(below.to(tl.float32), axis=1)
                level = tl.min(tl.where(valid, num / tl.maximum(den, 1.0), float("inf")), axis=0)
                new = tl.where(valid, tl.maximum(level - rest, 0.0), 0.0)
                load = load + tl.sum(tl.where(onehot, (new - sh)[:, None], 0.0), axis=0)
                tl.store(share_ptr + i * M + offs_m, new)

        offs_c = tl.arange(0, C)
        for i in range(R):
            e = tl.load(rep_log_ptr + i)
            mass = tl.load(counts_ptr + e)
            if mass > 0:
                col = tl.load(rep_colrank_ptr + i * C + offs_c)
                sh = tl.load(share_ptr + i * M + offs_m)
                per_rank = sh / tl.load(rep_copies_ptr + i * M + offs_m)
                val = tl.sum(tl.where(col[:, None] == offs_m[None, :], per_rank[None, :], 0.0), axis=1)
                # Normalise by the poured total, not by mass: level - rest
                # cancels rank loads of ~1e4 pairs in fp32, which leaves a
                # small expert's shares off its count by up to ~0.5%.
                tl.store(out_ptr + e * out_stride + offs_c, val / tl.sum(sh), mask=col >= 0)


class DeviceWaterfill:
    """Per-layer tables for _lplb_waterfill_kernel; solve() never syncs."""

    def __init__(self, phy2log: torch.Tensor, log2phy: torch.Tensor, num_gpus: int, passes: int, min_pairs: int):
        device = phy2log.device
        p2l = phy2log.cpu().to(torch.int64)
        l2p = log2phy.cpu().to(torch.int64)
        local = p2l.numel() // num_gpus
        valid = l2p >= 0
        num_valid = valid.sum(-1)
        self.uniform = (valid.float() / num_valid.clamp_min(1).unsqueeze(-1)).to(device)
        single = torch.nonzero(num_valid == 1).flatten()
        rep = torch.nonzero(num_valid > 1).flatten()
        self.num_logical = l2p.shape[0]
        self.G = num_gpus
        self.M = num_gpus
        self.C = triton.next_power_of_2(max(int(num_valid.max()), 1))
        self.R = rep.numel()
        rep_rank = torch.full((max(self.R, 1), self.M), -1, dtype=torch.int32)
        rep_copies = torch.ones((max(self.R, 1), self.M), dtype=torch.float32)
        rep_colrank = torch.full((max(self.R, 1), self.C), -1, dtype=torch.int32)
        for i, e in enumerate(rep.tolist()):
            col_rank = (l2p[e, : int(num_valid[e])] // local).tolist()
            ranks = sorted(set(col_rank))
            rep_rank[i, : len(ranks)] = torch.tensor(ranks, dtype=torch.int32)
            rep_copies[i, : len(ranks)] = torch.tensor([col_rank.count(r) for r in ranks], dtype=torch.float32)
            rep_colrank[i, : len(col_rank)] = torch.tensor([ranks.index(r) for r in col_rank], dtype=torch.int32)
        self.single_log = single.to(torch.int32).to(device)
        self.single_rank = (l2p[single, 0] // local).to(torch.int32).to(device)
        self.rep_log = rep.to(torch.int32).to(device) if self.R else torch.zeros(1, dtype=torch.int32, device=device)
        self.rep_rank = rep_rank.to(device)
        self.rep_copies = rep_copies.to(device)
        self.rep_colrank = rep_colrank.to(device)
        self.share = torch.zeros((max(self.R, 1), self.M), dtype=torch.float32, device=device)
        self.passes = passes
        self.min_pairs = float(min_pairs)

    def solve(self, global_counts: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        out.copy_(self.uniform)
        if self.R == 0:
            return out
        _lplb_waterfill_kernel[(1,)](
            global_counts.float(),
            self.single_log,
            self.single_rank,
            self.rep_log,
            self.rep_rank,
            self.rep_copies,
            self.rep_colrank,
            self.share,
            out,
            self.num_logical,
            self.single_log.numel(),
            self.R,
            out.stride(0),
            self.min_pairs,
            PASSES=self.passes,
            G=self.G,
            M=self.M,
            C=self.C,
            BLOCK_E=triton.next_power_of_2(self.num_logical),
            BLOCK_S=triton.next_power_of_2(max(self.single_log.numel(), 1)),
            num_warps=1,
        )
        return out
