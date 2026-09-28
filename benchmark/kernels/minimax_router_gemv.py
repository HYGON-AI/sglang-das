#!/usr/bin/env python3
"""Correctness and latency sweep for the MiniMax-M3 small-M router GEMV."""

import argparse

import torch
import triton

from sglang.kernels.ops.moe.minimax_router_gemv import minimax_router_gemv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--block-k", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--num-warps", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    weight = torch.randn((128, 6144), device="cuda", dtype=torch.bfloat16) / 64
    print("M  backend                 latency_us  max_abs_err  max_rel_err")
    for m in args.batch_sizes:
        hidden = torch.randn((m, 6144), device="cuda", dtype=torch.bfloat16) / 64
        reference = torch.mm(hidden, weight.t(), out_dtype=torch.float32)
        baseline_ms = triton.testing.do_bench(
            lambda: torch.mm(hidden, weight.t(), out_dtype=torch.float32),
            warmup=args.warmup,
            rep=args.rep,
        )
        print(
            f"{m:<2} torch.mm                {baseline_ms * 1000:10.3f}  "
            "0.000000e+00  0.000000e+00"
        )
        for block_k in args.block_k:
            for num_warps in args.num_warps:
                output = minimax_router_gemv(
                    hidden, weight, block_k=block_k, num_warps=num_warps
                )
                torch.cuda.synchronize()
                abs_err = (output - reference).abs()
                rel_err = abs_err / reference.abs().clamp_min(1e-6)
                latency_ms = triton.testing.do_bench(
                    lambda: minimax_router_gemv(
                        hidden,
                        weight,
                        block_k=block_k,
                        num_warps=num_warps,
                    ),
                    warmup=args.warmup,
                    rep=args.rep,
                )
                label = f"bk{block_k}-w{num_warps}"
                print(
                    f"{m:<2} {label:<22} {latency_ms * 1000:10.3f}  "
                    f"{abs_err.max().item():.6e}  {rel_err.max().item():.6e}"
                )


if __name__ == "__main__":
    main()
