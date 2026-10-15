# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: fused HIP rope_fp4_fake_quant vs Python reference on DCU."""

import sys

import torch
import triton
import triton.testing

import sgl_kernel
from sglang.kernels.ops.attention.dsv4.torch_quant import (
    fake_quant_compressed_kv,
    fake_quant_fp4,
)
from sglang.srt.layers.attention.dsv4.dsv41_sparse import rope_tail
from sglang.utils import is_in_ci

IS_CI = is_in_ci()
DEVICE = "cuda"
ROPE_DIM = 64

# In CI: one representative config to keep runtime short.
_batch_vals = [4] if IS_CI else [1, 4, 16, 64, 128, 256]

configs = [
    triton.testing.Benchmark(
        x_names=["B"],
        x_vals=_batch_vals,
        line_arg="provider",
        line_vals=["fused_hip", "python"],
        line_names=["Fused HIP", "Python fallback"],
        styles=[("green", "-"), ("red", "--")],
        ylabel="us (median)",
        plot_name="rope_fp4_d512_ue8m0",
        args={"D": 512, "compressed_kv": False},
    ),
    triton.testing.Benchmark(
        x_names=["B"],
        x_vals=_batch_vals,
        line_arg="provider",
        line_vals=["fused_hip", "python"],
        line_names=["Fused HIP", "Python fallback"],
        styles=[("green", "-"), ("red", "--")],
        ylabel="us (median)",
        plot_name="rope_fp4_d512_ckv",
        args={"D": 512, "compressed_kv": True},
    ),
    triton.testing.Benchmark(
        x_names=["B"],
        x_vals=_batch_vals,
        line_arg="provider",
        line_vals=["fused_hip", "python"],
        line_names=["Fused HIP", "Python fallback"],
        styles=[("green", "-"), ("red", "--")],
        ylabel="us (median)",
        plot_name="rope_fp4_d128_ue8m0",
        args={"D": 128, "compressed_kv": False},
    ),
]


@triton.testing.perf_report(configs)
def bench(B, D, compressed_kv, provider):
    x = torch.randn(B, D, dtype=torch.bfloat16, device=DEVICE)
    freqs = torch.polar(
        torch.ones(B, ROPE_DIM // 2, device=DEVICE),
        torch.rand(B, ROPE_DIM // 2, device=DEVICE) * 6.28,
    ).to(torch.complex64)
    quant = fake_quant_compressed_kv if compressed_kv else fake_quant_fp4

    if provider == "fused_hip":
        fn = lambda: sgl_kernel.rope_fp4_fake_quant(x, freqs, ROPE_DIM, compressed_kv)
    else:
        fn = lambda: quant(rope_tail(x, freqs, ROPE_DIM))

    ms = triton.testing.do_bench(fn)
    return ms * 1e3  # microseconds


if __name__ == "__main__":
    if torch.version.hip is None:
        print("Skipping: not a HIP/DCU device.", file=sys.stderr)
        sys.exit(0)
    bench.run(print_data=True)
