import argparse

import torch

from sglang.kernels.ops.attention.fla.hcu.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update as triton_impl,
)
from sglang.kernels.ops.attention.fla.hcu.fused_sigmoid_gating_recurrent_cuda import (
    fused_sigmoid_gating_delta_rule_update_cuda_hcu as cuda_impl,
)


def make_inputs(batch: int, heads: int, pool: int, dtype: torch.dtype):
    torch.manual_seed(7)
    q = torch.randn(1, batch, heads, 128, device="cuda", dtype=dtype) * 0.1
    k = torch.randn_like(q) * 0.1
    v = torch.randn(1, batch, heads, 128, device="cuda", dtype=dtype) * 0.1
    a = torch.randn(batch, heads * 128, device="cuda", dtype=dtype) * 0.5 - 1
    b = torch.randn(batch, heads, device="cuda", dtype=dtype) * 0.5
    A_log = torch.randn(heads, device="cuda", dtype=torch.float32) * 0.2
    dt_bias = torch.randn(heads * 128, device="cuda", dtype=torch.float32) * 0.1
    state = torch.randn(pool, heads, 128, 128, device="cuda", dtype=torch.float32) * 0.01
    indices = torch.arange(batch, device="cuda", dtype=torch.int32)
    cu_seqlens = torch.arange(batch + 1, device="cuda", dtype=torch.int64)
    return q, k, v, a, b, A_log, dt_bias, state, indices, cu_seqlens


def bench(fn, warmup=20, repeat=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--lower-bound", type=float, default=-5.0)
    args = parser.parse_args()
    x = make_inputs(args.batch, args.heads, max(args.batch, 256), torch.bfloat16)
    q, k, v, a, b, A_log, dt_bias, state, indices, cu_seqlens = x

    state_ref = state.clone()
    state_cuda = state.clone()
    out_ref = triton_impl(
        A_log, a, dt_bias, 1.0, 20.0, q, k, v, b, state_ref, indices,
        cu_seqlens=cu_seqlens, is_kda=True, lower_bound=args.lower_bound,
        use_qk_l2norm_in_kernel=True,
    )
    out_cuda = cuda_impl(
        A_log, a, dt_bias, q, k, v, b, state_cuda, indices,
        lower_bound=args.lower_bound, use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    out_diff = (out_ref.float() - out_cuda.float()).abs().max().item()
    state_diff = (state_ref[: args.batch] - state_cuda[: args.batch]).abs().max().item()
    print(f"correctness out_max_abs={out_diff:.6g} state_max_abs={state_diff:.6g}")
    torch.testing.assert_close(out_cuda, out_ref, atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(state_cuda[: args.batch], state_ref[: args.batch], atol=2e-3, rtol=1e-2)

    def run_triton():
        triton_impl(
            A_log, a, dt_bias, 1.0, 20.0, q, k, v, b, state_ref, indices,
            cu_seqlens=cu_seqlens, is_kda=True, lower_bound=args.lower_bound,
            use_qk_l2norm_in_kernel=True,
        )

    def run_cuda():
        cuda_impl(
            A_log, a, dt_bias, q, k, v, b, state_cuda, indices,
            lower_bound=args.lower_bound, use_qk_l2norm_in_kernel=True,
        )

    triton_us = bench(run_triton, repeat=args.repeat)
    cuda_us = bench(run_cuda, repeat=args.repeat)
    print(f"triton_us={triton_us:.3f} cuda_us={cuda_us:.3f} speedup={triton_us / cuda_us:.3f}x")


if __name__ == "__main__":
    main()
