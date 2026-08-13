import argparse

import torch
from sgl_kernel import concat_mla_absorb_q

from sglang.benchmark.bench_utils import run_bench


def benchmark(dim_0: int, dim_1: int):
    q_nope = torch.randn(dim_0, dim_1, 512, device="cuda", dtype=torch.bfloat16)
    q_rope = torch.randn(dim_0, dim_1, 64, device="cuda", dtype=torch.bfloat16)

    custom_ms, _, _ = run_bench(lambda: concat_mla_absorb_q(q_nope, q_rope))
    torch_ms, _, _ = run_bench(lambda: torch.cat((q_nope, q_rope), dim=-1))
    print(
        f"shape=({dim_0}, {dim_1}) custom={custom_ms:.6f} ms "
        f"torch={torch_ms:.6f} ms speedup={torch_ms / custom_ms:.2f}x"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim-0", type=int, default=16)
    parser.add_argument("--dim-1", type=int, default=128)
    args = parser.parse_args()
    benchmark(args.dim_0, args.dim_1)
