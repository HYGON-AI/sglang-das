from __future__ import annotations

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.layernorm.mhc_pre_splitk_triton import mhc_pre_partial_splitk
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=8, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)
register_amd_ci(est_time=8, stage="jit-kernel-benchmark", runner_config="amd")

# hc_mult = 4, hidden_size = 4096 -> K = 16384, N = 4 * 2 + 4 * 4 = 24
HC_HIDDEN_SIZE = 16384
HC_MULT3 = 24


def reference(x: torch.Tensor, fn: torch.Tensor):
    return x.float() @ fn.t(), (x.float() ** 2).sum(-1)


IMPL_MAP = {
    "split3": lambda x, fn: mhc_pre_partial_splitk(x, fn, variant="split3"),
    "ieee": lambda x, fn: mhc_pre_partial_splitk(x, fn, variant="ieee"),
    "torch": reference,
}


@marker.parametrize("num_tokens", [2048, 8192, 32768], [8192])
@marker.benchmark("impl", ["split3", "ieee", "torch"])
def benchmark(num_tokens: int, impl: str):
    torch.manual_seed(0)
    x = torch.randn(
        num_tokens, HC_HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    ) * 0.1
    fn = torch.randn(HC_MULT3, HC_HIDDEN_SIZE, device="cuda", dtype=torch.float32) * 0.01

    if impl != "torch":
        out, sqrsum = IMPL_MAP[impl](x, fn)
        ref_out, ref_sqrsum = reference(x, fn)
        # split3 rebuilds the FP32 weight from three BF16 terms, so it matches
        # the IEEE path only to that reconstruction's precision.
        rtol = 2e-2 if impl == "split3" else 1e-3
        torch.testing.assert_close(out.sum(0), ref_out, rtol=rtol, atol=rtol)
        torch.testing.assert_close(sqrsum.sum(0), ref_sqrsum, rtol=1e-3, atol=1e-3)

    return marker.do_bench(
        IMPL_MAP[impl],
        input_args=(x, fn),
        memory_args=(x, fn),
        memory_output="out",
    )


if __name__ == "__main__":
    benchmark.run()
