"""Benchmark backends used by the HCU BF16-to-FP32 GEMM selector."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.attention.dsv4.gemm import (
    _auto_dispatch_bf16_fp32,
    _get_aiter_tgemm,
    _linear_bf16_fp32_cublas,
)
from sglang.kernels.ops.attention.dsv4.hcu_linear_bf16_fp32 import (
    hcu_linear_bf16_fp32,
    hcu_linear_bf16_fp32_supported,
)
from sglang.test.ci.ci_register import register_hcu_ci

register_hcu_ci(est_time=60, suite="nightly-hcu-perf", nightly=True)


def _torch_impl(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x.float(), weight.float())


def _aiter_impl(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    tgemm = _get_aiter_tgemm()
    if tgemm is None:
        marker.skip("aiter.tuned_gemm is not installed")
    return tgemm.mm(x, weight, otype=x.dtype).float()


def _run_impl(impl: str, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if impl == "auto":
        return _auto_dispatch_bf16_fp32(x, weight)
    if impl == "sgl":
        if not hcu_linear_bf16_fp32_supported(x, weight):
            marker.skip("shape is outside the HCU kernel contract")
        return hcu_linear_bf16_fp32(x, weight)
    if impl == "torch":
        return _torch_impl(x, weight)
    if impl == "cublas":
        return _linear_bf16_fp32_cublas(x, weight)
    if impl == "aiter":
        return _aiter_impl(x, weight)
    raise ValueError(f"Unknown implementation: {impl}")


@marker.parametrize(
    "m",
    [
        1,
        2,
        4,
        8,
        16,
        32,
        48,
        64,
        65,
        128,
        256,
        512,
        768,
        1024,
        1280,
        1408,
        1536,
        1664,
        1792,
        1920,
        2048,
        2176,
        2304,
        2432,
        2560,
        2688,
        2816,
        3072,
        3584,
        4096,
    ],
    [1, 32, 64, 1024, 1408, 2048, 2816, 3072, 4096],
)
@marker.parametrize(
    "n,k",
    [
        (256, 4096),
        (512, 4096),
        (1024, 4096),
        (2048, 4096),
        (256, 6144),
        (512, 6144),
    ],
    [
        (256, 4096),
        (512, 4096),
        (1024, 4096),
        (2048, 4096),
        (512, 6144),
    ],
)
@marker.benchmark("impl", ["auto", "sgl", "torch", "cublas", "aiter"])
def benchmark(m: int, n: int, k: int, impl: str):
    x = create_random(m, k, dtype=torch.bfloat16)
    weight = create_random(n, k, dtype=torch.bfloat16)
    return marker.do_bench(
        _run_impl,
        input_args=(impl, x, weight),
        memory_args=(x, weight),
        graph_clone_args=(1, 2),
    )


if __name__ == "__main__":
    benchmark.run()
