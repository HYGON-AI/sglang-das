from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.utils import is_hcu

from .utils import make_name

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_SUPPORTED_N = frozenset((256, 512, 1024, 2048))
_MAX_INT32 = (1 << 31) - 1


@cache_once
def _jit_linear_bf16_fp32_module() -> Module:
    if not is_hcu():
        raise RuntimeError(
            "The HCU BF16-to-FP32 GEMM requires gfx936, gfx938, or gfx928"
        )
    return load_jit(
        make_name("linear_bf16_fp32_hcu"),
        cuda_files=["deepseek_v4/linear_bf16_fp32.cuh"],
        cuda_wrappers=[("run", "LinearBf16Fp32Kernel::run")],
    )


def hcu_linear_bf16_fp32_supported(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    if x.dim() != 2 or weight.dim() != 2:
        return False
    m, k = x.shape
    n, weight_k = weight.shape
    return (
        is_hcu()
        and 0 < m <= 64
        and n in _SUPPORTED_N
        and k == weight_k
        and k > 0
        and k % 128 == 0
        and m * k <= _MAX_INT32
        and n * k <= _MAX_INT32
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
    )


def _uses_splitk(*, m: int, n: int, k: int) -> bool:
    return k % 512 == 0 and (n == 1024 or (n == 2048 and m >= 8))


def hcu_linear_bf16_fp32(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if not hcu_linear_bf16_fp32_supported(x, weight):
        raise ValueError(
            "Unsupported HCU BF16-to-FP32 GEMM input; expected contiguous BF16 "
            "[M, K] x [N, K] with M in [1, 64], N in "
            "{256, 512, 1024, 2048}, and K divisible by 128"
        )

    m, k = x.shape
    n = weight.shape[0]
    if _uses_splitk(m=m, n=n, k=k):
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
    else:
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    _jit_linear_bf16_fp32_module().run(out, x, weight)
    return out
