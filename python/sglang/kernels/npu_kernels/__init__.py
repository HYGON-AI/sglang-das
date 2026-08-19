"""NPU/DCU Triton kernels (Ascend-op compatible).

Prefer importing symbols from submodules directly, e.g.::

    from sglang.kernels.npu_kernels.npu_grouped_matmul_triton import (
        grouped_matmul_triton,
    )

Or via this package (lazy)::

    from sglang.kernels import npu_kernels
    npu_kernels.grouped_matmul_triton(...)
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "grouped_matmul_triton",
    "npu_quant_matmul_triton",
    "npu_dynamic_quant_triton",
    "npu_dequant_swiglu_quant_triton",
    "npu_moe_init_routing_v2_triton",
    "npu_moe_finalize_routing_triton",
]

_LAZY_ATTRS = {
    "grouped_matmul_triton": (
        "sglang.kernels.npu_kernels.npu_grouped_matmul_triton",
        "grouped_matmul_triton",
    ),
    "npu_quant_matmul_triton": (
        "sglang.kernels.npu_kernels.npu_quant_matmul_triton",
        "npu_quant_matmul_triton",
    ),
    "npu_dynamic_quant_triton": (
        "sglang.kernels.npu_kernels.npu_dynamic_quant_triton",
        "npu_dynamic_quant_triton",
    ),
    "npu_dequant_swiglu_quant_triton": (
        "sglang.kernels.npu_kernels.npu_dequant_swiglu_quant_triton",
        "npu_dequant_swiglu_quant_triton",
    ),
    "npu_moe_init_routing_v2_triton": (
        "sglang.kernels.npu_kernels.npu_moe_init_routing_v2_triton",
        "npu_moe_init_routing_v2_triton",
    ),
    "npu_moe_finalize_routing_triton": (
        "sglang.kernels.npu_kernels.npu_moe_finalize_routing_triton",
        "npu_moe_finalize_routing_triton",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _LAZY_ATTRS[name]
    import importlib

    mod = importlib.import_module(module_name)
    value = getattr(mod, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)