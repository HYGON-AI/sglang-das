"""Compatibility entry point for the row-wise speculative argmax kernel.

HCU keeps the existing DSpark implementation, which has the same reduction and
tie-breaking semantics as the community ``row_argmax`` helper.
"""

import torch

from sglang.kernels.ops.speculative.dspark.fast_argmax import fast_row_argmax


def row_argmax(x: torch.Tensor) -> torch.Tensor:
    return fast_row_argmax(x)
