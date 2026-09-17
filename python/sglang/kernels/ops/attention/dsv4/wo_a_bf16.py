"""Compatibility exports for the split HCU WO-A BF16 kernels.

Upstream consolidated these kernels into this module.  The HCU tree keeps the
single-token and small-batch implementations in separate files, so expose the
same public API without replacing the platform-tuned implementations.
"""

from .wo_a_bf16_gemv import wo_a_bf16_gemv
from .wo_a_bf16_small_batch import (
    wo_a_bf16_small_batch,
    wo_a_bf16_small_batch_mxfp8,
)

__all__ = [
    "wo_a_bf16_gemv",
    "wo_a_bf16_small_batch",
    "wo_a_bf16_small_batch_mxfp8",
]
