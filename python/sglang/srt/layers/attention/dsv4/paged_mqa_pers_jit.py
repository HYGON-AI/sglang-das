"""JIT loader for the persistent INT8 paged-MQA kernel.

See csrc/paged_mqa_pers_jit.cu. Enabled with
SGLANG_MQA_PERSISTENT=1 on the indexer path; the default is lightop
gemmopt.paged_mqa_logits.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch


@functools.cache
def _module():
    import torch.utils.cpp_extension

    source_path = Path(__file__).with_name("csrc") / "paged_mqa_pers_jit.cu"
    return torch.utils.cpp_extension.load_inline(
        name="sgl_paged_mqa_pers_jit",
        cpp_sources="",
        cuda_sources=source_path.read_text(),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-w", "-DUSE_ROCM=1"],
        verbose=False,
    )


def persistent_int8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_seq_len: int,
    num_sms: int = 320,
) -> torch.Tensor:
    """Run the persistent INT8 Paged MQA kernel for the HCU indexer path."""
    if max_seq_len <= 0 or max_seq_len % 64 != 0:
        raise ValueError("persistent Paged MQA requires a positive 64-token width")
    if q.dim() == 4:
        q = q.reshape(q.shape[0], q.shape[-2], q.shape[-1])
    return _module().persistent(
        q,
        kv_cache,
        weights,
        seq_lens,
        block_table,
        int(max_seq_len),
        int(num_sms),
        0,
    )
