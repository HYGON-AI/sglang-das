"""Memory budget for the fp32 MQA-logits scratch of the DeepSeek sparse
attention indexers (DSA, DSV4, DSV4.1).

The indexer scores every query row against every key column into one fp32
logits matrix that no pool sized by mem_fraction_static accounts for. These
helpers bound that matrix and slice it by query rows.
"""

from __future__ import annotations

import functools
import math
import os
from typing import Callable, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_schedule
from sglang.srt.utils import get_device_module, is_hip, is_xpu
from sglang.srt.utils.common import ceil_div

MQA_LOGITS_BYTES_PER_ELEM = 4
MQA_LOGITS_STATIC_SKIP_ELEMS = 8_000_000
MQA_LOGITS_TOTAL_MEM_FRACTION = 0.3
# aiter's fp8_mqa_logits only compiles below 2 GiB of logits (buffer_store).
MQA_LOGITS_MAX_BYTES_ROCM = 2**31 - 1
# DeepGEMM pads the logits row stride to 1024 bytes, i.e. 256 fp32 columns.
MQA_LOGITS_ROW_ALIGN_ELEMS = 256


@functools.cache
def _dsv41_mqa_logits_budget_bytes(
    device_index: int, fraction: float, mem_fraction_static: Optional[float]
) -> int:
    # Query once per configuration, not once per layer: mem_get_info can sync.
    free, total = mqa_logits_free_mem_bytes(device_index, include_reusable_cache=True)
    guard = int(total * MQA_LOGITS_TOTAL_MEM_FRACTION)
    if mem_fraction_static is not None:
        guard = min(guard, int(total * max(0.0, 1.0 - mem_fraction_static) * fraction))
    return max(1, min(int(free * fraction), guard))


def dsv41_mqa_logits_rows_per_chunk(
    *, num_rows: int, num_cols: int, device_index: int
) -> Optional[int]:
    """Opt-in LightOp dense FP32 row budget; num_cols includes native padding.

    The V4.1 variable is independent of the DSA fraction. No device queries
    occur with the variable unset, during capture, or for small matrices.
    A one-row minimum is not a hard bound on total process/scratch memory.
    """
    name = "SGLANG_DSV41_PAGED_MQA_LOGITS_FREE_MEM_FRACTION"
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        fraction = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be finite and in (0, 1], got {raw!r}") from exc
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError(f"{name} must be finite and in (0, 1], got {raw!r}")
    if torch.cuda.is_current_stream_capturing() or not mqa_logits_needs_budget_check(
        num_rows=num_rows, num_cols=num_cols
    ):
        return None
    budget = _dsv41_mqa_logits_budget_bytes(
        device_index, fraction, get_schedule().mem_fraction_static
    )
    # Unlike DeepGEMM/aiter, this LightOp output has no 256-column stride pad.
    return mqa_logits_rows_per_chunk(
        num_rows=num_rows,
        row_bytes=num_cols * MQA_LOGITS_BYTES_PER_ELEM,
        budget_bytes=budget,
    )


def mqa_logits_free_mem_fraction() -> float:
    return envs.SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION.get()


def mqa_logits_needs_budget_check(*, num_rows: int, num_cols: int) -> bool:
    return num_rows * num_cols >= MQA_LOGITS_STATIC_SKIP_ELEMS


def mqa_logits_row_bytes(num_cols: int) -> int:
    aligned_cols = (
        ceil_div(num_cols, MQA_LOGITS_ROW_ALIGN_ELEMS) * MQA_LOGITS_ROW_ALIGN_ELEMS
    )
    return aligned_cols * MQA_LOGITS_BYTES_PER_ELEM


def mqa_logits_static_budget_bytes(*, device_index: int) -> int:
    """Budget from configuration alone (no device query); safe during graph capture."""
    total_mem = get_device_module().get_device_properties(device_index).total_memory
    total_mem_budget = int(total_mem * MQA_LOGITS_TOTAL_MEM_FRACTION)
    mem_fraction_static = get_schedule().mem_fraction_static
    if mem_fraction_static is None:
        budget = total_mem_budget
    else:
        static_free_mem = int(total_mem * max(0.0, 1.0 - mem_fraction_static))
        budget = min(
            int(static_free_mem * mqa_logits_free_mem_fraction()), total_mem_budget
        )
    return max(1, budget)


def mqa_logits_free_mem_bytes(
    device_index: int, *, include_reusable_cache: bool = False
) -> Tuple[int, int]:
    """(free, total) device memory. With include_reusable_cache, also count
    allocator-cached blocks that mem_get_info omits, excluding active
    allocations and inactive split blocks."""
    free_mem, total_mem = torch.cuda.mem_get_info(device_index)
    if include_reusable_cache:
        try:
            stats = torch.cuda.memory_stats(device_index)
            reusable_mem = max(
                0,
                int(stats["reserved_bytes.all.current"])
                - int(stats["active_bytes.all.current"])
                - int(stats["inactive_split_bytes.all.current"]),
            )
            free_mem = min(int(total_mem), int(free_mem) + reusable_mem)
        except (KeyError, RuntimeError, TypeError, ValueError):
            pass
    return int(free_mem), int(total_mem)


def mqa_logits_budget_bytes(
    *, device_index: int, allow_sync: bool, include_reusable_cache: bool = False
) -> int:
    """Static budget capped by current free memory; mem_get_info syncs, so
    callers pass allow_sync=False under CUDA graph capture."""
    budget = mqa_logits_static_budget_bytes(device_index=device_index)
    if allow_sync and not is_xpu():
        free_mem, _ = mqa_logits_free_mem_bytes(
            device_index, include_reusable_cache=include_reusable_cache
        )
        budget = min(int(free_mem * mqa_logits_free_mem_fraction()), budget)
    if is_hip():
        budget = min(budget, MQA_LOGITS_MAX_BYTES_ROCM)
    return max(1, budget)


def mqa_logits_should_chunk(
    *,
    num_rows: int,
    num_cols: int,
    get_budget_bytes: Callable[[], int],
    rocm: bool,
) -> Tuple[bool, int]:
    """Whether a [num_rows, num_cols] fp32 logits matrix must be row-chunked.

    Returns (need_chunk, effective_budget_bytes). Matrices below
    MQA_LOGITS_STATIC_SKIP_ELEMS never chunk and the budget is not queried
    (it may synchronize the host), hence the callable. On ROCm the budget is
    also capped by aiter's logits ceiling.
    """
    if not mqa_logits_needs_budget_check(num_rows=num_rows, num_cols=num_cols):
        return False, 0
    budget_bytes = get_budget_bytes()
    if rocm:
        budget_bytes = min(budget_bytes, MQA_LOGITS_MAX_BYTES_ROCM)
    logits_bytes = num_rows * num_cols * MQA_LOGITS_BYTES_PER_ELEM
    return logits_bytes > budget_bytes, budget_bytes


def mqa_logits_rows_per_chunk(
    *, num_rows: int, row_bytes: int, budget_bytes: int
) -> Optional[int]:
    """Query rows per chunk so one logits chunk fits the budget; None if all rows fit."""
    if num_rows * row_bytes <= budget_bytes:
        return None
    rows = max(budget_bytes // max(row_bytes, 1), 1)
    return int(rows) if rows < num_rows else None
