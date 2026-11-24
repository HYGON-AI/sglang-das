"""HCU prefill candidate indexer source/consumer optimizations (#40352)."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple


def hcu_prefill_40352_enabled() -> bool:
    return os.environ.get("SGLANG_HCU_OPT_DSV41_PREFILL_CONSUMER_40352", "1") == "1"


def cand_window_covers_all(indexer, lc_per_req: List[int]) -> bool:
    """True when every request's reachable compressed KV fits inside candidate_topk_blocks."""
    topk_blocks = getattr(indexer, "candidate_topk_blocks", 0)
    block_size = getattr(indexer, "candidate_block_size", 8)
    return topk_blocks > 0 and max(lc_per_req, default=0) <= topk_blocks * block_size


def get_late_tail_publish_target(
    backend, q_lens_cpu: List[int]
) -> Optional[Tuple[object, List[int]]]:
    """Return (tail_forward_metadata, tail_lens_cpu) when bounded SWA replay uses a tail."""
    tail_meta = getattr(backend, "tail_forward_metadata", None)
    if tail_meta is None or tail_meta is getattr(backend, "forward_metadata", None):
        return None
    tail = getattr(tail_meta, "late_layer_tail", None)
    if tail is None:
        return None
    tail_lens_cpu = (
        tail.local_lens_cpu
        if getattr(tail, "cp_metadata", None) is not None
        else tail.extend_seq_lens_cpu
    )
    if tail_lens_cpu is None or len(tail_lens_cpu) != len(q_lens_cpu):
        return None
    if any(t < 0 or t > q for t, q in zip(tail_lens_cpu, q_lens_cpu)):
        return None
    return tail_meta, list(tail_lens_cpu)
