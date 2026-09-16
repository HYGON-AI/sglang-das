from __future__ import annotations

import functools
import logging

from sglang.srt.environ import envs
from sglang.srt.utils import is_gfx95_supported, is_hip

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def is_unified_kv_triton() -> bool:
    # unified_kv_triton is only implemented on HIP (ROCm)
    return is_hip() and envs.SGLANG_HACK_FLASHMLA_BACKEND.get() == "unified_kv_triton"


@functools.lru_cache(maxsize=1)
def is_unified_kv_fp8() -> bool:
    if not (is_unified_kv_triton() and envs.SGLANG_DSV4_UNIFIED_KV_FP8.get()):
        return False
    if not is_gfx95_supported():
        logger.warning(
            "SGLANG_DSV4_UNIFIED_KV_FP8=1 needs an AMD gfx95 GPU; falling back to "
            "the bf16 unified_kv pool (see unified_fp8= in the DSV4 memory log)."
        )
        return False
    return True
