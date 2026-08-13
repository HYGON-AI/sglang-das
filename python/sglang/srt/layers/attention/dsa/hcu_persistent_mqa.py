# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runtime gate for the packaged gfx938 persistent paged-MQA fastpath."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_EXTENSION_API_NAME = "paged_mqa_logits_length_masked"

_selected_operation: Optional[Callable[..., torch.Tensor]] = None
_preload_failure: Optional[str] = None
_preloaded_hardware: Optional[tuple[str, int]] = None
_logged_hits: set[tuple[int, int, int]] = set()
_logged_misses: set[str] = set()


def resolve_persistent_ctas(
    configured_ctas: int, *, batch_size: int, max_context_len: int
) -> int:
    """Resolve graph-stable persistent grid size from capture-stable shapes."""

    _validate_configured_ctas(configured_ctas)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if max_context_len <= 0:
        raise ValueError(f"max_context_len must be positive, got {max_context_len}")
    if configured_ctas:
        return configured_ctas

    max_chunks = (max_context_len + 255) // 256
    batch_aware_floor = max(128, (2048 + batch_size - 1) // batch_size)
    return min(max_chunks, batch_aware_floor)


def _validate_configured_ctas(configured_ctas: int) -> None:
    if configured_ctas < 0 or configured_ctas > 4096:
        raise ValueError(
            "SGLANG_DSA_HCU_PERSISTENT_MQA_CTAS must be 0 (auto) or in "
            f"[1, 4096], got {configured_ctas}"
        )


def _input_gate_miss_reason(
    *,
    is_hcu: bool,
    page_size: int,
    q: torch.Tensor,
    fused_kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: Any,
    max_context_len: int,
) -> Optional[str]:
    if not is_hcu:
        return "device backend is not HCU"
    if page_size != 64:
        return f"page_size must be 64, got {page_size}"
    if schedule_meta is not None:
        return "schedule_meta must be None"
    if max_context_len <= 0:
        return f"max_context_len must be positive, got {max_context_len}"

    if q.device.type != "cuda":
        return f"q must be an HCU/HIP tensor, got device {q.device}"
    if q.dtype != torch.float8_e4m3fn:
        return f"q dtype must be float8_e4m3fn, got {q.dtype}"
    if q.dim() != 4 or tuple(q.shape[1:]) != (1, 32, 128):
        return f"q shape must be [batch, 1, 32, 128], got {tuple(q.shape)}"
    if q.shape[0] <= 0:
        return "q batch size must be positive"
    if not q.is_contiguous():
        return "q must be contiguous"

    reference_device = q.device
    for name, tensor in (
        ("fused_kv_cache", fused_kv_cache),
        ("weights", weights),
        ("context_lens", context_lens),
        ("block_table", block_table),
    ):
        if tensor.device != reference_device:
            return f"{name} must be on {reference_device}, got {tensor.device}"

    if fused_kv_cache.dtype != torch.uint8:
        return f"fused_kv_cache dtype must be uint8, got {fused_kv_cache.dtype}"
    if fused_kv_cache.dim() != 4 or tuple(fused_kv_cache.shape[1:]) != (
        64,
        1,
        132,
    ):
        return (
            "fused_kv_cache shape must be [num_blocks, 64, 1, 132], got "
            f"{tuple(fused_kv_cache.shape)}"
        )
    if fused_kv_cache.shape[0] <= 0:
        return "fused_kv_cache must contain at least one block"
    if tuple(fused_kv_cache.stride()) != (8448, 132, 132, 1):
        return (
            "fused_kv_cache stride must be [8448, 132, 132, 1], got "
            f"{tuple(fused_kv_cache.stride())}"
        )

    batch_size = q.shape[0]
    if (
        weights.dtype != torch.float32
        or weights.dim() != 2
        or tuple(weights.shape) != (batch_size, 32)
        or not weights.is_contiguous()
    ):
        return (
            "weights must be contiguous float32 [batch, 32], got "
            f"dtype={weights.dtype}, shape={tuple(weights.shape)}, "
            f"contiguous={weights.is_contiguous()}"
        )
    if (
        context_lens.dtype != torch.int32
        or context_lens.dim() != 1
        or tuple(context_lens.shape) != (batch_size,)
        or not context_lens.is_contiguous()
    ):
        return (
            "context_lens must be contiguous int32 [batch], got "
            f"dtype={context_lens.dtype}, shape={tuple(context_lens.shape)}, "
            f"contiguous={context_lens.is_contiguous()}"
        )
    if (
        block_table.dtype != torch.int32
        or block_table.dim() != 2
        or block_table.shape[0] != batch_size
        or block_table.shape[1] <= 0
        or block_table.stride(1) != 1
    ):
        return (
            "block_table must be int32 [batch, pages] with stride(1)=1, got "
            f"dtype={block_table.dtype}, shape={tuple(block_table.shape)}, "
            f"stride={tuple(block_table.stride())}"
        )
    block_table_capacity = block_table.shape[1] * page_size
    if max_context_len != block_table_capacity:
        return (
            f"max_context_len {max_context_len} must equal block-table capacity "
            f"{block_table_capacity}"
        )
    return None


def _hardware_gate_miss_reason(*, arch_name: str, num_cus: int) -> Optional[str]:
    if not arch_name.startswith("gfx938"):
        return f"GPU architecture must be gfx938, got {arch_name or 'unknown'}"
    if num_cus != 64:
        return f"gfx938 specialization requires 64 CUs, got {num_cus}"
    return None


def _consumer_gate_miss_reason(
    *,
    fuse_topk: bool,
    force_unfused_topk: bool,
    topk_transform_method_name: str,
    index_topk: int,
    score_context_lens: torch.Tensor,
    topk_context_lens: torch.Tensor,
) -> Optional[str]:
    if not fuse_topk:
        return "SGLANG_DSA_FUSE_TOPK must be enabled"
    if topk_transform_method_name != "PAGED":
        return (
            "metadata.topk_transform_method must be PAGED, got "
            f"{topk_transform_method_name or 'unknown'}"
        )
    if index_topk != 2048:
        return f"index_topk must be 2048, got {index_topk}"
    if score_context_lens is not topk_context_lens:
        topk_consumer = (
            "fast_topk_v2" if force_unfused_topk else "fast_topk_transform_fused"
        )
        return (
            "MQA context_lens must be the exact tensor consumed by production "
            f"{topk_consumer}"
        )
    return None


def _select_package_operation(package_operation: Any) -> None:
    global _selected_operation

    if _selected_operation is not None:
        return
    if not callable(package_operation):
        raise RuntimeError(
            "installed LightOp package does not export callable "
            f"{_EXTENSION_API_NAME}"
        )
    _selected_operation = package_operation
    logger.info("Using packaged LightOp persistent paged-MQA operation")


def _raise_unavailable(reason: str) -> None:
    raise RuntimeError(f"HCU persistent paged-MQA fastpath unavailable: {reason}")


def _log_consumer_fallback(reason: str) -> None:
    if reason not in _logged_misses:
        _logged_misses.add(reason)
        logger.warning(
            "HCU persistent paged-MQA fastpath missed; using production LightOp: %s",
            reason,
        )


def preload_hcu_persistent_mqa(
    *,
    is_hcu: bool,
    arch_name: str,
    num_cus: int,
    package_operation: Any = None,
) -> None:
    """Validate and select the packaged LightOp API before graph capture."""

    global _preload_failure, _preloaded_hardware

    if not envs.SGLANG_DSA_HCU_PERSISTENT_MQA_FASTPATH.get():
        return

    if _preload_failure is not None:
        _raise_unavailable(_preload_failure)

    if not is_hcu:
        reason = "device backend is not HCU"
    else:
        reason = _hardware_gate_miss_reason(
            arch_name=arch_name,
            num_cus=num_cus,
        )
    if reason is not None:
        _preload_failure = reason
        _raise_unavailable(reason)

    try:
        _validate_configured_ctas(envs.SGLANG_DSA_HCU_PERSISTENT_MQA_CTAS.get())
        _select_package_operation(package_operation)
        _preloaded_hardware = (arch_name, num_cus)
    except (RuntimeError, ValueError) as error:
        _preload_failure = str(error)
        _raise_unavailable(_preload_failure)


def _log_hit_once(*, q_rows: int, max_context_len: int, resolved_ctas: int) -> None:
    hit_key = (q_rows, max_context_len, resolved_ctas)
    if hit_key in _logged_hits:
        return
    _logged_hits.add(hit_key)
    logger.info(
        "HCU persistent paged-MQA fastpath HIT source=lightop-package q_rows=%d "
        "max_context_len=%d resolved_ctas=%d; kernel writes only the valid "
        "prefix and downstream production TopK must use the same context_lens",
        q_rows,
        max_context_len,
        resolved_ctas,
    )


def paged_mqa_logits_length_masked(
    q: torch.Tensor,
    fused_kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: Any,
    max_context_len: int,
    *,
    is_hcu: bool,
    page_size: int,
    fuse_topk: bool,
    force_unfused_topk: bool,
    topk_transform_method_name: str,
    index_topk: int,
    topk_context_lens: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Use the length-masked kernel only when producer and consumer agree."""

    if not envs.SGLANG_DSA_HCU_PERSISTENT_MQA_FASTPATH.get():
        return None

    if _preload_failure is not None:
        _raise_unavailable(_preload_failure)

    reason = _input_gate_miss_reason(
        is_hcu=is_hcu,
        page_size=page_size,
        q=q,
        fused_kv_cache=fused_kv_cache,
        weights=weights,
        context_lens=context_lens,
        block_table=block_table,
        schedule_meta=schedule_meta,
        max_context_len=max_context_len,
    )
    if reason is not None:
        _raise_unavailable(reason)

    reason = _consumer_gate_miss_reason(
        fuse_topk=fuse_topk,
        force_unfused_topk=force_unfused_topk,
        topk_transform_method_name=topk_transform_method_name,
        index_topk=index_topk,
        score_context_lens=context_lens,
        topk_context_lens=topk_context_lens,
    )
    if reason is not None:
        # Forward modes may use different TopK metadata. A mismatch must use the
        # regular kernel, which initializes all logits before that consumer.
        _log_consumer_fallback(reason)
        return None

    operation = _selected_operation
    if _preloaded_hardware is None or operation is None:
        _raise_unavailable("packaged LightOp operation was not preloaded")

    try:
        resolved_ctas = resolve_persistent_ctas(
            envs.SGLANG_DSA_HCU_PERSISTENT_MQA_CTAS.get(),
            batch_size=q.shape[0],
            max_context_len=max_context_len,
        )
    except ValueError as error:
        _raise_unavailable(str(error))

    logits = operation(
        q,
        fused_kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        True,
        4,
        resolved_ctas,
    )
    _log_hit_once(
        q_rows=q.shape[0],
        max_context_len=max_context_len,
        resolved_ctas=resolved_ctas,
    )
    return logits
