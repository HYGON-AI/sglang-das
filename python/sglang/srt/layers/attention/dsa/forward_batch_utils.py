from __future__ import annotations

from typing import Optional


def effective_forward_mode(forward_batch):
    """Return the algorithmic mode before MLP-sync remaps it to EXTEND."""

    original_forward_mode = getattr(forward_batch, "_original_forward_mode", None)
    return (
        forward_batch.forward_mode
        if original_forward_mode is None
        else original_forward_mode
    )


def get_flashmla_kv_valid_rows(forward_batch, num_rows: int) -> Optional[int]:
    """Return the real prefix length when MLP-sync padded FlashMLA KV rows."""

    forward_mode = effective_forward_mode(forward_batch)
    planned_batch_size = getattr(forward_batch, "forward_metadata_planned_bs", None)
    planned_num_tokens = getattr(
        forward_batch, "forward_metadata_planned_num_tokens", None
    )
    original_batch_size = getattr(forward_batch, "_original_batch_size", None)
    original_num_tokens = getattr(forward_batch, "_original_num_tokens", None)

    has_planned_layout = (
        planned_batch_size is not None or planned_num_tokens is not None
    )
    if has_planned_layout:
        if not (
            isinstance(planned_batch_size, int)
            and isinstance(planned_num_tokens, int)
            and planned_batch_size == original_batch_size
        ):
            return None
        real_batch_size = planned_batch_size
        real_num_tokens = planned_num_tokens
    else:
        real_batch_size = original_batch_size
        real_num_tokens = original_num_tokens

    if forward_mode.is_decode_or_idle():
        spec_info = getattr(forward_batch, "spec_info", None)
        if getattr(spec_info, "num_tokens_per_req", None) != 1:
            return None
        if isinstance(real_batch_size, int) and 0 <= real_batch_size < num_rows:
            return real_batch_size
        return None

    if not (forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2()):
        return None
    if not (
        isinstance(real_batch_size, int)
        and real_batch_size >= 0
        and isinstance(real_num_tokens, int)
        and 0 <= real_num_tokens < num_rows
    ):
        return None
    return real_num_tokens
