from __future__ import annotations


def effective_forward_mode(forward_batch):
    """Return the algorithmic mode before MLP-sync remaps it to EXTEND."""

    original_forward_mode = getattr(forward_batch, "_original_forward_mode", None)
    return (
        forward_batch.forward_mode
        if original_forward_mode is None
        else original_forward_mode
    )
