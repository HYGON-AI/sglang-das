"""Staging buffers and index remap kernels for CP Cache LayerSplit."""

from __future__ import annotations

from math import prod
from typing import Callable, Optional

import torch

from sglang.srt.utils import is_hcu

_is_hcu = is_hcu()


@torch.compile(dynamic=True)
def build_active_pages_mask(
    indices: torch.Tensor,
    page_size: int,
    max_pages: int,
) -> torch.Tensor:
    local_mask = torch.zeros(max_pages, dtype=torch.int32, device=indices.device)
    valid = (indices >= 0) & (indices < max_pages * page_size)
    safe_indices = torch.clamp(indices, min=0, max=max_pages * page_size - 1)
    page_ids = torch.div(safe_indices, page_size, rounding_mode="floor")
    local_mask.scatter_add_(
        0,
        page_ids.flatten().to(torch.long),
        valid.flatten().to(torch.int32),
    )
    return local_mask


def all_reduce_active_pages_mask(local_mask: torch.Tensor, pynccl_comm) -> torch.Tensor:
    """Sum the per-rank active-page mask across the attention-CP group."""
    with pynccl_comm.change_state(enable=True):
        pynccl_comm.all_reduce(local_mask)
    return local_mask


def _staging_page_map(
    selected_pages: torch.Tensor,
    max_pages: int,
    static_selected_pages: bool = False,
) -> torch.Tensor:
    """Pool page -> staging slot table; -1 marks pages outside the selection."""
    selected = selected_pages.to(torch.long)
    num_selected = selected.numel()
    if static_selected_pages:
        # nonzero_static pads with -1; park the padding past the real pages.
        selected = torch.where(
            selected >= 0,
            selected,
            max_pages
            + torch.arange(num_selected, dtype=torch.long, device=selected.device),
        )
        num_rows = max_pages + num_selected
    else:
        num_rows = max_pages
    page_map = torch.full((num_rows,), -1, dtype=torch.int32, device=selected.device)
    page_map[selected] = torch.arange(
        num_selected, dtype=torch.int32, device=selected.device
    )
    return page_map


def contiguous_request_page_capacity(
    total_tokens: int, num_reqs: int, page_size: int, max_pages: int
) -> int:
    """Bound pages touched by per-request contiguous token spans.

    A span can touch at most ceil(length / page_size) + 1 pages. Summing across
    requests is bounded by ceil(total_tokens / page_size) + 2 * num_reqs; the
    extra per-request allowance also covers padded entries mapped to page 0.
    """
    if total_tokens == 0:
        return min(max_pages, 1)
    return min(
        max_pages,
        (total_tokens + page_size - 1) // page_size + 2 * num_reqs,
    )


@torch.compile(dynamic=True, disable=_is_hcu)
def remap_indices_to_staging(
    indices: torch.Tensor,
    selected_pages: torch.Tensor,
    page_size: int,
    max_pages: int,
    static_selected_pages: bool = False,
) -> torch.Tensor:
    if _is_hcu:
        from lightop.attention import dsv41_remap_indices_to_staging

        return dsv41_remap_indices_to_staging(
            indices, selected_pages, page_size, max_pages, static_selected_pages
        )

    valid = (indices >= 0) & (indices < max_pages * page_size)
    safe_indices = torch.clamp(indices, min=0, max=max_pages * page_size - 1)
    page_ids = torch.div(safe_indices, page_size, rounding_mode="floor")
    offsets = safe_indices - page_ids * page_size
    page_map = _staging_page_map(selected_pages, max_pages, static_selected_pages)
    new_pages = page_map[page_ids.to(torch.long)]
    valid &= new_pages >= 0
    remapped = new_pages.to(indices.dtype) * page_size + offsets
    # An unmapped index must not fall through as a pool index: the caller reads
    # the staging buffer with it, which would silently fetch another page's KV.
    return torch.where(valid, remapped, torch.where(indices < 0, indices, -1))


@torch.compile(dynamic=True)
def remap_page_table_to_staging(
    page_table: torch.Tensor,
    selected_pages: torch.Tensor,
    max_pages: int,
) -> torch.Tensor:
    valid = (page_table >= 0) & (page_table < max_pages)
    safe_pages = torch.clamp(page_table, min=0, max=max_pages - 1)
    page_map = _staging_page_map(selected_pages, max_pages)
    remapped = page_map[safe_pages.to(torch.long)]
    valid &= remapped >= 0
    return torch.where(
        valid,
        remapped.to(page_table.dtype),
        torch.where(page_table < 0, page_table, -1),
    )


def active_pages_for_indices(
    indices: torch.Tensor,
    page_size: int,
    max_pages: int,
    pynccl_comm,
    static_size: Optional[int] = None,
) -> torch.Tensor:
    """Select pages touched by any CP rank; all ranks must call in the same order."""
    local_mask = build_active_pages_mask(indices, page_size, max_pages)
    local_mask = all_reduce_active_pages_mask(local_mask, pynccl_comm)
    if static_size is not None:
        return torch.nonzero_static(
            local_mask, size=static_size, fill_value=-1
        ).flatten()
    # Dynamic output size causes a device-to-host synchronization on HCU.
    return torch.nonzero(local_mask, as_tuple=False).flatten()


class StagingBufferManager:
    """Family-keyed staging buffers allocated before serving."""

    def __init__(self) -> None:
        self._buffers: dict[str, Optional[torch.Tensor]] = {}

    def allocate(
        self,
        family: str,
        num_pages: int,
        allocate_fn: Callable[[int], torch.Tensor],
    ) -> torch.Tensor:
        if family in self._buffers:
            raise RuntimeError(f"Staging buffer is already allocated: {family}")
        buffer = allocate_fn(num_pages)
        self._buffers[family] = buffer
        return buffer

    def get_existing(self, family: str) -> Optional[torch.Tensor]:
        return self._buffers.get(family)

    def allocate_shared(
        self, families: dict[str, tuple[int, Callable[[int], torch.Tensor]]]
    ) -> None:
        """Allocate aliased views for families with disjoint read/write lifetimes.

        Zero-page templates preserve each pool's padded page ABI without first
        allocating separate full-sized buffers. Callers must serialize reuse.
        """
        if not families:
            return
        for family, (num_pages, _) in families.items():
            if family in self._buffers:
                raise RuntimeError(f"Staging buffer is already allocated: {family}")
            if num_pages < 0:
                raise ValueError(f"Negative staging page count: {family}={num_pages}")

        templates = {
            family: allocate_fn(0)
            for family, (_, allocate_fn) in families.items()
        }
        first = next(iter(templates.values()))
        shapes = {}
        for family, template in templates.items():
            if template.ndim < 1 or template.shape[0] != 0:
                raise ValueError("Shared staging requires zero-page templates")
            if template.dtype != first.dtype or template.device != first.device:
                raise ValueError("Shared staging requires matching dtype and device")
            shapes[family] = (families[family][0], *template.shape[1:])

        storage = first.new_zeros(max(prod(shape) for shape in shapes.values()))
        for family, shape in shapes.items():
            self._buffers[family] = storage[: prod(shape)].view(shape)
