"""Staging buffers and index remap kernels for CP Cache LayerSplit."""

from __future__ import annotations

from math import prod
from typing import Callable, Optional

import torch


@torch.compile(dynamic=True)
def build_active_pages_mask(
    indices: torch.Tensor,
    page_size: int,
    max_pages: int,
) -> torch.Tensor:
    local_mask = torch.zeros(max_pages, dtype=torch.int32, device=indices.device)
    valid = indices >= 0
    safe_indices = torch.clamp(indices, min=0)
    page_ids = torch.div(safe_indices, page_size, rounding_mode="floor")
    local_mask.index_put_(
        (page_ids.flatten().to(torch.long),),
        valid.flatten().to(torch.int32),
        accumulate=True,
    )
    return local_mask


def all_reduce_active_pages_mask(local_mask: torch.Tensor, pynccl_comm) -> torch.Tensor:
    """Sum the per-rank active-page mask across the attention-CP group."""
    with pynccl_comm.change_state(enable=True):
        pynccl_comm.all_reduce(local_mask)
    return local_mask


@torch.compile(dynamic=True)
def remap_indices_to_staging(
    indices: torch.Tensor,
    selected_pages: torch.Tensor,
    page_size: int,
    max_pages: int,
) -> torch.Tensor:
    page_map = torch.full((max_pages,), -1, dtype=torch.int32, device=indices.device)
    page_map[selected_pages.to(torch.long)] = torch.arange(
        selected_pages.numel(), dtype=torch.int32, device=indices.device
    )

    valid = indices >= 0
    safe_indices = torch.clamp(indices, min=0)
    page_ids = torch.div(safe_indices, page_size, rounding_mode="floor")
    offsets = safe_indices - page_ids * page_size
    new_pages = page_map[page_ids.to(torch.long)].to(indices.dtype)
    remapped = new_pages * page_size + offsets
    return torch.where(valid, remapped, indices)


@torch.compile(dynamic=True)
def remap_page_table_to_staging(
    page_table: torch.Tensor,
    selected_pages: torch.Tensor,
    max_pages: int,
) -> torch.Tensor:
    page_map = torch.full((max_pages,), -1, dtype=torch.int32, device=page_table.device)
    page_map[selected_pages.to(torch.long)] = torch.arange(
        selected_pages.numel(), dtype=torch.int32, device=page_table.device
    )

    valid = page_table >= 0
    safe_pages = torch.clamp(page_table, min=0)
    remapped = page_map[safe_pages.to(torch.long)].to(page_table.dtype)
    return torch.where(valid, remapped, page_table)


def active_pages_for_indices(
    indices: torch.Tensor,
    page_size: int,
    max_pages: int,
    pynccl_comm,
) -> torch.Tensor:
    """Select pages touched by any CP rank; all ranks must call in the same order."""
    local_mask = build_active_pages_mask(indices, page_size, max_pages)
    local_mask = all_reduce_active_pages_mask(local_mask, pynccl_comm)
    # TODO: replace torch.nonzero with bounded GPU compaction to avoid a
    # dynamic-shape CUDA synchronization in this hot read path.
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
