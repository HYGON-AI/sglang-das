from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Optional

import torch


def get_flashmla_op(name: str, *, is_hcu: bool) -> Callable[..., Any]:
    module_name = "flash_mla.flash_mla_interface" if is_hcu else "sgl_kernel.flash_mla"
    return getattr(import_module(module_name), name)


@dataclass(frozen=True)
class DSAFlashMLAMetadata:
    """Metadata needed only by FlashMLA.

    CUDA FlashMLA stores scheduler metadata in tensors. HCU FlashMLA stores it
    in a mutable backend object and initializes the object's tensor fields on
    the first kernel invocation.
    """

    flashmla_metadata: Any
    num_splits: Optional[torch.Tensor]

    def slice(self, sli) -> DSAFlashMLAMetadata:
        return DSAFlashMLAMetadata(
            flashmla_metadata=self.flashmla_metadata,
            num_splits=(None if self.num_splits is None else self.num_splits[sli]),
        )

    def copy_(self, other: DSAFlashMLAMetadata) -> None:
        if isinstance(self.flashmla_metadata, torch.Tensor) and isinstance(
            other.flashmla_metadata, torch.Tensor
        ):
            self.flashmla_metadata.copy_(other.flashmla_metadata)
        else:
            object.__setattr__(self, "flashmla_metadata", other.flashmla_metadata)

        if isinstance(self.num_splits, torch.Tensor) and isinstance(
            other.num_splits, torch.Tensor
        ):
            self.num_splits.copy_(other.num_splits)
        else:
            object.__setattr__(self, "num_splits", other.num_splits)


def can_fuse_flashmla_metadata(
    *metadatas: Optional[DSAFlashMLAMetadata],
) -> bool:
    return all(
        metadata is not None
        and isinstance(metadata.flashmla_metadata, torch.Tensor)
        and isinstance(metadata.num_splits, torch.Tensor)
        for metadata in metadatas
    )


def wrap_flashmla_metadata_result(
    result: tuple[Any, Optional[torch.Tensor]], *, is_hcu: bool
) -> DSAFlashMLAMetadata:
    flashmla_metadata, num_splits = result
    if is_hcu:
        num_splits = getattr(flashmla_metadata, "num_splits", num_splits)
    return DSAFlashMLAMetadata(
        flashmla_metadata=flashmla_metadata,
        num_splits=num_splits,
    )
