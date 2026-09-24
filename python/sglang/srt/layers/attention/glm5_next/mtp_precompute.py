"""Multi-step precompute utilities for Native Sparse Attention backend.

This module provides optimization utilities for multi-step speculative decoding
by precomputing shared metadata once and copying it to multiple backend instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.glm5_next.utils import compute_dsa_seqlens

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.speculative.spec_info import SpecInput


@dataclass
class PrecomputedMetadata:
    """Precomputed metadata shared across multiple backend instances.

    Used for multi-step speculative decoding where multiple backends
    need identical metadata. Precomputing once and copying N times
    is much faster than computing N times.

    """

    # Basic seqlens
    cache_seqlens: torch.Tensor  # int32, [bs]
    cu_seqlens_k: torch.Tensor  # int32, [bs+1]

    # Page table
    page_indices: torch.Tensor  # int32, [bs, max_len] or [expanded_bs, max_len]
    real_page_table: Optional[torch.Tensor]  # int32, transformed version

    # DSA seqlens
    seqlens_expanded: torch.Tensor  # int32, [expanded_size]
    dsa_cache_seqlens: torch.Tensor  # int32, [expanded_size]
    dsa_cu_seqlens_k: torch.Tensor  # int32, [expanded_size+1]
    seqlens_expanded_size: int

    # Dimensions
    max_len: int  # for decode
    max_seqlen_k: int  # for target_verify

    # FlashMLA (optional)
    flashmla_metadata: Optional[torch.Tensor] = None

    # KPool write plans are backend-local CUDA-graph buffers.  Keep the
    # shared input so every backend can rebuild its own plan after metadata
    # has been copied from this precomputed object.
    req_pool_indices: Optional[torch.Tensor] = None


def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:
    """Compute cumulative sequence lengths with padding."""
    assert seqlens.dtype == torch.int32
    return torch.nn.functional.pad(
        torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
    )


@triton.jit
def _fill_decode_page_table_kernel(
    req_to_token,
    req_pool_indices,
    seq_lens,
    page_table,
    req_to_token_stride: tl.constexpr,
    page_table_stride: tl.constexpr,
    max_len: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    req_idx = tl.load(req_pool_indices + row)
    seq_len = tl.load(seq_lens + row)
    in_bounds = cols < max_len
    valid = in_bounds & (cols < seq_len)
    vals = tl.load(
        req_to_token + req_idx * req_to_token_stride + cols,
        mask=valid,
        other=0,
    ).to(tl.int32)
    tl.store(page_table + row * page_table_stride + cols, vals, mask=in_bounds)


def fill_decode_page_table_gpu(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    bs: int,
    max_fill_len: Optional[int] = None,
):
    """Fill decode page table from GPU seq_lens without materializing CPU lengths."""
    if bs == 0:
        return
    max_len = page_table.shape[1]
    if max_fill_len is not None:
        max_len = min(max_len, max_fill_len)
    if max_len == 0:
        return
    # Keep the optimized launch shape, but zero the invalid tail inside the
    # filled window so page-table transforms never consume uninitialized ids.
    block = 1024
    _fill_decode_page_table_kernel[(bs, triton.cdiv(max_len, block))](
        req_to_token,
        req_pool_indices,
        seq_lens,
        page_table,
        req_to_token.shape[1],
        page_table.stride(0),
        max_len,
        BLOCK=block,
        num_warps=8,
    )


class NativeSparseAttnBackendMTPPrecomputeMixin:
    """Mixin class providing metadata precomputation for multi-step speculative decoding.

    This mixin provides the _precompute_replay_metadata method and its helpers,
    which are used to optimize CUDA graph replay in multi-step scenarios.
    """

    def _precompute_replay_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ) -> PrecomputedMetadata:
        """Precompute all shared metadata for multi-step backends.

        This function extracts and computes all operations that are
        identical across different backend instances in multi-step
        speculative decoding.

        Args:
            bs: Batch size
            req_pool_indices: Request pool indices [bs]
            seq_lens: Sequence lengths [bs]
            seq_lens_cpu: Sequence lengths on CPU [bs]
            forward_mode: Forward mode (decode/target_verify)
            spec_info: Speculative decoding info (reserved for future use)

        Returns:
            PrecomputedMetadata containing all shared intermediate results
        """
        # Slice inputs to batch size
        seq_lens = seq_lens[:bs]
        if seq_lens_cpu is not None:
            seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]

        # Dispatch to mode-specific precomputation
        if forward_mode.is_decode_or_idle():
            return self._precompute_decode_mode(bs, req_pool_indices, seq_lens)
        elif forward_mode.is_target_verify():
            assert seq_lens_cpu is not None
            return self._precompute_target_verify_mode(
                bs, req_pool_indices, seq_lens, seq_lens_cpu
            )
        else:
            raise ValueError(f"Unsupported forward mode: {forward_mode}")

    def _precompute_decode_mode(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> PrecomputedMetadata:
        """Precompute metadata for normal decode mode."""
        # Convert to int32 and compute cumsum
        cache_seqlens = seq_lens.to(torch.int32)
        cu_seqlens_k = compute_cu_seqlens(cache_seqlens)

        # Get page indices from cache on device. The shape is the captured graph
        # width; the kernel masks each row by GPU seq_lens.
        metadata = self.decode_cuda_graph_metadata[bs]
        max_len = metadata.page_table_1.shape[1]
        page_indices = torch.empty(
            (bs, max_len), dtype=torch.int32, device=seq_lens.device
        )
        fill_decode_page_table_gpu(
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            page_indices,
            bs,
        )

        # Compute DSA seqlens
        dsa_cache_seqlens = compute_dsa_seqlens(
            cache_seqlens,
            dsa_index_topk=self.dsa_index_topk,
            index_kpool=self.dsa_index_kpool,
        )
        seqlens_expanded = cache_seqlens
        seqlens_expanded_size = seqlens_expanded.shape[0]

        # Compute DSA cumsum
        dsa_cu_seqlens_k = compute_cu_seqlens(dsa_cache_seqlens)

        # Transform page table if needed
        if self.real_page_size > 1:
            real_page_table = self._transform_table_1_to_real(page_indices)
        else:
            real_page_table = None  # Will use page_indices directly

        # Compute FlashMLA metadata if needed
        flashmla_metadata = None
        if self.dsa_decode_impl == "flashmla_kv":
            flashmla_metadata = self._compute_flashmla_metadata(
                cache_seqlens=dsa_cache_seqlens,
                seq_len_q=1,
            )

        return PrecomputedMetadata(
            cache_seqlens=cache_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            page_indices=page_indices,
            real_page_table=real_page_table,
            seqlens_expanded=seqlens_expanded,
            dsa_cache_seqlens=dsa_cache_seqlens,
            dsa_cu_seqlens_k=dsa_cu_seqlens_k,
            seqlens_expanded_size=seqlens_expanded_size,
            max_len=max_len,
            max_seqlen_k=max_len,
            flashmla_metadata=flashmla_metadata,
            req_pool_indices=req_pool_indices,
        )

    def _precompute_target_verify_mode(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
    ) -> PrecomputedMetadata:
        """Precompute metadata for target verify mode."""
        max_seqlen_k = int(
            seq_lens_cpu.max().item() + self.speculative_num_draft_tokens
        )

        # Cache seqlens with draft tokens
        cache_seqlens = (seq_lens + self.speculative_num_draft_tokens).to(torch.int32)
        cu_seqlens_k = compute_cu_seqlens(cache_seqlens)

        # Page indices (repeated for each draft token)
        page_indices = self.req_to_token[req_pool_indices, :max_seqlen_k]
        page_indices = torch.repeat_interleave(
            page_indices, repeats=self.speculative_num_draft_tokens, dim=0
        ).contiguous()

        # Generate expanded seqlens
        extend_seq_lens_cpu = [self.speculative_num_draft_tokens] * bs
        seqlens_int32_cpu = [
            self.speculative_num_draft_tokens + kv_len
            for kv_len in seq_lens_cpu.tolist()
        ]
        seqlens_expanded = torch.cat(
            [
                torch.arange(
                    kv_len - qo_len + 1,
                    kv_len + 1,
                    dtype=torch.int32,
                    device=self.device,
                )
                for qo_len, kv_len in zip(
                    extend_seq_lens_cpu,
                    seqlens_int32_cpu,
                    strict=True,
                )
            ]
        )

        # Compute DSA seqlens
        dsa_cache_seqlens = compute_dsa_seqlens(
            seqlens_expanded,
            self.dsa_index_topk,
            index_kpool=self.dsa_index_kpool,
        )
        seqlens_expanded_size = seqlens_expanded.shape[0]

        # DSA cumsum
        dsa_cu_seqlens_k = compute_cu_seqlens(dsa_cache_seqlens)

        # Transform page table
        if self.real_page_size > 1:
            real_page_table = self._transform_table_1_to_real(page_indices)
        else:
            real_page_table = None

        # FlashMLA metadata
        flashmla_metadata = None
        if self.dsa_decode_impl == "flashmla_kv":
            flashmla_metadata = self._compute_flashmla_metadata(
                cache_seqlens=dsa_cache_seqlens,
                seq_len_q=1,
            )

        return PrecomputedMetadata(
            cache_seqlens=cache_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            page_indices=page_indices,
            real_page_table=real_page_table,
            seqlens_expanded=seqlens_expanded,
            dsa_cache_seqlens=dsa_cache_seqlens,
            dsa_cu_seqlens_k=dsa_cu_seqlens_k,
            seqlens_expanded_size=seqlens_expanded_size,
            max_len=-1,  # Not used in this mode
            max_seqlen_k=max_seqlen_k,
            flashmla_metadata=flashmla_metadata,
            req_pool_indices=req_pool_indices,
        )
