from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute import (
    compute_cu_seqlens,
)
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsa_backend import DSAMetadata


class BaseIndexerMetadata(ABC):
    @abstractmethod
    def get_seqlens_int32(self) -> torch.Tensor:
        """
        Return: (batch_size,) int32 tensor
        """

    @abstractmethod
    def get_page_table_64(self) -> torch.Tensor:
        """
        Return: (batch_size, num_blocks) int32, page table.
                The page size of the table is 64.
        """

    @abstractmethod
    def get_page_table_1(self) -> torch.Tensor:
        """
        Return: (batch_size, num_blocks) int32, page table.
                The page size of the table is 1.
        """

    @abstractmethod
    def get_seqlens_expanded(self) -> torch.Tensor:
        """
        Return: (sum_extend_seq_len,) int32 tensor
        """

    def get_indexer_kvcache_range(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return: (tokens, ), (tokens, ) int32, k_start and k_end in kv cache(token,xxx) for each token.
        """

    def get_indexer_seq_len_cpu(self) -> torch.Tensor:
        """
        Return: seq lens for each batch.
        """

    def get_indexer_seq_len(self) -> torch.Tensor:
        """
        Return: seq lens for each batch.
        """

    def get_dsa_extend_len_cpu(self) -> List[int]:
        """
        Return: extend seq lens for each batch.
        """

    def get_token_to_batch_idx(self) -> torch.Tensor:
        """
        Return: batch idx for each token.
        """

    @abstractmethod
    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Perform topk selection on the logits and possibly transform the result.

        NOTE that attention backend may override this function to do some
        transformation, which means the result of this topk_transform may not
        be the topk indices of the input logits.

        Return: Anything, since it will be passed to the attention backend
                for further processing on sparse attention computation.
                Don't assume it is the topk indices of the input logits.
        """

    def topk_transform_sparse_mask(
        self,
        logits: torch.Tensor,
        nonzero_mask: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        """Consume sparse Page-MQA output with its mandatory mask."""

        raise NotImplementedError("mask-aware sparse TopK is not implemented")


@dataclass(frozen=True)
class DSAIndexerMetadata(BaseIndexerMetadata):
    attn_metadata: DSAMetadata
    topk_transform_method: TopkTransformMethod
    topk_backend: DSATopKBackend = DSATopKBackend.SGL_KERNEL
    paged_mqa_schedule_metadata: Optional[torch.Tensor] = None
    paged_mqa_ctx_lens_2d: Optional[torch.Tensor] = None
    force_unfused_topk: bool = False

    def get_seqlens_int32(self) -> torch.Tensor:
        return self.attn_metadata.cache_seqlens_int32

    def get_page_table_64(self) -> torch.Tensor:
        return self.attn_metadata.real_page_table

    def get_page_table_1(self) -> torch.Tensor:
        return self.attn_metadata.page_table_1

    def get_seqlens_expanded(self) -> torch.Tensor:
        return self.attn_metadata.dsa_seqlens_expanded

    def get_indexer_kvcache_range(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.attn_metadata.indexer_k_start_end

    def get_indexer_seq_len(self) -> torch.Tensor:
        return self.attn_metadata.indexer_seq_lens

    def get_indexer_seq_len_cpu(self) -> torch.Tensor:
        return self.attn_metadata.indexer_seq_lens_cpu

    def get_dsa_extend_len_cpu(self) -> List[int]:
        return self.attn_metadata.dsa_extend_seq_lens_list

    def get_token_to_batch_idx(self) -> torch.Tensor:
        return self.attn_metadata.token_to_batch_idx

    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        ks: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        ke_offset: Optional[torch.Tensor] = None,
        batch_idx_list: Optional[List[int]] = None,
        topk_indices_offset_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if topk_indices_offset_override is not None:
            cu_topk_indices_offset = topk_indices_offset_override
            cu_seqlens_q_topk = None
        elif cu_seqlens_q is not None:
            cu_seqlens_q = cu_seqlens_q.to(torch.int32)
            cu_seqlens_q_topk = compute_cu_seqlens(cu_seqlens_q)
            cu_topk_indices_offset = torch.repeat_interleave(
                cu_seqlens_q_topk[:-1],
                cu_seqlens_q,
                # Avoid reading sum(cu_seqlens_q) back to the host.
                output_size=logits.shape[0],
            )
        else:
            cu_seqlens_q_topk = self.attn_metadata.cu_seqlens_q
            cu_topk_indices_offset = self.attn_metadata.topk_indices_offset
        if ke_offset is not None:
            seq_lens_topk = ke_offset
        else:
            seq_lens_topk = self.get_seqlens_expanded()
        return self.topk_backend.topk_transform(
            logits=logits,
            lengths=seq_lens_topk,
            topk=topk,
            topk_transform_method=self.topk_transform_method,
            attn_metadata=self.attn_metadata,
            cu_seqlens_q_topk=cu_seqlens_q_topk,
            topk_indices_offset=cu_topk_indices_offset,
            row_starts=ks,
            batch_idx_list=batch_idx_list,
            force_unfused_topk=self.force_unfused_topk,
        )

    def topk_transform_sparse_mask(
        self,
        logits: torch.Tensor,
        nonzero_mask: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        """Run the only valid consumer for LightOp sparse Page-MQA logits."""

        if (
            not envs.SGLANG_DSA_HCU_LIGHTOP_MASK_TOPK.get()
            or not envs.SGLANG_DSA_FUSE_TOPK.get()
            or self.force_unfused_topk
            or not self.topk_backend.is_sgl_kernel()
            or self.topk_transform_method != TopkTransformMethod.PAGED
            or topk != 2048
        ):
            raise RuntimeError(
                "Sparse Page-MQA requires the LightOp fused paged mask-aware "
                "TopK=2048 consumer"
            )

        lengths = self.get_seqlens_expanded()
        page_table_size_1 = self.attn_metadata.page_table_1
        cu_seqlens_q = self.attn_metadata.cu_seqlens_q
        if page_table_size_1 is None or (
            logits.dim() != 2
            or logits.shape[0] != lengths.shape[0]
            or page_table_size_1.shape[0] != logits.shape[0]
            or cu_seqlens_q.shape[0] != logits.shape[0] + 1
        ):
            raise RuntimeError(
                "Sparse Page-MQA metadata rows must match its paired paged TopK"
            )

        from lightop.attention import fast_topk_transform_sparse_mask_fused

        return fast_topk_transform_sparse_mask_fused(
            score=logits,
            nonzero_mask=nonzero_mask,
            lengths=lengths,
            page_table_size_1=page_table_size_1,
            cu_seqlens_q=cu_seqlens_q,
            topk=topk,
        )
