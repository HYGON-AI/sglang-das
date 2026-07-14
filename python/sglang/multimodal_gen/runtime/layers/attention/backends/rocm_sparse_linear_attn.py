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

import inspect
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sparse_linear_attn import (
    SparseLinearAttentionMetadata,
    SparseLinearAttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.models.utils import set_weight_attrs
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


class RocmSparseLinearAttentionBackend(AttentionBackend):
    """SLA backend for ROCm/HCU using flash_attn.sparse_attn_with_sla."""

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SLA_ATTN

    @staticmethod
    def get_impl_cls() -> type["RocmSparseLinearAttentionImpl"]:
        return RocmSparseLinearAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[SparseLinearAttentionMetadata]:
        return SparseLinearAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type[SparseLinearAttentionMetadataBuilder]:
        return SparseLinearAttentionMetadataBuilder


class RocmSparseLinearAttentionImpl(AttentionImpl, nn.Module):
    """ROCm/HCU SLA implementation backed by flash-attention-cutlass."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool = False,
        softmax_scale: float | None = None,
        num_kv_heads: int | None = None,
        prefix: str = "",
        topk_ratio: float = 0.1,
        feature_map: str = "softmax",
        local_blocks: int = 0,
        skip_linear_branch: bool = True,
        use_bf16: bool = True,
        use_fp8: bool = False,
        **extra_impl_args,
    ) -> None:
        del causal, softmax_scale, num_kv_heads, prefix, extra_impl_args
        nn.Module.__init__(self)
        if local_blocks < 0:
            raise ValueError(f"local_blocks must be non-negative, got {local_blocks}")

        self.num_heads = num_heads
        self.head_size = head_size
        self.topk_ratio = topk_ratio
        self.feature_map = feature_map
        self.local_blocks = local_blocks
        self.skip_linear_branch = skip_linear_branch
        self.use_bf16 = use_bf16
        self.use_fp8 = use_fp8
        self.compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

        self.proj_l = nn.Linear(head_size, head_size, dtype=torch.float32)
        set_weight_attrs(self.proj_l.weight, {"missing_param_init": "zeros"})
        if self.proj_l.bias is not None:
            set_weight_attrs(self.proj_l.bias, {"missing_param_init": "zeros"})

        self.feature_map_q: Callable[[torch.Tensor], torch.Tensor]
        self.feature_map_k: Callable[[torch.Tensor], torch.Tensor]
        if feature_map == "elu":
            self.feature_map_q = lambda x: F.elu(x) + 1
            self.feature_map_k = lambda x: F.elu(x) + 1
        elif feature_map == "relu":
            self.feature_map_q = F.relu
            self.feature_map_k = F.relu
        elif feature_map == "softmax":
            self.feature_map_q = lambda x: F.softmax(x, dim=-1)
            self.feature_map_k = lambda x: F.softmax(x, dim=-1)
        else:
            raise ValueError(f"Unknown feature map: {feature_map}")

        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.proj_l.weight)
            nn.init.zeros_(self.proj_l.bias)  # type: ignore[arg-type]

    @staticmethod
    def _get_sparse_attn_with_sla():
        try:
            from flash_attn import sparse_attn_with_sla
        except ImportError as err:
            raise ImportError(
                "ROCm SLA backend requires flash_attn.sparse_attn_with_sla. "
                "Please build and install flash-attention-cutlass with SLA support."
            ) from err

        return sparse_attn_with_sla

    @staticmethod
    def _get_sparse_attn_helpers():
        try:
            from flash_attn.flash_attn_interface import (
                get_block_map_fast,
                sparse_attn_func,
            )
            from flash_attn.utils.sparse_utils import (
                block_map_to_block_offset_triton,
                get_block_map,
            )
        except ImportError as err:
            raise ImportError(
                "ROCm SLA local-block mode requires flash_attn sparse attention "
                "helpers from flash-attention-cutlass."
            ) from err

        return (
            sparse_attn_func,
            get_block_map,
            get_block_map_fast,
            block_map_to_block_offset_triton,
        )

    @staticmethod
    def _select_sparse_block_m(seq_len: int) -> int:
        # flash-attention-cutlass SLA dispatches 64-row blocks up to 2048 tokens
        # and 128-row blocks for longer sequences.
        return 64 if seq_len <= 2048 else 128

    @staticmethod
    def _add_local_blocks_(
        sparse_map: torch.Tensor,
        local_blocks: int,
        block_m: int = 64,
        block_k: int = 64,
    ) -> None:
        if local_blocks <= 0:
            return

        _, _, num_q_blocks, num_k_blocks = sparse_map.shape
        q_blocks = torch.arange(num_q_blocks, device=sparse_map.device)
        q_start_k_blocks = (q_blocks * block_m) // block_k
        q_span_k_blocks = (block_m + block_k - 1) // block_k
        offsets = torch.arange(
            -local_blocks,
            local_blocks + q_span_k_blocks,
            device=sparse_map.device,
        )
        k_blocks = q_start_k_blocks[:, None] + offsets[None, :]
        valid = (0 <= k_blocks) & (k_blocks < num_k_blocks)
        q_indices = q_blocks[:, None].expand_as(k_blocks)[valid]
        k_indices = k_blocks[valid]
        sparse_map[:, :, q_indices, k_indices] = 1

    def _attention_dtype(self, tensor: torch.Tensor) -> torch.dtype:
        if tensor.is_cuda:
            return self.compute_dtype
        return tensor.dtype

    def _call_sparse_attn_with_sla(
        self,
        sparse_attn_with_sla: Callable[..., torch.Tensor],
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        if self.local_blocks > 0:
            return self._call_sparse_attn_with_local_blocks(query, key, value)

        kwargs = {
            "topk": self.topk_ratio,
            "feature_map": self.feature_map,
            "return_sparsity": False,
        }
        supported_params = inspect.signature(sparse_attn_with_sla).parameters
        if "use_bf16" in supported_params:
            kwargs["use_bf16"] = self.use_bf16
        if "use_fp8" in supported_params:
            kwargs["use_fp8"] = self.use_fp8

        return sparse_attn_with_sla(query, key, value, **kwargs)

    def _call_sparse_attn_with_local_blocks(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        (
            sparse_attn_func,
            get_block_map,
            get_block_map_fast,
            block_map_to_block_offset_triton,
        ) = self._get_sparse_attn_helpers()
        _, seq_len, _, head_dim = query.shape
        block_m = self._select_sparse_block_m(seq_len)
        block_k = 64

        if head_dim == 128:
            sparse_map, _, _ = get_block_map_fast(
                query,
                key,
                topk_ratio=self.topk_ratio,
                BLKQ=block_m,
                BLKK=block_k,
            )
        else:
            query_bhld = query.transpose(1, 2).contiguous()
            key_bhld = key.transpose(1, 2).contiguous()
            sparse_map, _, _ = get_block_map(
                query_bhld,
                key_bhld,
                topk_ratio=self.topk_ratio,
                BLKQ=block_m,
                BLKK=block_k,
            )
        self._add_local_blocks_(
            sparse_map,
            self.local_blocks,
            block_m=block_m,
            block_k=block_k,
        )
        block_offset, block_count = block_map_to_block_offset_triton(sparse_map)
        block_offset = block_offset * block_k

        batch_size, _, num_heads, _ = query.shape
        num_blocks_q = sparse_map.shape[-2]
        column_count = torch.zeros(
            (batch_size, num_heads, num_blocks_q),
            dtype=torch.int32,
            device=query.device,
        )
        column_index = torch.zeros(
            (batch_size, num_heads, num_blocks_q, 1),
            dtype=torch.int32,
            device=query.device,
        )

        return sparse_attn_func(
            query,
            key,
            value,
            block_count=block_count,
            block_offset=block_offset,
            column_count=column_count,
            column_index=column_index,
            softmax_scale=head_dim**-0.5,
            is_sla=True,
        )

    def _calc_linear_attention_with_torch(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        kv = torch.matmul(key.transpose(-1, -2), value)
        key_sum = torch.sum(key, dim=-2, keepdim=True)
        return torch.matmul(query, kv) / (
            1e-5 + torch.matmul(query, key_sum.transpose(-1, -2))
        )

    def _linear_attention(self, query, key, value) -> torch.Tensor:
        dtype = query.dtype
        compute_dtype = self._attention_dtype(query)

        query = query.transpose(1, 2).contiguous().to(compute_dtype)
        key = key.transpose(1, 2).contiguous().to(compute_dtype)
        value = value.transpose(1, 2).contiguous().to(compute_dtype)

        query = self.feature_map_q(query).contiguous()
        key = self.feature_map_k(key).contiguous()
        output = self._calc_linear_attention_with_torch(query, key, value)
        output = output.transpose(1, 2).contiguous()

        if output.is_cuda:
            with torch.amp.autocast("cuda", dtype=compute_dtype):
                output = self.proj_l(output)
        else:
            output = self.proj_l(output.to(self.proj_l.weight.dtype))

        return output.to(dtype)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: SparseLinearAttentionMetadata = None,
    ) -> torch.Tensor:
        del attn_metadata
        dtype = query.dtype
        compute_dtype = self._attention_dtype(query)
        sparse_attn_with_sla = self._get_sparse_attn_with_sla()

        sparse_output = self._call_sparse_attn_with_sla(
            sparse_attn_with_sla,
            query.contiguous().to(compute_dtype),
            key.contiguous().to(compute_dtype),
            value.contiguous().to(compute_dtype),
        )
        if self.skip_linear_branch:
            return sparse_output.to(dtype)

        linear_output = self._linear_attention(query, key, value)

        return (sparse_output.to(dtype) + linear_output).to(dtype)
