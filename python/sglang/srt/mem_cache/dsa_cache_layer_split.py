# Copyright 2023-2026 SGLang Team
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
# ==============================================================================

"""Layer-sharded DSA KV cache pool for context-parallel prefill.

``LayerSplitDSATokenToKVPool`` splits the DSA (DeepSeek Sparse Attention) GPU
KV/indexer cache layers across context-parallel (CP) ranks so that each rank
only materializes the layers it owns, reducing per-rank KV memory. When a rank
needs to read a layer it does not own, the owning rank broadcasts that layer's
buffer into a small per-rank remote scratch buffer.

This subclass keeps the core ``KVCache`` / ``MLATokenToKVPool`` /
``DSATokenToKVPool`` pools untouched: all sharding, broadcast, and remote-scratch
bookkeeping lives here. Layer split is only ever enabled for DSA MLA models on
PD prefill workers under prefill-CP (see
``sglang.srt.layers.cp.utils.is_glm_dsa_cache_layer_split_enabled``).
"""

from __future__ import annotations

import logging
import weakref
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.kernels.ops.attention.dsa import index_buf_accessor
from sglang.srt.layers.attention.dsa.hcu_int8_index_k_cache import (
    IndexKCacheMode,
    quantize_and_store_index_k_int8,
)
from sglang.srt.layers.cp.utils import get_layer_owner, get_layer_shard_range
from sglang.srt.mem_cache.index_key_cache import IndexKeyCache
from sglang.srt.mem_cache.memory_pool import (
    GPU_MEMORY_TYPE_KV_CACHE,
    DSATokenToKVPool,
    RadixAttention,
    get_tensor_size_bytes,
    maybe_detect_oob,
    unwrap_write_loc,
)
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter

logger = logging.getLogger(__name__)


@dataclass(eq=False, frozen=True)
class MainKVPagePlan:
    """Physical pages needed by one ordinary prefill ``ForwardBatch``.

    ``history_page_ids`` is the unique payload copied from each layer owner.
    ``all_page_ids`` additionally contains pages receiving this step's newly
    computed KV. Object identity deliberately serves as the batch identity.
    """

    history_page_ids: torch.Tensor
    all_page_ids: torch.Tensor


def _build_unique_physical_pages(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    sequence_lens: Sequence[int],
    page_size: int,
) -> torch.Tensor:
    """Collect one physical page ID per logical page and remove duplicates."""

    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if req_to_token.ndim != 2:
        raise ValueError(
            f"req_to_token must be 2-D, got shape={tuple(req_to_token.shape)}"
        )

    batch_size = len(sequence_lens)
    if batch_size == 0:
        return torch.empty(0, dtype=torch.long, device=req_to_token.device)
    if req_pool_indices.numel() < batch_size:
        raise ValueError(
            "req_pool_indices is shorter than sequence_lens: "
            f"{req_pool_indices.numel()} < {batch_size}"
        )

    page_counts = []
    for sequence_len in sequence_lens:
        sequence_len = int(sequence_len)
        if sequence_len < 0:
            raise ValueError(
                f"sequence length must be non-negative, got {sequence_len}"
            )
        page_counts.append((sequence_len + page_size - 1) // page_size)

    total_pages = sum(page_counts)
    if total_pages == 0:
        return torch.empty(0, dtype=torch.long, device=req_to_token.device)

    max_pages = max(page_counts)
    if (max_pages - 1) * page_size >= req_to_token.shape[1]:
        raise ValueError(
            "sequence length exceeds req_to_token capacity: "
            f"max_pages={max_pages}, page_size={page_size}, "
            f"capacity={req_to_token.shape[1]}"
        )

    device = req_to_token.device
    page_counts_tensor = torch.tensor(page_counts, dtype=torch.long, device=device)
    request_rows = torch.repeat_interleave(
        req_pool_indices[:batch_size].to(device=device, dtype=torch.long),
        page_counts_tensor,
        output_size=total_pages,
    )
    request_page_starts = torch.cumsum(page_counts_tensor, dim=0) - page_counts_tensor
    repeated_page_starts = torch.repeat_interleave(
        request_page_starts,
        page_counts_tensor,
        output_size=total_pages,
    )
    logical_pages = (
        torch.arange(total_pages, dtype=torch.long, device=device)
        - repeated_page_starts
    )
    physical_page_starts = req_to_token[request_rows, logical_pages * page_size]
    physical_pages = torch.div(
        physical_page_starts, page_size, rounding_mode="floor"
    ).to(torch.long)
    # Physical page 0 is the pool's padded/dummy page, never request history.
    return torch.unique(physical_pages[physical_pages > 0], sorted=True).contiguous()


def build_main_kv_page_plan(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    prefix_lens: Sequence[int],
    current_locs: torch.Tensor,
    page_size: int,
) -> MainKVPagePlan:
    """Build the deterministic, CP-global compact layout for one batch."""

    history_page_ids = _build_unique_physical_pages(
        req_to_token,
        req_pool_indices,
        prefix_lens,
        page_size,
    )
    current_locs = current_locs.reshape(-1)
    valid_current_locs = current_locs[current_locs >= 0]
    current_page_ids = torch.unique(
        torch.div(valid_current_locs, page_size, rounding_mode="floor").to(torch.long),
        sorted=True,
    )
    current_page_ids = current_page_ids[current_page_ids > 0]
    all_page_ids = torch.unique(
        torch.cat((history_page_ids, current_page_ids)), sorted=True
    ).contiguous()
    return MainKVPagePlan(
        history_page_ids=history_page_ids,
        all_page_ids=all_page_ids,
    )


class LayerSplitIndexKeyCache(IndexKeyCache):
    def __init__(self, pool: LayerSplitDSATokenToKVPool, index_buf_size: int):
        num_pages = (index_buf_size + pool.page_size + 1) // pool.page_size
        if pool.index_k_cache_mode is IndexKCacheMode.BF16:
            self.pool = pool
            with (
                torch.cuda.use_mem_pool(pool.custom_mem_pool)
                if pool.custom_mem_pool
                else nullcontext()
            ):
                self.buffer = [
                    torch.zeros(
                        self._buffer_shape(self._layer_num_pages(i, num_pages)),
                        dtype=self._buffer_dtype(),
                        device=pool.device,
                    )
                    for i in range(pool.indexer_layer_num)
                ]
        else:
            super().__init__(pool, index_buf_size)

        with (
            torch.cuda.use_mem_pool(pool.custom_mem_pool)
            if pool.custom_mem_pool
            else nullcontext()
        ):
            self.remote_buffer = torch.zeros(
                self._buffer_shape(num_pages),
                dtype=self._buffer_dtype(),
                device=pool.device,
            )
        self.remote_layer_id: Optional[int] = None
        self.pending_event = pool.device_module.Event()
        self.pending_layer_id: Optional[int] = None
        self.pending_broadcast = False

    def _buffer_shape(self, num_pages: int) -> tuple[int, ...]:
        if self.pool.index_k_cache_mode is IndexKCacheMode.BF16:
            return (
                num_pages,
                self.pool.page_size,
                1,
                self.pool.index_head_dim,
            )
        return super()._buffer_shape(num_pages)

    def _buffer_dtype(self) -> torch.dtype:
        if self.pool.index_k_cache_mode is IndexKCacheMode.BF16:
            return self.pool.index_k_buffer_dtype
        return self.pool.index_k_with_scale_buffer_dtype

    def _layer_num_pages(self, layer_idx: int, num_pages: int) -> int:
        layer_id = self.pool.indexer_layer_ids[layer_idx]
        return num_pages if self.pool._is_layer_owned(layer_id) else 0

    def clear(self) -> None:
        super().clear()
        del self.remote_buffer

    def move(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor) -> None:
        if tgt_loc.numel() == 0:
            return
        # The owner buffers are about to change outside the regular mirrored
        # write path, so no cached remote layer remains valid.
        self.finalize_pending(set_remote_layer_id=False)
        self.remote_layer_id = None
        if self.pool.index_k_cache_mode is IndexKCacheMode.BF16:
            tgt_loc_flat = tgt_loc.view(-1).long()
            src_loc_flat = src_loc.view(-1).long()
            for index_k in self.buffer:
                if index_k.shape[0] == 0:
                    continue
                flat_index_k = index_k.view(-1, 1, self.pool.index_head_dim)
                flat_index_k[tgt_loc_flat] = flat_index_k[src_loc_flat]
            return

        super().move(tgt_loc, src_loc)

    def get_buffer(self, layer_id: int) -> torch.Tensor:
        if self.pool.layer_transfer_counter is not None:
            self.pool.layer_transfer_counter.wait_until(
                layer_id - self.pool.start_layer
            )
        return self.get_broadcastable_buffer(layer_id)

    def get_write_buffer(self, layer_id: int) -> torch.Tensor:
        if not self.prepare_remote_write(layer_id):
            # Correctness fallback for a missed prefetch. Broadcast history
            # before any rank writes this step's new Index-K into scratch.
            self.get_broadcastable_buffer(layer_id)

        if self.pool._is_layer_owned(layer_id):
            return super().get_write_buffer(layer_id)

        # Non-owner local buffers have zero rows. HCU still needs to run the
        # fused Q/K quant kernel to produce Q, so write this step's Index-K
        # directly into the already-prefetched remote scratch.
        return self.remote_buffer

    def commit_write_buffer(self, layer_id: int, loc: torch.Tensor) -> None:
        """Mirror an owner's fused Index-K writes into prefetched scratch."""

        if not self.pool._is_layer_owned(layer_id) or not self.prepare_remote_write(
            layer_id
        ):
            return

        loc = loc.reshape(-1)
        loc = loc[loc >= 0].to(torch.long)
        if loc.numel() == 0:
            return

        local_buffer = self.buffer[self.pool._get_indexer_cache_index(layer_id)]
        page_indices = torch.div(loc, self.pool.page_size, rounding_mode="floor")
        token_offsets = loc % self.pool.page_size
        k_bytes_per_page = self.pool.page_size * self.pool.index_head_dim
        scale_bytes_per_token = (
            self.pool.index_head_dim
            // self.pool.quant_block_size
            * torch.float32.itemsize
        )

        local_k = local_buffer[:, :k_bytes_per_page].view(
            -1, self.pool.page_size, self.pool.index_head_dim
        )
        remote_k = self.remote_buffer[:, :k_bytes_per_page].view(
            -1, self.pool.page_size, self.pool.index_head_dim
        )
        remote_k[page_indices, token_offsets] = local_k[page_indices, token_offsets]

        local_scale = local_buffer[:, k_bytes_per_page:].view(
            -1, self.pool.page_size, scale_bytes_per_token
        )
        remote_scale = self.remote_buffer[:, k_bytes_per_page:].view(
            -1, self.pool.page_size, scale_bytes_per_token
        )
        remote_scale[page_indices, token_offsets] = local_scale[
            page_indices, token_offsets
        ]

    def get_k_and_scale(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ):
        buf = self.get_buffer(layer_id)
        self.pool.prefetch_kv_buffer(layer_id)
        return index_buf_accessor.GetKAndS.execute(
            self.pool,
            buf,
            page_indices=page_indices,
            seq_len_tensor=seq_len_tensor,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
        )

    def store_quantized(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        if not self.prepare_remote_write(layer_id):
            self.get_broadcastable_buffer(layer_id)
        index_buf_accessor.SetKAndS.execute(
            pool=self.pool,
            buf=self.remote_buffer,
            loc=loc,
            index_k=index_k,
            index_k_scale=index_k_scale,
        )
        if self.pool._is_layer_owned(layer_id):
            super().store_quantized(layer_id, loc, index_k, index_k_scale)

    def store_bf16(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ) -> None:
        """Store BF16 Index-K into scratch and the owner's persistent shard."""

        if not self.prepare_remote_write(layer_id):
            self.get_broadcastable_buffer(layer_id)
        if index_k.dtype != self.pool.index_k_buffer_dtype:
            index_k = index_k.to(self.pool.index_k_buffer_dtype)

        page_indices = loc // self.pool.page_size
        token_offsets = loc % self.pool.page_size
        self.remote_buffer[page_indices, token_offsets] = index_k
        if self.pool._is_layer_owned(layer_id):
            cache_idx = self.pool._get_indexer_cache_index(layer_id)
            self.buffer[cache_idx][page_indices, token_offsets] = index_k

    def store_int8(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ) -> None:
        """Store current INT8 Index-K into scratch and the owner's shard."""

        if not self.prepare_remote_write(layer_id):
            self.get_broadcastable_buffer(layer_id)
        assert self.pool.index_k_int8_remote_aliases is not None
        remote_int8_k, remote_fp32_scales = self.pool.index_k_int8_remote_aliases
        quantize_and_store_index_k_int8(
            index_k,
            self.remote_buffer,
            loc,
            page_size=self.pool.page_size,
            int8_k=remote_int8_k,
            fp32_scales=remote_fp32_scales,
        )
        if not self.pool._is_layer_owned(layer_id):
            return

        cache_idx = self.pool._get_indexer_cache_index(layer_id)
        local_int8_k, local_fp32_scales = self.pool.index_k_int8_aliases[cache_idx]
        quantize_and_store_index_k_int8(
            index_k,
            self.buffer[cache_idx],
            loc,
            page_size=self.pool.page_size,
            int8_k=local_int8_k,
            fp32_scales=local_fp32_scales,
        )

    def finalize_pending(self, *, set_remote_layer_id: bool = True) -> None:
        if not self.pending_broadcast:
            return
        self.pool.device_module.current_stream().wait_event(self.pending_event)
        self.pending_broadcast = False
        if set_remote_layer_id and self.pending_layer_id is not None:
            self.remote_layer_id = self.pending_layer_id
        elif not set_remote_layer_id:
            self.remote_layer_id = None
        self.pending_layer_id = None

    def prefetch(
        self,
        layer_id: int,
        layer_transfer_counter: Optional[LayerDoneCounter] = None,
        layer_transfer_idx: Optional[int] = None,
        has_history: bool = True,
    ) -> None:
        """Broadcast a full Index-K buffer ahead of its indexer layer."""

        # GLM-5.2 only allocates caches for layers that really run the
        # indexer. skip-topk layers reuse prior results and need no collective.
        if layer_id not in self.pool.indexer_prefetch_layer_ids:
            return
        if self.remote_layer_id == layer_id:
            return
        if self.pending_broadcast:
            if self.pending_layer_id == layer_id:
                return
            self.finalize_pending(set_remote_layer_id=False)

        if not has_history:
            # This step's Index-K writes will populate every location read by
            # the indexer. Page zero was initialized when scratch was created.
            self.remote_layer_id = layer_id
            return

        cache_idx = self.pool._get_indexer_cache_index(layer_id)
        src_tensor = (
            self.buffer[cache_idx] if self.pool._is_layer_owned(layer_id) else None
        )
        transfer_counter = layer_transfer_counter or self.pool.layer_transfer_counter
        transfer_idx = (
            layer_transfer_idx
            if layer_transfer_idx is not None
            else layer_id - self.pool.start_layer
        )

        if self.pool.layer_broadcast_comm is None:
            if transfer_counter is not None:
                transfer_counter.wait_until(transfer_idx)
            self.pool._broadcast_tensor_from_owner(
                self.remote_buffer,
                layer_id,
                src_tensor=src_tensor,
                use_layer_broadcast_comm=True,
            )
            self.remote_layer_id = layer_id
            return

        current_stream = self.pool.device_module.current_stream()
        self.pool.kv_broadcast_stream.wait_stream(current_stream)
        with self.pool.device_module.stream(self.pool.kv_broadcast_stream):
            if transfer_counter is not None:
                transfer_counter.wait_until(transfer_idx)
            with torch.profiler.record_function(
                "layersplit_index_k_broadcast "
                f"layer={layer_id} bytes={self.remote_buffer.nbytes}"
            ):
                self.pool._broadcast_tensor_from_owner(
                    self.remote_buffer,
                    layer_id,
                    src_tensor=src_tensor,
                    use_layer_broadcast_comm=True,
                )
            self.pending_event.record()
        self.remote_layer_id = None
        self.pending_layer_id = layer_id
        self.pending_broadcast = True

    def prepare_remote_write(self, layer_id: int) -> bool:
        if self.pending_broadcast:
            self.finalize_pending(set_remote_layer_id=self.pending_layer_id == layer_id)
        return self.remote_layer_id == layer_id

    def invalidate(self, layer_id: int) -> None:
        if self.pending_broadcast and self.pending_layer_id == layer_id:
            self.finalize_pending(set_remote_layer_id=False)
        if self.remote_layer_id == layer_id:
            self.remote_layer_id = None

    def get_broadcastable_buffer(self, layer_id: int) -> torch.Tensor:
        if self.pending_broadcast:
            self.finalize_pending(set_remote_layer_id=self.pending_layer_id == layer_id)
        if self.remote_layer_id != layer_id:
            # Index-K and Main-KV share the dedicated communicator. A fallback
            # on the current stream must follow all side-stream collectives.
            self.pool._drain_pending_layer_broadcasts(discard_index=True)
            cache_idx = self.pool._get_indexer_cache_index(layer_id)
            src_tensor = (
                self.buffer[cache_idx] if self.pool._is_layer_owned(layer_id) else None
            )
            self.pool._broadcast_tensor_from_owner(
                self.remote_buffer,
                layer_id,
                src_tensor=src_tensor,
                use_layer_broadcast_comm=True,
            )
            self.remote_layer_id = layer_id
        return self.remote_buffer

    def state_buf_infos(self):
        owned_cache_indices = [
            cache_idx
            for cache_idx, layer_id in enumerate(self.pool.indexer_layer_ids)
            if self.pool._is_layer_owned(layer_id)
        ]
        data_ptrs = [self.buffer[i].data_ptr() for i in owned_cache_indices]
        data_lens = [self.buffer[i].nbytes for i in owned_cache_indices]
        item_lens = [self._item_len(i) for i in owned_cache_indices]
        return data_ptrs, data_lens, item_lens

    def cpu_copy(self, indices):
        page_indices = indices[:: self.pool.page_size] // self.pool.page_size
        torch.cuda.synchronize()
        index_k_cpu = []
        chunk_size = self.pool.cpu_offloading_chunk_size
        page_chunk_size = max(1, chunk_size // self.pool.page_size)
        for layer_id in range(self.pool.indexer_layer_num):
            index_k_cpu.append([])
            if self.buffer[layer_id].shape[0] == 0:
                continue
            for i in range(0, len(page_indices), page_chunk_size):
                chunk_page_indices = page_indices[i : i + page_chunk_size]
                idx_cpu = self.buffer[layer_id][chunk_page_indices].to(
                    "cpu", non_blocking=True
                )
                index_k_cpu[-1].append(idx_cpu)
        torch.cuda.synchronize()
        return index_k_cpu

    def load_cpu_copy(self, index_k_cpu, indices) -> None:
        page_indices = indices[:: self.pool.page_size] // self.pool.page_size
        torch.cuda.synchronize()
        chunk_size = self.pool.cpu_offloading_chunk_size
        page_chunk_size = max(1, chunk_size // self.pool.page_size)
        for layer_id in range(self.pool.indexer_layer_num):
            if self.buffer[layer_id].shape[0] == 0:
                continue
            for i in range(0, len(page_indices), page_chunk_size):
                chunk_page_indices = page_indices[i : i + page_chunk_size]
                idx_cpu = index_k_cpu[layer_id][i // page_chunk_size]
                assert idx_cpu.shape[0] == len(chunk_page_indices)
                idx_chunk = idx_cpu.to(self.buffer[layer_id].device, non_blocking=True)
                self.buffer[layer_id][chunk_page_indices] = idx_chunk
        torch.cuda.synchronize()


class LayerSplitDSATokenToKVPool(DSATokenToKVPool):
    """DSA KV pool that shards layers across CP ranks with owner-broadcast reads."""

    def __init__(
        self,
        *args,
        layer_shard_rank: int,
        layer_shard_size: int,
        indexer_prefetch_layer_ids: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        assert (
            layer_shard_rank is not None and layer_shard_size > 1
        ), "LayerSplitDSATokenToKVPool requires layer_shard_size > 1"
        self.layer_shard_rank = layer_shard_rank
        self.layer_shard_size = layer_shard_size
        self.layer_shard_enabled = True
        self.layer_broadcast_comm = None
        self.indexer_prefetch_layer_ids = (
            frozenset(indexer_prefetch_layer_ids)
            if indexer_prefetch_layer_ids is not None
            else None
        )
        super().__init__(*args, **kwargs)
        if self.indexer_prefetch_layer_ids is None:
            self.indexer_prefetch_layer_ids = frozenset(self.indexer_layer_ids)
        # First global layer index owned by this rank (used by PD transfer to
        # label the contiguous owned-buffer range).
        my_start, _ = self._owned_local_layer_range()
        self.layer_shard_start = self.start_layer + my_start

    # ---- layer ownership helpers ------------------------------------------

    def _local_layer_idx(self, layer_id: int) -> int:
        return layer_id - self.start_layer

    def _owned_local_layer_range(self) -> tuple[int, int]:
        return get_layer_shard_range(
            self.layer_shard_rank, self.layer_shard_size, self.layer_num
        )

    def _is_layer_owned(self, layer_id: int) -> bool:
        local_idx = self._local_layer_idx(layer_id)
        owned_start, owned_end = self._owned_local_layer_range()
        return owned_start <= local_idx < owned_end

    def _get_layer_owner_rank(self, layer_id: int) -> int:
        return get_layer_owner(
            self._local_layer_idx(layer_id), self.layer_shard_size, self.layer_num
        )

    def _log_layer_shard_plan(self) -> None:
        partitions = []
        for rank in range(self.layer_shard_size):
            st, ed = get_layer_shard_range(rank, self.layer_shard_size, self.layer_num)
            partitions.append(f"r{rank}:[{st},{ed})")
        my_start, my_end = self._owned_local_layer_range()
        logger.info(
            "Layer shard plan (continuous): "
            f"layer_num={self.layer_num}, shard_size={self.layer_shard_size}, "
            f"rank={self.layer_shard_rank}, local=[{my_start},{my_end}), "
            f"global=[{self.start_layer + my_start},{self.start_layer + my_end}), "
            f"partitions={'; '.join(partitions)}"
        )

    # ---- broadcast plumbing -----------------------------------------------

    def _init_layer_broadcast_comm(self) -> None:
        cp_group = get_parallel().attn_cp_group
        if cp_group.world_size <= 1 or cp_group.pynccl_comm is None:
            return

        from sglang.srt.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )

        self.layer_broadcast_comm = PyNcclCommunicator(
            group=cp_group.cpu_group,
            device=cp_group.device,
        )
        logger.info(
            "Initialized dedicated layer-shard broadcast NCCL communicator: "
            f"rank={cp_group.rank_in_group}, world_size={cp_group.world_size}"
        )

    def _broadcast_tensor_from_owner(
        self,
        tensor: torch.Tensor,
        layer_id: int,
        src_tensor: Optional[torch.Tensor] = None,
        use_layer_broadcast_comm: bool = False,
    ) -> torch.Tensor:
        owner_rank = self._get_layer_owner_rank(layer_id)
        if self.layer_shard_rank == owner_rank:
            assert src_tensor is not None
            if tensor.data_ptr() != src_tensor.data_ptr():
                tensor.copy_(src_tensor)

        cp_group = get_parallel().attn_cp_group
        comm = (
            self.layer_broadcast_comm
            if use_layer_broadcast_comm and self.layer_broadcast_comm is not None
            else cp_group.pynccl_comm
        )
        if comm is not None:
            # PyNcclCommunicator defaults to disabled=True (it is only enabled
            # inside CUDA-graph capture via change_state). Without re-enabling it
            # here, comm.broadcast() is a silent no-op and non-owner CP ranks read
            # stale remote buffers, corrupting layer-split attention. Mirror the
            # standard usage in parallel_state.py.
            with comm.change_state(enable=True):
                comm.broadcast(tensor, src=owner_rank)
        else:
            torch.distributed.broadcast(
                tensor,
                src=cp_group.ranks[owner_rank],
                group=cp_group.device_group,
            )
        return tensor

    # ---- buffer allocation (owned-only + remote scratch) ------------------

    def _create_buffers(self):
        self._log_layer_shard_plan()
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # Owned layers get the full buffer; non-owned layers allocate a
                # 0-row placeholder so ``kv_buffer`` stays index-aligned by layer.
                self.kv_buffer = [
                    torch.zeros(
                        (
                            (
                                (self.size + self.page_size)
                                if self._is_layer_owned(self.start_layer + i)
                                else 0
                            ),
                            1,
                            self.kv_cache_dim,
                        ),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for i in range(self.layer_num)
                ]
                self.remote_kv_buffer = torch.empty(
                    (self.size + self.page_size, 1, self.kv_cache_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                # Physical page 0 is translated to compact page 0 and may be
                # read for padded/dummy entries. Compact broadcasts start at
                # page 1, so initialize this page once instead of exposing the
                # uninitialized contents of the scratch allocation.
                self.remote_kv_buffer[: self.page_size].zero_()
                if (self.size + self.page_size) % self.page_size != 0:
                    raise ValueError(
                        "LayerSplit MLA KV buffer must contain whole pages: "
                        f"size={self.size}, page_size={self.page_size}"
                    )
                self.num_pool_pages = (self.size + self.page_size) // self.page_size
                self.physical_to_compact_main_kv_page = torch.full(
                    (self.num_pool_pages,),
                    -1,
                    dtype=torch.int32,
                    device=self.device,
                )
                # Keep this device-side to avoid a synchronous scalar H2D.
                self.physical_to_compact_main_kv_page[0:1].zero_()
                self._active_main_kv_page_plan: Optional[MainKVPagePlan] = None
                self._active_main_kv_batch_marker: Optional[
                    weakref.ReferenceType[Any]
                ] = None
                self._active_main_kv_compact_page_ids = torch.empty(
                    0, dtype=torch.long, device=self.device
                )
                self.remote_kv_layer_id: Optional[int] = None
                self.device_module = torch.get_device_module(self.device)
                self.kv_broadcast_stream = self.device_module.Stream()
                self.pending_remote_kv_layer_id: Optional[int] = None
                self.pending_remote_kv_broadcast = False
        self._init_layer_broadcast_comm()

    def _create_index_key_cache(self) -> IndexKeyCache:
        cache = LayerSplitIndexKeyCache(self, self.index_buf_size)
        self.index_k_buffer = (
            cache.buffer if self.index_k_cache_mode is IndexKCacheMode.BF16 else None
        )
        return cache

    @property
    def index_k_with_scale_buffer(self):
        if not self.use_scaled_index_k_cache:
            return None
        return self.index_key_cache.buffer

    def _clear_buffers(self):
        del self.kv_buffer
        del self.remote_kv_buffer
        del self.physical_to_compact_main_kv_page
        del self._active_main_kv_compact_page_ids
        self._active_main_kv_page_plan = None
        self._active_main_kv_batch_marker = None
        self._clear_index_k_buffers()

    # ---- MLA latent KV: owned-only writes, owner-broadcast reads ----------

    def get_kv_size_bytes(self):
        kv_size_bytes = 0
        for kv_cache in self.kv_buffer:
            kv_size_bytes += get_tensor_size_bytes(kv_cache)
        for index_k_cache in self.index_key_cache.buffer:
            kv_size_bytes += get_tensor_size_bytes(index_k_cache)
        if self.use_int8_index_k_cache:
            kv_size_bytes += get_tensor_size_bytes(self.index_k_dequant_workspace)
            kv_size_bytes += get_tensor_size_bytes(self.index_k_page_claims)
        return kv_size_bytes

    def get_contiguous_buf_infos(self):
        # Only report buffers owned by the current CP rank; non-owned layers
        # are empty and are pulled from their owner via PD transfer.
        owned_layer_ids = [
            i
            for i in range(self.layer_num)
            if self._is_layer_owned(self.start_layer + i)
        ]
        kv_data_ptrs = [self.kv_buffer[i].data_ptr() for i in owned_layer_ids]
        kv_data_lens = [self.kv_buffer[i].nbytes for i in owned_layer_ids]
        kv_item_lens = [
            self.kv_buffer[i][0].nbytes * self.page_size for i in owned_layer_ids
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_kv_layer_ids(self):
        """Global layer ids aligned with the owned-only KV transfer buffers."""
        return [
            self.start_layer + i
            for i in range(self.layer_num)
            if self._is_layer_owned(self.start_layer + i)
        ]

    def get_state_layer_ids(self):
        """Global layer ids aligned with the owned-only Index-K buffers."""
        return [
            layer_id
            for layer_id in self.indexer_layer_ids
            if self._is_layer_owned(layer_id)
        ]

    def get_kv_layer_ids(self):
        my_start, my_end = self._owned_local_layer_range()
        return list(range(self.start_layer + my_start, self.start_layer + my_end))

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        kv_buffer = self._get_broadcastable_kv_buffer(layer_id)
        if self.store_dtype != self.dtype:
            return kv_buffer.view(self.dtype)
        return kv_buffer

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        kv_buffer = self._get_broadcastable_kv_buffer(layer_id)
        if self.store_dtype != self.dtype:
            return kv_buffer[..., : self.kv_lora_rank].view(self.dtype)
        return kv_buffer[..., : self.kv_lora_rank]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        loc, _, _ = unwrap_write_loc(loc_info)
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MLA)")
        layer_id = layer.layer_id
        assert not self.dsa_kv_cache_store_fp8
        # A write invalidates any cached remote copy for this layer.
        if self.pending_remote_kv_layer_id == layer_id:
            self._finalize_pending_kv_broadcast(set_remote_layer_id=False)
        if self.remote_kv_layer_id == layer_id:
            self.remote_kv_layer_id = None
        if not self._is_layer_owned(layer_id):
            return
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k.view(
                self.store_dtype
            )
        else:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_mla_kv_buffer (MLA)")
        layer_id = layer.layer_id
        if self.pending_remote_kv_layer_id == layer_id:
            self._finalize_pending_kv_broadcast(set_remote_layer_id=True)
        remote_kv_updatable = self.remote_kv_layer_id == layer_id
        if remote_kv_updatable:
            remote_loc = self.translate_main_kv_loc_to_compact(loc)
            self._write_mla_kv_buffer(
                self.remote_kv_buffer, remote_loc, cache_k_nope, cache_k_rope
            )
        if not self._is_layer_owned(layer_id):
            return
        self._write_mla_kv_buffer(
            self.kv_buffer[layer_id - self.start_layer],
            loc,
            cache_k_nope,
            cache_k_rope,
        )
        if not remote_kv_updatable and self.remote_kv_layer_id == layer_id:
            self.remote_kv_layer_id = None

    def configure_main_kv_page_plan(
        self,
        page_plan: Optional[MainKVPagePlan],
        batch_marker: Any,
    ) -> None:
        """Install one batch's physical-to-compact Main-KV mapping.

        History pages occupy compact slots ``[1, 1 + N_history)`` so the
        transmitted payload is a single contiguous view. Pages used only by
        this step follow them and are populated locally after CP AllGather.
        """

        if (
            self._active_main_kv_batch_marker is not None
            and self._active_main_kv_batch_marker() is batch_marker
            and self._active_main_kv_page_plan is page_plan
        ):
            return

        # A late side-stream broadcast must not write through state owned by
        # the next ForwardBatch. Index scratch also depends on pool contents
        # that may have changed through allocation, movement, or HiCache load.
        self._drain_pending_layer_broadcasts(
            discard_index=True,
            discard_main=True,
        )
        self.index_key_cache.remote_layer_id = None
        self.remote_kv_layer_id = None
        self.physical_to_compact_main_kv_page.fill_(-1)
        # Keep this device-side to avoid a synchronous scalar H2D.
        self.physical_to_compact_main_kv_page[0:1].zero_()
        self._active_main_kv_batch_marker = None
        self._active_main_kv_page_plan = None
        self._active_main_kv_compact_page_ids = torch.empty(
            0, dtype=torch.long, device=self.device
        )
        if page_plan is None:
            self._active_main_kv_batch_marker = weakref.ref(batch_marker)
            return

        history_page_ids = page_plan.history_page_ids.to(
            device=self.device, dtype=torch.long
        ).contiguous()
        all_page_ids = page_plan.all_page_ids.to(
            device=self.device, dtype=torch.long
        ).contiguous()
        for name, page_ids in (
            ("history_page_ids", history_page_ids),
            ("all_page_ids", all_page_ids),
        ):
            if page_ids.numel() == 0:
                continue
            invalid_page_ids = page_ids[
                (page_ids <= 0) | (page_ids >= self.num_pool_pages)
            ]
            if invalid_page_ids.numel() != 0:
                raise RuntimeError(
                    "LayerSplit compact Main-KV page plan contains an out-of-range "
                    f"{name} entry: page_id={int(invalid_page_ids[0].item())}, "
                    f"valid_range=[1, {self.num_pool_pages})"
                )
        if all_page_ids.numel() > self.num_pool_pages - 1:
            raise RuntimeError(
                "LayerSplit compact Main-KV layout exceeds the remote buffer: "
                f"pages={all_page_ids.numel()}, "
                f"capacity_pages={self.num_pool_pages - 1}"
            )

        history_compact_ids = torch.arange(
            1,
            history_page_ids.numel() + 1,
            dtype=torch.int32,
            device=self.device,
        )
        self.physical_to_compact_main_kv_page.index_copy_(
            0, history_page_ids, history_compact_ids
        )

        current_only_mask = (
            self.physical_to_compact_main_kv_page.index_select(0, all_page_ids) < 0
        )
        current_only_page_ids = all_page_ids[current_only_mask]
        current_compact_ids = torch.arange(
            history_page_ids.numel() + 1,
            history_page_ids.numel() + current_only_page_ids.numel() + 1,
            dtype=torch.int32,
            device=self.device,
        )
        self.physical_to_compact_main_kv_page.index_copy_(
            0, current_only_page_ids, current_compact_ids
        )
        self._active_main_kv_compact_page_ids = torch.cat(
            (history_page_ids, current_only_page_ids)
        ).contiguous()
        self._active_main_kv_page_plan = page_plan
        self._active_main_kv_batch_marker = weakref.ref(batch_marker)

    def translate_main_kv_loc_to_compact(self, loc: torch.Tensor) -> torch.Tensor:
        """Translate physical token locations for the active compact buffer."""

        if self._active_main_kv_page_plan is None:
            return loc

        valid = loc >= 0
        safe_loc = loc.clamp_min(0)
        physical_page = torch.div(safe_loc, self.page_size, rounding_mode="floor").to(
            torch.long
        )
        page_offset = safe_loc % self.page_size
        compact_page = self.physical_to_compact_main_kv_page.index_select(
            0, physical_page.reshape(-1)
        ).reshape(physical_page.shape)
        compact_loc = compact_page.to(loc.dtype) * self.page_size + page_offset
        return torch.where(
            valid & (compact_page >= 0),
            compact_loc,
            torch.full_like(compact_loc, -1),
        )

    def _broadcast_compact_main_kv_pages(
        self,
        layer_id: int,
        src_tensor: Optional[torch.Tensor],
        *,
        include_current: bool = False,
    ) -> None:
        page_plan = self._active_main_kv_page_plan
        assert page_plan is not None
        num_pages = (
            self._active_main_kv_compact_page_ids.numel()
            if include_current
            else page_plan.history_page_ids.numel()
        )
        if num_pages == 0:
            return

        page_ids = self._active_main_kv_compact_page_ids[:num_pages]
        remote_pages = self.remote_kv_buffer.view(
            self.num_pool_pages,
            self.page_size,
            1,
            self.kv_cache_dim,
        )
        payload = remote_pages[1 : num_pages + 1]
        bytes_to_broadcast = payload.numel() * payload.element_size()

        if self._is_layer_owned(layer_id):
            assert src_tensor is not None
            local_pages = src_tensor.view(
                self.num_pool_pages,
                self.page_size,
                1,
                self.kv_cache_dim,
            )
            with torch.profiler.record_function(
                "layersplit_main_kv_pack "
                f"layer={layer_id} pages={num_pages} "
                f"bytes={bytes_to_broadcast}"
            ):
                torch.index_select(local_pages, 0, page_ids, out=payload)

        with torch.profiler.record_function(
            "layersplit_main_kv_broadcast "
            f"layer={layer_id} pages={num_pages} "
            f"bytes={bytes_to_broadcast}"
        ):
            self._broadcast_tensor_from_owner(
                payload,
                layer_id,
                src_tensor=(payload if self._is_layer_owned(layer_id) else None),
                use_layer_broadcast_comm=True,
            )

    def _finalize_pending_kv_broadcast(
        self, *, set_remote_layer_id: bool = True
    ) -> None:
        if not self.pending_remote_kv_broadcast:
            return
        self.device_module.current_stream().wait_stream(self.kv_broadcast_stream)
        self.pending_remote_kv_broadcast = False
        if set_remote_layer_id and self.pending_remote_kv_layer_id is not None:
            self.remote_kv_layer_id = self.pending_remote_kv_layer_id
        elif not set_remote_layer_id:
            self.remote_kv_layer_id = None
        self.pending_remote_kv_layer_id = None

    def _drain_pending_layer_broadcasts(
        self,
        *,
        discard_index: bool = False,
        discard_main: bool = False,
    ) -> None:
        """Order synchronous fallbacks after side-stream collectives."""

        self.index_key_cache.finalize_pending(set_remote_layer_id=not discard_index)
        self._finalize_pending_kv_broadcast(set_remote_layer_id=not discard_main)

    def prefetch_index_buffer(
        self,
        layer_id: int,
        layer_transfer_counter: Optional[LayerDoneCounter] = None,
        layer_transfer_idx: Optional[int] = None,
        has_history: bool = True,
    ) -> None:
        self.index_key_cache.prefetch(
            layer_id,
            layer_transfer_counter=layer_transfer_counter,
            layer_transfer_idx=layer_transfer_idx,
            has_history=has_history,
        )

    def prefetch_kv_buffer(
        self,
        layer_id: int,
        layer_transfer_counter: Optional[LayerDoneCounter] = None,
        layer_transfer_idx: Optional[int] = None,
        has_history: bool = True,
    ) -> None:
        """Kick off an async owner-broadcast of ``layer_id``'s latent KV.

        Called ahead of the layer's attention so the remote scratch buffer is
        ready by the time a non-owner rank reads it (see the prefetch wiring in
        ``DeepseekV2DecoderLayer``).
        """
        if self.remote_kv_layer_id == layer_id:
            return
        if self.pending_remote_kv_broadcast:
            if self.pending_remote_kv_layer_id == layer_id:
                return
            self._finalize_pending_kv_broadcast(set_remote_layer_id=False)

        compact_history_is_empty = (
            self._active_main_kv_page_plan is not None
            and self._active_main_kv_page_plan.history_page_ids.numel() == 0
        )
        if compact_history_is_empty or (
            self._active_main_kv_page_plan is None and not has_history
        ):
            # The gathered K will populate this scratch directly. Broadcasting
            # an empty history would only duplicate data movement.
            self.remote_kv_layer_id = layer_id
            return

        local_idx = self._local_layer_idx(layer_id)
        src_tensor = (
            self.kv_buffer[local_idx] if self._is_layer_owned(layer_id) else None
        )
        transfer_counter = layer_transfer_counter or self.layer_transfer_counter
        transfer_idx = (
            layer_transfer_idx if layer_transfer_idx is not None else local_idx
        )
        if self.layer_broadcast_comm is None:
            if transfer_counter is not None:
                transfer_counter.wait_until(transfer_idx)
            if self._active_main_kv_page_plan is not None:
                self._broadcast_compact_main_kv_pages(layer_id, src_tensor)
            else:
                self._broadcast_tensor_from_owner(
                    self.remote_kv_buffer,
                    layer_id,
                    src_tensor=src_tensor,
                    use_layer_broadcast_comm=True,
                )
            self.remote_kv_layer_id = layer_id
            return

        self.kv_broadcast_stream.wait_stream(self.device_module.current_stream())
        with self.device_module.stream(self.kv_broadcast_stream):
            if transfer_counter is not None:
                transfer_counter.wait_until(transfer_idx)
            if self._active_main_kv_page_plan is not None:
                self._broadcast_compact_main_kv_pages(layer_id, src_tensor)
            else:
                self._broadcast_tensor_from_owner(
                    self.remote_kv_buffer,
                    layer_id,
                    src_tensor=src_tensor,
                    use_layer_broadcast_comm=True,
                )
        self.pending_remote_kv_layer_id = layer_id
        self.pending_remote_kv_broadcast = True

    def _get_broadcastable_kv_buffer(self, layer_id: int) -> torch.Tensor:
        if self.pending_remote_kv_broadcast:
            self._finalize_pending_kv_broadcast(
                set_remote_layer_id=self.pending_remote_kv_layer_id == layer_id
            )
        if self.remote_kv_layer_id != layer_id:
            self._drain_pending_layer_broadcasts(discard_main=True)
            local_idx = self._local_layer_idx(layer_id)
            src_tensor = (
                self.kv_buffer[local_idx] if self._is_layer_owned(layer_id) else None
            )
            if self._active_main_kv_page_plan is not None:
                # A missed prefetch reaches this fallback after current KV may
                # already have been produced, so bootstrap every compact page.
                self._broadcast_compact_main_kv_pages(
                    layer_id,
                    src_tensor,
                    include_current=True,
                )
            else:
                self._broadcast_tensor_from_owner(
                    self.remote_kv_buffer,
                    layer_id,
                    src_tensor=src_tensor,
                    use_layer_broadcast_comm=True,
                )
            self.remote_kv_layer_id = layer_id
        return self.remote_kv_buffer

    def invalidate_remote_kv_buffer_for_layer(self, layer_id: int) -> None:
        """Invalidate a broadcast copy before HiCache restores its owner layer."""

        if self.pending_remote_kv_layer_id == layer_id:
            self._finalize_pending_kv_broadcast(set_remote_layer_id=False)
        if self.remote_kv_layer_id == layer_id:
            self.remote_kv_layer_id = None

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        return super().get_mla_kv_buffer(
            layer,
            self.translate_main_kv_loc_to_compact(loc),
            dst_dtype,
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        size_limit = self.size + self.page_size
        maybe_detect_oob(tgt_loc, 0, size_limit, "move_kv_cache tgt_loc")
        maybe_detect_oob(src_loc, 0, size_limit, "move_kv_cache src_loc")
        if tgt_loc.numel() == 0:
            return
        # Compaction mutates owner storage. Finish any side-stream reads before
        # moving it, and do not retain scratch identities across the mutation.
        self._drain_pending_layer_broadcasts(
            discard_index=True,
            discard_main=True,
        )
        self.index_key_cache.remote_layer_id = None
        self.remote_kv_layer_id = None
        tgt_loc_flat = tgt_loc.view(-1).long()
        src_loc_flat = src_loc.view(-1).long()
        for kv_cache in self.kv_buffer:
            if kv_cache.shape[0] == 0:
                continue
            kv_cache[tgt_loc_flat] = kv_cache[src_loc_flat]
        self.index_key_cache.move(tgt_loc, src_loc)

    # ---- DSA indexer buffer: owned-only writes, owner-broadcast reads -----

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        assert self.index_k_buffer is not None, "BF16 index K cache is not enabled"
        return self.index_key_cache.get_buffer(layer_id)

    def set_index_k_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ) -> None:
        assert self.index_k_buffer is not None, "BF16 index K cache is not enabled"
        self.index_key_cache.store_bf16(layer_id, loc, index_k)

    def get_broadcastable_index_k_with_scale_buffer(
        self, layer_id: int
    ) -> torch.Tensor:
        return self.index_key_cache.get_buffer(layer_id)

    def commit_index_k_with_scale_write_buffer(
        self, layer_id: int, loc: torch.Tensor
    ) -> None:
        self.index_key_cache.commit_write_buffer(layer_id, loc)

    def set_index_k_int8_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ) -> None:
        assert self.use_int8_index_k_cache, "INT8 index K cache is not enabled"
        self.index_key_cache.store_int8(layer_id, loc, index_k)

    def invalidate_index_buffer_for_layer(self, layer_id: int) -> None:
        self.index_key_cache.invalidate(layer_id)

    def _get_broadcastable_index_buffer(self, layer_id: int) -> torch.Tensor:
        return self.index_key_cache.get_broadcastable_buffer(layer_id)

    # ---- HiCache CPU offload: skip empty (non-owned) layers ---------------

    def get_cpu_copy(self, indices, mamba_indices=None):
        from sglang.srt.utils import current_platform

        current_platform.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            if self.kv_buffer[layer_id].shape[0] == 0:
                continue
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = self.kv_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append(kv_cpu)
        current_platform.synchronize()

        return {"kv": kv_cache_cpu, "index_k": self.index_key_cache.cpu_copy(indices)}

    def load_cpu_copy(self, kv_cache_cpu_dict, indices, mamba_indices=None):
        from sglang.srt.utils import current_platform

        # A resume overwrites owner storage outside the mirrored write path.
        # Drain broadcasts first so their completion cannot later make a stale
        # scratch layer appear valid.
        self._drain_pending_layer_broadcasts(
            discard_index=True,
            discard_main=True,
        )
        self.index_key_cache.remote_layer_id = None
        self.remote_kv_layer_id = None
        kv_cache_cpu = kv_cache_cpu_dict["kv"]
        current_platform.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            if self.kv_buffer[layer_id].shape[0] == 0:
                continue
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = kv_cache_cpu[layer_id][i // chunk_size]
                assert kv_cpu.shape[0] == len(chunk_indices)
                kv_chunk = kv_cpu.to(self.kv_buffer[layer_id].device, non_blocking=True)
                self.kv_buffer[layer_id][chunk_indices] = kv_chunk
        current_platform.synchronize()

        self.index_key_cache.load_cpu_copy(kv_cache_cpu_dict["index_k"], indices)
