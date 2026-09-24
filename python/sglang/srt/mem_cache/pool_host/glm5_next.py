"""HiCache adapters for the HCU GLM physical KPool and draft LayerSplit layout."""

from __future__ import annotations

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.pool_host import dsa, mla
from sglang.srt.mem_cache.pool_host.common import kernel_accessible_host_ptr
from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost


def is_hcu_glm_pool(pool):
    return bool(getattr(pool, "is_hcu_glm5_next_pool", False))


def index_bytes_per_page(pool):
    if pool.use_fp8_index_k_cache:
        row_bytes = (
            pool.index_head_dim + pool.index_head_dim // pool.quant_block_size * 4
        )
        return row_bytes * pool.slots_per_page
    return pool.index_head_dim * pool.index_k_buffer_dtype.itemsize * pool.page_size


class _HcuLayerOwnership:
    def _init_hcu_transfer_ptrs(self):
        if self.layout != "layer_first":
            return
        self._hcu_is_indexer = hasattr(self, "indexer_page_stride_size")
        device_refs = []
        host_refs = self.index_k_data_refs if self._hcu_is_indexer else self.data_refs
        host_ptrs = []
        for pool in (self.device_pool, *self.mtp_draft_device_pools):
            buffers = (
                self._get_device_index_buffers(pool)
                if self._hcu_is_indexer
                else pool.kv_buffer
            )
            for layer_id in self._owned_device_layer_ids(pool):
                host_layer_id, _ = self._transfer_layer_ids(
                    pool, layer_id, is_draft=pool is not self.device_pool
                )
                device_refs.append(buffers[layer_id])
                host_ptrs.append(kernel_accessible_host_ptr(host_refs[host_layer_id]))
        self._hcu_device_data_ptrs = torch.tensor(
            [x.data_ptr() for x in device_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        # Main reserves the worst-rank layer count, so host storage may include
        # padding layers. Pair only owned target/draft layers for the copy kernel.
        self._hcu_host_data_ptrs = torch.tensor(
            host_ptrs, dtype=torch.uint64, device=self.device_pool.device
        )

    def _backup_hcu_layer_first(self, host_indices, device_indices):
        from sgl_kernel.kvcacheio import transfer_kv_all_layer_mla_lf_lf_D2H_hcu

        if self._hcu_is_indexer:
            host_indices, device_indices = self._get_indexer_page_indices(
                host_indices, device_indices
            )
            item_size = self.indexer_page_stride_size
        else:
            item_size = self.token_stride_size
        host_ptrs = self._hcu_host_data_ptrs
        if host_ptrs.numel() == 0 or host_indices.numel() == 0:
            return
        transfer_kv_all_layer_mla_lf_lf_D2H_hcu(
            src_layers=self._hcu_device_data_ptrs,
            dst_layers=host_ptrs,
            src_indices=device_indices,
            dst_indices=host_indices,
            item_size=item_size,
            num_layers=host_ptrs.numel(),
        )

    def _draft_layer_num(self):
        return sum(
            len(self._owned_device_layer_ids(pool))
            for pool in self.mtp_draft_device_pools
        )

    def _transfer_layer_ids(self, device_pool, layer_id, is_draft):
        device_layer_id = 0 if is_draft else layer_id
        if not self._is_device_layer_owned(device_pool, device_layer_id):
            return None
        if not is_draft:
            return self._host_layer_index(layer_id), device_layer_id
        host_layer_id = self.target_layer_num
        for pool in self.mtp_draft_device_pools:
            if pool is device_pool:
                return host_layer_id, device_layer_id
            host_layer_id += len(self._owned_device_layer_ids(pool))
        raise ValueError("Draft pool is not registered with the HCU host cache")

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if (
            io_backend == "kernel"
            and self.layout == "layer_first"
            and envs.SGLANG_USE_HICACHE_OPTIMIZATION_KERNEL.get()
            and not self.can_use_jit
            and (
                getattr(self, "indexer_page_stride_size", None) is not None
                or self.page_size == 64
            )
        ):
            self._backup_hcu_layer_first(host_indices, device_indices)
            return
        if not self._is_device_layer_sharded(device_pool) and not any(
            self._is_device_layer_sharded(pool) for pool in self.mtp_draft_device_pools
        ):
            return super().backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend
            )
        for layer_id in self._owned_device_layer_ids(device_pool):
            self._backup_from_device_per_layer(
                device_pool, host_indices, device_indices, layer_id, io_backend
            )
        for depth, pool in enumerate(self.mtp_draft_device_pools):
            if self._owned_device_layer_ids(pool):
                self._backup_from_device_per_layer(
                    pool,
                    host_indices,
                    device_indices,
                    self.device_pool.layer_num + depth,
                    io_backend,
                    is_draft=True,
                )


class Glm5NextMLAPoolHost(_HcuLayerOwnership, MLATokenToKVPoolHost):
    def __init__(self, device_pool, *args, **kwargs):
        kwargs["override_kv_cache_dim"] = device_pool.kv_cache_dim
        for pool in kwargs.get("mtp_draft_device_pools", ()):
            if (
                pool.page_size,
                pool.kv_cache_dim,
                pool.store_dtype,
                index_bytes_per_page(pool),
            ) != (
                device_pool.page_size,
                device_pool.kv_cache_dim,
                device_pool.store_dtype,
                index_bytes_per_page(device_pool),
            ):
                raise ValueError("Target and draft HCU HiCache geometry must match")
        super().__init__(device_pool, *args, **kwargs)
        self.data_ptrs = torch.tensor(
            [kernel_accessible_host_ptr(x) for x in self.data_refs],
            dtype=torch.uint64,
            device=device_pool.device,
        )

        self._init_hcu_transfer_ptrs()

    def get_contiguous_buf_infos(self):
        return (
            [x.data_ptr() for x in self.data_refs],
            [x.nbytes for x in self.data_refs],
            [self.token_stride_size * self.page_size] * len(self.data_refs),
        )

    def get_size_per_token(self):
        super().get_size_per_token()
        self.layer_num = self.target_layer_num + self._draft_layer_num()
        # Reserve target + draft + indexer before allocating the anchor. The
        # worst-rank count keeps fixed-size host slot capacity identical across
        # CP ranks even though only the draft owner allocates its host layer.
        budget_layers = self.target_layer_num + sum(
            self._effective_host_layer_num(pool) for pool in self.mtp_draft_device_pools
        )
        return budget_layers * (
            self.kv_cache_dim * self.dtype.itemsize
            + index_bytes_per_page(self.device_pool) / self.page_size
        )

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        host_indices = self.maybe_dcp_kernel_indices(host_indices)
        device_indices = self.maybe_dcp_kernel_indices(device_indices)
        device_pool.invalidate_remote_kv_buffer_for_layer(
            device_pool.start_layer + (0 if is_draft else layer_id)
        )
        layer_ids = self._transfer_layer_ids(device_pool, layer_id, is_draft)
        if layer_ids is None:
            return
        host_layer_id, device_layer_id = layer_ids

        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    mla.jit_transfer_hicache_one_layer_mla(
                        cache_dst=device_pool.kv_buffer[device_layer_id],
                        cache_src=self.kv_buffer[host_layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.kv_cache_dim,
                    )
                elif envs.SGLANG_USE_HICACHE_OPTIMIZATION_KERNEL.get():
                    from sgl_kernel.kvcacheio import (
                        transfer_kv_per_layer_mla_lf_lf_H2D_hcu,
                    )

                    transfer_kv_per_layer_mla_lf_lf_H2D_hcu(
                        src=self.kv_buffer[host_layer_id],
                        dst=device_pool.kv_buffer[device_layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        item_size=self.token_stride_size,
                        page_size=self.page_size,
                    )
                else:
                    mla.transfer_kv_per_layer_mla(
                        src=self.kv_buffer[host_layer_id],
                        dst=device_pool.kv_buffer[device_layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        item_size=self.token_stride_size,
                    )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    mla.jit_transfer_hicache_one_layer_mla(
                        cache_dst=device_pool.kv_buffer[device_layer_id],
                        cache_src=self.data_refs[host_layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.kv_cache_dim,
                    )
                else:
                    mla.transfer_kv_per_layer_mla_pf_lf(
                        src=self.kv_buffer,
                        dst=device_pool.kv_buffer[device_layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        layer_id=host_layer_id,
                        item_size=self.token_stride_size,
                        src_layout_dim=self.layout_dim,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                mla.transfer_kv_direct(
                    src_layers=[self.kv_buffer[host_layer_id]],
                    dst_layers=[device_pool.kv_buffer[device_layer_id]],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                mla.transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.kv_buffer],
                    dst_ptrs=[device_pool.kv_buffer[device_layer_id]],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=host_layer_id,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "kernel_ascend":
            if self.layout == "page_first_kv_split":
                # Ascend-specific: transfer KV data for all layers when layer_id == 0
                if device_layer_id == 0:
                    mla.transfer_kv_dim_exchange(
                        device_indices=device_indices,
                        host_indices=host_indices,
                        device_k=device_pool.k_buffer,
                        host_k=self.k_buffer,
                        device_v=device_pool.v_buffer,
                        host_v=self.v_buffer,
                        device_index_k=device_pool.index_k_buffer,
                        host_index_k=self.index_k_buffer,
                        page_size=self.page_size,
                        direction=mla.TransferDirection.H2D,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def _backup_from_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        # Indices arrive already translated by backup_from_device_all_layer.
        layer_ids = self._transfer_layer_ids(device_pool, layer_id, is_draft)
        if layer_ids is None:
            return
        host_layer_id, device_layer_id = layer_ids

        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    mla.jit_transfer_hicache_one_layer_mla(
                        cache_dst=self.kv_buffer[host_layer_id],
                        cache_src=device_pool.kv_buffer[device_layer_id],
                        indices_dst=host_indices,
                        indices_src=device_indices,
                        element_dim=self.kv_cache_dim,
                    )
                else:
                    mla.transfer_kv_per_layer_mla(
                        src=device_pool.kv_buffer[device_layer_id],
                        dst=self.kv_buffer[host_layer_id],
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                    )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    mla.jit_transfer_hicache_one_layer_mla(
                        cache_dst=self.data_refs[host_layer_id],
                        cache_src=device_pool.kv_buffer[device_layer_id],
                        indices_dst=host_indices,
                        indices_src=device_indices,
                        element_dim=self.kv_cache_dim,
                    )
                else:
                    raise ValueError(
                        "Layer-sharded MLA HiCache backup with page_first layout "
                        "requires the JIT one-layer kernel."
                    )
            else:
                raise ValueError(
                    f"Layer-sharded HiCache backup does not support layout: {self.layout}"
                )
        elif io_backend == "direct":
            if self.layout == "layer_first":
                mla.transfer_kv_direct(
                    src_layers=[device_pool.kv_buffer[device_layer_id]],
                    dst_layers=[self.kv_buffer[host_layer_id]],
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(
                    "Layer-sharded direct HiCache backup only supports "
                    f"layer_first layout, got {self.layout}"
                )
        else:
            raise ValueError(
                f"Layer-sharded HiCache backup does not support IO backend: {io_backend}"
            )


class Glm5NextIndexerPoolHost(_HcuLayerOwnership, DSAIndexerPoolHost):
    def __init__(
        self,
        device_pool: dsa.DSATokenToKVPool,
        anchor_host: MLATokenToKVPoolHost,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ):
        self.device_pool = device_pool
        self.page_size = anchor_host.page_size
        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = dsa.get_allocator_from_storage(allocator_type)
        self.dtype = device_pool.store_dtype
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer
        self.target_layer_num = self._effective_host_layer_num()
        self.mtp_draft_device_pools = anchor_host.mtp_draft_device_pools
        self.layer_num = self.target_layer_num + self._draft_layer_num()

        self.index_head_dim = device_pool.index_head_dim
        self.indexer_quant_block_size = device_pool.quant_block_size
        self.indexer_dtype = dsa.DSATokenToKVPool.index_k_with_scale_buffer_dtype
        self.indexer_size_per_token = index_bytes_per_page(device_pool) / self.page_size
        self.size = anchor_host.size
        self.page_num = anchor_host.page_num

        self.indexer_page_stride_size = index_bytes_per_page(device_pool)
        self.indexer_layout_dim = self.indexer_page_stride_size * self.layer_num
        self.indexer_page_num = (self.size + self.page_size + 1) // self.page_size
        self.size_per_token = (
            self.indexer_size_per_token * self.layer_num * self.indexer_dtype.itemsize
        )

        buf_elem_size = self.page_num * self.layer_num * self.indexer_page_stride_size
        requested_bytes = buf_elem_size * self.indexer_dtype.itemsize
        available_bytes = dsa.host_memory_budget_bytes()
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory for DSA indexer hierarchical cache. "
                f"Requesting {requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free."
            )
        draft_layer_num = self.layer_num - self.target_layer_num
        if draft_layer_num > 0:
            dsa.logger.info(
                "Allocating %.2f GB host memory for DSA indexer (layout=%s), "
                "packed MTP layers: "
                "target_layers=%d, draft_layers=%d, total_layers=%d.",
                requested_bytes / 1e9,
                layout,
                self.target_layer_num,
                draft_layer_num,
                self.layer_num,
            )
        else:
            dsa.logger.info(
                "Allocating %.2f GB host memory for DSA indexer (layout=%s).",
                requested_bytes / 1e9,
                layout,
            )
        self.init_kv_buffer()
        self.can_use_jit = False
        self.can_use_write_back_jit = False
        self._init_write_back_staging_buffers()
        self.lock = dsa.threading.RLock()
        self.clear()

        self._init_hcu_transfer_ptrs()

    def _get_device_index_buffers(self, pool):
        if pool.use_fp8_index_k_cache:
            return pool.index_k_with_scale_buffer
        row_bytes = index_bytes_per_page(pool)
        return [
            buffer.view(torch.uint8).view(buffer.shape[0], row_bytes)
            for buffer in pool.index_k_buffer
        ]

    def init_kv_buffer(self):
        alloc_func = dsa.ALLOC_MEMORY_FUNCS[self.device_pool.device]
        device_pools = (self.device_pool, *self.mtp_draft_device_pools)
        self.packed_device_index_buffers = [
            buffer
            for pool in device_pools
            for buffer in self._get_device_index_buffers(pool)
        ]
        self.index_k_device_ptrs = torch.tensor(
            [x.data_ptr() for x in self.packed_device_index_buffers],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        if self.layout == "layer_first":
            self.index_k_with_scale_buffer = alloc_func(
                (self.layer_num, self.indexer_page_num, self.indexer_page_stride_size),
                dtype=self.indexer_dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.index_k_data_refs = [
                self.index_k_with_scale_buffer[i] for i in range(self.layer_num)
            ]
            self.index_k_data_ptrs = torch.tensor(
                [kernel_accessible_host_ptr(x) for x in self.index_k_data_refs],
                dtype=torch.uint64,
                device=self.device_pool.device,
            )
        elif self.layout in ["page_first", "page_first_direct"]:
            self.index_k_with_scale_buffer = alloc_func(
                (
                    self.indexer_page_num,
                    self.layer_num,
                    1,
                    self.indexer_page_stride_size,
                ),
                dtype=self.indexer_dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
                registration_granularity_bytes=self.indexer_layout_dim,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        device_pool.invalidate_index_buffer_for_layer(
            device_pool.start_layer + (0 if is_draft else layer_id)
        )
        layer_ids = self._transfer_layer_ids(device_pool, layer_id, is_draft)
        if layer_ids is None:
            return
        host_layer_id, device_layer_id = layer_ids

        host_page_indices, device_page_indices = self._get_indexer_page_indices(
            host_indices, device_indices
        )
        use_kernel = io_backend == "kernel" and self.indexer_page_stride_size % 8 == 0
        if use_kernel:
            if self.layout == "layer_first":
                if envs.SGLANG_USE_HICACHE_OPTIMIZATION_KERNEL.get():
                    from sgl_kernel.kvcacheio import (
                        transfer_kv_per_layer_mla_lf_lf_H2D_hcu,
                    )

                    transfer_kv_per_layer_mla_lf_lf_H2D_hcu(
                        src=self.index_k_with_scale_buffer[host_layer_id],
                        dst=self._get_device_index_buffers(device_pool)[
                            device_layer_id
                        ],
                        src_indices=host_page_indices,
                        dst_indices=device_page_indices,
                        item_size=self.indexer_page_stride_size,
                        page_size=1,
                    )
                    return
                dsa.transfer_kv_per_layer_mla(
                    src=self.index_k_with_scale_buffer[host_layer_id],
                    dst=self._get_device_index_buffers(device_pool)[device_layer_id],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    item_size=self.indexer_page_stride_size,
                )
            elif self.layout == "page_first":
                dsa.transfer_kv_per_layer_mla_pf_lf(
                    src=self.index_k_with_scale_buffer,
                    dst=self._get_device_index_buffers(device_pool)[device_layer_id],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    layer_id=host_layer_id,
                    item_size=self.indexer_page_stride_size,
                    src_layout_dim=self.indexer_layout_dim,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                dsa.transfer_kv_direct(
                    src_layers=[self.index_k_with_scale_buffer[host_layer_id]],
                    dst_layers=[
                        self._get_device_index_buffers(device_pool)[device_layer_id]
                    ],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    page_size=1,
                )
            elif self.layout == "page_first_direct":
                dsa.transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.index_k_with_scale_buffer],
                    dst_ptrs=[
                        self._get_device_index_buffers(device_pool)[device_layer_id]
                    ],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    layer_id=host_layer_id,
                    page_size=1,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def _backup_from_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        layer_ids = self._transfer_layer_ids(device_pool, layer_id, is_draft)
        if layer_ids is None:
            return
        host_layer_id, device_layer_id = layer_ids

        host_page_indices, device_page_indices = self._get_indexer_page_indices(
            host_indices, device_indices
        )
        use_kernel = io_backend == "kernel" and self.indexer_page_stride_size % 8 == 0
        if use_kernel:
            if self.layout == "layer_first":
                dsa.transfer_kv_per_layer_mla(
                    src=self._get_device_index_buffers(device_pool)[device_layer_id],
                    dst=self.index_k_with_scale_buffer[host_layer_id],
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    item_size=self.indexer_page_stride_size,
                )
            elif self.layout == "page_first":
                raise ValueError(
                    "Layer-sharded DSA indexer HiCache backup with page_first "
                    "layout is not supported without a per-layer LF->PF kernel."
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                dsa.transfer_kv_direct(
                    src_layers=[
                        self._get_device_index_buffers(device_pool)[device_layer_id]
                    ],
                    dst_layers=[self.index_k_with_scale_buffer[host_layer_id]],
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    page_size=1,
                )
            else:
                raise ValueError(
                    "Layer-sharded direct DSA indexer backup only supports "
                    f"layer_first layout, got {self.layout}"
                )
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")


def get_mla_host_pool_cls(pool):
    return Glm5NextMLAPoolHost if is_hcu_glm_pool(pool) else MLATokenToKVPoolHost


def get_dsa_host_pool_cls(pool):
    return Glm5NextIndexerPoolHost if is_hcu_glm_pool(pool) else DSAIndexerPoolHost
