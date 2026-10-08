from __future__ import annotations

import weakref
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.attention.glm5_next import index_buf_accessor
from sglang.srt.layers.attention.glm5_next.quant_k_cache import (
    quantize_k_cache_separate,
)
from sglang.srt.layers.cp.utils import get_layer_owner as _get_layer_owner
from sglang.srt.layers.cp.utils import get_layer_shard_range as _get_layer_shard_range
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.layer_split import MainKVPagePlan
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    KVCache,
    get_tensor_size_bytes,
    unwrap_write_loc,
)
from sglang.srt.mem_cache.utils import (
    get_mla_kv_buffer_triton,
    set_mla_kv_buffer_triton,
    set_mla_kv_buffer_triton_fp8_quant,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_hcu, is_hcu_native_fp8_supported, is_hip

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter

_is_hcu = is_hcu()
_is_hip = is_hip()


class Glm5NextMLATokenToKVPool(DSATokenToKVPool):
    is_hcu_glm5_next_pool = True

    def get_kv_layer_ids(self):
        return [
            self.start_layer + i
            for i in range(self.layer_num)
            if self._is_layer_owned(self.start_layer + i)
        ]

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        use_dsa: bool = False,
        override_kv_cache_dim: Optional[int] = None,
        layer_shard_rank: Optional[int] = None,
        layer_shard_size: int = 1,
        layer_shard_rank_offset: int = 0,
        mla_kv_prefetch_ring_size: int = 1,
        layer_split_scratch_source: Optional[Glm5NextMLATokenToKVPool] = None,
    ):
        KVCache.__init__(
            self,
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.layer_shard_rank = layer_shard_rank
        self.layer_shard_size = layer_shard_size
        self.layer_shard_rank_offset = layer_shard_rank_offset
        self.layer_shard_enabled = (
            _is_hcu and layer_shard_rank is not None and layer_shard_size > 1
        )
        self.layer_shard_start = 0
        self.layer_broadcast_comm = None
        if self.layer_shard_enabled:
            if not (
                0 <= layer_shard_rank < layer_shard_size
                and 0 <= layer_shard_rank_offset < layer_shard_size
            ):
                raise ValueError("Invalid LayerSplit rank or rank offset")
            self.layer_shard_start = self._owned_local_layer_range()[0]

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.use_dsa = use_dsa
        self.dsa_kv_cache_store_fp8 = (
            use_dsa
            and dtype == torch.float8_e4m3fn
            and override_kv_cache_dim is not None
        )
        # When override_kv_cache_dim is provided, we assume the caller already
        # selected the backend-specific physical KV cache width.
        self.kv_cache_dim = (
            override_kv_cache_dim
            if override_kv_cache_dim is not None
            else (kv_lora_rank + qk_rope_head_dim)
        )
        self.mla_kv_prefetch_ring_size = max(1, mla_kv_prefetch_ring_size)
        self.layer_split_scratch_source = layer_split_scratch_source
        self.shares_layer_split_scratch = layer_split_scratch_source is not None

        self._create_buffers()

        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        if not use_dsa:
            # DSA will allocate indexer KV cache later and then log the total size
            self._finalize_allocation_log(size)

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                self.kv_buffer = [
                    torch.zeros(
                        (
                            (
                                self.size + self.page_size
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
                if self.layer_shard_enabled:
                    if (self.size + self.page_size) % self.page_size != 0:
                        raise ValueError(
                            "LayerSplit MLA KV buffer must contain whole pages: "
                            f"size={self.size}, page_size={self.page_size}"
                        )
                    self.num_pool_pages = (self.size + self.page_size) // self.page_size
                    shared_buffers = self._get_shared_main_kv_scratch()
                    if shared_buffers is None:
                        self.remote_kv_buffers = [
                            torch.zeros(
                                (
                                    self.size + self.page_size,
                                    1,
                                    self.kv_cache_dim,
                                ),
                                dtype=self.store_dtype,
                                device=self.device,
                            )
                            for _ in range(self.mla_kv_prefetch_ring_size)
                        ]
                    else:
                        self.remote_kv_buffers = shared_buffers
                    self.remote_kv_layer_ids: List[Optional[int]] = [
                        None
                    ] * self.mla_kv_prefetch_ring_size
                    self.pending_remote_kv_layer_ids: List[Optional[int]] = [
                        None
                    ] * self.mla_kv_prefetch_ring_size
                    self.slot_broadcast_events: List[Optional[Any]] = [
                        None
                    ] * self.mla_kv_prefetch_ring_size
                    # A single batch-wide mapping is shared by every remote
                    # ring slot. Page 0 remains the padded/dummy page.
                    self.physical_to_compact_main_kv_page = torch.full(
                        (self.num_pool_pages,),
                        -1,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    self.physical_to_compact_main_kv_page[0] = 0
                    self._active_main_kv_page_plan: Optional[MainKVPagePlan] = None
                    self._active_main_kv_batch_marker: Optional[
                        weakref.ReferenceType[Any]
                    ] = None
                    self._active_main_kv_compact_page_ids = torch.empty(
                        0, dtype=torch.long, device=self.device
                    )
                    self.device_module = torch.get_device_module(self.device)
                    self.kv_broadcast_stream = self.device_module.Stream()
        self._init_layer_broadcast_comm()

    def _get_shared_main_kv_scratch(self) -> Optional[List[torch.Tensor]]:
        source = self.layer_split_scratch_source
        if source is None:
            return None
        if source is self:
            raise ValueError("LayerSplit scratch source cannot be the pool itself.")
        if not self.layer_shard_enabled or not source.layer_shard_enabled:
            raise ValueError(
                "LayerSplit scratch sharing requires both KV pools to be layer-sharded."
            )

        compatibility = {
            "size": (self.size, source.size),
            "page_size": (self.page_size, source.page_size),
            "kv_cache_dim": (self.kv_cache_dim, source.kv_cache_dim),
            "store_dtype": (self.store_dtype, source.store_dtype),
            "device": (self.device, source.device),
            "layer_shard_rank": (
                self.layer_shard_rank,
                source.layer_shard_rank,
            ),
            "layer_shard_size": (
                self.layer_shard_size,
                source.layer_shard_size,
            ),
            "mla_kv_prefetch_ring_size": (
                self.mla_kv_prefetch_ring_size,
                source.mla_kv_prefetch_ring_size,
            ),
        }
        mismatches = [
            f"{name}: draft={draft_value!r}, target={target_value!r}"
            for name, (draft_value, target_value) in compatibility.items()
            if draft_value != target_value
        ]
        if mismatches:
            raise ValueError(
                "Incompatible LayerSplit Main-KV scratch source: "
                + "; ".join(mismatches)
            )

        buffers = getattr(source, "remote_kv_buffers", None)
        if buffers is None or len(buffers) != self.mla_kv_prefetch_ring_size:
            raise ValueError(
                "LayerSplit scratch source does not expose the expected "
                f"Main-KV ring: expected={self.mla_kv_prefetch_ring_size}, "
                f"actual={0 if buffers is None else len(buffers)}"
            )
        expected_shape = (
            self.size + self.page_size,
            1,
            self.kv_cache_dim,
        )
        for slot, buffer in enumerate(buffers):
            if (
                tuple(buffer.shape) != expected_shape
                or buffer.dtype != self.store_dtype
            ):
                raise ValueError(
                    "Incompatible LayerSplit Main-KV scratch tensor: "
                    f"slot={slot}, shape={tuple(buffer.shape)}, "
                    f"dtype={buffer.dtype}, expected_shape={expected_shape}, "
                    f"expected_dtype={self.store_dtype}"
                )
        return list(buffers)

    def _clear_buffers(self):
        del self.kv_buffer
        if hasattr(self, "remote_kv_buffers"):
            del self.remote_kv_buffers
        if hasattr(self, "physical_to_compact_main_kv_page"):
            del self.physical_to_compact_main_kv_page

    def get_kv_size_bytes(self):
        assert hasattr(self, "kv_buffer")
        kv_size_bytes = 0
        for kv_cache in self.kv_buffer:
            kv_size_bytes += get_tensor_size_bytes(kv_cache)
        return kv_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        if self.layer_shard_enabled:
            owned_layer_ids = [
                i
                for i in range(self.layer_num)
                if self._is_layer_owned(self.start_layer + i)
            ]
        else:
            owned_layer_ids = list(range(self.layer_num))

        kv_data_ptrs = [self.kv_buffer[i].data_ptr() for i in owned_layer_ids]
        kv_data_lens = [self.kv_buffer[i].nbytes for i in owned_layer_ids]
        kv_item_lens = [
            (
                self.kv_buffer[i][0].nbytes * self.page_size
                if self.kv_buffer[i].shape[0] > 0
                else 0
            )
            for i in owned_layer_ids
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_key_buffer(self, layer_id: int):
        return self._get_key_buffer_impl(layer_id, prefetch_has_history=True)

    def get_key_buffer_with_prefetch_history(self, layer_id: int, *, has_history: bool):
        return self._get_key_buffer_impl(layer_id, prefetch_has_history=has_history)

    def _get_key_buffer_impl(self, layer_id: int, *, prefetch_has_history: bool):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        kv_buffer = self._get_broadcastable_kv_buffer(
            layer_id, prefetch_has_history=prefetch_has_history
        )
        if self.store_dtype != self.dtype:
            return kv_buffer.view(self.dtype)

        return kv_buffer

    def get_key_buffer_DeepSeekV2(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        kv_buffer = self._get_broadcastable_kv_buffer(layer_id)
        if self.store_dtype != self.dtype and self.dtype not in (
            torch.float8_e5m2,
            torch.float8_e4m3fn,
        ):
            return kv_buffer.view(self.dtype)
        return kv_buffer, self.dtype

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        kv_buffer = self._get_broadcastable_kv_buffer(layer_id)
        if self.store_dtype != self.dtype:
            return kv_buffer[..., : self.kv_lora_rank].view(self.dtype)
        return kv_buffer[..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def _store_kv_cache_to_buffer(
        self, kv_buffer: torch.Tensor, loc: torch.Tensor, cache_k: torch.Tensor
    ):
        if cache_k.shape[-1] == self.kv_cache_dim:
            kv_buffer[loc] = cache_k
        else:
            assert cache_k.shape[-1] < self.kv_cache_dim
            pad_width = self.kv_cache_dim - cache_k.shape[-1]
            padding = cache_k.new_zeros(*cache_k.shape[:-1], pad_width)
            kv_buffer[loc] = torch.cat([cache_k, padding], dim=-1)

    def _store_kv_cache(self, layer_id: int, loc: torch.Tensor, cache_k: torch.Tensor):
        kv_buffer = self.kv_buffer[layer_id - self.start_layer]
        self._store_kv_cache_to_buffer(kv_buffer, loc, cache_k)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        loc, _, _ = unwrap_write_loc(loc)
        layer_id = layer.layer_id
        assert not self.dsa_kv_cache_store_fp8
        if self.layer_shard_enabled:
            slot = self._remote_kv_slot(layer_id)
            if self.pending_remote_kv_layer_ids[slot] == layer_id:
                self._finalize_pending_kv_broadcast(slot, set_remote_layer_id=False)
            if self.remote_kv_layer_ids[slot] == layer_id:
                self.remote_kv_layer_ids[slot] = None
        if not self._is_layer_owned(layer_id):
            return
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)

        if self.store_dtype != self.dtype:
            self._store_kv_cache(layer_id, loc, cache_k.view(self.store_dtype))
        else:
            self._store_kv_cache(layer_id, loc, cache_k)

    def set_kv_buffer_opt(  # TODO: handwrite kernel
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        layer_id = layer.layer_id
        assert not (self.use_dsa and self.dsa_kv_cache_store_fp8)
        if self.layer_shard_enabled:
            slot = self._remote_kv_slot(layer_id)
            if self.pending_remote_kv_layer_ids[slot] == layer_id:
                self._finalize_pending_kv_broadcast(slot, set_remote_layer_id=False)
            if self.remote_kv_layer_ids[slot] == layer_id:
                self.remote_kv_layer_ids[slot] = None
        if not self._is_layer_owned(layer_id):
            return
        cache_k = torch.cat([cache_k_nope, cache_k_rope], dim=-1)
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
        if self.store_dtype != self.dtype:
            self._store_kv_cache(layer_id, loc, cache_k.view(self.store_dtype))
        else:
            self._store_kv_cache(layer_id, loc, cache_k)

    def _write_mla_kv_buffer(
        self,
        dst_buffer: torch.Tensor,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ) -> None:
        if _is_hip and not _is_hcu and self.use_dsa and self.dtype == fp8_dtype:
            # HIP FP8 path uses raw MLA KV layout (nope + rope) without per-block scales.
            # Fuse BF16/FP16 -> FP8 cast with paged KV write.
            set_mla_kv_buffer_triton_fp8_quant(
                dst_buffer,
                loc,
                cache_k_nope,
                cache_k_rope,
                fp8_dtype,
            )
        elif self.dsa_kv_cache_store_fp8:
            if _is_hcu and self.qk_rope_head_dim > 0:
                from lightop import op

                op.fused_quantize_and_store_mla_kv_cache(
                    cache_k_nope,
                    cache_k_rope,
                    dst_buffer,
                    loc,
                    "fp8_e4m3",
                    1e-6,
                )
            else:
                cache_k_nope_fp8, cache_k_rope_fp8 = quantize_k_cache_separate(
                    cache_k_nope, cache_k_rope
                )
                set_mla_kv_buffer_triton(
                    dst_buffer,
                    loc,
                    cache_k_nope_fp8,
                    cache_k_rope_fp8,
                )
        else:
            if cache_k_nope.dtype != self.dtype:
                cache_k_nope = cache_k_nope.to(self.dtype)
                cache_k_rope = cache_k_rope.to(self.dtype)
            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            if cache_k_nope.shape[-1] + cache_k_rope.shape[-1] == self.kv_cache_dim:
                set_mla_kv_buffer_triton(
                    dst_buffer,
                    loc,
                    cache_k_nope,
                    cache_k_rope,
                )
            else:
                cache_k = torch.cat([cache_k_nope, cache_k_rope], dim=-1)
                self._store_kv_cache_to_buffer(dst_buffer, loc, cache_k)

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        loc, _, _ = unwrap_write_loc(loc)
        layer_id = layer.layer_id
        remote_kv_updatable = False
        if self.layer_shard_enabled:
            slot = self._remote_kv_slot(layer_id)
            if self.pending_remote_kv_layer_ids[slot] == layer_id:
                self._finalize_pending_kv_broadcast(slot, set_remote_layer_id=True)
            remote_kv_updatable = self.remote_kv_layer_ids[slot] == layer_id

        if remote_kv_updatable:
            remote_loc = self.translate_main_kv_loc_to_compact(loc)
            self._write_mla_kv_buffer(
                self.remote_kv_buffers[slot],
                remote_loc,
                cache_k_nope,
                cache_k_rope,
            )
        if not self._is_layer_owned(layer_id):
            return

        self._write_mla_kv_buffer(
            self.kv_buffer[layer_id - self.start_layer],
            loc,
            cache_k_nope,
            cache_k_rope,
        )
        if (
            self.layer_shard_enabled
            and not remote_kv_updatable
            and self.remote_kv_layer_ids[slot] == layer_id
        ):
            self.remote_kv_layer_ids[slot] = None

    def _remote_kv_slot(self, layer_id: int) -> int:
        return layer_id % self.mla_kv_prefetch_ring_size

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

        if not self.layer_shard_enabled:
            return
        if (
            self._active_main_kv_batch_marker is not None
            and self._active_main_kv_batch_marker() is batch_marker
            and self._active_main_kv_page_plan is page_plan
        ):
            return

        # A late side-stream broadcast must not write through a mapping owned
        # by the next ForwardBatch.
        self._drain_pending_layer_broadcasts()
        self.remote_kv_layer_ids[:] = [None] * self.mla_kv_prefetch_ring_size
        self.pending_remote_kv_layer_ids[:] = [None] * self.mla_kv_prefetch_ring_size
        self.physical_to_compact_main_kv_page.fill_(-1)
        self.physical_to_compact_main_kv_page[0] = 0
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

        # ``all_page_ids`` is unique and sorted. The map identifies exactly
        # those pages not already represented by a history page. A boundary
        # page is already mapped by the history side and is patched in place.
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

        if not self.layer_shard_enabled or self._active_main_kv_page_plan is None:
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
        slot: int,
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
        remote_pages = self.remote_kv_buffers[slot].view(
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

    def invalidate_remote_kv_buffer_for_layer(self, layer_id: int) -> None:
        """Invalidate a broadcast copy after the owner layer is restored."""
        if not self.layer_shard_enabled:
            return
        slot = self._remote_kv_slot(layer_id)
        if self.pending_remote_kv_layer_ids[slot] == layer_id:
            self._finalize_pending_kv_broadcast(slot, set_remote_layer_id=False)
        if self.remote_kv_layer_ids[slot] == layer_id:
            self.remote_kv_layer_ids[slot] = None

    def _finalize_pending_kv_broadcast(
        self, slot: int, *, set_remote_layer_id: bool = True
    ) -> None:
        if not self.layer_shard_enabled:
            return
        pending = self.pending_remote_kv_layer_ids[slot]
        if pending is None:
            return
        event = self.slot_broadcast_events[slot]
        if event is not None:
            self.device_module.current_stream().wait_event(event)
            self.slot_broadcast_events[slot] = None
        if set_remote_layer_id:
            self.remote_kv_layer_ids[slot] = pending
        else:
            self.remote_kv_layer_ids[slot] = None
        self.pending_remote_kv_layer_ids[slot] = None

    def _drain_pending_layer_broadcasts(
        self,
        *,
        discard_index: bool = False,
        discard_main_slot: Optional[int] = None,
    ) -> None:
        """Order a synchronous fallback after all side-stream broadcasts."""
        finalize_index = getattr(self, "_finalize_pending_index_broadcast", None)
        if finalize_index is not None:
            finalize_index(set_remote_layer_id=not discard_index)
        for slot in range(self.mla_kv_prefetch_ring_size):
            self._finalize_pending_kv_broadcast(
                slot, set_remote_layer_id=(slot != discard_main_slot)
            )

    def _get_layer_prefetch_stream(self):
        # The HCU draft shares target Main-KV scratch but uses its own NCCL
        # communicator. Its side-stream broadcasts can stall at completion
        # after a HiCache reload. Keep both draft KV and Index-K broadcasts
        # on the forward stream; target layers retain asynchronous prefetch.
        if self.shares_layer_split_scratch:
            return self.device_module.current_stream()
        return self.kv_broadcast_stream

    def prefetch_kv_buffer(
        self,
        layer_id: int,
        layer_transfer_counter: Optional[LayerDoneCounter] = None,
        layer_transfer_idx: Optional[int] = None,
        has_history: bool = True,
    ) -> None:
        if not self.layer_shard_enabled:
            return
        if not (self.start_layer <= layer_id < self.start_layer + self.layer_num):
            return

        slot = self._remote_kv_slot(layer_id)
        if self.remote_kv_layer_ids[slot] == layer_id:
            return
        if self.pending_remote_kv_layer_ids[slot] == layer_id:
            return
        if self.pending_remote_kv_layer_ids[slot] is not None:
            self._finalize_pending_kv_broadcast(slot, set_remote_layer_id=False)

        compact_history_is_empty = (
            self._active_main_kv_page_plan is not None
            and self._active_main_kv_page_plan.history_page_ids.numel() == 0
        )
        if compact_history_is_empty or (
            self._active_main_kv_page_plan is None and not has_history
        ):
            # The current step starts without reusable tokens. The gathered K
            # will populate this scratch directly, so an empty owner broadcast
            # would only duplicate data movement.
            self.remote_kv_layer_ids[slot] = layer_id
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
                self._broadcast_compact_main_kv_pages(slot, layer_id, src_tensor)
            else:
                self._broadcast_tensor_from_owner(
                    self.remote_kv_buffers[slot],
                    layer_id,
                    src_tensor=src_tensor,
                    use_layer_broadcast_comm=True,
                )
            self.remote_kv_layer_ids[slot] = layer_id
            return

        broadcast_stream = self._get_layer_prefetch_stream()
        broadcast_stream.wait_stream(self.device_module.current_stream())
        with self.device_module.stream(broadcast_stream):
            if transfer_counter is not None:
                transfer_counter.wait_until(transfer_idx)
            if self._active_main_kv_page_plan is not None:
                self._broadcast_compact_main_kv_pages(slot, layer_id, src_tensor)
            else:
                self._broadcast_tensor_from_owner(
                    self.remote_kv_buffers[slot],
                    layer_id,
                    src_tensor=src_tensor,
                    use_layer_broadcast_comm=True,
                )
            event = self.device_module.Event()
            event.record()
            self.slot_broadcast_events[slot] = event
        self.pending_remote_kv_layer_ids[slot] = layer_id
        self.remote_kv_layer_ids[slot] = None

    def _get_broadcastable_kv_buffer(
        self, layer_id: int, *, prefetch_has_history: bool = True
    ) -> torch.Tensor:
        if not self.layer_shard_enabled:
            return self.kv_buffer[layer_id - self.start_layer]
        slot = self._remote_kv_slot(layer_id)
        if self.pending_remote_kv_layer_ids[slot] == layer_id:
            self._finalize_pending_kv_broadcast(slot, set_remote_layer_id=True)
        if self.remote_kv_layer_ids[slot] != layer_id:
            if self.pending_remote_kv_layer_ids[slot] is not None:
                self._finalize_pending_kv_broadcast(slot, set_remote_layer_id=False)
            # The main and index broadcasts share one communicator. A fallback
            # broadcast on the current stream must not race with work still
            # queued on the communication stream.
            self._drain_pending_layer_broadcasts(discard_main_slot=slot)
            local_idx = self._local_layer_idx(layer_id)
            src_tensor = (
                self.kv_buffer[local_idx] if self._is_layer_owned(layer_id) else None
            )
            if self._active_main_kv_page_plan is not None:
                # The normal prefetch path transmits history only. A missed
                # prefetch reaches this correctness fallback after current KV
                # may already have been produced, so bootstrap every compact
                # page from the owner's now-current persistent buffer.
                self._broadcast_compact_main_kv_pages(
                    slot,
                    layer_id,
                    src_tensor,
                    include_current=True,
                )
            else:
                self._broadcast_tensor_from_owner(
                    self.remote_kv_buffers[slot],
                    layer_id,
                    src_tensor=src_tensor,
                    use_layer_broadcast_comm=True,
                )
            self.remote_kv_layer_ids[slot] = layer_id

        self.prefetch_kv_buffer(
            layer_id + self.mla_kv_prefetch_ring_size - 1,
            has_history=prefetch_has_history,
        )
        return self.remote_kv_buffers[slot]

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        # get k nope and k rope from the kv buffer, and optionally cast them to dst_dtype.
        layer_id = layer.layer_id
        kv_buffer = self.get_key_buffer(layer_id)
        loc = self.translate_main_kv_loc_to_compact(loc)
        dst_dtype = dst_dtype or self.dtype
        cache_k_nope = torch.empty(
            (loc.shape[0], 1, self.kv_lora_rank),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        cache_k_rope = torch.empty(
            (loc.shape[0], 1, self.qk_rope_head_dim),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        get_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope, cache_k_rope)
        return cache_k_nope, cache_k_rope

    def get_cpu_copy(self, indices, mamba_indices=None):
        torch.cuda.synchronize()
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
        torch.cuda.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        torch.cuda.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            if self.kv_buffer[layer_id].shape[0] == 0:
                continue
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = kv_cache_cpu[layer_id][i // chunk_size]
                assert kv_cpu.shape[0] == len(chunk_indices)
                kv_chunk = kv_cpu.to(self.kv_buffer[0].device, non_blocking=True)
                self.kv_buffer[layer_id][chunk_indices] = kv_chunk
        torch.cuda.synchronize()

    def _local_layer_idx(self, layer_id: int) -> int:
        return layer_id - self.start_layer

    def _owned_local_layer_range(self) -> tuple[int, int]:
        assert self.layer_shard_rank is not None
        logical_rank = (
            self.layer_shard_rank - self.layer_shard_rank_offset
        ) % self.layer_shard_size
        return _get_layer_shard_range(
            logical_rank, self.layer_shard_size, self.layer_num
        )

    def _is_layer_owned(self, layer_id: int) -> bool:
        if not self.layer_shard_enabled:
            return True
        local_idx = self._local_layer_idx(layer_id)
        owned_start, owned_end = self._owned_local_layer_range()
        return owned_start <= local_idx < owned_end

    def _get_layer_owner_rank(self, layer_id: int) -> int:
        logical_owner = _get_layer_owner(
            self._local_layer_idx(layer_id), self.layer_shard_size, self.layer_num
        )
        return (logical_owner + self.layer_shard_rank_offset) % self.layer_shard_size

    def _init_layer_broadcast_comm(self) -> None:
        if not self.layer_shard_enabled:
            return

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

    def _broadcast_tensor_from_owner(
        self,
        tensor: torch.Tensor,
        layer_id: int,
        src_tensor: Optional[torch.Tensor] = None,
        use_layer_broadcast_comm: bool = False,
    ) -> torch.Tensor:
        if not self.layer_shard_enabled:
            return tensor

        owner_rank = self._get_layer_owner_rank(layer_id)
        if self.layer_shard_rank == owner_rank:
            assert src_tensor is not None
            # Compact Main-KV pages are packed directly into ``tensor``.
            if src_tensor is not tensor:
                tensor.copy_(src_tensor)

        if use_layer_broadcast_comm and self.layer_broadcast_comm is not None:
            with self.layer_broadcast_comm.change_state(enable=True):
                self.layer_broadcast_comm.broadcast(tensor, owner_rank)
        else:
            cp_group = get_parallel().attn_cp_group
            pynccl_comm = cp_group.pynccl_comm
            if pynccl_comm is not None:
                with pynccl_comm.change_state(enable=True):
                    pynccl_comm.broadcast(tensor, owner_rank)
            else:
                cp_group.broadcast(tensor, src=owner_rank)
        return tensor

    @property
    def index_k_with_scale_buffer(self):
        return self._glm_index_k_with_scale_buffer

    @index_k_with_scale_buffer.setter
    def index_k_with_scale_buffer(self, value):
        self._glm_index_k_with_scale_buffer = value

    @index_k_with_scale_buffer.deleter
    def index_k_with_scale_buffer(self):
        del self._glm_index_k_with_scale_buffer

    def move_kv_cache(self, tgt_loc, src_loc):
        if tgt_loc.numel() == 0:
            return
        if self.layer_shard_enabled:
            self._drain_pending_layer_broadcasts()
        for buffer in self.kv_buffer:
            if buffer.shape[0] > 0:
                buffer[tgt_loc.long()] = buffer[src_loc.long()]
        if self.layer_shard_enabled:
            self.remote_kv_layer_ids[:] = [None] * self.mla_kv_prefetch_ring_size
            self.remote_index_layer_id = None


class Glm5NextDSATokenToKVPool(Glm5NextMLATokenToKVPool):
    quant_block_size = 128
    index_k_with_scale_buffer_dtype = torch.uint8
    rope_storage_dtype = torch.bfloat16  # rope is always stored in bf16

    def __init__(
        self,
        size: int,
        page_size: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        index_head_dim: int,
        enable_memory_saver: bool,
        kv_cache_dim: int,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        index_buf_size: Optional[int] = None,
        index_kpool: int = 1,
        index_kpool_compress: bool = True,
        skip_topk_layers: Optional[List[bool]] = None,
        tail_extra_slots: int = 0,
        max_running_requests: Optional[int] = None,
        layer_shard_rank: Optional[int] = None,
        layer_shard_size: int = 1,
        layer_shard_rank_offset: int = 0,
        mla_kv_prefetch_ring_size: int = 1,
        layer_split_scratch_source: Optional[Glm5NextDSATokenToKVPool] = None,
    ):

        override_dim = (
            kv_cache_dim if kv_cache_dim != kv_lora_rank + qk_rope_head_dim else None
        )

        super().__init__(
            size,
            page_size,
            dtype,
            kv_lora_rank,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
            use_dsa=True,
            override_kv_cache_dim=override_dim,
            layer_shard_rank=layer_shard_rank,
            layer_shard_size=layer_shard_size,
            layer_shard_rank_offset=layer_shard_rank_offset,
            mla_kv_prefetch_ring_size=mla_kv_prefetch_ring_size,
            layer_split_scratch_source=layer_split_scratch_source,
        )
        # self.index_k_dtype = torch.float8_e4m3fn
        # self.index_k_scale_dtype = torch.float32
        self.index_head_dim = index_head_dim
        # assert index_kpool > 1 or tail_extra_slots == 0
        self.index_kpool = index_kpool
        self.index_kpool_compress = index_kpool_compress
        self.kpool_use_compress = index_kpool > 1
        self.skip_topk_layers = [False] * layer_num
        self.index_buf_size = size if index_buf_size is None else index_buf_size
        self.tail_extra_slots = tail_extra_slots
        self.slots_per_page = self.page_size // index_kpool
        self._tail_k = None
        self._tail_score = None
        if index_buf_size is None:
            index_buf_size = size
        num_pages = (index_buf_size + page_size + 1) // self.page_size
        # num head == 1 and head dim == 128 for index_k in DSA
        assert index_head_dim == 128
        if _is_hcu:
            self.use_fp8_index_k_cache = index_kpool > 1 or (
                dtype
                in (
                    torch.float8_e4m3fn,
                    torch.float8_e5m2,
                )
                and is_hcu_native_fp8_supported()
            )
        else:
            self.use_fp8_index_k_cache = True
        self.index_k_buffer_dtype = (
            torch.bfloat16 if _is_hcu and not self.use_fp8_index_k_cache else self.dtype
        )
        assert self.index_kpool <= 1 or self.use_fp8_index_k_cache, (
            "kpool layout requires FP8 index K cache"
        )

        if _is_hip and not _is_hcu:  # and  not _is_hcu:nhb
            assert self.page_size == 1
        else:
            assert self.page_size == 64
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            self.index_k_with_scale_buffer = None
            self.index_k_buffer = None
            if self.use_fp8_index_k_cache:
                self.index_k_with_scale_buffer = [
                    torch.zeros(
                        # Layout:
                        #     ref: test_attention.py :: kv_cache_cast_to_fp8
                        #     shape: (num_pages, page_size 64 * head_dim 128 + page_size 64 * fp32_nbytes 4)
                        #     data: for page i,
                        #         * buf[i, :page_size * head_dim] for fp8 data
                        #         * buf[i, page_size * head_dim:].view(float32) for scale
                        (
                            (
                                num_pages
                                if self._is_layer_owned(self.start_layer + i)
                                else 0
                            ),
                            self.slots_per_page
                            * (
                                index_head_dim
                                + index_head_dim // self.quant_block_size * 4
                            ),
                        ),
                        dtype=self.index_k_with_scale_buffer_dtype,
                        device=device,
                    )
                    for i in range(layer_num)
                ]
            else:
                self.index_k_buffer = [
                    torch.zeros(
                        (
                            (
                                num_pages
                                if self._is_layer_owned(self.start_layer + i)
                                else 0
                            ),
                            self.page_size,
                            1,
                            self.index_head_dim,
                        ),
                        dtype=self.index_k_buffer_dtype,
                        device=device,
                    )
                    for i in range(layer_num)
                ]
            if self.index_kpool > 1:
                assert max_running_requests is not None, (
                    "kpool layout requires max_running_requests for the per-req tail"
                )
                tail_shape = (
                    max_running_requests + 1,
                    self.index_kpool + self.tail_extra_slots,
                    self.index_head_dim,
                )
                self._tail_k = [
                    torch.zeros(tail_shape, dtype=torch.bfloat16, device=device)
                    for _ in range(layer_num)
                ]
                self._tail_score = [
                    torch.zeros(tail_shape, dtype=torch.bfloat16, device=device)
                    for _ in range(layer_num)
                ]
            if self.layer_shard_enabled:
                # Keep the draft Index-K scratch independent from the target
                # pool. Unlike Main-KV scratch, whose compact page plan is
                # invalidated for every ForwardBatch, Index-K scratch uses a
                # persistent ``remote_index_layer_id`` validity marker. If two
                # pools alias the tensor while keeping separate markers, one
                # pool can overwrite the bytes and the other can still treat
                # them as current, silently consuming another model's Index K.
                if self.use_fp8_index_k_cache:
                    self.remote_index_k_with_scale_buffer = torch.zeros(
                        (
                            num_pages,
                            self.slots_per_page
                            * (
                                index_head_dim
                                + index_head_dim // self.quant_block_size * 4
                            ),
                        ),
                        dtype=self.index_k_with_scale_buffer_dtype,
                        device=device,
                    )
                else:
                    self.remote_index_k_buffer = torch.zeros(
                        (num_pages, self.page_size, 1, self.index_head_dim),
                        dtype=self.index_k_buffer_dtype,
                        device=device,
                    )
                self.remote_index_layer_id: Optional[int] = None
                self.pending_remote_index_event = self.device_module.Event()
                self.pending_remote_index_layer_id: Optional[int] = None
                self.pending_remote_index_broadcast = False
        self._finalize_allocation_log(size)

    def _clear_buffers(self):
        super()._clear_buffers()
        if self.index_k_with_scale_buffer is not None:
            del self.index_k_with_scale_buffer
        if self.index_k_buffer is not None:
            del self.index_k_buffer
        if self._tail_k is not None:
            del self._tail_k
            del self._tail_score
        if hasattr(self, "remote_index_k_with_scale_buffer"):
            del self.remote_index_k_with_scale_buffer
        if hasattr(self, "remote_index_k_buffer"):
            del self.remote_index_k_buffer

    def get_tail_buffers(self, layer_id: int):
        assert self._tail_k is not None and self._tail_score is not None
        idx = layer_id - self.start_layer
        return self._tail_k[idx], self._tail_score[idx]

    def get_tail_buf_infos(self):
        if self._tail_k is None:
            return [], [], []
        bufs = self._tail_k + self._tail_score
        return (
            [b.data_ptr() for b in bufs],
            [b.nbytes for b in bufs],
            [b[0].nbytes for b in bufs],
        )

    def index_k_device_ptrs_for_host_transfer(self) -> list[int]:
        """Per-layer device index_k pointers for host hicache transfer paths."""
        if self.use_fp8_index_k_cache:
            return [buf.data_ptr() for buf in self.index_k_with_scale_buffer]

        row_width = (
            self.page_size
            * self.index_head_dim
            * torch.empty((), dtype=self.index_k_buffer_dtype).element_size()
        )
        return [
            buf.view(torch.uint8).view(buf.shape[0], row_width).data_ptr()
            for buf in self.index_k_buffer
        ]

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        assert self.use_fp8_index_k_cache, "FP8 index K cache is not enabled"
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_broadcastable_index_buffer(layer_id)

    def get_hcu_index_k_with_scale_write_buffer(self, layer_id: int) -> torch.Tensor:
        """Return the HCU fused-write target for the current index K."""
        assert self.use_fp8_index_k_cache, "FP8 index K cache is not enabled"
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        remote_updatable = self._prepare_remote_index_write(layer_id)

        local_idx = layer_id - self.start_layer
        if not self.layer_shard_enabled or self._is_layer_owned(layer_id):
            return self.index_k_with_scale_buffer[local_idx]
        if not remote_updatable:
            self.remote_index_layer_id = None
        return self.remote_index_k_with_scale_buffer

    def get_kpool_index_k_with_scale_write_buffer(self, layer_id: int) -> torch.Tensor:
        """Return a LayerSplit-aware persistent kpool write target.

        All ranks first make the shared scratch current for this layer (a cold
        layer with no history can skip the broadcast). The layer owner writes
        its persistent buffer while the other ranks update scratch. The caller
        must invoke
        ``commit_kpool_index_k_with_scale_write_buffer`` after the write (and
        after the CP slot all-gather, when present) so the owner's scratch
        copy remains valid for the following index read.
        """
        assert self.use_fp8_index_k_cache, "FP8 index K cache is not enabled"
        assert self.index_kpool > 1, "kpool write buffer requires index_kpool > 1"
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        broadcast_buf = self._get_broadcastable_index_buffer(layer_id)
        if not self.layer_shard_enabled or not self._is_layer_owned(layer_id):
            return broadcast_buf
        return self.index_k_with_scale_buffer[layer_id - self.start_layer]

    def commit_hcu_index_k_with_scale_write_buffer(
        self, layer_id: int, loc: torch.Tensor
    ) -> None:
        """Mirror owner fused writes into an already-prefetched scratch."""
        if not self.layer_shard_enabled or not self._is_layer_owned(layer_id):
            return
        if not self._prepare_remote_index_write(layer_id):
            return

        local_buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        remote_buf = self.remote_index_k_with_scale_buffer
        page_indices = loc // self.page_size
        token_offsets = loc % self.page_size
        k_bytes_per_page = self.page_size * self.index_head_dim
        scale_bytes_per_token = (
            self.index_head_dim // self.quant_block_size * torch.float32.itemsize
        )

        local_k = local_buf[:, :k_bytes_per_page].view(
            -1, self.page_size, self.index_head_dim
        )
        remote_k = remote_buf[:, :k_bytes_per_page].view(
            -1, self.page_size, self.index_head_dim
        )
        remote_k[page_indices, token_offsets] = local_k[page_indices, token_offsets]

        local_scale = local_buf[:, k_bytes_per_page:].view(
            -1, self.page_size, scale_bytes_per_token
        )
        remote_scale = remote_buf[:, k_bytes_per_page:].view(
            -1, self.page_size, scale_bytes_per_token
        )
        remote_scale[page_indices, token_offsets] = local_scale[
            page_indices, token_offsets
        ]

    def commit_kpool_index_k_with_scale_write_buffer(
        self, layer_id: int, loc: torch.Tensor
    ) -> None:
        """Mirror newly written owner kpool slots into broadcast scratch.

        ``loc`` addresses compressed pool slots, so each physical page has
        ``slots_per_page`` entries. It must not use the token-level
        ``page_size`` addressing from the regular HCU index write path.
        """
        if (
            not self.layer_shard_enabled
            or not self._is_layer_owned(layer_id)
            or loc.numel() == 0
        ):
            return
        if not self._prepare_remote_index_write(layer_id):
            return

        local_buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        remote_buf = self.remote_index_k_with_scale_buffer
        slots_per_page = self.slots_per_page
        page_indices = loc // slots_per_page
        slot_offsets = loc % slots_per_page
        k_bytes_per_page = slots_per_page * self.index_head_dim
        scale_bytes_per_slot = (
            self.index_head_dim // self.quant_block_size * torch.float32.itemsize
        )

        local_k = local_buf[:, :k_bytes_per_page].view(
            -1, slots_per_page, self.index_head_dim
        )
        remote_k = remote_buf[:, :k_bytes_per_page].view(
            -1, slots_per_page, self.index_head_dim
        )
        remote_k[page_indices, slot_offsets] = local_k[page_indices, slot_offsets]

        local_scale = local_buf[:, k_bytes_per_page:].view(
            -1, slots_per_page, scale_bytes_per_slot
        )
        remote_scale = remote_buf[:, k_bytes_per_page:].view(
            -1, slots_per_page, scale_bytes_per_slot
        )
        remote_scale[page_indices, slot_offsets] = local_scale[
            page_indices, slot_offsets
        ]

    def get_index_k_with_scale_buffer_broadcast(self, layer_id: int) -> torch.Tensor:
        """Return the layer-shard-aware FP8 index buffer for attention reads."""
        assert self.use_fp8_index_k_cache, "FP8 index K cache is not enabled"
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_broadcastable_index_buffer(layer_id)

    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        assert self.index_k_buffer is not None, "BF16 index K cache is not enabled"
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_broadcastable_index_buffer(layer_id)

    def get_index_k_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        if not self.use_fp8_index_k_cache:
            num_pages = (seq_len + self.page_size - 1) // self.page_size
            buf = self._get_broadcastable_index_buffer(layer_id)
            return buf[page_indices[:num_pages]].view(-1, 1, self.index_head_dim)[
                :seq_len
            ]
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        buf = self._get_broadcastable_index_buffer(layer_id)
        return index_buf_accessor.GetK.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def get_index_k_scale_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        buf = self._get_broadcastable_index_buffer(layer_id)
        return index_buf_accessor.GetS.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ):
        return self._get_index_k_scale_buffer_impl(
            layer_id,
            seq_len_tensor,
            page_indices,
            seq_len_sum,
            max_seq_len,
            prefetch_has_history=True,
        )

    def get_index_k_scale_buffer_with_prefetch_history(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
        *,
        has_history: bool,
    ):
        return self._get_index_k_scale_buffer_impl(
            layer_id,
            seq_len_tensor,
            page_indices,
            seq_len_sum,
            max_seq_len,
            prefetch_has_history=has_history,
        )

    def _get_index_k_scale_buffer_impl(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
        *,
        prefetch_has_history: bool,
    ):
        """
        Fused method to get both index K and scale data in a single call using Triton.
        More efficient than calling get_index_k_continuous and get_index_k_scale_continuous separately.

        :param layer_id: Layer index
        :param seq_len: Sequence length
        :param page_indices: Page indices tensor
        :return: tuple of (k_fp8, k_scale) where
                 k_fp8: (seq_len, index_head_dim), uint8
                 k_scale: (seq_len, 4), uint8
        """
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        buf = self._get_broadcastable_index_buffer(layer_id)
        if self.layer_shard_enabled:
            if layer_id == self.start_layer:
                for offset in range(self.mla_kv_prefetch_ring_size - 1):
                    self.prefetch_kv_buffer(
                        layer_id + offset,
                        has_history=prefetch_has_history,
                    )
            self.prefetch_kv_buffer(
                layer_id + self.mla_kv_prefetch_ring_size - 1,
                has_history=prefetch_has_history,
            )
        return index_buf_accessor.GetKAndS.execute(
            self,
            buf,
            page_indices=page_indices,
            seq_len_tensor=seq_len_tensor,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        assert self.use_fp8_index_k_cache, "FP8 index K cache is not enabled"
        remote_updatable = self._prepare_remote_index_write(layer_id)
        if remote_updatable:
            index_buf_accessor.SetKAndS.execute(
                pool=self,
                buf=self.remote_index_k_with_scale_buffer,
                loc=loc,
                index_k=index_k,
                index_k_scale=index_k_scale,
            )
        if not self._is_layer_owned(layer_id):
            return
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        index_buf_accessor.SetKAndS.execute(
            pool=self, buf=buf, loc=loc, index_k=index_k, index_k_scale=index_k_scale
        )

    def _finalize_pending_index_broadcast(
        self, *, set_remote_layer_id: bool = True
    ) -> None:
        if not self.layer_shard_enabled or not getattr(
            self, "pending_remote_index_broadcast", False
        ):
            return
        self.device_module.current_stream().wait_event(self.pending_remote_index_event)
        self.pending_remote_index_broadcast = False
        if set_remote_layer_id and self.pending_remote_index_layer_id is not None:
            self.remote_index_layer_id = self.pending_remote_index_layer_id
        elif not set_remote_layer_id:
            self.remote_index_layer_id = None
        self.pending_remote_index_layer_id = None

    def prefetch_index_buffer(
        self,
        layer_id: int,
        layer_transfer_counter: Optional[LayerDoneCounter] = None,
        layer_transfer_idx: Optional[int] = None,
        has_history: bool = True,
    ) -> None:
        if not self.layer_shard_enabled:
            return
        if self.remote_index_layer_id == layer_id:
            return
        if getattr(self, "pending_remote_index_broadcast", False):
            if self.pending_remote_index_layer_id == layer_id:
                return
            self._finalize_pending_index_broadcast(set_remote_layer_id=False)

        if not has_history:
            self.remote_index_layer_id = layer_id
            return

        if self.use_fp8_index_k_cache:
            local_buffer = self.index_k_with_scale_buffer
            remote_buffer = self.remote_index_k_with_scale_buffer
        else:
            local_buffer = self.index_k_buffer
            remote_buffer = self.remote_index_k_buffer
        local_idx = self._local_layer_idx(layer_id)
        src_tensor = local_buffer[local_idx] if self._is_layer_owned(layer_id) else None
        transfer_counter = layer_transfer_counter or self.layer_transfer_counter
        transfer_idx = (
            layer_transfer_idx if layer_transfer_idx is not None else local_idx
        )

        if self.layer_broadcast_comm is None:
            if transfer_counter is not None:
                transfer_counter.wait_until(transfer_idx)
            self._broadcast_tensor_from_owner(
                remote_buffer,
                layer_id,
                src_tensor=src_tensor,
                use_layer_broadcast_comm=True,
            )
            self.pending_remote_index_event.record()
            self.remote_index_layer_id = layer_id
            return

        broadcast_stream = self._get_layer_prefetch_stream()
        broadcast_stream.wait_stream(self.device_module.current_stream())
        with self.device_module.stream(broadcast_stream):
            if transfer_counter is not None:
                transfer_counter.wait_until(transfer_idx)
            self._broadcast_tensor_from_owner(
                remote_buffer,
                layer_id,
                src_tensor=src_tensor,
                use_layer_broadcast_comm=True,
            )
            self.pending_remote_index_event.record()
        self.remote_index_layer_id = None
        self.pending_remote_index_layer_id = layer_id
        self.pending_remote_index_broadcast = True

    def _prepare_remote_index_write(self, layer_id: int) -> bool:
        if not self.layer_shard_enabled:
            return False
        if getattr(self, "pending_remote_index_broadcast", False):
            self._finalize_pending_index_broadcast(
                set_remote_layer_id=(self.pending_remote_index_layer_id == layer_id)
            )
        return self.remote_index_layer_id == layer_id

    def invalidate_index_buffer_for_layer(self, layer_id: int) -> None:
        if (
            self.layer_shard_enabled
            and getattr(self, "pending_remote_index_broadcast", False)
            and self.pending_remote_index_layer_id == layer_id
        ):
            self._finalize_pending_index_broadcast(set_remote_layer_id=False)
        if (
            self.layer_shard_enabled
            and getattr(self, "remote_index_layer_id", None) == layer_id
        ):
            self.remote_index_layer_id = None

    def _get_broadcastable_index_buffer(self, layer_id: int) -> torch.Tensor:
        if self.use_fp8_index_k_cache:
            local_buffer = self.index_k_with_scale_buffer
            remote_attr = "remote_index_k_with_scale_buffer"
        else:
            local_buffer = self.index_k_buffer
            remote_attr = "remote_index_k_buffer"

        if not self.layer_shard_enabled:
            return local_buffer[layer_id - self.start_layer]
        if getattr(self, "pending_remote_index_broadcast", False):
            self._finalize_pending_index_broadcast(
                set_remote_layer_id=(self.pending_remote_index_layer_id == layer_id)
            )
        if self.remote_index_layer_id != layer_id:
            # Main KV prefetch is queued after index prefetch on the same
            # communicator. Drain both before issuing a synchronous fallback
            # from the current stream.
            self._drain_pending_layer_broadcasts()
            local_idx = self._local_layer_idx(layer_id)
            src_tensor = (
                local_buffer[local_idx] if self._is_layer_owned(layer_id) else None
            )
            remote_buffer = getattr(self, remote_attr)
            self._broadcast_tensor_from_owner(
                remote_buffer,
                layer_id,
                src_tensor=src_tensor,
                use_layer_broadcast_comm=True,
            )
            self.remote_index_layer_id = layer_id
        return getattr(self, remote_attr)

    def move_kv_cache(self, tgt_loc, src_loc):
        super().move_kv_cache(tgt_loc, src_loc)
        if tgt_loc.numel() == 0:
            return
        tgt = tgt_loc.reshape(-1).long()
        src = src_loc.reshape(-1).long()
        tgt_page, src_page = tgt // self.page_size, src // self.page_size
        if self.use_fp8_index_k_cache:
            tgt_offset = (tgt % self.page_size) // self.index_kpool
            src_offset = (src % self.page_size) // self.index_kpool
            k_bytes = self.slots_per_page * self.index_head_dim
            scale_bytes = self.index_head_dim // self.quant_block_size * 4
            for buf in self.index_k_with_scale_buffer:
                if buf.shape[0] == 0:
                    continue
                keys = buf[:, :k_bytes].view(
                    -1, self.slots_per_page, self.index_head_dim
                )
                scales = buf[:, k_bytes:].view(-1, self.slots_per_page, scale_bytes)
                keys[tgt_page, tgt_offset] = keys[src_page, src_offset]
                scales[tgt_page, tgt_offset] = scales[src_page, src_offset]
        else:
            for buf in self.index_k_buffer:
                if buf.shape[0] > 0:
                    buf[tgt_page, tgt % self.page_size] = buf[
                        src_page, src % self.page_size
                    ]

    def _get_compress_tail_cpu_copy(self, req_pool_index):
        if self._tail_k is None or req_pool_index is None:
            return None
        return tuple(
            [buf[req_pool_index].to("cpu", non_blocking=True) for buf in buffers]
            for buffers in (self._tail_k, self._tail_score)
        )

    def _load_compress_tail_cpu_copy(self, tail_k_cpu, tail_score_cpu, req_pool_index):
        if self._tail_k is None or req_pool_index is None or tail_k_cpu is None:
            return
        for buffers, saved_buffers in (
            (self._tail_k, tail_k_cpu),
            (self._tail_score, tail_score_cpu),
        ):
            for buf, saved in zip(buffers, saved_buffers):
                buf[req_pool_index] = saved.to(buf.device, non_blocking=True)

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        # DSA keeps a page-indexed indexer cache alongside kv_buffer.
        # Retract frees the slots/pages and they get reused by other reqs'
        # indexer cache writes, so we must offload it here too -- otherwise
        # resume restores kv_buffer but leaves foreign index/scale in place and
        # DSA attention reads garbage at those token positions.
        kv_cache_cpu = super().get_cpu_copy(indices)
        index_k_cache = (
            self.index_k_with_scale_buffer
            if self.use_fp8_index_k_cache
            else self.index_k_buffer
        )

        page_indices = indices[:: self.page_size] // self.page_size
        torch.cuda.synchronize()
        index_k_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        page_chunk_size = max(1, chunk_size // self.page_size)
        for layer_id in range(self.layer_num):
            index_k_cpu.append([])
            if index_k_cache[layer_id].shape[0] == 0:
                continue
            for i in range(0, len(page_indices), page_chunk_size):
                chunk_page_indices = page_indices[i : i + page_chunk_size]
                idx_cpu = index_k_cache[layer_id][chunk_page_indices].to(
                    "cpu", non_blocking=True
                )
                index_k_cpu[-1].append(idx_cpu)
        torch.cuda.synchronize()

        cpu_copy = {"kv": kv_cache_cpu, "index_k": index_k_cpu}
        tail = self._get_compress_tail_cpu_copy(req_pool_index)
        if tail is not None:
            cpu_copy["tail_k"], cpu_copy["tail_score"] = tail
        torch.cuda.synchronize()
        return cpu_copy

    def load_cpu_copy(
        self, kv_cache_cpu_dict, indices, mamba_indices=None, req_pool_index=None
    ):
        super().load_cpu_copy(kv_cache_cpu_dict["kv"], indices)

        page_indices = indices[:: self.page_size] // self.page_size
        index_k_cpu = kv_cache_cpu_dict["index_k"]
        index_k_cache = (
            self.index_k_with_scale_buffer
            if self.use_fp8_index_k_cache
            else self.index_k_buffer
        )
        torch.cuda.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        page_chunk_size = max(1, chunk_size // self.page_size)
        for layer_id in range(self.layer_num):
            if index_k_cache[layer_id].shape[0] == 0:
                continue
            for i in range(0, len(page_indices), page_chunk_size):
                chunk_page_indices = page_indices[i : i + page_chunk_size]
                idx_cpu = index_k_cpu[layer_id][i // page_chunk_size]
                assert idx_cpu.shape[0] == len(chunk_page_indices)
                idx_chunk = idx_cpu.to(index_k_cache[0].device, non_blocking=True)
                index_k_cache[layer_id][chunk_page_indices] = idx_chunk
        self._load_compress_tail_cpu_copy(
            kv_cache_cpu_dict.get("tail_k"),
            kv_cache_cpu_dict.get("tail_score"),
            req_pool_index,
        )
        torch.cuda.synchronize()

    def set_index_k_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ) -> None:
        assert self.index_k_buffer is not None, "BF16 index K cache is not enabled"
        if index_k.dtype != self.index_k_buffer_dtype:
            index_k = index_k.to(self.index_k_buffer_dtype)

        remote_updatable = self._prepare_remote_index_write(layer_id)
        if remote_updatable:
            self.remote_index_k_buffer[loc // self.page_size, loc % self.page_size] = (
                index_k
            )
        if not self._is_layer_owned(layer_id):
            return
        self.index_k_buffer[layer_id - self.start_layer][
            loc // self.page_size, loc % self.page_size
        ] = index_k

    def get_state_buf_infos(self):
        index_cache = (
            self.index_k_with_scale_buffer
            if self.use_fp8_index_k_cache
            else self.index_k_buffer
        )
        if self.layer_shard_enabled:
            owned_layer_ids = [
                i
                for i in range(self.layer_num)
                if self._is_layer_owned(self.start_layer + i)
            ]
        else:
            owned_layer_ids = list(range(self.layer_num))

        data_ptrs = [index_cache[i].data_ptr() for i in owned_layer_ids]
        data_lens = [index_cache[i].nbytes for i in owned_layer_ids]
        item_lens = [
            index_cache[i][0].nbytes if index_cache[i].shape[0] > 0 else 0
            for i in owned_layer_ids
        ]
        return data_ptrs, data_lens, item_lens

    def get_kv_size_bytes(self):
        kv_size_bytes = super().get_kv_size_bytes()
        index_cache = (
            self.index_k_with_scale_buffer
            if self.use_fp8_index_k_cache
            else self.index_k_buffer
        )
        for index_k_cache in index_cache:
            kv_size_bytes += get_tensor_size_bytes(index_k_cache)
        return kv_size_bytes

    def get_compress_tail_buffers(self, layer_id):
        return self.get_tail_buffers(layer_id)

    def get_compress_tail_buf_infos(self):
        return self.get_tail_buf_infos()

    def get_broadcastable_index_k_with_scale_buffer(self, layer_id):
        return self.get_index_k_with_scale_buffer_broadcast(layer_id)


def glm5_next_pool_kwargs(configurator) -> dict:
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    scratch = getattr(configurator, "glm5_next_layer_split_scratch_source", None)
    enabled = (
        _is_hcu
        and parallel.enable_dsa_cache_layer_split
        and (not configurator.is_draft_worker or scratch is not None)
        and parallel.attn_cp_size > 1
    )
    return dict(
        layer_shard_rank=parallel.attn_cp_rank if enabled else None,
        layer_shard_size=parallel.attn_cp_size if enabled else 1,
        layer_shard_rank_offset=parallel.attn_cp_size - 1
        if enabled and configurator.is_draft_worker
        else 0,
        mla_kv_prefetch_ring_size=parallel.mla_kv_prefetch_ring_size if enabled else 1,
        layer_split_scratch_source=scratch if enabled else None,
    )


def glm5_next_cache_cell_size(
    configurator, num_layers: int, draft_layers: int = 0
) -> int:
    """Worst-rank bytes/token for the source's physical KV and compressed index layout."""
    from sglang.srt.configs.model_config import (
        get_dsa_index_head_dim,
        get_dsa_index_kpool,
    )
    from sglang.srt.mem_cache.kv_cache_configurator import calculate_mla_kv_cache_dim
    from sglang.srt.runtime_context import get_parallel

    config = configurator.model_config
    dtype = configurator.kv_cache_dtype
    main_bytes = (
        calculate_mla_kv_cache_dim(model_config=config, kv_cache_dtype=dtype)
        * dtype.itemsize
    )
    from sglang.srt.configs.model_config import is_deepseek_dsa

    if not is_deepseek_dsa(config.hf_config):
        return int((num_layers + draft_layers) * main_bytes)
    kpool = get_dsa_index_kpool(config.hf_config)
    head_dim = get_dsa_index_head_dim(config.hf_config)
    fp8_index = kpool > 1 or (
        dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        and is_hcu_native_fp8_supported()
    )
    index_bytes = (
        (head_dim + head_dim // 128 * 4) / kpool
        if fp8_index
        else head_dim * torch.bfloat16.itemsize
    )
    parallel = get_parallel()
    if (
        not _is_hcu
        or not parallel.enable_dsa_cache_layer_split
        or parallel.attn_cp_size <= 1
    ):
        return int((num_layers + draft_layers) * (main_bytes + index_bytes))
    ranks = parallel.attn_cp_size
    ring = parallel.mla_kv_prefetch_ring_size
    costs = []
    for rank in range(ranks):
        start, end = _get_layer_shard_range(rank, ranks, num_layers)
        dstart, dend = _get_layer_shard_range(
            (rank - (ranks - 1)) % ranks, ranks, draft_layers
        )
        owned = end - start + dend - dstart
        # Draft aliases only Main-KV scratch. Index scratch retains independent validity.
        costs.append(
            (owned + ring) * main_bytes + (owned + 1 + bool(draft_layers)) * index_bytes
        )
    return int(max(costs))
