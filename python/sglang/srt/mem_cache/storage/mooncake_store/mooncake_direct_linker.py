from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future
from queue import Empty, Queue

import torch

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.unified_cache_linker import (
    ExternalLinkerLoadError,
    UnifiedCacheLinker,
)
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_STORAGE,
    StorageMetricsCollector,
    resolve_collector_class,
)
from sglang.srt.utils import freeze_gc, get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()


class LayerWiseLoadCounter:
    """CPU completion counter compatible with KV pools' layer wait hook."""

    def __init__(self, num_layers: int, sync_groups=()):
        self.num_layers = num_layers
        self.sync_groups = tuple(group for group in sync_groups if group is not None)
        self.producer_index = -1
        self.consumer_index = -1
        self.futures: dict[int, list[Future]] = {}
        self.errors: dict[int, BaseException] = {}
        self.active_indices: set[int] = set()

    def update_producer(self) -> int:
        self.producer_index += 1
        self.futures[self.producer_index] = [Future() for _ in range(self.num_layers)]
        self.active_indices.add(self.producer_index)
        return self.producer_index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def complete(self, index: int, layer: int) -> None:
        self.futures[index][layer].set_result(None)

    def fail(self, index: int, error: BaseException) -> None:
        for future in self.futures.get(index, ()):
            if not future.done():
                future.set_exception(error)

    def wait_until(self, threshold: int) -> None:
        index = self.consumer_index
        futures = self.futures.get(index)
        if futures is None:
            return
        try:
            futures[threshold].result()
        except BaseException as error:
            # Finish the model forward before reporting the error. Raising here
            # can leave peer TP ranks blocked in a later model collective.
            self.errors[index] = error
        finally:
            if threshold == self.num_layers - 1:
                self.futures.pop(index, None)

    def raise_if_failed(self) -> None:
        index = self.consumer_index
        if index not in self.active_indices:
            return
        error = self.errors.get(index)
        failed = torch.tensor(int(error is not None), dtype=torch.int, device="cpu")
        for group in self.sync_groups:
            if torch.distributed.get_world_size(group=group) > 1:
                torch.distributed.all_reduce(
                    failed, op=torch.distributed.ReduceOp.MAX, group=group
                )
        if failed.item():
            self.active_indices.discard(index)
            self.errors.pop(index, None)
            message = "Mooncake layer-wise KV load failed for the current batch."
            if error is None:
                message += " A peer cache rank reported the failure."
            raise ExternalLinkerLoadError(message) from error
        self.active_indices.discard(index)

    def reset(self) -> None:
        self.producer_index = -1
        self.consumer_index = -1
        self.futures.clear()
        self.errors.clear()
        self.active_indices.clear()


class MooncakeDirectLinker(UnifiedCacheLinker):
    def __init__(
        self,
        server_args,
        params: CacheInitParams,
        *,
        components,
        storage=None,
    ):
        self.page_size = params.page_size
        self.page_wise_load_threshold = (
            server_args.mooncake_page_wise_load_threshold
        )
        self.page_wise_load_batch_size = (
            server_args.mooncake_page_wise_load_batch_size
        )
        self.enable_page_wise_load = server_args.mooncake_enable_page_wise_load
        if self.page_wise_load_threshold <= 0:
            raise ValueError(
                "--mooncake-page-wise-load-threshold must be positive, got "
                f"{self.page_wise_load_threshold}."
            )
        if self.page_wise_load_batch_size <= 0:
            raise ValueError(
                "--mooncake-page-wise-load-batch-size must be positive, got "
                f"{self.page_wise_load_batch_size}."
            )
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        self.pool_group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=self.page_size,
            params=params,
            components=components,
        )
        self.pools = self.pool_group.entry_map
        self.num_layers = self.pool_group.num_layers

        tp_rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tp_rank = torch.distributed.get_rank(group=params.tp_cache_group)
        extra_config, *_ = HybridCacheController.parse_storage_backend_extra_config(
            server_args.hicache_storage_backend_extra_config
        )
        extra_config["dfs_replica_num"] = server_args.mooncake_dfs_replica_num
        storage_config = HiCacheStorageConfig(
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            pp_rank=params.pp_rank,
            pp_size=params.pp_size,
            attn_cp_rank=params.attn_cp_rank,
            attn_cp_size=params.attn_cp_size,
            is_mla_model=True,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name=server_args.model_path,
            extra_config=extra_config,
        )
        if storage is None:
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStore,
            )

            self.storage = MooncakeStore(storage_config, mem_pool=None)
        else:
            self.storage = storage
        self.storage.mem_pool_host = self.pool_group
        self.storage.registered_pools = self.pools
        rank_suffix = f"tp{tp_rank}_cp{params.attn_cp_rank}_pp{params.pp_rank}"
        self.storage.mla_suffix = rank_suffix
        self.storage.mha_suffix = rank_suffix

        self.storage_metrics_collector = None
        if params.enable_metrics:
            labels = {
                "storage_backend": "mooncake_direct",
                "tp_rank": tp_rank,
                "dp_rank": getattr(server_args, "dp_rank", 0),
                "pp_rank": params.pp_rank,
                "pp_size": params.pp_size,
                "attn_cp_rank": params.attn_cp_rank,
                "attn_cp_size": params.attn_cp_size,
            }
            if server_args.extra_metric_labels:
                labels.update(server_args.extra_metric_labels)
            collector_cls = resolve_collector_class(
                server_args, STAT_LOGGER_ROLE_STORAGE, StorageMetricsCollector
            )
            self.storage_metrics_collector = collector_cls(labels=labels)

        self.register_buffers()
        self.layer_done_counter = LayerWiseLoadCounter(
            self.num_layers,
            sync_groups=(params.attn_cp_cache_group, params.attn_tp_cache_group)
            if params.attn_cp_cache_group is not None
            or params.attn_tp_cache_group is not None
            else (params.tp_cache_group,),
        )
        if PoolName.MAMBA in self.pools:
            params.req_to_token_pool.register_layer_transfer_counter(
                self.layer_done_counter
            )
        self.pending_loads: dict[str, list[PoolTransfer]] = {}
        self.prepared_load_sessions: dict[str, list[str]] = {}
        self.session_refcounts: dict[str, int] = {}
        self.session_sources: dict[str, str] = {}
        self.prepared_load_sources: dict[str, str | None] = {}
        self.pending_load_metrics: dict[str, tuple[str | None, int, float]] = {}
        self.session_lock = threading.Lock()
        self.gc_frozen = False
        self.load_queue: Queue[
            tuple[int, list[tuple[str, list[PoolTransfer]]], object] | None
        ] = Queue()
        self.offload_queue: Queue[
            tuple[list[PoolTransfer], int, str, float, object] | None
        ] = Queue()
        self.offload_results: Queue[bool] = Queue()
        self.stats = {"lookup": 0, "load": 0, "load_fallback": 0, "offload": 0}
        self.load_thread = threading.Thread(
            target=self.load_thread_func,
            daemon=True,
            name=f"mooncake-load-tp{tp_rank}",
        )
        self.load_thread.start()
        self.offload_thread = threading.Thread(
            target=self.offload_thread_func,
            daemon=True,
            name=f"mooncake-offload-tp{tp_rank}",
        )
        self.offload_thread.start()

    def register_buffers(self) -> None:
        seen = set()
        for pool in self.pools.values():
            for buffer in pool.get_hybrid_pool_buffer():
                storage = buffer.untyped_storage()
                allocation = (int(storage.data_ptr()), int(storage.nbytes()))
                if allocation in seen:
                    continue
                seen.add(allocation)
                result = self.storage.store.register_buffer(*allocation)
                if result not in (0, None):
                    raise RuntimeError(
                        "Failed to register GPU KV buffer with Mooncake, "
                        f"error code: {result}."
                    )

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        expanded = self.pool_group.resolve_transfers(transfers)
        if not expanded:
            return []
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        page_keys = list(kv.keys)
        if not page_keys:
            return []
        result = self.storage.batch_exists_v2(page_keys, expanded)
        restorable = result.restorable_prefix_pages or []
        self.stats["lookup"] += 1
        if restorable:
            logger.info(
                "Mooncake direct linker lookup hit: rid=%s pages=%d candidates=%d",
                rid,
                restorable[-1],
                len(restorable),
            )
        return restorable

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        # Query establishes a boundary at which every component is restorable;
        # insert then removes pages already resident in L1. Loading is therefore
        # intentionally partial and may contain only a side pool such as SWA.
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        if not expanded:
            return False
        if rid not in self.prepared_load_sessions:
            # Keep direct callers safe, although UnifiedCacheLinkerWrapper
            # normally prepares the session before committing the radix hit.
            if not self.prepare_load(rid, transfers):
                return False
        if rid in self.pending_loads:
            raise RuntimeError(f"Mooncake load for rid={rid} is already queued.")
        self.pending_loads[rid] = expanded
        kv = next(
            (transfer for transfer in transfers if transfer.name == PoolName.KV),
            None,
        )
        page_count = (
            len(kv.keys)
            if kv is not None
            else max((len(transfer.keys) for transfer in transfers), default=0)
        )
        self.pending_load_metrics[rid] = (
            self.get_prepared_load_source(rid),
            page_count * self.page_size,
            time.perf_counter(),
        )
        return True

    def prepare_load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        if not expanded:
            return False
        if rid in self.prepared_load_sessions:
            raise RuntimeError(f"Mooncake load for rid={rid} is already prepared.")

        keys = []
        seen = set()
        for transfer in expanded:
            component_keys, _ = self.storage._get_hybrid_page_component_keys(
                list(transfer.keys), transfer
            )
            for key in self.storage._tag_keys(component_keys):
                if key not in seen:
                    seen.add(key)
                    keys.append(key)

        acquired = []
        with self.session_lock:
            session_sources = getattr(self, "session_sources", None)
            if session_sources is None:
                session_sources = self.session_sources = {}
            new_keys = []
            for key in keys:
                if key in self.session_refcounts:
                    self.session_refcounts[key] += 1
                    acquired.append(key)
                else:
                    new_keys.append(key)

            results = []
            sources = []
            try:
                if new_keys:
                    results = list(self.storage.store.batch_get_session_start(new_keys))
                    source_getter = getattr(
                        self.storage.store, "batch_get_session_sources", None
                    )
                    if source_getter is not None:
                        try:
                            sources = list(source_getter(new_keys))
                        except BaseException:
                            logger.warning(
                                "Mooncake session source lookup failed for rid=%s; "
                                "using the configured direct-storage source.",
                                rid,
                                exc_info=True,
                            )
                    else:
                        sources = []
                    if len(sources) != len(new_keys):
                        fallback_source = (
                            "dfs"
                            if getattr(self.storage, "dfs_replica_num", 0) > 0
                            else "local_disk"
                        )
                        sources = [fallback_source] * len(new_keys)
            except BaseException:
                logger.warning(
                    "Mooncake get session start raised for rid=%s; "
                    "falling back to prefill.",
                    rid,
                    exc_info=True,
                )
                self._rollback_session_refs_locked(acquired)
                self.stats["load_fallback"] = self.stats.get("load_fallback", 0) + 1
                return False

            failed = len(results) != len(new_keys) or any(
                result != 0 for result in results
            )
            for source_index, (key, result) in enumerate(zip(new_keys, results)):
                if result == 0:
                    self.session_refcounts[key] = 1
                    source = (
                        sources[source_index]
                        if source_index < len(sources)
                        else "unknown"
                    )
                    session_sources[key] = source
                    acquired.append(key)

            if failed:
                failed_objects = [
                    {
                        "key": key,
                        "result": results[index] if index < len(results) else None,
                    }
                    for index, key in enumerate(new_keys)
                    if index >= len(results) or results[index] != 0
                ]
                logger.debug(
                    "Mooncake lookup hit but get session start failed for rid=%s: "
                    "failed_objects=%s; falling back to prefill.",
                    rid,
                    failed_objects,
                )
                logger.warning(
                    "Mooncake lookup hit but get session start failed for rid=%s: "
                    "failed_objects=%d; falling back to prefill.",
                    rid,
                    len(failed_objects),
                )
                self._rollback_session_refs_locked(acquired)
                self.stats["load_fallback"] = self.stats.get("load_fallback", 0) + 1
                return False

            self.prepared_load_sessions[rid] = acquired
            acquired_sources = {session_sources.get(key) for key in acquired}
            source = (
                "dfs"
                if "dfs" in acquired_sources
                else "local_disk" if "local_disk" in acquired_sources else None
            )
            prepared_sources = getattr(self, "prepared_load_sources", None)
            if prepared_sources is None:
                prepared_sources = self.prepared_load_sources = {}
            prepared_sources[rid] = source
        return True

    def get_prepared_load_source(self, rid: str) -> str | None:
        return getattr(self, "prepared_load_sources", {}).get(rid)

    def _rollback_session_refs_locked(self, keys: list[str]) -> None:
        to_end = []
        for key in keys:
            count = self.session_refcounts.get(key, 0)
            if count <= 1:
                self.session_refcounts.pop(key, None)
                getattr(self, "session_sources", {}).pop(key, None)
                to_end.append(key)
            else:
                self.session_refcounts[key] = count - 1
        if to_end:
            try:
                self.storage.store.batch_get_session_end(to_end)
            except BaseException:
                logger.warning(
                    "Mooncake get session cleanup failed for %d keys.",
                    len(to_end),
                    exc_info=True,
                )

    def abort_prepared_load(self, rid: str) -> None:
        self.pending_loads.pop(rid, None)
        getattr(self, "prepared_load_sources", {}).pop(rid, None)
        with self.session_lock:
            keys = self.prepared_load_sessions.pop(rid, [])
            self._rollback_session_refs_locked(keys)

    def cancel_queued_load(self, rid: str) -> None:
        getattr(self, "pending_load_metrics", {}).pop(rid, None)
        self.abort_prepared_load(rid)

    def freeze_gc_once(self) -> None:
        if self.gc_frozen:
            return
        # Transfer metadata creates many short-lived lists. Keep the mature
        # model graph out of cyclic GC scans before load or offload traffic.
        freeze_gc("Mooncake direct linker")
        self.gc_frozen = True

    def start_layer_wise_loading(self) -> int:
        if not self.pending_loads:
            return -1
        self.freeze_gc_once()
        pending = self.pending_loads
        self.pending_loads = {}

        counter_index = self.layer_done_counter.update_producer()
        ready_event = device_module.Event()
        ready_event.record()
        self.load_queue.put((counter_index, list(pending.items()), ready_event))
        self.stats["load"] += len(pending)
        return counter_index

    def load_thread_func(self) -> None:
        while True:
            task = self.load_queue.get()
            try:
                if task is None:
                    return
                counter_index, transfers, ready_event = task
                ready_event.synchronize()
                self.load_layer_wise(counter_index, transfers)
            finally:
                self.load_queue.task_done()

    def load_layer_wise(
        self,
        counter_index: int,
        request_transfers: list[tuple[str, list[PoolTransfer]]],
    ) -> None:
        request_success = {rid: False for rid, _ in request_transfers}
        try:
            batches: dict[PoolName, tuple[list[str], list[int]]] = {}
            for _, transfers in request_transfers:
                for transfer in transfers:
                    keys, locations = batches.setdefault(transfer.name, ([], []))
                    component_keys, _ = self.storage._get_hybrid_page_component_keys(
                        list(transfer.keys), transfer
                    )
                    keys.extend(self.storage._tag_keys(component_keys))
                    locations.extend(
                        self.pools[transfer.name].prepare_locations(
                            transfer.host_indices
                        )
                    )

            if self.enable_page_wise_load and any(
                len(keys) >= self.page_wise_load_threshold
                for keys, _ in batches.values()
            ):
                request_success = self._load_page_wise(
                    counter_index, request_transfers, batches
                )
                if not all(request_success.values()):
                    raise RuntimeError(
                        "Mooncake page-wise load failed for one or more requests."
                    )
                return

            for layer in range(self.num_layers):
                for name, (keys, locations) in batches.items():
                    meta = self.pools[name].get_prepared_layer_range_meta(
                        locations, layer
                    )
                    if meta is None:
                        continue
                    ptrs, sizes, offsets = meta
                    result = self.storage.store.batch_get_into_multi_buffer_ranges(
                        keys,
                        ptrs,
                        sizes,
                        offsets,
                    )
                    expected = [sum(item) for item in sizes]
                    transferred = (
                        None
                        if result is None or isinstance(result, int)
                        else list(result)
                    )
                    if (
                        result is None
                        or isinstance(result, int)
                        or transferred != expected
                    ):
                        failed_objects = []
                        for index, key in enumerate(keys):
                            actual = (
                                result
                                if result is None or isinstance(result, int)
                                else (
                                    transferred[index]
                                    if index < len(transferred)
                                    else None
                                )
                            )
                            wanted = expected[index] if index < len(expected) else None
                            if actual != wanted:
                                failed_objects.append(
                                    {
                                        "key": key,
                                        "transferred": actual,
                                        "expected": wanted,
                                    }
                                )
                        logger.error(
                            "Mooncake lookup/session succeeded but range get failed: "
                            "rids=%s pool=%s layer=%d failed_objects=%s",
                            [rid for rid, _ in request_transfers],
                            name,
                            layer,
                            failed_objects,
                        )
                        raise RuntimeError(
                            f"Mooncake range get failed for pool={name}, "
                            f"layer={layer}, failed_objects={len(failed_objects)}."
                        )
                self.layer_done_counter.complete(counter_index, layer)
            request_success = {rid: True for rid, _ in request_transfers}
        except BaseException as error:
            self.layer_done_counter.fail(counter_index, error)
            logger.exception("Mooncake layer-wise load batch failed")
        finally:
            for rid, _ in request_transfers:
                self._finish_l4_metric(
                    "prefetch", rid, request_success.get(rid, False)
                )
                self.abort_prepared_load(rid)

    def _finish_l4_metric(self, operation: str, rid: str, success: bool) -> None:
        metric = getattr(self, "pending_load_metrics", {}).pop(rid, None)
        if metric is None:
            return
        source, tokens, started = metric
        if source not in ("local_disk", "dfs"):
            return
        duration = time.perf_counter() - started
        self._log_l4_metric(operation, source, tokens, duration, success)

    def _log_l4_metric(
        self,
        operation: str,
        source: str,
        tokens: int,
        duration: float,
        success: bool,
    ) -> None:
        collector = getattr(self, "storage_metrics_collector", None)
        if collector is None:
            return
        try:
            if operation == "prefetch":
                collector.log_l4_prefetch(source, tokens, duration, success)
            else:
                collector.log_l4_backup(source, tokens, duration, success)
        except BaseException:
            logger.warning(
                "Failed to record SGLang L4 %s metrics.",
                operation,
                exc_info=True,
            )

    def _load_page_wise(
        self,
        counter_index: int,
        request_transfers: list[tuple[str, list[PoolTransfer]]],
        batches: dict[PoolName, tuple[list[str], list[int]]],
    ) -> dict[str, bool]:
        """Load complete pages before releasing their layers to the consumer."""
        request_success = {rid: True for rid, _ in request_transfers}
        batch_rids: dict[PoolName, list[str]] = {}
        for rid, transfers in request_transfers:
            for transfer in transfers:
                component_keys, _ = self.storage._get_hybrid_page_component_keys(
                    list(transfer.keys), transfer
                )
                tagged_keys = self.storage._tag_keys(component_keys)
                batch_rids.setdefault(transfer.name, []).extend(
                    [rid] * len(tagged_keys)
                )

        for name, (keys, locations) in batches.items():
            ptrs: list[list[int]] = [[] for _ in keys]
            sizes: list[list[int]] = [[] for _ in keys]
            offsets: list[list[int]] = [[] for _ in keys]

            for layer in range(self.num_layers):
                meta = self.pools[name].get_prepared_layer_range_meta(
                    locations, layer
                )
                if meta is None:
                    continue
                layer_ptrs, layer_sizes, layer_offsets = meta
                if not (
                    len(layer_ptrs)
                    == len(layer_sizes)
                    == len(layer_offsets)
                    == len(keys)
                ):
                    raise ValueError(
                        f"Mooncake pool={name} layer={layer} produced "
                        f"{len(layer_ptrs)} range entries for {len(keys)} keys."
                    )
                for index in range(len(keys)):
                    ptrs[index].extend(layer_ptrs[index])
                    sizes[index].extend(layer_sizes[index])
                    offsets[index].extend(layer_offsets[index])

            for start in range(0, len(keys), self.page_wise_load_batch_size):
                end = start + self.page_wise_load_batch_size
                chunk_keys = keys[start:end]
                chunk_sizes = sizes[start:end]
                result = self.storage.store.batch_get_into_multi_buffer_ranges(
                    chunk_keys,
                    ptrs[start:end],
                    chunk_sizes,
                    offsets[start:end],
                )
                expected = [sum(item) for item in chunk_sizes]
                transferred = (
                    None if result is None or isinstance(result, int) else list(result)
                )
                chunk_rids = batch_rids.get(name, [None] * len(keys))[start:end]
                if transferred is None or transferred != expected:
                    failed_objects = []
                    for index, key in enumerate(chunk_keys):
                        actual = (
                            result
                            if result is None or isinstance(result, int)
                            else transferred[index]
                            if index < len(transferred)
                            else None
                        )
                        wanted = expected[index] if index < len(expected) else None
                        if actual != wanted:
                            rid = chunk_rids[index] if index < len(chunk_rids) else None
                            if rid is not None:
                                request_success[rid] = False
                            failed_objects.append(
                                {
                                    "key": key,
                                    "rid": rid,
                                    "transferred": actual,
                                    "expected": wanted,
                                }
                            )
                    logger.error(
                        "Mooncake page-wise range get failed: pool=%s "
                        "failed_objects=%s",
                        name,
                        failed_objects,
                    )

        # Page-wise loading intentionally gives up layer overlap: sessions must
        # be released only after every page is complete, and before any layer is
        # made visible to the model.
        for rid, _ in request_transfers:
            self.abort_prepared_load(rid)
        if all(request_success.values()):
            for layer in range(self.num_layers):
                self.layer_done_counter.complete(counter_index, layer)
        return request_success

    @staticmethod
    def _validate_range_get_result(
        result,
        keys: list[str],
        sizes: list[list[int]],
        request_transfers: list[tuple[str, list[PoolTransfer]]],
        name: PoolName,
        layer: int | None,
    ) -> None:
        expected = [sum(item) for item in sizes]
        transferred = (
            None if result is None or isinstance(result, int) else list(result)
        )
        if (
            result is not None
            and not isinstance(result, int)
            and transferred == expected
        ):
            return

        failed_objects = []
        for index, key in enumerate(keys):
            actual = (
                result
                if result is None or isinstance(result, int)
                else transferred[index] if index < len(transferred) else None
            )
            wanted = expected[index] if index < len(expected) else None
            if actual != wanted:
                failed_objects.append(
                    {"key": key, "transferred": actual, "expected": wanted}
                )
        location = "complete_page" if layer is None else f"layer={layer}"
        logger.error(
            "Mooncake lookup/session succeeded but range get failed: "
            "rids=%s pool=%s %s failed_objects=%s",
            [rid for rid, _ in request_transfers],
            name,
            location,
            failed_objects,
        )
        raise RuntimeError(
            f"Mooncake range get failed for pool={name}, {location}, "
            f"failed_objects={len(failed_objects)}."
        )

    def offload(self, transfers: list[PoolTransfer]) -> bool:
        expanded = self.pool_group.resolve_transfers(transfers, allow_partial=True)
        if not expanded:
            return False
        self.freeze_gc_once()
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        tokens = len(kv.keys) * self.page_size
        source = (
            "dfs"
            if getattr(self.storage, "dfs_replica_num", 0) > 0
            else "local_disk"
        )
        ready_event = device_module.Event()
        ready_event.record()
        self.offload_queue.put(
            (expanded, tokens, source, time.perf_counter(), ready_event)
        )
        return True

    def offload_thread_func(self) -> None:
        while True:
            task = self.offload_queue.get()
            metric_recorded = False
            try:
                if task is None:
                    return
                expanded, tokens, source, started, ready_event = task
                ready_event.synchronize()
                results = self.storage.batch_set_v2(expanded)
                success = all(all(pool_results) for pool_results in results.values())
                duration = time.perf_counter() - started
                self._log_l4_metric("backup", source, tokens, duration, success)
                metric_recorded = True
                if success:
                    self.stats["offload"] += 1
                    if self.stats["offload"] == 1:
                        logger.info("Mooncake direct linker offload: tokens=%d", tokens)
                self.offload_results.put(success)
            except BaseException:
                logger.exception("Mooncake offload failed")
                if task is not None and not metric_recorded:
                    _, tokens, source, started, _ = task
                    self._log_l4_metric(
                        "backup",
                        source,
                        tokens,
                        time.perf_counter() - started,
                        False,
                    )
                self.offload_results.put(False)
            finally:
                self.offload_queue.task_done()

    def num_completed_offloads(self) -> int:
        return self.offload_results.qsize()

    def pop_completed_offload(self) -> bool:
        return self.offload_results.get_nowait()

    def reset(self) -> None:
        for rid in list(self.pending_loads):
            getattr(self, "pending_load_metrics", {}).pop(rid, None)
            self.abort_prepared_load(rid)
        self.load_queue.join()
        self.offload_queue.join()
        for rid in list(self.prepared_load_sessions):
            self.abort_prepared_load(rid)
        while True:
            try:
                self.offload_results.get_nowait()
            except Empty:
                break
        self.layer_done_counter.reset()

    def close(self) -> None:
        self.reset()
        self.load_queue.put(None)
        self.offload_queue.put(None)
        self.load_thread.join()
        self.offload_thread.join()
        logger.info("Mooncake direct linker stats: %s", self.stats)
        self.storage.close()
