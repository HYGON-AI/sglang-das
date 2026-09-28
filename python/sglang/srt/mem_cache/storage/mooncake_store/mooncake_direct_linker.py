from __future__ import annotations

import logging
import os
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
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.unified_cache.linker_fault_injection import (
    arm_load_failure_injection,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinker,
)
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_STORAGE,
    StorageMetricsCollector,
    resolve_collector_class,
)
from sglang.srt.runtime_context import get_memory, get_model
from sglang.srt.utils import freeze_gc, get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()


def _get_mooncake_storage_metrics_dp_rank(server_args, params) -> int:
    if getattr(server_args, "enable_dp_attention", False):
        from sglang.srt.layers.dp_attention import get_attention_dp_rank

        return get_attention_dp_rank()
    return getattr(params, "dp_rank", None) or 0


def _storage_suffix(
    *, rank_replicated: bool, tp_rank: int, attn_cp_rank: int, pp_rank: int
) -> str:
    parts = []
    if not rank_replicated:
        parts.append(f"tp{tp_rank}")
    parts.extend((f"cp{attn_cp_rank}", f"pp{pp_rank}"))
    return "_".join(parts)


class LayerWiseLoadCounter:
    """CPU completion counter compatible with KV pools' layer wait hook."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.producer_index = -1
        self.consumer_index = -1
        self.futures: dict[int, list[Future]] = {}
        # Batch indices already logged: one line per failed load, not per layer.
        self.reported: set[int] = set()

    def update_producer(self) -> int:
        self.producer_index += 1
        self.futures[self.producer_index] = [Future() for _ in range(self.num_layers)]
        return self.producer_index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def complete(self, index: int, layer: int) -> None:
        self.futures[index][layer].set_result(None)

    def fail(self, index: int, error: BaseException) -> None:
        # A private copy, because wait_until clears the traceback of what it
        # catches and the caller still needs its own exception for the
        # logger.exception() that follows.
        failure = RuntimeError(f"{type(error).__name__}: {error}")
        for future in self.futures.get(index, ()):
            if not future.done():
                future.set_exception(failure)

    def wait_until(self, threshold: int) -> None:
        index = self.consumer_index
        futures = self.futures.get(index)
        if futures is None:
            return
        try:
            futures[threshold].result()
        except BaseException as error:
            # Never raise: this runs inside the model forward, where nothing
            # catches before the scheduler's top-level handler and the whole
            # engine goes down. The failure travels out through
            # pop_completed_load(), which aborts the affected requests.
            if index not in self.reported:
                self.reported.add(index)
                logger.error(
                    "Mooncake layer-wise KV load failed for batch %d; affected "
                    "requests will be aborted after this forward: %s",
                    index,
                    error,
                )
            # Every layer re-raises this same exception object, and each
            # appended traceback entry roots a frame chain that pins that
            # layer's activations: +31.9 GiB on one faulted 61-layer forward at
            # --chunked-prefill-size 2048, until the engine OOMed. Nothing
            # reads the traceback.
            error.__traceback__ = None
        finally:
            if threshold == self.num_layers - 1:
                self.futures.pop(index, None)
                self.reported.discard(index)

    def reset(self) -> None:
        self.producer_index = -1
        self.consumer_index = -1
        self.futures.clear()
        self.reported.clear()


class ReadPlanLoadCounter:
    """Publish one Mooncake ReadPlan; wait for each layer without holding the GIL."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.producer_index = self.consumer_index = -1
        self.plans: dict[int, Future] = {}
        # Batch indices already logged: one line per failed plan, not per layer.
        self.reported: set[int] = set()

    def update_producer(self) -> int:
        self.producer_index += 1
        self.plans[self.producer_index] = Future()
        return self.producer_index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def bind(self, index: int, plan) -> None:
        self.plans[index].set_result(plan)

    def fail(self, index: int, error: BaseException) -> None:
        future = self.plans.get(index)
        if future is not None and not future.done():
            future.set_exception(error)

    def wait_until(self, threshold: int) -> None:
        index = self.consumer_index
        future = self.plans.get(index)
        if future is None:
            return
        try:
            future.result().wait(threshold)
        except BaseException as error:
            # Do not raise from the model forward. A rank-local failure here can
            # strand peer TP/CP ranks in a later model collective and brings the
            # scheduler down. load_layer_wise reports False through the linker
            # completion queue; the cache MIN-reduces that verdict across the
            # attention group and aborts the affected requests after forward.
            if index not in self.reported:
                self.reported.add(index)
                logger.error(
                    "Mooncake ReadPlan KV load failed for batch %d; affected "
                    "requests will be aborted after this forward: %s",
                    index,
                    error,
                )
            # Repeated waits otherwise grow a traceback chain that keeps each
            # layer's activations alive until the failed plan is retired.
            error.__traceback__ = None
        finally:
            if threshold == self.num_layers - 1:
                self.plans.pop(index, None)
                self.reported.discard(index)

    def reset(self) -> None:
        self.producer_index = self.consumer_index = -1
        self.plans.clear()
        self.reported.clear()


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
        self.enable_page_wise_load = server_args.mooncake_enable_page_wise_load
        if self.page_wise_load_threshold <= 0:
            raise ValueError(
                "--mooncake-page-wise-load-threshold must be positive, got "
                f"{self.page_wise_load_threshold}."
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
        self._load_revalidation_groups = ()
        if self.pool_group.storage_layout_tag:
            self._load_revalidation_groups = tuple(
                group
                for group in (params.attn_cp_cache_group, params.attn_tp_cache_group)
                if group is not None
                and torch.distributed.get_world_size(group=group) > 1
            )

        tp_rank = 0
        tp_size = server_args.tp_size
        tp_group = params.attn_tp_cache_group or params.tp_cache_group
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tp_rank = torch.distributed.get_rank(group=tp_group)
            tp_size = torch.distributed.get_world_size(group=tp_group)
        rank_replicated = self.pool_group.rank_replicated
        self.offload_owner = not rank_replicated or tp_rank == 0
        self.tp_rank = tp_rank
        extra_config, *_ = HybridCacheController.parse_storage_backend_extra_config(
            get_memory().hicache_storage_backend_extra_config
        )
        storage_config = HiCacheStorageConfig(
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=params.pp_rank,
            pp_size=params.pp_size,
            attn_cp_rank=params.attn_cp_rank,
            attn_cp_size=params.attn_cp_size,
            is_mla_model=rank_replicated,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name=get_model().model_path,
            extra_config=extra_config,
            dp_rank=getattr(params, "dp_rank", None),
        )
        if storage is None:
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStore,
            )

            self.storage = MooncakeStore(
                storage_config,
                mem_pool=None,
                enable_client_http_server=params.enable_metrics,
            )
        else:
            self.storage = storage
        self.storage.mem_pool_host = self.pool_group
        self.storage.registered_pools = self.pools
        storage_suffix = _storage_suffix(
            rank_replicated=rank_replicated,
            tp_rank=tp_rank,
            attn_cp_rank=params.attn_cp_rank,
            pp_rank=params.pp_rank,
        )
        if self.pool_group.storage_layout_tag:
            storage_suffix = f"{self.pool_group.storage_layout_tag}_{storage_suffix}"
        self.storage.mla_suffix = storage_suffix
        self.storage.mha_suffix = storage_suffix
        logger.info(
            "Mooncake direct linker storage topology: "
            "rank_replicated=%s, tp_rank=%d/%d, offload_owner=%s, suffix=%s",
            rank_replicated,
            tp_rank,
            tp_size,
            self.offload_owner,
            storage_suffix,
        )
        self.read_plan_enabled = os.environ.get("SGLANG_MOONCAKE_READ_PLAN", "0") == "1"
        self.read_plan_reuse_ranges = (
            os.environ.get("SGLANG_MOONCAKE_READ_PLAN_REUSE_RANGES", "0") == "1"
        )
        self.host_prefetch_enabled = bool(
            getattr(
                server_args,
                "mooncake_enable_waiting_queue_dfs_prefetch",
                False,
            )
        )
        self.host_prefetch_limit = int(
            getattr(
                server_args,
                "mooncake_waiting_queue_dfs_prefetch_max_requests",
                8,
            )
        )
        self.host_prefetch_worker_count = int(
            getattr(
                server_args,
                "mooncake_waiting_queue_dfs_prefetch_workers",
                2,
            )
        )
        self.host_prefetch_max_bytes = int(
            getattr(
                server_args,
                "mooncake_waiting_queue_dfs_prefetch_max_bytes",
                1073741824,
            )
        )
        if self.host_prefetch_limit <= 0:
            raise ValueError(
                "--mooncake-waiting-queue-dfs-prefetch-max-requests must be "
                f"positive, got {self.host_prefetch_limit}."
            )
        if self.host_prefetch_worker_count <= 0:
            raise ValueError(
                "--mooncake-waiting-queue-dfs-prefetch-workers must be "
                f"positive, got {self.host_prefetch_worker_count}."
            )
        self.host_prefetch_worker_count = min(
            self.host_prefetch_worker_count, self.host_prefetch_limit
        )
        if self.host_prefetch_max_bytes <= 0:
            raise ValueError(
                "--mooncake-waiting-queue-dfs-prefetch-max-bytes must be "
                f"positive, got {self.host_prefetch_max_bytes}."
            )
        if self.read_plan_reuse_ranges and not self.read_plan_enabled:
            raise ValueError(
                "SGLANG_MOONCAKE_READ_PLAN_REUSE_RANGES requires "
                "SGLANG_MOONCAKE_READ_PLAN=1"
            )
        if self.read_plan_reuse_ranges and self.enable_page_wise_load:
            raise ValueError(
                "SGLANG_MOONCAKE_READ_PLAN_REUSE_RANGES is incompatible with "
                "page-wise Mooncake loads"
            )
        if self.read_plan_enabled:
            if not callable(getattr(self.storage.store, "create_read_plan", None)):
                raise RuntimeError(
                    "Mooncake ReadPlan requires a package with create_read_plan. "
                    "Install the ReadPlan-enabled Mooncake package."
                )
            logger.info(
                "Mooncake ReadPlan enabled; address reuse=%s",
                self.read_plan_reuse_ranges,
            )
        self.host_prefetch_enabled = self._resolve_host_prefetch_enabled(
            params, requested=self.host_prefetch_enabled
        )

        self.storage_metrics_collector = None
        if params.enable_metrics:
            labels = {
                "storage_backend": "mooncake_direct",
                "tp_rank": tp_rank,
                "dp_rank": _get_mooncake_storage_metrics_dp_rank(server_args, params),
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
        if self.read_plan_enabled:
            self.layer_done_counter = ReadPlanLoadCounter(self.num_layers)
        else:
            self.layer_done_counter = LayerWiseLoadCounter(self.num_layers)
        if PoolName.MAMBA in self.pools:
            params.req_to_token_pool.register_layer_transfer_counter(
                self.layer_done_counter
            )
        self.pending_loads: dict[str, list[PoolTransfer]] = {}
        self.pending_load_tokens: dict[str, int] = {}
        self.prepared_load_sessions: dict[str, list[str]] = {}
        self.session_refcounts: dict[str, int] = {}
        self.session_sources: dict[str, str] = {}
        self.session_lock = threading.Lock()
        self.host_prefetch_lock = threading.Lock()
        self.host_prefetch_entries: dict[str, dict[str, object]] = {}
        self.host_prefetch_generation = 0
        self.host_prefetch_queue: Queue[
            tuple[str, list[PoolTransfer]] | None
        ] = Queue()
        self.gc_frozen = False
        self.load_queue: Queue[
            tuple[int, dict[str, list[PoolTransfer]], object] | None
        ] = Queue()
        # (rids, success) per started load batch. Pushed from a finally so a
        # failed batch still releases the tree-side locks it pinned.
        self.completed_loads: Queue[tuple[list[str], bool]] = Queue()
        self.offload_queue: Queue[
            tuple[list[PoolTransfer], int, float, object] | None
        ] = Queue()
        self.offload_results: Queue[bool] = Queue()
        self.stats = {"lookup": 0, "load": 0, "offload": 0}
        self.load_thread = threading.Thread(
            target=self.load_thread_func,
            daemon=True,
            name=f"mooncake-load-tp{tp_rank}",
        )
        self.load_thread.start()
        self.host_prefetch_threads: list[threading.Thread] = []
        if self.host_prefetch_enabled:
            for worker_id in range(self.host_prefetch_worker_count):
                thread = threading.Thread(
                    target=self.host_prefetch_thread_func,
                    daemon=True,
                    name=f"mooncake-host-prefetch-tp{tp_rank}-{worker_id}",
                )
                self.host_prefetch_threads.append(thread)
                thread.start()
        self.offload_thread = threading.Thread(
            target=self.offload_thread_func,
            daemon=True,
            name=f"mooncake-offload-tp{tp_rank}",
        )
        self.offload_thread.start()

    def _local_host_prefetch_capability(self, requested: bool) -> tuple[bool, str]:
        """Return whether this rank can run waiting-queue DFS prefetch."""
        if not requested:
            return False, "disabled by --mooncake-enable-waiting-queue-dfs-prefetch"
        store = self.storage.store
        missing = [
            name
            for name in (
                "batch_get_session_start_with_sources",
                "batch_get_session_prefetch",
                "batch_get_session_refresh",
                "dfs_prefetch_arena_available",
            )
            if not callable(getattr(store, name, None))
        ]
        if missing:
            return False, "Mooncake package lacks " + ", ".join(missing)
        try:
            if store.dfs_prefetch_arena_available():
                return True, "ready"
            status = getattr(store, "dfs_prefetch_arena_status", None)
            reason = status() if callable(status) else "unavailable"
        except Exception as exc:
            reason = f"arena query failed: {exc!r}"
        return False, (
            f"DFS prefetch arena unavailable ({reason}); set "
            "MC_STORE_DFS_PREFETCH_ARENA_SIZE_BYTES on prefill"
        )

    def _resolve_host_prefetch_enabled(
        self, params: CacheInitParams, *, requested: bool
    ) -> bool:
        """Agree on waiting-queue prefetch once, before any worker starts.

        Every rank that builds this linker takes part unconditionally, so the
        scheduler's per-request prefetch collectives are entered by all ranks
        or by none. The groups and their order match
        ``UnifiedRadixCache._all_reduce_attn_groups``.
        """
        local_ok, reason = self._local_host_prefetch_capability(requested)
        verdict = torch.tensor([int(local_ok)], dtype=torch.int)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            reduced = False
            for group in (params.attn_cp_cache_group, params.attn_tp_cache_group):
                if (
                    group is not None
                    and torch.distributed.get_world_size(group=group) > 1
                ):
                    torch.distributed.all_reduce(
                        verdict, op=torch.distributed.ReduceOp.MIN, group=group
                    )
                    reduced = True
            tp_group = params.tp_cache_group
            if (
                not reduced
                and tp_group is not None
                and torch.distributed.get_world_size(group=tp_group) > 1
            ):
                torch.distributed.all_reduce(
                    verdict, op=torch.distributed.ReduceOp.MIN, group=tp_group
                )
        enabled = bool(verdict.item())
        if requested and not enabled:
            logger.warning(
                "Mooncake waiting-queue DFS prefetch disabled on all ranks; "
                "local status: %s (if ready, another rank is unavailable)",
                reason,
            )
        elif enabled:
            logger.info("Mooncake waiting-queue DFS prefetch enabled")
        return enabled

    def waiting_queue_prefetch_enabled(self) -> bool:
        return self.host_prefetch_enabled

    def disable_waiting_queue_prefetch(self) -> None:
        # The scheduler found a rank without prefetch before any submission;
        # the idle host-prefetch workers stay parked on an empty queue.
        self.host_prefetch_enabled = False

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
        # PP0 queries every PP shard; the wrapper's existing TP/CP reduction
        # then selects a boundary that all ranks can restore.
        result = self.storage.batch_exists_v2(page_keys, expanded, query_all_pp=True)
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

    def revalidate_load(self, transfers: list[PoolTransfer]) -> bool:
        valid = self._revalidate_load_local(transfers)
        groups = getattr(self, "_load_revalidation_groups", ())
        if groups:
            # LayerSplit shards can be evicted independently after lookup.
            # All CP ranks must either restore or recompute the same prefix.
            verdict = torch.tensor([int(valid)], dtype=torch.int)
            for group in groups:
                torch.distributed.all_reduce(
                    verdict, op=torch.distributed.ReduceOp.MIN, group=group
                )
            valid = bool(verdict.item())
        return valid

    def _revalidate_load_local(self, transfers: list[PoolTransfer]) -> bool:
        """Re-check remote existence before the scheduler commits device
        slots to the async layer-wise load.

        The match-time lookup and ``batch_get_session_start`` can be minutes
        apart; master-side memory-watermark eviction in that window expires
        replicas and would fail the layer-wise load fatally. Re-checking here
        lets the caller degrade the request to a plain cache miss instead.

        Failures inside the revalidation itself never propagate: this hook
        guards a hot scheduling path, so an unexpected error falls back to
        "validated" and the async load path retains its own semantics.
        """
        try:
            resolved = self.pool_group.resolve_transfers(
                transfers, allow_partial=True, allow_missing_kv=True
            )
            if not resolved:
                return True
            key_strs: list[str] = []
            for transfer in resolved:
                component_keys, _ = self.storage._get_hybrid_page_component_keys(
                    list(transfer.keys), transfer
                )
                key_strs.extend(self.storage._tag_keys(component_keys))
            if not key_strs:
                return True
            exist = self.storage._batch_exist(key_strs)
        except BaseException:
            logger.exception("Mooncake direct linker load revalidation error")
            return True
        missing = [key for key, state in zip(key_strs, exist) if state != 1]
        if missing:
            logger.warning(
                "Mooncake direct linker load revalidation failed: "
                "missing=%d/%d keys (master eviction race), "
                "degrading request to cache miss",
                len(missing),
                len(key_strs),
            )
            failed_get_cache = getattr(self.storage, "failed_get_cache", None)
            if failed_get_cache is not None:
                failed_get_cache.update_batch([], missing)
            return False
        return True

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        # Query establishes a boundary at which every component is restorable;
        # insert then removes pages already resident in L1. Loading is therefore
        # intentionally partial and may contain only a side pool such as SWA.
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        if not expanded:
            return False
        if rid in self.pending_loads:
            raise RuntimeError(f"Mooncake load for rid={rid} is already queued.")
        self.pending_loads[rid] = expanded
        with self.host_prefetch_lock:
            entry = self.host_prefetch_entries.get(rid)
            if entry is not None and entry.get("state") == "claimed":
                entry["state"] = "loading"
        logical_pages = {
            page_key for transfer in expanded for page_key in transfer.keys
        }
        pending_load_tokens = getattr(self, "pending_load_tokens", None)
        if pending_load_tokens is None:
            pending_load_tokens = self.pending_load_tokens = {}
        pending_load_tokens[rid] = len(logical_pages) * self.page_size
        return True

    def prepare_load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        """Acquire key-indexed Mooncake sessions for a future range read."""
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        if not expanded:
            return False
        return self._prepare_expanded_load(rid, expanded)

    def _prepare_expanded_load(
        self, rid: str, expanded: list[PoolTransfer]
    ) -> bool:
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
        if not keys:
            return False

        with self.session_lock:
            if rid in self.prepared_load_sessions:
                return True
            acquired = []
            new_keys = []
            for key in keys:
                if key in self.session_refcounts:
                    self.session_refcounts[key] += 1
                    self.session_sources.setdefault(key, "unknown")
                    acquired.append(key)
                else:
                    new_keys.append(key)

            try:
                if new_keys and self.host_prefetch_enabled:
                    results, sources = (
                        self.storage.store.batch_get_session_start_with_sources(
                            new_keys
                        )
                    )
                    results = list(results)
                    sources = list(sources)
                elif new_keys:
                    results = list(
                        self.storage.store.batch_get_session_start(new_keys)
                    )
                    sources = ["unknown"] * len(results)
                else:
                    results = []
                    sources = []
            except BaseException:
                logger.warning(
                    "Mooncake get session start raised for rid=%s", rid, exc_info=True
                )
                self._rollback_session_refs_locked(acquired)
                return False

            failed = len(results) != len(new_keys) or any(
                result != 0 for result in results
            )
            for index, key in enumerate(new_keys):
                if index < len(results) and results[index] == 0:
                    self.session_refcounts[key] = 1
                    source = sources[index] if index < len(sources) else "unknown"
                    self.session_sources[key] = str(source).lower()
                    acquired.append(key)
            if failed:
                self._rollback_session_refs_locked(acquired)
                return False

            self.prepared_load_sessions[rid] = acquired
        return True

    @staticmethod
    def _mooncake_dfs_alignment() -> int:
        try:
            return max(
                4096,
                int(os.environ.get("MOONCAKE_DFS_ALIGNMENT", "4096") or 4096),
            )
        except ValueError:
            return 4096

    @staticmethod
    def _estimate_pinned_arena_capacity(
        object_sizes: list[int], alignment: int
    ) -> int:
        """Conservatively mirror Mooncake's aligned DFS arena allocation."""
        arena_size = 0
        for object_size in object_sizes:
            aligned_object_size = (
                (object_size + alignment - 1) // alignment
            ) * alignment
            arena_size = ((arena_size + alignment - 1) // alignment) * alignment
            arena_size += aligned_object_size

        if not arena_size:
            return 0
        padded_size = arena_size + alignment - 1
        if padded_size <= 1 << 20:
            capacity = 4 << 10
            while capacity < padded_size:
                capacity *= 2
            return capacity
        large_alignment = 2 << 20
        return (
            (padded_size + large_alignment - 1) // large_alignment
        ) * large_alignment

    def _get_host_prefetch_object_sizes(
        self, transfers: list[PoolTransfer]
    ) -> dict[str, int] | None:
        """Map tagged storage keys to one page's exact payload size."""
        try:
            expanded = self.pool_group.resolve_transfers(
                transfers, allow_partial=True, allow_missing_kv=True
            )
            if not expanded:
                return None

            object_sizes: dict[str, int] = {}
            for transfer in expanded:
                page_keys = list(transfer.keys or ())
                if not page_keys:
                    continue
                component_keys, key_multiplier = (
                    self.storage._get_hybrid_page_component_keys(
                        page_keys, transfer
                    )
                )
                pool = self.pools.get(transfer.name)
                if pool is None or key_multiplier <= 0:
                    return None

                # Each metadata row describes one logical page. Group its
                # buffer sizes the same way MooncakeStore packs buffers into
                # the component keys returned above.
                page_buffer_sizes = [
                    size
                    for component in pool.buffer_meta
                    for _, _, size in component
                ]
                if (
                    not page_buffer_sizes
                    or len(page_buffer_sizes) % key_multiplier
                    or len(component_keys) != len(page_keys) * key_multiplier
                ):
                    return None
                buffers_per_object = len(page_buffer_sizes) // key_multiplier
                if buffers_per_object <= 0:
                    return None
                object_payload_sizes = [
                    sum(
                        page_buffer_sizes[
                            object_index
                            * buffers_per_object : (object_index + 1)
                            * buffers_per_object
                        ]
                    )
                    for object_index in range(key_multiplier)
                ]
                tagged_keys = self.storage._tag_keys(component_keys)
                for index, key in enumerate(tagged_keys):
                    size = object_payload_sizes[index % key_multiplier]
                    if size <= 0:
                        return None
                    object_sizes[key] = max(object_sizes.get(key, 0), size)
            return object_sizes or None
        except Exception:
            logger.warning(
                "Could not estimate Mooncake waiting-queue prefetch bytes.",
                exc_info=True,
            )
            return None

    def submit_host_prefetch(
        self, rid: str, transfers: list[PoolTransfer]
    ) -> bool:
        if not self.host_prefetch_enabled:
            return False
        kv_transfer = next(
            (transfer for transfer in transfers if transfer.name == PoolName.KV), None
        )
        if kv_transfer is None or not kv_transfer.keys:
            return False
        object_sizes = self._get_host_prefetch_object_sizes(transfers)
        if object_sizes is None:
            return False

        alignment = self._mooncake_dfs_alignment()
        estimated_bytes = self._estimate_pinned_arena_capacity(
            list(object_sizes.values()), alignment
        )
        if not estimated_bytes or estimated_bytes > self.host_prefetch_max_bytes:
            return False

        with self.host_prefetch_lock:
            active = sum(
                entry.get("state")
                in {
                    "queued",
                    "preparing",
                    "reading",
                    "dfs_prefetched",
                    "claimed",
                    "loading",
                }
                for entry in self.host_prefetch_entries.values()
            )
            reserved_bytes = sum(
                int(entry.get("reserved_bytes", 0))
                for entry in self.host_prefetch_entries.values()
            )
            if (
                rid in self.host_prefetch_entries
                or active >= self.host_prefetch_limit
                or reserved_bytes + estimated_bytes > self.host_prefetch_max_bytes
            ):
                return False
            self.host_prefetch_generation += 1
            generation = self.host_prefetch_generation
            queued_transfers = list(transfers)
            self.host_prefetch_entries[rid] = {
                "state": "queued",
                "cancelled": False,
                "transfers": queued_transfers,
                "session_rid": self._host_prefetch_session_rid(rid, generation),
                "object_sizes": object_sizes,
                "reserved_bytes": estimated_bytes,
            }

        # Session creation can contact DFS. The scheduler only publishes the
        # work item; this worker owns both preparation and the DFS read.
        self.host_prefetch_queue.put((rid, queued_transfers))
        return True

    def get_host_prefetch_status(self, rid: str) -> str | None:
        with self.host_prefetch_lock:
            entry = self.host_prefetch_entries.get(rid)
            if entry is None:
                return None
            return str(entry.get("state"))

    def revalidate_host_prefetch(self, rid: str) -> bool:
        with self.host_prefetch_lock:
            entry = self.host_prefetch_entries.get(rid)
            if (
                entry is None
                or entry.get("state") != "dfs_prefetched"
                or entry.get("cancelled")
            ):
                return False
            session_rid = str(entry["session_rid"])
        with self.session_lock:
            keys = list(self.prepared_load_sessions.get(session_rid, ()))
        if not keys:
            return False
        try:
            results = list(self.storage.store.batch_get_session_refresh(keys))
        except BaseException:
            logger.warning(
                "Mooncake waiting-queue prefetch refresh failed for rid=%s",
                rid,
                exc_info=True,
            )
            return False
        return len(results) == len(keys) and all(result == 0 for result in results)

    def revalidate_no_prefetch_needed(self, rid: str) -> bool:
        """Re-check the keys of a request that needed no DFS prefetch.

        The worker released its session on finding no DFS keys, so nothing
        pins the objects while the request waits for admission. Returning
        False sends every rank back to the normal path, whose revalidation
        degrades an evicted prefix to a cache miss.
        """
        with self.host_prefetch_lock:
            entry = self.host_prefetch_entries.get(rid)
            if (
                entry is None
                or entry.get("state") != "no_prefetch_needed"
                or entry.get("cancelled")
            ):
                return False
            keys = list(entry.get("no_prefetch_keys", ()))
        if not keys:
            return False
        try:
            exist = list(self.storage._batch_exist(keys))
        except BaseException:
            logger.warning(
                "Mooncake waiting-queue existence check failed for rid=%s",
                rid,
                exc_info=True,
            )
            return False
        missing = len(keys) - sum(1 for state in exist if state == 1)
        if len(exist) != len(keys) or missing:
            logger.warning(
                "Mooncake waiting-queue keys changed while queued for rid=%s: "
                "missing=%d/%d, falling back to the normal load path",
                rid,
                missing,
                len(keys),
            )
            return False
        return True

    def claim_ready_host_prefetch(self, rid: str) -> bool:
        with self.host_prefetch_lock:
            entry = self.host_prefetch_entries.get(rid)
            if entry is None or entry.get("state") != "dfs_prefetched":
                return False
            session_rid = str(entry["session_rid"])
            with self.session_lock:
                if (
                    rid in self.prepared_load_sessions
                    or session_rid not in self.prepared_load_sessions
                ):
                    return False
                self.prepared_load_sessions[rid] = self.prepared_load_sessions.pop(
                    session_rid
                )
            entry["state"] = "claimed"
        return True

    def cancel_host_prefetch(self, rid: str) -> None:
        with self.host_prefetch_lock:
            entry = self.host_prefetch_entries.get(rid)
            if entry is None:
                return
            state = str(entry.get("state"))
            session_rid = str(entry["session_rid"])
            if state in {"preparing", "reading", "loading"}:
                entry["cancelled"] = True
                return
            self.host_prefetch_entries.pop(rid, None)
        self._abort_prepared_load_now(rid if state == "claimed" else session_rid)

    @staticmethod
    def _host_prefetch_session_rid(rid: str, generation: int) -> str:
        return f"__sglang_waiting_queue_prefetch__:{rid}:{generation}"

    def host_prefetch_thread_func(self) -> None:
        while True:
            task = self.host_prefetch_queue.get()
            try:
                if task is None:
                    return
                rid, transfers = task
                with self.host_prefetch_lock:
                    entry = self.host_prefetch_entries.get(rid)
                    if entry is None or entry.get("transfers") is not transfers:
                        continue
                    session_rid = str(entry["session_rid"])
                    if entry.get("cancelled"):
                        self.host_prefetch_entries.pop(rid, None)
                        continue
                    entry["state"] = "preparing"

                try:
                    prepared = self.prepare_load(session_rid, transfers)
                except BaseException:
                    logger.warning(
                        "Mooncake waiting-queue session preparation failed "
                        "for rid=%s",
                        rid,
                        exc_info=True,
                    )
                    prepared = False
                with self.session_lock:
                    keys = list(self.prepared_load_sessions.get(session_rid, ()))
                if not prepared or not keys:
                    self._finish_host_prefetch(
                        rid, entry, session_rid, "failed"
                    )
                    continue

                with self.session_lock:
                    sources = {
                        key: self.session_sources.get(key, "unknown")
                        for key in keys
                    }
                object_sizes = entry.get("object_sizes", {})
                if not isinstance(object_sizes, dict) or any(
                    key not in object_sizes for key in keys
                ):
                    self._finish_host_prefetch(
                        rid, entry, session_rid, "failed"
                    )
                    continue
                dfs_keys = [
                    key
                    for key in keys
                    if sources[key] in {"dfs", "unknown"}
                ]
                if not dfs_keys:
                    self._finish_host_prefetch(
                        rid, entry, session_rid, "no_prefetch_needed", keys=keys
                    )
                    continue

                alignment = self._mooncake_dfs_alignment()
                dfs_estimated_bytes = self._estimate_pinned_arena_capacity(
                    [int(object_sizes[key]) for key in dfs_keys], alignment
                )
                with self.host_prefetch_lock:
                    current = self.host_prefetch_entries.get(rid)
                    if current is not entry or current.get("cancelled"):
                        cancelled = True
                        within_budget = False
                    else:
                        cancelled = False
                        other_reserved_bytes = sum(
                            int(other.get("reserved_bytes", 0))
                            for other_rid, other in self.host_prefetch_entries.items()
                            if other_rid != rid
                        )
                        within_budget = (
                            other_reserved_bytes + dfs_estimated_bytes
                            <= self.host_prefetch_max_bytes
                        )
                        if within_budget:
                            current["reserved_bytes"] = dfs_estimated_bytes
                            current["state"] = "reading"
                if cancelled:
                    self._finish_host_prefetch(
                        rid, entry, session_rid, "failed"
                    )
                    continue
                if not within_budget:
                    self._finish_host_prefetch(
                        rid, entry, session_rid, "failed"
                    )
                    continue

                try:
                    results = list(
                        self.storage.store.batch_get_session_prefetch(dfs_keys)
                    )
                    success = len(results) == len(dfs_keys) and all(
                        result == 0 for result in results
                    )
                except BaseException:
                    logger.warning(
                        "Mooncake waiting-queue DFS prefetch failed for rid=%s",
                        rid,
                        exc_info=True,
                    )
                    success = False
                self._finish_host_prefetch(
                    rid,
                    entry,
                    session_rid,
                    "dfs_prefetched" if success else "failed",
                )
            finally:
                self.host_prefetch_queue.task_done()

    def _finish_host_prefetch(
        self,
        rid: str,
        entry: dict[str, object],
        session_rid: str,
        outcome: str,
        *,
        keys: list[str] | None = None,
    ) -> None:
        keep_session = False
        with self.host_prefetch_lock:
            current = self.host_prefetch_entries.get(rid)
            cancelled = bool(
                current is not entry or current is None or current.get("cancelled")
            )
            if (
                current is entry
                and outcome in {"dfs_prefetched", "no_prefetch_needed"}
                and not cancelled
            ):
                if outcome == "no_prefetch_needed":
                    # The session is released below; keep the keys so
                    # admission can re-check them (see
                    # revalidate_no_prefetch_needed). Published together with
                    # the state under the same lock.
                    current["no_prefetch_keys"] = list(keys or ())
                    current["reserved_bytes"] = 0
                current["state"] = outcome
                keep_session = outcome == "dfs_prefetched"
            if current is entry:
                if cancelled:
                    self.host_prefetch_entries.pop(rid, None)
                elif outcome not in {"dfs_prefetched", "no_prefetch_needed"}:
                    current["state"] = "failed"
                    current["reserved_bytes"] = 0
        if not keep_session:
            self._abort_prepared_load_now(session_rid)

    def _rollback_session_refs_locked(self, keys: list[str]) -> None:
        to_end = []
        for key in keys:
            count = self.session_refcounts.get(key, 0)
            if count <= 1:
                self.session_refcounts.pop(key, None)
                self.session_sources.pop(key, None)
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

    def _abort_prepared_load_now(self, rid: str) -> None:
        with self.session_lock:
            keys = self.prepared_load_sessions.pop(rid, [])
            self._rollback_session_refs_locked(keys)

    def abort_prepared_load(self, rid: str) -> None:
        with self.host_prefetch_lock:
            entry = self.host_prefetch_entries.get(rid)
            if entry is not None and entry.get("state") in {"claimed", "loading"}:
                self.host_prefetch_entries.pop(rid, None)
        self._abort_prepared_load_now(rid)

    def cancel_queued_load(self, rid: str) -> bool:
        getattr(self, "pending_load_tokens", {}).pop(rid, None)
        cancelled = self.pending_loads.pop(rid, None) is not None
        if cancelled:
            self.abort_prepared_load(rid)
        return cancelled

    def num_completed_loads(self) -> int:
        return self.completed_loads.qsize()

    def pop_completed_load(self) -> tuple[list[str], bool]:
        return self.completed_loads.get_nowait()

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
        self.load_queue.put((counter_index, pending, ready_event))
        self.stats["load"] += len(pending)
        return counter_index

    def load_thread_func(self) -> None:
        while True:
            task = self.load_queue.get()
            try:
                if task is None:
                    return
                counter_index, pending, ready_event = task
                success = False
                try:
                    ready_event.synchronize()
                    success = self.load_layer_wise(
                        counter_index, list(pending.items())
                    )
                except BaseException as error:
                    self.layer_done_counter.fail(counter_index, error)
                    logger.exception("Mooncake layer-wise load batch failed")
                finally:
                    self._finish_prefetch_metrics(list(pending), success is True)
                    self.completed_loads.put((list(pending), success))
            finally:
                self.load_queue.task_done()

    def load_layer_wise(
        self,
        counter_index: int,
        request_transfers: list[tuple[str, list[PoolTransfer]]],
    ) -> bool:
        success = False
        maybe_fail = arm_load_failure_injection(self.tp_rank)
        try:
            for rid, transfers in request_transfers:
                with self.session_lock:
                    prepared = rid in self.prepared_load_sessions
                if not prepared and not self._prepare_expanded_load(rid, transfers):
                    raise RuntimeError(
                        f"Mooncake get session preparation failed for rid={rid}."
                    )
            if getattr(self, "read_plan_enabled", False):
                self.load_with_read_plan(counter_index, request_transfers)
                return True
            batches: dict[PoolName, tuple[list[str], list[int]]] = {}
            for _rid, transfers in request_transfers:
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
                self._load_page_wise(counter_index, batches, maybe_fail)
                success = True
                return success

            for layer in range(self.num_layers):
                for name, (keys, locations) in batches.items():
                    meta = self.pools[name].get_prepared_layer_range_meta(
                        locations, layer
                    )
                    if meta is None:
                        continue
                    ptrs, sizes, offsets = meta
                    maybe_fail(name, f"layer={layer}")
                    result = self.storage.store.batch_get_into_multi_buffer_ranges(
                        keys,
                        ptrs,
                        sizes,
                        offsets,
                    )
                    expected = [sum(item) for item in sizes]
                    if (
                        result is None
                        or isinstance(result, int)
                        or list(result) != expected
                    ):
                        raise RuntimeError(
                            f"Mooncake range get failed for pool={name}, "
                            f"layer={layer}: transferred={result}, "
                            f"expected={expected}"
                        )
                self.layer_done_counter.complete(counter_index, layer)
            success = True
        except BaseException as error:
            self.layer_done_counter.fail(counter_index, error)
            logger.exception("Mooncake layer-wise load batch failed")
        finally:
            for rid, _ in request_transfers:
                self.abort_prepared_load(rid)
        return success

    def _finish_prefetch_metrics(self, rids: list[str], success: bool) -> None:
        pending_load_tokens = getattr(self, "pending_load_tokens", {})
        tokens = sum(pending_load_tokens.pop(rid, 0) for rid in rids)
        if not success or tokens <= 0:
            return
        self._log_prefetched_tokens(tokens)
        recorder = getattr(
            getattr(self.storage, "store", None), "record_prefetched_tokens", None
        )
        if recorder is not None:
            try:
                recorder(tokens)
            except BaseException:
                logger.warning(
                    "Failed to record Mooncake prefetched token metric.",
                    exc_info=True,
                )

    def _log_prefetched_tokens(self, tokens: int) -> None:
        collector = getattr(self, "storage_metrics_collector", None)
        if collector is None or tokens <= 0:
            return
        try:
            collector.log_prefetched_tokens(tokens)
        except BaseException:
            logger.warning(
                "Failed to record SGLang direct-storage prefetch metrics.",
                exc_info=True,
            )

    def _load_page_wise(self, counter_index: int, batches, maybe_fail) -> None:
        """Load all layer ranges for each page before exposing the data."""
        all_keys: list[str] = []
        all_ptrs: list[list[int]] = []
        all_sizes: list[list[int]] = []
        all_offsets: list[list[int]] = []
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

            all_keys.extend(keys)
            all_ptrs.extend(ptrs)
            all_sizes.extend(sizes)
            all_offsets.extend(offsets)

        lengths = {
            "keys": len(all_keys),
            "ptrs": len(all_ptrs),
            "sizes": len(all_sizes),
            "offsets": len(all_offsets),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"Mooncake page-wise aggregated metadata mismatch: {lengths}."
            )

        # Mooncake's range API is key-major and does not take a pool argument,
        # so differently suffixed physical-pool objects can share one call.
        maybe_fail("aggregated", "complete_page")
        result = self.storage.store.batch_get_into_multi_buffer_ranges(
            all_keys, all_ptrs, all_sizes, all_offsets
        )
        expected = [sum(item) for item in all_sizes]
        if result is None or isinstance(result, int) or list(result) != expected:
            pool_counts = {
                str(name): len(keys) for name, (keys, _) in batches.items()
            }
            raise RuntimeError(
                "Mooncake aggregated range get failed for "
                f"pools={pool_counts}, complete_page: transferred={result}, "
                f"expected={expected}"
            )

        for layer in range(self.num_layers):
            self.layer_done_counter.complete(counter_index, layer)

    def _prepare_read_plan_layouts(
        self, request_transfers: list[tuple[str, list[PoolTransfer]]]
    ):
        # Consolidate index copies once per pool, preserving request/key order.
        # Each component describes (base, row stride, byte count, source offset).
        # Locations may be non-contiguous; Mooncake expands addresses in C++.
        batches = {}
        for _rid, transfers in request_transfers:
            for transfer in transfers:
                keys, indices = batches.setdefault(transfer.name, ([], []))
                component_keys, _ = self.storage._get_hybrid_page_component_keys(
                    list(transfer.keys), transfer
                )
                keys.extend(self.storage._tag_keys(component_keys))
                indices.append(transfer.host_indices)

        layouts = []
        for name, (keys, indices) in batches.items():
            pool = self.pools[name]
            joined = indices[0] if len(indices) == 1 else torch.cat(indices)
            locations = pool.prepare_locations(joined)
            layout = []
            for layer in range(self.num_layers):
                mapped = pool.layer_mapping.get(layer)
                buffer_indices = (
                    ()
                    if mapped is None
                    else (mapped,)
                    if isinstance(mapped, int)
                    else tuple(mapped)
                )
                layout.append(
                    [
                        (*component[buffer_index], offsets[buffer_index])
                        for component, offsets in zip(
                            pool.buffer_meta, pool._component_offsets
                        )
                        for buffer_index in buffer_indices
                    ]
                )
            layouts.append((keys, locations, pool.packed, layout))

        return layouts

    def load_with_read_plan(
        self,
        counter_index: int,
        request_transfers: list[tuple[str, list[PoolTransfer]]],
    ) -> None:
        layouts = self._prepare_read_plan_layouts(request_transfers)
        plan = self.storage.store.create_read_plan(
            layouts,
            self.num_layers,
            reuse_ranges=self.read_plan_reuse_ranges,
            page_wise=self.enable_page_wise_load,
            borrowed_sessions=True,
            buffer_owners=self.pools,
        )
        self.layer_done_counter.bind(counter_index, plan)
        # run() and wait() release the GIL. Each layer becomes visible only after
        # every pool's bytes have been checked; the last wait includes cleanup.
        # In page-wise mode a single batch_get carries all groups per key, so the
        # first wait(0) blocks until every page is complete and later waits are
        # no-ops, matching _load_page_wise's all-or-nothing release.
        plan.run()

    def offload(self, transfers: list[PoolTransfer]) -> bool:
        expanded = self.pool_group.resolve_transfers(transfers, allow_partial=True)
        if not expanded:
            return False
        self.freeze_gc_once()
        if not self.offload_owner:
            self.offload_results.put(True)
            return True
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        tokens = len(kv.keys) * self.page_size
        ready_event = device_module.Event()
        ready_event.record()
        self.offload_queue.put((expanded, tokens, time.perf_counter(), ready_event))
        return True

    def offload_thread_func(self) -> None:
        while True:
            task = self.offload_queue.get()
            metric_recorded = False
            try:
                if task is None:
                    return
                expanded, tokens, started, ready_event = task
                ready_event.synchronize()
                results = self.storage.batch_set_v2(expanded)
                success = all(all(pool_results) for pool_results in results.values())
                self._log_l4_backup_metric(
                    tokens, time.perf_counter() - started, success
                )
                metric_recorded = True
                if success:
                    self.stats["offload"] += 1
                    if self.stats["offload"] == 1:
                        logger.info("Mooncake direct linker offload: tokens=%d", tokens)
                self.offload_results.put(success)
            except BaseException:
                logger.exception("Mooncake offload failed")
                if task is not None and not metric_recorded:
                    _, tokens, started, _ = task
                    self._log_l4_backup_metric(
                        tokens, time.perf_counter() - started, False
                    )
                self.offload_results.put(False)
            finally:
                self.offload_queue.task_done()

    def _log_l4_backup_metric(
        self, tokens: int, duration: float, success: bool
    ) -> None:
        collector = getattr(self, "storage_metrics_collector", None)
        if collector is None:
            return
        try:
            # This branch intentionally does not guess DFS versus local disk.
            # Its Mooncake configuration owns that routing decision.
            collector.log_l4_backup("mooncake", tokens, duration, success)
        except BaseException:
            logger.warning(
                "Failed to record SGLang L4 backup metrics.",
                exc_info=True,
            )

    def num_completed_offloads(self) -> int:
        return self.offload_results.qsize()

    def pop_completed_offload(self) -> bool:
        return self.offload_results.get_nowait()

    def reset(self) -> None:
        with self.host_prefetch_lock:
            for entry in self.host_prefetch_entries.values():
                entry["cancelled"] = True
        if self.host_prefetch_enabled:
            self.host_prefetch_queue.join()
        for rid in list(self.pending_loads):
            self.abort_prepared_load(rid)
        self.pending_loads.clear()
        getattr(self, "pending_load_tokens", {}).clear()
        self.load_queue.join()
        self.offload_queue.join()
        for rid in list(self.prepared_load_sessions):
            self.abort_prepared_load(rid)
        with self.host_prefetch_lock:
            self.host_prefetch_entries.clear()
        while True:
            try:
                self.offload_results.get_nowait()
            except Empty:
                break
        while True:
            try:
                self.completed_loads.get_nowait()
            except Empty:
                break
        self.layer_done_counter.reset()

    def close(self) -> None:
        self.reset()
        self.load_queue.put(None)
        self.offload_queue.put(None)
        for _ in self.host_prefetch_threads:
            self.host_prefetch_queue.put(None)
        self.load_thread.join()
        self.offload_thread.join()
        for thread in self.host_prefetch_threads:
            thread.join()
        logger.info("Mooncake direct linker stats: %s", self.stats)
        self.storage.close()
