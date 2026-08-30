import threading
from array import array
from queue import Queue
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.mem_cache.base_prefix_cache import InsertResult
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.mooncake_store import mooncake_direct_linker
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.full_component import FullComponent
from sglang.srt.mem_cache.unified_cache.components.mamba_component import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_cache.components.swa_component import SWAComponent
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
)
from sglang.srt.mem_cache.unified_cache_linker import (
    DevicePoolEntry,
    DevicePoolGroup,
    ExternalCacheHitMarker,
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Allocator:
    def __init__(self, slots=None):
        self.slots = slots
        self.freed = []
        self.mapping = []

    def available_size(self):
        return 100

    def alloc(self, size):
        if self.slots is None:
            return torch.arange(1, size + 1, dtype=torch.int64)
        value = self.slots[:size].clone()
        self.slots = self.slots[size:]
        return value

    def free(self, value):
        self.freed.append(value.clone())

    def set_full_to_swa_mapping(self, full, swa):
        self.mapping.append((full.clone(), swa.clone()))


def test_sparse_multi_component_layer_ranges():
    k0 = torch.zeros((8, 3), dtype=torch.uint8)
    k2 = torch.zeros((8, 5), dtype=torch.uint8)
    v0 = torch.zeros((8, 7), dtype=torch.uint8)
    v2 = torch.zeros((8, 11), dtype=torch.uint8)
    pool = DevicePoolEntry(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        device_pool=None,
        components=[[k0, k2], [v0, v2]],
        layer_mapping={0: 0, 2: 1},
        page_size=2,
        rows_are_pages=False,
        packed=False,
    )

    indices = torch.tensor([0, 1, 4, 5])
    locations = pool.prepare_locations(indices)
    assert locations == [0, 4]
    pointers, sizes = pool.get_page_buffer_meta(indices)
    assert pointers == [
        buffer[row].data_ptr() for row in locations for buffer in (k0, k2, v0, v2)
    ]
    assert sizes == [6, 10, 14, 22] * 2
    assert pool.get_prepared_layer_range_meta(locations, 1) is None

    pointers, sizes, offsets = pool.get_prepared_layer_range_meta(locations, 2)
    assert pointers == [
        [k2[0].data_ptr()],
        [v2[0].data_ptr()],
        [k2[4].data_ptr()],
        [v2[4].data_ptr()],
    ]
    assert sizes == [[10], [22], [10], [22]]
    assert offsets == [[6], [14], [6], [14]]


def test_lookup_returns_sparse_mamba_boundaries():
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    pools = [
        SimpleNamespace(
            name=name,
            indices_from_pool=name,
            translate_indices=lambda indices: indices,
        )
        for name in (PoolName.KV, PoolName.MAMBA)
    ]
    linker.pool_group = DevicePoolGroup(pools, num_layers=4, page_size=1)
    linker.pools = linker.pool_group.entry_map
    linker.stats = {"lookup": 0}
    linker.storage = MooncakeStore.__new__(MooncakeStore)
    linker.storage.mem_pool_host = SimpleNamespace(kv_buffer=None)
    linker.storage._get_hybrid_page_component_keys = lambda keys, transfer: (
        [f"{key}_{transfer.name}" for key in keys],
        1,
    )
    linker.storage._tag_keys = lambda keys: keys
    linker.storage._batch_exist = lambda keys: [
        int(key.endswith("_kv") or key[0] in ("b", "d")) for key in keys
    ]
    valid = linker.lookup(
        "rid",
        [
            PoolTransfer(name=PoolName.KV, keys=["a", "b", "c", "d"]),
            PoolTransfer(
                name=PoolName.MAMBA,
                keys=["d"],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
        ],
    )
    assert valid == [2, 4]


def test_tail_hashes_honor_radix_key_limit():
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = SimpleNamespace(page_size=256)
    key = RadixKey(array("q", range(256)), limit=255)
    result = SimpleNamespace(last_device_node=None)

    assert wrapper._tail_hashes(key, result, device_hit_len=0) == []


def test_load_and_offload_share_gc_freeze(monkeypatch):
    calls = []
    monkeypatch.setattr(mooncake_direct_linker, "freeze_gc", calls.append)
    monkeypatch.setattr(
        mooncake_direct_linker.device_module,
        "Event",
        lambda: SimpleNamespace(record=lambda: None),
    )

    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    pool = SimpleNamespace(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        translate_indices=lambda indices: indices,
    )
    linker.page_size = 2
    linker.pool_group = DevicePoolGroup([pool], num_layers=1, page_size=2)
    linker.gc_frozen = False
    linker.offload_queue = Queue()
    linker.pending_loads = {"first": [object()]}
    linker.layer_done_counter = SimpleNamespace(update_producer=lambda: 3)
    linker.load_queue = Queue()
    linker.stats = {"load": 0}

    assert linker.offload(
        [
            PoolTransfer(
                name=PoolName.KV,
                keys=["page"],
                device_indices=torch.tensor([0, 1]),
            )
        ]
    )
    assert linker.start_layer_wise_loading() == 3
    linker.pending_loads = {"second": [object()]}
    assert linker.start_layer_wise_loading() == 3

    assert calls == ["Mooncake direct linker"]
    assert linker.stats["load"] == 2


def test_load_waits_for_scheduler_stream(monkeypatch):
    event_calls = []
    loaded = threading.Event()

    class _Event:
        def record(self):
            event_calls.append("record")

        def synchronize(self):
            event_calls.append("synchronize")

    monkeypatch.setattr(mooncake_direct_linker.device_module, "Event", _Event)

    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.gc_frozen = True
    linker.pending_loads = {"rid": [object()]}
    linker.layer_done_counter = SimpleNamespace(update_producer=lambda: 7)
    linker.load_queue = Queue()
    linker.stats = {"load": 0}

    def load_layer_wise(counter_index, transfers):
        assert event_calls == ["record", "synchronize"]
        assert counter_index == 7
        assert len(transfers) == 1
        loaded.set()

    linker.load_layer_wise = load_layer_wise
    thread = threading.Thread(target=linker.load_thread_func, daemon=True)
    thread.start()

    assert linker.start_layer_wise_loading() == 7
    assert loaded.wait(timeout=5)
    linker.load_queue.join()
    linker.load_queue.put(None)
    thread.join(timeout=5)


def test_session_start_negative_result_falls_back_and_logs_key(caplog):
    ended = []

    class _Store:
        def batch_get_session_start(self, keys):
            assert keys == ["page-a", "page-b"]
            return [0, -702]

        def batch_get_session_end(self, keys):
            ended.append(list(keys))

    pool = SimpleNamespace(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        translate_indices=lambda indices: indices,
    )
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.pool_group = DevicePoolGroup([pool], num_layers=1, page_size=1)
    linker.storage = SimpleNamespace(
        store=_Store(),
        _get_hybrid_page_component_keys=lambda keys, transfer: (keys, 1),
        _tag_keys=lambda keys: keys,
    )
    linker.prepared_load_sessions = {}
    linker.session_refcounts = {}
    linker.session_lock = threading.Lock()
    linker.pending_loads = {}
    linker.stats = {"load_fallback": 0}

    transfer = PoolTransfer(
        name=PoolName.KV,
        keys=["page-a", "page-b"],
        device_indices=torch.tensor([1, 2]),
    )
    assert not linker.prepare_load("rid", [transfer])
    assert linker.prepared_load_sessions == {}
    assert linker.session_refcounts == {}
    assert ended == [["page-a"]]
    assert linker.stats["load_fallback"] == 1
    assert "lookup hit but get session start failed" in caplog.text
    assert "page-b" in caplog.text
    assert "-702" in caplog.text


def test_prepare_load_reports_dfs_source():
    ended = []
    calls = []

    class _Store:
        def batch_get_session_start_with_sources(self, keys):
            calls.append("combined")
            return [0] * len(keys), ["local_disk", "dfs"]

        def batch_get_session_start(self, keys):
            pytest.fail("combined session start should be used")

        def batch_get_session_end(self, keys):
            ended.append(list(keys))

    pool = SimpleNamespace(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        translate_indices=lambda indices: indices,
    )
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.pool_group = DevicePoolGroup([pool], num_layers=1, page_size=1)
    linker.storage = SimpleNamespace(
        store=_Store(),
        dfs_replica_num=1,
        _get_hybrid_page_component_keys=lambda keys, transfer: (keys, 1),
        _tag_keys=lambda keys: keys,
    )
    linker.prepared_load_sessions = {}
    linker.prepared_load_sources = {}
    linker.session_refcounts = {}
    linker.session_sources = {}
    linker.session_lock = threading.Lock()
    linker.pending_loads = {}
    linker.stats = {"load_fallback": 0}

    transfer = PoolTransfer(
        name=PoolName.KV,
        keys=["page-a", "page-b"],
        device_indices=torch.tensor([1, 2]),
    )
    assert linker.prepare_load("rid", [transfer])
    assert linker.get_prepared_load_source("rid") == "dfs"
    assert calls == ["combined"]

    linker.abort_prepared_load("rid")
    assert ended == [["page-a", "page-b"]]
    assert linker.session_sources == {}


def test_prepare_load_combined_start_exception_does_not_retry_legacy():
    calls = []

    class _Store:
        def batch_get_session_start_with_sources(self, keys):
            calls.append("combined")
            raise RuntimeError("binding failed after dispatch")

        def batch_get_session_start(self, keys):
            calls.append("legacy")
            return [0] * len(keys)

        def batch_get_session_end(self, keys):
            pytest.fail("no session result was available to clean up")

    pool = SimpleNamespace(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        translate_indices=lambda indices: indices,
    )
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.pool_group = DevicePoolGroup([pool], num_layers=1, page_size=1)
    linker.storage = SimpleNamespace(
        store=_Store(),
        _get_hybrid_page_component_keys=lambda keys, transfer: (keys, 1),
        _tag_keys=lambda keys: keys,
    )
    linker.prepared_load_sessions = {}
    linker.prepared_load_sources = {}
    linker.session_refcounts = {}
    linker.session_sources = {}
    linker.session_lock = threading.Lock()
    linker.pending_loads = {}
    linker.stats = {"load_fallback": 0}

    transfer = PoolTransfer(
        name=PoolName.KV,
        keys=["page-a"],
        device_indices=torch.tensor([1]),
    )
    assert not linker.prepare_load("rid", [transfer])
    assert calls == ["combined"]
    assert linker.prepared_load_sessions == {}
    assert linker.session_refcounts == {}
    assert linker.stats["load_fallback"] == 1


def test_prepare_load_legacy_start_does_not_lookup_source():
    calls = []

    class _Store:
        def batch_get_session_start(self, keys):
            calls.append("legacy")
            return [0] * len(keys)

        def batch_get_session_sources(self, keys):
            pytest.fail("legacy fallback must not issue a source lookup")

        def batch_get_session_end(self, keys):
            pass

    pool = SimpleNamespace(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        translate_indices=lambda indices: indices,
    )
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.pool_group = DevicePoolGroup([pool], num_layers=1, page_size=1)
    linker.storage = SimpleNamespace(
        store=_Store(),
        dfs_replica_num=1,
        _get_hybrid_page_component_keys=lambda keys, transfer: (keys, 1),
        _tag_keys=lambda keys: keys,
    )
    linker.prepared_load_sessions = {}
    linker.prepared_load_sources = {}
    linker.session_refcounts = {}
    linker.session_sources = {}
    linker.session_lock = threading.Lock()
    linker.pending_loads = {}
    linker.stats = {"load_fallback": 0}

    transfer = PoolTransfer(
        name=PoolName.KV,
        keys=["page-a"],
        device_indices=torch.tensor([1]),
    )
    assert linker.prepare_load("rid", [transfer])
    assert calls == ["legacy"]
    assert linker.get_prepared_load_source("rid") is None


def test_prepare_load_source_mismatch_does_not_guess_dfs():
    class _Store:
        def batch_get_session_start_with_sources(self, keys):
            return [0] * len(keys), []

        def batch_get_session_end(self, keys):
            pass

    pool = SimpleNamespace(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        translate_indices=lambda indices: indices,
    )
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.page_size = 16
    linker.pool_group = DevicePoolGroup([pool], num_layers=1, page_size=16)
    linker.storage = SimpleNamespace(
        store=_Store(),
        dfs_replica_num=1,
        _get_hybrid_page_component_keys=lambda keys, transfer: (keys, 1),
        _tag_keys=lambda keys: keys,
    )
    linker.prepared_load_sessions = {}
    linker.prepared_load_sources = {}
    linker.session_refcounts = {}
    linker.session_sources = {}
    linker.session_lock = threading.Lock()
    linker.pending_loads = {}
    linker.stats = {"load_fallback": 0}
    transfer = PoolTransfer(
        name=PoolName.KV,
        keys=["page-a"],
        device_indices=torch.tensor([1]),
    )

    assert linker.prepare_load("rid", [transfer])
    assert linker.get_prepared_load_source("rid") is None
    assert linker.prepared_load_page_sources["rid"] == {(PoolName.KV, "page-a"): None}


def test_successful_prefetch_records_sglang_tokens():
    sglang_metrics = []
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.pending_load_metrics = {
        "rid": (
            128,
            {"local_disk": 128},
            mooncake_direct_linker.time.perf_counter(),
        )
    }
    linker.storage_metrics_collector = SimpleNamespace(
        log_l4_prefetch=lambda source, tokens, duration, success: sglang_metrics.append(
            (source, tokens, success, duration)
        )
    )
    linker.storage = SimpleNamespace(store=SimpleNamespace())

    linker._finish_l4_metric("prefetch", "rid", True)

    assert sglang_metrics[0][:3] == ("local_disk", 128, True)
    assert sglang_metrics[0][3] >= 0


def test_successful_memory_prefetch_records_mooncake_tokens():
    mooncake_tokens = []
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.pending_load_metrics = {
        "rid": (128, {}, mooncake_direct_linker.time.perf_counter())
    }
    linker.storage_metrics_collector = None
    linker.storage = SimpleNamespace(
        store=SimpleNamespace(record_prefetched_tokens=mooncake_tokens.append)
    )

    linker._finish_l4_metric("prefetch", "rid", True)

    assert mooncake_tokens == [128]


@pytest.mark.parametrize(
    ("total_tokens", "source_tokens", "expected_l4"),
    [
        (192, {"dfs": 64}, [("dfs", 64, True)]),
        (
            128,
            {"local_disk": 64, "dfs": 64},
            [("dfs", 64, True), ("local_disk", 64, True)],
        ),
    ],
)
def test_prefetch_metrics_keep_mooncake_total_and_split_actual_sources(
    total_tokens, source_tokens, expected_l4
):
    mooncake_tokens = []
    sglang_metrics = []
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.pending_load_metrics = {
        "rid": (
            total_tokens,
            source_tokens,
            mooncake_direct_linker.time.perf_counter(),
        )
    }
    linker.storage = SimpleNamespace(
        store=SimpleNamespace(record_prefetched_tokens=mooncake_tokens.append)
    )
    linker.storage_metrics_collector = SimpleNamespace(
        log_l4_prefetch=lambda source, tokens, duration, success: sglang_metrics.append(
            (source, tokens, success)
        )
    )

    linker._finish_l4_metric("prefetch", "rid", True)

    assert mooncake_tokens == [total_tokens]
    assert sorted(sglang_metrics) == expected_l4


def test_failed_mixed_source_prefetch_records_each_duration_without_tokens():
    mooncake_tokens = []
    sglang_metrics = []
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.pending_load_metrics = {
        "rid": (
            128,
            {"local_disk": 64, "dfs": 64},
            mooncake_direct_linker.time.perf_counter(),
        )
    }
    linker.storage = SimpleNamespace(
        store=SimpleNamespace(record_prefetched_tokens=mooncake_tokens.append)
    )
    linker.storage_metrics_collector = SimpleNamespace(
        log_l4_prefetch=lambda source, tokens, duration, success: sglang_metrics.append(
            (source, tokens, success)
        )
    )

    linker._finish_l4_metric("prefetch", "rid", False)

    assert mooncake_tokens == []
    assert sorted(sglang_metrics) == [
        ("dfs", 64, False),
        ("local_disk", 64, False),
    ]


def test_prepare_load_attributes_component_sources_once_per_logical_page():
    class _Store:
        def batch_get_session_start_with_sources(self, keys):
            return [0] * len(keys), ["memory", "memory", "local_disk", "dfs"]

        def batch_get_session_end(self, keys):
            pass

    kv_pool = SimpleNamespace(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        translate_indices=lambda indices: indices,
    )
    draft_pool = SimpleNamespace(
        name=PoolName.DRAFT,
        indices_from_pool=PoolName.DRAFT,
        translate_indices=lambda indices: indices,
    )
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.page_size = 16
    linker.pool_group = DevicePoolGroup(
        [kv_pool, draft_pool], num_layers=1, page_size=16
    )
    linker.storage = SimpleNamespace(
        store=_Store(),
        _get_hybrid_page_component_keys=lambda keys, transfer: (
            [
                component
                for key in keys
                for component in (
                    f"{key}-{transfer.name}-a",
                    f"{key}-{transfer.name}-b",
                )
            ],
            2,
        ),
        _tag_keys=lambda keys: keys,
    )
    linker.prepared_load_sessions = {}
    linker.prepared_load_sources = {}
    linker.prepared_load_page_sources = {}
    linker.session_refcounts = {}
    linker.session_sources = {}
    linker.session_lock = threading.Lock()
    linker.pending_loads = {}
    linker.stats = {"load_fallback": 0}
    transfer = PoolTransfer(
        name=PoolName.KV,
        keys=["page-a"],
        device_indices=torch.tensor([1]),
    )
    draft_transfer = PoolTransfer(
        name=PoolName.DRAFT,
        keys=["page-a"],
        device_indices=torch.tensor([1]),
    )

    assert linker.prepare_load("rid", [transfer, draft_transfer])
    assert linker.prepared_load_page_sources["rid"] == {
        (PoolName.KV, "page-a"): "memory",
        (PoolName.DRAFT, "page-a"): "dfs",
    }
    # DFS wins across actual components/pools and the shared logical page is
    # counted once.
    assert linker.load("rid", [transfer, draft_transfer])
    assert linker.pending_load_metrics["rid"][:2] == (16, {"dfs": 16})


def test_load_counts_only_adopted_pages_and_uses_actual_pool_sources():
    class _Store:
        def batch_get_session_start_with_sources(self, keys):
            return [0] * len(keys), ["local_disk", "memory", "dfs", "dfs"]

        def batch_get_session_end(self, keys):
            pass

    kv_pool = SimpleNamespace(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        translate_indices=lambda indices: indices,
    )
    draft_pool = SimpleNamespace(
        name=PoolName.DRAFT,
        indices_from_pool=PoolName.DRAFT,
        translate_indices=lambda indices: indices,
    )
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.page_size = 16
    linker.pool_group = DevicePoolGroup(
        [kv_pool, draft_pool], num_layers=1, page_size=16
    )
    linker.storage = SimpleNamespace(
        store=_Store(),
        _get_hybrid_page_component_keys=lambda keys, transfer: (
            [f"{key}-{transfer.name}" for key in keys],
            1,
        ),
        _tag_keys=lambda keys: keys,
    )
    linker.prepared_load_sessions = {}
    linker.prepared_load_sources = {}
    linker.prepared_load_page_sources = {}
    linker.session_refcounts = {}
    linker.session_sources = {}
    linker.session_lock = threading.Lock()
    linker.pending_loads = {}
    linker.stats = {"load_fallback": 0}
    kv_transfer = PoolTransfer(
        name=PoolName.KV,
        keys=["page-a", "page-b"],
        device_indices=torch.tensor([1, 2]),
    )
    draft_transfer = PoolTransfer(
        name=PoolName.DRAFT,
        keys=["page-a", "page-c"],
        device_indices=torch.tensor([1, 3]),
    )

    # prepare sees three logical pages, but the adopted load only includes KV.
    assert linker.prepare_load("rid", [kv_transfer, draft_transfer])
    assert linker.load("rid", [kv_transfer])
    assert linker.pending_load_metrics["rid"][:2] == (32, {"local_disk": 16})

    linker.cancel_queued_load("rid")
    assert linker.prepare_load("rid-all", [kv_transfer, draft_transfer])
    assert linker.load("rid-all", [kv_transfer, draft_transfer])
    total_tokens, source_tokens, _ = linker.pending_load_metrics["rid-all"]
    # Actual overlapping and distinct pool pages are unioned: a, b, c.
    assert (total_tokens, source_tokens) == (48, {"dfs": 32})
    assert sum(source_tokens.values()) <= total_tokens


def test_cached_tokens_details_exposes_direct_l4_source():
    streamer = SchedulerOutputStreamer.__new__(SchedulerOutputStreamer)
    streamer.enable_hicache_storage = lambda: False
    req = SimpleNamespace(
        cached_tokens_device=16,
        cached_tokens_host=0,
        cached_tokens_storage=32,
        cached_tokens_storage_source="mooncake_dfs",
        cached_tokens=48,
    )

    assert streamer.get_cached_tokens_details(req) == {
        "device": 16,
        "host": 0,
        "storage": 32,
        "storage_backend": "mooncake_dfs",
    }


def test_range_get_negative_result_logs_key(caplog):
    failures = []

    class _Store:
        def batch_get_into_multi_buffer_ranges(self, keys, ptrs, sizes, offsets):
            return [-702]

        def batch_get_session_end(self, keys):
            pass

    pool = SimpleNamespace(
        prepare_locations=lambda indices: [0],
        get_prepared_layer_range_meta=lambda locations, layer: (
            [[123]],
            [[8]],
            [[0]],
        ),
    )
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.num_layers = 1
    linker.page_wise_load_threshold = 10
    linker.page_wise_load_batch_size = 128
    linker.pools = {PoolName.KV: pool}
    linker.storage = SimpleNamespace(
        store=_Store(),
        _get_hybrid_page_component_keys=lambda keys, transfer: (keys, 1),
        _tag_keys=lambda keys: keys,
    )
    linker.layer_done_counter = SimpleNamespace(
        complete=lambda index, layer: None,
        fail=lambda index, error: failures.append(error),
    )
    linker.pending_loads = {}
    linker.prepared_load_sessions = {"rid": ["page-a"]}
    linker.session_refcounts = {"page-a": 1}
    linker.session_lock = threading.Lock()

    linker.load_layer_wise(
        0,
        [
            (
                "rid",
                [
                    PoolTransfer(
                        name=PoolName.KV,
                        keys=["page-a"],
                        host_indices=torch.tensor([0]),
                    )
                ],
            )
        ],
    )

    assert failures
    assert "lookup/session succeeded but range get failed" in caplog.text
    assert "page-a" in caplog.text
    assert "-702" in caplog.text


@pytest.mark.parametrize(
    ("key_count", "threshold", "uses_complete_page_flow"),
    [(9, 10, False), (10, 10, True), (10, 11, False)],
)
def test_large_load_switches_to_complete_page_flow(
    key_count, threshold, uses_complete_page_flow
):
    events = []
    calls = []

    class _Store:
        def batch_get_into_multi_buffer_ranges(self, keys, ptrs, sizes, offsets):
            calls.append((list(keys), ptrs, sizes, offsets))
            events.append(("get", len(keys)))
            return [sum(item) for item in sizes]

        def batch_get_session_end(self, keys):
            events.append(("session_end", len(keys)))

    keys = [f"page-{index}" for index in range(key_count)]
    pool = SimpleNamespace(
        prepare_locations=lambda indices: indices.tolist(),
        get_prepared_layer_range_meta=lambda locations, layer: (
            [[layer * 1000 + location] for location in locations],
            [[layer + 1] for _ in locations],
            [[layer * 10] for _ in locations],
        ),
    )
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.num_layers = 2
    linker.page_wise_load_threshold = threshold
    linker.page_wise_load_batch_size = 128
    linker.pools = {PoolName.KV: pool}
    linker.storage = SimpleNamespace(
        store=_Store(),
        _get_hybrid_page_component_keys=lambda values, transfer: (values, 1),
        _tag_keys=lambda values: values,
    )
    linker.layer_done_counter = SimpleNamespace(
        complete=lambda index, layer: events.append(("complete", layer)),
        fail=lambda index, error: pytest.fail(str(error)),
    )
    linker.pending_loads = {}
    linker.prepared_load_sessions = {"rid": list(keys)}
    linker.session_refcounts = dict.fromkeys(keys, 1)
    linker.session_lock = threading.Lock()

    linker.load_layer_wise(
        0,
        [
            (
                "rid",
                [
                    PoolTransfer(
                        name=PoolName.KV,
                        keys=keys,
                        host_indices=torch.arange(key_count),
                    )
                ],
            )
        ],
    )

    if not uses_complete_page_flow:
        assert len(calls) == 2
        assert calls[0][2][0] == [1]
        assert calls[1][2][0] == [2]
        assert events == [
            ("get", key_count),
            ("complete", 0),
            ("get", key_count),
            ("complete", 1),
            ("session_end", key_count),
        ]
    else:
        assert len(calls) == 1
        assert calls[0][2][0] == [1, 2]
        assert calls[0][3][0] == [0, 10]
        assert events == [
            ("get", 10),
            ("session_end", 10),
            ("complete", 0),
            ("complete", 1),
        ]


def test_complete_page_load_honors_configured_batch_size():
    calls = []

    class _Store:
        def batch_get_into_multi_buffer_ranges(self, keys, ptrs, sizes, offsets):
            calls.append((list(keys), sizes))
            return [sum(item) for item in sizes]

        def batch_get_session_end(self, keys):
            pass

    key_count = 129
    keys = [f"page-{index}" for index in range(key_count)]
    pool = SimpleNamespace(
        prepare_locations=lambda indices: indices.tolist(),
        get_prepared_layer_range_meta=lambda locations, layer: (
            [[location] for location in locations],
            [[8] for _ in locations],
            [[layer * 8] for _ in locations],
        ),
    )
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.num_layers = 3
    linker.page_wise_load_threshold = 10
    linker.page_wise_load_batch_size = 64
    linker.pools = {PoolName.KV: pool}
    linker.storage = SimpleNamespace(
        store=_Store(),
        _get_hybrid_page_component_keys=lambda values, transfer: (values, 1),
        _tag_keys=lambda values: values,
    )
    linker.layer_done_counter = SimpleNamespace(
        complete=lambda index, layer: None,
        fail=lambda index, error: pytest.fail(str(error)),
    )
    linker.pending_loads = {}
    linker.prepared_load_sessions = {"rid": list(keys)}
    linker.session_refcounts = dict.fromkeys(keys, 1)
    linker.session_lock = threading.Lock()

    linker.load_layer_wise(
        0,
        [
            (
                "rid",
                [
                    PoolTransfer(
                        name=PoolName.KV,
                        keys=keys,
                        host_indices=torch.arange(key_count),
                    )
                ],
            )
        ],
    )

    assert [len(call_keys) for call_keys, _ in calls] == [64, 64, 1]
    assert all(sizes == [8, 8, 8] for _, chunk in calls for sizes in chunk)


def test_page_wise_batch_failure_marks_every_request_metric_failed():
    mooncake_tokens = []
    sglang_metrics = []

    class _Store:
        def batch_get_into_multi_buffer_ranges(self, keys, ptrs, sizes, offsets):
            expected = [sum(item) for item in sizes]
            return [expected[0], expected[1] - 1]

        def batch_get_session_end(self, keys):
            pass

        def record_prefetched_tokens(self, tokens):
            mooncake_tokens.append(tokens)

    pool = SimpleNamespace(
        prepare_locations=lambda indices: indices.tolist(),
        get_prepared_layer_range_meta=lambda locations, layer: (
            [[location] for location in locations],
            [[8] for _ in locations],
            [[0] for _ in locations],
        ),
    )
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.num_layers = 1
    linker.page_wise_load_threshold = 1
    linker.page_wise_load_batch_size = 8
    linker.enable_page_wise_load = True
    linker.page_size = 1
    linker.pools = {PoolName.KV: pool}
    linker.storage = SimpleNamespace(
        store=_Store(),
        _get_hybrid_page_component_keys=lambda values, transfer: (values, 1),
        _tag_keys=lambda values: values,
    )
    linker.storage_metrics_collector = SimpleNamespace(
        log_l4_prefetch=lambda source, tokens, duration, success: sglang_metrics.append(
            (source, tokens, success)
        )
    )
    failures = []
    linker.layer_done_counter = SimpleNamespace(
        complete=lambda index, layer: pytest.fail("failed batch must not complete"),
        fail=lambda index, error: failures.append(error),
    )
    linker.pending_loads = {}
    linker.pending_load_metrics = {
        "rid-a": (1, {"dfs": 1}, mooncake_direct_linker.time.perf_counter()),
        "rid-b": (1, {"dfs": 1}, mooncake_direct_linker.time.perf_counter()),
    }
    linker.prepared_load_sessions = {"rid-a": ["page-a"], "rid-b": ["page-b"]}
    linker.session_refcounts = {"page-a": 1, "page-b": 1}
    linker.session_lock = threading.Lock()

    linker.load_layer_wise(
        0,
        [
            (
                "rid-a",
                [
                    PoolTransfer(
                        name=PoolName.KV,
                        keys=["page-a"],
                        host_indices=torch.tensor([0]),
                    )
                ],
            ),
            (
                "rid-b",
                [
                    PoolTransfer(
                        name=PoolName.KV,
                        keys=["page-b"],
                        host_indices=torch.tensor([1]),
                    )
                ],
            ),
        ],
    )

    assert failures
    assert mooncake_tokens == []
    assert sorted(sglang_metrics) == [("dfs", 1, False), ("dfs", 1, False)]


def test_page_wise_attribution_mismatch_fails_the_batch():
    sglang_metrics = []

    class _Store:
        def batch_get_into_multi_buffer_ranges(self, keys, ptrs, sizes, offsets):
            return [sum(item) for item in sizes]

        def batch_get_session_end(self, keys):
            pass

    pool = SimpleNamespace(
        prepare_locations=lambda indices: indices.tolist(),
        get_prepared_layer_range_meta=lambda locations, layer: (
            [[location] for location in locations],
            [[8] for _ in locations],
            [[0] for _ in locations],
        ),
    )
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.num_layers = 1
    linker.page_wise_load_threshold = 1
    linker.page_wise_load_batch_size = 8
    linker.enable_page_wise_load = True
    linker.page_size = 1
    linker.pools = {PoolName.KV: pool}
    tag_call_count = 0

    def tag_keys(keys):
        nonlocal tag_call_count
        tag_call_count += 1
        return keys if tag_call_count == 1 else []

    linker.storage = SimpleNamespace(
        store=_Store(),
        _get_hybrid_page_component_keys=lambda values, transfer: (values, 1),
        _tag_keys=tag_keys,
    )
    linker.storage_metrics_collector = SimpleNamespace(
        log_l4_prefetch=lambda source, tokens, duration, success: sglang_metrics.append(
            (source, tokens, success)
        )
    )
    failures = []
    linker.layer_done_counter = SimpleNamespace(
        complete=lambda index, layer: pytest.fail("mismatched batch must not complete"),
        fail=lambda index, error: failures.append(error),
    )
    linker.pending_loads = {}
    linker.pending_load_metrics = {
        "rid": (1, {"dfs": 1}, mooncake_direct_linker.time.perf_counter())
    }
    linker.prepared_load_sessions = {"rid": ["page-a"]}
    linker.session_refcounts = {"page-a": 1}
    linker.session_lock = threading.Lock()

    linker.load_layer_wise(
        0,
        [
            (
                "rid",
                [
                    PoolTransfer(
                        name=PoolName.KV,
                        keys=["page-a"],
                        host_indices=torch.tensor([0]),
                    )
                ],
            )
        ],
    )

    assert failures
    assert sglang_metrics == [("dfs", 1, False)]


def test_wrapper_peer_prepare_failure_falls_back_before_tree_insert():
    aborted_components = []
    aborted_sessions = []

    class _Component:
        component_type = ComponentType.FULL

        def build_external_linker_transfer(self, phase, node, keys):
            assert phase == LinkerTransferPhase.LOAD
            return PoolTransfer(
                name=PoolName.KV,
                keys=list(keys),
                device_indices=torch.tensor([11, 12]),
            )

        def update_external_linker_load(
            self, phase, req, full, transfer, prefix_len, **kwargs
        ):
            if phase == ExternalLinkerLoadPhase.ABORT:
                aborted_components.append(transfer.device_indices.clone())
                return None
            return transfer

    empty = torch.empty((0,), dtype=torch.int64)
    collectives = []
    cache = SimpleNamespace(
        page_size=1,
        _components_tuple=(_Component(),),
        tree_core=SimpleNamespace(
            empty_match_result=SimpleNamespace(device_indices=empty)
        ),
        _all_reduce_attn_groups=lambda value, op: (
            collectives.append((value.item(), op)),
            value.fill_(0),
        ),
    )
    backend = SimpleNamespace(
        prepare_load=lambda rid, transfers: True,
        get_prepared_load_source=lambda rid: "local_disk",
        abort_prepared_load=aborted_sessions.append,
        load=lambda rid, transfers: pytest.fail("peer failure must not queue a load"),
    )
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = cache
    wrapper.cache_linker = backend
    wrapper.hit_markers = {
        "rid": ExternalCacheHitMarker(
            prefix_key=RadixKey(array("q", [1, 2])),
            tail_hashes=["page-a", "page-b"],
            device_hit_len=0,
        )
    }
    req = SimpleNamespace(
        rid="rid",
        last_node=0,
        prefix_indices=empty,
        host_hit_length=2,
        swa_host_hit_length=2,
        mamba_host_hit_length=0,
        storage_hit_length=2,
        cached_tokens_storage_source="mooncake_local_disk",
    )

    indices, node = wrapper.load_back(req)

    assert indices.numel() == 0
    assert node == 0
    assert aborted_sessions == ["rid"]
    assert [value.tolist() for value in aborted_components] == [[11, 12]]
    assert req.host_hit_length == 0
    assert req.swa_host_hit_length == 0
    assert req.storage_hit_length == 0
    assert req.cached_tokens_storage_source is None
    assert len(collectives) == 1


def test_wrapper_combines_prepare_and_source_in_one_collective():
    collectives = []

    class _Component:
        component_type = ComponentType.FULL

        def build_external_linker_transfer(self, phase, node, keys):
            return PoolTransfer(
                name=PoolName.KV,
                keys=list(keys),
                device_indices=torch.tensor([11, 12]),
            )

        def update_external_linker_load(
            self, phase, req, full, transfer, prefix_len, **kwargs
        ):
            return transfer

    empty = torch.empty((0,), dtype=torch.int64)
    node = SimpleNamespace(id=0, parent=None, external_cache_stored=False)

    def reduce_source(value, op):
        collectives.append((value.item(), op))
        value.fill_(3)  # A peer selected DFS; it wins over local_disk under MIN.

    cache = SimpleNamespace(
        page_size=1,
        _components_tuple=(_Component(),),
        tree_core=SimpleNamespace(
            empty_match_result=SimpleNamespace(device_indices=empty),
            collect_full_device_indices=lambda last, previous: torch.tensor([11, 12]),
        ),
        _all_reduce_attn_groups=reduce_source,
        insert=lambda params: SimpleNamespace(
            mamba_exist=False,
            last_device_node=0,
            adopted_ranges={ComponentType.FULL: [(0, 2)]},
        ),
        resolve_node_handle=lambda value: node,
    )
    backend = SimpleNamespace(
        prepare_load=lambda rid, transfers: True,
        get_prepared_load_source=lambda rid: "local_disk",
        load=lambda rid, transfers: True,
        abort_prepared_load=lambda rid: None,
    )
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = cache
    wrapper.cache_linker = backend
    wrapper.hit_markers = {
        "rid": ExternalCacheHitMarker(
            prefix_key=RadixKey(array("q", [1, 2])),
            tail_hashes=["page-a", "page-b"],
            device_hit_len=0,
        )
    }
    req = SimpleNamespace(
        rid="rid",
        last_node=0,
        prefix_indices=empty,
        host_hit_length=2,
        swa_host_hit_length=0,
        mamba_host_hit_length=0,
        storage_hit_length=0,
        cached_tokens_storage_source=None,
        kv=None,
    )

    wrapper.load_back(req)

    assert len(collectives) == 1
    assert req.cached_tokens_storage_source == "mooncake_dfs"


def test_offload_runs_on_background_thread(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    caller_thread = threading.get_ident()
    worker_threads = []
    event_calls = []

    monkeypatch.setattr(mooncake_direct_linker, "freeze_gc", lambda _: None)

    class _Event:
        def record(self):
            event_calls.append("record")

        def synchronize(self):
            event_calls.append("synchronize")

    monkeypatch.setattr(mooncake_direct_linker.device_module, "Event", _Event)

    class _Storage:
        def batch_set_v2(self, transfers):
            assert event_calls == ["record", "synchronize"]
            worker_threads.append(threading.get_ident())
            started.set()
            assert release.wait(timeout=5)
            return {
                transfer.name: [True] * len(transfer.keys) for transfer in transfers
            }

    pool = SimpleNamespace(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        translate_indices=lambda indices: indices,
        get_hybrid_pool_buffer=lambda: [],
    )
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.page_size = 2
    linker.pool_group = DevicePoolGroup([pool], num_layers=1, page_size=2)
    linker.pools = linker.pool_group.entry_map
    linker.storage = _Storage()
    linker.gc_frozen = False
    linker.stats = {"lookup": 0, "load": 0, "offload": 0}
    linker.offload_queue = Queue()
    linker.offload_results = Queue()
    linker.offload_thread = threading.Thread(
        target=linker.offload_thread_func, daemon=True
    )
    linker.offload_thread.start()

    assert linker.offload(
        [
            PoolTransfer(
                name=PoolName.KV,
                keys=["page"],
                device_indices=torch.tensor([0, 1]),
            )
        ]
    )
    assert started.wait(timeout=5)
    assert linker.num_completed_offloads() == 0
    assert worker_threads == [linker.offload_thread.ident]
    assert worker_threads[0] != caller_thread

    release.set()
    linker.offload_queue.join()
    assert linker.num_completed_offloads() == 1
    assert linker.pop_completed_offload()
    linker.offload_queue.put(None)
    linker.offload_thread.join(timeout=5)


def test_async_offload_pins_node_until_completion():
    class _Component:
        def build_external_linker_transfer(self, phase, node, keys):
            assert phase == LinkerTransferPhase.OFFLOAD
            return PoolTransfer(name=PoolName.KV, keys=["page"])

    results = []
    linker = SimpleNamespace(
        offload=lambda transfers: True,
        num_completed_offloads=lambda: len(results),
        pop_completed_offload=lambda: results.pop(0),
    )
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache_linker = linker
    wrapper.pending_offloads = []
    lock_params = object()
    locks = []
    unlocks = []

    def inc_lock_ref(node):
        locks.append(node)
        return SimpleNamespace(to_dec_params=lambda: lock_params)

    node_id = 7
    node = SimpleNamespace(id=node_id, external_cache_stored=False)
    wrapper.cache = SimpleNamespace(
        _components_tuple=(_Component(),),
        inc_lock_ref=inc_lock_ref,
        dec_lock_ref=lambda node, params: unlocks.append((node, params)),
        resolve_node_handle=lambda value: node if value == node_id else None,
    )

    wrapper.offload_nodes([node_id])
    assert locks == [node_id]
    assert node.external_cache_stored
    assert not unlocks

    results.append(False)
    assert wrapper.num_completed_offloads() == 1
    wrapper.drain_offloads(finish_count=1)
    assert not node.external_cache_stored
    assert unlocks == [(node_id, lock_params)]


def test_check_hicache_events_drains_common_tp_offloads():
    drained = []
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.linker = SimpleNamespace(
        num_completed_offloads=lambda: 3,
        drain_offloads=drained.append,
    )

    def reduce_to_common_count(count, op):
        assert op == torch.distributed.ReduceOp.MIN
        count.fill_(1)

    cache._all_reduce_attn_groups = reduce_to_common_count
    cache.check_hicache_events()

    assert drained == [1]


def test_deepseek_v4_device_pool_group_maps_sparse_sidecars():
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
        DeepSeekV4LayerItem,
        DeepSeekV4TokenToKVPool,
    )

    def state_pool():
        return SimpleNamespace(
            ring_size=2,
            kv_score_buffer=SimpleNamespace(kv_score=torch.zeros((8, 3))),
        )

    kvcache = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
    kvcache._unified_kv = False
    kvcache.start_layer = 0
    kvcache.end_layer = 3
    kvcache.swa_page_size = 2
    kvcache.swa_kv_pool = SimpleNamespace(
        kv_buffer=[torch.zeros((8, 3), dtype=torch.uint8) for _ in range(3)]
    )
    kvcache.c4_kv_pool = SimpleNamespace(
        kv_buffer=[torch.zeros((8, 5), dtype=torch.uint8) for _ in range(2)]
    )
    kvcache.c4_indexer_kv_pool = SimpleNamespace(
        index_k_with_scale_buffer=[
            torch.zeros((8, 7), dtype=torch.uint8) for _ in range(2)
        ]
    )
    kvcache.c128_kv_pool = SimpleNamespace(
        kv_buffer=[torch.zeros((8, 11), dtype=torch.uint8)]
    )
    kvcache.layer_mapping = [
        DeepSeekV4LayerItem(4, 0),
        DeepSeekV4LayerItem(128, 0),
        DeepSeekV4LayerItem(4, 1),
    ]
    kvcache.compress_state_pools = [state_pool(), None, state_pool()]
    kvcache.indexer_compress_state_pools = [state_pool(), None, state_pool()]

    group = resolve_hybrid_device_pool_group(
        kvcache=kvcache,
        page_size=2,
        params=SimpleNamespace(req_to_token_pool=None),
        components={ComponentType.FULL, ComponentType.SWA},
    )
    assert group.num_layers == 3
    assert set(group.entry_map) == {
        PoolName.SWA,
        PoolName.DEEPSEEK_V4_C4,
        PoolName.DEEPSEEK_V4_C4_INDEXER,
        PoolName.DEEPSEEK_V4_C128,
        PoolName.DEEPSEEK_V4_C4_STATE,
        PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
    }
    assert group.sources[PoolName.DEEPSEEK_V4_C4] == PoolName.KV
    assert group.sources[PoolName.DEEPSEEK_V4_C4_STATE] == PoolName.SWA
    c4_pool = group.entry_map[PoolName.DEEPSEEK_V4_C4]
    pointers, sizes = c4_pool.get_page_buffer_meta(torch.tensor([0, 1]))
    assert len(pointers) == 2
    assert sizes == [5, 5]
    _, sizes, offsets = c4_pool.get_prepared_layer_range_meta([0], 2)
    assert sizes == [[5]]
    assert offsets == [[5]]
    assert c4_pool.get_prepared_layer_range_meta([0], 1) is None


def test_mamba_strategy_rejects_direct_linker():
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    kvcache = HybridLinearKVPool.__new__(HybridLinearKVPool)
    kvcache.use_mla = False
    kvcache.full_attention_layer_id_mapping = {0: 0, 2: 1}
    kvcache.full_kv_pool = SimpleNamespace(
        size=6,
        k_scale_buffer=None,
        k_buffer=[torch.zeros((8, 3)), torch.zeros((8, 5))],
        v_buffer=[torch.zeros((8, 7)), torch.zeros((8, 11))],
    )
    req_pool = SimpleNamespace(
        mamba_ckpt_pool=None,
        mamba_map={1: 0, 3: 1},
        mamba_pool=SimpleNamespace(
            mamba_cache=SimpleNamespace(
                temporal=torch.zeros((2, 5, 2, 3)),
                conv=[torch.zeros((2, 5, 4))],
            )
        ),
        translate_mamba_indices=lambda indices: indices,
    )

    with pytest.raises(ValueError, match="does not support the direct external linker"):
        resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=2,
            params=SimpleNamespace(req_to_token_pool=req_pool),
            components={ComponentType.FULL, ComponentType.MAMBA},
        )


def test_dsa_device_pool_group_uses_assembler_strategy():
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

    kvcache = DSATokenToKVPool.__new__(DSATokenToKVPool)
    kvcache.page_size = 2
    kvcache.layer_num = 2
    kvcache.kv_buffer = [
        torch.zeros((8, 3), dtype=torch.uint8),
        torch.zeros((8, 5), dtype=torch.uint8),
    ]
    kvcache.index_key_cache = SimpleNamespace(
        buffer=[
            torch.zeros((4, 7), dtype=torch.uint8),
            torch.zeros((4, 11), dtype=torch.uint8),
        ]
    )

    group = resolve_hybrid_device_pool_group(
        kvcache=kvcache,
        page_size=2,
        params=SimpleNamespace(req_to_token_pool=None),
        components={ComponentType.FULL},
    )

    assert group.num_layers == 2
    assert set(group.entry_map) == {PoolName.KV, PoolName.INDEXER}
    assert group.sources == {
        PoolName.KV: PoolName.KV,
        PoolName.INDEXER: PoolName.KV,
    }


def test_device_pool_group_allows_partial_side_pool_load():
    swa_pool = SimpleNamespace(
        name=PoolName.SWA,
        indices_from_pool=PoolName.SWA,
        translate_indices=lambda indices: indices + 100,
    )
    group = DevicePoolGroup([swa_pool], num_layers=1, page_size=2)
    transfer = PoolTransfer(
        name=PoolName.SWA,
        keys=["b", "d"],
        device_indices=torch.tensor([20, 21, 24, 25]),
        hit_policy=PoolHitPolicy.TRAILING_PAGES,
    )

    assert group.resolve_transfers([transfer]) == []
    [resolved] = group.resolve_transfers(
        [transfer], allow_partial=True, allow_missing_kv=True
    )
    assert resolved.name == PoolName.SWA
    assert resolved.keys == ["b", "d"]
    assert resolved.host_indices.tolist() == [120, 121, 124, 125]


def test_swa_linker_load_prepare_commit_and_abort():
    swa_allocator = _Allocator()
    allocator = SimpleNamespace(
        swa_attn_allocator=swa_allocator,
        set_full_to_swa_mapping=swa_allocator.set_full_to_swa_mapping,
    )
    component = SWAComponent.__new__(SWAComponent)
    component.cache = SimpleNamespace(
        page_size=1,
        token_to_kv_pool_allocator=allocator,
    )
    component.sliding_window_size = 2
    req = SimpleNamespace(kv=SimpleNamespace(swa_evicted_seqlen=0))
    full = PoolTransfer(name=PoolName.KV, device_indices=torch.tensor([1, 2, 3, 4]))
    swa = PoolTransfer(name=PoolName.SWA, device_indices=torch.tensor([20, 21]))

    component.update_external_linker_load(
        ExternalLinkerLoadPhase.PREPARE, req, full, swa, prefix_len=4
    )
    mapped_full, mapped_swa = swa_allocator.mapping[0]
    assert mapped_full.tolist() == [3, 4]
    assert mapped_swa.tolist() == [20, 21]
    assert req.kv.swa_evicted_seqlen == 2

    insert_result = InsertResult(
        prefix_len=2,
        adopted_ranges={ComponentType.SWA: [(2, 4)]},
    )
    canonical = torch.tensor([10, 11, 12, 13])
    committed = component.update_external_linker_load(
        ExternalLinkerLoadPhase.COMMIT,
        req,
        full,
        swa,
        prefix_len=4,
        insert_result=insert_result,
        canonical_full=canonical[2:],
    )
    assert committed is swa
    mapped_full, mapped_swa = swa_allocator.mapping[1]
    assert mapped_full.tolist() == [12, 13]
    assert mapped_swa.tolist() == [20, 21]

    aborted = PoolTransfer(name=PoolName.SWA, device_indices=torch.tensor([30, 31]))
    component.update_external_linker_load(
        ExternalLinkerLoadPhase.ABORT, req, full, aborted, prefix_len=4
    )
    assert swa_allocator.freed[0].tolist() == [30, 31]


def test_mamba_component_rejects_external_linker():
    component = MambaComponent.__new__(MambaComponent)
    with pytest.raises(AssertionError, match="does not support external linker mode"):
        component.build_external_linker_transfer(
            LinkerTransferPhase.LOAD, None, ["a", "b"]
        )


def test_component_commit_filters_overlapping_full_and_swa_load_pages():
    mapping = _Allocator()
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = SimpleNamespace(
        page_size=2,
        token_to_kv_pool_allocator=SimpleNamespace(
            set_full_to_swa_mapping=mapping.set_full_to_swa_mapping
        ),
    )
    full_component = FullComponent.__new__(FullComponent)
    full_component.cache = wrapper.cache
    full_component.component_type = ComponentType.FULL
    swa_component = SWAComponent.__new__(SWAComponent)
    swa_component.cache = wrapper.cache
    swa_component.component_type = ComponentType.SWA
    full = PoolTransfer(
        name=PoolName.KV,
        keys=["a", "b", "c", "d"],
        device_indices=torch.tensor([100, 101, 102, 103, 104, 105, 106, 107]),
    )
    canonical_tail = torch.tensor([10, 11, 102, 103, 14, 15, 106, 107])
    swa = PoolTransfer(
        name=PoolName.SWA,
        keys=["a", "b", "c", "d"],
        device_indices=torch.tensor([200, 201, 202, 203, 204, 205, 206, 207]),
    )
    insert_result = InsertResult(
        prefix_len=0,
        adopted_ranges={
            ComponentType.FULL: [(2, 4), (6, 8)],
            ComponentType.SWA: [(2, 4), (6, 8)],
        },
    )
    filtered = wrapper._update_load(
        ExternalLinkerLoadPhase.COMMIT,
        SimpleNamespace(),
        [(full_component, full), (swa_component, swa)],
        prefix_len=8,
        insert_result=insert_result,
        canonical_full=canonical_tail,
    )
    assert filtered == [full, swa]
    assert full.keys == ["b", "d"]
    assert full.device_indices.tolist() == [102, 103, 106, 107]
    assert swa.keys == ["b", "d"]
    assert swa.device_indices.tolist() == [202, 203, 206, 207]
    mapped_full, mapped_swa = mapping.mapping[0]
    assert mapped_full.tolist() == [102, 103, 106, 107]
    assert mapped_swa.tolist() == [202, 203, 206, 207]
