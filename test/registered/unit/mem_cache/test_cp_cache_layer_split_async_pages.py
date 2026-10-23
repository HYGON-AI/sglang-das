"""CPU dependency and remap regressions for static LayerSplit page prefetch.

Run the real pool/staging/slot logic with CPU tensors; only accelerator stream
and collective boundaries are emulated. These tests do not prove GPU overlap.
"""

import ast
import unittest
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Iterator, Optional

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _source_file(name):
    relative = Path("python/sglang/srt/mem_cache/cp_cache_layer_split") / name
    for parent in Path(__file__).resolve().parents:
        for root in (parent, parent / "stage", parent / "source"):
            candidate = root / relative
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(relative)


def _load_nodes(name, names, namespace, *, methods=False):
    path = _source_file(name)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = tree.body
    if methods:
        nodes = next(
            node.body for node in nodes
            if isinstance(node, ast.ClassDef)
            and node.name == "CpCacheLayerSplitDeepSeekV4TokenToKVPool"
        )
    chosen = [node for node in nodes if getattr(node, "name", None) in names]
    if {node.name for node in chosen} != set(names):
        raise AssertionError(f"Missing production definitions in {path}")
    for node in chosen:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=chosen, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


class _Device:
    def __init__(self, events):
        self.events = events
        self.main = self._new_stream("main")
        self.current = self.main
        self.next_id = 0

    def _new_stream(self, name):
        return SimpleNamespace(
            name=name, device=torch.device("cpu"),
            wait_stream=lambda other: self.events.append(
                ("wait", name, other.name)
            ),
        )

    def Stream(self, device):
        self.next_id += 1
        return self._new_stream(f"side{self.next_id}")

    def current_stream(self, device=None):
        return self.current

    @contextmanager
    def stream(self, stream):
        old, self.current = self.current, stream
        self.events.append(("enter", stream.name))
        try:
            yield
        finally:
            self.events.append(("exit", stream.name))
            self.current = old


class _TrackedTensor:
    def __init__(self, value, harness):
        self.value, self.harness = value, harness

    @property
    def shape(self):
        return self.value.shape

    @property
    def device(self):
        return self.value.device

    def __getitem__(self, key):
        return _TrackedTensor(self.value[key], self.harness)

    def copy_(self, value):
        self.harness.record("copy")
        self.value.copy_(value)
        return self


class _Harness:
    def __init__(self, *, enabled=True, graph=False, rank=0, peer_pages=()):
        self.events = []
        self.device = _Device(self.events)
        self.peer_pages = peer_pages
        self.captured = graph
        self.reference = torch.arange(8 * 2 * 3).reshape(8, 2, 3).float()
        self.staging = _TrackedTensor(torch.full_like(self.reference, -99), self)
        self.namespace = {
            "__name__": __name__, "Optional": Optional, "Iterable": Iterable,
            "Iterator": Iterator, "dataclass": dataclass,
            "nullcontext": nullcontext, "contextmanager": contextmanager,
            "NopeFp8RopeBf16Pack": object,
            "envs": SimpleNamespace(
                SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=SimpleNamespace(get=lambda: True)
            ),
            "_is_hcu": False,
        }
        harness = self

        class TorchProxy:
            cuda = SimpleNamespace(
                Stream=object, is_current_stream_capturing=lambda: graph
            )

            def __getattr__(self, name):
                if name == "get_device_module":
                    return lambda device: harness.device
                if name == "index_select":

                    def gather(source, dim, index, *, out):
                        harness.record("gather_out")
                        torch.index_select(source, dim, index, out=out.value)
                        return out

                    return gather
                if name in ("nonzero", "nonzero_static"):
                    def select(*args, **kwargs):
                        harness.record(name)
                        return getattr(torch, name)(*args, **kwargs)
                    return select
                return getattr(torch, name)

        self.namespace["torch"] = TorchProxy()
        _load_nodes("broadcast.py", ["PendingBroadcast", "BroadcastSlots"], self.namespace)
        slots_type = self.namespace["BroadcastSlots"]

        class Slots(slots_type):
            def launch(self, *args, **kwargs):
                harness.record("launch")
                return super().launch(*args, **kwargs)

            def clear(self, *args, **kwargs):
                harness.record("clear")
                return super().clear(*args, **kwargs)

            def start(self, *args, **kwargs):
                harness.record("start")
                return super().start(*args, **kwargs)

        @contextmanager
        def change_state(**kwargs):
            yield

        def all_reduce(mask):
            self.record("all_reduce")
            for page in self.peer_pages:
                mask[page] += 1

        def broadcast(value, owner):
            self.record("broadcast")
            self.events.append(("owner", owner))
            if rank != owner:
                swa = self.pool._broadcast_slots.pending("swa")
                layer = self.pool._broadcast_slots.pending("extra").layer_id
                family = "swa" if swa.active else self.pool._extra_family_name(
                    self.pool.compression_ratios[layer]
                )
                selected = self.pool._batch_active_pages[family].selected_pages
                if selected.numel():
                    value.value.copy_(self.reference[selected.clamp_min(0).long()])
                else:
                    value.value.copy_(self.reference[:1])

        comm = SimpleNamespace(change_state=change_state, all_reduce=all_reduce,
                               broadcast=broadcast)
        self.namespace["get_pynccl_broadcast_comm"] = lambda: comm
        _load_nodes("staging.py", ["build_active_pages_mask", "all_reduce_active_pages_mask",
                    "active_pages_for_indices", "contiguous_request_page_capacity",
                    "remap_indices_to_staging"], self.namespace)
        for name, event in (("build_active_pages_mask", "build"),
                            ("remap_indices_to_staging", "remap")):
            function = self.namespace[name]

            def tracked(*args, _function=function, _event=event, **kwargs):
                self.record(_event)
                return _function(*args, **kwargs)

            self.namespace[name] = tracked
        methods = ["_pool_num_pages", "_active_pages_for_indices",
                   "_compact_broadcast_for_read", "_extra_family_name",
                   "_clear_extra_read_state", "prefetch_low_ratio_extra_layer",
                   "prefetch_swa_layer", "reset_batch_active_pages",
                   "_clear_swa_read_state", "_clear_indexer_read_state",
                   "clear_staging_remap_for_read", "prepare_forward",
                   "get_extra_key_buffer_for_write", "set_extra_key_buffer",
                   "set_extra_key_buffer_fused"]
        _load_nodes("deepseek_v4_pool.py", methods, self.namespace, methods=True)
        cls = type("PoolUnderTest", (), {name: self.namespace[name] for name in methods})
        cls._pool_num_pages = staticmethod(self.namespace["_pool_num_pages"])
        self.namespace["_USE_STATIC_CP_LAYER_SPLIT_PAGES"] = enabled
        self.pool = cls()
        self.pool.compression_ratios = {8: 1, 9: 2}
        self.pool.cp_rank = rank
        self.pool.cp_size = 2
        self.pool._extra_full_read_key = None
        self.pool.request_window = None
        self.pool._broadcast_slots = Slots(("extra", "swa"))
        kv_pool = SimpleNamespace(page_size=2, size=13,
                                 kv_buffer=[self.reference + 999, self.reference])
        self.pool.kv_pools = {1: kv_pool, 2: kv_pool}
        self.pool.swa_kv_pool = kv_pool
        self.pool.sliding_window = 128
        self.pool._swa_local_layer_id = lambda layer: 1
        self.pool._low_owner_local_by_source = {4: 1, 6: 1}
        self.pool.source_layer_of = lambda layer: {8: 4, 9: 6}[layer]
        self.pool._get_layer_owner_rank = lambda source: {4: 0, 6: 1}.get(source, 0)
        self.pool._is_layer_owned = lambda source: rank == {4: 0, 6: 1}[source]
        self.pool._get_staging_buffer = lambda family: self.staging
        self.pool.wait_layer_transfer = lambda layer: self.events.append(("transfer", layer))
        self.pool._batch_active_pages = {
            name: SimpleNamespace(selected_pages=None, static_selected_pages=False)
            for name in ("extra_c1", "extra_c2", "swa")
        }

    def record(self, name):
        self.events.append((name, self.device.current.name))

    def fetch(self, layer=8, *, indices=None, read_indices=None, num_reqs=1):
        if indices is None:
            indices = torch.tensor([2, 3, 8, 9, 12, -1], dtype=torch.int32)
        if read_indices is None:
            read_indices = indices
        self.pool.prefetch_low_ratio_extra_layer(
            layer, indices, read_indices=read_indices, num_reqs=num_reqs
        )

    def fetch_tail(self, *, layer=8, indices=None, seq_lens=(13,), tail=True):
        if indices is None:
            indices = torch.tensor([2, 3, 8, 9, -1], dtype=torch.int32)
        core = SimpleNamespace(sparse_page_indices=lambda ratio: indices)
        metadata = SimpleNamespace(
            sparse_prefill_cache=None, late_layer_tail=object() if tail else None,
            core_attn_metadata=core,
        )
        self.namespace["get_attn_backend"] = lambda: SimpleNamespace(
            forward_metadata=metadata
        )
        self.namespace["_should_sync_cp_cache_layer_split"] = lambda *args: True
        _load_nodes("deepseek_v4_helpers.py",
                    ["_get_core_attn_metadata", "maybe_prefetch_cp_kv_extra"],
                    self.namespace)
        batch = SimpleNamespace(seq_lens_cpu=torch.tensor(seq_lens))
        self.namespace["maybe_prefetch_cp_kv_extra"](self.pool, layer, batch)
        return metadata

    def fetch_swa(self, *, layer=0, indices=None, seq_lens=(13,),
                  extend_lens=(8,), tail_lens=None, cache=None):
        if indices is None:
            indices = torch.tensor([2, 3, 8, 9, -1], dtype=torch.int32)
        tail = (
            SimpleNamespace(extend_seq_lens_cpu=tail_lens, local_lens_cpu=[0])
            if tail_lens is not None else None
        )
        metadata = SimpleNamespace(
            sparse_prefill_cache=cache, late_layer_tail=tail,
            core_attn_metadata=SimpleNamespace(swa_page_indices=indices),
        )
        self.namespace["get_attn_backend"] = lambda: SimpleNamespace(
            forward_metadata=metadata
        )
        self.namespace["_should_sync_cp_cache_layer_split"] = lambda *args: True
        _load_nodes("deepseek_v4_helpers.py",
                    ["_get_core_attn_metadata", "maybe_prefetch_cp_kv_swa"],
                    self.namespace)
        batch = SimpleNamespace(seq_lens_cpu=torch.tensor(seq_lens),
                                extend_seq_lens_cpu=list(extend_lens))
        self.namespace["maybe_prefetch_cp_kv_swa"](self.pool, layer, batch)
        return metadata


class TestLayerSplitAsyncPages(CustomTestCase):
    def _cache_source(self, harness, *, cache_full_read=True):
        pool = harness.pool
        pool.compression_ratios[4] = 1
        pool.source_layer_of = lambda layer: {4: 4, 8: 4, 9: 6}[layer]
        pool.prefetch_low_ratio_extra_layer(
            4,
            torch.tensor([2, 3, -1], dtype=torch.int32),
            read_indices=torch.tensor([2, 3, 8, 9, 12, -1], dtype=torch.int32),
            num_reqs=1,
            cache_full_read=cache_full_read,
        )

    def test_full_source_payload_reuses_pages_not_in_source_topk(self):
        for rank in (0, 1):
            for empty_consumer in (False, True):
                for consume_source in (False, True):
                    with self.subTest(rank=rank, empty=empty_consumer,
                                      consume_source=consume_source):
                        harness = _Harness(rank=rank, peer_pages=(6,))
                        self._cache_source(harness)
                        pool = harness.pool
                        self.assertEqual(pool._extra_full_read_key, (1, 4))
                        selected = pool._batch_active_pages["extra_c1"].selected_pages
                        storage = harness.staging.value.clone()
                        if consume_source:
                            pool._broadcast_slots.finish("extra", 4)
                        # Bounded replay rebuilds query plans at the layer21
                        # boundary, but the source KV and its full page set stay.
                        pool._batch_active_pages["swa"].selected_pages = torch.tensor([0])
                        pool.reset_batch_active_pages()
                        self.assertIsNone(pool._batch_active_pages["swa"].selected_pages)
                        self.assertIs(pool._batch_active_pages["extra_c1"].selected_pages, selected)
                        self.assertEqual(pool._extra_full_read_key, (1, 4))
                        # SWA prefetch resets layer-local Extra read state only.
                        pool._clear_extra_read_state()
                        indices = torch.empty(0, dtype=torch.int32) if empty_consumer else torch.tensor(
                            [8, 12, -1], dtype=torch.int32
                        )
                        pool.prefetch_low_ratio_extra_layer(
                            8, indices, num_reqs=1, read_token_capacity=13
                        )
                        expected = torch.empty(0, dtype=torch.int32) if empty_consumer else torch.tensor(
                            [2, 4, -1], dtype=torch.int32
                        )
                        torch.testing.assert_close(pool._extra_remapped_indices, expected)
                        torch.testing.assert_close(harness.staging.value, storage)
                        self.assertIs(pool._batch_active_pages["extra_c1"].selected_pages, selected)
                        self.assertEqual(pool._extra_staging_layer_id, 8)
                        self.assertEqual(pool._extra_remapped_layer_id, 8)
                        self.assertFalse(pool._broadcast_slots.pending("extra").active)
                        for operation in ("build", "all_reduce", "broadcast"):
                            self.assertEqual(sum(e[0] == operation for e in harness.events), 1)
                        self.assertIn(("wait", "main", "side1"), harness.events)

    def test_full_payload_is_invalidated_by_new_forward_or_shared_extra_write(self):
        for rank in (0, 1):
            for invalidate in ("forward", "other_ratio", "write_view", "write", "fused_write"):
                with self.subTest(rank=rank, invalidate=invalidate):
                    harness = _Harness(rank=rank, peer_pages=(6,))
                    self._cache_source(harness)
                    pool = harness.pool
                    if invalidate == "forward":
                        pool.prepare_forward(require_prefetched_reads=True)
                    elif invalidate == "other_ratio":
                        harness.fetch(layer=9)
                    else:
                        pool._owns_extra_key_layer_id = lambda layer: False
                        if invalidate == "write_view":
                            pool.get_extra_key_buffer_for_write(4)
                        elif invalidate == "write":
                            pool.set_extra_key_buffer(4, None, None)
                        else:
                            pool.set_extra_key_buffer_fused(4, None, None)
                    self.assertIsNone(pool._extra_full_read_key)
                    before = sum(e[0] == "broadcast" for e in harness.events)
                    harness.fetch(layer=8)
                    self.assertEqual(sum(e[0] == "broadcast" for e in harness.events), before + 1)
                    torch.testing.assert_close(
                        pool._extra_remapped_indices,
                        torch.tensor([0, 1, 2, 3, 4, -1], dtype=torch.int32),
                    )

    def test_sparse_subsets_dynamic_and_capture_do_not_cache_payload(self):
        for options in ({"enabled": False}, {"graph": True}, {"full": False}):
            with self.subTest(options=options):
                harness = _Harness(enabled=options.get("enabled", True),
                                   graph=options.get("graph", False))
                self._cache_source(harness, cache_full_read=options.get("full", True))
                self.assertIsNone(harness.pool._extra_full_read_key)
                harness.fetch()
                self.assertEqual(sum(e[0] == "broadcast" for e in harness.events), 2)

    def test_new_source_with_same_ratio_replaces_cached_payload(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                harness = _Harness(rank=rank)
                self._cache_source(harness)
                pool = harness.pool
                pool.compression_ratios.update({6: 1, 9: 1})
                pool.source_layer_of = lambda layer: {4: 4, 6: 6, 8: 4, 9: 6}[layer]
                harness.reference = harness.reference + 100
                pool.kv_pools[1].kv_buffer[1] = harness.reference
                indices = torch.tensor([2, 3, 8, 9, 12, -1], dtype=torch.int32)
                pool.prefetch_low_ratio_extra_layer(
                    6, indices[:2], read_indices=indices, num_reqs=1, cache_full_read=True
                )
                self.assertEqual(pool._extra_full_read_key, (1, 6))
                pool.prefetch_low_ratio_extra_layer(
                    9, indices, num_reqs=1, read_token_capacity=13
                )
                self.assertEqual(sum(e[0] == "broadcast" for e in harness.events), 2)
                torch.testing.assert_close(harness.staging.value[:3], harness.reference[[1, 4, 6]])
                torch.testing.assert_close(
                    pool._extra_remapped_indices,
                    torch.tensor([0, 1, 2, 3, 4, -1], dtype=torch.int32),
                )

    def test_helper_only_marks_uniform_full_history_source_reads(self):
        for sparse, extends in ((True, [16]), (True, [1]), (True, []), (False, [16])):
            with self.subTest(sparse=sparse, extends=extends):
                harness = _Harness()
                full_ids = torch.tensor([2, 3, 8, 9, 12, -1], dtype=torch.int32)
                core = SimpleNamespace(page_table=torch.zeros((1, 8)),
                                       sparse_page_indices=lambda ratio: full_ids[:2])
                metadata = SimpleNamespace(
                    sparse_prefill_cache=SimpleNamespace(
                        num_reqs=1,
                        ensure_compressed=lambda *args: SimpleNamespace(flat_token_ids=full_ids),
                    ),
                    core_attn_metadata=core,
                )
                harness.namespace["get_attn_backend"] = lambda: SimpleNamespace(forward_metadata=metadata)
                harness.namespace["_should_sync_cp_cache_layer_split"] = lambda *args: True
                harness.namespace["envs"].SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get = lambda: sparse
                _load_nodes("deepseek_v4_helpers.py",
                            ["_get_core_attn_metadata", "maybe_prefetch_cp_kv_extra"], harness.namespace)
                captured = {}
                harness.pool.prefetch_low_ratio_extra_layer = lambda *args, **kwargs: captured.update(kwargs)
                harness.namespace["maybe_prefetch_cp_kv_extra"](
                    harness.pool, 8, SimpleNamespace(extend_seq_lens_cpu=extends)
                )
                self.assertIs(captured["read_indices"], full_ids)
                self.assertEqual(captured["cache_full_read"], sparse and bool(extends) and min(extends) >= 2)

    def test_compact_pack_writes_existing_staging_without_payload_copy(self):
        for dtype in (torch.uint8, torch.bfloat16, torch.float32):
            for shape in ((8, 31), (8, 2, 3)):
                for static in (False, True):
                    with self.subTest(dtype=dtype, shape=shape, static=static):
                        harness = _Harness()
                        source = (
                            torch.arange(torch.tensor(shape).prod())
                            .reshape(shape)
                            .to(dtype)
                        )
                        storage = torch.full_like(source, 99)
                        staging = _TrackedTensor(storage, harness)
                        pointer = storage.data_ptr()
                        pages = torch.tensor(
                            [6, 1, 6, -1 if static else 0], dtype=torch.int32
                        )
                        indices = torch.tensor([2, 3, -1], dtype=torch.int32)
                        result = harness.pool._compact_broadcast_for_read(
                            8,
                            indices,
                            source,
                            staging,
                            pages,
                            page_size=2,
                            max_pages=8,
                            broadcast_kind="extra",
                            remap_indices=False,
                            selected_pages_static=static,
                        )
                        self.assertIs(result, indices)
                        self.assertEqual(storage.data_ptr(), pointer)
                        torch.testing.assert_close(
                            storage[:4], source[pages.clamp_min(0).long()]
                        )
                        torch.testing.assert_close(
                            storage[4:], torch.full_like(storage[4:], 99)
                        )
                        self.assertEqual(
                            harness.events.count(("gather_out", "main")), 1
                        )
                        self.assertFalse(
                            any(event[0] == "copy" for event in harness.events)
                        )
                        self.assertEqual(
                            harness.events.count(("broadcast", "side1")), 1
                        )

    def test_swa_first_layer_and_replay_tail_use_static_union_without_chunk_cache(self):
        for layer, tail_lens in ((0, None), (21, (8,))):
            for rank in (0, 1):
                for empty_rank in (False, True):
                    with self.subTest(layer=layer, rank=rank, empty_rank=empty_rank):
                        harness = _Harness(rank=rank, peer_pages=(1, 4, 6))
                        indices = torch.empty(0, dtype=torch.int32) if empty_rank else None
                        metadata = harness.fetch_swa(
                            layer=layer, tail_lens=tail_lens, indices=indices,
                        )
                        selected = harness.pool._batch_active_pages["swa"].selected_pages
                        torch.testing.assert_close(selected[:3], torch.tensor([1, 4, 6]))
                        self.assertTrue(bool((selected[3:] == -1).all()))
                        self.assertIsNone(metadata.sparse_prefill_cache)
                        self.assertIn(("nonzero_static", "main"), harness.events)
                        self.assertFalse(any(e[0] == "nonzero" for e in harness.events))
                        expected = (
                            torch.empty(0, dtype=torch.int32) if empty_rank else
                            torch.tensor([0, 1, 2, 3, -1], dtype=torch.int32)
                        )
                        torch.testing.assert_close(harness.pool._swa_remapped_indices, expected)
                        torch.testing.assert_close(
                            harness.staging.value[:3], harness.reference[[1, 4, 6]]
                        )
                        self.assertEqual(harness.events.count(("all_reduce", "main")), 1)
                        self.assertEqual(harness.events.count(("broadcast", "side1")), 1)
                        harness.pool._broadcast_slots.finish("swa", layer)
                        self.assertEqual(harness.events[-1], ("wait", "main", "side1"))

    def test_swa_capacity_uses_global_chunk_history_or_floored_tail(self):
        cases = (
            ((), ()), ((0,), (0,)), ((4,), (4,)), ((129,), (1,)),
            ((32768,), (32768,)), ((1048576,), (32768,)),
            ((1, 129, 65536, 1048576), (1, 1, 32768, 7)),
        )
        for lengths, extends in cases:
            for tail in (False, True):
                with self.subTest(lengths=lengths, tail=tail):
                    harness = _Harness()
                    captured = {}
                    harness.pool.prefetch_swa_layer = (
                        lambda *args, **kwargs: captured.update(kwargs)
                    )
                    tail_lens = tuple(min(n, 128) for n in extends) if tail else None
                    harness.fetch_swa(seq_lens=lengths, extend_lens=extends,
                                      tail_lens=tail_lens)
                    spans = tail_lens if tail else tuple(
                        min(seq, ext + 127) for seq, ext in zip(lengths, extends)
                    )
                    self.assertEqual(captured["read_token_capacity"], sum(spans))
                    self.assertEqual(captured["num_reqs"], len(lengths))
                    self.assertIsNone(captured["read_indices"])
                    page_size = 256
                    # Each request starts at the worst offset. Physical pages
                    # can be fragmented; CP peers see arbitrary subsets.
                    pages = sum((n + 2 * page_size - 2) // page_size
                                for n in spans if n > 0)
                    max_pages = pages + 31
                    owners = torch.randperm(
                        max_pages, generator=torch.Generator().manual_seed(42)
                    )[:pages]
                    static_size = harness.namespace["contiguous_request_page_capacity"](
                        sum(spans), len(lengths), page_size, max_pages
                    )
                    self.assertGreaterEqual(static_size, pages)
                    for rank in range(8):
                        harness.peer_pages = owners.tolist()
                        selected = harness.namespace["active_pages_for_indices"](
                            owners[rank::8].to(torch.int32) * page_size,
                            page_size, max_pages,
                            harness.namespace["get_pynccl_broadcast_comm"](), static_size,
                        )
                        torch.testing.assert_close(selected[:pages], owners.sort().values)
                        self.assertTrue(bool((selected[pages:] == -1).all()))

    def test_swa_cache_span_keeps_priority_over_capacity(self):
        harness = _Harness(peer_pages=(6,))
        cache = SimpleNamespace(swa_token_ids=torch.tensor([2, 3, 8, 9, 12, -1]),
                                num_reqs=1)
        harness.fetch_swa(cache=cache)
        self.assertIn(("nonzero_static", "main"), harness.events)
        selected = harness.pool._batch_active_pages["swa"].selected_pages
        torch.testing.assert_close(selected[:3], torch.tensor([1, 4, 6]))
        # A second layer reuses this selection instead of running nonzero again.
        harness.pool._broadcast_slots.finish("swa", 0)
        harness.fetch_swa(layer=1, cache=cache)
        self.assertIs(harness.pool._batch_active_pages["swa"].selected_pages, selected)
        self.assertEqual(harness.events.count(("nonzero_static", "main")), 1)

    def test_swa_disabled_request_window_and_legacy_call_keep_dynamic_fallback(self):
        for options in ({"enabled": False}, {"window": 16}, {"legacy": True}):
            with self.subTest(options=options):
                harness = _Harness(enabled=options.get("enabled", True), peer_pages=(6,))
                harness.pool.request_window = options.get("window")
                if options.get("legacy"):
                    harness.pool.prefetch_swa_layer(
                        0, torch.tensor([2, 3, 8, 9, -1], dtype=torch.int32)
                    )
                else:
                    harness.fetch_swa()
                self.assertIn(("nonzero", "main"), harness.events)
                self.assertFalse(any(e[0] == "nonzero_static" for e in harness.events))
                torch.testing.assert_close(
                    harness.pool._swa_remapped_indices,
                    torch.tensor([0, 1, 2, 3, -1], dtype=torch.int32),
                )

    def test_tail_without_chunk_cache_selects_static_pages_on_broadcast_slot(self):
        for layer in (8, 9):
            for rank in (0, 1):
                with self.subTest(layer=layer, rank=rank):
                    harness = _Harness(rank=rank, peer_pages=(6,))
                    metadata = harness.fetch_tail(layer=layer)
                    family = harness.pool._extra_family_name(
                        harness.pool.compression_ratios[layer]
                    )
                    selected = harness.pool._batch_active_pages[family].selected_pages
                    torch.testing.assert_close(selected[:3], torch.tensor([1, 4, 6]))
                    self.assertTrue(bool((selected[3:] == -1).all()))
                    torch.testing.assert_close(
                        harness.pool._extra_remapped_indices,
                        torch.tensor([0, 1, 2, 3, -1], dtype=torch.int32),
                    )
                    torch.testing.assert_close(
                        harness.staging.value[:3], harness.reference[[1, 4, 6]]
                    )
                    self.assertIn(("nonzero_static", "side1"), harness.events)
                    self.assertFalse(any(e[0] == "nonzero" for e in harness.events))
                    self.assertIsNone(metadata.sparse_prefill_cache)
                    self.assertIsNotNone(metadata.late_layer_tail)
                    harness.pool._broadcast_slots.finish("extra", layer)
                    self.assertEqual(harness.events[-1], ("wait", "main", "side1"))

    def test_tail_empty_rank_still_fetches_peer_pages(self):
        harness = _Harness(peer_pages=(1, 4, 6))
        harness.fetch_tail(indices=torch.empty(0, dtype=torch.int32))
        selected = harness.pool._batch_active_pages["extra_c1"].selected_pages
        torch.testing.assert_close(selected[:3], torch.tensor([1, 4, 6]))
        self.assertEqual(harness.pool._extra_remapped_indices.numel(), 0)
        self.assertEqual(harness.events.count(("all_reduce", "side1")), 1)
        self.assertEqual(harness.events.count(("broadcast", "side1")), 1)
        self.assertIn(("nonzero_static", "side1"), harness.events)

    def test_tail_history_bound_covers_fragmented_cp_page_union(self):
        for layer, ratio in ((8, 1), (9, 2)):
            for lengths in ((), (0,), (1,), (32768,), (1048576,),
                            (1, 129, 32768, 1048575)):
                harness = _Harness()
                captured = {}
                harness.pool.prefetch_low_ratio_extra_layer = (
                    lambda *args, **kwargs: captured.update(kwargs)
                )
                harness.fetch_tail(layer=layer, seq_lens=lengths)
                capacity = captured["read_token_capacity"]
                self.assertEqual(capacity, sum((n + ratio - 1) // ratio for n in lengths))
                self.assertEqual(captured["num_reqs"], len(lengths))
                self.assertIsNone(captured["read_indices"])
                page_size = 256 // ratio
                # Maximal boundary misalignment, with unrelated physical page IDs.
                compressed_lens = [(n + ratio - 1) // ratio for n in lengths]
                pages = sum((n + 2 * page_size - 2) // page_size
                            for n in compressed_lens if n > 0)
                max_pages = pages + 31
                owners = torch.randperm(max_pages, generator=torch.Generator().manual_seed(42))[:pages]
                static_size = harness.namespace["contiguous_request_page_capacity"](
                    capacity, len(lengths), page_size, max_pages
                )
                self.assertGreaterEqual(static_size, pages)
                for rank in range(8):
                    with self.subTest(layer=layer, lengths=lengths, rank=rank):
                        # The local rank sees only its part; peers can select any
                        # other page in the same full request histories.
                        local = owners[rank::8]
                        harness.peer_pages = owners.tolist()
                        selected = harness.namespace["active_pages_for_indices"](
                            local.to(torch.int32) * page_size, page_size, max_pages,
                            harness.namespace["get_pynccl_broadcast_comm"](), static_size,
                        )
                        torch.testing.assert_close(selected[:pages], owners.sort().values)
                        self.assertTrue(bool((selected[pages:] == -1).all()))
                        self.assertLess(static_size, max_pages)

    def test_tail_disabled_window_and_missing_metadata_keep_dynamic_fallback(self):
        for options in ({"enabled": False}, {"window": 16}, {"tail": False}):
            with self.subTest(options=options):
                harness = _Harness(enabled=options.get("enabled", True), peer_pages=(6,))
                harness.pool.request_window = options.get("window")
                harness.fetch_tail(tail=options.get("tail", True))
                self.assertIn(("nonzero", "main"), harness.events)
                torch.testing.assert_close(
                    harness.pool._extra_remapped_indices,
                    torch.tensor([0, 1, 2, 3, -1], dtype=torch.int32),
                )

    def test_static_selection_copy_broadcast_and_remap_share_one_slot(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                harness = _Harness(rank=rank, peer_pages=(6,))
                harness.fetch(read_indices=torch.tensor([2, 3, 8, 9, -1, -1]))
                events, pool = harness.events, harness.pool
                torch.testing.assert_close(pool._extra_remapped_indices,
                                           torch.tensor([0, 1, 2, 3, 4, -1], dtype=torch.int32))
                torch.testing.assert_close(harness.staging.value[:3], harness.reference[[1, 4, 6]])
                for operation in ("build", "all_reduce", "nonzero_static", "broadcast", "remap"):
                    self.assertEqual([item for item in events if item[0] == operation],
                                     [(operation, "side1")])
                self.assertEqual(events.count(("launch", "main")), 1)
                self.assertEqual(events.count(("clear", "main")), 1)
                self.assertFalse(any(item[0] == "start" for item in events))
                self.assertIn(("owner", 0), events)  # Source layer 4 owns consumer layer 8.
                self.assertEqual([item for item in events if item[0] == "gather_out"],
                                 [("gather_out", "side1")] if rank == 0 else [])
                self.assertFalse(any(item[0] == "copy" for item in events))
                harness.record("independent_q")
                self.assertTrue(pool._broadcast_slots.pending("extra").active)
                pool._broadcast_slots.finish("extra", 8)
                self.assertFalse(pool._broadcast_slots.pending("extra").active)
                self.assertLess(events.index(("remap", "side1")), events.index(("independent_q", "main")))
                self.assertEqual(events[-1], ("wait", "main", "side1"))

    def test_dynamic_disabled_and_capture_keep_original_launch_boundaries(self):
        cases = ({"enabled": False}, {"graph": True}, {"num_reqs": None},
                 {"request_window": 16})
        for options in cases:
            with self.subTest(options=options):
                options = dict(options)
                num_reqs = options.pop("num_reqs", 1)
                request_window = options.pop("request_window", None)
                harness = _Harness(**options)
                harness.pool.request_window = request_window
                harness.fetch(num_reqs=num_reqs)
                events = harness.events
                self.assertIn(("build", "main"), events)
                self.assertIn(("all_reduce", "main"), events)
                self.assertIn(("start", "main"), events)
                self.assertIn(("broadcast", "side1"), events)
                self.assertIn(("remap", "main"), events)
                selector = "nonzero_static" if options.get("graph") else "nonzero"
                self.assertIn((selector, "main"), events)
                torch.testing.assert_close(harness.pool._extra_remapped_indices,
                                           torch.tensor([0, 1, 2, 3, 4, -1], dtype=torch.int32))

    def test_empty_rank_keeps_collectives_and_one_dummy_broadcast_page(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                harness = _Harness(rank=rank)
                empty = torch.empty(0, dtype=torch.int32)
                harness.fetch(indices=empty, read_indices=empty, num_reqs=0)
                self.assertEqual(harness.pool._extra_remapped_indices.numel(), 0)
                self.assertEqual(harness.events.count(("all_reduce", "side1")), 1)
                self.assertEqual(harness.events.count(("broadcast", "side1")), 1)
                self.assertEqual(harness.events.count(("launch", "main")), 1)
                self.assertEqual(harness.events.count(("clear", "main")), 1)
                self.assertTrue(harness.pool._broadcast_slots.pending("extra").active)
                torch.testing.assert_close(harness.staging.value[:1], harness.reference[:1])

    def test_next_layer_drains_previous_slot_and_rebuilds_pages(self):
        harness = _Harness(rank=0)
        harness.fetch()
        harness.fetch(layer=9, indices=torch.tensor([0, 1, 6, -1], dtype=torch.int32))
        torch.testing.assert_close(harness.pool._extra_remapped_indices,
                                   torch.tensor([0, 1, 2, -1], dtype=torch.int32))
        selected = harness.pool._batch_active_pages["extra_c2"].selected_pages
        torch.testing.assert_close(selected[:2], torch.tensor([0, 3]))
        self.assertEqual(harness.pool._extra_remapped_layer_id, 9)
        self.assertEqual(harness.pool._extra_staging_layer_id, 9)
        self.assertEqual(harness.pool._broadcast_slots.pending("extra").layer_id, 9)
        self.assertEqual(harness.events.count(("build", "side1")), 2)
        self.assertEqual(harness.events.count(("all_reduce", "side1")), 2)
        self.assertEqual(harness.events.count(("wait", "main", "side1")), 1)
        second_build = [i for i, event in enumerate(harness.events)
                        if event == ("build", "side1")][1]
        self.assertLess(harness.events.index(("wait", "main", "side1")), second_build)
        self.assertIn(("owner", 1), harness.events)


if __name__ == "__main__":
    unittest.main()
