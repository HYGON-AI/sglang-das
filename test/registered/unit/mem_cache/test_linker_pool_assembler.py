"""Unit tests for external-linker device pool assembly."""

import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
    _build_deepseek_v4_device_pool_group,
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDevicePoolEntry(CustomTestCase):
    def test_sparse_multi_component_layer_ranges(self):
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
        self.assertEqual(locations, [0, 4])
        pointers, sizes = pool.get_page_buffer_meta(indices)
        self.assertEqual(
            pointers,
            [
                buffer[row].data_ptr()
                for row in locations
                for buffer in (k0, k2, v0, v2)
            ],
        )
        self.assertEqual(sizes, [6, 10, 14, 22] * 2)
        self.assertIsNone(pool.get_prepared_layer_range_meta(locations, 1))

        pointers, sizes, offsets = pool.get_prepared_layer_range_meta(locations, 2)
        self.assertEqual(
            pointers,
            [
                [k2[0].data_ptr()],
                [v2[0].data_ptr()],
                [k2[4].data_ptr()],
                [v2[4].data_ptr()],
            ],
        )
        self.assertEqual(sizes, [[10], [22], [10], [22]])
        self.assertEqual(offsets, [[6], [14], [6], [14]])

    def test_rejects_invalid_pages_and_empty_buffers(self):
        with self.assertRaisesRegex(ValueError, "has no storage buffers"):
            DevicePoolEntry(
                name=PoolName.KV,
                indices_from_pool=PoolName.KV,
                device_pool=None,
                components=[],
                layer_mapping={},
                page_size=2,
                rows_are_pages=False,
            )

        pool = DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[[torch.zeros((8, 3), dtype=torch.uint8)]],
            layer_mapping={0: 0},
            page_size=2,
            rows_are_pages=False,
        )
        for indices, error in (
            (torch.tensor([0]), "multiple of page_size"),
            (torch.tensor([1, 2]), "aligned contiguous pages"),
            (torch.tensor([0, 2]), "aligned contiguous pages"),
            (torch.tensor([8, 9]), "exceeds buffer shapes"),
        ):
            with self.subTest(indices=indices.tolist()):
                with self.assertRaisesRegex(ValueError, error):
                    pool.prepare_locations(indices)


class TestDevicePoolGroup(CustomTestCase):
    def test_resolve_transfers_expands_physical_pools(self):
        entries = [
            SimpleNamespace(
                name=PoolName.KV,
                indices_from_pool=PoolName.KV,
                translate_indices=lambda indices: indices,
            ),
            SimpleNamespace(
                name=PoolName.INDEXER,
                indices_from_pool=PoolName.KV,
                translate_indices=lambda indices: indices + 100,
            ),
        ]
        group = DevicePoolGroup(entries, num_layers=2, page_size=2)
        transfer = PoolTransfer(
            name=PoolName.KV,
            keys=["a", "b"],
            device_indices=torch.tensor([0, 1, 4, 5]),
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

        resolved = group.resolve_transfers([transfer])

        self.assertEqual(
            [item.name for item in resolved], [PoolName.KV, PoolName.INDEXER]
        )
        self.assertEqual(resolved[0].host_indices.tolist(), [0, 1, 4, 5])
        self.assertEqual(resolved[1].host_indices.tolist(), [100, 101, 104, 105])
        self.assertTrue(
            all(item.hit_policy == PoolHitPolicy.ALL_PAGES for item in resolved)
        )

    def test_partial_side_pool_requires_explicit_opt_in(self):
        entry = SimpleNamespace(
            name=PoolName.SWA,
            indices_from_pool=PoolName.SWA,
            translate_indices=lambda indices: indices + 100,
        )
        group = DevicePoolGroup([entry], num_layers=1, page_size=2)
        transfer = PoolTransfer(
            name=PoolName.SWA,
            keys=["b", "d"],
            device_indices=torch.tensor([20, 21, 24, 25]),
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

        self.assertEqual(group.resolve_transfers([transfer]), [])
        resolved = group.resolve_transfers(
            [transfer], allow_partial=True, allow_missing_kv=True
        )

        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].name, PoolName.SWA)
        self.assertEqual(resolved[0].keys, ["b", "d"])
        self.assertEqual(resolved[0].host_indices.tolist(), [120, 121, 124, 125])
        self.assertEqual(resolved[0].hit_policy, PoolHitPolicy.TRAILING_PAGES)


class TestHybridDevicePoolAssembler(CustomTestCase):
    def test_deepseek_v4_maps_sparse_sidecars(self):
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
        kvcache.start_layer = 1
        kvcache.end_layer = 4
        kvcache.swa_page_size = 2
        kvcache.swa_kv_pool = SimpleNamespace(
            page_size=2,
            kv_buffer=[torch.zeros((8, 3), dtype=torch.uint8) for _ in range(3)],
        )
        kvcache.c4_kv_pool = SimpleNamespace(
            kv_buffer=[torch.zeros((8, 5), dtype=torch.uint8) for _ in range(2)],
            bytes_per_page_padded=5,
        )
        kvcache.c4_indexer_kv_pool = SimpleNamespace(
            index_k_with_scale_buffer=[
                torch.zeros((8, 7), dtype=torch.uint8) for _ in range(2)
            ]
        )
        kvcache.c128_kv_pool = SimpleNamespace(
            kv_buffer=[torch.zeros((8, 11), dtype=torch.uint8)],
            bytes_per_page_padded=11,
        )
        kvcache.layer_mapping = [
            DeepSeekV4LayerItem(0, -1),
            DeepSeekV4LayerItem(4, 0),
            DeepSeekV4LayerItem(128, 0),
            DeepSeekV4LayerItem(4, 1),
        ]
        kvcache.compress_state_pools = [None, state_pool(), None, state_pool()]
        kvcache.indexer_compress_state_pools = [
            None,
            state_pool(),
            None,
            state_pool(),
        ]
        draft_swa_buffers = [
            torch.zeros((8, 13), dtype=torch.uint8),
            torch.zeros((8, 17), dtype=torch.uint8),
        ]

        group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=2,
            params=SimpleNamespace(
                mtp_draft_device_pools=(
                    SimpleNamespace(
                        swa_kv_pool=SimpleNamespace(kv_buffer=draft_swa_buffers)
                    ),
                )
            ),
            components={ComponentType.FULL, ComponentType.SWA},
        )

        self.assertEqual(group.num_layers, 3)
        self.assertTrue(group.rank_replicated)
        self.assertEqual(
            set(group.entry_map),
            {
                PoolName.SWA,
                PoolName.DEEPSEEK_V4_C4,
                PoolName.DEEPSEEK_V4_C4_INDEXER,
                PoolName.DEEPSEEK_V4_C128,
                PoolName.DEEPSEEK_V4_C4_STATE,
                PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
            },
        )
        self.assertEqual(group.sources[PoolName.DEEPSEEK_V4_C4], PoolName.KV)
        self.assertEqual(group.sources[PoolName.DEEPSEEK_V4_C4_STATE], PoolName.SWA)

        c4_pool = group.entry_map[PoolName.DEEPSEEK_V4_C4]
        pointers, sizes = c4_pool.get_page_buffer_meta(torch.tensor([0, 1]))
        self.assertEqual(len(pointers), 2)
        self.assertEqual(sizes, [5, 5])
        _, sizes, offsets = c4_pool.get_prepared_layer_range_meta([0], 2)
        self.assertEqual(sizes, [[5]])
        self.assertEqual(offsets, [[5]])
        self.assertIsNone(c4_pool.get_prepared_layer_range_meta([0], 1))

        swa_pool = group.entry_map[PoolName.SWA]
        _, sizes, offsets = swa_pool.get_prepared_layer_range_meta([0], 0)
        self.assertEqual(sizes, [[3, 13]])
        self.assertEqual(offsets, [[0, 9]])
        _, sizes, offsets = swa_pool.get_prepared_layer_range_meta([0], 1)
        self.assertEqual(sizes, [[3, 17]])
        self.assertEqual(offsets, [[3, 22]])

    def test_unified_deepseek_v4_uses_only_compressed_pools(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4LayerItem

        for split_indexer in (False, True):
            with self.subTest(split_indexer=split_indexer):
                c4 = [torch.zeros((4, width), dtype=torch.uint8) for width in (5, 7)]
                c128 = [torch.zeros((4, 11), dtype=torch.uint8)]
                expected = {
                    PoolName.DEEPSEEK_V4_C4: c4,
                    PoolName.DEEPSEEK_V4_C128: c128,
                }
                if split_indexer:
                    payload = [
                        torch.zeros((4, 1, 4, 1, 16), dtype=torch.uint8)
                        for _ in range(2)
                    ]
                    scale = [
                        torch.zeros((4, 1, 4, 1), dtype=torch.uint8) for _ in range(2)
                    ]
                    indexer = SimpleNamespace(
                        index_k_with_scale_buffer=None,
                        index_k_payload_buffer=payload,
                        index_k_scale_buffer=scale,
                    )
                    expected[PoolName.DEEPSEEK_V4_C4_INDEXER] = [
                        b.flatten(1) for b in payload
                    ]
                    expected[PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE] = [
                        b.flatten(1) for b in scale
                    ]
                else:
                    buffers = [
                        torch.zeros((4, width), dtype=torch.uint8) for width in (13, 17)
                    ]
                    indexer = SimpleNamespace(index_k_with_scale_buffer=buffers)
                    expected[PoolName.DEEPSEEK_V4_C4_INDEXER] = buffers
                regions = {4: (c4, 7), 128: (c128, 11)}
                kvcache = SimpleNamespace(
                    _unified_kv=True,
                    start_layer=0,
                    end_layer=3,
                    layer_mapping=[
                        DeepSeekV4LayerItem(4, 1),
                        DeepSeekV4LayerItem(128, 0),
                        DeepSeekV4LayerItem(4, 0),
                    ],
                    # Unified KV has neither paged KV nor an index-addressed SWA pool.
                    swa_kv_pool=None,
                    c4_kv_pool=None,
                    c128_kv_pool=None,
                    swa_page_size=3,
                    c4_indexer_kv_pool=indexer,
                    unified_region_buffers=Mock(side_effect=regions.__getitem__),
                    # bf16 unified rows are a single pool: no rope sibling.
                    unified_rope_region_buffers=Mock(return_value=None),
                )
                group = _build_deepseek_v4_device_pool_group(kvcache, page_size=2)

                # expected carries no *_ROPE entry, so this also pins that the
                # bf16 layout ships exactly what it did before.
                self.assertEqual(set(group.entry_map), set(expected))
                self.assertEqual(set(group.sources.values()), {PoolName.KV})
                self.assertTrue(group.rank_replicated)
                for name, buffers in expected.items():
                    entry = group.entry_map[name]
                    actual = entry.components[0]
                    self.assertEqual(len(actual), len(buffers))
                    for got, want in zip(actual, buffers):
                        self.assertEqual(got.data_ptr(), want.data_ptr())
                        self.assertEqual(got.shape, want.shape)
                self.assertEqual(
                    kvcache.unified_region_buffers.call_args_list, [call(4), call(128)]
                )
                resolved = group.resolve_transfers(
                    [
                        PoolTransfer(
                            name=PoolName.KV,
                            keys=["page-0"],
                            device_indices=torch.tensor([0, 1]),
                        )
                    ]
                )
                self.assertEqual({t.name for t in resolved}, set(expected))

    def test_unified_deepseek_v4_fp8_ships_the_rope_half(self):
        """fp8 rows are two pools; shipping nope alone leaves stale rope."""
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4LayerItem

        c4 = [torch.zeros((4, width), dtype=torch.uint8) for width in (5, 7)]
        c128 = [torch.zeros((4, 11), dtype=torch.uint8)]
        c4_rope = [torch.zeros((4, 3), dtype=torch.uint8) for _ in range(2)]
        c128_rope = [torch.zeros((4, 3), dtype=torch.uint8)]
        indexer = [torch.zeros((4, width), dtype=torch.uint8) for width in (13, 17)]
        regions = {4: (c4, 7), 128: (c128, 11)}
        rope_regions = {4: (c4_rope, 3), 128: (c128_rope, 3)}
        kvcache = SimpleNamespace(
            _unified_kv=True,
            start_layer=0,
            end_layer=3,
            layer_mapping=[
                DeepSeekV4LayerItem(4, 1),
                DeepSeekV4LayerItem(128, 0),
                DeepSeekV4LayerItem(4, 0),
            ],
            swa_kv_pool=None,
            c4_kv_pool=None,
            c128_kv_pool=None,
            swa_page_size=3,
            c4_indexer_kv_pool=SimpleNamespace(index_k_with_scale_buffer=indexer),
            unified_region_buffers=Mock(side_effect=regions.__getitem__),
            unified_rope_region_buffers=Mock(side_effect=rope_regions.__getitem__),
        )
        group = _build_deepseek_v4_device_pool_group(kvcache, page_size=2)

        expected_rope = {
            PoolName.DEEPSEEK_V4_C4_ROPE: c4_rope,
            PoolName.DEEPSEEK_V4_C128_ROPE: c128_rope,
        }
        self.assertTrue(set(expected_rope).issubset(group.entry_map))
        for name, buffers in expected_rope.items():
            actual = group.entry_map[name].components[0]
            self.assertEqual(len(actual), len(buffers))
            for got, want in zip(actual, buffers):
                self.assertEqual(got.data_ptr(), want.data_ptr())
                self.assertEqual(got.shape, want.shape)
        self.assertEqual(
            kvcache.unified_rope_region_buffers.call_args_list, [call(4), call(128)]
        )

        # One row index addresses both halves: same layer mapping, same source.
        for rope_name, nope_name in (
            (PoolName.DEEPSEEK_V4_C4_ROPE, PoolName.DEEPSEEK_V4_C4),
            (PoolName.DEEPSEEK_V4_C128_ROPE, PoolName.DEEPSEEK_V4_C128),
        ):
            self.assertEqual(
                group.entry_map[rope_name].layer_mapping,
                group.entry_map[nope_name].layer_mapping,
            )
            self.assertEqual(group.sources[rope_name], group.sources[nope_name])

        # One KV transfer must expand to both halves.
        resolved = group.resolve_transfers(
            [
                PoolTransfer(
                    name=PoolName.KV,
                    keys=["page-0"],
                    device_indices=torch.tensor([0, 1]),
                )
            ]
        )
        self.assertTrue(set(expected_rope).issubset({t.name for t in resolved}))

    def test_deepseek_v4_still_rejects_hisparse(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import HiSparseC4DevicePool

        kvcache = SimpleNamespace(
            c4_kv_pool=HiSparseC4DevicePool.__new__(HiSparseC4DevicePool)
        )
        with self.assertRaisesRegex(ValueError, "does not support HiSparse"):
            _build_deepseek_v4_device_pool_group(kvcache, 2)

    def test_dsa_uses_hybrid_assembler_strategy(self):
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

        def dsa_pool(kv_width, index_width):
            pool = DSATokenToKVPool.__new__(DSATokenToKVPool)
            pool.page_size = 2
            pool.layer_num = 1
            pool.kv_buffer = [torch.zeros((8, kv_width), dtype=torch.uint8)]
            pool.index_key_cache = SimpleNamespace(
                buffer=[torch.zeros((4, index_width), dtype=torch.uint8)]
            )
            return pool

        kvcache = dsa_pool(3, 7)
        kvcache.layer_num = 2
        kvcache.kv_buffer.append(torch.zeros((8, 5), dtype=torch.uint8))
        kvcache.index_key_cache.buffer.append(torch.zeros((4, 11), dtype=torch.uint8))
        draft_pools = (dsa_pool(13, 17), dsa_pool(19, 23))

        group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=2,
            params=SimpleNamespace(mtp_draft_device_pools=draft_pools),
            components={ComponentType.FULL},
        )

        self.assertEqual(group.num_layers, 2)
        self.assertTrue(group.rank_replicated)
        self.assertEqual(set(group.entry_map), {PoolName.KV, PoolName.INDEXER})
        self.assertEqual(
            group.sources,
            {
                PoolName.KV: PoolName.KV,
                PoolName.INDEXER: PoolName.KV,
            },
        )
        _, sizes, offsets = group.entry_map[PoolName.KV].get_prepared_layer_range_meta(
            [0], 0
        )
        self.assertEqual(sizes, [[6, 26]])
        self.assertEqual(offsets, [[0, 16]])
        _, sizes, offsets = group.entry_map[
            PoolName.INDEXER
        ].get_prepared_layer_range_meta([0], 0)
        self.assertEqual(sizes, [[7, 17]])
        self.assertEqual(offsets, [[0, 18]])
        _, sizes, offsets = group.entry_map[
            PoolName.INDEXER
        ].get_prepared_layer_range_meta([0], 1)
        self.assertEqual(sizes, [[11, 23]])
        self.assertEqual(offsets, [[7, 35]])

    def test_linker_requires_packed_draft(self):
        """Do not accept draft state that the linker would omit from storage."""
        from sglang.srt.speculative import base_spec_worker as spec
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        draft = SimpleNamespace(
            token_to_kv_pool=object(),
            model_config=SimpleNamespace(
                num_nextn_predict_layers=0,
                hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
            ),
        )
        target = SimpleNamespace(spec_algorithm=SpeculativeAlgorithm.EAGLE)
        worker = SimpleNamespace(
            target_worker=SimpleNamespace(model_runner=target),
            _draft_model_runners=lambda: (draft,),
        )
        for linker_enabled, nextn_layers in (
            (False, 0),
            (True, 0),
            (False, 1),
            (True, 1),
        ):
            draft.model_config.num_nextn_predict_layers = nextn_layers
            with (
                self.subTest(linker=linker_enabled, nextn=nextn_layers),
                patch.object(
                    spec,
                    "get_memory",
                    return_value=SimpleNamespace(
                        enable_hierarchical_cache=not linker_enabled,
                        enable_unified_cache_external_linker=linker_enabled,
                    ),
                ),
            ):
                if linker_enabled and not nextn_layers:
                    with self.assertRaisesRegex(
                        NotImplementedError, "only supports packed"
                    ):
                        spec.BaseSpecWorker._build_hicache_draft_plan(worker)
                    self.assertEqual(target.mtp_draft_device_pools, ())
                else:
                    plan = spec.BaseSpecWorker._build_hicache_draft_plan(worker)
                    self.assertEqual(
                        plan.mode,
                        (
                            spec.HiCacheDraftMode.PACKED
                            if nextn_layers
                            else spec.HiCacheDraftMode.SIDECAR
                        ),
                    )
                    self.assertEqual(plan.device_pools, (draft.token_to_kv_pool,))
                    self.assertEqual(
                        target.mtp_draft_device_pools,
                        plan.device_pools if nextn_layers else (),
                    )

    def test_unsupported_strategy_fails_with_context(self):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        kvcache = HybridLinearKVPool.__new__(HybridLinearKVPool)
        with self.assertRaisesRegex(
            ValueError,
            "does not support the direct external linker: _MambaStrategy",
        ):
            resolve_hybrid_device_pool_group(
                kvcache=kvcache,
                page_size=2,
                params=SimpleNamespace(),
                components={ComponentType.FULL, ComponentType.MAMBA},
            )


class TestDeepSeekV41DevicePoolAssembler(CustomTestCase):
    def make_pool(self, cp_rank=None, cp_size=8, split_indexer=False, start=0):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        pool._unified_kv = False
        pool.start_layer, pool.end_layer = start, start + 40
        pool._stage_start, pool._stage_end = start, start + 40
        pool.page_size = pool.swa_page_size = 8
        pool.compression_ratios = [0] * start + [0, 0] + [2] * 18 + [1] * 20
        pool.sources_by_ratio = {1: [start + 20], 2: [start + i for i in (2, 8, 14)]}
        pool.c4_kv_pool = pool.c128_kv_pool = pool.c4_indexer_kv_pool = None
        owned = list(range(start, start + 40))
        if cp_rank is not None:
            width = 40 // cp_size
            owned = owned[cp_rank * width : (cp_rank + 1) * width]
            pool._swa_global_to_local = {layer: i for i, layer in enumerate(owned)}
            pool._low_owner_local_by_source = {}
            for sources in pool.sources_by_ratio.values():
                counts = [0] * cp_size
                for source in sources:
                    owner = (source - start) // width
                    pool._low_owner_local_by_source[source] = counts[owner]
                    counts[owner] += 1
        pool.swa_kv_pool = SimpleNamespace(
            page_size=8,
            kv_buffer=[torch.zeros((8, 3), dtype=torch.uint8) for _ in owned],
        )
        pool.kv_pools, pool.index_pools = {}, {}
        for ratio, sources in pool.sources_by_ratio.items():
            kv_sources = [source for source in sources if source in owned]
            pool.kv_pools[ratio] = SimpleNamespace(
                page_size=8 // ratio,
                kv_buffer=[
                    torch.zeros((8, 5 + ratio), dtype=torch.uint8) for _ in kv_sources
                ],
            )
            # Include a partial padding group: it must not become a FULL page.
            rows = 8 * (8 // ratio // 2) + 1
            if split_indexer:
                index_pool = SimpleNamespace(
                    page_size=2,
                    index_k_with_scale_buffer=None,
                    index_k_payload_buffer=[
                        torch.zeros((rows, 1, 4, 2, 16), dtype=torch.uint8)
                        for _ in sources
                    ],
                    index_k_scale_buffer=[
                        torch.zeros((rows, 1, 4, 2), dtype=torch.uint8) for _ in sources
                    ],
                )
            else:
                index_pool = SimpleNamespace(
                    page_size=2,
                    index_k_with_scale_buffer=[
                        torch.zeros((rows, 11 + ratio), dtype=torch.uint8)
                        for _ in sources
                    ],
                )
            pool.index_pools[ratio] = index_pool
        draft = SimpleNamespace(
            swa_page_size=8,
            sources_by_ratio={},
            swa_kv_pool=SimpleNamespace(
                page_size=8,
                kv_buffer=[
                    torch.zeros((8, w), dtype=torch.uint8) for w in (13, 17, 19)
                ],
            ),
        )
        return pool, draft

    def build(self, pool, draft):
        return resolve_hybrid_device_pool_group(
            kvcache=pool,
            page_size=pool.page_size,
            params=SimpleNamespace(mtp_draft_device_pools=(draft,)),
            components={ComponentType.FULL, ComponentType.SWA},
        )

    def test_shared_source_mapping_and_index_page_grouping(self):
        for split in (False, True):
            with self.subTest(split=split):
                pool, draft = self.make_pool(split_indexer=split, start=3)
                group = self.build(pool, draft)
                self.assertEqual(group.num_layers, 40)
                self.assertTrue(group.rank_replicated)
                c2 = group.entry_map[PoolName.DEEPSEEK_V4_C2]
                self.assertEqual(c2.layer_mapping, {2: 0, 8: 1, 14: 2})
                self.assertIsNone(c2.get_prepared_layer_range_meta([1], 3))
                ptrs, sizes, offsets = c2.get_prepared_layer_range_meta([1], 8)
                self.assertEqual(ptrs, [[pool.kv_pools[2].kv_buffer[1][1].data_ptr()]])
                self.assertEqual(sizes, [[7]])
                self.assertEqual(offsets, [[7]])
                for ratio, name in (
                    (1, PoolName.DEEPSEEK_V4_C1_INDEXER),
                    (2, PoolName.DEEPSEEK_V4_C2_INDEXER),
                ):
                    entry = group.entry_map[name]
                    original = pool.index_pools[ratio]
                    buffers = (
                        original.index_k_payload_buffer
                        if split
                        else original.index_k_with_scale_buffer
                    )
                    factor = 8 // ratio // original.page_size
                    for view, buffer in zip(entry.kv_buffer, buffers):
                        self.assertEqual(view.data_ptr(), buffer.data_ptr())
                        self.assertEqual(view.shape[0], 8)
                        self.assertEqual(view.stride(0), buffer[0].numel() * factor)
                    pointers, _ = entry.get_page_buffer_meta(torch.arange(8, 16))
                    self.assertEqual(pointers[0], buffers[0][factor].data_ptr())
                self.assertEqual(len(group.entries), 7 if split else 5)

    def test_cp_ownership_and_packed_draft_on_all_ranks(self):
        for cp_size in (1, 2, 8):
            for rank in range(cp_size):
                with self.subTest(cp_size=cp_size, rank=rank):
                    pool, draft = self.make_pool(cp_rank=rank, cp_size=cp_size)
                    group = self.build(pool, draft)
                    for ratio, kv_name, index_name in (
                        (1, PoolName.DEEPSEEK_V4_C1, PoolName.DEEPSEEK_V4_C1_INDEXER),
                        (2, PoolName.DEEPSEEK_V4_C2, PoolName.DEEPSEEK_V4_C2_INDEXER),
                    ):
                        sources = pool.sources_by_ratio[ratio]
                        local = [s for s in sources if s in pool._swa_global_to_local]
                        self.assertEqual(kv_name in group.entry_map, bool(local))
                        if local:
                            self.assertEqual(
                                group.entry_map[kv_name].layer_mapping,
                                {s: i for i, s in enumerate(local)},
                            )
                        self.assertEqual(
                            group.entry_map[index_name].layer_mapping,
                            {s: i for i, s in enumerate(sources)},
                        )
                    swa = group.entry_map[PoolName.SWA]
                    for depth, buffer in enumerate(draft.swa_kv_pool.kv_buffer):
                        ptrs, _, offsets = swa.get_prepared_layer_range_meta([1], depth)
                        self.assertEqual(ptrs[0][-1], buffer[1].data_ptr())
                        self.assertEqual(
                            offsets[0][-1],
                            len(pool.swa_kv_pool.kv_buffer) * 3
                            + sum((13, 17, 19)[:depth]),
                        )

    def test_page_roundtrip_uses_all_bytes_and_restores_different_slots(self):
        for split in (False, True):
            for rank in range(8):
                with self.subTest(split=split, rank=rank):
                    pool, draft = self.make_pool(cp_rank=rank, split_indexer=split)
                    group = self.build(pool, draft)
                    for entry in group.entries:
                        for i, buffer in enumerate(entry.kv_buffer):
                            buffer.copy_(
                                ((torch.arange(buffer.numel()) + 31 * i) % 251)
                                .to(torch.uint8)
                                .reshape(buffer.shape)
                            )
                    outgoing = group.resolve_transfers(
                        [
                            PoolTransfer(
                                name=PoolName.KV,
                                keys=["a", "b"],
                                device_indices=torch.tensor(
                                    [*range(8, 16), *range(24, 32)]
                                ),
                            ),
                            PoolTransfer(
                                name=PoolName.SWA,
                                keys=["b"],
                                device_indices=torch.arange(16, 24),
                                hit_policy=PoolHitPolicy.TRAILING_PAGES,
                            ),
                        ]
                    )
                    objects = {}
                    for transfer in outgoing:
                        entry = group.entry_map[transfer.name]
                        ptrs, sizes = entry.get_page_buffer_meta(transfer.host_indices)
                        width = len(ptrs) // len(transfer.keys)
                        for i, key in enumerate(transfer.keys):
                            objects[(transfer.name, key)] = b"".join(
                                ctypes.string_at(p, n)
                                for p, n in zip(
                                    ptrs[i * width : (i + 1) * width],
                                    sizes[i * width : (i + 1) * width],
                                )
                            )
                        for buffer in entry.kv_buffer:
                            buffer.zero_()
                    incoming = group.resolve_transfers(
                        [
                            PoolTransfer(
                                name=PoolName.KV,
                                keys=["a", "b"],
                                device_indices=torch.tensor(
                                    [*range(16, 24), *range(40, 48)]
                                ),
                            ),
                            PoolTransfer(
                                name=PoolName.SWA,
                                keys=["b"],
                                device_indices=torch.arange(32, 40),
                                hit_policy=PoolHitPolicy.TRAILING_PAGES,
                            ),
                        ]
                    )
                    for transfer in incoming:
                        entry = group.entry_map[transfer.name]
                        locations = entry.prepare_locations(transfer.host_indices)
                        coverage = [set() for _ in transfer.keys]
                        for layer in range(group.num_layers):
                            meta = entry.get_prepared_layer_range_meta(locations, layer)
                            if meta is None:
                                continue
                            for i, (ptrs, sizes, offsets) in enumerate(zip(*meta)):
                                data = objects[(transfer.name, transfer.keys[i])]
                                for pointer, size, offset in zip(ptrs, sizes, offsets):
                                    segment = set(range(offset, offset + size))
                                    self.assertFalse(coverage[i] & segment)
                                    coverage[i].update(segment)
                                    ctypes.memmove(
                                        pointer, data[offset : offset + size], size
                                    )
                        for i, key in enumerate(transfer.keys):
                            expected = objects[(transfer.name, key)]
                            self.assertEqual(coverage[i], set(range(len(expected))))
                            slots = transfer.host_indices[i * 8 : (i + 1) * 8]
                            ptrs, sizes = entry.get_page_buffer_meta(slots)
                            self.assertEqual(
                                b"".join(
                                    ctypes.string_at(p, n) for p, n in zip(ptrs, sizes)
                                ),
                                expected,
                            )

    def test_rejects_incompatible_layouts(self):
        for kind, error in (
            ("odd", "even tree page"),
            ("swa", "SWA page size"),
            ("swa_missing", "page-addressed SWA"),
            ("draft", "SWA page size"),
            ("index", "indexer pages"),
            ("mixed", "paged C1/C2"),
            ("draft_compressed", "SWA-only packed drafts"),
            ("missing_indexer", "does not cover"),
        ):
            pool, draft = self.make_pool()
            if kind == "odd":
                pool.page_size = 7
            elif kind == "swa":
                pool.swa_page_size = pool.swa_kv_pool.page_size = 4
            elif kind == "swa_missing":
                pool.swa_kv_pool = None
            elif kind == "draft":
                draft.swa_page_size = draft.swa_kv_pool.page_size = 4
            elif kind == "index":
                pool.index_pools[2].page_size = 3
            elif kind == "mixed":
                pool.sources_by_ratio[4] = [0]
            elif kind == "draft_compressed":
                draft.sources_by_ratio = {2: [0]}
            elif kind == "missing_indexer":
                pool.index_pools[2].index_k_with_scale_buffer.pop()
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, error):
                self.build(pool, draft)

    def test_real_v41_pools_with_cp8_and_packed_draft(self):
        from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
        from sglang.srt.mem_cache.cp_cache_layer_split.deepseek_v4_pool import (
            CpCacheLayerSplitDeepSeekV4TokenToKVPool,
        )
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args(page_size=256)
        override.install()
        self.addCleanup(override.restore)

        def make_real(ratios, sources, rank=None):
            cls = (
                DeepSeekV4TokenToKVPool
                if rank is None
                else CpCacheLayerSplitDeepSeekV4TokenToKVPool
            )
            return cls(
                max_num_reqs=2,
                swa_size=512,
                full_size=512,
                c4_size=0,
                c128_size=0,
                c4_state_pool_size=0,
                c128_state_pool_size=0,
                page_size=256,
                swa_page_size=256,
                dtype=torch.float8_e4m3fn,
                c4_state_dtype=torch.float32,
                c128_state_dtype=torch.float32,
                qk_nope_head_dim=448,
                qk_rope_head_dim=64,
                indexer_head_dim=128,
                layer_num=len(ratios),
                device="cpu",
                enable_memory_saver=False,
                compression_ratios=ratios,
                kv_source_layers=sources,
                kv_layout=KVLayout.V41,
                **({} if rank is None else {"cp_rank": rank, "cp_size": 8}),
            )

        draft = make_real([0, 0, 0], [])
        for rank in (None, *range(8)):
            with self.subTest(rank=rank):
                pool = make_real([0, 0] + [2] * 18 + [1] * 20, [2, 8, 14, 20], rank)
                group = self.build(pool, draft)
                self.assertIsNone(pool.c4_kv_pool)
                self.assertIsNone(pool.c128_kv_pool)
                self.assertEqual(group.num_layers, 40)
                for entry in group.entries:
                    # The last allocatable page includes the reserved page-0 gap.
                    ptrs, sizes = entry.get_page_buffer_meta(torch.arange(512, 768))
                    self.assertTrue(ptrs)
                    self.assertTrue(all(size > 0 for size in sizes))
                    original_allocations = {
                        buffer.untyped_storage().data_ptr()
                        for buffer in (
                            entry.device_pool.index_k_with_scale_buffer
                            if "indexer" in entry.name.value
                            else entry.device_pool.kv_buffer
                        )
                    }
                    if entry.name == PoolName.SWA:
                        original_allocations.update(
                            buffer.untyped_storage().data_ptr()
                            for buffer in draft.swa_kv_pool.kv_buffer
                        )
                    self.assertEqual(
                        {b.untyped_storage().data_ptr() for b in entry.kv_buffer},
                        original_allocations,
                    )


if __name__ == "__main__":
    unittest.main()
