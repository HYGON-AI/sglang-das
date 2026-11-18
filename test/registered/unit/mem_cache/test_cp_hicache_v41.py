"""CP8 L2 ownership and HCU byte-exact eviction/reload contracts."""

from math import prod
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as a


def _page_bytes(shape, offset, device):
    return (
        torch.arange(prod(shape), device=device, dtype=torch.int32)
        .to(torch.uint8)
        .reshape(shape)
        + offset
    )


def fixture(rank, device, fused=False):
    owned = {2 * rank: 0, 2 * rank + 1: 1}
    sources = {1: [0, 4, 8, 12], 2: [2, 6, 10, 14]}
    kv, ix, local = {}, {}, {}
    for ratio, ids in sources.items():
        mine = [s for s in ids if s in owned]
        local.update({s: i for i, s in enumerate(mine)})
        kv[ratio] = NS(
            kv_buffer=[
                torch.arange(5 * 128, device=device, dtype=torch.int32)
                .to(torch.uint8)
                .reshape(5, 128)
                for _ in mine
            ]
        )
        ix[ratio] = NS(
            page_size=2,
            index_k_with_scale_buffer=None,
            index_k_payload_buffer=[
                _page_bytes((10 // ratio, 2, 1, 8), s + 1, device) for s in ids
            ],
            index_k_scale_buffer=[
                _page_bytes((10 // ratio, 2, 1, 4), s + 2, device) for s in ids
            ],
        )
        if fused:
            # A real fused FP8 indexer row is 128 payload bytes + 8 scale
            # bytes. FULL pages remain aligned for the native copy backend.
            ix[ratio].index_k_with_scale_buffer = [
                _page_bytes((10 // ratio, 2, 1, 136), s + 1, device) for s in ids
            ]
            ix[ratio].index_k_payload_buffer = None
            ix[ratio].index_k_scale_buffer = None
    return NS(
        start_layer=0,
        end_layer=16,
        _swa_global_to_local=owned,
        _low_owner_local_by_source=local,
        sources_by_ratio=sources,
        kv_pools=kv,
        index_pools=ix,
        swa_kv_pool=NS(
            kv_buffer=[
                torch.zeros((5, 128), dtype=torch.uint8, device=device) for _ in owned
            ]
        ),
        layer_mapping=[NS(compress_ratio=0) for _ in range(16)],
    )


@pytest.mark.parametrize("fused", [False, True])
def test_cp8_host_mapping(fused):
    for rank in range(8):
        pool = fixture(rank, "cpu", fused)
        assert a._resolve_deepseek_v4_layer_mappings(pool).swa == {
            2 * rank: 0,
            2 * rank + 1: 1,
        }
        with (
            patch.object(
                a, "DeepSeekV4PagedHostPool", side_effect=lambda **kw: NS(**kw)
            ),
            patch.object(a, "build_pool_entry", side_effect=lambda **kw: NS(**kw)),
            patch.object(
                a, "get_memory", return_value=NS(hicache_mem_layout="layer_first")
            ),
            patch.object(a, "_get_allocator_type", return_value="default"),
        ):
            entries = a._dsv4_low_ratio_entries(pool, 4, 5, 16)
        for entry in entries:
            assert set(entry.layer_mapping.values()) == set(
                range(len(entry.host_pool.device_buffers))
            )
            if any(entry.device_pool is p for p in pool.kv_pools.values()):
                assert set(entry.layer_mapping) <= set(pool._swa_global_to_local)
            else:
                assert len(entry.layer_mapping) == 4


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires HCU/CUDA host registration and copy kernels",
)
@pytest.mark.parametrize("backend", ["kernel", "direct"])
@pytest.mark.parametrize("fused", [False, True])
def test_cp8_device_eviction_reload(backend, fused):
    # Sequential logical ranks use the same device: ownership, empty KV families,
    # packed payload/scale rows, permuted host pages, and relocated device pages.
    for rank in range(8):
        pool = fixture(rank, "cuda", fused)
        with (
            patch.object(
                a, "get_memory", return_value=NS(hicache_mem_layout="layer_first")
            ),
            patch.object(a, "_get_allocator_type", return_value="default"),
        ):
            entries = a._dsv4_low_ratio_entries(pool, 4, 5, 16)
        hi = torch.tensor([8, 9, 10, 11, 4, 5, 6, 7], device="cuda", dtype=torch.int64)
        di = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11], device="cuda", dtype=torch.int64)
        dst = torch.tensor(
            [12, 13, 14, 15, 16, 17, 18, 19], device="cuda", dtype=torch.int64
        )
        for entry in entries:
            h = entry.host_pool
            expected = [b[1:3].clone() for b in h.device_buffers]
            h.backup_from_device_all_layer(entry.device_pool, hi, di, backend)
            torch.cuda.synchronize()
            for b in h.device_buffers:
                b.zero_()
            for layer in range(h.layer_num):
                h.load_to_device_per_layer(entry.device_pool, hi, dst, layer, backend)
            torch.cuda.synchronize()
            for b, e in zip(h.device_buffers, expected):
                assert torch.equal(b[3:5], e), (rank, entry.name, backend)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU/CUDA")
@pytest.mark.parametrize("backend", ["kernel"])
def test_swa_device_eviction_reload(backend):
    from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool

    buffers = [
        torch.randint(0, 255, (5, 128), device="cuda", dtype=torch.uint8)
        for _ in range(5)
    ]
    h = DeepSeekV4PagedHostPool("swa", buffers, 128, 5, 4, page_aligned_only=True)
    hi = torch.arange(4, 12, device="cuda", dtype=torch.int64)
    dst = torch.arange(12, 20, device="cuda", dtype=torch.int64)
    expected = [b[1:3].clone() for b in buffers]
    h.backup_from_device_all_layer(None, hi, hi, backend)
    torch.cuda.synchronize()
    for b in buffers:
        b.zero_()
    for layer in range(5):
        h.load_to_device_per_layer(None, hi, dst, layer, backend)
    torch.cuda.synchronize()
    assert all(torch.equal(b[3:5], e) for b, e in zip(buffers, expected))


def test_host_budget_books_cumulative_pools():
    from sglang.srt.mem_cache.pool_host.base import (
        host_memory_budget_bytes,
        host_memory_budget_scope,
    )

    with host_memory_budget_scope(100):
        assert host_memory_budget_bytes(60) == 100
        assert host_memory_budget_bytes(35) == 40
        assert host_memory_budget_bytes(10) == 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires host registration")
def test_host_budget_rejects_cumulative_paged_pool_allocations():
    from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
    from sglang.srt.mem_cache.pool_host.base import host_memory_budget_scope

    buffers = [torch.zeros((5, 128), dtype=torch.uint8, device="cuda")]
    with host_memory_budget_scope(1000):
        first = DeepSeekV4PagedHostPool("swa", buffers, 128, 5, 4)
        assert first.layer_num == 1
        with pytest.raises(ValueError, match="Not enough host memory"):
            DeepSeekV4PagedHostPool("swa", buffers, 128, 5, 4)


def test_v41_capacity_counts_owner_kv_and_replicated_indexers():
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.mem_cache.hicache_auto_size import _estimate_hicache_bytes

    for rank in range(8):
        f = fixture(rank, "cpu")
        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        pool.__dict__.update(f.__dict__)
        pool.compression_ratios = [0, 1, 2]
        params = NS(
            token_to_kv_pool_allocator=NS(get_kvcache=lambda: pool),
            mtp_draft_device_pools=(),
        )
        expected = 2 * 5 * 128 + 5 * 128 + 4 * 10 * 2 * 12 + 4 * 5 * 2 * 12
        assert _estimate_hicache_bytes(params, None) == expected


def test_capacity_uses_packed_or_sidecar_draft_swa_slots():
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.mem_cache.hicache_auto_size import _estimate_hicache_bytes

    pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
    pool.__dict__.update(fixture(0, "cpu").__dict__)
    pool.compression_ratios = [0, 1, 2]
    pool.swa_kv_pool.size = 20
    packed = NS(
        swa_kv_pool=NS(
            size=8,
            kv_buffer=[torch.zeros((3, 128), dtype=torch.uint8) for _ in range(2)],
        )
    )
    sidecar = NS(
        swa_kv_pool=NS(size=4, kv_buffer=[torch.zeros((2, 128), dtype=torch.uint8)])
    )
    params = NS(
        token_to_kv_pool_allocator=NS(get_kvcache=lambda: pool),
        mtp_draft_device_pools=(),
    )
    target_bytes = _estimate_hicache_bytes(params, None)
    params.mtp_draft_device_pools = (packed,)
    assert _estimate_hicache_bytes(params, None) == target_bytes + 2 * 3 * 128 * 20 // 8
    plan = NS(mode="sidecar", device_pools=(sidecar,))
    assert _estimate_hicache_bytes(params, plan) == target_bytes + 2 * 128 * 20 // 4


def test_owner_with_multiple_sources_uses_local_offsets():
    pool = fixture(0, "cpu")
    pool._swa_global_to_local = {0: 0, 4: 1, 8: 2, 12: 3}
    pool._low_owner_local_by_source = {0: 0, 4: 1, 8: 2, 12: 3}
    pool.kv_pools[1].kv_buffer = [
        torch.zeros((5, 128), dtype=torch.uint8) for _ in range(4)
    ]
    pool.kv_pools[2].kv_buffer = []
    with (
        patch.object(a, "DeepSeekV4PagedHostPool", side_effect=lambda **kw: NS(**kw)),
        patch.object(a, "build_pool_entry", side_effect=lambda **kw: NS(**kw)),
        patch.object(
            a, "get_memory", return_value=NS(hicache_mem_layout="layer_first")
        ),
        patch.object(a, "_get_allocator_type", return_value="default"),
    ):
        entries = a._dsv4_low_ratio_entries(pool, 4, 5, 16)
    owned = [e for e in entries if e.device_pool is pool.kv_pools[1]]
    assert len(owned) == 1
    assert owned[0].layer_mapping == {0: 0, 4: 1, 8: 2, 12: 3}
    assert not any(e.device_pool is pool.kv_pools[2] for e in entries)


def test_assembler_packs_draft_after_owner_local_swa_buffers():
    from unittest.mock import Mock

    pool = fixture(3, "cpu")
    pool.swa_page_size = 4
    pool.swa_kv_pool.page_size = 4
    pool.swa_kv_pool.bytes_per_page_padded = 128
    draft_buffers = [torch.zeros((5, 128), dtype=torch.uint8) for _ in range(2)]
    draft = NS(swa_kv_pool=NS(kv_buffer=draft_buffers))
    params = NS(
        page_size=4,
        mtp_draft_device_pools=(draft,),
        token_to_kv_pool_allocator=NS(swa_attn_allocator=NS(alloc=Mock(), free=Mock())),
        tp_cache_group=None,
        attn_cp_cache_group=None,
        attn_tp_cache_group=None,
        pp_cache_group=None,
    )
    memory = NS(
        hicache_mem_layout="layer_first",
        hicache_write_policy="write_through",
        hicache_io_backend="kernel",
        hicache_host_memory_mode="cache",
    )
    with (
        patch.object(a, "get_memory", return_value=memory),
        patch.object(a, "_get_allocator_type", return_value="default"),
        patch.object(a, "_deepseek_v4_num_host_pages", return_value=(5, 5, 0)),
        patch.object(a, "LogicalHostPool", return_value=NS()),
        patch.object(a, "DeepSeekV4PagedHostPool", side_effect=lambda **kw: NS(**kw)),
        patch.object(a, "build_pool_entry", side_effect=lambda **kw: NS(**kw)),
        patch.object(
            a, "HostPoolGroup", side_effect=lambda entries: NS(entries=entries)
        ),
        patch.object(a, "HybridCacheController", return_value=NS()),
    ):
        group, _ = a.build_deepseek_v4_hicache_stack(
            params=params, kvcache=pool, load_cache_event=None, storage_backend=None
        )
    swa = next(e for e in group.entries if e.name == a.PoolName.SWA)
    assert swa.layer_mapping == {6: 0, 7: 1, 16: 2, 17: 3}
    assert len(swa.host_pool.device_buffers) == 4
    assert all(x is y for x, y in zip(swa.host_pool.device_buffers[2:], draft_buffers))
    assert swa.transfer_layer_id_max == 18


def test_l2_validation_keeps_unverified_modes_blocked():
    from test_cp_cache_layer_split_dspark import TestCpCacheLayerSplitDspark

    t = TestCpCacheLayerSplitDspark()
    t._validate(t._args(enable_hierarchical_cache=True))
    t._validate(
        t._args(
            enable_hierarchical_cache=True,
            enable_decoder_swa_bounded_replay=True,
        )
    )
    for kw in (
        {"hicache_io_backend": "direct"},
        {"hicache_mem_layout": "page_first"},
        {"hicache_storage_backend": "file"},
        {"enable_encoder_swa_bounded_replay": True},
        {"page_size": 255},
    ):
        with pytest.raises(ValueError):
            t._validate(t._args(enable_hierarchical_cache=True, **kw))
