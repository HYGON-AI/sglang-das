from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.cp_cache_layer_split.deepseek_v4_layout import (
    build_cp_cache_layer_split_deepseek_v4_pool_layout,
)
from sglang.srt.mem_cache.cp_cache_layer_split.deepseek_v4_pool import (
    CpCacheLayerSplitDeepSeekV4TokenToKVPool as Pool,
)
from sglang.srt.mem_cache.cp_cache_layer_split.transfer import (
    match_transfer_entries,
    decode_v4_transfer_metadata,
)


def test_flash0731_layer_coverage():
    ratios = [0, 0] + [4, 128] * 20 + [4]
    layouts = [
        build_cp_cache_layer_split_deepseek_v4_pool_layout(rank, 8, 0, 43, ratios)
        for rank in range(8)
    ]
    assert sum(x.swa_layer_num for x in layouts) == 43
    assert sum(x.c4_layer_num for x in layouts) == 21
    assert sum(x.c128_layer_num for x in layouts) == 20
    assert max(x.swa_layer_num for x in layouts) == 6


def test_descriptors_disambiguate_c4_and_indexer():
    src = [("c4_indexer", 2), ("c4", 4)]
    dst = [("c4", 2), ("c4", 4), ("c4_indexer", 2), ("c4_indexer", 4)]
    assert match_transfer_entries(
        src, dst, [8448, 37376], [37376, 37376, 8448, 8448]
    ) == [(0, 2), (1, 1)]


@pytest.mark.parametrize(
    "dst,sizes", [([], []), ([("c4", 2)], [1]), ([("c4", 2), ("c4", 2)], [2, 2])]
)
def test_transfer_fails_before_wrong_copy(dst, sizes):
    with pytest.raises(RuntimeError):
        match_transfer_entries([("c4", 2)], dst, [2], sizes)


def test_old_wire_without_v4_extension():
    assert decode_v4_transfer_metadata([b""] * 19) == {}


def test_hcu_int8_nonowner_reads_staging():
    pool = Pool.__new__(Pool)
    staging = torch.tensor([[1, 2, 127, 255]], dtype=torch.uint8)
    pool.get_index_k_with_scale_buffer = lambda layer: staging
    assert pool.get_index_k_int8_packed_buffer(42) is staging


def test_hcu_sparse_prefill_page_geometry_on_nonowner():
    pool = Pool.__new__(Pool)
    pool.compression_ratios = [4, 128]
    pool.c4_kv_pool = SimpleNamespace(page_size=64)
    pool.c128_kv_pool = SimpleNamespace(page_size=2)
    assert pool.get_extra_key_page_size(0) == 64
    assert pool.get_extra_key_page_size(1) == 2


def test_sparse_dequant_flat_ids_are_remapped_independently():
    pool = Pool.__new__(Pool)
    pool._swa_remapped_layer_id = 5
    pool.swa_kv_pool = SimpleNamespace(size=1024, page_size=256)
    pool._batch_active_pages = {
        "swa": SimpleNamespace(selected_pages=torch.tensor([1, 3]))
    }
    pool._swa_remapped_indices = torch.tensor([[0, 1]])
    got = pool.remap_flat_token_ids_for_read(
        5, torch.tensor([256, 257, 768, 769, -1]), family="swa"
    )
    torch.testing.assert_close(got, torch.tensor([0, 1, 256, 257, -1]))
    assert pool._swa_remapped_indices.shape == (1, 2)
