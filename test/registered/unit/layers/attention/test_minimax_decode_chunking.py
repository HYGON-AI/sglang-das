import pytest

from sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx import (
    _select_decode_score_num_kv_chunks,
)


@pytest.mark.parametrize(
    "batch_size,num_heads,target_grid,max_chunks,expected",
    [
        (1, 1, 4096, 256, 256),
        (7, 1, 4096, 256, 256),
        (7, 1, 4096, 128, 128),
        (7, 1, 1024, 256, 128),
        (8, 4, 256, 16, 8),
        (64, 8, 64, 256, 1),
    ],
)
def test_select_decode_score_num_kv_chunks(
    batch_size, num_heads, target_grid, max_chunks, expected
):
    assert (
        _select_decode_score_num_kv_chunks(
            batch_size,
            num_heads,
            target_grid=target_grid,
            max_chunks=max_chunks,
        )
        == expected
    )


@pytest.mark.parametrize(
    "batch_size,num_heads,target_grid,max_chunks",
    [(0, 1, 4096, 256), (1, 0, 4096, 256), (1, 1, 0, 256), (1, 1, 4096, 0)],
)
def test_select_decode_score_num_kv_chunks_rejects_invalid_values(
    batch_size, num_heads, target_grid, max_chunks
):
    with pytest.raises(ValueError):
        _select_decode_score_num_kv_chunks(
            batch_size,
            num_heads,
            target_grid=target_grid,
            max_chunks=max_chunks,
        )
