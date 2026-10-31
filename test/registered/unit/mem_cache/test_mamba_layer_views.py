from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache import memory_pool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_cached_layer_views_share_storage_and_follow_slot_lifecycle(monkeypatch):
    monkeypatch.setattr(memory_pool, "_cpu_has_amx_support", False)
    params = SimpleNamespace(
        shape=SimpleNamespace(conv=[(4, 3)], temporal=(2, 4, 4)),
        dtype=SimpleNamespace(conv=torch.float32, temporal=torch.float32),
        is_kda=False,
        use_hcu_kda=False,
    )
    pool = memory_pool.MambaPool(
        size=4,
        spec_state_size=4,
        cache_params=params,
        mamba_layer_ids=[0, 1, 2],
        device="cpu",
    )

    layer = pool.mamba2_layer_cache(1)
    assert layer is pool.mamba2_layer_cache(1)
    assert layer.temporal._base is None
    assert layer.conv[0]._base is None
    assert layer.temporal.data_ptr() == pool.mamba_cache.temporal[1].data_ptr()
    assert layer.conv[0].data_ptr() == pool.mamba_cache.conv[0][1].data_ptr()

    layer.temporal[1] = 3
    layer.conv[0][1] = 5
    assert torch.all(pool.mamba_cache.temporal[1, 1] == 3)
    assert torch.all(pool.mamba_cache.conv[0][1, 1] == 5)
    assert torch.all(pool.mamba_cache.temporal[0] == 0)

    pool.copy_from(torch.tensor([1]), torch.tensor([2]))
    assert torch.all(layer.temporal[2] == 3)
    assert torch.all(layer.conv[0][2] == 5)
    pool.clear_slots(torch.tensor([2]))
    assert torch.all(layer.temporal[2] == 0)
    assert torch.all(layer.conv[0][2] == 0)


def test_speculative_and_replay_layer_views_preserve_aliases_and_none():
    full = memory_pool.MambaPool.SpeculativeState(
        conv=[torch.zeros(3, 4, 2)],
        temporal=torch.zeros(3, 4, 2, 2),
        intermediate_ssm=None,
        intermediate_conv_window=[torch.zeros(3, 4, 2, 2)],
        replayssm_d=torch.zeros(3, 4, 2, 2),
        replayssm_k=None,
        replayssm_g=torch.zeros(3, 4, 2),
    )
    layer = full.at_layer_idx(2)
    assert layer.intermediate_ssm is None
    assert layer.replayssm_k is None
    for view, parent in (
        (layer.conv[0], full.conv[0]),
        (layer.temporal, full.temporal),
        (layer.intermediate_conv_window[0], full.intermediate_conv_window[0]),
        (layer.replayssm_d, full.replayssm_d),
        (layer.replayssm_g, full.replayssm_g),
    ):
        assert view._base is None
        assert view.data_ptr() == parent[2].data_ptr()
        view.fill_(7)
        assert torch.all(parent[2] == 7)
        assert torch.all(parent[:2] == 0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
