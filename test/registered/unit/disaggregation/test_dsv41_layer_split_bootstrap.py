from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


def manager():
    obj = SimpleNamespace(
        prefill_info_table={},
        kv_args=SimpleNamespace(page_size=256, engine_rank=0),
        kv_cache_layout=None,
        kv_cache_dtype_str="fp8_e4m3",
        dsv41_spec_layout={"kv_layer_ids": [20, 20, 2, 8, 14, 2, 8, 14]},
        attn_tp_size=1, attn_cp_size=1, attn_cp_rank=0,
        pp_size=1, pp_rank=0, dcp_size=1,
        is_mla_backend=True, is_hybrid_mla_backend=False,
        enable_all_cp_ranks_for_transfer=False,
    )
    obj._resolve_rank_mapping = lambda info: CommonKVManager._resolve_rank_mapping(obj, info)
    return obj


def fetch(obj, **changes):
    data = dict(
        attn_tp_size=1, attn_cp_size=8, dp_size=1, pp_size=1,
        page_size=256, kv_cache_dtype="fp8_e4m3", follow_bootstrap_room=True,
        cp_cache_layer_split=True, dsv41_spec_layout=obj.dsv41_spec_layout.copy(),
    )
    data.update(changes)
    with patch("sglang.srt.disaggregation.common.conn.requests.get",
               return_value=Mock(status_code=200, json=lambda: data)):
        return CommonKVManager.try_ensure_parallel_info(obj, "prefill:8998")


def test_layer_split_routes_all_eight_owners():
    obj = manager()
    assert fetch(obj)
    info = obj.prefill_info_table["prefill:8998"]
    assert info.target_cp_ranks == list(range(8))
    assert info.required_prefill_response_num == 8


def test_non_layer_split_cp_remains_rejected():
    with pytest.raises(RuntimeError, match="CP=1"):
        fetch(manager(), cp_cache_layer_split=False)


def test_decode_cp_remains_rejected():
    obj = manager()
    obj.attn_cp_size = 2
    with pytest.raises(RuntimeError, match="CP=1"):
        fetch(obj)


def test_layout_mismatch_remains_rejected():
    with pytest.raises(RuntimeError, match="layout mismatch"):
        fetch(manager(), dsv41_spec_layout={"kv_layer_ids": [2]})


def test_existing_cp1_mla_path_preserved():
    obj = manager()
    assert fetch(obj, attn_cp_size=1, attn_tp_size=8, cp_cache_layer_split=False)
