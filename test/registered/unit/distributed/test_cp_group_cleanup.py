from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.distributed import bootstrap, parallel_state


@pytest.mark.parametrize("dp_group", ["tp", "attn_tp", "dedicated"])
def test_model_group_cleanup_destroys_aliases_once(monkeypatch, dp_group):
    groups = {
        name: SimpleNamespace(destroy=Mock())
        for name in ("tp", "attn_tp", "dedicated", "overlap")
    }
    bindings = {
        "_TP": groups["tp"],
        "_ATTN_TP": groups["attn_tp"],
        "_SHARED_EXPERTS_TP": groups["tp"],
        "_PP": None,
        "_DCP": None,
        "_MOE_EP": groups["tp"],
        "_MOE_TP": groups["tp"],
        "_MOE_DP": groups["tp"],
        "_ATTN_CP": groups["tp"],
        "_ATTN_CP_OVERLAP": groups["overlap"],
        "_ATTN_DP_TP": groups[dp_group],
        "_PDMUX_PREFILL_TP_GROUP": None,
    }
    for name, value in bindings.items():
        monkeypatch.setattr(parallel_state, name, value)
    monkeypatch.setattr(
        parallel_state,
        "get_parallel",
        lambda: SimpleNamespace(clear_stamp=lambda: None),
    )
    monkeypatch.setattr(parallel_state, "get_global_dwdp_manager", lambda: None)
    monkeypatch.setattr(bootstrap, "reset_parallel_initialised", lambda: None)
    parallel_state.destroy_model_parallel()
    for name, group in groups.items():
        assert group.destroy.call_count == (
            1 if name != "dedicated" or dp_group == "dedicated" else 0
        )
    assert all(getattr(parallel_state, name) is None for name in bindings)
