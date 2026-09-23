from types import SimpleNamespace

import torch

import sglang.srt.distributed.parallel_state as parallel_state
from sglang.srt.distributed.parallel_state import GroupCoordinator


class _FakeAiterCollectives:
    disabled = False
    _IS_CAPTURING = True

    def __init__(self) -> None:
        self.reduce_scatter_calls = []
        self.all_gather_calls = []

    @staticmethod
    def should_custom_ar(_input: torch.Tensor) -> bool:
        return True

    @staticmethod
    def should_custom_ag(_input: torch.Tensor) -> bool:
        return True

    def reduce_scatter(
        self, input: torch.Tensor, output: torch.Tensor, registered: bool
    ) -> None:
        self.reduce_scatter_calls.append(registered)
        output.fill_(7)

    def all_gather_unreg(
        self, input: torch.Tensor, out: torch.Tensor, dim: int
    ) -> None:
        self.all_gather_calls.append(dim)
        out.fill_(9)


def _patch_graph_warmup(monkeypatch) -> None:
    monkeypatch.setattr(parallel_state, "is_hip", lambda: True)
    monkeypatch.setattr(parallel_state, "_is_hcu", False)
    monkeypatch.setattr(
        parallel_state.torch.cuda, "is_current_stream_capturing", lambda: False
    )
    monkeypatch.setattr(parallel_state, "is_in_tc_piecewise_cuda_graph", lambda: False)
    monkeypatch.setattr(
        parallel_state.envs.SGLANG_DP_USE_REDUCE_SCATTER, "get", lambda: True
    )
    monkeypatch.setattr(parallel_state.envs.SGLANG_USE_AITER_AG, "get", lambda: True)
    monkeypatch.setenv("SGLANG_AITER_AR_REAL_GRAPH_WARMUP", "1")


def test_aiter_reduce_scatter_graph_warmup_runs_real_collective(monkeypatch) -> None:
    _patch_graph_warmup(monkeypatch)
    ca_comm = _FakeAiterCollectives()
    group = SimpleNamespace(
        ca_comm=ca_comm,
        world_size=2,
        _has_aiter_custom_reduce_scatter=lambda: True,
    )
    input = torch.arange(8, dtype=torch.bfloat16).reshape(4, 2)
    output = torch.zeros((2, 2), dtype=torch.bfloat16)

    assert GroupCoordinator._maybe_aiter_reduce_scatter(group, output, input)
    assert ca_comm.reduce_scatter_calls == [False]
    assert torch.equal(output, torch.full_like(output, 7))


def test_aiter_all_gather_graph_warmup_runs_real_collective(monkeypatch) -> None:
    _patch_graph_warmup(monkeypatch)
    ca_comm = _FakeAiterCollectives()
    group = SimpleNamespace(
        ca_comm=ca_comm,
        world_size=2,
        _has_aiter_custom_all_gather=lambda: True,
    )
    input = torch.arange(4, dtype=torch.bfloat16).reshape(2, 2)
    output = torch.zeros((4, 2), dtype=torch.bfloat16)

    GroupCoordinator._all_gather_into_tensor(group, output, input)
    assert ca_comm.all_gather_calls == [0]
    assert torch.equal(output, torch.full_like(output, 9))
