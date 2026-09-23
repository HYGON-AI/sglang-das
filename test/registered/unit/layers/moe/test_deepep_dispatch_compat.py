from unittest.mock import Mock

import torch

from sglang.srt.layers.moe.token_dispatcher import deepep
from sglang.srt.layers.moe.utils import MoeRunnerBackend


def _inputs():
    return (
        torch.empty(2, 8),
        torch.zeros(2, 2, dtype=torch.int64),
        torch.ones(2, 2),
    )


def test_vendor_deepep_quant_type_fp8(monkeypatch):
    monkeypatch.setattr(deepep, "_DEEPEP_LL_DISPATCH_USES_QUANT_TYPE", True)
    buffer = Mock()
    buffer.low_latency_dispatch.return_value = (None,) * 5
    hidden, ids, weights = _inputs()

    deepep._low_latency_dispatch_compat(
        buffer,
        hidden,
        ids,
        weights,
        16,
        128,
        use_fp8=True,
        use_nvfp4=False,
        input_global_scale=None,
        round_scale=False,
        use_ue8m0=False,
        async_finish=True,
        return_recv_hook=False,
    )

    kwargs = buffer.low_latency_dispatch.call_args.kwargs
    assert kwargs["topk_weight"] is weights
    assert kwargs["quant_type"] == 2
    assert kwargs["quant_group_size"] == 0
    assert kwargs["fp8_round_scale"] is False


def test_vendor_deepep_quant_type_bf16(monkeypatch):
    monkeypatch.setattr(deepep, "_DEEPEP_LL_DISPATCH_USES_QUANT_TYPE", True)
    buffer = Mock()
    buffer.low_latency_dispatch.return_value = (None,) * 5
    hidden, ids, weights = _inputs()

    deepep._low_latency_dispatch_compat(
        buffer,
        hidden,
        ids,
        weights,
        16,
        128,
        use_fp8=False,
        use_nvfp4=False,
        input_global_scale=None,
        round_scale=False,
        use_ue8m0=False,
        async_finish=False,
        return_recv_hook=True,
    )

    assert buffer.low_latency_dispatch.call_args.kwargs["quant_type"] == 0


def test_installed_hcu_deepgemm_does_not_change_aiter_dispatch(monkeypatch):
    monkeypatch.setattr(deepep.deep_gemm_wrapper, "ENABLE_HCU_DEEPGEMM", True)
    monkeypatch.setattr(deepep.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", False)
    monkeypatch.setattr(
        deepep, "get_moe_runner_backend", lambda: MoeRunnerBackend.AITER
    )

    assert not deepep._hcu_deepgemm_is_selected()
    assert not deepep._deepgemm_is_selected()


def test_hcu_deepgemm_dispatch_requires_explicit_runner(monkeypatch):
    monkeypatch.setattr(deepep.deep_gemm_wrapper, "ENABLE_HCU_DEEPGEMM", True)
    monkeypatch.setattr(deepep.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", False)
    monkeypatch.setattr(
        deepep, "get_moe_runner_backend", lambda: MoeRunnerBackend.DEEP_GEMM
    )

    assert deepep._hcu_deepgemm_is_selected()
    assert deepep._deepgemm_is_selected()
