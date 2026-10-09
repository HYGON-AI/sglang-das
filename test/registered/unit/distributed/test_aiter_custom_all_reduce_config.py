from types import SimpleNamespace

import pytest

from sglang.srt.distributed.device_communicators import custom_all_reduce as ca
from sglang.srt.distributed.device_communicators.custom_all_reduce import (
    _aiter_enable_register_for_capturing,
    _aiter_max_size_bytes,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-c-test-cpu")


@pytest.mark.parametrize("tms_cudagraph", [False, True])
@pytest.mark.parametrize("env_value", [None, "0", "1", "true", "false", "invalid"])
@pytest.mark.parametrize("is_hcu", [False, True])
def test_aiter_ipc_registration_matches_pre_8effe135(
    monkeypatch, tms_cudagraph, env_value, is_hcu
):
    monkeypatch.setattr(ca, "_is_hcu", is_hcu)
    if env_value is None:
        monkeypatch.delenv("AITER_AR_ENABLE_REG_CAPTURE", raising=False)
    else:
        monkeypatch.setenv("AITER_AR_ENABLE_REG_CAPTURE", env_value)

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy selection must not inspect GPU architecture")

    monkeypatch.setattr(ca.torch.cuda, "current_device", forbidden)
    monkeypatch.setattr(ca.torch.cuda, "get_device_properties", forbidden)
    assert _aiter_enable_register_for_capturing(tms_cudagraph) is tms_cudagraph


def test_aiter_legacy_default_is_copy_in(monkeypatch):
    monkeypatch.delenv("SGLANG_MEMORY_SAVER_CUDA_GRAPH", raising=False)
    monkeypatch.delenv("AITER_AR_ENABLE_REG_CAPTURE", raising=False)
    assert ca.envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get() is False
    assert (
        _aiter_enable_register_for_capturing(
            ca.envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()
        )
        is False
    )


@pytest.mark.parametrize("env_value,expected", [("0", False), ("1", True)])
def test_aiter_legacy_selector_follows_memory_saver_env(
    monkeypatch, env_value, expected
):
    monkeypatch.setenv("SGLANG_MEMORY_SAVER_CUDA_GRAPH", env_value)
    assert (
        _aiter_enable_register_for_capturing(
            ca.envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()
        )
        is expected
    )


@pytest.mark.parametrize("tms_cudagraph", [False, True])
def test_hcu_glm52_uses_copy_in_for_aiter_graph(monkeypatch, tms_cudagraph):
    from sglang.srt.distributed.device_communicators import custom_all_reduce

    monkeypatch.setattr(custom_all_reduce, "_is_hcu", True)
    monkeypatch.setenv("AITER_AR_ENABLE_REG_CAPTURE", "1")
    glm52 = SimpleNamespace(
        model_type="glm_moe_dsa",
        head_dim=64,
        index_topk_freq=4,
        index_skip_topk_offset=3,
    )
    glm51 = SimpleNamespace(model_type="glm_moe_dsa", head_dim=64)
    other = SimpleNamespace(
        model_type="other", index_topk_freq=4, index_skip_topk_offset=3
    )

    assert not _aiter_enable_register_for_capturing(tms_cudagraph, glm52)
    assert _aiter_enable_register_for_capturing(tms_cudagraph, glm51) is tms_cudagraph
    assert _aiter_enable_register_for_capturing(tms_cudagraph, other) is tms_cudagraph


@pytest.mark.parametrize("tms_cudagraph", [False, True])
def test_hcu_deepseek_v3_uses_copy_in_for_aiter_graph(monkeypatch, tms_cudagraph):
    from sglang.srt.distributed.device_communicators import custom_all_reduce

    monkeypatch.setattr(custom_all_reduce, "_is_hcu", True)
    monkeypatch.setenv("AITER_AR_ENABLE_REG_CAPTURE", "1")
    deepseek_v3 = SimpleNamespace(model_type="deepseek_v3")
    deepseek_v32 = SimpleNamespace(model_type="deepseek_v32")

    assert not _aiter_enable_register_for_capturing(tms_cudagraph, deepseek_v3)
    assert (
        _aiter_enable_register_for_capturing(tms_cudagraph, deepseek_v32)
        is tms_cudagraph
    )


@pytest.mark.parametrize("tms_cudagraph", [False, True])
def test_non_hcu_deepseek_v3_keeps_legacy_opt_in(monkeypatch, tms_cudagraph):
    from sglang.srt.distributed.device_communicators import custom_all_reduce

    monkeypatch.setattr(custom_all_reduce, "_is_hcu", False)
    monkeypatch.setenv("AITER_AR_ENABLE_REG_CAPTURE", "1")
    deepseek_v3 = SimpleNamespace(model_type="deepseek_v3")

    assert (
        _aiter_enable_register_for_capturing(tms_cudagraph, deepseek_v3)
        is tms_cudagraph
    )


def test_aiter_max_size_bytes_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AITER_AR_MAX_SIZE_MB", raising=False)
    assert _aiter_max_size_bytes() is None


def test_aiter_max_size_bytes_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AITER_AR_MAX_SIZE_MB", "64")
    assert _aiter_max_size_bytes() == 64 * 1024 * 1024


@pytest.mark.parametrize("value", ["0", "-1", "bad"])
def test_aiter_max_size_bytes_rejects_invalid(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("AITER_AR_MAX_SIZE_MB", value)
    with pytest.raises(ValueError):
        _aiter_max_size_bytes()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
