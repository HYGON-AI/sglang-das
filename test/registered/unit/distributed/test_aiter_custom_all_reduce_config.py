from types import SimpleNamespace

import pytest

from sglang.srt.distributed.device_communicators.custom_all_reduce import (
    _aiter_enable_register_for_capturing,
    _aiter_max_size_bytes,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-c-test-cpu")


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [(None, True), ("1", True), ("true", True), ("0", False), ("false", False)],
)
def test_aiter_register_capture_honors_environment(
    monkeypatch: pytest.MonkeyPatch, env_value: str | None, expected: bool
) -> None:
    if env_value is None:
        monkeypatch.delenv("AITER_AR_ENABLE_REG_CAPTURE", raising=False)
    else:
        monkeypatch.setenv("AITER_AR_ENABLE_REG_CAPTURE", env_value)
    assert _aiter_enable_register_for_capturing(False) is expected


def test_aiter_register_capture_disabled_by_memory_saver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AITER_AR_ENABLE_REG_CAPTURE", "1")
    assert not _aiter_enable_register_for_capturing(True)


def test_hcu_glm52_uses_copy_in_for_aiter_graph(monkeypatch: pytest.MonkeyPatch) -> None:
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

    assert not _aiter_enable_register_for_capturing(False, glm52)
    assert _aiter_enable_register_for_capturing(False, glm51)
    assert _aiter_enable_register_for_capturing(False, other)


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
