"""CPU tests for serving-time Triton compilation diagnostics."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.utils import triton_load_watch as watch
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-c-test-cpu")


@pytest.fixture(autouse=True)
def reset_watch(monkeypatch):
    monkeypatch.setattr(watch, "_serving_started", True)
    monkeypatch.setattr(watch, "_prev_compile_listener", None)
    monkeypatch.setattr(watch, "_warned_compile_kernels", set())
    monkeypatch.setenv("SGLANG_TRITON_SLOW_COMPILE_THRESHOLD_SECS", "1")


def compile_kernel(name="sparse_decode", seconds=2.0, cache_hit=False):
    watch._on_compilation(
        src=SimpleNamespace(name=name),
        metadata=None,
        metadata_group=None,
        times=SimpleNamespace(total=seconds * 1e6),
        cache_hit=cache_hit,
    )


def test_slow_compile_warns_once_per_kernel(caplog):
    with caplog.at_level(logging.WARNING, logger=watch.__name__):
        compile_kernel()
        compile_kernel(seconds=3.0)
        compile_kernel(name="other_kernel")
    assert len(caplog.records) == 2
    assert "sparse_decode" in caplog.records[0].message
    assert "suppressed in this worker" in caplog.records[0].message
    assert "other_kernel" in caplog.records[1].message


@pytest.mark.parametrize("seconds,expected", [(0.5, 2.0), (1.0, 1.0)])
def test_threshold_is_preserved(caplog, seconds, expected):
    compile_kernel(seconds=seconds)
    compile_kernel()
    assert len(caplog.records) == 1
    assert f"{expected:.2f} s" in caplog.text


@pytest.mark.parametrize("cache_hit,serving", [(True, True), (False, False)])
def test_ignored_compilation_does_not_suppress_later_warning(
    monkeypatch, caplog, cache_hit, serving
):
    monkeypatch.setattr(watch, "_serving_started", serving)
    compile_kernel(cache_hit=cache_hit)
    assert not caplog.records
    monkeypatch.setattr(watch, "_serving_started", True)
    compile_kernel()
    assert len(caplog.records) == 1


def test_previous_listener_receives_every_event(monkeypatch):
    listener = Mock()
    monkeypatch.setattr(watch, "_prev_compile_listener", listener)
    compile_kernel()
    compile_kernel()
    compile_kernel(cache_hit=True)
    assert listener.call_count == 3
    assert listener.call_args.kwargs["cache_hit"] is True


@pytest.mark.parametrize("crash", [False, True])
def test_kernel_load_checks_are_not_suppressed(monkeypatch, caplog, crash):
    monkeypatch.setattr(watch.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(watch.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(watch, "get_available_gpu_memory", lambda *a, **kw: 0.25)
    monkeypatch.setenv("SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB", "1")
    monkeypatch.setenv("SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY", "1" if crash else "0")
    compile_kernel()
    caplog.clear()
    for _ in range(2):
        if crash:
            with pytest.raises(RuntimeError, match="device-loaded after serving"):
                watch._on_kernel_load(None, None, "sparse_decode", None, None)
        else:
            watch._on_kernel_load(None, None, "sparse_decode", None, None)
    assert len(caplog.records) == (0 if crash else 2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
