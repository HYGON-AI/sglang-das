"""L4 metric outcomes must not double-count failures or drop modern L3 metrics."""

import ast
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


@pytest.fixture
def collector():
    root = next(
        p for p in Path(__file__).resolve().parents if (p / "python/sglang").is_dir()
    )
    path = root / "python/sglang/srt/observability/metrics_collector.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "StorageMetricsCollector"
    )
    # Execute the production collector with real Prometheus metrics, isolated from
    # unrelated model imports and the process-global metric registry.
    registry = CollectorRegistry()
    scope = {
        "_StatLoggerDIMixin": type(
            "MetricBase",
            (),
            {
                "_counter_cls": staticmethod(partial(Counter, registry=registry)),
                "_gauge_cls": staticmethod(partial(Gauge, registry=registry)),
                "_histogram_cls": staticmethod(partial(Histogram, registry=registry)),
            },
        )
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    labels = {"storage_backend": "mooncake_direct", "tp_rank": "0"}
    return SimpleNamespace(
        obj=scope["StorageMetricsCollector"](labels), registry=registry, labels=labels
    )


@pytest.mark.parametrize("operation", ["prefetch", "backup"])
def test_l4_success_failure_and_empty(collector, operation):
    log = getattr(collector.obj, "log_l4_" + operation)
    log("mooncake", 32, 0.25, True)
    log("mooncake", 64, 0.5, False)
    log("mooncake", 0, 0.125, True)
    labels = {**collector.labels, "source": "mooncake"}
    get = collector.registry.get_sample_value
    token_prefix = "prefetched" if operation == "prefetch" else "backuped"
    assert get(f"sglang:l4_{token_prefix}_tokens_total", labels) == 32
    assert get(f"sglang:{token_prefix}_tokens_total", collector.labels) == 32
    assert (
        get(
            f"sglang:l4_{operation}_duration_seconds_count",
            {**labels, "result": "success"},
        )
        == 2
    )
    assert (
        get(
            f"sglang:l4_{operation}_duration_seconds_sum",
            {**labels, "result": "success"},
        )
        == 0.375
    )
    assert (
        get(
            f"sglang:l4_{operation}_duration_seconds_count",
            {**labels, "result": "failure"},
        )
        == 1
    )


def test_modern_l3_metrics_preserved(collector):
    c = collector.obj
    c.log_storage_prefetch_hit_tokens(128)
    c.log_storage_prefetch_unfulfilled_tokens(16, "anchor_lost")
    c.log_storage_prefetch_deferred_tokens(8, "host_capacity")
    get = collector.registry.get_sample_value
    assert get("sglang:storage_prefetch_hit_tokens_total", collector.labels) == 128
    assert (
        get(
            "sglang:storage_prefetch_unfulfilled_tokens_total",
            {**collector.labels, "reason": "anchor_lost"},
        )
        == 16
    )
    assert (
        get(
            "sglang:storage_prefetch_deferred_tokens_total",
            {**collector.labels, "reason": "host_capacity"},
        )
        == 8
    )
