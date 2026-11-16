"""Regression coverage for runtime Mooncake storage attachment."""

from types import SimpleNamespace

from sglang.srt.mem_cache.unified_cache.storage_attachment import StorageAttachment
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_runtime_mooncake_attach_injects_cli_health_config(monkeypatch):
    controller = SimpleNamespace(
        storage_backend_type=None,
        mem_pool_host=SimpleNamespace(entries=[]),
    )
    attached = []
    controller.attach_storage_backend = lambda **kwargs: attached.append(kwargs)
    cache = SimpleNamespace(
        cache_controller=controller,
        enable_storage=False,
        _enable_metrics_flag=False,
        extra_metric_labels=None,
    )
    attachment = StorageAttachment(cache)
    monkeypatch.setattr(attachment, "apply_runtime_config", lambda **kwargs: None)
    monkeypatch.setattr(
        "sglang.srt.mem_cache.unified_cache.storage_attachment."
        "HybridCacheController.parse_storage_backend_extra_config",
        lambda raw: (
            {"master_server_address": "127.0.0.1:50051"},
            256,
            1.0,
            0.25,
            False,
        ),
    )
    server_args = SimpleNamespace(
        mooncake_dfs_health_probe_interval_s=13,
        mooncake_dfs_health_failure_threshold=3,
        mooncake_dfs_health_success_threshold=4,
        mooncake_dfs_health_probe_timeout_s=7,
    )
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_server_args", lambda: server_args
    )

    ok, _ = attachment.attach("mooncake")

    assert ok
    assert attached[0]["storage_backend_extra_config"][
        "_mooncake_dfs_health_config"
    ] == {
        "probe_interval_s": 13,
        "failure_threshold": 3,
        "success_threshold": 4,
        "probe_timeout_s": 7,
    }
