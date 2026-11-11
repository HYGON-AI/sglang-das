import json
import os
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    DEFAULT_TENANT_ID,
    MooncakeStoreConfig,
    _get_mooncake_client_http_setup_kwargs,
    _get_service_endpoint,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fake_mooncake_modules(fake_store_cls):
    mooncake = types.ModuleType("mooncake")
    mooncake_store = types.ModuleType("mooncake.store")
    mooncake_store.MooncakeDistributedStore = fake_store_cls

    class ReplicateConfig:
        pass

    mooncake_store.ReplicateConfig = ReplicateConfig
    return {
        "mooncake": mooncake,
        "mooncake.store": mooncake_store,
    }


class FakeMooncakeDistributedStore:
    instances = []

    def __init__(self):
        self.setup_calls = []
        self.objects = {}
        type(self).instances.append(self)

    def setup(self, *args, **kwargs):
        self.setup_calls.append((args, kwargs))
        return 0

    def register_buffer(self, *args, **kwargs):
        return 0

    def disable_dfs_prefetch(self):
        return True

    def put(self, key, value, *args):
        self.objects[key] = value
        return 0

    def is_exist(self, key):
        return 1 if key in self.objects else 0

    def get(self, key):
        return self.objects.get(key)


class OldMooncakeDistributedStore(FakeMooncakeDistributedStore):
    instances = []

    def setup(self, *args, **kwargs):
        if "tenant_id" in kwargs:
            raise TypeError("tenant_id is an invalid keyword argument")
        return super().setup(*args, **kwargs)


def _make_storage_config(tenant_id=DEFAULT_TENANT_ID):
    extra_config = {
        "master_server_address": "127.0.0.1:50051",
        "check_server": False,
        "global_segment_size": 1024 * 1024,
    }
    if tenant_id is not None:
        extra_config["tenant_id"] = tenant_id

    return SimpleNamespace(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="test",
        tp_lcm_size=None,
        should_split_heads=False,
        extra_config=extra_config,
    )


class TestMooncakeTenantConfig(unittest.TestCase):
    def test_load_from_extra_config_normalizes_tenant(self):
        cfg = MooncakeStoreConfig.load_from_extra_config(
            {
                "master_server_address": "127.0.0.1:50051",
                "tenant_id": "  tenant-extra  ",
            }
        )
        self.assertEqual(cfg.tenant_id, "tenant-extra")

        cfg = MooncakeStoreConfig.load_from_extra_config(
            {
                "master_server_address": "127.0.0.1:50051",
                "tenant_id": "   ",
            }
        )
        self.assertEqual(cfg.tenant_id, DEFAULT_TENANT_ID)

        cfg = MooncakeStoreConfig.load_from_extra_config(
            {
                "master_server_address": "127.0.0.1:50051",
                "tenant_id": None,
            }
        )
        self.assertEqual(cfg.tenant_id, DEFAULT_TENANT_ID)

    def test_load_from_env_reads_mooncake_tenant_id(self):
        with (
            envs.MOONCAKE_MASTER.override("127.0.0.1:50051"),
            envs.MOONCAKE_TENANT_ID.override("tenant-env"),
        ):
            cfg = MooncakeStoreConfig.load_from_env()

        self.assertEqual(cfg.tenant_id, "tenant-env")

    def test_load_from_file_reads_tenant_id(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as config_file:
            json.dump(
                {
                    "master_server_address": "127.0.0.1:50051",
                    "tenant_id": "tenant-file",
                },
                config_file,
            )
            config_file.flush()

            with envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.override(config_file.name):
                cfg = MooncakeStoreConfig.from_file()

        self.assertEqual(cfg.tenant_id, "tenant-file")

    def test_mooncake_store_forwards_non_default_tenant_id(self):
        FakeMooncakeDistributedStore.instances = []
        with patch.dict(
            "sys.modules",
            _fake_mooncake_modules(FakeMooncakeDistributedStore),
        ):
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStore,
            )

            MooncakeStore(_make_storage_config("tenant-a"))

        fake_store = FakeMooncakeDistributedStore.instances[-1]
        self.assertEqual(fake_store.setup_calls[0][1]["tenant_id"], "tenant-a")

    def test_mooncake_store_keeps_default_tenant_backward_compatible(self):
        FakeMooncakeDistributedStore.instances = []
        with patch.dict(
            "sys.modules",
            _fake_mooncake_modules(FakeMooncakeDistributedStore),
        ):
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStore,
            )

            MooncakeStore(_make_storage_config())

        fake_store = FakeMooncakeDistributedStore.instances[-1]
        self.assertNotIn("tenant_id", fake_store.setup_calls[0][1])

    def test_non_default_tenant_requires_new_mooncake(self):
        OldMooncakeDistributedStore.instances = []
        with patch.dict(
            "sys.modules",
            _fake_mooncake_modules(OldMooncakeDistributedStore),
        ):
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStore,
            )

            with self.assertRaisesRegex(RuntimeError, "tenant_id"):
                MooncakeStore(_make_storage_config("tenant-a"))

    def test_embedding_store_forwards_tenant_id(self):
        FakeMooncakeDistributedStore.instances = []
        storage_config = SimpleNamespace(
            extra_config={
                "master_server_address": "127.0.0.1:50051",
                "tenant_id": "tenant-embedding",
            }
        )

        with patch.dict(
            "sys.modules",
            _fake_mooncake_modules(FakeMooncakeDistributedStore),
        ):
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_embedding_store import (
                MooncakeEmbeddingStore,
            )

            MooncakeEmbeddingStore(storage_config)

        fake_store = FakeMooncakeDistributedStore.instances[-1]
        self.assertEqual(fake_store.setup_calls[0][1]["tenant_id"], "tenant-embedding")


class TestMooncakeServiceMetrics(unittest.TestCase):
    def test_independent_tp_services_request_auto_http_ports(self):
        with (
            patch.dict(os.environ),
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank") as get_rank,
        ):
            envs.MOONCAKE_CLIENT_METRICS_PORT_BASE.clear()
            # Four independent TP=2 services all repeat rank 0 and rank 1.
            # Every server must bind port zero itself, without rank offsets.
            for _ in range(4):
                for rank in range(2):
                    get_rank.return_value = rank
                    self.assertEqual(
                        _get_mooncake_client_http_setup_kwargs(True),
                        {"enable_client_http_server": True, "client_http_port": 0},
                    )

    def test_explicit_http_port_base_preserves_rank_offsets(self):
        with (
            envs.MOONCAKE_CLIENT_METRICS_PORT_BASE.override(15000),
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=1),
            patch("torch.distributed.get_world_size", return_value=2),
        ):
            self.assertEqual(
                _get_mooncake_client_http_setup_kwargs(True, dp_rank=1),
                {"enable_client_http_server": True, "client_http_port": 15003},
            )
            self.assertEqual(_get_mooncake_client_http_setup_kwargs(False), {})

    def test_uses_existing_business_endpoint(self):
        self.assertEqual(
            _get_service_endpoint(
                SimpleNamespace(host="10.0.0.20", port=30000, nnodes=1)
            ),
            "10.0.0.20:30000",
        )
        self.assertEqual(
            _get_service_endpoint(
                SimpleNamespace(host="2001:db8::20", port=30000, nnodes=1)
            ),
            "[2001:db8::20]:30000",
        )

    def test_single_node_wildcard_uses_existing_ip_discovery(self):
        with patch(
            "sglang.srt.mem_cache.storage.mooncake_store.mooncake_store.get_local_ip_auto",
            return_value="10.0.0.20",
        ):
            for host in ("0.0.0.0", "::", None):
                self.assertEqual(
                    _get_service_endpoint(
                        SimpleNamespace(host=host, port=30000, nnodes=1)
                    ),
                    "10.0.0.20:30000",
                )

    def test_multiple_nodes_use_leader_ip_and_business_port(self):
        for leader, expected in (
            ("10.0.0.20:5000", "10.0.0.20:30000"),
            ("[2001:db8::20]:5000", "[2001:db8::20]:30000"),
        ):
            args = SimpleNamespace(
                host="0.0.0.0", port=30000, nnodes=2, dist_init_addr=leader
            )
            with patch(
                "sglang.srt.mem_cache.storage.mooncake_store.mooncake_store.get_local_ip_auto",
                side_effect=AssertionError("must not use a remote worker's IP"),
            ):
                self.assertEqual(_get_service_endpoint(args), expected)

    def test_reports_before_warmup_with_client_http_disabled(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        config = _make_storage_config()
        calls = []
        with (
            patch.dict(
                "sys.modules", _fake_mooncake_modules(FakeMooncakeDistributedStore)
            ),
            patch.object(
                FakeMooncakeDistributedStore,
                "set_service_endpoint",
                side_effect=lambda endpoint: calls.append(endpoint) or 0,
                create=True,
            ),
            patch.object(
                MooncakeStore, "warmup", side_effect=lambda: calls.append("warmup")
            ),
            patch(
                "sglang.srt.mem_cache.storage.mooncake_store.mooncake_store.get_server_args",
                return_value=SimpleNamespace(
                    host="10.0.0.20",
                    port=30000,
                    nnodes=1,
                    enable_hierarchical_cache=False,
                    enable_unified_cache_external_linker=True,
                ),
            ),
        ):
            store = MooncakeStore(config, enable_client_http_server=False)

        self.assertEqual(calls, ["10.0.0.20:30000", "warmup"])
        self.assertNotIn("enable_client_http_server", store.store.setup_calls[0][1])

    def test_shared_daemon_endpoint_conflict_warns_and_finishes_setup(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        shared = FakeMooncakeDistributedStore()
        registered = []

        def set_endpoint(endpoint):
            if registered and registered[0] != endpoint:
                return -1
            registered[:] = [endpoint]
            return 0

        with (
            patch.dict("sys.modules", _fake_mooncake_modules(lambda: shared)),
            patch.object(shared, "set_service_endpoint", set_endpoint, create=True),
            patch.object(MooncakeStore, "warmup") as warmup,
            patch(
                "sglang.srt.mem_cache.storage.mooncake_store.mooncake_store.get_server_args",
                side_effect=[
                    SimpleNamespace(host="10.0.0.20", port=30000, nnodes=1),
                    SimpleNamespace(host="10.0.0.20", port=30001, nnodes=1),
                ],
            ),
        ):
            first = MooncakeStore(_make_storage_config())
            with self.assertLogs(
                "sglang.srt.mem_cache.storage.mooncake_store.mooncake_store",
                level="WARNING",
            ) as logs:
                second = MooncakeStore(_make_storage_config())

        self.assertIs(first.store, second.store)
        self.assertEqual(warmup.call_count, 2)
        self.assertEqual(registered, ["10.0.0.20:30000"])
        self.assertTrue(any("30001" in message for message in logs.output))
        self.assertTrue(any("original endpoint" in message for message in logs.output))


class TestMooncakePrefetchSetup(unittest.TestCase):
    def test_hicache_factory_disables_unused_waiting_queue_arena(self):
        from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory

        backend_class = Mock()
        storage_config, host_pool = object(), object()
        result = StorageBackendFactory._create_builtin_backend(
            "mooncake", backend_class, storage_config, host_pool
        )
        backend_class.assert_called_once_with(
            storage_config, host_pool, enable_dfs_prefetch=False
        )
        self.assertIs(result, backend_class.return_value)

    def test_disabled_prefetch_overrides_client_without_changing_environment(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        with (
            patch.dict(
                "sys.modules", _fake_mooncake_modules(FakeMooncakeDistributedStore)
            ),
            patch.dict(os.environ, {"MC_STORE_DFS_PREFETCH_ARENA_SIZE_BYTES": "4096"}),
        ):
            MooncakeStore(_make_storage_config(), enable_dfs_prefetch=False)
            self.assertEqual(
                os.environ["MC_STORE_DFS_PREFETCH_ARENA_SIZE_BYTES"], "4096"
            )

        fake_store = FakeMooncakeDistributedStore.instances[-1]
        self.assertIs(fake_store.setup_calls[0][1]["enable_dfs_prefetch"], False)

    def test_enabled_or_unspecified_prefetch_preserves_environment_setup(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        for enabled in (True, None):
            with (
                self.subTest(enabled=enabled),
                patch.dict(
                    "sys.modules", _fake_mooncake_modules(FakeMooncakeDistributedStore)
                ),
            ):
                MooncakeStore(_make_storage_config(), enable_dfs_prefetch=enabled)
                self.assertNotIn(
                    "enable_dfs_prefetch",
                    FakeMooncakeDistributedStore.instances[-1].setup_calls[0][1],
                )

    def test_disabled_prefetch_cannot_silently_drop_unsupported_gate(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        class UnsupportedPrefetchStore(FakeMooncakeDistributedStore):
            attempts = 0

            def setup(self, *args, **kwargs):
                type(self).attempts += 1
                if "enable_dfs_prefetch" in kwargs:
                    raise TypeError(
                        "enable_dfs_prefetch is an invalid keyword argument"
                    )
                return super().setup(*args, **kwargs)

        with patch.dict(
            "sys.modules", _fake_mooncake_modules(UnsupportedPrefetchStore)
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot disable DFS prefetch"):
                MooncakeStore(_make_storage_config(), enable_dfs_prefetch=False)
        self.assertEqual(UnsupportedPrefetchStore.attempts, 1)

    def test_prefetch_requires_release_api_before_setup(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        class UnsupportedReleaseStore(FakeMooncakeDistributedStore):
            disable_dfs_prefetch = None

        with patch.dict("sys.modules", _fake_mooncake_modules(UnsupportedReleaseStore)):
            with self.assertRaisesRegex(RuntimeError, "disable_dfs_prefetch"):
                MooncakeStore(_make_storage_config(), enable_dfs_prefetch=True)
        self.assertEqual(UnsupportedReleaseStore.instances[-1].setup_calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
