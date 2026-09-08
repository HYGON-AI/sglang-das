import unittest
from types import SimpleNamespace
from typing import Dict, Optional

from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


_SOURCE_KEYS = (
    "l1_device",
    "l3_mooncake_memory",
    "l4_mooncake_dfs",
    "l4_mooncake_local_disk",
)


def _make_req(
    prompt_tokens: int,
    cached_tokens: int,
    *,
    device: int = 0,
    host: int = 0,
    storage: int = 0,
    storage_source: Optional[str] = None,
    source_counts: Optional[Dict[str, int]] = None,
):
    return SimpleNamespace(
        origin_input_ids=[0] * prompt_tokens,
        cached_tokens=cached_tokens,
        cached_tokens_device=device,
        cached_tokens_host=host,
        cached_tokens_storage=storage,
        cached_tokens_storage_source=storage_source,
        cached_tokens_by_source=source_counts or {source: 0 for source in _SOURCE_KEYS},
    )


class TestCacheHitRates(unittest.TestCase):
    def setUp(self):
        self.streamer = SchedulerOutputStreamer.__new__(SchedulerOutputStreamer)

    def test_storage_hit_is_not_attributed_to_l1(self):
        req = _make_req(
            64176,
            64000,
            storage=64000,
            storage_source="mooncake_dfs",
        )

        rates = self.streamer.get_cache_hit_rates(req)

        self.assertEqual(rates["l1_device_tokens"], 0)
        self.assertEqual(rates["l4_mooncake_dfs_tokens"], 64000)
        self.assertEqual(rates["uncached_tokens"], 176)
        self.assertAlmostEqual(rates["overall_hit_rate"], 64000 / 64176)

    def test_original_66894_sample_is_reconciled_to_storage(self):
        req = _make_req(
            66894,
            66816,
            storage=66816,
            storage_source="mooncake_memory",
        )

        rates = self.streamer.get_cache_hit_rates(req)

        self.assertEqual(rates["l1_device_tokens"], 0)
        self.assertEqual(rates["l3_mooncake_memory_tokens"], 66816)
        self.assertEqual(rates["l4_mooncake_dfs_tokens"], 0)
        self.assertEqual(rates["uncached_tokens"], 78)
        self.assertAlmostEqual(rates["overall_hit_rate"], 66816 / 66894)

    def test_device_and_dfs_hits_are_accounted_separately(self):
        req = _make_req(
            68644,
            68608,
            device=58880,
            storage=9728,
            storage_source="mooncake_dfs",
            source_counts={
                "l1_device": 58880,
                "l3_mooncake_memory": 0,
                "l4_mooncake_dfs": 9728,
                "l4_mooncake_local_disk": 0,
            },
        )

        rates = self.streamer.get_cache_hit_rates(req)

        self.assertEqual(rates["l1_device_tokens"], 58880)
        self.assertEqual(rates["l4_mooncake_dfs_tokens"], 9728)
        self.assertEqual(rates["uncached_tokens"], 36)
        self.assertAlmostEqual(rates["overall_hit_rate"], 68608 / 68644)

    def test_all_three_cache_tiers_can_hit_in_one_request(self):
        req = _make_req(
            100,
            96,
            device=32,
            storage=64,
            storage_source="mooncake_mixed",
            source_counts={
                "l1_device": 32,
                "l3_mooncake_memory": 32,
                "l4_mooncake_dfs": 32,
                "l4_mooncake_local_disk": 0,
            },
        )

        rates = self.streamer.get_cache_hit_rates(req)

        self.assertEqual(rates["l1_device_tokens"], 32)
        self.assertEqual(rates["l3_mooncake_memory_tokens"], 32)
        self.assertEqual(rates["l4_mooncake_dfs_tokens"], 32)
        self.assertEqual(rates["l4_mooncake_local_disk_tokens"], 0)
        self.assertEqual(rates["uncached_tokens"], 4)
        self.assertAlmostEqual(rates["overall_hit_rate"], 0.96)

    def test_aggregate_only_legacy_path_still_uses_l1(self):
        req = _make_req(10, 7)

        rates = self.streamer.get_cache_hit_rates(req)

        self.assertEqual(rates["l1_device_tokens"], 7)
        self.assertEqual(rates["uncached_tokens"], 3)
        self.assertAlmostEqual(rates["overall_hit_rate"], 0.7)

    def test_overlapping_source_counts_are_rejected(self):
        req = _make_req(
            64176,
            64000,
            source_counts={
                "l1_device": 64000,
                "l3_mooncake_memory": 0,
                "l4_mooncake_dfs": 64000,
                "l4_mooncake_local_disk": 0,
            },
        )

        self.assertIsNone(self.streamer.get_cache_hit_rates(req))

    def test_overall_rate_uses_aggregate_when_host_is_not_exposed_as_a_tier(self):
        req = _make_req(10, 10, host=10)

        rates = self.streamer.get_cache_hit_rates(req)

        self.assertEqual(rates["uncached_tokens"], 0)
        self.assertEqual(rates["overall_hit_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
