import threading
import types
import unittest
from queue import Queue
from unittest.mock import Mock, call

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
    ReadPlanLoadCounter,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMooncakeDirectLinkerReadPlan(CustomTestCase):
    def test_read_plan_failure_is_deferred_until_after_forward(self):
        counter = ReadPlanLoadCounter(num_layers=2)
        index = counter.update_producer()
        counter.set_consumer(index)
        plan = Mock()
        plan.wait.side_effect = RuntimeError("range get failed: rc=-707")
        counter.bind(index, plan)

        with self.assertLogs(
            "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker",
            level="ERROR",
        ) as logs:
            counter.wait_until(0)
            counter.wait_until(1)

        self.assertEqual(len(logs.output), 1)
        self.assertNotIn(index, counter.plans)
        self.assertNotIn(index, counter.reported)

    def test_successful_read_plan_load_reports_success(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.read_plan_enabled = True
        linker.host_prefetch_enabled = False
        linker.tp_rank = 0
        linker.load_with_read_plan = Mock()

        success = linker.load_layer_wise(7, [])

        self.assertIs(success, True)
        linker.load_with_read_plan.assert_called_once_with(7, [])

    def _read_plan_linker(self, *, prefetch):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.read_plan_enabled = True
        linker.host_prefetch_enabled = prefetch
        linker.read_plan_reuse_ranges = False
        linker.enable_page_wise_load = False
        linker.tp_rank = 0
        linker.num_layers = 1
        linker.pools = {}
        linker.storage = types.SimpleNamespace(store=Mock())
        linker._prepare_read_plan_layouts = Mock(return_value=[])
        linker._prepare_expanded_load = Mock(
            side_effect=AssertionError("unexpected Python session preparation")
        )
        linker.abort_prepared_load = Mock()
        linker.layer_done_counter = ReadPlanLoadCounter(num_layers=1)
        linker.layer_done_counter.update_producer()
        return linker

    def test_prefetch_off_read_plan_owns_sessions(self):
        linker = self._read_plan_linker(prefetch=False)
        transfers = [[PoolTransfer(name=PoolName.KV, keys=["page"])]]

        self.assertTrue(linker.load_layer_wise(0, transfers))

        linker._prepare_expanded_load.assert_not_called()
        linker.abort_prepared_load.assert_not_called()
        linker.storage.store.create_read_plan.assert_called_once_with(
            [],
            1,
            reuse_ranges=False,
            page_wise=False,
            borrowed_sessions=False,
            buffer_owners=linker.pools,
        )
        linker.storage.store.create_read_plan.return_value.run.assert_called_once()

    def test_prefetch_on_read_plan_borrows_prepared_sessions(self):
        linker = self._read_plan_linker(prefetch=True)
        linker.session_lock = threading.Lock()
        linker.prepared_load_sessions = {"rid": ["tagged-page"]}
        transfers = [[PoolTransfer(name=PoolName.KV, keys=["page"])]]

        self.assertTrue(linker.load_layer_wise(0, transfers, request_ids=["rid"]))

        linker._prepare_expanded_load.assert_not_called()
        self.assertTrue(
            linker.storage.store.create_read_plan.call_args.kwargs["borrowed_sessions"]
        )
        linker.abort_prepared_load.assert_called_once_with("rid")

    def test_prefetch_load_without_prepared_session_fails_closed(self):
        for read_plan_enabled in (True, False):
            with self.subTest(read_plan_enabled=read_plan_enabled):
                linker = self._read_plan_linker(prefetch=True)
                linker.read_plan_enabled = read_plan_enabled
                linker.session_lock = threading.Lock()
                linker.prepared_load_sessions = {}
                transfers = [[PoolTransfer(name=PoolName.KV, keys=["page"])]]

                with self.assertLogs(
                    "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker",
                    level="ERROR",
                ) as logs:
                    self.assertFalse(
                        linker.load_layer_wise(
                            0, transfers, request_ids=["missing-rid"]
                        )
                    )

                self.assertIn(
                    "Load admitted without prepared session: rid=missing-rid",
                    logs.output[0],
                )
                linker._prepare_expanded_load.assert_not_called()
                store = linker.storage.store
                store.create_read_plan.assert_not_called()
                store.batch_get_session_start.assert_not_called()
                store.batch_get_session_start_with_sources.assert_not_called()
                store.batch_get_session_refresh.assert_not_called()
                store.batch_get_session_prefetch.assert_not_called()
                store.batch_get_into_multi_buffer_ranges.assert_not_called()
                linker.abort_prepared_load.assert_called_once_with("missing-rid")

    def test_prefetch_off_read_plan_failure_skips_python_session_cleanup(self):
        linker = self._read_plan_linker(prefetch=False)
        linker.storage.store.create_read_plan.return_value.run.side_effect = (
            RuntimeError("read failed")
        )
        transfers = [[PoolTransfer(name=PoolName.KV, keys=["page"])]]

        with self.assertLogs(
            "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker",
            level="ERROR",
        ):
            self.assertFalse(linker.load_layer_wise(0, transfers))

        linker._prepare_expanded_load.assert_not_called()
        linker.abort_prepared_load.assert_not_called()

    def test_prefetch_off_legacy_load_batches_sessions_by_pool(self):
        for page_wise in (False, True):
            with self.subTest(page_wise=page_wise):
                linker = self._read_plan_linker(prefetch=False)
                linker.read_plan_enabled = False
                linker.enable_page_wise_load = page_wise
                linker.page_wise_load_threshold = 1
                del linker._prepare_read_plan_layouts
                linker.storage._get_hybrid_page_component_keys = Mock(
                    side_effect=lambda keys, _: (keys, 1)
                )
                linker.storage._tag_keys = lambda keys: [
                    f"tagged:{key}" for key in keys
                ]
                linker.pools = {
                    PoolName.KV: types.SimpleNamespace(
                        prepare_locations=lambda indices: list(indices),
                        get_prepared_layer_range_meta=lambda locations, _: (
                            [[index] for index in locations],
                            [[8] for _ in locations],
                            [[0] for _ in locations],
                        ),
                    )
                }
                store = linker.storage.store
                store.batch_get_session_start.return_value = [0, 0]
                store.batch_get_into_multi_buffer_ranges.return_value = [8, 8]
                linker.layer_done_counter = Mock()
                events = []
                store.batch_get_session_end.side_effect = lambda _: events.append("end")
                linker.layer_done_counter.complete.side_effect = (
                    lambda *_: events.append("complete")
                )
                transfers = [
                    [PoolTransfer(name=PoolName.KV, keys=["a"], host_indices=[1])],
                    [PoolTransfer(name=PoolName.KV, keys=["b"], host_indices=[2])],
                ]

                self.assertTrue(linker.load_layer_wise(0, transfers))

                store.batch_get_session_start.assert_called_once_with(
                    ["tagged:a", "tagged:b"]
                )
                store.batch_get_session_end.assert_called_once_with(
                    ["tagged:a", "tagged:b"]
                )
                self.assertEqual(
                    events, ["end", "complete"] if page_wise else ["complete", "end"]
                )
                self.assertEqual(
                    linker.storage._get_hybrid_page_component_keys.call_count, 2
                )
                linker._prepare_expanded_load.assert_not_called()
                linker.abort_prepared_load.assert_not_called()

    def _legacy_load_linker(self, store):
        linker = self._read_plan_linker(prefetch=False)
        linker.read_plan_enabled = False
        linker.storage.store = store
        linker.pools = {
            name: types.SimpleNamespace(prepare_locations=lambda indices: list(indices))
            for name in (PoolName.KV, PoolName.SWA)
        }
        linker.storage._get_hybrid_page_component_keys = lambda keys, _: (keys, 1)
        linker.storage._tag_keys = lambda keys: list(keys)
        linker.storage.failed_get_cache = Mock()
        linker.layer_done_counter = Mock()
        return linker

    def test_partial_session_start_failures_release_every_started_batch(self):
        scenarios = (
            ("first_batch_partial", False),
            ("earlier_then_partial", True),
        )
        for scenario, include_earlier in scenarios:
            with self.subTest(scenario=scenario):
                store = Mock()
                sessions = set()

                def start(keys):
                    if keys == ["earlier"]:
                        sessions.add("earlier")
                        return [0]
                    sessions.add("present")
                    return [0, -702]

                def end(keys):
                    sessions.difference_update(keys)

                store.batch_get_session_start.side_effect = start
                store.batch_get_session_end.side_effect = end
                linker = self._legacy_load_linker(store)
                transfers = []
                if include_earlier:
                    transfers.append(
                        [
                            PoolTransfer(
                                name=PoolName.KV,
                                keys=["earlier"],
                                host_indices=[0],
                            )
                        ]
                    )
                transfers.append(
                    [
                        PoolTransfer(
                            name=(PoolName.SWA if include_earlier else PoolName.KV),
                            keys=["present", "missing"],
                            host_indices=[0, 1],
                        )
                    ]
                )

                with self.assertLogs(
                    "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker",
                    level="ERROR",
                ):
                    self.assertFalse(linker.load_layer_wise(0, transfers))

                self.assertEqual(sessions, set())
                expected_start_batches = (
                    [["earlier"], ["present", "missing"], ["present", "missing"]]
                    if include_earlier
                    else [["present", "missing"], ["present", "missing"]]
                )
                self.assertEqual(
                    store.batch_get_session_start.call_args_list,
                    [call(keys) for keys in expected_start_batches],
                )
                self.assertEqual(
                    [
                        call.args[0]
                        for call in store.batch_get_session_end.call_args_list
                    ],
                    (
                        [["earlier"], ["present", "missing"]]
                        if include_earlier
                        else [["present", "missing"]]
                    ),
                )
                linker.storage.failed_get_cache.update_batch.assert_called_once_with(
                    [], ["missing"]
                )
                linker.layer_done_counter.fail.assert_called_once()
                linker.layer_done_counter.complete.assert_not_called()

    def test_partial_session_start_retry_exception_releases_current_batch(self):
        store = Mock()
        sessions = set()
        retry_error = RuntimeError("retry failed")

        def start(keys):
            if store.batch_get_session_start.call_count == 1:
                sessions.add("present")
                return [0, -702]
            raise retry_error

        store.batch_get_session_start.side_effect = start
        store.batch_get_session_end.side_effect = (
            lambda keys: sessions.difference_update(keys)
        )
        linker = self._legacy_load_linker(store)
        transfers = [
            [
                PoolTransfer(
                    name=PoolName.KV,
                    keys=["present", "missing"],
                    host_indices=[0, 1],
                )
            ]
        ]

        with self.assertLogs(
            "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker",
            level="ERROR",
        ):
            self.assertFalse(linker.load_layer_wise(0, transfers))

        self.assertEqual(sessions, set())
        store.batch_get_session_start.assert_has_calls(
            [call(["present", "missing"])] * 2
        )
        store.batch_get_session_end.assert_called_once_with(["present", "missing"])
        linker.storage.failed_get_cache.update_batch.assert_not_called()
        linker.layer_done_counter.fail.assert_called_once_with(0, retry_error)
        linker.layer_done_counter.complete.assert_not_called()

    def test_layout_expands_packed_layer_mapping(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.num_layers = 3
        linker.storage = types.SimpleNamespace(
            _get_hybrid_page_component_keys=lambda keys, _: (keys, 1),
            _tag_keys=lambda keys: [f"tagged:{key}" for key in keys],
        )

        pool = types.SimpleNamespace(
            layer_mapping={0: (0, 2), 1: 1},
            buffer_meta=[
                [(100, 10, 1), (110, 11, 2), (120, 12, 3)],
                [(200, 20, 4), (210, 21, 5), (220, 22, 6)],
            ],
            _component_offsets=[[7, 8, 9], [17, 18, 19]],
            packed=True,
            prepare_locations=lambda indices: [int(value) for value in indices],
        )
        linker.pools = {PoolName.KV: pool}

        transfer = PoolTransfer(
            name=PoolName.KV,
            host_indices=torch.tensor([4, 5]),
            keys=["page-0"],
        )
        layouts = linker._prepare_read_plan_layouts([[transfer]])

        self.assertEqual(len(layouts), 1)
        keys, locations, packed, layers = layouts[0]
        self.assertEqual(keys, ["tagged:page-0"])
        self.assertEqual(locations, [4, 5])
        self.assertTrue(packed)
        self.assertEqual(
            layers,
            [
                [
                    (100, 10, 1, 7),
                    (120, 12, 3, 9),
                    (200, 20, 4, 17),
                    (220, 22, 6, 19),
                ],
                [(110, 11, 2, 8), (210, 21, 5, 18)],
                [],
            ],
        )

    def test_host_prefetch_submission_does_not_prepare_inline(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.host_prefetch_enabled = True
        linker.host_prefetch_limit = 8
        linker.host_prefetch_generation = 0
        linker.host_prefetch_lock = threading.Lock()
        linker.host_prefetch_entries = {}
        linker.host_prefetch_queue = Queue()
        linker.prepare_load = Mock(
            side_effect=AssertionError(
                "session preparation must run on the prefetch worker"
            )
        )
        transfer = PoolTransfer(name=PoolName.KV, keys=["page-a"])

        self.assertTrue(linker.submit_host_prefetch("rid", [transfer]))
        linker.prepare_load.assert_not_called()
        self.assertEqual(linker.get_host_prefetch_status("rid"), "queued")
        queued_rid, queued_transfers = linker.host_prefetch_queue.get_nowait()
        self.assertEqual(queued_rid, "rid")
        self.assertEqual(queued_transfers, [transfer])
        linker.host_prefetch_queue.task_done()

if __name__ == "__main__":
    unittest.main()
