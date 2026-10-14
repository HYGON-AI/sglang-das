import threading
import types
import unittest
from queue import Queue
from unittest.mock import Mock

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
