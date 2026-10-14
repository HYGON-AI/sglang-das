"""Startup agreement on waiting-queue prefetch, forward ingress and final poll.

create_custom_parallel_group gathers on the whole world, so the switches that
decide whether the prefetch groups are created must be agreed by every
scheduler rank, whatever its own flags, capability or KV canary mode.
SGLANG_MOONCAKE_FINAL_POLL_TIMEOUT_MS only matters on a PD prefill with
pp_size == 1, so other roles neither read it nor fail on it.
"""

import datetime
import os
import socket
import time
import types
import unittest
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.distributed.parallel_state import (  # noqa: E402
    create_custom_parallel_group,
)
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _scheduler(*, flag=True, capable=True, canary="none", mode="prefill", pp=1):
    s = Scheduler.__new__(Scheduler)
    s.server_args = types.SimpleNamespace(
        mooncake_enable_waiting_queue_dfs_prefetch=flag,
        enable_unified_cache_external_linker=True,
        unified_cache_external_linker_backend="mooncake",
        kv_canary=canary,
    )
    s.disaggregation_mode = {
        "prefill": DisaggregationMode.PREFILL,
        "decode": DisaggregationMode.DECODE,
        "null": DisaggregationMode.NULL,
    }[mode]
    s.schedule_policy = "fcfs"
    s.ps = types.SimpleNamespace(pp_size=pp)
    s.tree_cache = MagicMock()
    s.tree_cache.waiting_queue_prefetch_enabled.return_value = capable
    return s


class TestSingleRank(CustomTestCase):
    def test_all_enabled(self):
        s = _scheduler()
        with envs.SGLANG_MOONCAKE_FINAL_POLL_TIMEOUT_MS.override(400.0):
            self.assertEqual(s._resolve_prefetch_startup_switches(None), (True, True))
        s.tree_cache.disable_waiting_queue_prefetch.assert_not_called()

    def test_canary_keeps_prefetch_but_not_ingress(self):
        s = _scheduler(canary="log")
        self.assertEqual(s._resolve_prefetch_startup_switches(None), (True, False))

    def test_not_configured_skips_the_capability_query(self):
        s = _scheduler(flag=False)
        s.tree_cache = object()
        self.assertEqual(s._resolve_prefetch_startup_switches(None), (False, False))

    def test_invalid_final_poll_reads_as_off_like_the_runtime(self):
        # The typed getter falls back to its default (0.0, off) on a bad value,
        # exactly as the supplemental poll reads it at run time.
        s = _scheduler()
        os.environ["SGLANG_MOONCAKE_FINAL_POLL_TIMEOUT_MS"] = "not-a-number"
        try:
            self.assertEqual(s._resolve_prefetch_startup_switches(None), (True, True))
            self.assertEqual(envs.SGLANG_MOONCAKE_FINAL_POLL_TIMEOUT_MS.get(), 0.0)
        finally:
            os.environ.pop("SGLANG_MOONCAKE_FINAL_POLL_TIMEOUT_MS", None)

    def test_fractional_final_poll_counts_as_on(self):
        s = _scheduler()
        with envs.SGLANG_MOONCAKE_FINAL_POLL_TIMEOUT_MS.override(0.5):
            self.assertEqual(s._resolve_prefetch_startup_switches(None), (True, True))


# Per-rank settings for a 4-rank world laid out as two attention domains
# {0, 1} and {2, 3}. Each case: (settings per rank, expected result or "raise").
CASES = {
    "all_on": ([{}] * 4, (True, True)),
    "one_domain_lacks_capability": (
        [{}, {}, {"capable": False}, {"capable": False}],
        (False, False),
    ),
    "one_rank_flag_off": ([{}, {"flag": False}, {}, {}], (False, False)),
    "all_off": ([{"flag": False}] * 4, (False, False)),
    "unsupported_backend_on_one_rank": (
        [{}, {}, {}, {"backend": "other"}],
        (False, False),
    ),
    "canary_on_one_rank": ([{}, {}, {"canary": "raise"}, {}], (True, False)),
    "final_poll_on_off_split": (
        [{"final_poll": 400.0}, {"final_poll": 400.0}, {"final_poll": 0.0}, {}],
        "raise",
    ),
    "final_poll_fraction_vs_zero": (
        [{"final_poll": 0.5}, {"final_poll": 0.0}, {"final_poll": 0.0}, {}],
        "raise",
    ),
    "final_poll_invalid_vs_on": (
        [{"final_poll": "bad"}, {"final_poll": 400.0}, {"final_poll": 400.0}, {}],
        "raise",
    ),
    "final_poll_values_differ_but_all_on": (
        [{"final_poll": 400.1}, {"final_poll": 400.9}, {"final_poll": 1.0}, {}],
        (True, True),
    ),
    # The supplemental poll never runs outside a PD prefill with pp_size == 1:
    # a disagreeing or invalid value there must not stop the server.
    "decode_final_poll_split": (
        [
            {"mode": "decode", "final_poll": 400.0},
            {"mode": "decode", "final_poll": 400.0},
            {"mode": "decode", "final_poll": 0.0},
            {"mode": "decode", "final_poll": "bad"},
        ],
        (False, False),
    ),
    "null_final_poll_split": (
        [
            {"mode": "null", "final_poll": 400.0},
            {"mode": "null", "final_poll": 0.0},
            {"mode": "null", "final_poll": 0.0},
            {"mode": "null", "final_poll": "bad"},
        ],
        (True, True),
    ),
    "prefill_pp2_final_poll_split": (
        [
            {"pp": 2, "final_poll": 400.0},
            {"pp": 2, "final_poll": 0.0},
            {"pp": 2, "final_poll": 0.0},
            {"pp": 2, "final_poll": 400.0},
        ],
        (False, False),
    ),
}


def _gloo_rank(rank, world_size, port, case, results):
    settings = dict(CASES[case][0][rank])
    final_poll = settings.pop("final_poll", CASES[case][0][0].get("final_poll", 0.0))
    os.environ["SGLANG_MOONCAKE_FINAL_POLL_TIMEOUT_MS"] = str(final_poll)
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=20),
    )
    try:
        backend = settings.pop("backend", "mooncake")
        s = _scheduler(**settings)
        s.server_args.unified_cache_external_linker_backend = backend
        try:
            decision = s._resolve_prefetch_startup_switches(
                torch.distributed.group.WORLD
            )
        except ValueError:
            results[rank] = ("raise", 0)
            return
        created = 0
        if decision[0]:
            # What init_disaggregation does next: every rank must enter it.
            domain = [0, 1] if rank < 2 else [2, 3]
            if create_custom_parallel_group(domain, backend="gloo") is not None:
                created = 1
        results[rank] = (
            decision,
            created,
            s.tree_cache.disable_waiting_queue_prefetch.call_count,
        )
    finally:
        torch.distributed.destroy_process_group()


class TestWorldConsensus(CustomTestCase):
    def _run(self, case):
        import torch.multiprocessing as mp

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        world_size = 4
        with mp.Manager() as manager:
            results = manager.dict()
            context = mp.spawn(
                _gloo_rank,
                args=(world_size, port, case, results),
                nprocs=world_size,
                join=False,
            )
            deadline = time.monotonic() + 120
            while not context.join(timeout=2):
                if time.monotonic() > deadline:
                    for process in context.processes:
                        if process.is_alive():
                            process.kill()
                    self.fail(f"{case}: ranks did not finish (startup hang)")
            return [results[rank] for rank in range(world_size)]

    def test_cases(self):
        for case, (settings, expected) in CASES.items():
            with self.subTest(case=case):
                outcomes = self._run(case)
                if expected == "raise":
                    self.assertEqual([o[0] for o in outcomes], ["raise"] * 4)
                    continue
                for rank, (decision, created, disabled) in enumerate(outcomes):
                    self.assertEqual(decision, expected)
                    self.assertEqual(created, int(expected[0]))
                    # Only a rank that was locally able turns its linker off.
                    locally_able = (
                        settings[rank].get("flag", True)
                        and settings[rank].get("capable", True)
                        and settings[rank].get("backend", "mooncake") == "mooncake"
                        and settings[rank].get("mode", "prefill") != "decode"
                        and settings[rank].get("pp", 1) == 1
                    )
                    self.assertEqual(disabled, int(locally_able and not expected[0]))


class TestRuntimeInit(CustomTestCase):
    """_init_waiting_queue_prefetch_runtime applies the agreed switches."""

    def _init(self, **kwargs):
        from unittest.mock import patch

        import sglang.srt.managers.scheduler as scheduler_module

        s = _scheduler(**kwargs)
        if kwargs.get("flag", True) is False:
            # A disabled rank must not query or call the prefetch-only API.
            s.tree_cache = object()
        s.world_group = types.SimpleNamespace(world_size=1, cpu_group=None)
        s.attn_cp_cpu_group = "cp"
        s.attn_tp_cpu_group = "tp_attn"
        s.tp_cpu_group = "tp"
        created = []
        with patch.object(
            scheduler_module,
            "create_custom_parallel_group",
            side_effect=lambda ranks, backend: created.append(ranks) or "dup",
        ), patch("torch.distributed.get_world_size", return_value=2), patch(
            "torch.distributed.get_process_group_ranks", return_value=[0, 1]
        ):
            s._init_waiting_queue_prefetch_runtime()
        return s, created

    def test_canary_creates_groups_but_no_executor_and_forwards_directly(self):
        s, created = self._init(canary="log")
        self.assertTrue(s.enable_waiting_queue_dfs_prefetch)
        self.assertFalse(s._forward_ingress_enabled)
        self.assertEqual(len(created), 5)  # 2 background + 3 ingress groups
        self.assertIsNone(s._forward_launch_executor)
        s.model_worker = MagicMock()
        s.model_worker.forward_batch_generation.return_value = "direct"
        self.assertEqual(s._forward_with_waiting_queue_ingress("batch"), "direct")
        s.model_worker.forward_batch_generation.assert_called_once_with("batch")

    def test_off_creates_nothing(self):
        s, created = self._init(flag=False)
        self.assertFalse(s.enable_waiting_queue_dfs_prefetch)
        self.assertEqual(created, [])
        self.assertIsNone(s._forward_launch_executor)
        self.assertIsNone(s._bg_thread)
        for name in (
            "_bg_prefetch_jobs",
            "_bg_prefetch_acks",
            "_bg_pending_prefetch_jobs",
            "_bg_prefetch_join_age",
            "_bg_prefetch_rooms",
            "_bg_condition",
        ):
            self.assertFalse(hasattr(s, name), name)

    def test_on_creates_the_executor(self):
        s, created = self._init()
        self.assertTrue(s._forward_ingress_enabled)
        self.assertEqual(len(created), 5)
        self.assertIsNotNone(s._forward_launch_executor)
        s._forward_launch_executor.shutdown(wait=True)

    def test_final_poll_getter_error_fails_everywhere(self):
        from unittest.mock import patch

        s = _scheduler()
        with patch.object(
            type(envs.SGLANG_MOONCAKE_FINAL_POLL_TIMEOUT_MS),
            "get",
            side_effect=RuntimeError("unreadable"),
        ):
            with self.assertRaises(ValueError):
                s._resolve_prefetch_startup_switches(None)

    def test_final_poll_not_read_outside_pd_prefill_pp1(self):
        from unittest.mock import patch

        for kwargs in ({"mode": "decode"}, {"mode": "null"}, {"pp": 2}):
            with self.subTest(**kwargs):
                s = _scheduler(**kwargs)
                with patch.object(
                    type(envs.SGLANG_MOONCAKE_FINAL_POLL_TIMEOUT_MS),
                    "get",
                    side_effect=RuntimeError("unreadable"),
                ) as getter:
                    s._resolve_prefetch_startup_switches(None)
                getter.assert_not_called()


class TestRunBatchOffRoute(CustomTestCase):
    def test_disabled_ingress_calls_worker_directly(self):
        from contextlib import nullcontext
        from unittest.mock import patch

        import sglang.srt.managers.scheduler as scheduler_module

        class StopForward(Exception):
            pass

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.forward_ct = 0
        scheduler._sched_idled = False
        scheduler.scripted_scheduler_hook = None
        scheduler.profiler_manager = MagicMock()
        scheduler.forward_sleep_time = None
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.is_generation = True
        scheduler.enable_overlap = False
        scheduler.enable_pdmux = False
        scheduler._forward_ingress_enabled = False
        scheduler.future_map = MagicMock()
        scheduler._forward_isolation = MagicMock(return_value=nullcontext())
        scheduler.model_worker = MagicMock()
        scheduler.model_worker.forward_batch_generation.side_effect = StopForward
        scheduler._forward_with_waiting_queue_ingress = MagicMock(
            side_effect=AssertionError("disabled ingress wrapper called")
        )
        batch = types.SimpleNamespace(
            forward_mode=types.SimpleNamespace(is_prebuilt=lambda: False),
            spec_algorithm=types.SimpleNamespace(is_none=lambda: False),
            reqs=[],
        )

        with patch.object(scheduler_module, "resolve_forward_inputs"):
            with self.assertRaises(StopForward):
                scheduler.run_batch(batch)

        scheduler.model_worker.forward_batch_generation.assert_called_once_with(batch)
        scheduler._forward_with_waiting_queue_ingress.assert_not_called()


if __name__ == "__main__":
    unittest.main()
