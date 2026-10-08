"""Forward ingress: receive-broadcast agreement, health checks, background
worker shutdown and fail-fast propagation of thread errors."""

import datetime
import signal
import socket
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import sglang.srt.managers.scheduler as scheduler_module  # noqa: E402
import sglang.srt.managers.scheduler_components.request_receiver as receiver_module  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.request_receiver import (  # noqa: E402
    SchedulerRequestReceiver,
)
from sglang.srt.managers.utils import HEALTH_CHECK_RID_PREFIX  # noqa: E402

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


class FakeRequest:
    mm_inputs = None
    input_embeds = None
    session_id = None
    session_params = None
    lora_id = None
    positional_embed_overrides = None

    def __init__(self, rid, ipc=None):
        self.rid = rid
        self.http_worker_ipc = ipc
        self.sampling_params = SimpleNamespace()


def _ingress_scheduler(forward, receiver):
    s = Scheduler.__new__(Scheduler)
    s.enable_waiting_queue_dfs_prefetch = True
    s._forward_ingress_enabled = True
    s._bg_error = None
    s.forward_ct = 0
    s.enable_overlap = False
    s.device = "cpu"
    s.device_module = SimpleNamespace(StreamContext=lambda _stream: nullcontext())
    s.forward_stream = SimpleNamespace(wait_stream=lambda _stream: None)
    s.schedule_stream = SimpleNamespace(
        wait_stream=lambda _stream: None, synchronize=lambda: None
    )
    s.model_worker = SimpleNamespace(forward_batch_generation=forward)
    s.request_receiver = receiver
    s._request_bg_prefetch_epoch = lambda: None
    s._forward_deferred_reqs = []
    s.return_health_check_ipcs = []
    s.handled = []
    s.handle_generate_request = lambda req: s.handled.append(req.rid)
    s._forward_launch_executor = ThreadPoolExecutor(max_workers=1)
    s.tree_cache = MagicMock()
    return s


def _run_ingress(s):
    batch = SimpleNamespace()
    try:
        with patch.object(scheduler_module, "TokenizedGenerateReqInput", FakeRequest):
            return s._forward_with_waiting_queue_ingress(batch)
    finally:
        s._forward_launch_executor.shutdown(wait=True)


class _ScriptedReceiver:
    """Returns each scripted list once, then empty lists; no collectives."""

    def __init__(self, *batches):
        self.batches = list(batches)

    def ingress_sync_groups(self):
        return []

    def recv_requests(self):
        return self.batches.pop(0) if self.batches else []


def _slow_forward(seconds, result="done"):
    def forward(_batch):
        time.sleep(seconds)
        return result

    return forward


# ---- N2: which groups the ingress loop agrees on --------------------------


class TestIngressSyncGroups(CustomTestCase):
    def _receiver(self, *, tp, attn_tp, attn_cp):
        # A frozen, slotted dataclass: set only the fields the method reads.
        r = SchedulerRequestReceiver.__new__(SchedulerRequestReceiver)
        fields = dict(
            ps=SimpleNamespace(tp_size=tp, attn_tp_size=attn_tp, attn_cp_size=attn_cp),
            tp_cpu_group="tp",
            attn_tp_cpu_group="attn_tp",
            attn_cp_cpu_group="attn_cp",
        )
        for name, value in fields.items():
            object.__setattr__(r, name, value)
        return r

    def _groups(self, receiver, *, dp, local):
        parallel = SimpleNamespace(
            enable_dp_attention=dp,
            enable_dp_attention_local_control_broadcast=local,
        )
        with patch.object(
            receiver_module, "get_parallel", return_value=parallel
        ), patch.object(receiver_module, "is_ep_scale_joiner", return_value=False):
            return receiver.ingress_sync_groups()

    def test_mapping_follows_the_receive_broadcasts(self):
        cases = [
            # (tp, attn_tp, attn_cp, dp, local) -> groups
            ((4, 4, 1, False, False), ["tp"]),
            ((8, 1, 8, False, False), ["tp"]),
            ((1, 1, 1, False, False), []),
            ((4, 2, 1, True, False), ["tp"]),
            ((4, 2, 1, True, True), ["attn_tp"]),
            ((4, 1, 1, True, True), []),
            ((8, 2, 2, True, True), ["attn_tp", "attn_cp"]),
        ]
        for (tp, attn_tp, attn_cp, dp, local), expected in cases:
            with self.subTest(
                tp=tp, attn_tp=attn_tp, attn_cp=attn_cp, dp=dp, local=local
            ):
                receiver = self._receiver(tp=tp, attn_tp=attn_tp, attn_cp=attn_cp)
                self.assertEqual(self._groups(receiver, dp=dp, local=local), expected)


class TestIngressSyncGroupsCoverRealBroadcasts(CustomTestCase):
    """Drive the real _broadcast_reqs_across_ranks and check that the groups
    it broadcasts on are covered by ingress_sync_groups (full TP covers the
    attention sub-groups), so every broadcast participant takes part in the
    loop's agreement."""

    def _receiver(self, *, tp, attn_tp, attn_cp):
        r = SchedulerRequestReceiver.__new__(SchedulerRequestReceiver)
        # This rank is the leader (rank 0) of every group, which is valid for
        # any group size, including size-1 groups; groups of size 1 hold only it.
        group = lambda name, size: SimpleNamespace(
            rank=0, ranks=list(range(size)), name=name
        )
        fields = dict(
            ps=SimpleNamespace(
                tp_size=tp,
                attn_tp_size=attn_tp,
                attn_cp_size=attn_cp,
                attn_tp_rank=0,
                attn_cp_rank=0,
            ),
            tp_group=group("tp", tp),
            attn_tp_group=group("attn_tp", attn_tp),
            attn_cp_group=group("attn_cp", attn_cp),
            tp_cpu_group="tp",
            attn_tp_cpu_group="attn_tp",
            attn_cp_cpu_group="attn_cp",
        )
        for name, value in fields.items():
            object.__setattr__(r, name, value)
        return r

    def test_sync_groups_cover_every_broadcast(self):
        cases = [
            # (tp, attn_tp, attn_cp, dp, local, joiner)
            (4, 4, 1, False, False, False),
            (8, 1, 8, False, False, False),
            (1, 1, 1, False, False, False),
            (4, 2, 1, True, False, False),
            (4, 2, 1, True, True, False),
            (4, 1, 1, True, True, False),
            (8, 2, 2, True, True, False),
            (8, 2, 2, True, False, False),
            (8, 2, 2, True, False, True),  # EP scale joiner: local control
            (4, 1, 4, True, False, False),
        ]
        for tp, attn_tp, attn_cp, dp, local, joiner in cases:
            with self.subTest(
                tp=tp,
                attn_tp=attn_tp,
                attn_cp=attn_cp,
                dp=dp,
                local=local,
                joiner=joiner,
            ):
                receiver = self._receiver(tp=tp, attn_tp=attn_tp, attn_cp=attn_cp)
                parallel = SimpleNamespace(
                    enable_dp_attention=dp,
                    enable_dp_attention_local_control_broadcast=local,
                )
                broadcasts = []

                def fake_broadcast(data, rank, group, src):
                    broadcasts.append(group)
                    return []

                with patch.object(
                    receiver_module, "get_parallel", return_value=parallel
                ), patch.object(
                    receiver_module, "is_ep_scale_joiner", return_value=joiner
                ), patch.object(
                    receiver_module, "broadcast_pyobj", side_effect=fake_broadcast
                ):
                    # The leader splits what it received; an empty batch
                    # gives two empty lists and still drives every broadcast.
                    self.assertEqual(receiver._broadcast_reqs_across_ranks([]), [])
                    sync = receiver.ingress_sync_groups()
                for group in broadcasts:
                    self.assertTrue(
                        group in sync or "tp" in sync,
                        f"broadcast on {group} not covered by {sync}",
                    )
                # No agreement when nothing is broadcast, and none missing
                # when something is.
                self.assertEqual(bool(sync), bool(broadcasts), (sync, broadcasts))


def _dp_rank(rank, world_size, port, local_ctrl, results):
    """DP2 x attnTP2: domains {0,1} and {2,3} whose forwards end 0.4 s apart."""
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=20),
    )
    try:
        full = dist.new_group([0, 1, 2, 3], backend="gloo")
        domains = [dist.new_group(r, backend="gloo") for r in ([0, 1], [2, 3])]
        original = dist.new_group([0, 1, 2, 3], backend="gloo")
        domain = domains[rank // 2]
        leader = 0 if rank < 2 else 2

        class Receiver:
            calls = 0

            def ingress_sync_groups(self):
                return [domain] if local_ctrl else [full]

            def recv_requests(self):
                Receiver.calls += 1
                work = [None]
                dist.broadcast_object_list(work, src=leader, group=domain)
                if not local_ctrl:
                    control = [None]
                    dist.broadcast_object_list(control, src=0, group=full)
                return []

        forward = _slow_forward(0.1 if rank < 2 else 0.5)
        s = _ingress_scheduler(forward, Receiver())
        assert _run_ingress(s) == "done"
        # The next iteration's DP sync on the original group.
        dist.all_reduce(torch.ones(1), group=original)
        results[rank] = Receiver.calls
    finally:
        dist.destroy_process_group()


def _spawn(test, fn, world_size, *args, deadline_s=90):
    import torch.multiprocessing as mp

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with mp.Manager() as manager:
        results = manager.dict()
        context = mp.spawn(
            fn, args=(world_size, port, *args, results), nprocs=world_size, join=False
        )
        # A rank that raises fails the test through join(); ranks whose
        # errors are expected catch them and report them in results.
        deadline = time.monotonic() + deadline_s
        try:
            while not context.join(timeout=2):
                if time.monotonic() > deadline:
                    test.fail("ranks did not finish")
        finally:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
                process.join(timeout=10)
        return dict(results)


class TestIngressAcrossDpDomains(CustomTestCase):
    def test_full_tp_control_broadcast(self):
        calls = _spawn(self, _dp_rank, 4, False)
        self.assertEqual(sorted(calls), [0, 1, 2, 3])
        self.assertEqual(len(set(calls.values())), 1, calls)

    def test_local_control_broadcast(self):
        calls = _spawn(self, _dp_rank, 4, True)
        self.assertEqual(sorted(calls), [0, 1, 2, 3])
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(calls[2], calls[3])
        # Domains exit on their own: the faster one received fewer times.
        self.assertLess(calls[0], calls[2])


# ---- N3: health checks during forward -------------------------------------


class TestHealthCheckDuringForward(CustomTestCase):
    def test_answered_as_busy_not_dispatched(self):
        health = FakeRequest(HEALTH_CHECK_RID_PREFIX + "x", ipc="ipc-1")
        s = _ingress_scheduler(
            _slow_forward(0.2), _ScriptedReceiver([health, FakeRequest("a")])
        )
        self.assertEqual(_run_ingress(s), "done")
        self.assertEqual(s.handled, ["a"])
        self.assertEqual(s.return_health_check_ipcs, ["ipc-1"])
        self.assertEqual(s._forward_deferred_reqs, [])

    def test_order_is_kept_after_a_deferred_request(self):
        control = object()
        health = FakeRequest(HEALTH_CHECK_RID_PREFIX + "y", ipc="ipc-2")
        s = _ingress_scheduler(_slow_forward(0.2), _ScriptedReceiver([control, health]))
        _run_ingress(s)
        self.assertEqual(s._forward_deferred_reqs, [control, health])
        self.assertEqual(s.return_health_check_ipcs, [])


# ---- N4: background worker shutdown --------------------------------------


def _bg_scheduler(round_fn):
    s = Scheduler.__new__(Scheduler)
    s.enable_waiting_queue_dfs_prefetch = True
    s._bg_condition = threading.Condition()
    s._bg_requested_epoch = 0
    s._bg_completed_epoch = 0
    s._bg_stop_flag = False
    s._bg_error = None
    s._run_bg_prefetch_round = round_fn
    s._bg_thread = threading.Thread(target=s._bg_worker_loop, daemon=True)
    s.tree_cache = MagicMock()
    s.hisparse_coordinator = None
    s.decode_offload_manager = None
    s._forward_launch_executor = MagicMock()
    return s


class TestBackgroundWorkerShutdown(CustomTestCase):
    def setUp(self):
        patcher = patch.object(scheduler_module, "rank_consensus_checker")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_idle_worker_stops_before_resources_close(self):
        s = _bg_scheduler(MagicMock())
        s._bg_thread.start()
        seen = []
        s.tree_cache.release_host_resources.side_effect = lambda: seen.append(
            s._bg_thread.is_alive()
        )
        s.release_host_resources()
        self.assertEqual(seen, [False])
        s._forward_launch_executor.shutdown.assert_called_once_with(wait=True)

    def test_stuck_round_leaves_resources_alone(self):
        entered, release = threading.Event(), threading.Event()

        def round_fn():
            entered.set()
            release.wait(timeout=30)  # a peer that never joins this round

        s = _bg_scheduler(round_fn)
        s._bg_thread.start()
        s._request_bg_prefetch_epoch()
        self.assertTrue(entered.wait(timeout=10))
        s._stop_bg_prefetch_worker = lambda: Scheduler._stop_bg_prefetch_worker(
            s, timeout_s=0.2
        )
        s.release_host_resources()
        s.tree_cache.release_host_resources.assert_not_called()
        s._forward_launch_executor.shutdown.assert_called_once_with(wait=False)
        s.release_host_resources()  # repeated call: nothing more
        s._forward_launch_executor.shutdown.assert_called_once()
        release.set()
        s._bg_thread.join(timeout=10)

    def test_requested_round_not_yet_entered_is_dropped(self):
        round_fn = MagicMock()
        s = _bg_scheduler(round_fn)
        s._bg_requested_epoch = 1  # requested, the worker has not woken yet
        s._bg_stop_flag = True
        s._bg_thread.start()
        s._bg_thread.join(timeout=10)
        self.assertFalse(s._bg_thread.is_alive())
        round_fn.assert_not_called()

    def test_error_is_reported_before_stop(self):
        s = _bg_scheduler(MagicMock())
        s._bg_error = RuntimeError("round failed")
        s._bg_stop_flag = True
        with self.assertRaises(RuntimeError):
            s._request_bg_prefetch_epoch()
        s._bg_error = None
        s._request_bg_prefetch_epoch()
        self.assertEqual(s._bg_requested_epoch, 0)


def _stop_interleaved_rank(rank, world_size, port, results):
    """Rank 0's worker is inside a background round whose peer (rank 1) stops
    before entering it: rank 0 must time out, skip closing what the worker
    still uses; rank 1 drops its not-yet-entered round and closes normally."""
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=5),
    )
    try:
        # The stuck round must end by this group's own timeout (longer than
        # the 0.5 s stop wait), not because the peer went away; the peer stays
        # alive on a separate control group until that has happened.
        bg_group = dist.new_group(
            [0, 1], backend="gloo", timeout=datetime.timedelta(seconds=3)
        )
        ctrl_group = dist.new_group(
            [0, 1], backend="gloo", timeout=datetime.timedelta(seconds=60)
        )
        entered = threading.Event()
        calls = []

        def round_fn():
            calls.append(1)
            entered.set()
            dist.all_reduce(torch.ones(1), group=bg_group)  # never joined

        with patch.object(scheduler_module, "rank_consensus_checker"):
            s = _bg_scheduler(round_fn)
            s._stop_bg_prefetch_worker = lambda: Scheduler._stop_bg_prefetch_worker(
                s, timeout_s=0.5
            )
            if rank == 0:
                s._bg_thread.start()
                s._request_bg_prefetch_epoch()
                assert entered.wait(timeout=10)
            else:
                # Requested, but the stop arrives before the worker wakes.
                s._bg_requested_epoch = 1
                s._bg_stop_flag = True
                s._bg_thread.start()
            # Rank 0 is inside the round before anyone stops.
            dist.barrier(group=ctrl_group)
            s.release_host_resources()
            alive_at_release = s._bg_thread.is_alive()
            closed = s.tree_cache.release_host_resources.call_count
            shutdown = s._forward_launch_executor.shutdown.call_args
            # Rank 1 stays alive until rank 0 has sampled its stop.
            dist.barrier(group=ctrl_group)
        # Rank 0's stuck round ends by bg_group's timeout while rank 1 is
        # still alive; rank 1 waits for that before tearing the groups down.
        s._bg_thread.join(timeout=30)
        dist.barrier(group=ctrl_group)
        results[rank] = (
            len(calls),
            alive_at_release,
            closed,
            dict(shutdown.kwargs) if shutdown else None,
            s._bg_thread.is_alive(),
            s._bg_error is not None,
        )
    finally:
        dist.destroy_process_group()


class TestInterleavedStopAcrossRanks(CustomTestCase):
    def test_bounded_best_effort_stop(self):
        results = _spawn(self, _stop_interleaved_rank, 2, deadline_s=120)
        # Rank 0: in the round at stop, join timed out, nothing closed, the
        # executor is not waited on; the worker ends later on its own, with
        # the round's collective timeout recorded as its error.
        self.assertEqual(results[0], (1, True, 0, {"wait": False}, False, True))
        # Rank 1: the requested round was never entered; normal close.
        self.assertEqual(results[1], (0, False, 1, {"wait": True}, False, False))


# ---- N5: fail fast --------------------------------------------------------


def _fail_fast_rank(rank, world_size, port, results):
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=5),
    )
    group = dist.new_group([0, 1], backend="gloo")

    class Receiver:
        def ingress_sync_groups(self):
            return [group]

        def recv_requests(self):
            return []

    def failing(_batch):
        raise ValueError("forward failed on rank 0")

    forward = failing if rank == 0 else _slow_forward(3.0)
    s = _ingress_scheduler(forward, Receiver())
    start = time.monotonic()
    try:
        _run_ingress(s)
        results[rank] = ("returned", time.monotonic() - start)
    except BaseException as exc:
        # Gloo reports a vanished peer with RuntimeError subclasses.
        kind = type(exc).__name__
        if isinstance(exc, RuntimeError):
            kind = "RuntimeError"
        results[rank] = (kind, time.monotonic() - start)


class TestFailFast(CustomTestCase):
    def test_forward_error_raises_before_the_peer_finishes(self):
        results = _spawn(self, _fail_fast_rank, 2)
        kind, elapsed = results[0]
        self.assertEqual(kind, "ValueError")
        # Without fail-fast rank 0 waits for rank 1's 3 s forward.
        self.assertLess(elapsed, 1.5)
        # Rank 1 either finishes its forward or fails on the vanished peer.
        self.assertIn(results[1][0], {"returned", "RuntimeError"})

    def test_background_error_raises_during_ingress(self):
        s = _ingress_scheduler(_slow_forward(2.0), _ScriptedReceiver())
        threading.Timer(0.1, lambda: setattr(s, "_bg_error", OSError("bg"))).start()
        start = time.monotonic()
        with self.assertRaises(RuntimeError):
            with patch.object(
                scheduler_module, "TokenizedGenerateReqInput", FakeRequest
            ):
                s._forward_with_waiting_queue_ingress(SimpleNamespace())
        self.assertLess(time.monotonic() - start, 1.0)
        s._forward_launch_executor.shutdown(wait=True)

    def test_forward_error_reaches_the_process_fatal_path(self):
        for killpg in (False, True):
            with self.subTest(killpg=killpg):
                self._fatal_path(killpg)

    def _fatal_path(self, killpg_enabled):
        from sglang.srt.environ import envs

        ingress = _ingress_scheduler(_slow_forward(0.0), _ScriptedReceiver())

        def failing(_batch):
            raise ValueError("forward failed")

        ingress.model_worker = SimpleNamespace(forward_batch_generation=failing)

        class FakeScheduler:
            gracefully_exit = False
            metrics_reporter = SimpleNamespace(_shutdown_fpm=lambda: None)

            def __init__(self, *args, **kwargs):
                pass

            def get_init_info(self):
                return {}

            def run_event_loop(self):
                _run_ingress(ingress)

            def release_host_resources(self):
                raise AssertionError("not on the exception path")

        parent = MagicMock()
        # Never signal the real process group, whatever the environment says.
        with envs.SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION.override(
            killpg_enabled
        ), patch.object(scheduler_module.os, "killpg") as killpg, patch.object(
            scheduler_module, "load_plugins"
        ), patch.object(
            scheduler_module, "publish"
        ), patch.object(
            scheduler_module, "configure_scheduler_process", return_value=0
        ), patch.object(
            scheduler_module.psutil, "Process"
        ) as process, patch.object(
            scheduler_module,
            "get_observability",
            return_value=SimpleNamespace(enable_trace=False),
        ), patch.object(
            scheduler_module, "Scheduler", FakeScheduler
        ), patch.object(
            scheduler_module, "TokenizedGenerateReqInput", FakeRequest
        ):
            process.return_value.parent.return_value = parent
            scheduler_module.run_scheduler_process(
                MagicMock(), MagicMock(), 0, 0, 0, 0, 0, 0, None, MagicMock()
            )
        parent.send_signal.assert_called_once_with(signal.SIGQUIT)
        self.assertEqual(killpg.call_count, int(killpg_enabled))


if __name__ == "__main__":
    unittest.main()
