"""CPU checks for scheduler shared-buffer read/write dependencies."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def scheduler_methods():
    source = Path(__file__).resolve().parents[4] / "python/sglang/srt/managers/scheduler.py"
    tree = ast.parse(source.read_text())
    scheduler = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                     and node.name == "Scheduler")
    methods = [node for node in scheduler.body if isinstance(node, ast.FunctionDef)
               and node.name in ("run_event_loop", "_apply_war_barrier")]
    scope = {}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), scope)
    return scope


class TestSchedulerWarBarrier(CustomTestCase):
    def test_backend_default_and_explicit_override(self):
        for cuda, hcu, override, expected in (
            (True, False, False, True),
            (False, True, False, True),
            (False, False, False, False),
            (False, False, True, True),
        ):
            with self.subTest(cuda=cuda, hcu=hcu, override=override):
                scope = scheduler_methods()
                scope.update(
                    triton_load_watch=SimpleNamespace(install=Mock(), mark_serving_started=Mock()),
                    use_mlx=lambda: False,
                    is_cuda=lambda: cuda,
                    is_hcu=lambda: hcu,
                    _is_hip=hcu,
                    get_parallel=lambda: SimpleNamespace(pp_size=1),
                    envs=SimpleNamespace(SGLANG_ENABLE_WAR_BARRIER=SimpleNamespace(get=lambda: override)),
                    dispatch_event_loop=Mock(),
                )
                scheduler = SimpleNamespace(
                    device="cuda", enable_overlap=True,
                    schedule_stream=SimpleNamespace(cuda_stream=1),
                    forward_stream=SimpleNamespace(cuda_stream=2),
                    device_module=SimpleNamespace(StreamContext=lambda stream: nullcontext()),
                    metrics_reporter=SimpleNamespace(start_scheduler_time_accounting=Mock()),
                )
                scope["run_event_loop"](scheduler)
                self.assertEqual(scheduler._war_barrier_enabled, expected)
                scope["dispatch_event_loop"].assert_called_once_with(scheduler)

    def test_waits_for_gpu_event_without_cpu_synchronization(self):
        scope = scheduler_methods()
        scope["envs"] = SimpleNamespace(
            SGLANG_FORCE_COARSE_WAR_BARRIER=SimpleNamespace(get=lambda: False))
        event = object()
        runner = SimpleNamespace(shared_read_done_event=event)
        stream = SimpleNamespace(wait_event=Mock(), wait_stream=Mock())
        scheduler = SimpleNamespace(
            _war_barrier_enabled=True, schedule_stream=stream,
            forward_stream=object(), model_worker=SimpleNamespace(last_shared_read_runner=runner))
        scope["_apply_war_barrier"](scheduler)
        stream.wait_event.assert_called_once_with(event)
        stream.wait_stream.assert_not_called()
        self.assertIsNone(runner.shared_read_done_event)

    def test_missing_event_falls_back_to_gpu_stream_dependency(self):
        scope = scheduler_methods()
        scope["envs"] = SimpleNamespace(
            SGLANG_FORCE_COARSE_WAR_BARRIER=SimpleNamespace(get=lambda: False))
        stream = SimpleNamespace(wait_event=Mock(), wait_stream=Mock())
        forward = object()
        scheduler = SimpleNamespace(
            _war_barrier_enabled=True, schedule_stream=stream, forward_stream=forward,
            model_worker=SimpleNamespace(last_shared_read_runner=SimpleNamespace(shared_read_done_event=None)))
        scope["_apply_war_barrier"](scheduler)
        stream.wait_stream.assert_called_once_with(forward)
        stream.wait_event.assert_not_called()


if __name__ == "__main__":
    unittest.main()
