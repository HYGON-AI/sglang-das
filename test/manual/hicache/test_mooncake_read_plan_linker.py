"""CPU tests for the linker bridge, without importing SGLang's GPU runtime."""

import ast
import logging
import threading
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

LINKER = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_direct_linker.py"
)
tree = ast.parse(LINKER.read_text())
namespace = {
    "Future": Future,
    "torch": torch,
    "logger": logging.getLogger(__name__),
    "arm_load_failure_injection": lambda _rank: lambda *_args: None,
}
counter = next(
    n
    for n in tree.body
    if isinstance(n, ast.ClassDef) and n.name == "ReadPlanLoadCounter"
)
linker = next(
    n
    for n in tree.body
    if isinstance(n, ast.ClassDef) and n.name == "MooncakeDirectLinker"
)
methods = [
    n
    for n in linker.body
    if isinstance(n, ast.FunctionDef)
    and n.name
    in ("_prepare_read_plan_layouts", "load_with_read_plan", "load_layer_wise")
]
for node in [counter, *methods]:
    exec(
        compile(
            "from __future__ import annotations\n" + ast.unparse(node),
            str(LINKER),
            "exec",
        ),
        namespace,
    )
ReadPlanLoadCounter = namespace["ReadPlanLoadCounter"]


def test_prepare_merges_indices_preserving_key_order():
    copied = []
    pool = SimpleNamespace(
        packed=True,
        layer_mapping={0: 0, 2: 1},
        buffer_meta=[[(4096, 64, 32), (8192, 64, 32)]],
        _component_offsets=[[0, 32]],
        prepare_locations=lambda indices: copied.append(indices.tolist())
        or indices.tolist(),
    )
    storage = SimpleNamespace(
        _get_hybrid_page_component_keys=lambda keys, transfer: (keys, None),
        _tag_keys=lambda keys: ["tag/" + key for key in keys],
    )
    transfers = [
        [SimpleNamespace(name="kv", keys=["a"], host_indices=torch.tensor([3]))],
        [SimpleNamespace(name="kv", keys=["b"], host_indices=torch.tensor([7]))],
    ]
    result = namespace["_prepare_read_plan_layouts"](
        SimpleNamespace(storage=storage, pools={"kv": pool}, num_layers=3), transfers
    )
    assert copied == [[3, 7]]  # One copy/validation per pool, not per request.
    assert result == [
        (
            ["tag/a", "tag/b"],
            [3, 7],
            True,
            [[(4096, 64, 32, 0)], [], [(8192, 64, 32, 32)]],
        )
    ]


def test_stream_preparation_failure_wakes_waiter_and_keeps_worker_alive():
    import logging
    from queue import Queue

    path = LINKER
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MooncakeDirectLinker"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "load_thread_func"
    )
    namespace = {"logger": logging.getLogger("read-plan-test")}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    counter = ReadPlanLoadCounter(2)
    first = counter.update_producer()
    second = counter.update_producer()
    queue = Queue()
    completed_results = Queue()
    completed = []

    def fail():
        raise ValueError("stream failed")

    queue.put((first, {"first": []}, SimpleNamespace(synchronize=fail)))
    queue.put((second, {"second": []}, SimpleNamespace(synchronize=lambda: None)))
    queue.put(None)
    linker = SimpleNamespace(
        _finish_prefetch_metrics=lambda rids, success: None,
        load_queue=queue,
        completed_loads=completed_results,
        layer_done_counter=counter,
        load_layer_wise=lambda index, transfers: completed.append(index) or True,
    )
    thread = threading.Thread(
        target=namespace["load_thread_func"], args=(linker,), daemon=True
    )
    thread.start()
    thread.join(2)
    assert not thread.is_alive() and completed == [second]
    assert completed_results.get_nowait() == (["first"], False)
    assert completed_results.get_nowait() == (["second"], True)
    counter.set_consumer(first)
    counter.wait_until(0)
    counter.wait_until(1)
    assert first not in counter.plans and not counter.reported


def test_plan_publication_and_final_wait_release():
    counter = ReadPlanLoadCounter(2)
    index = counter.update_producer()
    counter.set_consumer(index)
    entered, release = threading.Event(), threading.Event()

    class Plan:
        def wait(self, group):
            entered.set()
            assert release.wait(2)

    counter.bind(index, Plan())
    done = Future()

    def consume():
        try:
            counter.wait_until(1)
            done.set_result(None)
        except BaseException as error:
            done.set_exception(error)

    worker = threading.Thread(target=consume, daemon=True)
    worker.start()
    try:
        assert entered.wait(2)
        assert not done.done()
    finally:
        release.set()
    done.result(timeout=2)
    worker.join(2)
    assert index not in counter.plans


def test_linker_passes_layouts_and_owners_to_packaged_plan():
    calls = []
    plan = SimpleNamespace(run=lambda: calls.append("run"))

    def create(layouts, groups, **kwargs):
        calls.append((layouts, groups, kwargs))
        return plan

    pools = {"pool": object()}
    linker = SimpleNamespace(
        _prepare_read_plan_layouts=lambda transfers: ["layout"],
        storage=SimpleNamespace(store=SimpleNamespace(create_read_plan=create)),
        pools=pools,
        num_layers=2,
        read_plan_reuse_ranges=False,
        layer_done_counter=SimpleNamespace(
            bind=lambda index, p: calls.append((index, p))
        ),
    )
    namespace["load_with_read_plan"](linker, 7, [])
    assert calls == [
        (["layout"], 2, {"reuse_ranges": False, "buffer_owners": pools}),
        (7, plan),
        "run",
    ]


@pytest.mark.parametrize("failure_stage", ["create", "run"])
def test_read_plan_failure_reaches_consumer(failure_stage, caplog):
    counter = ReadPlanLoadCounter(2)
    index = counter.update_producer()
    counter.set_consumer(index)
    error = ValueError("read plan " + failure_stage + " failed")

    class Plan:
        failure = None

        def run(self):
            self.failure = error
            raise error

        def wait(self, group):
            if self.failure is not None:
                raise self.failure
            pytest.fail("wait should observe the failed plan")

    def create(*args, **kwargs):
        if failure_stage == "create":
            raise error
        return Plan()

    linker = SimpleNamespace(
        read_plan_enabled=True,
        read_plan_reuse_ranges=False,
        _prepare_read_plan_layouts=lambda transfers: [],
        storage=SimpleNamespace(store=SimpleNamespace(create_read_plan=create)),
        pools={},
        num_layers=2,
        tp_rank=0,
        layer_done_counter=counter,
    )
    linker.load_with_read_plan = lambda i, t: namespace["load_with_read_plan"](
        linker, i, t
    )
    assert namespace["load_layer_wise"](linker, index, []) is False
    assert counter.plans[index].done()
    for _ in range(3):
        counter.wait_until(0)
    counter.wait_until(1)
    assert index not in counter.plans and not counter.reported
    assert (
        sum("affected requests will be aborted" in r.message for r in caplog.records)
        == 1
    )


def test_failure_does_not_retain_forward_tracebacks():
    counter = ReadPlanLoadCounter(3)
    index = counter.update_producer()
    counter.set_consumer(index)
    try:
        raise ValueError("construction failed")
    except ValueError as original:
        traceback = original.__traceback__
        counter.fail(index, original)
        failure = counter.plans[index].exception()
        assert failure is not original
        for _ in range(10):
            counter.wait_until(0)
            assert failure.__traceback__ is None
        assert original.__traceback__ is traceback
    counter.reset()
    assert not counter.plans and not counter.reported


@pytest.mark.parametrize("failed_rank", [0, 3])
def test_injected_failure_is_queued_and_next_batch_runs(monkeypatch, failed_rank):
    from queue import Queue

    # Execute the real injection logic with a 100% probability on one rank.
    injection_path = LINKER.parents[2] / "unified_cache/linker_fault_injection.py"
    injection_tree = ast.parse(injection_path.read_text())
    injection_tree.body = [
        node
        for node in injection_tree.body
        if not (
            isinstance(node, ast.ImportFrom) and node.module == "sglang.srt.environ"
        )
    ]
    injection_ns = {
        "envs": SimpleNamespace(
            SGLANG_TEST_LINKER_LOAD_FAILURE_PROB=SimpleNamespace(get=lambda: 1.0),
            SGLANG_TEST_LINKER_LOAD_FAILURE_RANKS=SimpleNamespace(
                get=lambda: str(failed_rank)
            ),
        )
    }
    exec(compile(injection_tree, str(injection_path), "exec"), injection_ns)
    monkeypatch.setitem(
        namespace,
        "arm_load_failure_injection",
        injection_ns["arm_load_failure_injection"],
    )
    method = next(
        n
        for n in linker.body
        if isinstance(n, ast.FunctionDef) and n.name == "load_thread_func"
    )
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(LINKER), "exec"),
        namespace,
    )
    counter = ReadPlanLoadCounter(2)
    failed = counter.update_producer()
    healthy = counter.update_producer()
    queue, completed = Queue(), Queue()
    calls = []
    instance = SimpleNamespace(
        read_plan_enabled=True,
        tp_rank=failed_rank,
        layer_done_counter=counter,
        _finish_prefetch_metrics=lambda rids, success: None,
        load_queue=queue,
        completed_loads=completed,
    )

    def load_plan(index, transfers):
        calls.append(index)
        counter.bind(index, SimpleNamespace(wait=lambda group: None))

    instance.load_with_read_plan = load_plan
    instance.load_layer_wise = lambda i, t: namespace["load_layer_wise"](instance, i, t)
    queue.put(
        (failed, {"failed-request": []}, SimpleNamespace(synchronize=lambda: None))
    )

    def select_healthy_rank():
        instance.tp_rank = failed_rank + 1

    queue.put(
        (
            healthy,
            {"healthy-request": []},
            SimpleNamespace(synchronize=select_healthy_rank),
        )
    )
    queue.put(None)
    worker = threading.Thread(
        target=namespace["load_thread_func"], args=(instance,), daemon=True
    )
    worker.start()
    worker.join(2)
    assert not worker.is_alive()
    assert completed.get_nowait() == (["failed-request"], False)
    assert completed.get_nowait() == (["healthy-request"], True)
    assert calls == [healthy]  # Failed injection never creates/binds a native plan.
    for index in (failed, healthy):
        counter.set_consumer(index)
        counter.wait_until(0)
        counter.wait_until(1)
    assert not counter.plans and not counter.reported


@pytest.mark.parametrize("draft_indices", [(0, 2), [0, 2]])
def test_read_plan_matches_pool_ranges_with_packed_draft(draft_indices):
    # Execute the real pool metadata method; mocks with integer-only mappings
    # miss the target+draft tuple used by DSPARK after the pool refactor.
    path = LINKER.parents[2] / "hybrid_cache/linker_pool_assembler.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and any(
            isinstance(m, ast.FunctionDef) and m.name == "get_prepared_layer_range_meta"
            for m in n.body
        )
    )
    method = next(
        m
        for m in cls.body
        if isinstance(m, ast.FunctionDef) and m.name == "get_prepared_layer_range_meta"
    )
    scope = {}
    exec(
        compile(
            "from __future__ import annotations\n" + ast.unparse(method),
            str(path),
            "exec",
        ),
        scope,
    )
    pool = SimpleNamespace(
        packed=True,
        layer_mapping={0: draft_indices, 2: 1},
        buffer_meta=[
            [(1000, 64, 8), (2000, 64, 8), (3000, 64, 8)],
            [(4000, 64, 4), (5000, 64, 4), (6000, 64, 4)],
        ],
        _component_offsets=[[0, 8, 16], [24, 28, 32]],
        prepare_locations=lambda x: x.tolist(),
    )
    storage = SimpleNamespace(
        _get_hybrid_page_component_keys=lambda keys, transfer: (keys, 1),
        _tag_keys=lambda keys: keys,
    )
    transfers = [
        [SimpleNamespace(name="kv", keys=["a", "b"], host_indices=torch.tensor([1, 3]))]
    ]
    layouts = namespace["_prepare_read_plan_layouts"](
        SimpleNamespace(storage=storage, pools={"kv": pool}, num_layers=3), transfers
    )
    keys, rows, packed, groups = layouts[0]
    assert keys == ["a", "b"] and packed
    for layer, items in enumerate(groups):
        legacy = scope["get_prepared_layer_range_meta"](pool, rows, layer)
        if legacy is None:
            assert items == []
            continue
        ptrs = [
            [base + row * stride for base, stride, size, offset in items]
            for row in rows
        ]
        sizes = [[size for base, stride, size, offset in items] for row in rows]
        offsets = [[offset for base, stride, size, offset in items] for row in rows]
        assert (ptrs, sizes, offsets) == legacy
