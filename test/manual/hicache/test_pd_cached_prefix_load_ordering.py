"""CPU regression tests executing the real source method and scheduler gate.

AST extraction avoids importing device dependencies; no method body is reimplemented.
"""

import ast
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[3] / "python/sglang/srt"


def method(path, name):
    tree = ast.parse((ROOT / path).read_text())
    f = next(
        n
        for c in tree.body
        if isinstance(c, ast.ClassDef)
        for n in c.body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    f.decorator_list = []
    return f


ns = {
    "Scheduler": object,
    "Req": object,
    "envs": SimpleNamespace(
        SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX=SimpleNamespace(get=lambda: True)
    ),
}
exec(
    compile(
        ast.Module(
            body=[
                method("disaggregation/prefill.py", "maybe_send_cached_prefix_chunk")
            ],
            type_ignores=[],
        ),
        "source_prefill",
        "exec",
    ),
    ns,
)
run = method("managers/scheduler.py", "run_batch")
block = next(
    n
    for n in run.body
    if isinstance(n, ast.If)
    and any(
        isinstance(x, ast.Attribute) and x.attr == "maybe_send_cached_prefix_chunk"
        for x in ast.walk(n)
    )
)
f = ast.FunctionDef(
    name="early_send_block",
    args=ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg="self"), ast.arg(arg="batch")],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
    ),
    body=[block],
    decorator_list=[],
)
gate = {"DisaggregationMode": SimpleNamespace(PREFILL="prefill")}
exec(
    compile(
        ast.fix_missing_locations(ast.Module(body=[f], type_ignores=[])),
        "source_scheduler",
        "exec",
    ),
    gate,
)


def request(rid="shared", n=512, host=0, bootstrap=False):
    return SimpleNamespace(
        rid=rid,
        prefix_indices=[0] * n,
        host_hit_length=host,
        pending_bootstrap=bootstrap,
        start_send_idx=0,
        early_send_prefix_end=None,
    )


def scheduler(staging=False):
    reads = []
    s = SimpleNamespace(
        disaggregation_mode="prefill",
        enable_staging=staging,
        enable_overlap=False,
        token_to_kv_pool_allocator=SimpleNamespace(page_size=256),
    )
    s.send_kv_chunk = lambda r, **kwargs: reads.append((r.rid, kwargs["end_idx"]))
    s.maybe_send_cached_prefix_chunk = MethodType(
        ns["maybe_send_cached_prefix_chunk"], s
    )
    return s, reads


class OrderingTests(unittest.TestCase):
    def test_first_load_and_shared_device_hit_are_both_deferred(self):
        for index in [0, 1, 25]:
            s, reads = scheduler()
            reqs = [request("load_owner", host=512), request()]
            gate["early_send_block"](
                s, SimpleNamespace(reqs=reqs, hicache_consumer_index=index)
            )
            self.assertEqual(reads, [])

    def test_warm_device_hit_keeps_early_send(self):
        s, reads = scheduler()
        gate["early_send_block"](
            s, SimpleNamespace(reqs=[request()], hicache_consumer_index=-1)
        )
        self.assertEqual(reads, [("shared", 512)])

    def test_staging_snapshot_survives_deferred_first_batch(self):
        s, reads = scheduler(staging=True)
        r = request(n=768, host=256)
        gate["early_send_block"](s, SimpleNamespace(reqs=[r], hicache_consumer_index=0))
        self.assertEqual(reads, [])
        self.assertEqual(r.early_send_prefix_end, 512)
        r.prefix_indices = [0] * 1024
        r.host_hit_length = 0
        s.maybe_send_cached_prefix_chunk(r)
        self.assertEqual(reads, [("shared", 512)])
        self.assertEqual(r.early_send_prefix_end, 512)

    def test_bootstrap_still_defers_and_snapshots(self):
        s, reads = scheduler(staging=True)
        r = request(bootstrap=True)
        s.maybe_send_cached_prefix_chunk(r)
        self.assertEqual(reads, [])
        self.assertEqual(r.early_send_prefix_end, 512)

    def test_unaligned_prefix_still_defers(self):
        s, reads = scheduler()
        s.maybe_send_cached_prefix_chunk(request(n=511))
        self.assertEqual(reads, [])

    def test_already_sent_prefix_is_not_repeated(self):
        s, reads = scheduler()
        r = request()
        r.start_send_idx = 512
        s.maybe_send_cached_prefix_chunk(r)
        self.assertEqual(reads, [])

    def test_decode_does_not_early_send(self):
        s, reads = scheduler()
        s.disaggregation_mode = "decode"
        gate["early_send_block"](
            s, SimpleNamespace(reqs=[request()], hicache_consumer_index=-1)
        )
        self.assertEqual(reads, [])

    def test_overlap_warm_hit_keeps_source_event(self):
        records = []

        class Event:
            def record(self, stream):
                records.append(stream)

        previous = ns.get("torch")
        ns["torch"] = SimpleNamespace(cuda=SimpleNamespace(Event=Event))
        try:
            s, reads = scheduler()
            s.enable_overlap = True
            s.forward_stream = object()
            r = request()
            r.disagg_kv_sender = SimpleNamespace()
            s.maybe_send_cached_prefix_chunk(r)
            self.assertEqual(records, [s.forward_stream])
            self.assertIsInstance(r.disagg_kv_sender._early_send_wait_event, Event)
            self.assertEqual(reads, [("shared", 512)])
        finally:
            if previous is None:
                ns.pop("torch")
            else:
                ns["torch"] = previous

    def test_global_disable_still_works(self):
        control = ns["envs"].SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX
        before = control.get
        control.get = lambda: False
        try:
            s, reads = scheduler(staging=True)
            r = request()
            s.maybe_send_cached_prefix_chunk(r)
            self.assertEqual(reads, [])
            self.assertIsNone(r.early_send_prefix_end)
        finally:
            control.get = before


if __name__ == "__main__":
    unittest.main(verbosity=2)
