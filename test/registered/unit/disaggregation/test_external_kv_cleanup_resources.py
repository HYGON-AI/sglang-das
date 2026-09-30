"""Exercise real cleanup helpers and CPU pools, including partial side effects."""

import ast
import importlib.util
import threading
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

from test_external_kv_abort_drain import ROOT, register_cpu_ci


register_cpu_ci(est_time=2, suite="base-a-test-cpu")

def load_functions(path, names, env):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(funcs) == len(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *funcs,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), env)
    return env


path = ROOT / "python/sglang/srt/mem_cache/cleanup.py"
spec = importlib.util.spec_from_file_location("cleanup_under_test", path)
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


class TestResourceCleanup(unittest.TestCase):
    def setUp(self):
        self.req = SimpleNamespace(
            rid="partial",
            external_kv_abort_requested=True,
            external_kv_cleanup_steps={},
            metadata_buffer_index=0,
            disagg_kv_sender=None,
        )
        self.state = SimpleNamespace(src_indices=[0, 1])
        self.env = dict(
            run_cleanup_step=cleanup.run_cleanup_step,
            pd_hidden_state=lambda req: self.state,
        )
        load_functions(
            ROOT / "python/sglang/srt/disaggregation/prefill.py",
            ["maybe_release_metadata_buffer", "clear_pd_hidden_request_state"],
            self.env,
        )
        utils = ROOT / "python/sglang/srt/disaggregation/utils.py"
        # Real allocator methods; no sglang/GPU imports are needed.
        tree = ast.parse(utils.read_text(encoding="utf-8"))
        nodes = [
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef)
            and n.name in ("ReqToMetadataIdxAllocator", "PDHiddenRowPool")
        ]
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *nodes,
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)
        pool_env = dict(
            deque=deque,
            threading=threading,
            torch=SimpleNamespace(zeros=lambda *args, **kwargs: None),
        )
        exec(compile(module, str(utils), "exec"), pool_env)
        self.allocator = pool_env["ReqToMetadataIdxAllocator"](1)
        self.assertEqual(self.allocator.alloc(), 0)
        self.pool = pool_env["PDHiddenRowPool"](4, 8, None)
        self.assertEqual(self.pool.alloc(2), [0, 1])

    def release_metadata(self):
        self.env["maybe_release_metadata_buffer"](self.req, self.allocator, self.pool)

    def test_metadata_index_success_hidden_rows_retry_does_not_repeat_index(self):
        free = self.pool.free
        self.pool.free = Mock(
            side_effect=[cleanup.RetryableCleanupError("before free"), None]
        )
        with self.assertRaises(cleanup.RetryableCleanupError):
            self.release_metadata()
        self.assertEqual(self.req.metadata_buffer_index, -1)
        self.assertEqual(self.allocator.available_size(), 1)
        self.pool.free.side_effect = free
        self.release_metadata()
        self.assertEqual(self.allocator.available_size(), 1)
        self.assertEqual(self.pool.available_size(), 4)
        self.assertIsNone(self.state.src_indices)

    def test_metadata_free_then_exception_never_frees_reallocated_slot(self):
        free = self.allocator.free

        def fail_after_free(index):
            free(index)
            raise RuntimeError("after returning index to pool")

        self.allocator.free = Mock(side_effect=fail_after_free)
        with self.assertRaises(RuntimeError):
            self.release_metadata()
        self.assertEqual(self.allocator.alloc(), 0)  # A different owner reuses it.
        with self.assertRaises(cleanup.CleanupOutcomeUnknown):
            self.release_metadata()
        self.allocator.free.assert_called_once()
        self.assertEqual(self.allocator.available_size(), 0)
        self.assertEqual(self.pool.available_size(), 2)

    def test_worker_release_receipt_survives_later_state_clear_retry(self):
        self.pool.free([0, 1])  # Worker already freed the rows.
        self.req.disagg_kv_sender = SimpleNamespace(
            bootstrap_room=7,
            kv_mgr=SimpleNamespace(
                pop_pd_hidden_request_done=Mock(side_effect=[True, False])
            ),
        )
        clear = self.env["clear_pd_hidden_request_state"]
        self.env["clear_pd_hidden_request_state"] = Mock(
            side_effect=cleanup.RetryableCleanupError("before clearing state")
        )
        with self.assertRaises(cleanup.RetryableCleanupError):
            self.release_metadata()
        self.assertEqual(self.pool.alloc(2), [0, 1])  # Reused by another request.
        self.env["clear_pd_hidden_request_state"] = clear
        self.release_metadata()
        self.req.disagg_kv_sender.kv_mgr.pop_pd_hidden_request_done.assert_called_once()
        self.assertEqual(self.pool.available_size(), 2)

    def setup_kv(self):
        self.req.req_pool_idx = 1
        self.req.kv = SimpleNamespace(kv_allocated_len=4)
        self.req.mamba_pool_idx = None
        self.req.cache_protected_len = 0
        self.req.skip_radix_cache_insert = True
        self.req.effective_kv_committed_len = lambda: 2
        self.req.dsv4_donated_swa_len = 0
        # Select ReqToTokenPool.free explicitly, since the file has other pools.
        tree = ast.parse(
            (ROOT / "python/sglang/srt/mem_cache/memory_pool.py").read_text(
                encoding="utf-8"
            )
        )
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "ReqToTokenPool"
        )
        free = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "free"
        )
        env = {}
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                free,
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)
        exec(compile(module, "pool_free", "exec"), env)

        class Slots:
            def __getitem__(self, key):
                if isinstance(key, tuple):
                    return [10, 11, 12, 13][key[1]]
                return [10, 11, 12, 13]

        pool = SimpleNamespace(free_slots=[], req_to_token=Slots())
        pool.free = lambda req, free=env["free"]: free(pool, req)
        allocator = SimpleNamespace(page_size=1, free=Mock(), free_segment=Mock())
        # Execute ChunkCache.cache_finished_req itself, not a mock of that action.
        chunk_tree = ast.parse(
            (ROOT / "python/sglang/srt/mem_cache/chunk_cache.py").read_text(
                encoding="utf-8"
            )
        )
        chunk_cls = next(
            n
            for n in chunk_tree.body
            if isinstance(n, ast.ClassDef) and n.name == "ChunkCache"
        )
        finished = next(
            n
            for n in chunk_cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "cache_finished_req"
        )
        mod = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                finished,
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(mod)
        cache_env = {}
        exec(compile(mod, "chunk_cache", "exec"), cache_env)
        cache = SimpleNamespace(
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            supports_mamba=lambda: False,
        )
        cache.cache_finished_req = lambda *args, **kwargs: cache_env[
            "cache_finished_req"
        ](cache, *args, **kwargs)
        env = dict(
            run_cleanup_step=cleanup.run_cleanup_step,
            HybridReqToTokenPool=type("Hybrid", (), {}),
            get_spec=lambda: SimpleNamespace(speculative_algorithm="test"),
        )
        load_functions(
            ROOT / "python/sglang/srt/mem_cache/common.py",
            [
                "release_kv_cache",
                "_release_aborted_kv_cache",
                "_release_overallocated_kv_indices",
                "_release_donated_swa_slots",
            ],
            env,
        )
        self.kv_env, self.cache, self.token_allocator = env, cache, allocator
        return env["release_kv_cache"]

    def test_cache_success_then_tail_retry_skips_actual_chunk_cache_free(self):
        release = self.setup_kv()
        self.token_allocator.free_segment.side_effect = [
            cleanup.RetryableCleanupError("before tail free"),
            None,
        ]
        with self.assertRaises(cleanup.RetryableCleanupError):
            release(self.req, self.cache, is_insert=False)
        self.token_allocator.free.assert_called_once_with([10, 11])
        release(self.req, self.cache, is_insert=False)
        self.token_allocator.free.assert_called_once()
        self.assertEqual(self.token_allocator.free_segment.call_count, 2)
        self.assertEqual(self.cache.req_to_token_pool.free_slots, [1])
        self.assertIsNone(self.req.kv)

    def test_partial_cache_free_unknown_outcome_is_not_repeated(self):
        release = self.setup_kv()
        returned = []

        def partial(indices):
            returned.append(indices[0])
            raise RuntimeError("some pages already freed")

        self.token_allocator.free.side_effect = partial
        with self.assertRaises(RuntimeError):
            release(self.req, self.cache, is_insert=False)
        with self.assertRaises(cleanup.CleanupOutcomeUnknown):
            release(self.req, self.cache, is_insert=False)
        self.assertEqual(returned, [10])
        self.token_allocator.free_segment.assert_not_called()
        self.assertEqual(self.cache.req_to_token_pool.free_slots, [])

    def test_request_slot_returned_then_exception_never_duplicates_free_slot(self):
        release = self.setup_kv()
        free = self.cache.req_to_token_pool.free

        def fail_after_free(req):
            free(req)
            raise RuntimeError("after request slot was returned")

        self.cache.req_to_token_pool.free = fail_after_free
        with self.assertRaises(RuntimeError):
            release(self.req, self.cache, is_insert=False)
        self.assertIsNone(self.req.req_pool_idx)
        with self.assertRaises(cleanup.CleanupOutcomeUnknown):
            release(self.req, self.cache, is_insert=False)
        self.assertEqual(self.cache.req_to_token_pool.free_slots, [1])
        self.token_allocator.free.assert_called_once()
        self.token_allocator.free_segment.assert_called_once()
        self.assertIsNotNone(self.req.kv)


if __name__ == "__main__":
    unittest.main()
