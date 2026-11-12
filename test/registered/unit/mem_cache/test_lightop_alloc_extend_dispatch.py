"""HCU extend routing and capacity checks; GPU kernel equality is tested separately."""

import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.allocator import paged
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLightOpAllocExtendDispatch(unittest.TestCase):
    def call(self, bs, ext, *, enabled=True, available=None, lightop_missing=False):
        p = 256
        need = bs * ((ext + p - 1) // p)
        n = need + bs + 2 if available is None else available
        a = object.__new__(paged.PagedTokenToKVPoolAllocator)
        a.page_size = p
        a.device = "cpu"
        a.debug_mode = False
        a.sglang_kvalloc_kernel = True
        a.lightop_kvalloc_kernel = enabled
        a.use_triton_kernels = False
        a.free_pages = torch.arange(100, 100 + n, dtype=torch.int64)
        a.staged_pages = []
        a.num_staged_pages = 0
        original = a.free_pages.clone()
        pre = torch.zeros(bs, dtype=torch.int64)
        seq = pre + ext
        last = pre - 1

        def reference(**k):
            paged.alloc_extend_naive(
                k["pre_lens_ptr"],
                k["seq_lens_ptr"],
                k["last_loc_ptr"],
                k["free_page_ptr"],
                k["out_indices"],
                k["page_size"],
                "cpu",
            )

        with (
            patch.object(
                paged,
                "sgl_kernel_alloc_extend_kernel",
                side_effect=reference,
                create=True,
            ) as old,
            patch.object(paged, "_get_lightop_alloc_extend_kernel") as get,
        ):
            fast = Mock(side_effect=reference)
            get.return_value = None if lightop_missing else fast
            out = a.alloc_extend(pre, pre, seq, seq, last, bs * ext)
            if need > n:
                self.assertIsNone(out)
                old.assert_not_called()
                get.assert_not_called()
                self.assertTrue(torch.equal(a.free_pages, original))
            else:
                expect = torch.empty(bs * ext, dtype=torch.int64)
                paged.alloc_extend_naive(pre, seq, last, original, expect, p, "cpu")
                self.assertTrue(torch.equal(out, expect))
                self.assertTrue(torch.equal(a.free_pages, original[need:]))
            return (old.call_count, fast.call_count, get.call_count)

    def test_supported_long_extend(self):
        for bs, ext in [(1, 8192), (1, 32768), (64, 8192)]:
            self.assertEqual(self.call(bs, ext), (0, 1, 1))

    def test_tiny_extend_falls_back(self):
        self.assertEqual(self.call(1, 127), (1, 0, 0))

    def test_short_request_batch_falls_back(self):
        for bs, ext in [(1, 8191), (64, 4096)]:
            self.assertEqual(self.call(bs, ext), (1, 0, 0))

    def test_large_batch_falls_back(self):
        for bs in [65, 128]:
            self.assertEqual(self.call(bs, 8192), (1, 0, 0))

    def test_explicit_disable(self):
        self.assertEqual(self.call(1, 32768, enabled=False), (1, 0, 0))

    def test_missing_lightop(self):
        self.assertEqual(self.call(1, 32768, lightop_missing=True), (1, 0, 1))

    def test_short_pool_before_dispatch(self):
        self.assertEqual(self.call(1, 32768, available=127), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
