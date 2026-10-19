"""LayerSplit fused writers preserve replicated latents/state without owning KV."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.kernels.ops.attention.dsv4.low_ratio_compress import (
    c1_decode_norm_rope_store,
    c2_prefill_norm_rope_store,
)
from sglang.srt.mem_cache.cp_cache_layer_split.deepseek_v4_pool import (
    CpCacheLayerSplitDeepSeekV4TokenToKVPool as SplitPool,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")


class TestLayerSplitWriteDestination(unittest.TestCase):
    def test_owner_uses_persistent_buffer_not_read_staging(self):
        pool = SplitPool.__new__(SplitPool)
        pool.compression_ratios = [0, 0, 2]
        persistent = object()
        kv_pool = Mock()
        kv_pool.get_key_buffer.return_value = persistent
        with (
            patch.object(SplitPool, "_owns_extra_key_layer_id", return_value=True),
            patch.object(SplitPool, "get_extra_key_buffer", side_effect=AssertionError("read staging")),
            patch.object(SplitPool, "wait_layer_transfer"),
            patch.object(SplitPool, "_owner_compress_layer_item", return_value=SimpleNamespace(compress_kv_pool=kv_pool, compress_layer_id=0)),
        ):
            self.assertIs(pool.get_extra_key_buffer_for_write(2), persistent)
            kv_pool.get_key_buffer.assert_called_once_with(0)

    def test_non_owner_does_not_read_or_allocate_persistent_kv(self):
        pool = SplitPool.__new__(SplitPool)
        with (
            patch.object(SplitPool, "_owns_extra_key_layer_id", return_value=False),
            patch.object(DeepSeekV4TokenToKVPool, "get_extra_key_buffer", side_effect=AssertionError("unowned")),
        ):
            self.assertIsNone(pool.get_extra_key_buffer_for_write(2))


@unittest.skipUnless(torch.cuda.is_available(), "requires GPU")
class TestLayerSplitFusedStore(unittest.TestCase):
    def test_owner_nonowner_latents_and_chunk_state(self):
        torch.manual_seed(1234)
        device = "cuda"
        weight = torch.randn(512, device=device, dtype=torch.bfloat16)
        angles = torch.randn(64, 32, device=device)
        freqs = torch.view_as_real(torch.polar(torch.ones_like(angles), angles)).flatten(-2)
        for layout in (KVLayout.V4, KVLayout.V41, KVLayout.V41_FP4):
            with self.subTest(layout=layout):
                cache = torch.zeros((4, layout.page_bytes(64)), device=device, dtype=torch.uint8)
                x = torch.randn(8, 512, device=device, dtype=torch.bfloat16)
                pos = torch.arange(8, device=device, dtype=torch.int64)
                loc = pos + 64
                loc[-1] = 0
                a = c1_decode_norm_rope_store(x, weight, pos, loc, 1e-6, freqs, cache, page_size=64, layout=layout)
                b = c1_decode_norm_rope_store(x, weight, pos, loc, 1e-6, freqs, None, page_size=64, layout=layout)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                self.assertTrue(bool(cache.any()))

                states = [torch.zeros((16, 1024), device=device) for _ in range(2)]
                # An odd chunk boundary carries a pending partner into the next
                # call; a chunk longer than the ring exercises write-back too.
                start = 0
                for n in (5, 11, 3):
                    x = torch.randn(n + 1, 1024, device=device)
                    pos = torch.cat((torch.arange(start, start + n, device=device), torch.zeros(1, device=device))).to(torch.int64)
                    loc = pos + 64
                    loc[-1] = 0
                    req = torch.zeros_like(pos)
                    starts = torch.tensor([0], device=device, dtype=torch.int32)
                    lens = torch.tensor([n], device=device, dtype=torch.int32)
                    outs = []
                    for state, dest in zip(states, (cache, None)):
                        outs.append(c2_prefill_norm_rope_store(x, state, weight, pos, req, loc, 1e-6, freqs, dest, starts, lens, page_size=64, ring_size=8, layout=layout))
                    valid = (pos % 2 == 1) & (loc != 0)
                    torch.testing.assert_close(outs[0][valid], outs[1][valid], rtol=0, atol=0)
                    torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)
                    start += n
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
