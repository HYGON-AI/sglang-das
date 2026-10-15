"""CPU regression tests for bounded interleave hidden-state gathers."""

import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.cp import base, interleave

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class TestInterleaveCpGatherMemory(CustomTestCase):
    def _gather(self, *, cp_size=4, rows=7, rank=0, dtype=torch.bfloat16,
                hcu=True, small=False, uneven=False, indexed=False, kv=False,
                noncontiguous=False, padded=False):
        width = 3
        source = torch.arange(cp_size * rows * width, dtype=dtype).reshape(
            rows * cp_size, width
        )
        rank_inputs = [source[i::cp_size].contiguous() for i in range(cp_size)]
        logical = [rows] * cp_size
        if uneven:
            logical[-1] -= 1
            rank_inputs[-1][-1].zero_()
        actual = logical
        if padded:
            rank_inputs = [torch.cat((x, x.new_zeros((2, width)))) for x in rank_inputs]
            actual = [rows + 2] * cp_size
        gather_index = torch.tensor([0, rows, 1, rows + 1]) if indexed else None
        metadata = interleave.InterleaveContextParallelMetadata(
            total_seq_lens=sum(logical), per_rank_actual_token=actual,
            per_rank_logical_token=logical, gather_index=gather_index,
        )
        batch = SimpleNamespace(attn_cp_metadata=metadata)
        strategy = interleave.InterleaveCPStrategy(cp_size)
        calls, offset = [], 0

        def gather(output, local):
            nonlocal offset
            count = local.shape[0]
            torch.testing.assert_close(local, rank_inputs[rank][offset:offset + count])
            output.copy_(torch.cat([x[offset:offset + count] for x in rank_inputs]))
            calls.append((output.numel() * output.element_size(), output.data_ptr()))
            offset += count

        local = rank_inputs[rank][:logical[rank]]
        if noncontiguous:
            local = local.t().contiguous().t()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                interleave, "_is_hcu", hcu, create=True
            ))
            stack.enter_context(mock.patch.object(
                interleave, "_HCU_GATHER_CHUNK_BYTES", 48, create=True
            ))
            stack.enter_context(mock.patch.object(
                interleave, "_HCU_GATHER_CHUNK_THRESHOLD_BYTES",
                100000 if small else 1, create=True,
            ))
            stack.enter_context(mock.patch.object(
                base, "get_parallel", return_value=SimpleNamespace(attn_cp_rank=rank)
            ))
            stack.enter_context(mock.patch.object(
                interleave, "get_parallel",
                return_value=SimpleNamespace(attn_cp_group=None)
            ))
            stack.enter_context(mock.patch.object(
                interleave, "use_symmetric_memory",
                side_effect=lambda *a, **kw: nullcontext()
            ))
            stack.enter_context(mock.patch.object(
                interleave, "is_allocation_symmetric", return_value=False
            ))
            stack.enter_context(mock.patch.object(
                interleave, "attn_cp_all_gather_into_tensor", side_effect=gather
            ))
            gather_method = (
                strategy.gather_kv_cache if kv else strategy.gather_hidden_states
            )
            result = gather_method(local, batch)
        if indexed:
            expected = torch.cat(rank_inputs).index_select(0, gather_index)
        else:
            expected = source[:sum(logical)]
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        self.assertTrue(result.is_contiguous())
        return calls

    def test_bounded_scratch_preserves_order_and_final_partial_chunk(self):
        for dtype in (torch.bfloat16, torch.float32):
            for cp_size in (2, 4, 8):
                for rank in range(cp_size):
                    with self.subTest(dtype=dtype, cp_size=cp_size, rank=rank):
                        calls = self._gather(cp_size=cp_size, rank=rank, dtype=dtype)
                        self.assertGreater(len(calls), 1)
                        # Keep at least one row per rank even if it exceeds the cap.
                        cap = max(
                            48, cp_size * 3 * torch.empty((), dtype=dtype).element_size()
                        )
                        self.assertLessEqual(max(n for n, _ in calls), cap)
                        self.assertEqual(len({ptr for _, ptr in calls}), 1)

    def test_noncontiguous_local_input(self):
        self._gather(noncontiguous=True)

    def test_empty_rank_has_no_collective(self):
        self.assertEqual(self._gather(rows=0), [])

    def test_uneven_and_physically_padded_tail_preserves_order(self):
        for cp_size in (2, 4, 8):
            for rank in range(cp_size):
                for padded in (False, True):
                    with self.subTest(cp_size=cp_size, rank=rank, padded=padded):
                        calls = self._gather(
                            cp_size=cp_size, rank=rank, uneven=True, padded=padded
                        )
                        self.assertGreater(len(calls), 1)
                        self.assertEqual(len({ptr for _, ptr in calls}), 1)

    def test_small_non_hcu_indexed_and_kv_keep_original_gather(self):
        for kwargs in (dict(small=True), dict(hcu=False),
                       dict(indexed=True), dict(kv=True)):
            with self.subTest(**kwargs):
                self.assertEqual(len(self._gather(**kwargs)), 1)


if __name__ == "__main__":
    unittest.main()
