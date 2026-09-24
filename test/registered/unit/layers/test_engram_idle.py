"""Regression tests for padded Engram IDLE batches."""

from types import SimpleNamespace

import torch

from sglang.srt.layers.engram import EngramHasher
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestEngramIdle(CustomTestCase):
    def test_padded_idle_returns_dummy_hashes_without_committing_history(self):
        hasher = EngramHasher.__new__(EngramHasher)
        torch.nn.Module.__init__(hasher)
        hasher.max_ngram_size = 4
        hasher.history = torch.full((3, 3), 17, dtype=torch.int32)
        hasher.primes = torch.empty((2, 3, 4), dtype=torch.int64)
        hasher.offsets = torch.empty((2, 12), dtype=torch.int64)

        history_before = hasher.history.clone()
        input_ids = torch.tensor([11, 22, 33], dtype=torch.int64)
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.IDLE,
            req_pool_indices=torch.empty(0, dtype=torch.int64),
        )

        output = hasher(input_ids, forward_batch)

        self.assertEqual(output.shape, (3, 2, 12))
        self.assertEqual(output.dtype, torch.int64)
        self.assertEqual(output.device, input_ids.device)
        self.assertTrue(torch.count_nonzero(output).item() == 0)
        self.assertTrue(torch.equal(hasher.history, history_before))


if __name__ == "__main__":
    import unittest

    unittest.main()
