"""Regression tests for HCU-only ``ForwardBatch`` state."""

import unittest

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestForwardBatchHcuState(CustomTestCase):
    def test_fused_rms_quant_flag_defaults_to_false(self):
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=1,
            input_ids=torch.zeros(1, dtype=torch.long),
            req_pool_indices=torch.zeros(1, dtype=torch.long),
            seq_lens=torch.ones(1, dtype=torch.long),
            out_cache_loc=torch.zeros(1, dtype=torch.long),
            seq_lens_sum=1,
        )

        self.assertFalse(forward_batch.rms_quant_flag)


if __name__ == "__main__":
    unittest.main()
