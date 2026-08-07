import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_output_processor_mixin import (  # noqa: E402
    SchedulerOutputProcessorMixin,
)

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class TestSchedulerOutputProcessor(unittest.TestCase):
    def test_prefill_converts_batched_logprob_tensors(self):
        logits_output = SimpleNamespace(
            next_token_logprobs=None,
            input_token_logprobs=None,
            input_top_logprobs_val=torch.tensor([[[-1.0, -2.0]]]),
            input_top_logprobs_idx=torch.tensor([[[7, 8]]]),
            input_token_ids_logprobs_val=torch.tensor([[[-3.0, -4.0]]]),
            next_token_top_logprobs_val=torch.tensor([[1.0, 2.0]]),
            next_token_top_logprobs_idx=torch.tensor([[3, 4]]),
            next_token_token_ids_logprobs_val=torch.tensor([[5.0, 6.0]]),
        )
        result = SimpleNamespace(
            copy_done=None,
            routed_experts_output=None,
            indexer_topk_output=None,
            logits_output=logits_output,
            next_token_ids=torch.empty(0, dtype=torch.int64),
            extend_input_len_per_req=[],
            extend_logprob_start_len_per_req=[],
        )
        batch = SimpleNamespace(
            return_logprob=True,
            reqs=[],
            prefill_stats=None,
            dp_cooperation_info=None,
        )
        scheduler = SimpleNamespace(
            is_generation=True,
            stream_output=MagicMock(),
            report_prefill_stats=MagicMock(),
        )

        SchedulerOutputProcessorMixin.process_batch_result_prefill(
            scheduler, batch, result
        )

        self.assertEqual(logits_output.next_token_top_logprobs_val, [[1.0, 2.0]])
        self.assertEqual(logits_output.next_token_top_logprobs_idx, [[3, 4]])
        self.assertEqual(logits_output.input_top_logprobs_val, [[[-1.0, -2.0]]])
        self.assertEqual(logits_output.input_top_logprobs_idx, [[[7, 8]]])
        self.assertEqual(logits_output.input_token_ids_logprobs_val, [[[-3.0, -4.0]]])
        self.assertEqual(logits_output.next_token_token_ids_logprobs_val, [[5.0, 6.0]])


if __name__ == "__main__":
    unittest.main()
