import unittest
from unittest.mock import Mock

import torch

from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
    DSparkWorkerV2,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDsparkWorkerPdPrefill(CustomTestCase):
    def test_shared_decode_modules_are_not_attached(self):
        worker = object.__new__(DSparkWorkerV2)
        worker._is_pd_prefill = True
        worker.draft_model = Mock()

        worker._attach_shared_modules()

        worker.draft_model.attach_shared_modules.assert_not_called()

    def test_prefill_keeps_hidden_states_for_remote_decode(self):
        worker = object.__new__(DSparkWorkerV2)
        worker._is_pd_prefill = True
        worker._target_worker = Mock()

        hidden_states = torch.randn(4, 3)
        logits_output = Mock(hidden_states=hidden_states)
        batch_output = Mock(
            logits_output=logits_output,
            next_token_ids=torch.tensor([7], dtype=torch.int64),
        )
        worker._target_worker.forward_batch_generation.return_value = batch_output

        batch = Mock()
        batch.forward_mode.is_idle.return_value = False
        batch.seq_lens = torch.tensor([4], dtype=torch.int64)

        result = worker._forward_prefill(batch, on_publish=None)

        self.assertIs(result, batch_output)
        self.assertIs(result.logits_output.hidden_states, hidden_states)
        self.assertIsNotNone(result.next_draft_input)


if __name__ == "__main__":
    unittest.main()
