import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class TestTokenizerManagerLogprobs(unittest.TestCase):
    def test_detokenizes_tensor_top_logprobs_without_tensor_truthiness(self):
        manager = object.__new__(TokenizerManager)

        result = manager.detokenize_top_logprobs_tokens(
            [torch.tensor([-0.5, -1.5]), torch.tensor([])],
            [torch.tensor([5, 7]), torch.tensor([], dtype=torch.int64)],
            decode_to_text=False,
        )

        self.assertEqual(result, [[(-0.5, 5, None), (-1.5, 7, None)], None])


if __name__ == "__main__":
    unittest.main()
