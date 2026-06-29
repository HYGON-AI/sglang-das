# DCU-specific copy of test/registered/models/test_reward_models.py.
# Changes:
# - use a local BW1100 reward model path override
# - force trust_remote_code for the local replacement model
# - keep one small SRT reward smoke; the HF custom model path is incompatible
#   with the container transformers DynamicCache API

import unittest

import torch

from sglang.test.ci.ci_register import register_dcu_ci
from sglang.test.dcu_utils import get_model_path
from sglang.test.runners import SRTRunner
from sglang.test.test_utils import CustomTestCase

register_dcu_ci(
    est_time=600,
    suite="stage-b-test-1-gpu-small-dcu",
)
register_dcu_ci(est_time=600, suite="nightly-dcu-api-models", nightly=True)

DEFAULT_DCU_REWARD_MODEL = (
    "/public/opendas/DL_DATA/llm-models/internlm2/internlm2-1.8b-reward"
)

PROMPT = "What is the range of the numeric output of a sigmoid node in a neural network?"
RESPONSE1 = "The output of a sigmoid node is bounded between -1 and 1."
RESPONSE2 = "The output of a sigmoid node is bounded between 0 and 1."

CONVS = [
    [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": RESPONSE1}],
    [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": RESPONSE2}],
]


class TestRewardModelsDCU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = get_model_path("SGLANG_DCU_REWARD_MODEL", DEFAULT_DCU_REWARD_MODEL)

    def test_reward_scores(self):
        with SRTRunner(
            self.model,
            torch_dtype=torch.float16,
            model_type="reward",
            attention_backend="fa3",
            page_size=64,
            trust_remote_code=True,
            json_model_override_args={"rope_scaling": None},
        ) as srt_runner:
            prompts = srt_runner.tokenizer.apply_chat_template(
                CONVS, tokenize=False, return_dict=False
            )
            srt_outputs = srt_runner.forward(prompts)

        srt_scores = torch.tensor(srt_outputs.scores)
        print("srt_scores={!r}".format(srt_scores))
        self.assertEqual(srt_scores.numel(), len(CONVS))
        self.assertTrue(torch.all(torch.isfinite(srt_scores)))


if __name__ == "__main__":
    unittest.main()
