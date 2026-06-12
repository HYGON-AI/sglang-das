# DCU-specific smoke for test/registered/core/test_score_api.py.
# Keeps a small Engine.score coverage point with explicit DCU backend settings.

import unittest

import torch

from sglang.srt.entrypoints.engine import Engine
from sglang.test.ci.ci_register import register_dcu_ci
from sglang.test.dcu_utils import get_model_path
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

register_dcu_ci(
    est_time=600,
    suite="stage-b-test-1-gpu-small-dcu",
    disabled=(
        "DCU disabled retest candidate: verify whether score API passes with "
        "explicit fa3/page-size-64 server settings before enabling."
    ),
)

DEFAULT_DCU_SCORE_MODEL = DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestScoreAPIDCU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = get_model_path("SGLANG_DCU_SCORE_MODEL", DEFAULT_DCU_SCORE_MODEL)

    def setUp(self):
        self.engine = Engine(
            model_path=self.model,
            attention_backend="fa3",
            page_size=64,
            trust_remote_code=True,
            log_level="warning",
        )

    def tearDown(self):
        self.engine.shutdown()
        torch.cuda.empty_cache()

    def test_score_basic(self):
        result = self.engine.score(
            query="The capital of France is",
            items=[" Paris", " Berlin"],
            label_token_ids=[1, 2, 3],
            apply_softmax=True,
        )

        self.assertEqual(len(result.scores), 2)
        self.assertGreater(result.prompt_tokens, 0)
        for score_list in result.scores:
            self.assertEqual(len(score_list), 3)
            self.assertTrue(all(isinstance(value, float) for value in score_list))
            self.assertAlmostEqual(sum(score_list), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
