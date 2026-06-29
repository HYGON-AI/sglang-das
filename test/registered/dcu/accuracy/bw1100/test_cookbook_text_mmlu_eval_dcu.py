import os
import unittest

from sglang.test.ci.ci_register import register_dcu_ci
from sglang.test.dcu_cookbook_utils import (
    assert_cookbook_min_score,
    COOKBOOK_MMLU_EVAL_MODELS,
    run_cookbook_accuracy_eval,
    selected_configs,
)
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST

register_dcu_ci(
    est_time=7200,
    suite="nightly-dcu-accuracy-text",
    nightly=True,
)


COOKBOOK_MMLU_MIN_SCORE = {"Qwen3-32B": 0.80}


class TestCookbookTextMMLUEvalDCU(unittest.TestCase):
    def test_cookbook_text_mmlu(self):
        configs = selected_configs(
            COOKBOOK_MMLU_EVAL_MODELS, "SGLANG_DCU_COOKBOOK_MMLU_MODEL_FILTER"
        )
        num_examples = int(os.environ.get("SGLANG_DCU_COOKBOOK_MMLU_NUM_EXAMPLES", "100"))
        num_threads = int(os.environ.get("SGLANG_DCU_COOKBOOK_MMLU_NUM_THREADS", "128"))
        max_tokens = int(os.environ.get("SGLANG_DCU_COOKBOOK_MMLU_MAX_TOKENS", "2048"))

        for config in configs:
            with self.subTest(model=config.name):
                metrics = run_cookbook_accuracy_eval(
                    config,
                    DEFAULT_URL_FOR_TEST,
                    "mmlu",
                    num_examples=num_examples,
                    num_threads=num_threads,
                    max_tokens=max_tokens,
                )
                self.assertIn("score", metrics)
                assert_cookbook_min_score(
                    config,
                    metrics,
                    COOKBOOK_MMLU_MIN_SCORE,
                    "SGLANG_DCU_COOKBOOK_MMLU_MIN_SCORE",
                )


if __name__ == "__main__":
    unittest.main()
