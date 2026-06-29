import os
import unittest

from sglang.test.ci.ci_register import register_dcu_ci
from sglang.test.dcu_cookbook_utils import (
    assert_cookbook_min_score,
    COOKBOOK_GSM8K_EVAL_MODELS,
    run_cookbook_accuracy_eval,
    selected_configs,
)
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST

register_dcu_ci(
    est_time=7200,
    suite="nightly-dcu-accuracy-text",
    nightly=True,
)


COOKBOOK_GSM8K_MIN_SCORE = {"Qwen3-32B": 0.90, "Qwen3-30B-A3B": 0.88}


class TestCookbookTextGSM8KEvalDCU(unittest.TestCase):
    def test_cookbook_text_gsm8k(self):
        configs = selected_configs(
            COOKBOOK_GSM8K_EVAL_MODELS, "SGLANG_DCU_COOKBOOK_GSM8K_MODEL_FILTER"
        )
        num_examples = int(os.environ.get("SGLANG_DCU_COOKBOOK_GSM8K_NUM_EXAMPLES", "100"))
        num_threads = int(os.environ.get("SGLANG_DCU_COOKBOOK_GSM8K_NUM_THREADS", "128"))
        num_shots = int(os.environ.get("SGLANG_DCU_COOKBOOK_GSM8K_NUM_SHOTS", "5"))
        max_tokens = int(os.environ.get("SGLANG_DCU_COOKBOOK_GSM8K_MAX_TOKENS", "2048"))

        for config in configs:
            with self.subTest(model=config.name):
                metrics = run_cookbook_accuracy_eval(
                    config,
                    DEFAULT_URL_FOR_TEST,
                    "gsm8k",
                    num_examples=num_examples,
                    num_threads=num_threads,
                    num_shots=num_shots,
                    max_tokens=max_tokens,
                )
                self.assertIn("score", metrics)
                assert_cookbook_min_score(
                    config,
                    metrics,
                    COOKBOOK_GSM8K_MIN_SCORE,
                    "SGLANG_DCU_COOKBOOK_GSM8K_MIN_SCORE",
                )


if __name__ == "__main__":
    unittest.main()
