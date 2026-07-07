import os
import unittest

from sglang.test.ci.ci_register import register_dcu_ci
from sglang.test.dcu_cookbook_utils import (
    assert_cookbook_min_score,
    COOKBOOK_MMMU_EVAL_MODELS,
    run_cookbook_accuracy_eval,
    selected_configs,
)
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST

register_dcu_ci(
    est_time=7200,
    suite="nightly-dcu-vlm",
    nightly=True,
)


COOKBOOK_MMMU_MIN_SCORE = {
    "Qwen3-VL-4B-Instruct": 0.25,
    "Qwen3-VL-32B-Instruct": 0.23,
}


class TestCookbookMMMUEvalDCU(unittest.TestCase):
    def test_cookbook_mmmu(self):
        configs = selected_configs(
            COOKBOOK_MMMU_EVAL_MODELS, "SGLANG_DCU_COOKBOOK_MMMU_MODEL_FILTER"
        )
        num_examples = int(os.environ.get("SGLANG_DCU_COOKBOOK_MMMU_NUM_EXAMPLES", "100"))
        num_threads = int(os.environ.get("SGLANG_DCU_COOKBOOK_MMMU_NUM_THREADS", "4"))
        max_tokens = int(os.environ.get("SGLANG_DCU_COOKBOOK_MMMU_MAX_TOKENS", "64"))

        for config in configs:
            with self.subTest(model=config.name):
                metrics = run_cookbook_accuracy_eval(
                    config,
                    DEFAULT_URL_FOR_TEST,
                    "mmmu",
                    num_examples=num_examples,
                    num_threads=num_threads,
                    max_tokens=max_tokens,
                )
                self.assertIn("score", metrics)
                assert_cookbook_min_score(
                    config,
                    metrics,
                    COOKBOOK_MMMU_MIN_SCORE,
                    "SGLANG_DCU_COOKBOOK_MMMU_MIN_SCORE",
                )


if __name__ == "__main__":
    unittest.main()
