# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest

from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.hcu_accuracy_report import write_hcu_accuracy_result
from sglang.test.hcu_cookbook_utils import (
    COOKBOOK_GSM8K_EVAL_MODELS,
    assert_cookbook_min_score,
    get_cookbook_threshold,
    run_cookbook_accuracy_eval,
    selected_configs,
)
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST

register_hcu_ci(
    est_time=7200,
    suite="nightly-hcu-accuracy-text",
    nightly=True,
)


COOKBOOK_GSM8K_MIN_SCORE = {
    "Qwen3-32B": 0.90,
    "Qwen3-30B-A3B": 0.93,
    "Qwen3.6-35B-A3B": 0.93,
}
COOKBOOK_GSM8K_MODEL_KEYS = {
    "Qwen3-32B": "qwen3_32b",
    "Qwen3-30B-A3B": "qwen3_30b_a3b",
    "Qwen3.6-35B-A3B": "qwen36_35b_a3b",
}


class TestCookbookTextGSM8KEvalHCU(unittest.TestCase):
    def test_cookbook_text_gsm8k(self):
        configs = selected_configs(
            COOKBOOK_GSM8K_EVAL_MODELS, "SGLANG_HCU_COOKBOOK_GSM8K_MODEL_FILTER"
        )
        num_examples = int(
            os.environ.get("SGLANG_HCU_COOKBOOK_GSM8K_NUM_EXAMPLES", "100")
        )
        num_threads = int(
            os.environ.get("SGLANG_HCU_COOKBOOK_GSM8K_NUM_THREADS", "128")
        )
        num_shots = int(os.environ.get("SGLANG_HCU_COOKBOOK_GSM8K_NUM_SHOTS", "5"))
        max_tokens = int(os.environ.get("SGLANG_HCU_COOKBOOK_GSM8K_MAX_TOKENS", "2048"))

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
                threshold = get_cookbook_threshold(
                    config,
                    COOKBOOK_GSM8K_MIN_SCORE,
                    "SGLANG_HCU_COOKBOOK_GSM8K_MIN_SCORE",
                )
                if threshold is None:
                    raise AssertionError(f"missing GSM8K threshold for {config.name}")
                write_hcu_accuracy_result(
                    model_key=COOKBOOK_GSM8K_MODEL_KEYS[config.name],
                    model=config.name,
                    score=metrics["score"],
                    threshold=threshold,
                    num_examples=num_examples,
                    invalid_rate=metrics.get("invalid"),
                    latency_seconds=metrics.get("latency"),
                    source_test=__file__,
                )
                assert_cookbook_min_score(
                    config,
                    metrics,
                    COOKBOOK_GSM8K_MIN_SCORE,
                    "SGLANG_HCU_COOKBOOK_GSM8K_MIN_SCORE",
                )


if __name__ == "__main__":
    unittest.main()
