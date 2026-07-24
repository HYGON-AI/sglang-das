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
from sglang.test.hcu_cookbook_utils import (
    assert_cookbook_min_score,
    COOKBOOK_MMLU_EVAL_MODELS,
    run_cookbook_accuracy_eval,
    selected_configs,
)
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST

register_hcu_ci(
    est_time=7200,
    suite="nightly-hcu-accuracy-text",
    nightly=True,
)


COOKBOOK_MMLU_MIN_SCORE = {"Qwen3-32B": 0.80}


class TestCookbookTextMMLUEvalHCU(unittest.TestCase):
    def test_cookbook_text_mmlu(self):
        configs = selected_configs(
            COOKBOOK_MMLU_EVAL_MODELS, "SGLANG_HCU_COOKBOOK_MMLU_MODEL_FILTER"
        )
        num_examples = int(os.environ.get("SGLANG_HCU_COOKBOOK_MMLU_NUM_EXAMPLES", "100"))
        num_threads = int(os.environ.get("SGLANG_HCU_COOKBOOK_MMLU_NUM_THREADS", "128"))
        max_tokens = int(os.environ.get("SGLANG_HCU_COOKBOOK_MMLU_MAX_TOKENS", "2048"))

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
                    "SGLANG_HCU_COOKBOOK_MMLU_MIN_SCORE",
                )


if __name__ == "__main__":
    unittest.main()
