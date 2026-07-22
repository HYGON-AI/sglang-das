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

# HCU-specific smoke for test/registered/core/test_score_api.py.
# Keeps a small Engine.score coverage point with explicit HCU backend settings.

import unittest

import torch

from sglang.srt.entrypoints.engine import Engine
from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.hcu_utils import get_model_path
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

register_hcu_ci(
    est_time=600,
    suite="stage-b-test-1-hcu-small",
    disabled=(
        "HCU disabled retest candidate: verify whether score API passes with "
        "explicit fa3/page-size-64 server settings before enabling."
    ),
)

DEFAULT_HCU_SCORE_MODEL = DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestScoreAPIHCU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = get_model_path("SGLANG_HCU_SCORE_MODEL", DEFAULT_HCU_SCORE_MODEL)

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
