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

import unittest
from types import SimpleNamespace

from sglang.srt.utils import is_hip, kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_hcu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_ACCURACY_TEST_FP8,
    DEFAULT_MODEL_NAME_FOR_DYNAMIC_QUANT_ACCURACY_TEST_FP8,
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=250, suite="stage-b-test-1-gpu-large")
register_amd_ci(est_time=600, suite="stage-b-test-1-gpu-small-amd")


# HCU_CSV_COVERED_UNVERIFIED: Enabled from sglang.csv historical HCU coverage; not re-tested in this framework pass.
register_hcu_ci(
    est_time=120,
    suite="stage-b-test-1-gpu-small-hcu",
    disabled="HCU PR baseline deferred: quantization path needs BW1100 numeric/backend validation before required CI.",
)

class TestEvalFP8Accuracy(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_ACCURACY_TEST_FP8
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model, cls.base_url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
            temperature=0.1,
        )

        metrics = run_eval(args)
        if is_hip():
            # Another threshold for AMD because fp8 dtype is difference
            self.assertGreaterEqual(metrics["score"], 0.60)
        else:
            self.assertGreaterEqual(metrics["score"], 0.60)


class TestEvalFP8DynamicQuantAccuracy(CustomTestCase):

    def _run_test(self, model, other_args, expected_score):
        base_url = DEFAULT_URL_FOR_TEST
        other_args = other_args or []

        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
                temperature=0.1,
            )

            metrics = run_eval(args)
            self.assertGreaterEqual(metrics["score"], expected_score)
        finally:
            kill_process_tree(process.pid)

    def test_mmlu_offline_only(self):
        """Test with offline quantization only."""
        self._run_test(
            model=DEFAULT_MODEL_NAME_FOR_DYNAMIC_QUANT_ACCURACY_TEST_FP8,
            other_args=[],
            expected_score=0.64,
        )

    def test_mmlu_offline_and_online_override(self):
        """Test with both offline and online quantization."""
        self._run_test(
            model=DEFAULT_MODEL_NAME_FOR_DYNAMIC_QUANT_ACCURACY_TEST_FP8,
            other_args=["--quantization", "w8a8_fp8"],
            # inference will use sgl kernel w/ online quant override
            # we observed that the accuracy is higher then offline only
            expected_score=0.64,
        )

    def test_mmlu_online_only(self):
        """Test with online quantization only."""
        self._run_test(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            # inference will use sgl kernel w/ online quantization only
            # we observed that the accuracy is higher then offline only
            other_args=["--quantization", "w8a8_fp8"],
            expected_score=0.64,
        )

    def test_mmlu_fp16_baseline(self):
        """Test with unquantized fp16 baseline."""
        self._run_test(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            other_args=[],
            expected_score=0.64,
        )


if __name__ == "__main__":
    unittest.main()
