"""HCU DeepSeek-V3.2 GSM8K completion evaluation on eight BW1100 cards."""

import os
import unittest

from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.hcu_accuracy_report import write_hcu_accuracy_result
from sglang.test.hcu_cookbook_utils import (
    DEEPSEEK_V32_CHANNEL_FP8_8GPU,
    CookbookServer,
    run_gsm8k_completion_benchmark,
)
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST

register_hcu_ci(
    est_time=4200,
    suite="nightly-hcu-accuracy-text",
    nightly=True,
)

DEFAULT_ACCURACY_THRESHOLD = 0.75


class TestDeepSeekV32EvalHCU(unittest.TestCase):
    def test_deepseek_v32_accuracy(self):
        num_questions = int(
            os.environ.get(
                "SGLANG_HCU_DEEPSEEK_V32_GSM8K_NUM_QUESTIONS",
                os.environ.get("GSM8K_NUM_QUESTIONS", "200"),
            )
        )
        num_shots = int(os.environ.get("GSM8K_NUM_SHOTS", "5"))
        parallel = int(os.environ.get("GSM8K_PARALLEL", "64"))
        threshold = float(
            os.environ.get(
                "SGLANG_HCU_DEEPSEEK_V32_GSM8K_THRESHOLD",
                str(DEFAULT_ACCURACY_THRESHOLD),
            )
        )

        with CookbookServer(DEEPSEEK_V32_CHANNEL_FP8_8GPU, DEFAULT_URL_FOR_TEST):
            accuracy, invalid, latency = run_gsm8k_completion_benchmark(
                DEFAULT_URL_FOR_TEST,
                num_questions=num_questions,
                num_shots=num_shots,
                parallel=parallel,
            )

        print(
            "HCU DeepSeek-V3.2 GSM8K: "
            f"accuracy={accuracy:.3f}, invalid={invalid:.3f}, "
            f"latency={latency:.1f}s, threshold={threshold:.3f}"
        )
        write_hcu_accuracy_result(
            model_key="deepseek_v32_channel_fp8",
            model="DeepSeek-V3.2-Channel-FP8",
            score=accuracy,
            threshold=threshold,
            num_examples=num_questions,
            invalid_rate=invalid,
            latency_seconds=latency,
            source_test=__file__,
        )
        self.assertGreaterEqual(accuracy, threshold)


if __name__ == "__main__":
    unittest.main()
