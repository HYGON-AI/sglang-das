"""HCU Qwen3.5 397B Channel FP8 GSM8K evaluation on four BW1100 cards."""

import os
import unittest

from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.hcu_accuracy_report import write_hcu_accuracy_result
from sglang.test.hcu_cookbook_utils import (
    QWEN35_397B_A17B_CHANNEL_FP8_4GPU,
    run_cookbook_accuracy_eval,
)
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST

register_hcu_ci(
    est_time=7200,
    suite="nightly-hcu-accuracy-text",
    nightly=True,
)

# AMD expects 0.9704 with rtol=0.05; 0.92 is the equivalent lower bound.
DEFAULT_ACCURACY_THRESHOLD = 0.92


class TestQwen35EvalHCU(unittest.TestCase):
    def test_qwen35_accuracy(self):
        num_questions = int(
            os.environ.get(
                "SGLANG_HCU_QWEN35_GSM8K_NUM_QUESTIONS",
                os.environ.get("GSM8K_NUM_QUESTIONS", "200"),
            )
        )
        num_threads = int(os.environ.get("GSM8K_PARALLEL", "64"))
        num_shots = int(os.environ.get("GSM8K_NUM_SHOTS", "5"))
        max_tokens = int(os.environ.get("SGLANG_HCU_QWEN35_MAX_TOKENS", "8192"))
        threshold = float(
            os.environ.get(
                "SGLANG_HCU_QWEN35_GSM8K_THRESHOLD",
                str(DEFAULT_ACCURACY_THRESHOLD),
            )
        )

        metrics = run_cookbook_accuracy_eval(
            QWEN35_397B_A17B_CHANNEL_FP8_4GPU,
            DEFAULT_URL_FOR_TEST,
            "gsm8k",
            num_examples=num_questions,
            num_threads=num_threads,
            num_shots=num_shots,
            max_tokens=max_tokens,
        )
        accuracy = float(metrics["score"])

        print(
            "HCU Qwen3.5-397B-A17B GSM8K: "
            f"accuracy={accuracy:.3f}, threshold={threshold:.3f}"
        )
        write_hcu_accuracy_result(
            model_key="qwen35_397b_channel_fp8",
            model="Qwen3.5-397B-A17B-Channel-FP8",
            score=accuracy,
            threshold=threshold,
            num_examples=num_questions,
            invalid_rate=metrics.get("invalid"),
            latency_seconds=metrics.get("latency"),
            source_test=__file__,
        )
        self.assertGreaterEqual(accuracy, threshold)


if __name__ == "__main__":
    unittest.main()
