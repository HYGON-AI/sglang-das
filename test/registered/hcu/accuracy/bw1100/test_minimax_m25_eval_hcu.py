"""HCU MiniMax-M2.5 GSM8K completion evaluation on eight BW1100 cards."""

import os
import unittest

from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.hcu_cookbook_utils import (
    CookbookServer,
    MINIMAX_M25_FP8_8GPU,
    run_gsm8k_completion_benchmark,
)
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST

register_hcu_ci(
    est_time=4200,
    suite="nightly-hcu-accuracy-text",
    nightly=True,
)

DEFAULT_ACCURACY_THRESHOLD = 0.93


class TestMiniMaxM25EvalHCU(unittest.TestCase):
    def test_minimax_m25_accuracy(self):
        num_questions = int(
            os.environ.get(
                "SGLANG_HCU_MINIMAX_M25_GSM8K_NUM_QUESTIONS",
                os.environ.get("GSM8K_NUM_QUESTIONS", "200"),
            )
        )
        num_shots = int(os.environ.get("GSM8K_NUM_SHOTS", "5"))
        parallel = int(os.environ.get("GSM8K_PARALLEL", "64"))
        threshold = float(
            os.environ.get(
                "SGLANG_HCU_MINIMAX_M25_GSM8K_THRESHOLD",
                str(DEFAULT_ACCURACY_THRESHOLD),
            )
        )

        with CookbookServer(MINIMAX_M25_FP8_8GPU, DEFAULT_URL_FOR_TEST):
            accuracy, invalid, latency = run_gsm8k_completion_benchmark(
                DEFAULT_URL_FOR_TEST,
                num_questions=num_questions,
                num_shots=num_shots,
                parallel=parallel,
            )

        print(
            "HCU MiniMax-M2.5 GSM8K: "
            f"accuracy={accuracy:.3f}, invalid={invalid:.3f}, "
            f"latency={latency:.1f}s, threshold={threshold:.3f}"
        )
        self.assertGreaterEqual(accuracy, threshold)


if __name__ == "__main__":
    unittest.main()
