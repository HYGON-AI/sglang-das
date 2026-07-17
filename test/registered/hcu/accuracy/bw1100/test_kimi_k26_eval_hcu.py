"""HCU Kimi-K2.6 GSM8K completion evaluation on eight BW1100 cards."""

import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.hcu_cookbook_utils import (
    DEFAULT_HCU_GSM8K_DATA_PATH,
    KIMI_K26_8GPU,
)
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, popen_launch_server

register_hcu_ci(
    est_time=4200,
    suite="nightly-hcu-accuracy-text",
    nightly=True,
)

DEFAULT_ACCURACY_THRESHOLD = 0.80


class TestKimiK26EvalHCU(unittest.TestCase):
    def test_kimi_k26_accuracy(self):
        num_questions = int(
            os.environ.get("SGLANG_HCU_KIMI_K26_GSM8K_NUM_QUESTIONS", "1319")
        )
        num_shots = int(
            os.environ.get("SGLANG_HCU_KIMI_K26_GSM8K_NUM_SHOTS", "8")
        )
        parallel = int(
            os.environ.get(
                "SGLANG_HCU_KIMI_K26_GSM8K_PARALLEL", str(num_questions)
            )
        )
        threshold = float(
            os.environ.get(
                "SGLANG_HCU_KIMI_K26_GSM8K_THRESHOLD",
                str(DEFAULT_ACCURACY_THRESHOLD),
            )
        )
        data_path = (
            os.environ.get("SGLANG_HCU_COOKBOOK_GSM8K_DATA_PATH")
            or os.environ.get("SGLANG_HCU_GSM8K_DATA_PATH")
            or DEFAULT_HCU_GSM8K_DATA_PATH
        )
        if not os.path.isfile(data_path):
            raise AssertionError(f"Local GSM8K data path does not exist: {data_path}")
        model_path = KIMI_K26_8GPU.resolve_model_path()

        process = popen_launch_server(
            model_path,
            DEFAULT_URL_FOR_TEST,
            timeout=KIMI_K26_8GPU.timeout,
            other_args=list(KIMI_K26_8GPU.server_args),
            env=KIMI_K26_8GPU.merged_env(),
        )
        try:
            response = requests.get(
                DEFAULT_URL_FOR_TEST.rstrip("/") + "/flush_cache", timeout=60
            )
            response.raise_for_status()

            args = SimpleNamespace(
                num_shots=num_shots,
                data_path=data_path,
                num_questions=num_questions,
                parallel=parallel,
                max_new_tokens=512,
                temperature=0,
                host="http://127.0.0.1",
                port=int(DEFAULT_URL_FOR_TEST.rsplit(":", 1)[-1]),
            )
            metrics = run_eval_few_shot_gsm8k(args)
        finally:
            kill_process_tree(process.pid)

        accuracy = float(metrics["accuracy"])
        invalid = float(metrics["invalid"])
        latency = float(metrics["latency"])
        print(
            "HCU Kimi-K2.6 GSM8K: "
            f"accuracy={accuracy:.3f}, invalid={invalid:.3f}, "
            f"latency={latency:.1f}s, threshold={threshold:.3f}"
        )
        self.assertGreaterEqual(accuracy, threshold)


if __name__ == "__main__":
    unittest.main()
