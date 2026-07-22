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

import json
import os
import shlex
import unittest
import warnings
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.hcu_accuracy_report import write_hcu_accuracy_result
from sglang.test.hcu_server_guard import HcuServerGuard
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    check_evaluation_test_results,
    write_results_to_json,
)

register_hcu_ci(est_time=3600, suite="nightly-hcu-accuracy-text", nightly=True)

DEFAULT_HCU_GSM8K_MODEL = (
    "/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-7B-Instruct"
)
DEFAULT_HCU_GSM8K_DATA_PATH = (
    "/public/opendas/DL_DATA/opencompass_data/gsm8k/test.jsonl"
)
DEFAULT_HCU_SERVER_ARGS = [
    "--attention-backend",
    "fa3",
    "--page-size",
    "64",
    "--log-level",
    "warning",
    "--log-level-http",
    "warning",
    "--trust-remote-code",
]


def _get_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def _get_int_env_with_fallback(name: str, fallback_name: str, default: int) -> int:
    value = os.environ.get(name)
    if value not in (None, ""):
        return int(value)
    return _get_int_env(fallback_name, default)


def _get_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def _get_model_env(name: str, default: str) -> str:
    model = os.environ.get(name, default)
    if model.startswith(("/", ".")) and not os.path.exists(model):
        raise AssertionError(f"{name} points to a missing local model path: {model}")
    return model


def _get_data_path_env(name: str, default: str) -> str:
    value = os.environ.get(name, default)
    if not os.path.exists(value):
        raise AssertionError(f"{name} points to a missing path: {value}")
    return value


def _get_server_args_env(name: str) -> list[str]:
    value = os.environ.get(name)
    if value:
        return shlex.split(value)
    return list(DEFAULT_HCU_SERVER_ARGS)


class TestBW1100GSM8KEvalHCU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _get_model_env("SGLANG_HCU_GSM8K_MODEL", DEFAULT_HCU_GSM8K_MODEL)
        cls.threshold = _get_float_env("SGLANG_HCU_GSM8K_THRESHOLD", 0.88)
        cls.num_examples = _get_int_env_with_fallback(
            "SGLANG_HCU_GSM8K_NUM_EXAMPLES", "SGLANG_HCU_EVAL_NUM_EXAMPLES", 100
        )
        cls.num_threads = _get_int_env("SGLANG_HCU_GSM8K_NUM_THREADS", 128)
        cls.num_shots = _get_int_env("SGLANG_HCU_GSM8K_NUM_SHOTS", 5)
        cls.data_path = _get_data_path_env(
            "SGLANG_HCU_GSM8K_DATA_PATH", DEFAULT_HCU_GSM8K_DATA_PATH
        )
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_gsm8k(self):
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*socket"
        )
        all_results = []

        try:
            with HcuServerGuard(
                self.model,
                self.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=_get_server_args_env("SGLANG_HCU_GSM8K_SERVER_ARGS"),
            ):
                args = SimpleNamespace(
                    base_url=self.base_url,
                    model=self.model,
                    eval_name="gsm8k",
                    api="completion",
                    num_examples=self.num_examples,
                    num_threads=self.num_threads,
                    num_shots=self.num_shots,
                    max_tokens=512,
                    gsm8k_data_path=self.data_path,
                )
                metrics = run_eval(args)

            metrics["score"] = round(metrics["score"], 4)
            write_hcu_accuracy_result(
                model_key="qwen25_7b_instruct",
                model="Qwen2.5-7B-Instruct",
                score=metrics["score"],
                threshold=self.threshold,
                num_examples=self.num_examples,
                invalid_rate=metrics.get("invalid"),
                latency_seconds=metrics.get("latency"),
                source_test=__file__,
            )
            write_results_to_json(self.model, metrics, "w")
            all_results.append((self.model, metrics["score"], 0.0, None))
        except Exception as exc:
            all_results.append((self.model, None, None, str(exc)))
            raise

        try:
            with open("results.json", "r") as f:
                print("\nFinal Results from results.json:")
                print(json.dumps(json.load(f), indent=2))
        except Exception as exc:
            print(f"Error reading results.json: {exc}")

        check_evaluation_test_results(
            all_results,
            self.__class__.__name__,
            model_accuracy_thresholds={self.model: self.threshold},
        )


if __name__ == "__main__":
    unittest.main()
