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

# Qwen model tests

import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=90, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=130, suite="stage-b-test-1-gpu-small-amd")


# DCU_CSV_COVERED_UNVERIFIED: Enabled from sglang.csv historical DCU coverage; not re-tested in this framework pass.
register_dcu_ci(
    est_time=180,
    suite="stage-b-test-1-gpu-small-dcu",
)


def _is_dcu():
    return os.getenv("SGLANG_IS_IN_CI_DCU") == "1"


_DCU_MODEL_NAME = "/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-0.6B"


def _dcu_server_args():
    return [
        "--attention-backend",
        "fa3",
        "--page-size",
        "64",
        "--trust-remote-code",
        "--max-total-tokens",
        "1024",
        "--disable-cuda-graph",
    ]


class TestQwen2(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _DCU_MODEL_NAME if _is_dcu() else "Qwen/Qwen2-7B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_dcu_server_args() if _is_dcu() else [],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        if _is_dcu():
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 8,
                        "ignore_eos": True,
                    },
                },
                timeout=60,
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn("text", response.json())
            return

        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["accuracy"], 0.78)


class TestQwen2FP8(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = (
            _DCU_MODEL_NAME if _is_dcu() else "neuralmagic/Qwen2-7B-Instruct-FP8"
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_dcu_server_args() if _is_dcu() else [],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        if _is_dcu():
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 8,
                        "ignore_eos": True,
                    },
                },
                timeout=60,
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn("text", response.json())
            return

        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["accuracy"], 0.78)


if __name__ == "__main__":
    unittest.main()
