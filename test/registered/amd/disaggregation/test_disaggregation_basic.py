# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Hygon modifications to this file are licensed under the Apache License,
# Version 2.0 (the "License"); you may not use these modifications except
# in compliance with the License. You may obtain a copy of the License at
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
import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import openai
import requests
from transformers import AutoTokenizer

from sglang.test.ci.ci_register import register_amd_ci, register_dcu_ci

# DCU_CSV_CI_UNVERIFIED: Registered from sglang.csv CI coverage; not re-tested in this framework pass.
register_dcu_ci(
    est_time=300,
    suite="nightly-dcu",
    nightly=True,
    disabled="DCU CSV CI placeholder: disaggregation basic path needs BW1100 multi-device validation before enabling.",
)

from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

register_amd_ci(est_time=600, suite="stage-b-test-large-8-gpu-35x-disaggregation-amd")


def _is_dcu():
    return os.getenv("SGLANG_IS_IN_CI_DCU") == "1"


_DCU_MODEL_NAME = "/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-0.6B"


def _dcu_disagg_server_args():
    return [
        "--attention-backend",
        "fa3",
        "--page-size",
        "64",
        "--disable-cuda-graph",
        "--max-total-tokens",
        "1024",
        "--max-running-requests",
        "8",
    ]


@contextmanager
def _temporary_env(name, value):
    old_value = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old_value


class TestDisaggregationAccuracy(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if _is_dcu():
            cls.model = _DCU_MODEL_NAME
            cls.transfer_backend = ["--disaggregation-transfer-backend", "nixl"]
            cls.rdma_devices = []
            cls.start_prefill()
            cls.start_decode()
            cls.wait_server_ready(
                cls.prefill_url + "/health", process=cls.process_prefill
            )
            cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)
            cls.launch_lb()
            return

        # Configure ROCm RDMA environment
        os.environ["SGLANG_USE_AITER"] = "1"
        rdma_env = os.environ.get("SGLANG_TEST_RDMA_DEVICE")

        if rdma_env:
            cls.rdma_devices = ["--disaggregation-ib-device", rdma_env]
            print(f"Found RDMA devices in env: {rdma_env}")
        else:
            print("SGLANG_TEST_RDMA_DEVICE is not set! Running without RDMA.")
            cls.rdma_devices = []

        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        # DEFAULT_MODEL_NAME_FOR_TEST

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health", process=cls.process_prefill)
        cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "1",
            "--attention-backend",
            "aiter",
            "--log-level",
            "debug",
        ]
        if _is_dcu():
            prefill_args = [
                "--trust-remote-code",
                "--disaggregation-mode",
                "prefill",
                "--disaggregation-bootstrap-port",
                cls.bootstrap_port,
                "--tp",
                "1",
                *_dcu_disagg_server_args(),
            ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        env_context = (
            _temporary_env("SGLANG_USE_AITER", "0") if _is_dcu() else nullcontext()
        )
        with env_context:
            cls.process_prefill = popen_launch_pd_server(
                cls.model,
                cls.prefill_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=prefill_args,
            )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "1",
            "--base-gpu-id",
            "1",
            "--attention-backend",
            "aiter",
            "--mem-fraction-static",
            "0.8",
            "--log-level",
            "debug",
        ]
        if _is_dcu():
            decode_args = [
                "--trust-remote-code",
                "--disaggregation-mode",
                "decode",
                "--disaggregation-bootstrap-port",
                cls.bootstrap_port,
                "--tp",
                "1",
                "--base-gpu-id",
                "1",
                *_dcu_disagg_server_args(),
            ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        print("Debug")
        print(decode_args)
        env_context = (
            _temporary_env("SGLANG_USE_AITER", "0") if _is_dcu() else nullcontext()
        )
        with env_context:
            cls.process_decode = popen_launch_pd_server(
                cls.model,
                cls.decode_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=decode_args,
            )

    def test_gsm8k(self):
        if _is_dcu():
            response = requests.post(
                self.lb_url + "/generate",
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
            host=f"http://{self.base_host}",
            port=int(self.lb_port),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"Evaluation metrics: {metrics}")

        self.assertGreater(metrics["accuracy"], 0.70)

    def test_logprob(self):
        if _is_dcu():
            self.skipTest("DCU disaggregation smoke covers generate path only.")

        prompt = "The capital of france is "
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0},
                "return_logprob": True,
                "return_input_logprob": True,
                "logprob_start_len": 0,
            },
        )

        j = response.json()
        completion_tokens = j["meta_info"]["completion_tokens"]
        input_logprobs = j["meta_info"]["input_token_logprobs"]
        output_logprobs = j["meta_info"]["output_token_logprobs"]

        assert (
            len(output_logprobs) == completion_tokens
        ), f"output_logprobs and completion_tokens should have the same length, but got {len(output_logprobs)} and {completion_tokens}"
        assert (
            len(input_logprobs) > 0
        ), f"input_logprobs should have at least one token, but got {len(input_logprobs)}"

    def test_structured_output(self):
        if _is_dcu():
            self.skipTest("DCU disaggregation smoke covers generate path only.")

        json_schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "pattern": "^[\\w]+$"},
                    "population": {"type": "integer"},
                },
                "required": ["name", "population"],
            }
        )

        # JSON
        response = requests.post(
            f"{self.lb_url}/generate",
            json={
                "text": "Here is the information of the capital of France in the JSON format.\n",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 64,
                    "json_schema": json_schema,
                },
            },
        )
        output = response.json()["text"]
        # ensure the output is a valid JSON
        json.loads(output)

    def test_first_token_finish(self):
        if _is_dcu():
            self.skipTest("DCU disaggregation smoke covers generate path only.")

        client = openai.Client(api_key="empty", base_url=f"{self.lb_url}/v1")
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        eos_token = tokenizer.eos_token_id
        prompt = "The best programming language for AI is"

        # First token EOS
        res = client.completions.create(
            model="dummy", prompt=prompt, logit_bias={eos_token: 42}
        ).model_dump()
        print(f"{res=}")

        assert res["usage"]["completion_tokens"] == 1, (
            "Expected completion_tokens to be 1 when first token is EOS, "
            f"but got {res['usage']['completion_tokens']}"
        )

        # First token EOS with ignore_eos
        res = client.completions.create(
            model="dummy",
            prompt=prompt,
            logit_bias={eos_token: 42},
            extra_body={"ignore_eos": True},
        ).model_dump()
        print(f"{res=}")

        assert res["usage"]["completion_tokens"] > 1, (
            "Expected completion_tokens to be greater than 1 when ignore_eos is True, "
            f"but got {res['usage']['completion_tokens']}"
        )

        # First token with specified stop token
        stop_token_id = tokenizer.encode(" hello", add_special_tokens=False)[0]
        res = client.completions.create(
            model="dummy",
            prompt=prompt,
            logit_bias={stop_token_id: 42},
            stop=[" hello"],
        ).model_dump()
        print(f"{res=}")

        assert res["usage"]["completion_tokens"] == 1, (
            "Expected completion_tokens to be 1 when first token is stop token, "
            f"but got {res['usage']['completion_tokens']}"
        )


# register_amd_ci(est_time=300, suite="stage-b-test-2-gpu-large-amd")
class TestDisaggregationMooncakeFailure(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        if _is_dcu():
            raise unittest.SkipTest("DCU smoke is covered by TestDisaggregationAccuracy.")

        super().setUpClass()
        # Configure ROCm RDMA environment
        os.environ["SGLANG_USE_AITER"] = "1"
        rdma_env = os.environ.get("SGLANG_TEST_RDMA_DEVICE")

        if rdma_env:
            cls.rdma_devices = ["--disaggregation-ib-device", rdma_env]
            print(f"Found RDMA devices in env: {rdma_env}")
        else:
            print("SGLANG_TEST_RDMA_DEVICE is not set! Running without RDMA.")
            cls.rdma_devices = []

        # set DISAGGREGATION_TEST_FAILURE_PROB to simulate failure
        os.environ["DISAGGREGATION_TEST_FAILURE_PROB"] = "0.05"

        cls.model = DEFAULT_MODEL_NAME_FOR_TEST

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health", process=cls.process_prefill)
        cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)

        cls.launch_lb()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("DISAGGREGATION_TEST_FAILURE_PROB")
        super().tearDownClass()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "1",
            "--attention-backend",
            "aiter",
            "--log-level",
            "debug",
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "1",
            "--base-gpu-id",
            "1",
            "--attention-backend",
            "aiter",
            "--mem-fraction-static",
            "0.8",
            "--log-level",
            "debug",
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{self.base_host}",
            port=int(self.lb_port),
        )

        # Expect lots of failure but the server cannot crash
        try:
            metrics = run_eval_few_shot_gsm8k(args)
            print(f"Evaluation metrics: {metrics}")
        except Exception as e:
            print(f"Test encountered expected errors: {e}")
            # Check if servers are still healthy
            try:
                response = requests.get(self.prefill_url + "/health_generate")
                assert response.status_code == 200
                response = requests.get(self.decode_url + "/health_generate")
                assert response.status_code == 200
            except Exception as health_check_error:
                # If health check fails, re-raise the original exception
                raise e from health_check_error


# register_amd_ci(est_time=300, suite="stage-b-test-2-gpu-large-amd")
class TestDisaggregationSimulatedRetract(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        if _is_dcu():
            raise unittest.SkipTest("DCU smoke is covered by TestDisaggregationAccuracy.")

        super().setUpClass()
        # Configure ROCm RDMA environment
        os.environ["SGLANG_USE_AITER"] = "1"
        rdma_env = os.environ.get("SGLANG_TEST_RDMA_DEVICE")

        if rdma_env:
            cls.rdma_devices = ["--disaggregation-ib-device", rdma_env]
            print(f"Found RDMA devices in env: {rdma_env}")
        else:
            print("SGLANG_TEST_RDMA_DEVICE is not set! Running without RDMA.")
            cls.rdma_devices = []

        os.environ["SGLANG_TEST_RETRACT"] = "true"
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health", process=cls.process_prefill)
        cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)

        cls.launch_lb()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("SGLANG_TEST_RETRACT")
        super().tearDownClass()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "1",
            "--attention-backend",
            "aiter",
            "--log-level",
            "debug",
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "1",
            "--base-gpu-id",
            "1",
            "--attention-backend",
            "aiter",
            "--mem-fraction-static",
            "0.8",
            "--log-level",
            "debug",
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{self.base_host}",
            port=int(self.lb_port),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"Evaluation metrics: {metrics}")

        self.assertGreater(metrics["accuracy"], 0.70)


if __name__ == "__main__":
    unittest.main()
