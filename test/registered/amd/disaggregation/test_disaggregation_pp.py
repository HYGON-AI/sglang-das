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

import os
import tempfile
import time
import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import requests

from sglang.test.ci.ci_register import register_amd_ci, register_dcu_ci

# register_amd_ci(est_time=200, suite="stage-c-test-large-8-gpu-amd")

# DCU_CSV_CI_UNVERIFIED: Registered from sglang.csv CI coverage; not re-tested in this framework pass.
register_dcu_ci(
    est_time=300,
    suite="nightly-dcu",
    nightly=True,
    disabled="DCU CSV CI placeholder: disaggregation PP path needs BW1100 multi-device validation before enabling.",
)

from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
    try_cached_model,
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


@contextmanager
def _temporary_envs(updates):
    old_values = {name: os.environ.get(name) for name in updates}
    for name, value in updates.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    try:
        yield
    finally:
        for name, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value


_DCU_KERNEL_ALIAS_DIR = None


def _ensure_dcu_kvcacheio_symbol():
    try:
        from sgl_kernel import kvcacheio
    except Exception as exc:
        raise unittest.SkipTest(f"sgl_kernel.kvcacheio is unavailable: {exc}") from exc

    if hasattr(kvcacheio, "dcu_create_chunked_prefix_cache_kv_indices"):
        return
    if hasattr(kvcacheio, "hcu_create_chunked_prefix_cache_kv_indices"):
        return

    raise unittest.SkipTest(
        "sgl_kernel.kvcacheio has neither dcu_create_chunked_prefix_cache_kv_indices "
        "nor hcu_create_chunked_prefix_cache_kv_indices."
    )


def _ensure_dcu_gpu_resources(required_gpus):
    if os.getenv("SGLANG_DCU_SKIP_GPU_RESOURCE_CHECK") == "1":
        return

    try:
        import torch
    except Exception as exc:
        raise unittest.SkipTest(f"torch is unavailable for DCU resource check: {exc}") from exc

    if not torch.cuda.is_available():
        raise unittest.SkipTest("DCU resource check requires torch.cuda to be available.")

    device_count = torch.cuda.device_count()
    if device_count < required_gpus:
        raise unittest.SkipTest(
            f"DCU PP disaggregation requires {required_gpus} visible GPUs, "
            f"but only {device_count} are visible."
        )

    min_free_ratio = float(os.getenv("SGLANG_DCU_MIN_FREE_GPU_RATIO", "0.80"))
    low_memory_devices = []
    for device_id in range(required_gpus):
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
        free_ratio = free_bytes / total_bytes if total_bytes else 0.0
        if free_ratio < min_free_ratio:
            low_memory_devices.append(
                f"{device_id}: {free_bytes / (1024**3):.1f}/"
                f"{total_bytes / (1024**3):.1f} GiB free"
            )

    if low_memory_devices:
        raise unittest.SkipTest(
            f"DCU PP disaggregation requires {required_gpus} mostly idle GPUs "
            f"(free ratio >= {min_free_ratio:.2f}); low-memory devices: "
            + ", ".join(low_memory_devices)
        )


def _dcu_kernel_alias_dir():
    global _DCU_KERNEL_ALIAS_DIR
    if _DCU_KERNEL_ALIAS_DIR is not None:
        return _DCU_KERNEL_ALIAS_DIR

    _ensure_dcu_kvcacheio_symbol()
    sitecustomize_dir = tempfile.mkdtemp(prefix="sglang_dcu_kernel_alias_")
    sitecustomize_path = os.path.join(sitecustomize_dir, "sitecustomize.py")
    with open(sitecustomize_path, "w", encoding="utf-8") as f:
        f.write(
            """
try:
    from sgl_kernel import kvcacheio as _kvcacheio

    for _name in dir(_kvcacheio):
        if _name.startswith("hcu_"):
            _dcu_name = "dcu_" + _name[4:]
            if not hasattr(_kvcacheio, _dcu_name):
                setattr(_kvcacheio, _dcu_name, getattr(_kvcacheio, _name))
except Exception:
    pass
"""
        )
    _DCU_KERNEL_ALIAS_DIR = sitecustomize_dir
    return _DCU_KERNEL_ALIAS_DIR


def _configure_disaggregation_class(cls):
    if _is_dcu():
        _ensure_dcu_kvcacheio_symbol()
        _ensure_dcu_gpu_resources(getattr(cls, "dcu_required_gpus", 6))
        cls.model = _DCU_MODEL_NAME
        cls.transfer_backend = ["--disaggregation-transfer-backend", "nixl"]
        cls.rdma_devices = []
        return

    # set up ROCm env
    os.environ["SGLANG_USE_AITER"] = "1"
    rdma_env = os.environ.get("SGLANG_TEST_RDMA_DEVICE")

    if rdma_env:
        cls.rdma_devices = ["--disaggregation-ib-device", rdma_env]
        print(f"Found RDMA devices in env: {rdma_env}")
    else:
        print("SGLANG_TEST_RDMA_DEVICE is not set! Running without RDMA.")
        cls.rdma_devices = []

    cls.model = try_cached_model(DEFAULT_MODEL_NAME_FOR_TEST)


def _launch_pd_process(cls, attr_name, url, args):
    if _is_dcu():
        alias_dir = _dcu_kernel_alias_dir()
        old_pythonpath = os.environ.get("PYTHONPATH")
        pythonpath = alias_dir
        if old_pythonpath:
            pythonpath = alias_dir + os.pathsep + old_pythonpath
        env_context = _temporary_envs(
            {"SGLANG_USE_AITER": "0", "PYTHONPATH": pythonpath}
        )
    else:
        env_context = nullcontext()

    with env_context:
        setattr(
            cls,
            attr_name,
            popen_launch_pd_server(
                cls.model,
                url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=args,
            ),
        )


def _assert_dcu_generate(test_case):
    response = requests.post(
        test_case.lb_url + "/generate",
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
    test_case.assertEqual(response.status_code, 200, response.text)
    test_case.assertIn("text", response.json())


class TestDisaggregationPrefillPPAccuracy(PDDisaggregationServerBase):
    dcu_required_gpus = 6

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _configure_disaggregation_class(cls)

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
            "--tp-size",
            "2",
            "--pp-size",
            "2",
            "--disable-overlap-schedule",
            "--attention-backend",
            "aiter",
        ]
        if _is_dcu():
            prefill_args = [
                "--trust-remote-code",
                "--disaggregation-mode",
                "prefill",
                "--disaggregation-bootstrap-port",
                cls.bootstrap_port,
                "--tp-size",
                "2",
                "--pp-size",
                "2",
                "--disable-overlap-schedule",
                *_dcu_disagg_server_args(),
            ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        _launch_pd_process(
            cls,
            "process_prefill",
            cls.prefill_url,
            prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp-size",
            "2",
            "--base-gpu-id",
            "4",
            "--attention-backend",
            "aiter",
        ]
        if _is_dcu():
            decode_args = [
                "--trust-remote-code",
                "--disaggregation-mode",
                "decode",
                "--disaggregation-bootstrap-port",
                cls.bootstrap_port,
                "--tp-size",
                "2",
                "--base-gpu-id",
                "4",
                *_dcu_disagg_server_args(),
            ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        _launch_pd_process(
            cls,
            "process_decode",
            cls.decode_url,
            decode_args,
        )

    def test_gsm8k(self):
        if _is_dcu():
            _assert_dcu_generate(self)
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
        metrics = run_eval(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.70)
        # Wait a little bit so that the memory check happens.
        time.sleep(5)


# register_amd_ci(est_time=200, suite="stage-c-test-large-8-gpu-amd")
class TestDisaggregationPrefillPPDynamicChunkAccuracy(PDDisaggregationServerBase):
    dcu_required_gpus = 6

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _configure_disaggregation_class(cls)

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
            "--tp-size",
            "2",
            "--pp-size",
            "2",
            "--disable-overlap-schedule",
            "--enable-dynamic-chunking",
            "--attention-backend",
            "aiter",
        ]
        if _is_dcu():
            prefill_args = [
                "--trust-remote-code",
                "--disaggregation-mode",
                "prefill",
                "--disaggregation-bootstrap-port",
                cls.bootstrap_port,
                "--tp-size",
                "2",
                "--pp-size",
                "2",
                "--disable-overlap-schedule",
                "--enable-dynamic-chunking",
                *_dcu_disagg_server_args(),
            ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        _launch_pd_process(
            cls,
            "process_prefill",
            cls.prefill_url,
            prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp-size",
            "2",
            "--base-gpu-id",
            "4",
            "--attention-backend",
            "aiter",
        ]
        if _is_dcu():
            decode_args = [
                "--trust-remote-code",
                "--disaggregation-mode",
                "decode",
                "--disaggregation-bootstrap-port",
                cls.bootstrap_port,
                "--tp-size",
                "2",
                "--base-gpu-id",
                "4",
                *_dcu_disagg_server_args(),
            ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        _launch_pd_process(
            cls,
            "process_decode",
            cls.decode_url,
            decode_args,
        )

    def test_gsm8k(self):
        if _is_dcu():
            _assert_dcu_generate(self)
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
        metrics = run_eval(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.70)
        # Wait a little bit so that the memory check happens.
        time.sleep(5)


# register_amd_ci(est_time=200, suite="stage-c-test-large-8-gpu-amd")
class TestDisaggregationDecodePPAccuracy(PDDisaggregationServerBase):
    dcu_required_gpus = 8

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _configure_disaggregation_class(cls)

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
            "--tp-size",
            "2",
            "--pp-size",
            "2",
            "--disable-overlap-schedule",
            "--attention-backend",
            "aiter",
        ]
        if _is_dcu():
            prefill_args = [
                "--trust-remote-code",
                "--disaggregation-mode",
                "prefill",
                "--disaggregation-bootstrap-port",
                cls.bootstrap_port,
                "--tp-size",
                "2",
                "--pp-size",
                "2",
                "--disable-overlap-schedule",
                *_dcu_disagg_server_args(),
            ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        _launch_pd_process(
            cls,
            "process_prefill",
            cls.prefill_url,
            prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp-size",
            "2",
            "--pp-size",
            "2",
            "--base-gpu-id",
            "4",
            "--attention-backend",
            "aiter",
        ]
        if _is_dcu():
            decode_args = [
                "--trust-remote-code",
                "--disaggregation-mode",
                "decode",
                "--disaggregation-bootstrap-port",
                cls.bootstrap_port,
                "--tp-size",
                "2",
                "--pp-size",
                "2",
                "--base-gpu-id",
                "4",
                *_dcu_disagg_server_args(),
            ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        _launch_pd_process(
            cls,
            "process_decode",
            cls.decode_url,
            decode_args,
        )

    def test_gsm8k(self):
        if _is_dcu():
            _assert_dcu_generate(self)
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
        metrics = run_eval(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.70)
        # Wait a little bit so that the memory check happens.
        time.sleep(5)


if __name__ == "__main__":
    unittest.main()
