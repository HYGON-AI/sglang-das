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
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci, register_dcu_ci

# DCU_CSV_CI_UNVERIFIED: Registered from sglang.csv CI coverage; not re-tested in this framework pass.
register_dcu_ci(
    est_time=600,
    suite="nightly-dcu",
    nightly=True,
    disabled="DCU CSV CI placeholder: disaggregation different-TP path needs BW1100 multi-device validation before enabling.",
)

from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
    try_cached_model,
)

register_cuda_ci(est_time=375, stage="stage-c", runner_config="8-gpu-h20")


def _is_dcu():
    return os.getenv("SGLANG_IS_IN_CI_DCU") == "1"


_DCU_MODEL_NAME = "/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-0.6B"
_DCU_TRANSFER_BACKEND_ENV = "SGLANG_DCU_DISAGG_TRANSFER_BACKEND"
_DCU_TRANSFER_BACKEND_MODULES = {
    "nixl": "nixl._api",
    "mooncake": "mooncake",
    "mori": "mori",
}
_DCU_KERNEL_ALIAS_DIR = None


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


def _check_dcu_transfer_backend(backend):
    module_name = _DCU_TRANSFER_BACKEND_MODULES.get(backend)
    if module_name is None:
        raise unittest.SkipTest(
            f"Unsupported DCU disaggregation transfer backend: {backend}."
        )

    try:
        __import__(module_name)
    except Exception as exc:
        raise unittest.SkipTest(
            f"DCU disaggregation transfer backend {backend!r} is unavailable: {exc}"
        ) from exc


def _resolve_dcu_transfer_backend():
    requested_backend = os.getenv(_DCU_TRANSFER_BACKEND_ENV)
    if requested_backend:
        _check_dcu_transfer_backend(requested_backend)
        return requested_backend

    unavailable = []
    for backend in ("nixl", "mooncake", "mori"):
        try:
            _check_dcu_transfer_backend(backend)
            return backend
        except unittest.SkipTest as exc:
            unavailable.append(str(exc))

    raise unittest.SkipTest(
        "No DCU disaggregation transfer backend is installed; checked nixl, "
        "mooncake, and mori. Details: " + "; ".join(unavailable)
    )


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
            f"DCU different-TP disaggregation requires {required_gpus} visible GPUs, "
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
            f"DCU different-TP disaggregation requires {required_gpus} mostly idle GPUs "
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


def _configure_disaggregation_class(cls, default_model):
    if _is_dcu():
        backend = _resolve_dcu_transfer_backend()
        _ensure_dcu_kvcacheio_symbol()
        _ensure_dcu_gpu_resources(required_gpus=6)
        cls.model = _DCU_MODEL_NAME
        cls.transfer_backend = ["--disaggregation-transfer-backend", backend]
        cls.rdma_devices = []
        return

    cls.model = try_cached_model(default_model)


def _dcu_process_env(env=None):
    if not _is_dcu():
        return env

    alias_dir = _dcu_kernel_alias_dir()
    base_env = dict(env or {})
    old_pythonpath = base_env.get("PYTHONPATH", os.environ.get("PYTHONPATH"))
    pythonpath = alias_dir
    if old_pythonpath:
        pythonpath = alias_dir + os.pathsep + old_pythonpath
    base_env["PYTHONPATH"] = pythonpath
    base_env["SGLANG_USE_AITER"] = "0"
    return base_env


def _launch_pd_process(cls, attr_name, url, args, env=None):
    setattr(
        cls,
        attr_name,
        popen_launch_pd_server(
            cls.model,
            url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            env=_dcu_process_env(env),
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


class TestDisaggregationMooncakePrefillLargerTP(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Temporarily disable JIT DeepGEMM
        envs.SGLANG_ENABLE_JIT_DEEPGEMM.set(False)

        _configure_disaggregation_class(cls, DEFAULT_MODEL_NAME_FOR_TEST_MLA)

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
            "4",
            "--enable-metrics",
            "--enable-request-time-stats-logging",
        ]
        if _is_dcu():
            prefill_args += _dcu_disagg_server_args()
        prefill_args += cls.transfer_backend + cls.rdma_devices
        _launch_pd_process(cls, "process_prefill", cls.prefill_url, prefill_args)

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "2",
            "--base-gpu-id",
            "4",
            "--enable-metrics",
            "--enable-request-time-stats-logging",
        ]
        if _is_dcu():
            decode_args += _dcu_disagg_server_args()
        decode_args += cls.transfer_backend + cls.rdma_devices
        _launch_pd_process(cls, "process_decode", cls.decode_url, decode_args)

    def test_gsm8k(self):
        if _is_dcu():
            _assert_dcu_generate(self)
            return

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Evaluation metrics: {metrics}")

        self.assertGreater(metrics["score"], 0.60)


class TestDisaggregationMooncakeDecodeLargerTP(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Temporarily disable JIT DeepGEMM
        envs.SGLANG_ENABLE_JIT_DEEPGEMM.set(False)

        _configure_disaggregation_class(cls, DEFAULT_MODEL_NAME_FOR_TEST_MLA)

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
            "2",
            "--enable-metrics",
            "--enable-request-time-stats-logging",
        ]
        if _is_dcu():
            prefill_args += _dcu_disagg_server_args()
        prefill_args += cls.transfer_backend + cls.rdma_devices
        _launch_pd_process(cls, "process_prefill", cls.prefill_url, prefill_args)

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "4",
            "--base-gpu-id",
            "4",
            "--enable-metrics",
            "--enable-request-time-stats-logging",
        ]
        if _is_dcu():
            decode_args += _dcu_disagg_server_args()
        decode_args += cls.transfer_backend + cls.rdma_devices
        _launch_pd_process(cls, "process_decode", cls.decode_url, decode_args)

    def test_gsm8k(self):
        if _is_dcu():
            _assert_dcu_generate(self)
            return

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Evaluation metrics: {metrics}")

        self.assertGreater(metrics["score"], 0.60)


class TestDisaggregationMooncakeMHAPrefillLargerTP(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Temporarily disable JIT DeepGEMM
        envs.SGLANG_ENABLE_JIT_DEEPGEMM.set(False)

        _configure_disaggregation_class(cls, DEFAULT_MODEL_NAME_FOR_TEST)

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
            "4",
            "--enable-metrics",
            "--enable-request-time-stats-logging",
        ]
        if _is_dcu():
            prefill_args += _dcu_disagg_server_args()
        prefill_args += cls.transfer_backend + cls.rdma_devices
        _launch_pd_process(cls, "process_prefill", cls.prefill_url, prefill_args)

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "2",
            "--base-gpu-id",
            "4",
            "--enable-metrics",
            "--enable-request-time-stats-logging",
        ]
        if _is_dcu():
            decode_args += _dcu_disagg_server_args()
        decode_args += cls.transfer_backend + cls.rdma_devices
        _launch_pd_process(cls, "process_decode", cls.decode_url, decode_args)

    def test_gsm8k(self):
        if _is_dcu():
            _assert_dcu_generate(self)
            return

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Evaluation metrics: {metrics}")

        self.assertGreater(metrics["score"], 0.60)


class TestDisaggregationMooncakeMHADecodeLargerTP(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Temporarily disable JIT DeepGEMM
        envs.SGLANG_ENABLE_JIT_DEEPGEMM.set(False)

        _configure_disaggregation_class(cls, DEFAULT_MODEL_NAME_FOR_TEST)

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
            "2",
            "--enable-metrics",
            "--enable-request-time-stats-logging",
        ]
        if _is_dcu():
            prefill_args += _dcu_disagg_server_args()
        prefill_args += cls.transfer_backend + cls.rdma_devices
        _launch_pd_process(cls, "process_prefill", cls.prefill_url, prefill_args)

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "4",
            "--base-gpu-id",
            "4",
            "--enable-metrics",
            "--enable-request-time-stats-logging",
        ]
        if _is_dcu():
            decode_args += _dcu_disagg_server_args()
        decode_args += cls.transfer_backend + cls.rdma_devices
        _launch_pd_process(cls, "process_decode", cls.decode_url, decode_args)

    def test_gsm8k(self):
        if _is_dcu():
            _assert_dcu_generate(self)
            return

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Evaluation metrics: {metrics}")

        self.assertGreater(metrics["score"], 0.60)


STAGING_ENV = {
    "SGLANG_DISAGG_STAGING_BUFFER": "1",
    "SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB": "64",
    "SGLANG_DISAGG_STAGING_POOL_SIZE_MB": "1024",
}


class TestDisaggregationStagingPrefillLargerTP(PDDisaggregationServerBase):
    """Prefill TP=4 -> Decode TP=2 with staging buffer enabled (MHA model)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        envs.SGLANG_ENABLE_JIT_DEEPGEMM.set(False)

        _configure_disaggregation_class(cls, DEFAULT_MODEL_NAME_FOR_TEST)

        cls.start_prefill()
        cls.start_decode()

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
            "4",
            "--enable-metrics",
            "--enable-request-time-stats-logging",
        ]
        if _is_dcu():
            prefill_args += _dcu_disagg_server_args()
        prefill_args += cls.transfer_backend + cls.rdma_devices
        env = {**os.environ, **STAGING_ENV}
        _launch_pd_process(cls, "process_prefill", cls.prefill_url, prefill_args, env)

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "2",
            "--base-gpu-id",
            "4",
            "--enable-metrics",
            "--enable-request-time-stats-logging",
        ]
        if _is_dcu():
            decode_args += _dcu_disagg_server_args()
        decode_args += cls.transfer_backend + cls.rdma_devices
        env = {**os.environ, **STAGING_ENV}
        _launch_pd_process(cls, "process_decode", cls.decode_url, decode_args, env)

    def test_gsm8k(self):
        if _is_dcu():
            _assert_dcu_generate(self)
            return

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"[Staging PrefillLargerTP] Evaluation metrics: {metrics}")
        self.assertGreater(metrics["score"], 0.60)


class TestDisaggregationStagingDecodeLargerTP(PDDisaggregationServerBase):
    """Prefill TP=2 -> Decode TP=4 with staging buffer enabled (MHA model)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        envs.SGLANG_ENABLE_JIT_DEEPGEMM.set(False)

        _configure_disaggregation_class(cls, DEFAULT_MODEL_NAME_FOR_TEST)

        cls.start_prefill()
        cls.start_decode()

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
            "2",
            "--enable-metrics",
            "--enable-request-time-stats-logging",
        ]
        if _is_dcu():
            prefill_args += _dcu_disagg_server_args()
        prefill_args += cls.transfer_backend + cls.rdma_devices
        env = {**os.environ, **STAGING_ENV}
        _launch_pd_process(cls, "process_prefill", cls.prefill_url, prefill_args, env)

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "4",
            "--base-gpu-id",
            "4",
            "--enable-metrics",
            "--enable-request-time-stats-logging",
        ]
        if _is_dcu():
            decode_args += _dcu_disagg_server_args()
        decode_args += cls.transfer_backend + cls.rdma_devices
        env = {**os.environ, **STAGING_ENV}
        _launch_pd_process(cls, "process_decode", cls.decode_url, decode_args, env)

    def test_gsm8k(self):
        if _is_dcu():
            _assert_dcu_generate(self)
            return

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"[Staging DecodeLargerTP] Evaluation metrics: {metrics}")
        self.assertGreater(metrics["score"], 0.60)


if __name__ == "__main__":
    unittest.main()
