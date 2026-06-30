import os
import time
import unittest

from sglang.srt.utils import is_blackwell, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci, register_dcu_ci
from sglang.test.dcu_utils import (
    DCU_TEXT_SERVER_ARGS,
    assert_generate_non_empty,
    get_model_path,
    get_server_args,
)
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
)

register_cuda_ci(est_time=132, suite="stage-b-test-2-gpu-large")


# DCU_CSV_CI_UNVERIFIED: Registered from sglang.csv CI coverage; not re-tested in this framework pass.
register_dcu_ci(
    est_time=120,
    suite="stage-b-test-1-gpu-small-dcu",
    nightly=False,
)

DCU_NEMOTRON_MODEL = "/public/opendas/DL_DATA/llm-models/vllm-optest-models/nvidia/Nemotron-H-8B-Base-8K"
DCU_NEMOTRON_TINY_MODEL = (
    "/public/opendas/DL_DATA/llm-models/vllm-optest-models/tiiuae/falcon-mamba-tiny-dev"
)


def _is_dcu():
    return os.environ.get("SGLANG_IS_IN_CI_DCU") == "1"


class DCUNemotronServerBase(CustomTestCase):
    __test__ = False
    model = None
    base_url = None
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    other_args = []

    @classmethod
    def setUpClass(cls):
        if cls.model is None:
            raise unittest.SkipTest("Base DCU Nemotron server fixture is not a concrete test class.")

        if not _is_dcu():
            super().setUpClass()
            return

        cls.model = get_model_path("SGLANG_DCU_NEMOTRON_MODEL", cls.model)
        port = find_available_port(11001)
        cls.base_url = f"http://127.0.0.1:{port}"
        server_args = get_server_args(
            "SGLANG_DCU_NEMOTRON_SERVER_ARGS",
            DCU_TEXT_SERVER_ARGS
            + [
                "--max-mamba-cache-size",
                "64",
                "--disable-cuda-graph",
                "--disable-radix-cache",
                "--max-total-tokens",
                "2048",
                "--max-running-requests",
                "4",
            ],
        )
        env = {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            "SGLANG_USE_MODELSCOPE": os.environ.get("SGLANG_USE_MODELSCOPE", "1"),
            "SGLANG_USE_LIGHTOP": os.environ.get("SGLANG_USE_LIGHTOP", "1"),
            "SGLANG_USE_CAUSAL_CONV1D": "1",
        }
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=cls.timeout,
            other_args=server_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if cls.model is None:
            return

        if not _is_dcu():
            super().tearDownClass()
            return

        if hasattr(cls, "process"):
            try:
                kill_process_tree(cls.process.pid, wait_timeout=60)
            except RuntimeError as exc:
                print(f"Warning: DCU Nemotron server cleanup did not fully reap: {exc}")
        time.sleep(2)

    def test_gsm8k(self):
        if not _is_dcu():
            return super().test_gsm8k()

        text = assert_generate_non_empty(
            self.base_url,
            text="The capital of France is",
            max_new_tokens=8,
        )
        self.assertTrue(text.strip())


class TestNvidiaNemotronNanoV2BF16(DCUNemotronServerBase, GSM8KMixin, DefaultServerBase):
    model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    gsm8k_accuracy_thres = 0.87
    other_args = ["--max-mamba-cache-size", "256"]
    if _is_dcu():
        model = DCU_NEMOTRON_MODEL


class TestNvidiaNemotronNanoV2BF16PP(DCUNemotronServerBase, GSM8KMixin, DefaultServerBase):
    model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    gsm8k_accuracy_thres = 0.87
    other_args = ["--max-mamba-cache-size", "256", "--pp-size", "2"]
    if _is_dcu():
        model = DCU_NEMOTRON_TINY_MODEL
        __unittest_skip__ = True
        __unittest_skip_why__ = "DCU single-card CI covers Nemotron/Mamba causal-conv smoke in BF16 class; PP path requires multi-card."


class TestNvidiaNemotronNanoV2FP8(DCUNemotronServerBase, GSM8KMixin, DefaultServerBase):
    gsm8k_accuracy_thres = 0.87
    model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8"
    other_args = ["--max-mamba-cache-size", "256"]
    if _is_dcu():
        model = DCU_NEMOTRON_TINY_MODEL
        __unittest_skip__ = True
        __unittest_skip_why__ = "DCU path validates causal-conv with BF16/local model; FP8 Nemotron asset is not available locally."


@unittest.skipIf(not is_blackwell(), "NVFP4 only supported on blackwell")
class TestNvidiaNemotronNanoV2NVFP4(GSM8KMixin, DefaultServerBase):
    gsm8k_accuracy_thres = 0.855
    model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4"
    other_args = ["--max-mamba-cache-size", "256"]


@unittest.skip(
    "STANDALONE speculative decoding does not yet support target and draft models "
    "with different hidden sizes (Nemotron-9B: 4480, Llama-3.2-1B: 2048)"
)
class TestNvidiaNemotronNanoV2SpeculativeDecoding(GSM8KMixin, DefaultServerBase):
    gsm8k_accuracy_thres = 0.87
    model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    other_args = [
        "--speculative-algorithm",
        "STANDALONE",
        "--speculative-num-steps",
        "2",
        "--speculative-eagle-topk",
        "3",
        "--speculative-num-draft-tokens",
        "5",
        "--speculative-draft-model-path",
        "meta-llama/Llama-3.2-1B",
        "--speculative-draft-load-format",
        "dummy",
        "--max-running-requests",
        "8",
        "--max-total-tokens",
        "2048",
        "--json-model-override-args",
        '{"vocab_size": 131072}',
    ]


@unittest.skip(
    "STANDALONE speculative decoding does not yet support target and draft models "
    "with different hidden sizes (Nemotron-9B: 4480, Llama-3.2-1B: 2048)"
)
class TestNvidiaNemotronNanoV2SpeculativeDecodingBF16Cache(
    GSM8KMixin, DefaultServerBase
):
    gsm8k_accuracy_thres = 0.87
    model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    other_args = [
        "--speculative-algorithm",
        "STANDALONE",
        "--speculative-num-steps",
        "2",
        "--speculative-eagle-topk",
        "3",
        "--speculative-num-draft-tokens",
        "5",
        "--speculative-draft-model-path",
        "meta-llama/Llama-3.2-1B",
        "--speculative-draft-load-format",
        "dummy",
        "--max-running-requests",
        "8",
        "--max-total-tokens",
        "2048",
        "--json-model-override-args",
        '{"vocab_size": 131072}',
        "--mamba-ssm-dtype",
        "bfloat16",
    ]


if __name__ == "__main__":
    unittest.main()
