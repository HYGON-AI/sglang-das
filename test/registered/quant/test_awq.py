import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
from sglang.test.dcu_utils import (
    DCU_TEXT_SERVER_ARGS,
    assert_generate_non_empty,
    get_model_path,
    get_server_args,
)

from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_AWQ_MOE_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    find_available_port,
    is_in_amd_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=160, stage="stage-b", runner_config="1-gpu-large")
register_amd_ci(est_time=200, suite="stage-b-test-1-gpu-large-amd")
register_dcu_ci(est_time=180, suite="stage-b-test-1-gpu-small-dcu")

DEFAULT_DCU_AWQ_MODEL = (
    "/public/opendas/DL_DATA/llm-models/vllm-gptq-models/qwen2.5/"
    "Qwen2.5-3B-Instruct-AWQ"
)


def _is_dcu():
    return os.environ.get("SGLANG_IS_IN_CI_DCU") == "1"


def _dcu_awq_server_args():
    return get_server_args(
        "SGLANG_DCU_AWQ_SERVER_ARGS",
        DCU_TEXT_SERVER_ARGS + ["--disable-cuda-graph"],
    )


def _dcu_awq_env():
    return {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "SGLANG_USE_LIGHTOP": "0",
        "SGLANG_USE_MODELSCOPE": os.environ.get("SGLANG_USE_MODELSCOPE", "1"),
    }


class TestAWQ(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if _is_dcu():
            cls.model = get_model_path("SGLANG_DCU_AWQ_MODEL", DEFAULT_DCU_AWQ_MODEL)
            port = find_available_port(11001)
            cls.base_url = f"http://127.0.0.1:{port}"
            other_args = _dcu_awq_server_args()
            env = _dcu_awq_env()
        else:
            cls.model = DEFAULT_AWQ_MOE_MODEL_NAME_FOR_TEST
            cls.base_url = DEFAULT_URL_FOR_TEST
            other_args = ["--trust-remote-code"]
            env = None

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        if _is_dcu():
            output = assert_generate_non_empty(
                self.base_url,
                text="The capital of France is",
                max_new_tokens=8,
            )
            self.assertGreater(len(output.strip()), 0)
            return

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.64)


@unittest.skipIf(is_in_amd_ci(), "AWQ Marlin is not supported on AMD GPUs")
@unittest.skipIf(_is_dcu(), "DCU AWQ CI uses TestAWQ smoke coverage.")
class TestAWQMarlinBfloat16(CustomTestCase):
    """
    Verify that the model can be loaded with bfloat16 dtype and awq_marlin quantization
    """

    @classmethod
    def setUpClass(cls):
        cls.model = "QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--dtype", "bfloat16", "--quantization", "awq_marlin"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.83)


if __name__ == "__main__":
    unittest.main()
