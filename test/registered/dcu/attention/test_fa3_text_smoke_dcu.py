import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_dcu_ci
from sglang.test.dcu_utils import (
    DCU_TEXT_SERVER_ARGS,
    assert_generate_non_empty,
    get_model_path,
    get_server_args,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    find_available_port,
    popen_launch_server,
)

register_dcu_ci(est_time=600, suite="stage-b-test-1-gpu-small-dcu")

DEFAULT_DCU_FA3_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class TestFA3TextSmokeDCU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = get_model_path("SGLANG_DCU_FA3_MODEL", DEFAULT_DCU_FA3_MODEL)
        port = find_available_port(11001)
        cls.base_url = f"http://127.0.0.1:{port}"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=get_server_args("SGLANG_DCU_FA3_SERVER_ARGS", DCU_TEXT_SERVER_ARGS),
            env={
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                "SGLANG_USE_MODELSCOPE": os.environ.get("SGLANG_USE_MODELSCOPE", "1"),
                "SGLANG_USE_LIGHTOP": os.environ.get("SGLANG_USE_LIGHTOP", "1"),
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_generate_non_empty(self):
        output = assert_generate_non_empty(
            self.base_url,
            text="The capital of France is",
            max_new_tokens=16,
        )
        self.assertGreater(len(output.strip()), 0)


if __name__ == "__main__":
    unittest.main()
