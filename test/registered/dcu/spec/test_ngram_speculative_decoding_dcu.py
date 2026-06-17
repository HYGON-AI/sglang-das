# DCU-specific copy of test/registered/spec/test_ngram_speculative_decoding.py.
# Changes:
# - use a local BW1100 model path override
# - keep only the FA3 paged-attention path supported on DCU
# - set speculative top-k to 1 because page-size 64 with top-k > 1 is unsupported

import unittest

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_dcu_ci
from sglang.test.dcu_utils import get_model_path
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    find_available_port,
    popen_launch_server,
)

register_dcu_ci(
    est_time=600,
    suite="stage-b-test-1-gpu-small-dcu",
    disabled=(
        "DCU disabled retest: page-size/top-k parameters can be adapted, "
        "but runtime still requires missing sgl_kernel.reconstruct_indices_from_tree_mask."
    ),
)

DEFAULT_DCU_NGRAM_MODEL = "/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-7B-Instruct"


class TestNgramSpeculativeDecodingDCU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.set(False)
        envs.SGLANG_ENABLE_JIT_DEEPGEMM.set(False)

        cls.model = get_model_path("SGLANG_DCU_NGRAM_MODEL", DEFAULT_DCU_NGRAM_MODEL)
        port = find_available_port(11001)
        cls.base_url = f"http://127.0.0.1:{port}"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "fa3",
                "--page-size",
                "64",
                "--cuda-graph-max-bs",
                "8",
                "--speculative-algorithm",
                "NGRAM",
                "--speculative-num-draft-tokens",
                "16",
                "--speculative-eagle-topk",
                "1",
                "--speculative-ngram-max-bfs-breadth",
                "1",
                "--mem-fraction-static",
                "0.75",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.clear()
        envs.SGLANG_ENABLE_JIT_DEEPGEMM.clear()

    def test_generate(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        self.assertTrue(payload.get("text", "").strip())


if __name__ == "__main__":
    unittest.main()
