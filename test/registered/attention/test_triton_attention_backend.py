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

"""
Usage:
python3 -m unittest test_triton_attention_backend.TestTritonAttnBackend.test_mmlu
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_hcu_ci

register_hcu_ci(est_time=1400, suite="stage-b-test-1-gpu-large-hcu", disabled='HCU Full Enabled run 26941698027 failed; keep disabled until BW1100 failure is fixed or revalidated.')

from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    run_bench_offline_throughput,
)

# Triton attention backend integration test with latency benchmark and MMLU eval
register_cuda_ci(est_time=177, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=1400, suite="stage-b-test-1-gpu-small-amd")


class TestTritonAttnBackend(CustomTestCase):
    def test_latency(self):
        output_throughput = run_bench_offline_throughput(
            DEFAULT_MODEL_NAME_FOR_TEST,
            [
                "--attention-backend",
                "triton",
                "--enable-torch-compile",
                "--cuda-graph-max-bs-decode",
                4,
            ],
        )

        print(f"{output_throughput=}")

        if is_in_ci():
            self.assertGreater(output_throughput, 153)

    def test_mmlu(self):
        model = DEFAULT_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "triton"],
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
            )

            metrics = run_eval(args)
            self.assertGreaterEqual(metrics["score"], 0.65)
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
