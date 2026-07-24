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
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci, register_hcu_ci

# HCU_CSV_CI_UNVERIFIED: Registered from sglang.csv CI coverage; not re-tested in this framework pass.
register_hcu_ci(
    est_time=660,
    suite="nightly-hcu",
    nightly=True,
    disabled="HCU CSV CI placeholder: Mooncake EP path needs BW1100 multi-device validation before enabling.",
)

from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import get_rdma_devices_args
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=82, stage="base-c", runner_config="deepep-4-gpu-h100")

ib_devices = get_rdma_devices_args()


class TestTP(CustomTestCase):
    extra_args = []

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--elastic-ep-backend",
                "mooncake",
                "--mooncake-ib-device",
                ib_devices,
                "--moe-a2a-backend",
                "mooncake",
                "--deepep-mode",
                "low_latency",
                "--moe-dense-tp-size",
                "1",
                "--enable-dp-lm-head",
                "--enable-two-batch-overlap",
                "--disable-custom-all-reduce",
                "--enable-eplb",
                "--ep-num-redundant-experts",
                "72",
                "--chunked-prefill-size",
                "512",
                "--cuda-graph-max-bs-decode",
                "128",
                "--max-running-requests",
                "512",
                "--mem-fraction-static",
                "0.5",
                *cls.extra_args,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
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
        print(metrics)

        self.assertGreater(metrics["score"], 0.60)


@unittest.skipIf(is_in_ci(), "Skip since mooncake-ep fault-tolerant test is flaky.")
class TestPureDP(TestTP):
    extra_args = [
        "--enable-dp-attention",
        "--dp",
        "4",
    ]

    pkill_process_1 = "sglang::scheduler_DP1_TP1_EP1"
    pkill_process_2 = "sglang::scheduler_DP3_TP3_EP3"

    def test_gsm8k_fault_1(self):
        """
        Kill one rank and the system should remain operational.
        """
        os.system(f"pkill -f {self.pkill_process_1}")
        super().test_gsm8k()

    @unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
    def test_gsm8k_fault_2(self):
        """
        Kill another rank and the system should remain operational.
        """
        os.system(f"pkill -f {self.pkill_process_2}")
        super().test_gsm8k()


@unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
class TestHybridDPTP(TestPureDP):
    extra_args = [
        "--enable-dp-attention",
        "--dp",
        "2",
    ]

    pkill_process_1 = "sglang::scheduler_DP1_TP2_EP2"
    pkill_process_2 = "sglang::scheduler_DP1_TP3_EP3"


if __name__ == "__main__":
    unittest.main()
