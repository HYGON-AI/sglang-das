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

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_hcu_ci
# HCU_CSV_COVERED_UNVERIFIED: Enabled from sglang.csv historical HCU coverage; not re-tested in this framework pass.
register_hcu_ci(
    est_time=300,
    suite="stage-b-test-1-hcu-small",
    disabled="HCU Stage-B deferred: hierarchical cache server fails during BW1100 startup with VMFault, scheduler exit -6/EOFError before MMLU begins.",
)


register_cuda_ci(est_time=228, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=300, suite="stage-b-test-1-gpu-small-amd")

import subprocess
import time
import unittest

from sglang.srt.utils import is_hip, kill_process_tree
from sglang.test.kits.eval_accuracy_kit import MMLUMixin
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

_is_hip = is_hip()


class TestHiCache(CustomTestCase, MMLUMixin):
    mmlu_score_threshold = 0.64
    mmlu_num_examples = 256
    mmlu_num_threads = 32

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-hierarchical-cache",
                "--mem-fraction-static",
                0.7,
                "--hicache-size",
                100 if not _is_hip else 200,
                "--page-size",
                "64",
                "--hicache-storage-backend",
                "file",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        # Graceful stop first so the server unregisters its large pinned host KV
        # pool in userspace; a bare SIGKILL leaves the kernel to unpin it during
        # reclaim, holding GPU memory long enough to fail the next test.
        cls.process.terminate()
        try:
            cls.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            pass
        kill_process_tree(cls.process.pid)
        time.sleep(5)


if __name__ == "__main__":
    unittest.main()
