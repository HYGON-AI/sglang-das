# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
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

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cuda_ci,
    register_hcu_ci,
)
from sglang.test.kits.eval_accuracy_kit import MMLUMixin
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=60, stage="stage-b", runner_config="1-gpu-small")
register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")
register_hcu_ci(
    est_time=60,
    suite="stage-b-test-1-hcu-small",
    disabled="HCU Full Enabled run 26941698027 failed; keep disabled until BW1100 failure is fixed or revalidated.",
)


class TestPageSize(CustomTestCase, MMLUMixin):
    mmlu_score_threshold = 0.65
    mmlu_num_examples = 64
    mmlu_num_threads = 32

    @classmethod
    def setUpClass(cls):
        os.environ["SGLANG_DEBUG_MEMORY_POOL"] = "1"
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--page-size", 4, "--chunked-prefill-size", 128],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
