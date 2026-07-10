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

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
from sglang.test.dcu_utils import DCU_TEXT_SERVER_ARGS, get_server_args

from sglang.test.kits.matched_stop_kit import MatchedStopMixin
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
)

register_cuda_ci(est_time=52, stage="stage-b", runner_config="1-gpu-small")
register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")
register_dcu_ci(est_time=120, suite="stage-b-test-1-gpu-small-dcu")


def _is_dcu():
    return os.environ.get("SGLANG_IS_IN_CI_DCU") == "1"


class TestMatchedStop(CustomTestCase, MatchedStopMixin):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        if _is_dcu():
            port = find_available_port(11001)
            cls.base_url = f"http://127.0.0.1:{port}"
            cls.eos_token_ids = [151643, 151645]
            other_args = get_server_args(
                "SGLANG_DCU_MATCHED_STOP_SERVER_ARGS",
                DCU_TEXT_SERVER_ARGS + ["--max-running-requests", "10"],
            )
            env = {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                "SGLANG_USE_MODELSCOPE": os.environ.get("SGLANG_USE_MODELSCOPE", "1"),
                "SGLANG_USE_LIGHTOP": os.environ.get("SGLANG_USE_LIGHTOP", "1"),
            }
        else:
            cls.base_url = DEFAULT_URL_FOR_TEST
            other_args = ["--max-running-requests", "10"]
            env = None

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=300,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
