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

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_hcu_ci

register_hcu_ci(est_time=800, suite="stage-b-test-1-gpu-large-hcu", disabled='HCU Full Enabled run 26941698027 failed; keep disabled until BW1100 failure is fixed or revalidated.')

from sglang.test.kits.eval_accuracy_kit import MGSMEnMixin
from sglang.test.test_utils import (
    DEFAULT_MLA_FP8_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    popen_launch_server,
)

# MLA FP8 KV cache test with MGSM evaluation
register_cuda_ci(est_time=104, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=800, suite="stage-b-test-1-gpu-small-amd")


class TestMLA(CustomTestCase, MGSMEnMixin):
    mgsm_en_score_threshold = 0.8

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MLA_FP8_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--kv-cache-dtype",
            "fp8_e5m2",
            # Pin MoE expert dispatch and kernel reduction order so MGSM
            # scores don't drift across runs. The eval already uses greedy
            # decoding, but FP8 dequant + non-deterministic MoE top-k
            # tie-breaks produce ~1–3 point swings without this flag and
            # straddle the 0.8 threshold. With deterministic inference,
            # the score becomes a fixed function of (model, weights, CUDA
            # stack), so threshold-edge flakes stop being random noise.
        ]
        if not is_in_amd_ci():
            # On AMD, the default attention backend (aiter) is not in the deterministic-inference allowlist, so the server fails to start, disable it.
            other_args.append("--enable-deterministic-inference")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
