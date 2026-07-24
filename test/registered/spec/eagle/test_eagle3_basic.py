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

import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_hcu_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.eagle_fixture import EagleServerBase
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_TARGET_MODEL_EAGLE3,
)

register_cuda_ci(est_time=50, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=50, suite="stage-b-test-1-gpu-small")

register_hcu_ci(
    est_time=120,
    suite="stage-b-test-1-gpu-small-hcu",
    disabled="HCU RL/speculative path needs local model mapping and quick validation before enabling.",
)

_is_hip = is_hip()


class TestEagle3Basic(EagleServerBase):
    target_model = DEFAULT_TARGET_MODEL_EAGLE3
    draft_model = DEFAULT_DRAFT_MODEL_EAGLE3

    spec_algo = "EAGLE3"
    spec_steps = 2
    spec_topk = 1
    spec_tokens = 3
    extra_args = (
        [
            "--dtype=float16",
            "--chunked-prefill-size",
            1024,
            "--attention-backend",
            "aiter",
        ]
        if _is_hip
        else ["--dtype=float16", "--chunked-prefill-size", 1024]
    )

    def test_mmlu(self):
        """Override to add EAGLE-specific assertions"""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.target_model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.72)

        server_info = requests.get(self.base_url + "/server_info").json()
        avg_spec_accept_length = server_info["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")
        if _is_hip:
            self.assertGreater(avg_spec_accept_length, 2.24)
        else:
            self.assertGreater(avg_spec_accept_length, 2.26)


if __name__ == "__main__":

    unittest.main()
