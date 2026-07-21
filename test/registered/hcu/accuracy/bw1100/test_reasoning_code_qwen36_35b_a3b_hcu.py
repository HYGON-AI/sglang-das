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

"""HCU MATH-500 and HumanEval evaluation for Qwen3.6-35B-A3B."""

import unittest

from sglang.test import hcu_reasoning_code_utils
from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.hcu_cookbook_utils import QWEN36_35B_A3B_2GPU

register_hcu_ci(
    est_time=14400,
    suite="nightly-hcu-accuracy-reasoning-code",
    nightly=True,
)


class TestQwen36ReasoningCodeHCU(hcu_reasoning_code_utils.HcuReasoningCodeTestBase):
    config = QWEN36_35B_A3B_2GPU
    model_key = "qwen36_35b_a3b"
    math500_threshold = 0.91
    humaneval_threshold = 0.73
    human_max_tokens = 4096


if __name__ == "__main__":
    unittest.main()
