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

"""Second minimal Stage B HCU CI flow smoke for matrix partitioning."""

import os
import unittest

from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.test_utils import CustomTestCase

register_hcu_ci(est_time=10, suite="stage-b-test-1-hcu-small")
register_hcu_ci(est_time=10, suite="nightly-hcu-core-functional", nightly=True)


class TestHCUStageBFlowOne(CustomTestCase):
    def test_hcu_device_nodes_are_mounted(self):
        if os.environ.get("SGLANG_IS_IN_CI_HCU") != "1":
            self.skipTest("Not running inside HCU CI container; skipping device check.")

        self.assertTrue(os.path.exists("/dev/kfd"))
        self.assertTrue(os.path.isdir("/dev/dri"))

    def test_hcu_runner_visible_environment_is_stable(self):
        if os.environ.get("SGLANG_IS_IN_CI_HCU") != "1":
            self.skipTest("Not running inside HCU CI container; skipping environment check.")

        self.assertEqual(os.environ.get("SGLANG_USE_AITER"), "0")
        self.assertEqual(os.environ.get("SGLANG_ROCM_USE_AITER_MOE"), "0")


if __name__ == "__main__":
    unittest.main()
