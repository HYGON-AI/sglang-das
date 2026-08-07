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

"""Minimal Stage B HCU CI flow smoke.

This test is intentionally lightweight. The PR-required Stage B signal first
proves CI orchestration, runner placement, container env propagation, and
matrix partitioning. Model/server coverage can grow once the runtime image is
ready for those dependencies.
"""

import os
import unittest

from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.test_utils import CustomTestCase

register_hcu_ci(est_time=10, suite="stage-b-test-1-hcu-small")
register_hcu_ci(est_time=10, suite="nightly-hcu-core-functional", nightly=True)


class TestHCUStageBFlowZero(CustomTestCase):
    def test_hcu_ci_markers_are_exported(self):
        if os.environ.get("SGLANG_IS_IN_CI_HCU") != "1":
            self.skipTest("Not running inside HCU CI container; skipping marker check.")

        self.assertEqual(os.environ.get("SGLANG_IS_IN_CI"), "1")
        self.assertEqual(os.environ.get("SGLANG_IS_IN_CI_HCU"), "1")

    def test_checkout_source_is_preferred(self):
        if os.environ.get("SGLANG_IS_IN_CI_HCU") != "1":
            self.skipTest("Not running inside HCU CI container; skipping PYTHONPATH check.")

        if os.environ.get("HCU_CI_USE_INSTALLED_WHEELS") == "1":
            self.skipTest("Wheel mode: using installed package, not checkout source.")

        import sglang

        pythonpath = os.environ.get("PYTHONPATH", "")
        checkout_python = pythonpath.split(os.pathsep)[0]
        self.assertTrue(checkout_python, "PYTHONPATH must prefer the checkout")
        self.assertIn(
            os.path.join(os.path.realpath(checkout_python), "sglang"),
            os.path.realpath(sglang.__file__),
        )


if __name__ == "__main__":
    unittest.main()
