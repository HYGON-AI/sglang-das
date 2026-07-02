"""Minimal Stage B DCU CI flow smoke.

This test is intentionally lightweight. The PR-required Stage B signal first
proves CI orchestration, runner placement, container env propagation, and
matrix partitioning. Model/server coverage can grow once the runtime image is
ready for those dependencies.
"""

import os
import unittest

from sglang.test.ci.ci_register import register_dcu_ci
from sglang.test.test_utils import CustomTestCase

register_dcu_ci(est_time=10, suite="stage-b-test-1-gpu-small-dcu")
register_dcu_ci(est_time=10, suite="nightly-dcu-core-functional", nightly=True)


class TestDCUStageBFlowZero(CustomTestCase):
    def test_dcu_ci_markers_are_exported(self):
        if os.environ.get("SGLANG_IS_IN_CI_DCU") != "1":
            self.skipTest("Not running inside DCU CI container; skipping marker check.")

        self.assertEqual(os.environ.get("SGLANG_IS_IN_CI"), "1")
        self.assertEqual(os.environ.get("SGLANG_IS_IN_CI_DCU"), "1")

    def test_checkout_source_is_preferred(self):
        if os.environ.get("SGLANG_IS_IN_CI_DCU") != "1":
            self.skipTest("Not running inside DCU CI container; skipping PYTHONPATH check.")

        if os.environ.get("DCU_CI_USE_INSTALLED_WHEELS") == "1":
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
