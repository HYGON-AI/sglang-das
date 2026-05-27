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


class TestDCUStageBFlowZero(CustomTestCase):
    def test_dcu_ci_markers_are_exported(self):
        if os.environ.get("SGLANG_IS_IN_CI_DCU") != "1":
            self.skipTest("Not running inside DCU CI container; skipping marker check.")

        self.assertEqual(os.environ.get("SGLANG_IS_IN_CI"), "1")
        self.assertEqual(os.environ.get("SGLANG_IS_IN_CI_DCU"), "1")

    def test_checkout_source_is_preferred(self):
        if os.environ.get("SGLANG_IS_IN_CI_DCU") != "1":
            self.skipTest("Not running inside DCU CI container; skipping PYTHONPATH check.")

        import sglang

        self.assertIn("/sglang-checkout/python/sglang", os.path.realpath(sglang.__file__))


if __name__ == "__main__":
    unittest.main()
