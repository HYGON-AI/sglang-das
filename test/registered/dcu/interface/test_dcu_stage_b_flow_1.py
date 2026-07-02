"""Second minimal Stage B DCU CI flow smoke for matrix partitioning."""

import os
import unittest

from sglang.test.ci.ci_register import register_dcu_ci
from sglang.test.test_utils import CustomTestCase

register_dcu_ci(est_time=10, suite="stage-b-test-1-gpu-small-dcu")
register_dcu_ci(est_time=10, suite="nightly-dcu-core-functional", nightly=True)


class TestDCUStageBFlowOne(CustomTestCase):
    def test_dcu_device_nodes_are_mounted(self):
        if os.environ.get("SGLANG_IS_IN_CI_DCU") != "1":
            self.skipTest("Not running inside DCU CI container; skipping device check.")

        self.assertTrue(os.path.exists("/dev/kfd"))
        self.assertTrue(os.path.isdir("/dev/dri"))

    def test_dcu_runner_visible_environment_is_stable(self):
        if os.environ.get("SGLANG_IS_IN_CI_DCU") != "1":
            self.skipTest("Not running inside DCU CI container; skipping environment check.")

        self.assertEqual(os.environ.get("SGLANG_USE_AITER"), "0")
        self.assertEqual(os.environ.get("SGLANG_ROCM_USE_AITER_MOE"), "0")


if __name__ == "__main__":
    unittest.main()
