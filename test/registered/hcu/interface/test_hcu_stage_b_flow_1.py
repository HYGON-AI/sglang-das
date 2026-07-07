"""Second minimal Stage B HCU CI flow smoke for matrix partitioning."""

import os
import unittest

from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.test_utils import CustomTestCase

register_hcu_ci(est_time=10, suite="stage-b-test-1-gpu-small-hcu")


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
