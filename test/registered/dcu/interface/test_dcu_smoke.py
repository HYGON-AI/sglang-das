"""DCU smoke placeholder.

Purpose:
    Provide one minimal registered test under test/registered/dcu/ so that
    `python3 test/run_suite.py --hw dcu --suite stage-a-test-1-gpu-small-dcu`
    is able to collect and execute at least one file end-to-end.

This file deliberately performs no real DCU work. It only validates that
the sglang package imports successfully. Heavier DCU smoke / accuracy /
perf tests will be added in subsequent PRs.
"""

import os
import unittest

from sglang.test.ci.ci_register import register_dcu_ci
from sglang.test.test_utils import CustomTestCase

register_dcu_ci(est_time=10, suite="stage-a-test-1-gpu-small-dcu")


class TestDCUSmoke(CustomTestCase):
    def test_import_sglang(self):
        import sglang

        self.assertTrue(hasattr(sglang, "__version__"))
        self.assertIsInstance(sglang.__version__, str)
        self.assertGreater(len(sglang.__version__), 0)

    def test_dcu_ci_env_marker(self):
        # SGLANG_IS_IN_CI_DCU is set by scripts/ci/dcu/dcu_ci_exec.sh.
        # When not running inside the DCU CI container (e.g. local dry-run),
        # the marker is absent and we just skip silently.
        if os.environ.get("SGLANG_IS_IN_CI_DCU") != "1":
            self.skipTest("Not running inside DCU CI container; skipping marker check.")
        self.assertEqual(os.environ.get("SGLANG_IS_IN_CI"), "1")


if __name__ == "__main__":
    unittest.main()
# CI auto-trigger verification: 20260610T103733Z
