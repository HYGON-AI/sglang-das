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

"""HCU smoke placeholder.

Purpose:
    Provide one minimal registered test under test/registered/hcu/ so that
    `python3 test/run_suite.py --hw hcu --suite stage-a-test-1-gpu-small-hcu`
    is able to collect and execute at least one file end-to-end.

This file deliberately performs no real HCU work. It only validates that
the sglang package imports successfully. Heavier HCU smoke / accuracy /
perf tests will be added in subsequent PRs.
"""

import os
import unittest

from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.test_utils import CustomTestCase

register_hcu_ci(est_time=10, suite="stage-a-test-1-gpu-small-hcu")


class TestHCUSmoke(CustomTestCase):
    def test_import_sglang(self):
        import sglang

        self.assertTrue(hasattr(sglang, "__version__"))
        self.assertIsInstance(sglang.__version__, str)
        self.assertGreater(len(sglang.__version__), 0)

    def test_hcu_ci_env_marker(self):
        # SGLANG_IS_IN_CI_HCU is set by scripts/ci/hcu/hcu_ci_exec.sh.
        # When not running inside the HCU CI container (e.g. local dry-run),
        # the marker is absent and we just skip silently.
        if os.environ.get("SGLANG_IS_IN_CI_HCU") != "1":
            self.skipTest("Not running inside HCU CI container; skipping marker check.")
        self.assertEqual(os.environ.get("SGLANG_IS_IN_CI"), "1")


if __name__ == "__main__":
    unittest.main()
# CI auto-trigger verification: 20260610T103733Z
