# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Run the HCU server lifecycle and failure-path unit tests in nightly CI."""

import os
import subprocess
import sys
import unittest

from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.hcu_utils import repo_root_from_test_file

register_hcu_ci(
    est_time=60,
    suite="nightly-hcu-core-functional",
    nightly=True,
)
register_hcu_ci(est_time=60, suite="stage-b-test-1-hcu-small")


class TestHCUServerGuardFailurePaths(unittest.TestCase):
    def test_server_guard_unit_suite(self):
        repo_root = repo_root_from_test_file(__file__)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root / "python")
        result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/ci/hcu/test_hcu_server_guard.py"),
            ],
            cwd=str(repo_root),
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
