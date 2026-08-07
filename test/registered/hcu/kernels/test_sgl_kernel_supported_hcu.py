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

import os
import subprocess
import sys
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.hcu_utils import repo_root_from_test_file

register_hcu_ci(est_time=2400, suite="nightly-hcu-core-functional", nightly=True)

SMOKE_KERNEL_TESTS = [
    "tests/speculative/test_eagle_utils.py",
    "tests/test_activation.py",
    "tests/test_apply_token_bitmask_inplace.py",
    "tests/test_merge_state_v2.py",
    # Disabled until its random-input top-k index comparison is stable on HCU.
    # "tests/test_moe_topk_sigmoid.py",
    "tests/test_moe_topk_softmax.py",
    "tests/test_topk.py",
    "tests/test_torch_defaults_reset.py",
]

NIGHTLY_KERNEL_TESTS = SMOKE_KERNEL_TESTS + [
    "tests/test_amd_deterministic_custom_allreduce.py",
    "tests/test_amd_nccl_allreduce_determinism.py",
    "tests/test_kvcacheio.py",
    "tests/test_moe_align.py",
]

KERNEL_TEST_SETS = {
    "smoke": SMOKE_KERNEL_TESTS,
    "nightly": NIGHTLY_KERNEL_TESTS,
    "all-supported": NIGHTLY_KERNEL_TESTS,
}


def _sanitize_hcu_log_text(text: str) -> str:
    return (
        text.replace("AMD", "HCU")
        .replace("amd", "hcu")
        .replace("XGMI", "HSL")
        .replace("xgmi", "hsl")
    )


class TestBW1100SupportedSGLKernelHCU(unittest.TestCase):
    def test_supported_kernel_whitelist(self):
        test_set = os.environ.get("SGLANG_HCU_KERNEL_TEST_SET", "nightly")
        if test_set not in KERNEL_TEST_SETS:
            raise AssertionError(
                "SGLANG_HCU_KERNEL_TEST_SET must be one of "
                f"{sorted(KERNEL_TEST_SETS)}, got {test_set!r}"
            )

        repo_root = repo_root_from_test_file(__file__)
        kernel_root = repo_root / "sgl-kernel"
        if not kernel_root.exists():
            raise unittest.SkipTest(f"sgl-kernel directory is missing: {kernel_root}")

        test_files = [str(kernel_root / name) for name in KERNEL_TEST_SETS[test_set]]
        missing = [name for name in test_files if not Path(name).exists()]
        if missing:
            missing_text = _sanitize_hcu_log_text(str(missing))
            raise AssertionError(f"Missing sgl-kernel tests: {missing_text}")

        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["PYTHONPATH"] = str(repo_root / "python")

        result = subprocess.run(
            [sys.executable, "-m", "pytest", *test_files, "-q"],
            cwd=str(kernel_root),
            env=env,
            text=True,
            capture_output=True,
            timeout=int(os.environ.get("SGLANG_HCU_KERNEL_TEST_TIMEOUT", "900")),
        )

        if result.returncode != 0:
            print("sgl-kernel stdout:")
            print(_sanitize_hcu_log_text(result.stdout))
            print("sgl-kernel stderr:")
            print(_sanitize_hcu_log_text(result.stderr))
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
