import os
import subprocess
import sys
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_dcu_ci
from sglang.test.dcu_utils import repo_root_from_test_file

register_dcu_ci(est_time=900, suite="stage-b-test-1-gpu-small-dcu")

SMOKE_KERNEL_TESTS = [
    "tests/speculative/test_eagle_utils.py",
    "tests/test_activation.py",
    "tests/test_apply_token_bitmask_inplace.py",
    "tests/test_merge_state_v2.py",
    "tests/test_moe_topk_sigmoid.py",
    "tests/test_moe_topk_softmax.py",
    "tests/test_topk.py",
    "tests/test_torch_defaults_reset.py",
]


def _sanitize_dcu_log_text(text: str) -> str:
    return text.replace("AMD", "DCU").replace("amd", "dcu")


class TestBW1100SmokeSGLKernelDCU(unittest.TestCase):
    def test_smoke_kernel_whitelist(self):
        repo_root = repo_root_from_test_file(__file__)
        kernel_root = repo_root / "sgl-kernel"
        if not kernel_root.exists():
            raise unittest.SkipTest(f"sgl-kernel directory is missing: {kernel_root}")

        test_files = [str(kernel_root / name) for name in SMOKE_KERNEL_TESTS]
        missing = [name for name in test_files if not Path(name).exists()]
        if missing:
            raise AssertionError(f"Missing sgl-kernel tests: {missing}")

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
            timeout=int(os.environ.get("SGLANG_DCU_KERNEL_TEST_TIMEOUT", "900")),
        )

        if result.returncode != 0:
            print("sgl-kernel stdout:")
            print(_sanitize_dcu_log_text(result.stdout))
            print("sgl-kernel stderr:")
            print(_sanitize_dcu_log_text(result.stderr))
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
