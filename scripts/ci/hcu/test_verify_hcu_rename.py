#!/usr/bin/env python3
"""Tests for the repository-wide HCU rename safety check.

Author: Codex
"""

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("verify_hcu_rename.py")


def load_checker():
    if not SCRIPT_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location("verify_hcu_rename", str(SCRIPT_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VerifyHCURenameTest(unittest.TestCase):
    def setUp(self):
        self.checker = load_checker()
        self.assertIsNotNone(self.checker, "verify_hcu_rename.py must exist")

    def test_rejects_old_product_path(self):
        self.assertTrue(hasattr(self.checker, "validate_path"))
        self.assertTrue(self.checker.validate_path("test/registered/dcu/test_model.py"))
        self.assertFalse(self.checker.validate_path("test/registered/hcu/test_model.py"))

    def test_rejects_old_runtime_symbol(self):
        violations = self.checker.validate_text_line(
            "python/sglang/srt/example.py", 7, "backend = 'dcu_mla'", {}
        )
        self.assertTrue(violations)

    def test_rejects_malformed_compiler_define(self):
        violations = self.checker.validate_text_line(
            "sgl-kernel/CMakeLists.txt", 3, '"-HCUTLASS_ENABLE_TESTS"', {}
        )
        self.assertTrue(violations)

    def test_allows_real_compiler_defines(self):
        for value in ("-DCUTLASS_TEST_LEVEL=0", "-DCUTE_USE_PACKED_TUPLE=1", "-DCUDA_ENABLED"):
            self.assertFalse(
                self.checker.validate_text_line("CMakeLists.txt", 1, value, {})
            )

    def test_rejects_leaked_migration_placeholder(self):
        violations = self.checker.validate_text_line(
            ".github/workflows/nightly-test-hcu.yml",
            3,
            "image: harbor.example@@HCU_SYNC_HARBOR_HCU_ADMIN@@base/dev:tag",
            {},
        )
        self.assertTrue(violations)

    def test_allows_only_explicit_harbor_entry(self):
        allowlist = {
            ".github/workflows/lint-hcu.yml": [
                "harbor.sourcefind.cn:5443/dcu/admin/base/dev"
            ]
        }
        allowed = self.checker.validate_text_line(
            ".github/workflows/lint-hcu.yml",
            14,
            "image: harbor.sourcefind.cn:5443/dcu/admin/base/dev:tag",
            allowlist,
        )
        rejected = self.checker.validate_text_line(
            ".github/workflows/other.yml",
            14,
            "image: harbor.sourcefind.cn:5443/dcu/admin/base/dev:tag",
            allowlist,
        )
        self.assertFalse(allowed)
        self.assertTrue(rejected)

    def test_rejects_duplicate_hcu_runtime_helper(self):
        source = """
def is_hcu():
    return True

def is_hcu():
    return is_hcu()
"""
        violations = self.checker.validate_python_structure(
            "python/sglang/srt/utils/common.py", source
        )
        self.assertTrue(violations)

    def test_rejects_duplicate_hcu_import(self):
        source = """
from sglang.srt.utils.common import (
    is_hcu,
    is_hcu,
)
"""
        violations = self.checker.validate_python_structure(
            "python/sglang/srt/server_args.py", source
        )
        self.assertTrue(violations)


if __name__ == "__main__":
    unittest.main()
