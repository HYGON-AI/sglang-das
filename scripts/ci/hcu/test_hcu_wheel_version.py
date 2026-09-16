# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for the shared PR wheel version configuration."""

import os
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

WORKFLOW = Path(__file__).resolve().parents[3] / ".github/workflows/release-pr-hcu.yml"


class TestHCUWheelVersion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.job = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"][
            "compile"
        ]

    def test_target_version_and_build_binding(self):
        expression = self.job["env"]["SGLANG_PACKAGE_VERSION"][3:-2].strip()
        release = "release/20260825_v0.5.18"
        for base, ref, expected in [
            (release, "feature", "0.5.18"),
            ("main", release, "0.5.18"),
            ("", release, "0.5.18"),
            ("", "main", "0.5.18"),
            ("v0.5.15.post1_dev", release, "0.5.15.post1"),
        ]:
            with self.subTest(base=base, ref=ref):
                code = expression.replace(
                    "github.event.pull_request.base.ref", repr(base)
                )
                code = code.replace("github.ref_name", repr(ref))
                code = code.replace("inputs.target_branch", repr(""))
                code = code.replace("&&", " and ").replace("||", " or ")
                self.assertEqual(eval(code, {"__builtins__": {}}), expected)
        build = next(
            s
            for s in self.job["steps"]
            if s.get("name") == "Build sglang python package"
        )
        self.assertEqual(
            build["env"]["SETUPTOOLS_SCM_PRETEND_VERSION"],
            "${{ env.SGLANG_PACKAGE_VERSION }}",
        )

    def test_repaired_wheel_version_guard(self):
        step = next(
            s
            for s in self.job["steps"]
            if s.get("name") == "Apply DAS local version to wheels"
        )
        script = step["run"]
        start = script.index('    version = metadata.get("Version", "")')
        end = script.index("    _, separator, local_version", start)
        guard = textwrap.dedent(script[start:end])
        for name, filename_version, metadata_version, passes in [
            ("sglang", "0.5.18+torch2110", "0.5.18+torch2110", True),
            ("sglang", "0.5.15.post1+torch2110", "0.5.15.post1+torch2110", False),
            ("sglang", "0.5.18+torch2110", "0.5.15.post1+torch2110", False),
            ("sglang-kernel", "0.4.6.post1+torch2110", "0.4.6.post1+torch2110", True),
        ]:
            with self.subTest(
                name=name,
                filename_version=filename_version,
                metadata_version=metadata_version,
            ):
                scope = {
                    "os": os,
                    "metadata": {"Name": name, "Version": metadata_version},
                    "path": SimpleNamespace(
                        name=f"{name.replace('-', '_')}-{filename_version}-cp310-cp310-linux_x86_64.whl"
                    ),
                }
                with patch.dict(os.environ, SGLANG_PACKAGE_VERSION="0.5.18"):
                    if passes:
                        exec(guard, scope)
                    else:
                        with self.assertRaises(RuntimeError):
                            exec(guard, scope)


if __name__ == "__main__":
    unittest.main()
