# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Run the HCU workflow's path filters against large PR file lists."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github/workflows/pr-test-hcu.yml"


class TestHcuChangeDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        job = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"][
            "check-changes"
        ]
        cls.scripts = {
            step["name"]: step["run"] for step in job["steps"] if "run" in step
        }

    def run_filter(self, name, paths, **overrides):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "changed-files"
            fixture.write_text("\n".join(paths) + "\n", encoding="utf-8")
            gh = root / "gh"
            gh.write_text('#!/usr/bin/env bash\ncat "$CHANGED_FILES_FIXTURE"\n')
            gh.chmod(0o755)
            output = root / "output"
            output.touch()
            result = subprocess.run(
                ["bash", "-c", self.scripts[name]],
                env={
                    **os.environ,
                    "PATH": f"{root}{os.pathsep}{os.environ['PATH']}",
                    "CHANGED_FILES_FIXTURE": str(fixture),
                    "GITHUB_OUTPUT": str(output),
                    "GITHUB_REPOSITORY": "HYGON-AI/sglang-das",
                    "PR_NUMBER": "560",
                    "EVENT_NAME": "pull_request_target",
                    "FORCE_WHEEL_MODE": "false",
                    **overrides,
                },
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return dict(line.split("=", 1) for line in output.read_text().splitlines())

    @staticmethod
    def large_paths(first):
        # Exceed Linux and macOS pipe capacity with the first match near the start.
        return [first] + [
            f"docs/large-sync/section-{i:05d}/reference.md" for i in range(10000)
        ]

    def test_large_pr_runs_hcu(self):
        outputs = self.run_filter(
            "Detect HCU-relevant changes",
            self.large_paths("python/sglang/srt/server.py"),
        )
        self.assertEqual(outputs, {"hcu": "true", "auto_run_ci": "true"})

    def test_large_pr_requires_python_wheels(self):
        outputs = self.run_filter(
            "Classify HCU wheel sync mode",
            self.large_paths("python/sglang/srt/server.py"),
        )
        self.assertEqual(
            outputs, {"wheel_required": "true", "gateway_wheel_required": "false"}
        )

    def test_large_pr_requires_gateway_wheels(self):
        outputs = self.run_filter(
            "Classify HCU wheel sync mode",
            self.large_paths("sgl-model-gateway/src/lib.rs"),
        )
        self.assertEqual(
            outputs, {"wheel_required": "true", "gateway_wheel_required": "true"}
        )

    def test_docs_only_skips_hcu_and_wheels(self):
        paths = ["docs/guide.md", "README.md"]
        self.assertEqual(
            self.run_filter("Detect HCU-relevant changes", paths),
            {"hcu": "false", "auto_run_ci": "false"},
        )
        self.assertEqual(
            self.run_filter("Classify HCU wheel sync mode", paths),
            {"wheel_required": "false", "gateway_wheel_required": "false"},
        )

    def test_non_pr_run_mode(self):
        self.assertEqual(
            self.run_filter("Detect HCU-relevant changes", [], PR_NUMBER=""),
            {"hcu": "true", "auto_run_ci": "false"},
        )
        self.assertEqual(
            self.run_filter(
                "Classify HCU wheel sync mode", [], EVENT_NAME="workflow_dispatch"
            ),
            {"wheel_required": "false", "gateway_wheel_required": "false"},
        )

    def test_forced_wheel_mode(self):
        self.assertEqual(
            self.run_filter(
                "Classify HCU wheel sync mode", [], FORCE_WHEEL_MODE="true"
            ),
            {"wheel_required": "true", "gateway_wheel_required": "false"},
        )


if __name__ == "__main__":
    unittest.main()
