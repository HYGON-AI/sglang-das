# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import hcu_ci_publish_reasoning_code as publish


def _write_source(path: Path, content: str = "{}\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class ReasoningCodePublisherTest(unittest.TestCase):
    def test_publishes_complete_model_atomically_with_permissions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "local" / "qwen3_32b"
            _write_source(model_dir / "manifest.json")
            _write_source(model_dir / "math500_results.jsonl")
            _write_source(model_dir / "math500_summary.json")
            _write_source(model_dir / "humaneval_samples.jsonl")
            _write_source(model_dir / "humaneval_generation_summary.json")
            _write_source(
                model_dir / "humaneval_judge" / "humaneval_summary.json"
            )
            run_log = root / "suite.log"
            judge_log = root / "judge.log"
            _write_source(run_log, "suite\n")
            _write_source(judge_log, "judge\n")

            final_dir = publish.publish_model(
                local_model_dir=model_dir,
                log_files=[run_log, judge_log],
                shared_root=root / "shared",
                run_id=123,
                run_attempt=2,
                model_key="qwen3_32b",
                outcome="success",
                target_ref="v0.5.15.post1_dev",
                commit_sha="a" * 40,
                image_ref="example/sglang:latest",
                image_id="sha256:" + "b" * 64,
                runner_name="runner-1",
            )

            self.assertEqual(
                final_dir,
                root / "shared" / "run-123" / "attempt-2" / "qwen3_32b",
            )
            self.assertTrue((final_dir / "manifest.json").is_file())
            self.assertTrue((final_dir / "logs" / "suite.log").is_file())
            status = json.loads((final_dir / "status.json").read_text())
            self.assertEqual(status["outcome"], "success")
            self.assertEqual(status["image_id"], "sha256:" + "b" * 64)
            self.assertFalse(
                any(path.name.startswith(".qwen3_32b.tmp-") for path in final_dir.parent.iterdir())
            )
            for path in final_dir.rglob("*"):
                expected_mode = 0o2775 if path.is_dir() else 0o664
                self.assertEqual(path.stat().st_mode & 0o7777, expected_mode)
                self.assertEqual(path.stat().st_gid, publish.SHARED_GID)

    def test_failure_without_results_still_publishes_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            final_dir = publish.publish_model(
                local_model_dir=root / "missing",
                log_files=[],
                shared_root=root / "shared",
                run_id=1,
                run_attempt=1,
                model_key="qwen3_30b_a3b",
                outcome="failure",
                target_ref="ref",
                commit_sha="c" * 40,
                image_ref="image",
                image_id="id",
                runner_name="runner",
            )
            status = json.loads((final_dir / "status.json").read_text())
            self.assertEqual(status["outcome"], "failure")
            self.assertEqual(status["files"], [])

    def test_existing_model_result_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            kwargs = dict(
                local_model_dir=root / "missing",
                log_files=[],
                shared_root=root / "shared",
                run_id=1,
                run_attempt=1,
                model_key="qwen36_35b_a3b",
                outcome="failure",
                target_ref="ref",
                commit_sha="d" * 40,
                image_ref="image",
                image_id="id",
                runner_name="runner",
            )
            publish.publish_model(**kwargs)
            with self.assertRaises(FileExistsError):
                publish.publish_model(**kwargs)

    def test_existing_shared_root_permissions_are_not_changed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "shared"
            root.mkdir(mode=0o755)
            before = root.stat().st_mode & 0o7777
            publish.publish_model(
                local_model_dir=root / "missing",
                log_files=[],
                shared_root=root,
                run_id=2,
                run_attempt=1,
                model_key="qwen35_397b",
                outcome="failure",
                target_ref="ref",
                commit_sha="e" * 40,
                image_ref="image",
                image_id="id",
                runner_name="runner",
            )
            self.assertEqual(root.stat().st_mode & 0o7777, before)

    def test_prunes_only_expired_run_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            old_run = root / "run-9"
            current_run = root / "run-10"
            unrelated = root / "other"
            for path in (old_run, current_run, unrelated):
                path.mkdir()
            old_time = time.time() - 20 * 86400
            os.utime(old_run, (old_time, old_time))
            os.utime(current_run, (old_time, old_time))
            os.utime(unrelated, (old_time, old_time))
            removed = publish.prune_expired_runs(
                root,
                retention_days=14,
                current_run_id=10,
            )
            self.assertEqual(removed, [old_run])
            self.assertTrue(current_run.exists())
            self.assertTrue(unrelated.exists())

    def test_rejects_invalid_model_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                publish.publish_model(
                    local_model_dir=Path(tmpdir) / "local",
                    log_files=[],
                    shared_root=Path(tmpdir) / "shared",
                    run_id=1,
                    run_attempt=1,
                    model_key="../escape",
                    outcome="failure",
                    target_ref="ref",
                    commit_sha="f" * 40,
                    image_ref="image",
                    image_id="id",
                    runner_name="runner",
                )


if __name__ == "__main__":
    unittest.main()
