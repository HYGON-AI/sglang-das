# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
import stat
import sys
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from unittest import mock


def _load_module(name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


REPO_ROOT = Path(__file__).resolve().parents[3]
notify = _load_module(
    "hcu_ci_notify_feishu",
    REPO_ROOT / "scripts" / "ci" / "hcu" / "hcu_ci_notify_feishu.py",
)
accuracy_report = _load_module(
    "hcu_accuracy_report",
    REPO_ROOT / "python" / "sglang" / "test" / "hcu_accuracy_report.py",
)
publisher = _load_module(
    "hcu_ci_publish_accuracy",
    REPO_ROOT / "scripts" / "ci" / "hcu" / "hcu_ci_publish_accuracy.py",
)

TEST_RUN_ID = 12345
TEST_RUN_ATTEMPT = 2


def _result_payload(
    model_key: str,
    model: str,
    *,
    score: float = 0.95,
    threshold: float = 0.90,
) -> dict:
    return {
        "schema_version": 1,
        "model_key": model_key,
        "model": model,
        "dataset": "gsm8k",
        "score": score,
        "threshold": threshold,
        "passed": score >= threshold,
        "num_examples": 100,
        "invalid_rate": 0.0,
        "latency_seconds": 12.5,
        "source_test": "test_accuracy.py",
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_partition_status(
    path: Path,
    partition: str,
    outcome: str,
    *,
    run_id: int = TEST_RUN_ID,
    run_attempt: int = TEST_RUN_ATTEMPT,
) -> None:
    _write_json(
        path / "partition-status.json",
        {
            "schema_version": 1,
            "partition": partition,
            "outcome": outcome,
            "run_id": run_id,
            "run_attempt": run_attempt,
            "target_ref": "0713-hcu-sglang-test",
            "commit_sha": "a" * 40,
            "image_ref": "example/sglang:latest",
            "image_id": "sha256:" + "b" * 64,
            "runner_name": "nmz4-ci-pr",
            "result_files": [],
        },
    )


def _write_complete_results(root: Path) -> None:
    for model_key, model in notify.EXPECTED_MODELS:
        _write_json(root / f"{model_key}.json", _result_payload(model_key, model))
    _write_partition_status(root / "artifact-0", "accuracy-text-0", "success")
    _write_partition_status(root / "artifact-1", "accuracy-text-1", "success")


def _column_sets(card: dict) -> list:
    return [
        element
        for element in card["body"]["elements"]
        if element.get("tag") == "column_set"
    ]


def _accuracy_table(card: dict) -> dict:
    tables = [
        element for element in card["body"]["elements"] if element.get("tag") == "table"
    ]
    assert len(tables) == 1
    return tables[0]


class AccuracyResultWriterTest(unittest.TestCase):
    def test_writes_normalized_json_and_rejects_duplicate(self):
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {accuracy_report.RESULT_DIR_ENV: tmpdir},
            clear=False,
        ):
            output = accuracy_report.write_hcu_accuracy_result(
                model_key="qwen3_32b",
                model="Qwen3-32B",
                score=0.94,
                threshold=0.90,
                num_examples=100,
                invalid_rate=None,
                latency_seconds=15.0,
                source_test="/tmp/test_qwen.py",
            )
            self.assertIsNotNone(output)
            payload = json.loads(Path(output).read_text(encoding="utf-8"))
            self.assertEqual(payload["model_key"], "qwen3_32b")
            self.assertTrue(payload["passed"])
            self.assertIsNone(payload["invalid_rate"])
            with self.assertRaises(FileExistsError):
                accuracy_report.write_hcu_accuracy_result(
                    model_key="qwen3_32b",
                    model="Qwen3-32B",
                    score=0.94,
                    threshold=0.90,
                    num_examples=100,
                    invalid_rate=0.0,
                    latency_seconds=15.0,
                    source_test="/tmp/test_qwen.py",
                )

    def test_rejects_invalid_rate(self):
        with self.assertRaises(ValueError):
            accuracy_report.write_hcu_accuracy_result(
                model_key="qwen3_32b",
                model="Qwen3-32B",
                score=0.94,
                threshold=0.90,
                num_examples=100,
                invalid_rate=1.5,
                latency_seconds=15.0,
                source_test="test_qwen.py",
            )


class ResultCollectionTest(unittest.TestCase):
    def test_complete_results_build_green_card(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_complete_results(root)
            collected = notify.collect_results(root)
            self.assertEqual(len(collected.results), len(notify.EXPECTED_MODELS))
            self.assertFalse(collected.missing_models)
            self.assertFalse(collected.diagnostics)
            card = notify.build_card(
                collected,
                branch="0713-hcu-sglang-test",
                target_ref="0713-hcu-sglang-test",
                commit_sha="a" * 40,
                image="example/sglang:test",
                workflow_result="success",
                run_url="https://github.com/HYGON-AI/sglang-das/actions/runs/1",
            )
            self.assertEqual(card["header"]["template"], "green")
            self.assertEqual(card["schema"], "2.0")
            self.assertNotIn("elements", card)
            self.assertNotIn(
                "note",
                [element.get("tag") for element in card["body"]["elements"]],
            )
            self.assertNotIn(
                "action",
                [element.get("tag") for element in card["body"]["elements"]],
            )
            metadata = card["body"]["elements"][-2]
            self.assertEqual(metadata["tag"], "div")
            self.assertEqual(metadata["text"]["text_size"], "notation")
            button = card["body"]["elements"][-1]
            self.assertEqual(button["tag"], "button")
            self.assertEqual(
                button["behaviors"],
                [
                    {
                        "type": "open_url",
                        "default_url": (
                            "https://github.com/HYGON-AI/sglang-das/actions/runs/1"
                        ),
                    }
                ],
            )
            self.assertIn(
                f"{len(notify.EXPECTED_MODELS)}/{len(notify.EXPECTED_MODELS)} 通过",
                card["header"]["title"]["content"],
            )
            self.assertNotIn("0713", card["header"]["title"]["content"])
            self.assertIn(
                f"本次 {len(notify.EXPECTED_MODELS)} 个模型全部达到阈值",
                card["body"]["elements"][0]["text"]["content"],
            )
            column_sets = _column_sets(card)
            self.assertEqual(len(column_sets), 1)
            table = _accuracy_table(card)
            self.assertEqual(
                [column["display_name"] for column in table["columns"]],
                ["模型", "精度", "阈值", "样本数", "结论"],
            )
            self.assertEqual(
                [column["width"] for column in table["columns"]],
                ["250px", "80px", "80px", "80px", "80px"],
            )
            self.assertTrue(table["freeze_first_column"])
            self.assertEqual(table["page_size"], len(notify.EXPECTED_MODELS))
            self.assertEqual(len(table["rows"]), len(notify.EXPECTED_MODELS))
            self.assertEqual(
                table["rows"][0],
                {
                    "model": notify.EXPECTED_MODELS[0][1],
                    "score": "95.00%",
                    "threshold": "90.00%",
                    "samples": "100",
                    "status": [{"text": "通过", "color": "green"}],
                },
            )
            rendered = json.dumps(card, ensure_ascii=False)
            self.assertNotIn("基础设施诊断", rendered)
            self.assertNotIn("精度回归", rendered)
            payload = {
                "timestamp": "1700000000",
                "sign": "x" * 44,
                "msg_type": "interactive",
                "card": card,
            }
            self.assertLess(
                len(json.dumps(payload, ensure_ascii=False).encode("utf-8")),
                notify.MAX_CARD_BYTES,
            )

    def test_empty_artifact_directory_builds_orange_missing_card(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collected = notify.collect_results(Path(tmpdir))
            card = notify.build_card(
                collected,
                branch="0713-hcu-sglang-test",
                target_ref="0713-hcu-sglang-test",
                commit_sha="d" * 40,
                image="example/sglang:test",
                workflow_result="failure",
                run_url="https://github.com/HYGON-AI/sglang-das/actions/runs/4",
            )
            self.assertEqual(len(collected.missing_models), len(notify.EXPECTED_MODELS))
            self.assertEqual(card["header"]["template"], "orange")
            self.assertIn(
                f"{len(notify.EXPECTED_MODELS)} 个模型未完成",
                card["body"]["elements"][0]["text"]["content"],
            )
            self.assertEqual(
                _accuracy_table(card)["rows"][0],
                {
                    "model": notify.EXPECTED_MODELS[0][1],
                    "score": "--",
                    "threshold": "--",
                    "samples": "--",
                    "status": [{"text": "未完成", "color": "orange"}],
                },
            )

    def test_regression_is_red_even_with_missing_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            key, model = notify.EXPECTED_MODELS[0]
            _write_json(
                root / f"{key}.json",
                _result_payload(key, model, score=0.70, threshold=0.88),
            )
            _write_partition_status(root / "artifact-0", "accuracy-text-0", "success")
            _write_partition_status(root / "artifact-1", "accuracy-text-1", "success")
            collected = notify.collect_results(root)
            card = notify.build_card(
                collected,
                branch="0713-hcu-sglang-test",
                target_ref="0713-hcu-sglang-test",
                commit_sha="b" * 40,
                image="example/sglang:test",
                workflow_result="failure",
                run_url="https://github.com/HYGON-AI/sglang-das/actions/runs/2",
            )
            self.assertEqual(card["header"]["template"], "red")
            self.assertIn(key, collected.regressions)
            rendered = json.dumps(card, ensure_ascii=False)
            self.assertIn("未达标", rendered)
            self.assertNotIn("精度回归", rendered)

    def test_duplicate_and_malformed_results_are_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            key, model = notify.EXPECTED_MODELS[0]
            payload = _result_payload(key, model)
            _write_json(root / "artifact-0" / f"{key}.json", payload)
            _write_json(root / "artifact-1" / f"{key}.json", payload)
            (root / "broken.json").write_text("{broken", encoding="utf-8")
            _write_partition_status(root / "artifact-0", "accuracy-text-0", "success")
            _write_partition_status(root / "artifact-1", "accuracy-text-1", "success")
            collected = notify.collect_results(root)
            self.assertIn(key, collected.duplicate_models)
            self.assertIn(key, collected.missing_models)
            self.assertGreaterEqual(len(collected.diagnostics), 2)

    def test_failed_non_accuracy_matrix_with_complete_scores_is_green(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_complete_results(root)
            collected = notify.collect_results(root)
            card = notify.build_card(
                collected,
                branch="0713-hcu-sglang-test",
                target_ref="0713-hcu-sglang-test",
                commit_sha="c" * 40,
                image="example/sglang:test",
                workflow_result="failure",
                run_url="https://github.com/HYGON-AI/sglang-das/actions/runs/3",
            )
            self.assertEqual(card["header"]["template"], "green")
            self.assertIn(
                f"本次 {len(notify.EXPECTED_MODELS)} 个模型全部达到阈值",
                card["body"]["elements"][0]["text"]["content"],
            )

    def test_failed_accuracy_partition_with_complete_scores_is_green(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_complete_results(root)
            _write_partition_status(
                root / "artifact-0",
                "accuracy-text-0",
                "failure",
            )
            collected = notify.collect_results(root)
            card = notify.build_card(
                collected,
                branch="0713-hcu-sglang-test",
                target_ref="0713-hcu-sglang-test",
                commit_sha="c" * 40,
                image="example/sglang:test",
                workflow_result="success",
                run_url="https://github.com/HYGON-AI/sglang-das/actions/runs/3",
            )
            self.assertEqual(card["header"]["template"], "green")
            self.assertEqual(
                collected.failed_partitions,
                {"accuracy-text-0": "failure"},
            )

    def test_preview_is_clearly_labeled_and_uses_all_models(self):
        collected = notify.build_preview_results(
            branch="v0.5.12_dev",
            target_ref="a" * 40,
            commit_sha="a" * 40,
            image="notification-preview/no-model",
            run_id=TEST_RUN_ID,
            run_attempt=TEST_RUN_ATTEMPT,
        )
        card = notify.build_card(
            collected,
            branch="v0.5.12_dev",
            target_ref="a" * 40,
            commit_sha="a" * 40,
            image="notification-preview/no-model",
            workflow_result="success",
            run_url="https://github.com/HYGON-AI/sglang-das/actions/runs/5",
            preview=True,
        )
        self.assertIn("精度预览", card["header"]["title"]["content"])
        self.assertIn(
            "合成数据",
            card["body"]["elements"][0]["text"]["content"],
        )
        self.assertEqual(
            len(_accuracy_table(card)["rows"]),
            len(notify.EXPECTED_MODELS),
        )

    def test_wrong_run_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_partition_status(
                root / "partition-0",
                "accuracy-text-0",
                "success",
                run_id=TEST_RUN_ID - 1,
            )
            collected = notify.collect_results(
                root,
                expected_run_id=TEST_RUN_ID,
                expected_run_attempt=TEST_RUN_ATTEMPT,
            )
            self.assertIn("accuracy-text-0", collected.missing_partitions)
            self.assertTrue(
                any("does not match expected" in item for item in collected.diagnostics)
            )

    def test_duplicate_partition_never_becomes_valid_again(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for index in range(3):
                _write_partition_status(
                    root / f"partition-{index}",
                    "accuracy-text-0",
                    "success",
                )
            collected = notify.collect_results(root)
            self.assertIn("accuracy-text-0", collected.duplicate_partitions)
            self.assertIn("accuracy-text-0", collected.missing_partitions)

    def test_hidden_staging_directory_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            key, model = notify.EXPECTED_MODELS[0]
            _write_json(
                root / ".accuracy-text-0.tmp" / f"{key}.json",
                _result_payload(key, model),
            )
            collected = notify.collect_results(root)
            self.assertNotIn(key, collected.results)


class SharedResultPublisherTest(unittest.TestCase):
    def test_existing_shared_directory_does_not_require_ownership(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shared_root = Path(tmpdir)
            with mock.patch.object(
                Path,
                "chmod",
                side_effect=PermissionError("not the directory owner"),
            ) as chmod:
                publisher._ensure_directory(shared_root)
            chmod.assert_not_called()

    def test_existing_shared_directory_must_be_writable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shared_root = Path(tmpdir)
            with mock.patch.object(publisher.os, "access", return_value=False):
                with self.assertRaisesRegex(
                    PermissionError,
                    "shared result directory is not writable",
                ):
                    publisher._ensure_directory(shared_root)

    def test_publishes_partition_atomically_with_group_permissions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local_results = root / "local"
            shared_root = root / "shared"
            key, model = notify.EXPECTED_MODELS[0]
            _write_json(
                local_results / f"{key}.json",
                _result_payload(key, model),
            )

            published = publisher.publish_partition(
                local_results_dir=local_results,
                shared_root=shared_root,
                run_id=TEST_RUN_ID,
                run_attempt=TEST_RUN_ATTEMPT,
                partition="accuracy-text-0",
                outcome="success",
                target_ref="0713-hcu-sglang-test",
                commit_sha="a" * 40,
                image_ref="example/sglang:latest",
                image_id="sha256:" + "b" * 64,
                runner_name="nmz4-ci-pr",
            )

            status_payload = json.loads(
                (published / "partition-status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status_payload["run_id"], TEST_RUN_ID)
            self.assertEqual(status_payload["run_attempt"], TEST_RUN_ATTEMPT)
            self.assertEqual(status_payload["result_files"], [f"{key}.json"])
            self.assertEqual(stat.S_IMODE(published.stat().st_mode), 0o2775)
            self.assertEqual(
                stat.S_IMODE((published / f"{key}.json").stat().st_mode), 0o664
            )
            self.assertFalse(
                any(path.name.startswith(".") for path in published.parent.iterdir())
            )

    def test_existing_partition_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            kwargs = {
                "local_results_dir": root / "local",
                "shared_root": root / "shared",
                "run_id": TEST_RUN_ID,
                "run_attempt": TEST_RUN_ATTEMPT,
                "partition": "accuracy-text-0",
                "outcome": "failure",
                "target_ref": "0713-hcu-sglang-test",
                "commit_sha": "a" * 40,
                "image_ref": "example/sglang:latest",
                "image_id": "sha256:" + "b" * 64,
                "runner_name": "nmz4-ci-pr",
            }
            publisher.publish_partition(**kwargs)
            with self.assertRaises(FileExistsError):
                publisher.publish_partition(**kwargs)

    def test_prunes_only_expired_run_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shared_root = Path(tmpdir)
            expired = shared_root / "run-1"
            current = shared_root / f"run-{TEST_RUN_ID}"
            unrelated = shared_root / "model-cache"
            for path in (expired, current, unrelated):
                path.mkdir()
            old_time = time.time() - 20 * 86400
            os.utime(expired, (old_time, old_time))
            os.utime(unrelated, (old_time, old_time))

            removed = publisher.prune_expired_runs(
                shared_root,
                retention_days=14,
                current_run_id=TEST_RUN_ID,
                now=time.time(),
            )
            self.assertEqual(removed, [expired])
            self.assertFalse(expired.exists())
            self.assertTrue(current.exists())
            self.assertTrue(unrelated.exists())


class FeishuSendTest(unittest.TestCase):
    class _Response:
        def __init__(self, payload: dict):
            self.body = json.dumps(payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self):
            return self.body

    def test_signature_matches_known_vector(self):
        self.assertEqual(
            notify.feishu_signature(1700000000, "test-secret"),
            "mbm4Y4oluIPQ00qlBIhX8vAZ0EKv3nw0LuTb91jPL84=",
        )

    def test_retries_then_succeeds(self):
        attempts = []
        sleeps = []

        def opener(request, timeout):
            attempts.append((request, timeout))
            if len(attempts) < 3:
                raise urllib.error.URLError("temporary failure")
            return self._Response({"code": 0, "msg": "success"})

        notify.send_card(
            "https://open.feishu.cn/open-apis/bot/v2/hook/test",
            "test-secret",
            {"schema": "2.0", "header": {}, "body": {"elements": []}},
            urlopen_func=opener,
            sleep_func=sleeps.append,
            time_func=lambda: 1700000000,
        )
        self.assertEqual(len(attempts), 3)
        self.assertEqual(sleeps, [2, 4])

    def test_raises_after_final_failure(self):
        attempts = []

        def opener(request, timeout):
            attempts.append((request, timeout))
            raise urllib.error.URLError("still unavailable")

        with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
            notify.send_card(
                "https://open.feishu.cn/open-apis/bot/v2/hook/test",
                "test-secret",
                {"schema": "2.0", "header": {}, "body": {"elements": []}},
                urlopen_func=opener,
                sleep_func=lambda _: None,
                time_func=lambda: 1700000000,
            )
        self.assertEqual(len(attempts), 3)


if __name__ == "__main__":
    unittest.main()
