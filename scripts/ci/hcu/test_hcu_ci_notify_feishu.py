import importlib.util
import json
import os
import sys
import tempfile
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


def _write_partition_status(path: Path, partition: str, outcome: str) -> None:
    _write_json(
        path / "partition-status.json",
        {"schema_version": 1, "partition": partition, "outcome": outcome},
    )


def _write_complete_results(root: Path) -> None:
    for model_key, model in notify.EXPECTED_MODELS:
        _write_json(root / f"{model_key}.json", _result_payload(model_key, model))
    _write_partition_status(root / "artifact-0", "accuracy-text-0", "success")
    _write_partition_status(root / "artifact-1", "accuracy-text-1", "success")


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
            self.assertEqual(len(collected.results), 10)
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
            self.assertIn("全部通过", card["header"]["title"]["content"])

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
            self.assertEqual(len(collected.missing_models), 10)
            self.assertEqual(card["header"]["template"], "orange")

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

    def test_failed_matrix_with_complete_scores_is_orange(self):
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
            self.assertEqual(card["header"]["template"], "orange")


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
            {"header": {}, "elements": []},
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
                {"header": {}, "elements": []},
                urlopen_func=opener,
                sleep_func=lambda _: None,
                time_func=lambda: 1700000000,
            )
        self.assertEqual(len(attempts), 3)


if __name__ == "__main__":
    unittest.main()
