# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("hcu_evalscope_accuracy.py")
SPEC = importlib.util.spec_from_file_location("hcu_evalscope_accuracy", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestHcuEvalScopeAccuracy(unittest.TestCase):
    def test_extract_final_code_uses_last_block_after_think(self):
        prediction = """
```python
def wrong():
    return 0
```
</think>
The final implementation is:
```python
    def answer():
        return 42
```
"""
        self.assertEqual(
            MODULE.extract_final_code(prediction),
            "def answer():\n    return 42\n",
        )

    def test_validate_jsonl_checks_count_and_digest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.jsonl"
            path.write_bytes(b'{"id": 1}\n{"id": 2}\n')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            result = MODULE.validate_jsonl(path, 2, digest)
            self.assertEqual(result["records"], 2)
            self.assertEqual(result["sha256"], digest)

    def test_build_summary_passes_and_supports_report_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_report(root, "gsm8k", 0.91, 200)
            self._write_report(root, "humaneval", 0.80, 164)
            summary = MODULE.build_summary(
                root,
                "model-key",
                "Model",
                {"gsm8k": 200, "humaneval": 164},
                {"gsm8k": 0.88, "humaneval": None},
            )
            self.assertEqual(summary["status"], "passed")
            self.assertEqual(summary["datasets"]["gsm8k"]["status"], "passed")
            self.assertEqual(
                summary["datasets"]["humaneval"]["status"], "report_only"
            )

    def test_build_summary_marks_regression_and_missing_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_report(root, "math_500", 0.71, 200)
            summary = MODULE.build_summary(
                root,
                "model-key",
                "Model",
                {"math_500": 200, "humaneval": 164},
                {"math_500": 0.72, "humaneval": 0.75},
            )
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["datasets"]["math_500"]["status"], "failed")
            self.assertEqual(summary["datasets"]["humaneval"]["status"], "missing")

    def test_load_report_rejects_incomplete_sample_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_report(root, "gsm8k", 0.90, 199)
            with self.assertRaisesRegex(ValueError, "expected 200"):
                MODULE.load_report(root, "gsm8k", 200)

    def test_load_report_counts_max_token_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_report(root, "math_500", 0.90, 2)
            review_dir = root / "math_500" / "reviews" / "Model"
            review_dir.mkdir(parents=True)
            rows = [
                {
                    "messages": [
                        {
                            "role": "assistant",
                            "perf_metrics": {"output_tokens": 16384},
                        }
                    ]
                },
                {
                    "messages": [
                        {
                            "role": "assistant",
                            "perf_metrics": {"output_tokens": 1024},
                        }
                    ]
                },
            ]
            (review_dir / "math_500.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            report = MODULE.load_report(
                root, "math_500", 2, max_tokens=16384
            )
            self.assertEqual(report["truncated_outputs"], 1)

    @staticmethod
    def _write_report(root: Path, dataset: str, score: float, count: int):
        report_dir = root / dataset / "reports" / "Model"
        report_dir.mkdir(parents=True)
        report = {
            "dataset_name": dataset,
            "score": score,
            "num": count,
            "metrics": [{"name": "mean_acc", "score": score, "num": count}],
            "perf_metrics": {
                "summary": {
                    "n_samples": count,
                    "latency": {"mean": 1.5},
                    "throughput": {"avg_output_tps": 12.0},
                }
            },
        }
        (report_dir / f"{dataset}.json").write_text(
            json.dumps(report), encoding="utf-8"
        )


if __name__ == "__main__":
    unittest.main()
