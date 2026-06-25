import ast
import io
import json
import os
import shlex
import unittest
import warnings
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_dcu_ci
from sglang.test.run_eval import run_eval, run_eval_once
from sglang.test.simple_eval_common import set_ulimit
from sglang.test.simple_eval_mmmu_vlm import MMMUVLMEval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    check_evaluation_test_results,
    popen_launch_server,
    write_results_to_json,
)

register_dcu_ci(est_time=7200, suite="nightly-dcu-vlm", nightly=True)

DEFAULT_DCU_VLM_SERVER_ARGS = [
    "--attention-backend",
    "fa3",
    "--mm-attention-backend",
    "fa3",
    "--page-size",
    "64",
    "--log-level",
    "warning",
    "--log-level-http",
    "warning",
    "--enable-multimodal",
    "--trust-remote-code",
]

DEFAULT_DCU_MMMU_MODEL_CANDIDATES = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2-VL-2B-Instruct",
]

DEFAULT_DCU_MMMU_DATASET_PATH = ""


class DCULocalParquetMMMUEval(MMMUVLMEval):
    def __init__(
        self,
        dataset_path: str,
        num_examples: int,
        num_threads: int,
        response_answer_regex: str = None,
    ):
        self.num_examples = num_examples
        self.num_threads = num_threads
        self.seed = 42
        self.response_answer_regex = response_answer_regex
        self.samples = self._prepare_local_parquet_samples(dataset_path, num_examples)

    @staticmethod
    def _image_to_data_uri(image_obj) -> str | None:
        if image_obj is None:
            return None
        try:
            import pandas as pd

            if pd.isna(image_obj):
                return None
        except Exception:
            pass

        from PIL import Image

        if isinstance(image_obj, dict):
            image_bytes = image_obj.get("bytes")
            image_path = image_obj.get("path")
            if image_bytes:
                image = Image.open(io.BytesIO(image_bytes))
            elif image_path and os.path.exists(image_path):
                image = Image.open(image_path)
            else:
                return None
        elif hasattr(image_obj, "convert"):
            image = image_obj
        else:
            return None
        return MMMUVLMEval._to_data_uri(image)

    @classmethod
    def _prepare_local_parquet_samples(cls, dataset_path: str, k: int) -> list[dict]:
        import pandas as pd

        df = pd.read_parquet(dataset_path)
        samples = []
        for _, row in df.iterrows():
            image_data = None
            for col in [f"image_{idx}" for idx in range(1, 8)]:
                if col in row:
                    image_data = cls._image_to_data_uri(row[col])
                    if image_data:
                        break
            if not image_data:
                continue

            raw_options = row.get("options")
            options = None
            index2ans = None
            all_choices = None
            question_type = row.get("question_type") or "open"
            if raw_options:
                try:
                    options = (
                        raw_options
                        if isinstance(raw_options, list)
                        else ast.literal_eval(str(raw_options))
                    )
                    if isinstance(options, list) and options:
                        index2ans, all_choices = cls._build_mc_mapping(options)
                        question_type = "multiple-choice"
                except Exception:
                    options = None

            prompt_text = f"{row.get('question', '')}\n"
            if options:
                letters = [chr(ord("A") + i) for i in range(len(options))]
                for letter, opt in zip(letters, options):
                    prompt_text += f"{letter}. {opt}\n"
                prompt_text += (
                    "\nAnswer the following multiple-choice question. "
                    "The last line of your response should be of the "
                    "following format: 'Answer: $LETTER' (without quotes) "
                    "where LETTER is one of the options. "
                    "Think step by step before answering."
                )
            else:
                prompt_text += "\nAnswer: "

            samples.append(
                {
                    "id": row.get("id", f"local:{len(samples)}"),
                    "final_input_prompt": prompt_text,
                    "image_data": image_data,
                    "answer": row.get("answer"),
                    "question_type": question_type,
                    "index2ans": index2ans,
                    "all_choices": all_choices,
                    "category": row.get("__subject__") or row.get("subject") or "Math",
                }
            )
            if k and len(samples) >= k:
                break

        if not samples:
            raise RuntimeError(f"No usable MMMU samples loaded from {dataset_path}")
        return samples


def _get_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def _get_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def _get_model_env(name: str) -> str:
    model = os.environ.get(name, "")
    if not model:
        return DEFAULT_DCU_MMMU_MODEL_CANDIDATES[0]
    if model.startswith(("/", ".")) and not os.path.exists(model):
        raise AssertionError(f"{name} points to a missing local model path: {model}")
    return model


def _get_optional_path_env(name: str) -> str | None:
    path = os.environ.get(name)
    if not path:
        if DEFAULT_DCU_MMMU_DATASET_PATH and os.path.exists(
            DEFAULT_DCU_MMMU_DATASET_PATH
        ):
            return DEFAULT_DCU_MMMU_DATASET_PATH
        return None
    if not os.path.exists(path):
        raise AssertionError(f"{name} points to a missing path: {path}")
    return path


def _get_server_args_env(name: str) -> list[str]:
    value = os.environ.get(name)
    if value:
        return shlex.split(value)
    return list(DEFAULT_DCU_VLM_SERVER_ARGS)


class TestBW1100MMMUEvalDCU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _get_model_env("SGLANG_DCU_MMMU_MODEL")
        cls.threshold = _get_float_env("SGLANG_DCU_MMMU_THRESHOLD", 0.35)
        cls.latency_threshold = _get_float_env("SGLANG_DCU_MMMU_LATENCY_THRESHOLD", 1e9)
        cls.num_examples = _get_int_env("SGLANG_DCU_MMMU_NUM_EXAMPLES", 100)
        cls.num_threads = _get_int_env("SGLANG_DCU_MMMU_NUM_THREADS", 4)
        cls.max_tokens = _get_int_env("SGLANG_DCU_MMMU_MAX_TOKENS", 30)
        cls.dataset_path = _get_optional_path_env("SGLANG_DCU_MMMU_DATASET_PATH")
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_mmmu(self):
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*socket"
        )
        process = None
        all_results = []

        try:
            process = popen_launch_server(
                model=self.model,
                base_url=self.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=_get_server_args_env("SGLANG_DCU_MMMU_SERVER_ARGS"),
            )

            set_ulimit()
            os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
            args = SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="mmmu",
                num_examples=self.num_examples,
                num_threads=self.num_threads,
                max_tokens=self.max_tokens,
                dataset_path=self.dataset_path,
                return_latency=True,
            )
            if self.dataset_path:
                eval_obj = DCULocalParquetMMMUEval(
                    self.dataset_path, self.num_examples, self.num_threads
                )
                result, latency, sampler = run_eval_once(
                    args, f"{self.base_url}/v1", eval_obj
                )
                metrics = result.metrics | {"score": result.score}
            else:
                metrics, latency = run_eval(args)
            metrics["score"] = round(metrics["score"], 4)
            metrics["latency"] = round(latency, 4)
            write_results_to_json(self.model, metrics, "w")
            all_results.append((self.model, metrics["score"], metrics["latency"], None))
        except Exception as exc:
            all_results.append((self.model, None, None, str(exc)))
            raise
        finally:
            if process is not None:
                kill_process_tree(process.pid)

        try:
            with open("results.json", "r") as f:
                print("\nFinal Results from results.json:")
                print(json.dumps(json.load(f), indent=2))
        except Exception as exc:
            print(f"Error reading results.json: {exc}")

        check_evaluation_test_results(
            all_results,
            self.__class__.__name__,
            model_accuracy_thresholds={self.model: self.threshold},
            model_latency_thresholds={self.model: self.latency_threshold},
        )


if __name__ == "__main__":
    unittest.main()
