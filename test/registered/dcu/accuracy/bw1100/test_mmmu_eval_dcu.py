import json
import os
import shlex
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_dcu_ci
from sglang.test.run_eval import run_eval_once
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
    "--tp-size",
    "4",
    "--disable-cuda-graph",
    "--disable-custom-all-reduce",
    "--cuda-graph-max-bs",
    "16",
    "--mem-fraction-static",
    "0.55",
]

DEFAULT_DCU_MMMU_MODEL_CANDIDATES = [
    "/public/opendas/DL_DATA/llm-models/qwen2.5/Qwen2.5-VL-72B-Instruct",
    "/public/opendas/DL_DATA/llm-models/vllm-gptq-models/qwen2.5/Qwen2.5-VL-72B-Instruct",
    "/public/opendas/DL_DATA/llm-models/vllm-gptq-models/qwen2.5/Qwen2.5-VL-3B-Instruct",
    "/public/opendas/DL_DATA/llm-models/qwen2/Qwen2-VL-2B-Instruct",
]

DEFAULT_DCU_MMMU_DATASET_CANDIDATES = [
    "/tmp/dcu_datasets/mmmu/mmmu/Math/test-00000-of-00001.parquet",
    "/home/github/data/mmmu/Math/test-00000-of-00001.parquet",
    "/public/opendas/DL_DATA/llm-models/multimodal-datasets/MMMU/Math/validation-00000-of-00001.parquet",
]


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

        import io

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
        import ast

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
        for candidate in DEFAULT_DCU_MMMU_MODEL_CANDIDATES:
            if os.path.exists(candidate):
                return candidate
        raise AssertionError(
            "No local DCU MMMU model path found. Set "
            f"{name} to one of: {DEFAULT_DCU_MMMU_MODEL_CANDIDATES}"
        )
    if model.startswith(("/", ".")) and not os.path.exists(model):
        raise AssertionError(f"{name} points to a missing local model path: {model}")
    return model


def _find_mmmu_parquet(path: str) -> str:
    dataset_path = Path(path)
    if dataset_path.is_file():
        if dataset_path.suffix != ".parquet":
            raise AssertionError(
                f"{path} is not a parquet file. Set a valid MMMU parquet path."
            )
        return str(dataset_path)

    if not dataset_path.is_dir():
        raise AssertionError(f"{path} is neither a file nor a directory")

    candidates = sorted(dataset_path.rglob("*.parquet"))
    if not candidates:
        raise AssertionError(f"No parquet files found under MMMU path: {path}")

    preferred = []
    for split in ("validation", "dev", "test"):
        preferred.extend(
            p
            for p in candidates
            if p.parent.name == "Math" and p.name.startswith(f"{split}-")
        )
    selected = preferred[0] if preferred else candidates[0]
    print(f"Resolved MMMU parquet dataset: {selected}")
    return str(selected)


def _get_dataset_path_env(name: str) -> str:
    path = os.environ.get(name)
    if not path:
        for candidate in DEFAULT_DCU_MMMU_DATASET_CANDIDATES:
            if os.path.exists(candidate):
                return _find_mmmu_parquet(candidate)
        raise AssertionError(
            "No local DCU MMMU parquet dataset found. Set "
            f"{name} to a parquet file or dataset directory."
        )
    if not os.path.exists(path):
        raise AssertionError(f"{name} points to a missing path: {path}")
    return _find_mmmu_parquet(path)


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
        cls.dataset_path = _get_dataset_path_env("SGLANG_DCU_MMMU_DATASET_PATH")
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
            eval_obj = DCULocalParquetMMMUEval(
                self.dataset_path, self.num_examples, self.num_threads
            )
            result, latency, sampler = run_eval_once(
                args, f"{self.base_url}/v1", eval_obj
            )
            metrics = result.metrics | {"score": result.score}
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
