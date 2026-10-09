# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Hygon modifications to this file are licensed under the Apache License,
# Version 2.0 (the "License"); you may not use these modifications except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/openai/simple-evals/

"""
Measuring Massive Multitask Language Understanding
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2009.03300
"""

import glob
import os
import random
import re
from typing import Optional

import pandas

from sglang.test import simple_eval_common as common
from sglang.test.simple_eval_common import (
    ANSWER_PATTERN_MULTICHOICE,
    HTML_JINJA,
    QUERY_TEMPLATE_MULTICHOICE,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
    format_multichoice_question,
)

subject2category = {
    "abstract_algebra": "stem",
    "anatomy": "other",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}


class MMLUEval(Eval):
    def __init__(
        self,
        filename: str,
        num_examples: Optional[int],
        num_threads: int,
        query_template: str | None = None,
    ):
        if os.path.isdir(filename):
            examples = self._load_local_hf_mmlu(filename)
        elif "://" in filename:
            df = pandas.read_csv(filename, storage_options={"timeout": 30})
            examples = [row.to_dict() for _, row in df.iterrows()]
        else:
            df = pandas.read_csv(filename)
            examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples
        self.num_threads = num_threads
        self.query_template = query_template

    @staticmethod
    def _load_local_hf_mmlu(dataset_path: str) -> list[dict]:
        from datasets import concatenate_datasets, load_dataset

        answer_letters = ["A", "B", "C", "D"]
        datasets = []
        for subject in sorted(subject2category):
            files = sorted(
                glob.glob(os.path.join(dataset_path, subject, "test-*.parquet"))
            )
            if not files:
                files = sorted(
                    glob.glob(
                        os.path.join(dataset_path, subject, "validation-*.parquet")
                    )
                )
            if not files:
                continue
            d = load_dataset("parquet", data_files=files, split="train")
            datasets.append(d)
        if not datasets:
            raise RuntimeError(f"Failed to load local MMLU dataset from {dataset_path}")

        merged = concatenate_datasets(datasets)
        examples = []
        for row in merged:
            choices = row.get("choices") or []
            if len(choices) < 4:
                continue
            answer = row.get("answer")
            if isinstance(answer, int):
                answer = answer_letters[answer]
            examples.append(
                {
                    "Question": row.get("question", ""),
                    "A": choices[0],
                    "B": choices[1],
                    "C": choices[2],
                    "D": choices[3],
                    "Answer": answer,
                    "Subject": row.get("subject", "unknown"),
                }
            )
        return examples

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(
                    content=format_multichoice_question(
                        row,
                        (
                            self.query_template
                            if self.query_template is not None
                            else QUERY_TEMPLATE_MULTICHOICE
                        ),
                    ),
                    role="user",
                )
            ]
            response_text = sampler(prompt_messages)
            response_text = response_text or ""
            match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
            extracted_answer = match.group(1) if match else None
            score = 1.0 if extracted_answer == row["Answer"] else 0.0
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            category = subject2category.get(row["Subject"], "other")
            return SingleEvalResult(
                html=html, score=score, metrics={category: score}, convo=convo
            )

        results = common.map_with_progress(fn, self.examples, self.num_threads)
        return common.aggregate_results(results)
