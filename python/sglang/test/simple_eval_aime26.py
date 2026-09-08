# SPDX-License-Identifier: MIT
# Copyright (c) 2024 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Adapted from https://github.com/openai/simple-evals/

"""
AIME 2026 - American Invitational Mathematics Examination 2026
Dataset: MathArena/aime_2026
https://huggingface.co/datasets/MathArena/aime_2026

The American Invitational Mathematics Examination (AIME) is a challenging
competition math exam. All answers are integers from 000 to 999.
"""

from sglang.test.simple_eval_aime25 import AIME25Eval
from sglang.test.simple_eval_common import import_load_dataset


class AIME26Eval(AIME25Eval):
    def _load_examples(self) -> list[dict]:
        load_dataset = import_load_dataset()

        dataset = load_dataset("MathArena/aime_2026", split="train")
        return [
            {"question": row["problem"], "answer": str(row["answer"])}
            for row in dataset
        ]
