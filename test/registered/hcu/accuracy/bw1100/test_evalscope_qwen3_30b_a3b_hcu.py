# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import unittest

from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.hcu_evalscope_utils import run_hcu_evalscope_case

register_hcu_ci(
    est_time=14400, suite="nightly-hcu-accuracy-evalscope", nightly=True
)


class TestEvalScopeQwen330BA3BHCU(unittest.TestCase):
    def test_evalscope_accuracy(self):
        run_hcu_evalscope_case("qwen3_30b_a3b")


if __name__ == "__main__":
    unittest.main()
