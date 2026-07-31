# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import unittest

from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.hcu_evalscope_utils import run_hcu_evalscope_case

register_hcu_ci(
    est_time=21600, suite="nightly-hcu-accuracy-evalscope", nightly=True
)


class TestEvalScopeQwen35397BHCU(unittest.TestCase):
    def test_evalscope_accuracy(self):
        run_hcu_evalscope_case("qwen35_397b_a17b_channel_fp8")


if __name__ == "__main__":
    unittest.main()
