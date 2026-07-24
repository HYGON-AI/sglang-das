# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest

from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.hcu_cookbook_utils import (
    assert_cookbook_min_output_throughput,
    CookbookServer,
    DEEPSEEK_V32_8GPU_PERF_MODELS,
    run_random_serving_perf,
    selected_configs,
)
from sglang.test.hcu_utils import get_int_env
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST

register_hcu_ci(
    est_time=14400,
    suite="nightly-hcu-perf-text",
    nightly=True,
)


COOKBOOK_DEEPSEEK_V32_8GPU_MIN_OUTPUT_TPS = {
    "DeepSeek-V3.2-Channel-FP8": 95.0,
}


class TestDeepSeekV32Cookbook8GpuServingPerfHCU(unittest.TestCase):
    def test_deepseek_v32_random_serving_perf(self):
        output_dir = os.environ.get(
            "SGLANG_HCU_PERF_OUTPUT_DIR", "performance_profiles_hcu_text"
        )
        num_prompts = get_int_env("SGLANG_HCU_PERF_NUM_PROMPTS", 32)
        input_len = get_int_env("SGLANG_HCU_PERF_INPUT_LEN", 2048)
        output_len = get_int_env("SGLANG_HCU_PERF_OUTPUT_LEN", 256)
        configs = selected_configs(
            DEEPSEEK_V32_8GPU_PERF_MODELS,
            "SGLANG_HCU_DEEPSEEK_V32_8GPU_PERF_MODEL_FILTER",
        )

        for config in configs:
            with self.subTest(model=config.name):
                with CookbookServer(config, DEFAULT_URL_FOR_TEST):
                    result = run_random_serving_perf(
                        config,
                        DEFAULT_URL_FOR_TEST,
                        output_dir,
                        num_prompts=num_prompts,
                        input_len=input_len,
                        output_len=output_len,
                    )
                self.assertGreater(result["request_throughput"], 0)
                self.assertGreater(result["input_throughput"], 0)
                self.assertGreater(result["output_throughput"], 0)
                assert_cookbook_min_output_throughput(
                    config,
                    result,
                    COOKBOOK_DEEPSEEK_V32_8GPU_MIN_OUTPUT_TPS,
                    "SGLANG_HCU_DEEPSEEK_V32_8GPU_MIN_OUTPUT_TPS",
                )


if __name__ == "__main__":
    unittest.main()
