import os
import unittest

from sglang.test.ci.ci_register import register_dcu_ci
from sglang.test.dcu_cookbook_utils import (
    assert_cookbook_min_output_throughput,
    CookbookServer,
    GLM51_8GPU_PERF_MODELS,
    run_random_serving_perf,
    selected_configs,
)
from sglang.test.dcu_utils import get_int_env
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST

register_dcu_ci(
    est_time=14400,
    suite="nightly-dcu-perf-text",
    nightly=True,
)


COOKBOOK_GLM51_8GPU_MIN_OUTPUT_TPS = {
    "GLM-5.1-Channel-FP8": 100.0,
}


class TestGlm51Cookbook8GpuServingPerfDCU(unittest.TestCase):
    def test_glm51_random_serving_perf(self):
        output_dir = os.environ.get(
            "SGLANG_DCU_PERF_OUTPUT_DIR", "performance_profiles_dcu_text"
        )
        num_prompts = get_int_env("SGLANG_DCU_PERF_NUM_PROMPTS", 32)
        input_len = get_int_env("SGLANG_DCU_PERF_INPUT_LEN", 2048)
        output_len = get_int_env("SGLANG_DCU_PERF_OUTPUT_LEN", 256)
        configs = selected_configs(
            GLM51_8GPU_PERF_MODELS, "SGLANG_DCU_GLM51_8GPU_PERF_MODEL_FILTER"
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
                    COOKBOOK_GLM51_8GPU_MIN_OUTPUT_TPS,
                    "SGLANG_DCU_GLM51_8GPU_MIN_OUTPUT_TPS",
                )


if __name__ == "__main__":
    unittest.main()
