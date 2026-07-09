import unittest

from sglang.test.ci.ci_register import register_cuda_ci, register_dcu_ci
from sglang.test.gpt_oss_common import BaseTestGptOss

# DCU_CSV_CI_UNVERIFIED: Registered from sglang.csv CI coverage; not re-tested in this framework pass.
register_dcu_ci(
    est_time=300,
    suite="nightly-dcu-4-gpu",
    nightly=True,
    disabled="DCU CSV CI placeholder: 4-GPU GPT-OSS BF16 path needs BW1100 large-model validation before enabling.",
)

register_cuda_ci(est_time=220, stage="base-c", runner_config="4-gpu-h100")
register_cuda_ci(est_time=220, stage="base-c", runner_config="4-gpu-b200")


class TestGptOss4GpuBf16(BaseTestGptOss):
    def test_bf16_120b(self):
        self.run_test(
            model_variant="120b",
            quantization="bf16",
            expected_score_of_reasoning_effort={
                "low": 0.58,
            },
            other_args=["--tp", "4", "--cuda-graph-max-bs", "200"],
        )


if __name__ == "__main__":
    unittest.main()
