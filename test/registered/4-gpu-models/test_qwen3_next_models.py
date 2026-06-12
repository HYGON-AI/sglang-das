import unittest

from sglang.test.ci.ci_register import register_cuda_ci, register_dcu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.kits.prefix_cache_branching_kit import PrefixCacheBranchingMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=350, suite="stage-c-test-4-gpu-h100")

# DCU_CSV_CI_UNVERIFIED: Registered from sglang.csv CI coverage; not re-tested in this framework pass.
register_dcu_ci(
    est_time=300,
    suite="nightly-dcu-4-gpu",
    nightly=True,
    disabled="DCU CSV CI placeholder: 4-GPU Qwen3-Next model path needs local model mapping before enabling.",
)

QWEN3_NEXT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"


class TestQwen3Next(
    GSM8KMixin, KLDivergenceMixin, PrefixCacheBranchingMixin, DefaultServerBase
):
    model = QWEN3_NEXT_MODEL
    cache_chunk_size = 64
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.0025
    other_args = [
        "--tp-size",
        "4",
        "--chunked-prefill-size",
        "2048",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "128",
    ]


if __name__ == "__main__":
    unittest.main()
