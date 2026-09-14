import unittest

from sglang.test.ci.ci_register import register_cuda_ci, register_hcu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.kits.prefix_cache_branching_kit import PrefixCacheBranchingMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=343, stage="base-c", runner_config="4-gpu-h100")

# HCU_CSV_CI_UNVERIFIED: Registered from sglang.csv CI coverage; not re-tested in this framework pass.
register_hcu_ci(
    est_time=300,
    suite="nightly-hcu-4",
    nightly=True,
    disabled="HCU CSV CI placeholder: 4-GPU Qwen3-Next model path needs local model mapping before enabling.",
)

QWEN3_NEXT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"

_COMMON_ARGS = [
    "--trust-remote-code",
    "--tp-size",
    "4",
    "--chunked-prefill-size",
    "2048",
    "--mamba-radix-cache-strategy",
    "extra_buffer_lazy",
    "--attention-backend",
    "triton",
]


def _make_args(*, page_size=1, track_interval=2):
    return [
        *_COMMON_ARGS,
        "--mamba-track-interval",
        str(track_interval),
        "--page-size",
        str(page_size),
    ]


class TestQwen3NextLazyExtraBuffer(
    GSM8KMixin, KLDivergenceMixin, PrefixCacheBranchingMixin, DefaultServerBase
):
    model = QWEN3_NEXT_MODEL
    cache_chunk_size = 64
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.002
    other_args = _make_args(page_size=1, track_interval=2)


class TestQwen3NextLazyExtraBufferLargePage(
    GSM8KMixin, KLDivergenceMixin, PrefixCacheBranchingMixin, DefaultServerBase
):
    model = QWEN3_NEXT_MODEL
    cache_chunk_size = 64
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.002
    other_args = _make_args(page_size=2, track_interval=2)


if __name__ == "__main__":
    unittest.main()
