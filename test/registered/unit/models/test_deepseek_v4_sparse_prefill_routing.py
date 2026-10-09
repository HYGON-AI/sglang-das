import os
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.deepseek_v4_hook import apply_deepseek_v4_defaults
from sglang.srt.environ import envs
from sglang.srt.layers.attention import deepseek_v4_backend
from sglang.test.test_utils import CustomTestCase


class TestDeepseekV4SparsePrefillRouting(CustomTestCase):
    def _apply_hip_defaults(self):
        cfg = SimpleNamespace(
            dsv4_attn_backend="auto",
            max_running_requests=1,
            speculative_algorithm=None,
        )
        with (
            patch(
                "sglang.srt.arg_groups.deepseek_v4_hook.get_platform",
                return_value=SimpleNamespace(is_hip=True),
            ),
            patch(
                "sglang.srt.arg_groups.deepseek_v4_hook.resolving_view",
                return_value=cfg,
            ),
            patch("sglang.srt.arg_groups.deepseek_v4_hook.run_post_process_pass"),
        ):
            apply_deepseek_v4_defaults(SimpleNamespace(), "DeepseekV4ForCausalLM")

    def test_explicit_sparse_prefill_env_survives_hip_defaults(self):
        with envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.override(True):
            self._apply_hip_defaults()
            self.assertTrue(envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get())

    def test_hip_default_disables_sparse_prefill_when_env_is_unset(self):
        name = envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.name
        old_value = os.environ.pop(name, None)
        try:
            self._apply_hip_defaults()
            self.assertFalse(envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get())
        finally:
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value

    def test_v41_sparse_prefill_allows_cp(self):
        q = SimpleNamespace(shape=(128, 1, 8, 512))
        with (
            patch.object(deepseek_v4_backend, "_is_sm120", False),
            patch.object(
                deepseek_v4_backend, "dsa_use_prefill_cp", return_value=True
            ),
            envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.override(True),
        ):
            self.assertTrue(
                deepseek_v4_backend._should_use_sparse_prefill(
                    q, SimpleNamespace(), allow_cp=True
                )
            )

    def test_legacy_v4_sparse_prefill_still_rejects_cp(self):
        q = SimpleNamespace(shape=(128, 1, 8, 512))
        with (
            patch.object(deepseek_v4_backend, "_is_sm120", False),
            patch.object(
                deepseek_v4_backend, "dsa_use_prefill_cp", return_value=True
            ),
            envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.override(True),
        ):
            self.assertFalse(
                deepseek_v4_backend._should_use_sparse_prefill(
                    q, SimpleNamespace(), allow_cp=False
                )
            )


if __name__ == "__main__":
    TestDeepseekV4SparsePrefillRouting.main()
