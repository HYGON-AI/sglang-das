import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

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
            patch.object(deepseek_v4_backend, "dsa_use_prefill_cp", return_value=True),
            envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.override(True),
        ):
            self.assertTrue(
                deepseek_v4_backend._should_use_sparse_prefill(
                    q, SimpleNamespace(), allow_cp=True
                )
            )

    def test_explicit_sparse_prefill_override_allows_cp(self):
        q = SimpleNamespace(shape=(128, 1, 8, 512))
        with (
            patch.object(deepseek_v4_backend, "_is_sm120", False),
            patch.object(deepseek_v4_backend, "dsa_use_prefill_cp", return_value=True),
            envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.override(True),
        ):
            self.assertTrue(
                deepseek_v4_backend._should_use_sparse_prefill(
                    q, SimpleNamespace(), allow_cp=False
                )
            )


class TestDeepseekV4SparsePrefillStorage(CustomTestCase):
    def test_swa_only_sparse_prefill_does_not_access_compressed_pool(self):
        cache = SimpleNamespace(
            swa_token_ids=torch.tensor([1, 2], dtype=torch.int64),
            swa_page_size=256,
            c0_combined_indices=torch.tensor([[0, 1]], dtype=torch.int32),
            c0_combined_lens=torch.tensor([2], dtype=torch.int32),
        )
        workspace = torch.zeros((2, 512), dtype=torch.bfloat16)
        backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(sparse_prefill_cache=cache),
            sparse_prefill_workspace=SimpleNamespace(get=Mock(return_value=workspace)),
            softmax_scale=0.125,
            head_dim_v=512,
        )
        pool = SimpleNamespace(
            get_extra_key_layout=Mock(side_effect=AssertionError("no compressed pool")),
            get_swa_key_layout=Mock(return_value=deepseek_v4_backend.KVLayout.V4),
            get_swa_key_buffer_radix=Mock(return_value=torch.zeros((1, 576))),
        )
        expected = torch.ones((1, 1, 512), dtype=torch.bfloat16)
        with (
            patch.object(deepseek_v4_backend, "_is_hcu", True),
            envs.SGLANG_LIGHTOP_DEQUANTIZE_K_CACHE_PAGED.override(False),
            patch.object(deepseek_v4_backend, "dequantize_k_cache_paged") as dequantize,
            patch(
                "flash_mla.flash_mla_interface.flash_mla_sparse_fwd",
                return_value=(expected, None, None),
            ),
        ):
            output = deepseek_v4_backend.DeepseekV4AttnBackend._forward_prefill_sparse(
                backend,
                torch.zeros((1, 1, 1, 512)),
                0,
                0,
                SimpleNamespace(),
                pool,
                SimpleNamespace(),
                torch.zeros(1),
            )
        self.assertIs(output, expected)
        pool.get_extra_key_layout.assert_not_called()
        dequantize.assert_called_once()
        self.assertIs(dequantize.call_args.kwargs["out"], workspace)
        self.assertIs(
            dequantize.call_args.kwargs["layout"], deepseek_v4_backend.KVLayout.V4
        )


if __name__ == "__main__":
    TestDeepseekV4SparsePrefillRouting.main()
