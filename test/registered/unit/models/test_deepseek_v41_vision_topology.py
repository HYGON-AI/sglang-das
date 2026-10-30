"""Coverage for DeepSeek-V4.1 HCU vision topology gating."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepseekV41VisionTopology(unittest.TestCase):
    def _should_disable(
        self,
        *,
        backend=MoeA2ABackend.NONE,
        disaggregation_mode="null",
        attn_cp_size=1,
        pp_size=1,
    ):
        parallel = SimpleNamespace(
            attn_cp_size=attn_cp_size,
            pp_group=SimpleNamespace(world_size=pp_size),
        )
        disagg = SimpleNamespace(disaggregation_mode=disaggregation_mode)
        with (
            patch.object(deepseek_v4, "_is_hcu", True),
            patch.object(deepseek_v4, "get_disagg", return_value=disagg),
            patch.object(deepseek_v4, "get_parallel", return_value=parallel),
            patch.object(deepseek_v4, "get_moe_a2a_backend", return_value=backend),
        ):
            return deepseek_v4._should_disable_v41_vision_for_hcu(
                SimpleNamespace(model_type="deepseek_v41", vision_n_layers=2)
            )

    def test_standalone_cp_keeps_vision(self):
        # CP prefill embeds globally and shards at the model boundary.
        self.assertFalse(self._should_disable(attn_cp_size=8))

    def test_cp_with_deepep_keeps_vision(self):
        self.assertFalse(
            self._should_disable(backend=MoeA2ABackend.DEEPEP, attn_cp_size=8)
        )

    def test_megamoe_prefill_cp_keeps_vision(self):
        """A dropped tower lets image pad ids reach the hash-MoE lookup unremapped."""
        self.assertFalse(
            self._should_disable(backend=MoeA2ABackend.MEGAMOE, attn_cp_size=8)
        )

    def test_unvalidated_a2a_backend_disables_vision(self):
        self.assertTrue(self._should_disable(backend=MoeA2ABackend.MORI))

    def test_prefill_role_keeps_vision(self):
        # The prefill role owns image processing, with or without CP.
        self.assertFalse(self._should_disable(disaggregation_mode="prefill"))
        self.assertFalse(
            self._should_disable(disaggregation_mode="prefill", attn_cp_size=8)
        )

    def test_decode_role_disables_vision(self):
        # The decode role never sees image inputs.
        self.assertTrue(self._should_disable(disaggregation_mode="decode"))

    def test_pipeline_parallel_disables_vision(self):
        self.assertTrue(
            self._should_disable(backend=MoeA2ABackend.MEGAMOE, pp_size=2)
        )

    def test_supported_tp_topology_keeps_vision(self):
        self.assertFalse(self._should_disable())

    @staticmethod
    def _encoder_only_shell():
        model = deepseek_v4.DeepseekV4ForCausalLM.__new__(
            deepseek_v4.DeepseekV4ForCausalLM
        )
        deepseek_v4.nn.Module.__init__(model)
        model.config = SimpleNamespace(model_type="deepseek_v41")
        model.encoder_only = True
        model.model = None
        model.lm_head = None
        model._routed_experts_weights_of_layer = deepseek_v4.LazyValue(
            lambda: {}
        )
        return model

    def test_encoder_only_lifecycle_matches_glm5(self):
        model = self._encoder_only_shell()

        self.assertEqual(model.routed_experts_weights_of_layer, {})
        self.assertEqual(model.start_layer, 0)
        self.assertEqual(model.end_layer, 0)
        self.assertFalse(model.wants_prefill_autotune())
        self.assertEqual(
            model.autotune_prefill_kernels(16, dtype=deepseek_v4.torch.float16),
            0,
        )

        with self.assertRaisesRegex(
            AttributeError, "not available in encoder-only mode"
        ):
            model.get_input_embeddings()
        with self.assertRaisesRegex(
            AttributeError, "not available in encoder-only mode"
        ):
            model.get_embed_and_head()
        with self.assertRaisesRegex(
            AttributeError, "not available in encoder-only mode"
        ):
            model.set_embed_and_head(None, None)
        with self.assertRaisesRegex(
            AttributeError, "not available in encoder-only mode"
        ):
            model.set_dspark_layers_to_capture([0])


if __name__ == "__main__":
    unittest.main()
