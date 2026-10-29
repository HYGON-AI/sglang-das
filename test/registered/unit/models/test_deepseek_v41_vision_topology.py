"""Coverage for DeepSeek-V4.1 HCU text-only vision fallback."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepseekV41VisionTopology(unittest.TestCase):
    def _should_disable(
        self, *, backend=MoeA2ABackend.NONE, attn_cp_size=1, pp_size=1
    ):
        parallel = SimpleNamespace(
            attn_cp_size=attn_cp_size,
            pp_group=SimpleNamespace(world_size=pp_size),
        )
        with (
            patch.object(deepseek_v4, "_is_hcu", True),
            patch.object(deepseek_v4, "get_parallel", return_value=parallel),
            patch.object(deepseek_v4, "get_moe_a2a_backend", return_value=backend),
        ):
            return deepseek_v4._should_disable_v41_vision_for_hcu(
                SimpleNamespace(model_type="deepseek_v41", vision_n_layers=2)
            )

    def test_megamoe_prefill_cp_keeps_vision(self):
        """A dropped tower lets image pad ids reach the hash-MoE lookup unremapped."""
        self.assertFalse(
            self._should_disable(backend=MoeA2ABackend.MEGAMOE, attn_cp_size=8)
        )

    def test_unvalidated_a2a_backend_disables_vision(self):
        self.assertTrue(self._should_disable(backend=MoeA2ABackend.MORI))

    def test_pipeline_parallel_disables_vision(self):
        self.assertTrue(
            self._should_disable(backend=MoeA2ABackend.MEGAMOE, pp_size=2)
        )


if __name__ == "__main__":
    unittest.main()
