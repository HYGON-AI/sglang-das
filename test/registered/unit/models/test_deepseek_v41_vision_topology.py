"""Coverage for DeepSeek-V4.1 HCU text-only vision fallback."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepseekV41VisionTopology(unittest.TestCase):
    @staticmethod
    def _config():
        return SimpleNamespace(model_type="deepseek_v41", vision_n_layers=2)

    @staticmethod
    def _parallel(*, attn_cp_size=1, pp_size=1):
        return SimpleNamespace(
            attn_cp_size=attn_cp_size,
            pp_group=SimpleNamespace(world_size=pp_size),
        )

    def _should_disable(
        self,
        *,
        disaggregation_mode="null",
        attn_cp_size=1,
        pp_size=1,
        moe_a2a_none=True,
    ):
        backend = SimpleNamespace(is_none=lambda: moe_a2a_none)
        with (
            patch.object(deepseek_v4, "_is_hcu", True),
            patch.object(
                deepseek_v4,
                "get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_mode=disaggregation_mode
                ),
            ),
            patch.object(
                deepseek_v4,
                "get_parallel",
                return_value=self._parallel(
                    attn_cp_size=attn_cp_size, pp_size=pp_size
                ),
            ),
            patch.object(
                deepseek_v4, "get_moe_a2a_backend", return_value=backend
            ),
        ):
            return deepseek_v4._should_disable_v41_vision_for_hcu(
                self._config()
            )

    def test_standalone_prefill_cp_disables_vision(self):
        self.assertTrue(self._should_disable(attn_cp_size=8))

    def test_standalone_moe_a2a_disables_vision(self):
        self.assertTrue(self._should_disable(moe_a2a_none=False))

    def test_disaggregated_text_serving_disables_vision(self):
        self.assertTrue(self._should_disable(disaggregation_mode="prefill"))

    def test_supported_tp_topology_keeps_vision(self):
        self.assertFalse(self._should_disable())


if __name__ == "__main__":
    unittest.main()
