import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.models.deepseek_common.amd import deepseek_v4_hip
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite='base-a-test-cpu')

# (eligibility fn, lightop probe, per-linear capability check) per operand dtype.
_ROUTES = {
    'fp8': (
        'fused_rmsnorm_fp8_operand_eligible',
        'is_lightop_rms_fp8_quant_available',
        'linear_takes_fp8_operand',
    ),
    'int8': (
        'fused_rmsnorm_int8_operand_eligible',
        'is_lightop_sglang_rms_quant_available',
        'opt_into_int8_operand',
    ),
}


class TestDeepseekV4HcuRmsQuantGuard(unittest.TestCase):
    def _eligible(self, route, *linears, linear_eligible=True):
        eligible_fn, probe, takes_operand = _ROUTES[route]
        with (
            patch.object(deepseek_v4_hip, '_is_hcu', True),
            patch.object(
                deepseek_v4_hip.envs.SGLANG_USE_FUSED_RMS_QUANT,
                'get',
                return_value=True,
            ),
            patch.object(deepseek_v4_hip, probe, return_value=True),
            patch.object(
                deepseek_v4_hip, takes_operand, return_value=linear_eligible
            ),
        ):
            return getattr(deepseek_v4_hip, eligible_fn)(*linears)

    def test_eligible_linear_allows_fused_operand(self):
        for route in _ROUTES:
            with self.subTest(route=route):
                self.assertTrue(self._eligible(route, object()))

    def test_ineligible_linear_disables_fused_operand(self):
        for route in _ROUTES:
            with self.subTest(route=route):
                self.assertFalse(
                    self._eligible(route, object(), linear_eligible=False)
                )

    def test_no_linears_disables_fused_operand(self):
        for route in _ROUTES:
            with self.subTest(route=route):
                self.assertFalse(self._eligible(route))

    def test_only_v41_fuses_q_norm(self):
        """DeepSeek V4 (not V4.1) keeps the unfused q_norm even when eligible."""
        with (
            patch.object(
                deepseek_v4_hip, 'fused_rmsnorm_fp8_operand_eligible', return_value=True
            ),
            patch.object(
                deepseek_v4_hip, 'fused_rmsnorm_int8_operand_eligible', return_value=True
            ),
        ):
            for is_dsv41 in (True, False):
                with self.subTest(is_dsv41=is_dsv41):
                    attn = SimpleNamespace(is_dsv41=is_dsv41, wq_b=object())
                    deepseek_v4_hip.init_fused_rmsnorm_operand(attn)
                    self.assertIs(attn.fused_rmsnorm_fp8_operand, is_dsv41)
                    self.assertIs(attn.fused_rmsnorm_int8_operand, is_dsv41)


if __name__ == '__main__':
    unittest.main()
