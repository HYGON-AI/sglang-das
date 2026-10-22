"""HCU DSpark PD admission uses normalized attention-DP widths."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import deepseek_v4_hook, parallel_hook
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.speculative.ragged_verify import RaggedVerifyMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDsv41HcuPdTopology(unittest.TestCase):
    def check(self, *, hcu=True, verify=RaggedVerifyMode.STATIC, **overrides):
        cfg = SimpleNamespace(
            tp_size=8,
            dp_size=1,
            attn_dp_size=8,
            enable_dp_attention=False,
            attn_cp_size=1,
            dcp_size=1,
            enable_prefill_cp=False,
            cp_strategy="interleave",
            disaggregation_mode="decode",
            disaggregation_transfer_backend="mooncake",
            speculative_algorithm="DSPARK",
            dsv4_attn_backend="flashmla",
            enable_encoder_swa_bounded_replay=False,
            enable_decoder_swa_bounded_replay=False,
            enable_hisparse=False,
            enable_two_batch_overlap=False,
            moe_a2a_backend="none",
            pp_size=1,
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend=Backend.DISABLED)
            ),
        )
        vars(cfg).update(overrides)

        def declare(target, reason, **values):
            vars(target).update(values)

        with ExitStack() as stack:
            for module in (parallel_hook, deepseek_v4_hook):
                stack.enter_context(patch.object(module, "resolving_view", lambda x: x))
            stack.enter_context(
                patch.object(parallel_hook, "declare_resolution", declare)
            )
            stack.enter_context(
                patch.object(deepseek_v4_hook, "is_hcu", return_value=hcu)
            )
            stack.enter_context(
                patch.object(
                    deepseek_v4_hook,
                    "model_config_of",
                    return_value=SimpleNamespace(
                        hf_config=SimpleNamespace(model_type="deepseek_v41")
                    ),
                )
            )
            stack.enter_context(
                patch(
                    "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
                    return_value=False,
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.speculative.ragged_verify.read_ragged_verify_mode",
                    return_value=verify,
                )
            )
            # Exercise main's actual legacy normalization, not a reimplementation.
            parallel_hook.handle_deprecated_dp_attention(cfg)
            deepseek_v4_hook.validate_deepseek_v41_features(cfg)
        return cfg

    def test_legacy_decode_flags_are_accepted_after_normalization(self):
        cfg = self.check(dp_size=8, attn_dp_size=1, enable_dp_attention=True)
        self.assertEqual(
            (cfg.dp_size, cfg.attn_dp_size, cfg.enable_dp_attention), (1, 8, False)
        )

    def test_native_attention_dp_decode_is_accepted(self):
        self.check()

    def test_invalid_decode_topologies_are_rejected(self):
        for values in (
            {"dp_size": 8},
            {"attn_dp_size": 4},
            {"attn_dp_size": 1},
            {"attn_cp_size": 2},
            {"enable_prefill_cp": True},
        ):
            with self.subTest(**values), self.assertRaisesRegex(
                ValueError, "DSpark PD on HCU"
            ):
                self.check(**values)

    def test_prefill_cp_topology_is_preserved(self):
        self.check(
            disaggregation_mode="prefill",
            attn_dp_size=1,
            attn_cp_size=8,
            enable_prefill_cp=True,
        )

    def test_prefill_rejects_attention_dp(self):
        with self.assertRaisesRegex(ValueError, "attention DP=1"):
            self.check(
                disaggregation_mode="prefill", attn_cp_size=8, enable_prefill_cp=True
            )

    def test_transport_and_verify_constraints_are_preserved(self):
        for values in (
            {"dcp_size": 2},
            {"disaggregation_transfer_backend": "nixl"},
            {"verify": RaggedVerifyMode.COMPACT},
        ):
            with self.subTest(**values), self.assertRaisesRegex(
                ValueError, "static verify"
            ):
                self.check(**values)

    def test_non_hcu_attention_dp_remains_supported(self):
        self.check(hcu=False)


if __name__ == "__main__":
    unittest.main()
