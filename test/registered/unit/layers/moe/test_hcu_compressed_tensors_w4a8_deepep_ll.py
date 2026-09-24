"""CPU regression tests for HCU W4A8 DeepEP low-latency dispatch."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPLLDispatchOutput,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a8_int8_moe import (
    HCUCompressedTensorsW4A8Int8DynamicMoE,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestHCUCompressedTensorsW4A8DeepEPLL(CustomTestCase):
    def setUp(self):
        self.method = HCUCompressedTensorsW4A8Int8DynamicMoE.__new__(
            HCUCompressedTensorsW4A8Int8DynamicMoE
        )
        self.method.runner_backend = MoeRunnerBackend.DEEP_GEMM
        self.hidden_states = torch.empty((2, 8, 16), dtype=torch.int8)
        self.hidden_states_scale = torch.ones((2, 8, 1), dtype=torch.float32)
        self.topk_ids = torch.zeros((3, 2), dtype=torch.int64)
        self.topk_weights = torch.ones((3, 2), dtype=torch.float32)
        self.masked_m = torch.tensor([3, 2], dtype=torch.int32)

    def make_dispatch_output(self, hidden_states=None, hidden_states_scale=None):
        return DeepEPLLDispatchOutput(
            hidden_states=(
                self.hidden_states if hidden_states is None else hidden_states
            ),
            hidden_states_scale=(
                self.hidden_states_scale
                if hidden_states_scale is None
                else hidden_states_scale
            ),
            topk_ids=self.topk_ids,
            topk_weights=self.topk_weights,
            masked_m=self.masked_m,
            expected_m=12,
        )

    def test_low_latency_reuses_dispatch_quantization_and_returns_ll_combine(
        self,
    ):
        layer = SimpleNamespace()
        output = torch.empty((2, 8, 16), dtype=torch.bfloat16)
        self.method._run_deep_gemm_masked = Mock(return_value=output)
        dispatch_output = self.make_dispatch_output()

        combine_input = self.method.apply_weights(layer, dispatch_output)

        self.assertIsInstance(combine_input, DeepEPLLCombineInput)
        self.assertIs(combine_input.hidden_states, output)
        self.assertIs(combine_input.topk_ids, self.topk_ids)
        self.assertIs(combine_input.topk_weights, self.topk_weights)
        self.method._run_deep_gemm_masked.assert_called_once_with(
            layer,
            self.hidden_states,
            self.masked_m,
            12,
            hidden_states_scale=self.hidden_states_scale,
        )

    def test_masked_gemm_reuses_int8_dispatch_input_and_returns_bf16(self):
        self.method.moe_runner_config = SimpleNamespace(
            activation="silu", swiglu_limit=10.0
        )
        layer = SimpleNamespace(
            w13_weight_packed=torch.empty((2, 32, 8), dtype=torch.int8),
            w13_weight_scale=torch.ones((2, 24, 1), dtype=torch.float32),
            w2_weight_packed=torch.empty((2, 16, 8), dtype=torch.int8),
            w2_weight_scale=torch.ones((2, 16, 1), dtype=torch.float32),
            w4a8_padded_intermediate_size=16,
        )
        q_a2 = torch.empty((2, 8, 12), dtype=torch.int8)
        q_a2_scale = torch.ones((2, 8, 1), dtype=torch.float32)

        with (
            patch(
                "lightop.quant.per_token_quant_int8",
            ) as quantize,
            patch(
                "lightop.fuse_silu_mul_clamp_quant_ep",
                return_value=(q_a2, q_a2_scale),
            ) as fused_activation,
            patch.object(
                torch.ops.sglang,
                "m_grouped_w4a8_gemm_nt_masked",
                create=True,
            ) as grouped_gemm,
        ):
            output = self.method._run_deep_gemm_masked(
                layer,
                self.hidden_states,
                self.masked_m,
                expected_m=12,
                hidden_states_scale=self.hidden_states_scale,
            )

        quantize.assert_not_called()
        self.assertEqual(grouped_gemm.call_count, 2)
        first_gemm_args = grouped_gemm.call_args_list[0].args
        self.assertIs(first_gemm_args[0], self.hidden_states)
        self.assertIs(first_gemm_args[1], self.hidden_states_scale)
        self.assertEqual(first_gemm_args[4].dtype, torch.bfloat16)
        self.assertEqual(first_gemm_args[6], self.hidden_states.shape[1])
        fused_activation.assert_called_once_with(
            input=first_gemm_args[4],
            limit=10.0,
            mask_m=self.masked_m,
            expect_m=self.hidden_states.shape[1],
        )
        second_gemm_args = grouped_gemm.call_args_list[1].args
        self.assertEqual(second_gemm_args[0].shape, (2, 8, 16))
        self.assertEqual(second_gemm_args[0][..., 12:].count_nonzero().item(), 0)
        self.assertIs(second_gemm_args[1], q_a2_scale)
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_low_latency_rejects_unquantized_activations(self):
        dispatch_output = self.make_dispatch_output(
            hidden_states=torch.empty((2, 8, 16), dtype=torch.bfloat16)
        )

        with self.assertRaisesRegex(RuntimeError, "requires INT8"):
            self.method.apply_weights(SimpleNamespace(), dispatch_output)

    def test_low_latency_rejects_invalid_scale_shape(self):
        dispatch_output = self.make_dispatch_output(
            hidden_states_scale=torch.ones((2, 8), dtype=torch.float32)
        )

        with self.assertRaisesRegex(RuntimeError, "scale shape"):
            self.method.apply_weights(SimpleNamespace(), dispatch_output)


if __name__ == "__main__":
    unittest.main()
