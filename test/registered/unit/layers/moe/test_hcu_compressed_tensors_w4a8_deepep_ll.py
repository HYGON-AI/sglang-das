"""CPU regression tests for HCU W4A8 DeepEP low-latency dispatch."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch

from sglang.srt.batch_overlap.two_batch_overlap import MaybeTboDeepEPDispatcher
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPLLDispatchOutput,
    _DeepEPDispatcherImplLowLatency,
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
        layer = SimpleNamespace(dispatcher=Mock())
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
        layer.dispatcher.record_combine_input_ready_event.assert_called_once_with()

    def test_maybe_tbo_dispatcher_forwards_ready_event_to_its_only_inner(self):
        dispatcher = MaybeTboDeepEPDispatcher.__new__(MaybeTboDeepEPDispatcher)
        inner = Mock()
        dispatcher._inners = [inner]

        dispatcher.record_combine_input_ready_event()

        inner.record_combine_input_ready_event.assert_called_once_with()

    def test_maybe_tbo_dispatcher_rejects_ambiguous_tbo_inner(self):
        dispatcher = MaybeTboDeepEPDispatcher.__new__(MaybeTboDeepEPDispatcher)
        dispatcher._inners = [Mock(), Mock()]

        with self.assertRaisesRegex(RuntimeError, "do not support TBO"):
            dispatcher.record_combine_input_ready_event()

    def test_low_latency_combine_waits_for_w4a8_producer_event(self):
        impl = _DeepEPDispatcherImplLowLatency.__new__(
            _DeepEPDispatcherImplLowLatency
        )
        producer_event = Mock()
        finish_event = Mock()
        finish_hook = Mock()
        compute_stream = Mock()
        timeline = Mock()
        compute_stream.wait_event = timeline.wait_event
        device_module = SimpleNamespace(
            current_stream=Mock(return_value=compute_stream)
        )
        combined = torch.empty((3, 16), dtype=torch.bfloat16)
        handle = object()
        buffer = Mock()
        buffer.low_latency_combine = timeline.low_latency_combine
        timeline.low_latency_combine.return_value = (
            combined,
            finish_event,
            finish_hook,
        )

        impl.device_module = device_module
        impl._combine_input_ready_event = producer_event
        impl._combine_input_ready_event_pending = False
        impl._get_buffer = Mock(return_value=buffer)
        impl.handle = handle
        impl.overlap_args = None
        impl.meta_overlap_args = None
        impl.return_recv_hook = False
        impl.packed_recv_count = None

        impl.record_combine_input_ready_event()
        with patch(
            "sglang.srt.layers.moe.token_dispatcher.deepep."
            "_deepep_precompile_tp_barrier"
        ):
            result = impl._combine_core(
                combined,
                self.topk_ids,
                self.topk_weights,
            )

        producer_event.record.assert_called_once_with(compute_stream)
        self.assertEqual(
            timeline.mock_calls[:2],
            [
                call.wait_event(producer_event),
                call.low_latency_combine(
                    x=combined,
                    topk_idx=self.topk_ids,
                    topk_weights=self.topk_weights,
                    handle=handle,
                    zero_copy=False,
                    async_finish=True,
                    return_recv_hook=False,
                ),
            ],
        )
        self.assertIs(result[0], combined)
        self.assertIs(result[1], finish_event)
        self.assertIs(result[2], finish_hook)
        self.assertFalse(impl._combine_input_ready_event_pending)

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
