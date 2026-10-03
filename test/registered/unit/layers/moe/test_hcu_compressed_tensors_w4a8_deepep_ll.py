"""CPU regression tests for HCU W4A8 DeepEP low-latency dispatch."""

import unittest
import weakref
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

    def test_masked_gemm_releases_first_quantization_before_activation(self):
        self.method.moe_runner_config = SimpleNamespace(
            activation="silu", swiglu_limit=10.0
        )
        hidden_states = torch.zeros((2, 8, 16), dtype=torch.bfloat16)
        layer = SimpleNamespace(
            w13_weight_packed=torch.empty((2, 32, 8), dtype=torch.int8),
            w13_weight_scale=torch.ones((2, 32, 1), dtype=torch.float32),
            w2_weight_packed=torch.empty((2, 16, 8), dtype=torch.int8),
            w2_weight_scale=torch.ones((2, 16, 1), dtype=torch.float32),
            w4a8_padded_intermediate_size=16,
        )
        first_quantization_refs = []
        gemm_calls = 0
        activation_calls = 0

        def quantize(input):
            return (
                torch.zeros_like(input, dtype=torch.int8),
                torch.ones((*input.shape[:2], 1), dtype=torch.float32),
            )

        def assert_first_quantization_released():
            self.assertEqual(len(first_quantization_refs), 2)
            for tensor_ref in first_quantization_refs:
                self.assertIsNone(
                    tensor_ref(), "GEMM1 quantization survives into GEMM2"
                )

        def grouped_gemm(input, scale, weight, weight_scale, output, *args):
            nonlocal gemm_calls
            gemm_calls += 1
            if gemm_calls == 1:
                first_quantization_refs.extend((weakref.ref(input), weakref.ref(scale)))
            else:
                assert_first_quantization_released()
            output.zero_()

        def fused_activation(input, limit, mask_m, expect_m):
            nonlocal activation_calls
            activation_calls += 1
            assert_first_quantization_released()
            return (
                torch.zeros((2, 8, 16), dtype=torch.int8),
                torch.ones((2, 8, 1), dtype=torch.float32),
            )

        with (
            patch("lightop.quant.per_token_quant_int8", new=quantize),
            patch("lightop.fuse_silu_mul_clamp_quant_ep", new=fused_activation),
            patch.object(
                torch.ops.sglang,
                "m_grouped_w4a8_gemm_nt_masked",
                new=grouped_gemm,
                create=True,
            ),
        ):
            output = self.method._run_deep_gemm_masked(
                layer,
                hidden_states,
                self.masked_m,
                expected_m=8,
                guard_activation_scales=True,
            )

        self.assertEqual(gemm_calls, 2)
        self.assertEqual(activation_calls, 1)
        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(output.count_nonzero().item(), 0)

    def test_masked_gemm_guards_both_normal_activation_scales(self):
        self.method.moe_runner_config = SimpleNamespace(
            activation="silu", swiglu_limit=10.0
        )
        hidden_states = torch.zeros((2, 8, 16), dtype=torch.bfloat16)
        q_a1 = torch.zeros_like(hidden_states, dtype=torch.int8)
        q_a1_scale = torch.arange(16, dtype=torch.float32).view(2, 8, 1)
        q_a2 = torch.zeros((2, 8, 16), dtype=torch.int8)
        q_a2_scale = q_a1_scale + 1
        layer = SimpleNamespace(
            w13_weight_packed=torch.empty((2, 32, 8), dtype=torch.int8),
            w13_weight_scale=torch.ones((2, 32, 1), dtype=torch.float32),
            w2_weight_packed=torch.empty((2, 16, 8), dtype=torch.int8),
            w2_weight_scale=torch.ones((2, 16, 1), dtype=torch.float32),
            w4a8_padded_intermediate_size=16,
        )
        with (
            patch(
                "lightop.quant.per_token_quant_int8",
                return_value=(q_a1, q_a1_scale),
            ),
            patch(
                "lightop.fuse_silu_mul_clamp_quant_ep",
                return_value=(q_a2, q_a2_scale),
            ),
            patch.object(
                torch.ops.sglang,
                "m_grouped_w4a8_gemm_nt_masked",
                create=True,
            ) as grouped_gemm,
        ):
            output = self.method._run_deep_gemm_masked(
                layer,
                hidden_states,
                self.masked_m,
                expected_m=8,
                guard_activation_scales=True,
            )

        self.assertEqual(grouped_gemm.call_count, 2)
        for invocation, original in zip(
            grouped_gemm.call_args_list, (q_a1_scale, q_a2_scale)
        ):
            guarded = invocation.args[1]
            torch.testing.assert_close(guarded, original, atol=0, rtol=0)
            self.assertEqual(guarded.shape, original.shape)
            self.assertNotEqual(guarded.data_ptr(), original.data_ptr())
            self.assertGreaterEqual(
                guarded.untyped_storage().nbytes()
                - guarded.numel() * guarded.element_size(),
                2 * 1024 * 1024,
            )
        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_normal_enables_scale_guard_without_splitting_expert_compute(self):
        x = torch.arange(32, dtype=torch.bfloat16).view(2, 16)
        topk_ids = torch.tensor([[0, 1], [1, -1]], dtype=torch.int64)
        topk_weights = torch.tensor([[0.25, 0.75], [1.0, 0.0]])
        layer = SimpleNamespace(w13_weight_scale=torch.ones((2, 32, 1)))
        dispatch_output = SimpleNamespace(
            hidden_states=x,
            hidden_states_scale=None,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            num_recv_tokens_per_expert=[1, 2],
        )

        def scatter(x, ids, counts, starts, padded, offsets, indices, **kwargs):
            positions = [0, 0]
            for token in range(ids.shape[0]):
                for slot in range(ids.shape[1]):
                    expert = int(ids[token, slot])
                    if expert >= 0:
                        row = int(starts[expert]) + positions[expert]
                        padded[row].copy_(x[token])
                        indices[token, slot] = row
                        positions[expert] += 1

        def gemm(layer, padded, counts, expected_m, guard_activation_scales=False):
            self.assertTrue(guard_activation_scales)
            self.assertEqual(padded.shape, (2, 256, 16))
            self.assertEqual(counts.tolist(), [1, 2])
            self.assertEqual(expected_m, 2)
            return padded * 2

        def gather(padded, ids, weights, indices, output):
            output.zero_()
            for token in range(ids.shape[0]):
                for slot in range(ids.shape[1]):
                    if ids[token, slot] >= 0:
                        output[token].add_(
                            padded[indices[token, slot]] * weights[token, slot]
                        )

        with (
            patch(
                "sglang.kernels.ops.moe.ep_moe_kernels.ep_scatter_no_scale",
                side_effect=scatter,
            ),
            patch(
                "sglang.kernels.ops.moe.ep_moe_kernels.ep_gather",
                side_effect=gather,
            ),
            patch.object(
                self.method, "_run_deep_gemm_masked", side_effect=gemm
            ) as grouped_compute,
        ):
            result = self.method._apply_deepep_normal_deep_gemm(layer, dispatch_output)

        self.assertEqual(grouped_compute.call_count, 1)
        torch.testing.assert_close(result.hidden_states, x * 2, atol=0, rtol=0)
        self.assertIs(result.topk_ids, topk_ids)
        self.assertIs(result.topk_weights, topk_weights)

    def test_low_latency_rejects_invalid_scale_shape(self):
        dispatch_output = self.make_dispatch_output(
            hidden_states_scale=torch.ones((2, 8), dtype=torch.float32)
        )

        with self.assertRaisesRegex(RuntimeError, "scale shape"):
            self.method.apply_weights(SimpleNamespace(), dispatch_output)


if __name__ == "__main__":
    unittest.main()
