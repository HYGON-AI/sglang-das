"""CPU regression tests for HCU W4A8 DeepEP low-latency dispatch."""

import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch
from compressed_tensors.quantization import QuantizationArgs

from sglang.srt.batch_overlap.two_batch_overlap import MaybeTboDeepEPDispatcher
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPLLDispatchOutput,
    DeepEPNormalCombineInput,
    DeepEPNormalDispatchOutput,
    _DeepEPDispatcherImplLowLatency,
    _DeepEPDispatcherImplNormal,
)
from sglang.srt.layers.moe.utils import DeepEPMode, MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.slimquant_w4a8_marlin import (
    HCUW4A8Int8DeepEPMoEMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a8_int8_moe import (
    HCUCompressedTensorsW4A8Int8DynamicMoE,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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



class TestHCUW4A8Reference(CustomTestCase):
    def setUp(self):
        for target, value in (
            ("sglang.srt.layers.moe.ep_moe.layer._use_w4a8_contiguous_hipc", True),
            ("sglang.srt.layers.moe.ep_moe.layer._use_w4a8_masked_hipc", True),
        ):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch(
            "sglang.srt.layers.quantization.slimquant_w4a8_marlin.get_deepep_mode",
            return_value=DeepEPMode.NORMAL,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.scheme = {
            "weights": QuantizationArgs(
                num_bits=4,
                type="int",
                strategy="channel",
                symmetric=True,
                group_size=-1,
            ),
            "input_activations": QuantizationArgs(
                num_bits=8, type="int", strategy="token", dynamic=True, symmetric=True
            ),
        }
        self.config = CompressedTensorsConfig(
            {"Linear": self.scheme}, [], "pack-quantized", {}, []
        )

    def make_layer(self, rank=0, tp=1):
        layer = FusedMoE.__new__(FusedMoE)
        torch.nn.Module.__init__(layer)
        layer.quant_config = self.config
        layer.scheme = None
        layer.quant_method = HCUW4A8Int8DeepEPMoEMethod(self.config)
        layer.quant_method.moe_runner_config = SimpleNamespace(swiglu_limit=10.0)
        layer.moe_tp_rank = rank
        layer.moe_tp_size = tp
        layer.moe_runner_config = SimpleNamespace(is_gated=True)
        layer.use_triton_kernels = False
        layer.use_padded_loading = False
        layer.use_presharded_weights = False
        layer.use_flashinfer_trtllm_moe = False
        layer._has_fused_shared = False
        return layer

    def make_reference_layer(self, **weights):
        layer = DeepEPMoE.__new__(DeepEPMoE)
        torch.nn.Module.__init__(layer)
        layer.scheme = None
        layer.quant_method = Mock(hcu_w4a8_hipc=True)
        layer.moe_runner_config = SimpleNamespace(activation="silu", swiglu_limit=10.0)
        layer.use_hcu_w4a8_reference = True
        layer.deprecate_flag = False
        layer.quant_config = self.config
        layer.use_w4a8_marlin = True
        layer.use_fp8_w8a8 = False
        layer.use_w4afp8 = False
        layer.use_w8a8_marlin = False
        layer.use_bf16_marlin = False
        layer.use_w4a16_marlin = False
        layer.dispatcher = Mock()
        for name, value in weights.items():
            setattr(layer, name, value)
        return layer

    def make_reference_dispatch_output(
        self, hidden_states=None, hidden_states_scale=None
    ):
        return DeepEPLLDispatchOutput(
            hidden_states=(
                torch.ones((2, 8, 16), dtype=torch.int8)
                if hidden_states is None
                else hidden_states
            ),
            hidden_states_scale=(
                torch.ones((2, 8, 1), dtype=torch.float32)
                if hidden_states_scale is None
                else hidden_states_scale
            ),
            topk_ids=torch.zeros((3, 2), dtype=torch.int64),
            topk_weights=torch.ones((3, 2), dtype=torch.float32),
            masked_m=torch.tensor([3, 2], dtype=torch.int32),
            expected_m=12,
        )

    def test_reference_normal_compact_layout_and_weighted_gather(self):
        x = torch.arange(32, dtype=torch.int8).view(2, 16)
        ids = torch.tensor([[0, 1], [1, -1]], dtype=torch.int64)
        weights = torch.tensor([[0.25, 0.75], [1.0, 0.0]])
        dispatch = DeepEPNormalDispatchOutput(
            hidden_states=x,
            hidden_states_scale=torch.ones((2, 1)),
            topk_ids=ids,
            topk_weights=weights,
            num_recv_tokens_per_expert=[256, 512],
        )
        layer = self.make_reference_layer(
            w13_weight_scale=torch.ones((2, 32, 1)),
            w2_weight_scale=torch.ones((2, 16, 1)),
            w13_weight=object(),
            w2_weight=object(),
            w4a8_padded_intermediate_size=16,
        )

        def scatter(q, scale, topk, counts, output, out_scale, total, **kwargs):
            self.assertEqual(output.shape, (768, 16))
            output.zero_()
            out_scale.fill_(1)
            output[0].copy_(q[0])
            output[256].copy_(q[0])
            output[257].copy_(q[1])
            return (
                torch.cat((torch.zeros(256), torch.ones(512))).int(),
                torch.tensor([[0, 256], [257, -1]], dtype=torch.int32),
            )

        def gemm(a, b, output, indices):
            self.assertEqual(output.ndim, 2)
            output.copy_(a[0].float().repeat(1, output.shape[1] // 16))

        def activation(input, limit):
            self.assertEqual(limit, 10.0)
            return input[:, :16].to(torch.int8), torch.ones((768, 1))

        def gather(input, topk, topk_weights, indices, output):
            output.zero_()
            for token in range(2):
                for slot in range(2):
                    if indices[token, slot] >= 0:
                        output[token].add_(
                            input[indices[token, slot]] * topk_weights[token, slot]
                        )

        with (
            patch(
                "lightop.quant.per_token_quant_int8",
            ) as quantize,
            patch(
                "sglang.srt.layers.moe.ep_moe.layer._ep_scatter_with_optional_lightop",
                new=scatter,
            ),
            patch(
                "sglang.srt.layers.moe.ep_moe.layer._ep_gather_with_optional_lightop",
                new=gather,
            ),
            patch(
                "deepgemm.m_grouped_w4a8_gemm_nt_contiguous_hipc", new=gemm, create=True
            ),
            patch("lightop.fuse_silu_mul_clamp_quant", new=activation),
        ):
            result = layer.run_moe_core(dispatch)
        quantize.assert_not_called()
        torch.testing.assert_close(
            result.hidden_states, x.to(torch.bfloat16), atol=0, rtol=0
        )
        self.assertIsInstance(result, DeepEPNormalCombineInput)
        self.assertIs(result.topk_ids, ids)
        layer.quant_method.apply.assert_not_called()

    def test_reference_normal_rejects_unaligned_counts(self):
        dispatch = DeepEPNormalDispatchOutput(
            hidden_states=torch.empty((1, 16), dtype=torch.int8),
            hidden_states_scale=torch.ones((1, 1)),
            topk_ids=torch.zeros((1, 1)).long(),
            topk_weights=torch.ones((1, 1)),
            num_recv_tokens_per_expert=[1],
        )
        with self.assertRaisesRegex(RuntimeError, "256-aligned"):
            layer = self.make_reference_layer(
                w13_weight_scale=torch.ones((1, 32, 1))
            )
            layer.run_moe_core(dispatch)

    def test_reference_normal_rejects_bf16_or_invalid_scales(self):
        layer = self.make_reference_layer()
        for dtype, scale, error in (
            (torch.bfloat16, None, "requires INT8"),
            (torch.int8, torch.ones((2, 1), dtype=torch.bfloat16), "requires INT8"),
            (torch.int8, torch.ones((2,)), "scale shape"),
        ):
            with self.subTest(dtype=dtype, scale=scale):
                dispatch = DeepEPNormalDispatchOutput(
                    hidden_states=torch.ones((2, 16), dtype=dtype),
                    hidden_states_scale=scale,
                    topk_ids=torch.zeros((2, 1), dtype=torch.int64),
                    topk_weights=torch.ones((2, 1)),
                    num_recv_tokens_per_expert=[256, 256],
                )
                with self.assertRaisesRegex(RuntimeError, error):
                    layer.run_moe_core(dispatch)

    def test_reference_masked_does_not_call_legacy_kernel(self):
        layer = self.make_reference_layer(
            w13_weight=torch.empty((2, 32, 8), dtype=torch.int8),
            w13_weight_scale=torch.ones((2, 32, 1)),
            w2_weight=torch.empty((2, 16, 8), dtype=torch.int8),
            w2_weight_scale=torch.ones((2, 16, 1)),
            w4a8_padded_intermediate_size=16,
        )

        timeline = []
        layer.dispatcher.record_combine_input_ready_event.side_effect = lambda: (
            timeline.append("ready")
        )

        def gemm(a, scale, weight, weight_scale, output, counts, expected):
            timeline.append("gemm")
            self.assertEqual(expected, 8)
            output.fill_(2)

        with (
            patch.object(
                torch.ops.sglang,
                "m_grouped_w4a8_gemm_nt_masked_hipc",
                new=gemm,
                create=True,
            ),
            patch.object(
                torch.ops.sglang, "m_grouped_w4a8_gemm_nt_masked", create=True
            ) as legacy,
            patch(
                "lightop.fuse_silu_mul_clamp_quant_ep",
                return_value=(
                    torch.ones((2, 8, 16), dtype=torch.int8),
                    torch.ones((2, 8, 1)),
                ),
            ),
        ):
            dispatch = self.make_reference_dispatch_output()
            result = layer.run_moe_core(dispatch)
            output = result.hidden_states
        legacy.assert_not_called()
        layer.quant_method.apply.assert_not_called()
        layer.dispatcher.record_combine_input_ready_event.assert_called_once_with()
        self.assertEqual(timeline, ["gemm", "gemm", "ready"])
        self.assertIsInstance(result, DeepEPLLCombineInput)
        self.assertIs(result.topk_ids, dispatch.topk_ids)
        self.assertIs(result.topk_weights, dispatch.topk_weights)
        self.assertEqual(output.shape, (2, 8, 16))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(output.min().item(), 2)

    def test_reference_low_latency_rejects_bad_inputs_before_ready_event(self):
        layer = self.make_reference_layer()
        for dispatch, error in (
            (
                self.make_reference_dispatch_output(
                    hidden_states=torch.empty((2, 8, 16), dtype=torch.bfloat16)
                ),
                "requires INT8",
            ),
            (
                self.make_reference_dispatch_output(
                    hidden_states_scale=torch.ones((2, 8), dtype=torch.float32)
                ),
                "scale shape",
            ),
        ):
            with self.subTest(error=error), self.assertRaisesRegex(RuntimeError, error):
                layer.run_moe_core(dispatch)
        layer.dispatcher.record_combine_input_ready_event.assert_not_called()

    def test_reference_constructor_uses_quant_method_marker_on_hcu_only(self):
        for hcu, marker, expected in (
            (True, True, True),
            (True, False, False),
            (False, True, False),
        ):
            with self.subTest(hcu=hcu, marker=marker):
                def init_parent(layer, **kwargs):
                    torch.nn.Module.__init__(layer)
                    layer.scheme = None
                    layer.quant_method = SimpleNamespace(hcu_w4a8_hipc=marker)
                    layer.quant_config = kwargs.get("quant_config")
                    layer.w13_weight = torch.empty((1, 1), dtype=torch.int8)
                    layer.dispatcher = Mock()

                with (
                    patch(
                        "sglang.srt.layers.moe.ep_moe.layer.FusedMoE.__init__",
                        new=init_parent,
                    ),
                    patch("sglang.srt.layers.moe.ep_moe.layer._is_hcu", hcu),
                    patch(
                        "sglang.srt.layers.moe.ep_moe.layer.get_moe_runner_backend",
                        return_value=MoeRunnerBackend.DEEP_GEMM,
                    ),
                    patch(
                        "sglang.srt.layers.moe.ep_moe.layer.get_moe_a2a_backend",
                        return_value=MoeA2ABackend.DEEPEP,
                    ),
                    patch(
                        "sglang.srt.layers.moe.ep_moe.layer.get_deepep_mode",
                        return_value=DeepEPMode.NORMAL,
                    ),
                ):
                    layer = DeepEPMoE(
                        num_experts=2,
                        top_k=2,
                        hidden_size=16,
                        intermediate_size=16,
                        layer_id=0,
                        quant_config=self.config if expected else None,
                    )
                self.assertEqual(layer.use_hcu_w4a8_reference, expected)
                self.assertFalse(hasattr(layer, "use_hcu_w4a8_hipc"))
                if expected:
                    self.assertFalse(layer.deprecate_flag)
                    self.assertTrue(layer.use_w4a8_marlin)
                    self.assertIs(layer.quant_config, self.config)
                    self.assertFalse(layer.use_fp8_w8a8)
                    self.assertFalse(layer.use_w4afp8)
                    self.assertFalse(layer.use_w8a8_marlin)
                    self.assertFalse(layer.use_bf16_marlin)
                    self.assertFalse(layer.use_w4a16_marlin)
                    self.assertFalse(layer.use_block_quant)
                    self.assertIsNone(layer.block_shape)
                    self.assertIsNone(layer.activation_scheme)
                    self.assertEqual(layer.deepep_mode, DeepEPMode.NORMAL)
                    layer.dispatcher.set_quant_config.assert_not_called()

    def test_non_reference_retains_modern_quant_method(self):
        layer = self.make_reference_layer()
        layer.use_hcu_w4a8_reference = False
        layer.deprecate_flag = True
        dispatch = self.make_reference_dispatch_output()
        expected = dispatch.hidden_states.to(torch.bfloat16) * 2
        layer.quant_method.apply.return_value = DeepEPLLCombineInput(
            expected, dispatch.topk_ids, dispatch.topk_weights
        )
        with patch.object(layer, "forward_groupgemm_w4a8_marlin_masked") as w4a8:
            result = layer.run_moe_core(dispatch)
        w4a8.assert_not_called()
        torch.testing.assert_close(result.hidden_states, expected)
        self.assertIs(result.topk_ids, dispatch.topk_ids)
        layer.dispatcher.record_combine_input_ready_event.assert_not_called()

    def test_reference_forward_dispatches_computes_combines_and_returns(self):
        for mode in ("normal", "low_latency"):
            with self.subTest(mode=mode):
                layer = self.make_reference_layer()
                x = torch.ones((3, 16), dtype=torch.bfloat16)
                topk = object()
                expected = x * 2
                dispatch = (
                    DeepEPNormalDispatchOutput(
                        x.to(torch.int8),
                        torch.ones((3, 1)),
                        torch.zeros((3, 2), dtype=torch.int64),
                        torch.ones((3, 2)),
                        [256, 256],
                    )
                    if mode == "normal"
                    else self.make_reference_dispatch_output()
                )
                expert_output = torch.full_like(
                    dispatch.hidden_states, 3, dtype=torch.bfloat16
                )
                timeline = []

                def dispatch_tokens(**kwargs):
                    self.assertIs(kwargs["hidden_states"], x)
                    self.assertIs(kwargs["topk_output"], topk)
                    timeline.append("dispatch")
                    return dispatch

                def compute(output):
                    self.assertIs(output, dispatch)
                    timeline.append("compute")
                    return expert_output

                def combine_tokens(*, combine_input):
                    self.assertIsInstance(
                        combine_input,
                        DeepEPNormalCombineInput
                        if mode == "normal"
                        else DeepEPLLCombineInput,
                    )
                    self.assertIs(combine_input.hidden_states, expert_output)
                    self.assertIs(combine_input.topk_ids, dispatch.topk_ids)
                    self.assertIs(combine_input.topk_weights, dispatch.topk_weights)
                    timeline.append("combine")
                    return expected

                layer.dispatcher.dispatch.side_effect = dispatch_tokens
                layer.dispatcher.combine.side_effect = combine_tokens
                layer.dispatcher.record_combine_input_ready_event.side_effect = (
                    lambda: timeline.append("ready")
                )
                compute_name = (
                    "forward_deepgemm_w4a8_marlin_contiguous"
                    if mode == "normal"
                    else "forward_groupgemm_w4a8_marlin_masked"
                )
                with (
                    patch.object(layer, compute_name, side_effect=compute),
                    patch.object(
                        FusedMoE,
                        "forward_impl",
                        side_effect=AssertionError("must use direct DeepEP forward"),
                    ),
                ):
                    result = layer.forward_impl(x, topk)
                self.assertIs(result, expected)
                torch.testing.assert_close(result, x * 2)
                self.assertEqual(
                    timeline,
                    ["dispatch", "compute", "combine"]
                    if mode == "normal"
                    else ["dispatch", "compute", "ready", "combine"],
                )

    def test_reference_quant_method_apply_uses_shared_core_without_recursion(self):
        layer = self.make_reference_layer()
        method = HCUW4A8Int8DeepEPMoEMethod(self.config)
        layer.quant_method = method
        dispatch = self.make_reference_dispatch_output()
        expected = dispatch.hidden_states.to(torch.bfloat16) * 2
        with patch.object(
            layer, "forward_groupgemm_w4a8_marlin_masked", return_value=expected
        ):
            result = method.apply(layer, dispatch)
        self.assertIsInstance(result, DeepEPLLCombineInput)
        torch.testing.assert_close(result.hidden_states, expected)
        self.assertIs(result.topk_ids, dispatch.topk_ids)
        layer.dispatcher.record_combine_input_ready_event.assert_called_once_with()

    def test_factory_bypasses_old_scheme_without_hipc_env(self):
        layer = self.make_layer()
        with (
            patch.dict("os.environ", {"SGLANG_HCU_W4A8_USE_HIPC": "0"}),
            patch(
                "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors._is_hcu",
                True,
            ),
            patch(
                "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors.get_moe_runner_backend",
                return_value=MoeRunnerBackend.DEEP_GEMM,
            ),
            patch(
                "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors.get_moe_a2a_backend",
                return_value=MoeA2ABackend.DEEPEP,
            ),
            patch.object(
                self.config,
                "get_moe_scheme",
                side_effect=AssertionError("old scheme must not be created"),
            ),
        ):
            method = self.config.get_quant_method(layer, "model.layers.0.mlp.experts")
        self.assertIsInstance(method, HCUW4A8Int8DeepEPMoEMethod)
        self.assertIs(method.quant_config, self.config)
        self.assertIsNone(layer.scheme)

    def test_unsupported_actorder_retains_scheme_path(self):
        self.scheme["weights"].actorder = "group"
        layer = self.make_layer()
        sentinel = object()
        with (
            patch(
                "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors._is_hcu",
                True,
            ),
            patch(
                "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors.get_moe_runner_backend",
                return_value=MoeRunnerBackend.DEEP_GEMM,
            ),
            patch(
                "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors.get_moe_a2a_backend",
                return_value=MoeA2ABackend.DEEPEP,
            ),
            patch.object(self.config, "get_moe_scheme", return_value=sentinel),
        ):
            method = self.config.get_quant_method(layer, "model.layers.0.mlp.experts")
        self.assertNotIsInstance(method, HCUW4A8Int8DeepEPMoEMethod)
        self.assertIs(layer.scheme, sentinel)

    def test_checkpoint_weight_and_scale_tp_partition(self):
        for rank in range(2):
            with self.subTest(rank=rank):
                layer = self.make_layer(rank, 2)
                layer.quant_method.create_weights(layer, 1, 32, 16, torch.float32)
                for shard in ("w1", "w3", "w2"):
                    prefix = "w2" if shard == "w2" else "w13"
                    # Real CT safetensors: [N,K/8] weight, [N,1] scale.
                    raw = torch.arange(128, dtype=torch.int32).reshape(32, 4)
                    layer._weight_loader_impl(
                        getattr(layer, prefix + "_weight_packed"),
                        raw,
                        prefix + "_weight_packed",
                        shard,
                        0,
                    )
                    scale = torch.arange(32, dtype=torch.float32).view(32, 1)
                    layer._weight_loader_impl(
                        getattr(layer, prefix + "_weight_scale"),
                        scale,
                        prefix + "_weight_scale",
                        shard,
                        0,
                    )
                    if shard == "w2":
                        expected = raw[:, rank * 2 : rank * 2 + 2].T
                        torch.testing.assert_close(layer.w2_weight_packed[0], expected)
                        torch.testing.assert_close(layer.w2_weight_scale[0], scale.T)
                    else:
                        offset = 0 if shard == "w1" else 16
                        expected = raw[rank * 16 : (rank + 1) * 16].T
                        torch.testing.assert_close(
                            layer.w13_weight_packed[0, :, offset : offset + 16],
                            expected,
                        )
                        torch.testing.assert_close(
                            layer.w13_weight_scale[0, :, offset : offset + 16],
                            scale[rank * 16 : (rank + 1) * 16].T,
                        )

    def test_factory_keeps_other_hardware_and_backends_on_original_path(self):
        for hcu, runner, a2a in (
            (False, MoeRunnerBackend.DEEP_GEMM, MoeA2ABackend.DEEPEP),
            (True, MoeRunnerBackend.TRITON, MoeA2ABackend.DEEPEP),
            (True, MoeRunnerBackend.DEEP_GEMM, MoeA2ABackend.DEEPEP_V2),
        ):
            with self.subTest(hcu=hcu, runner=runner, a2a=a2a):
                layer = self.make_layer()
                sentinel = object()
                with (
                    patch(
                        "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors._is_hcu",
                        hcu,
                    ),
                    patch(
                        "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors.get_moe_runner_backend",
                        return_value=runner,
                    ),
                    patch(
                        "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors.get_moe_a2a_backend",
                        return_value=a2a,
                    ),
                    patch.object(self.config, "get_moe_scheme", return_value=sentinel),
                ):
                    method = self.config.get_quant_method(
                        layer, "model.layers.0.mlp.experts"
                    )
                self.assertNotIsInstance(method, HCUW4A8Int8DeepEPMoEMethod)
                self.assertIs(layer.scheme, sentinel)

    def test_asymmetric_activations_do_not_select_symmetric_int8_kernel(self):
        self.scheme["input_activations"].symmetric = False
        layer = self.make_layer()
        sentinel = object()
        with (
            patch(
                "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors._is_hcu",
                True,
            ),
            patch(
                "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors.get_moe_runner_backend",
                return_value=MoeRunnerBackend.DEEP_GEMM,
            ),
            patch(
                "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors.get_moe_a2a_backend",
                return_value=MoeA2ABackend.DEEPEP,
            ),
            patch.object(self.config, "get_moe_scheme", return_value=sentinel),
        ):
            method = self.config.get_quant_method(layer, "model.layers.0.mlp.experts")
        self.assertNotIsInstance(method, HCUW4A8Int8DeepEPMoEMethod)
        self.assertIs(layer.scheme, sentinel)

    def test_conversion_scale_once_padding_and_no_duplicate_weights(self):
        layer = self.make_layer()
        method = layer.quant_method
        method.create_weights(layer, 2, 32, 24, torch.float32)
        layer.dispatcher = Mock()
        originals = {}
        for prefix, n, k in (("w13", 48, 32), ("w2", 32, 24)):
            w = (torch.arange(2 * n * k).reshape(2, n, k) % 16 - 8).to(torch.int8)
            u = w.to(torch.uint8) + 8
            packed = (
                (u[..., 0::2] | (u[..., 1::2] << 4))
                .contiguous()
                .view(torch.int32)
                .transpose(1, 2)
                .contiguous()
            )
            getattr(layer, prefix + "_weight_packed").data.copy_(packed)
            getattr(layer, prefix + "_weight_scale").data.fill_(0.5)
            originals[prefix] = ((w[..., 0::2].to(torch.uint8) & 15) << 4) | (
                w[..., 1::2].to(torch.uint8) & 15
            )
        with patch(
            "deepgemm.pack_w4a8_moe_hipc_weight", side_effect=lambda w: w, create=True
        ):
            method.process_weights_after_loading(layer)
        torch.testing.assert_close(layer.w13_weight.view(torch.uint8), originals["w13"])
        torch.testing.assert_close(
            layer.w2_weight[..., :12].view(torch.uint8), originals["w2"]
        )
        self.assertEqual(layer.w2_weight[..., 12:].count_nonzero(), 0)
        self.assertEqual(layer.w4a8_padded_intermediate_size, 128)
        for prefix in ("w13", "w2"):
            self.assertFalse(hasattr(layer, prefix + "_weight_packed"))
            scale = getattr(layer, prefix + "_weight_scale")
            torch.testing.assert_close(scale, torch.full_like(scale, 0.5 / 16))
            self.assertGreaterEqual(
                scale.untyped_storage().nbytes() - scale.numel() * 4, 2 * 1024 * 1024
            )
        self.assertFalse(any("g_idx" in name for name, _ in layer.named_parameters()))
        config = layer.dispatcher.set_quant_config.call_args.args[0]
        self.assertEqual(config["normal_expert_alignment"], 256)
        self.assertTrue(config["hcu_w4a8_int8_dispatch"])

    def test_reference_weight_layout_follows_mode_and_real_hipc_switches(self):
        cases = (
            (DeepEPMode.NORMAL, True, False, "hipc", None),
            (DeepEPMode.NORMAL, False, True, "marlin", None),
            (DeepEPMode.LOW_LATENCY, False, True, "hipc", None),
            (DeepEPMode.LOW_LATENCY, True, False, "marlin_masked", None),
            (DeepEPMode.AUTO, True, True, "hipc", None),
            (DeepEPMode.AUTO, True, False, "hipc", "marlin_masked"),
            (DeepEPMode.AUTO, False, True, "marlin", "hipc"),
            (DeepEPMode.AUTO, False, False, "marlin", "marlin_masked"),
        )
        for mode, normal_hipc, masked_hipc, layout, ll_layout in cases:
            with self.subTest(mode=mode, normal=normal_hipc, masked=masked_hipc):
                layer = self.make_layer()
                method = layer.quant_method
                method.moe_runner_config.swiglu_limit = None
                method.create_weights(layer, 2, 32, 24, torch.float32)
                layer.dispatcher = Mock()
                for prefix in ("w13", "w2"):
                    getattr(layer, prefix + "_weight_packed").data.zero_()
                    getattr(layer, prefix + "_weight_scale").data.fill_(0.5)
                seen = []

                def hipc_pack(weight):
                    # The real packer mutates input; no other layout may see it.
                    self.assertEqual(weight[..., :12].min().item(), -120)
                    seen.append("hipc")
                    weight.fill_(23)
                    return weight

                def marlin_pack(weight, use_deepep):
                    self.assertEqual(weight.min().item(), -120)
                    self.assertEqual(weight.max().item(), -120)
                    selected = "marlin_masked" if use_deepep else "marlin"
                    seen.append(selected)
                    return torch.full(
                        (2, 2, 2), 37 if use_deepep else 31, dtype=torch.int32
                    )

                with (
                    patch(
                        "sglang.srt.layers.moe.ep_moe.layer._use_w4a8_contiguous_hipc",
                        normal_hipc,
                    ),
                    patch(
                        "sglang.srt.layers.moe.ep_moe.layer._use_w4a8_masked_hipc",
                        masked_hipc,
                    ),
                    patch(
                        "sglang.srt.layers.quantization.slimquant_w4a8_marlin.get_deepep_mode",
                        return_value=mode,
                    ),
                    patch(
                        "deepgemm.pack_w4a8_moe_hipc_weight",
                        side_effect=hipc_pack,
                        create=True,
                    ),
                    patch(
                        "sglang.srt.layers.quantization.slimquant_w4a8_marlin.w4a8_weight_repack_impl",
                        side_effect=marlin_pack,
                    ),
                ):
                    method.process_weights_after_loading(layer)
                layouts = {layout} | ({ll_layout} if ll_layout else set())
                self.assertCountEqual(seen, list(layouts) * 2)
                for prefix in ("w13", "w2"):
                    weight = getattr(layer, prefix + "_weight")
                    self.assertEqual(
                        weight.dtype, torch.int8 if layout == "hipc" else torch.int32
                    )
                    expected = (
                        23 if layout == "hipc" else 37 if layout == "marlin_masked" else 31
                    )
                    self.assertTrue(torch.all(weight == expected))
                    self.assertFalse(hasattr(layer, prefix + "_weight_packed"))
                    if ll_layout:
                        ll_weight = getattr(layer, prefix + "_weight_low_latency")
                        self.assertNotEqual(weight.data_ptr(), ll_weight.data_ptr())
                        self.assertTrue(
                            torch.all(ll_weight == (23 if ll_layout == "hipc" else 37))
                        )
                    else:
                        self.assertFalse(
                            hasattr(layer, prefix + "_weight_low_latency")
                        )
                    scale = getattr(layer, prefix + "_weight_scale")
                    torch.testing.assert_close(scale, torch.full_like(scale, 0.5 / 16))
                self.assertEqual(
                    layer.w4a8_padded_intermediate_size,
                    128 if layout == "hipc" else 24,
                )
                self.assertEqual(
                    layer.w4a8_ll_padded_intermediate_size, 128 if masked_hipc else 24
                )

    def test_reference_normal_switch_off_delegates_apply_ep_without_clamp(self):
        method = HCUW4A8Int8DeepEPMoEMethod(self.config)
        method.moe_runner_config = SimpleNamespace(swiglu_limit=None)
        layer = self.make_reference_layer(
            w13_weight=object(),
            w2_weight=object(),
            w13_weight_scale=torch.ones((2, 32, 1)),
            w2_weight_scale=torch.ones((2, 16, 1)),
        )
        layer.quant_method = method
        layer.num_experts = 16
        layer.expert_map = torch.tensor([-1] * 6 + [0, 1] + [-1] * 8)
        layer.moe_runner_config = SimpleNamespace(
            activation="silu",
            swiglu_limit=None,
            num_experts=16,
            apply_router_weight_on_input=False,
            routed_scaling_factor=1.0,
        )
        x = torch.arange(32, dtype=torch.int8).view(2, 16)
        ids = torch.tensor([[0, 1], [1, -1]], dtype=torch.int64)
        dispatch = DeepEPNormalDispatchOutput(
            x, torch.ones((2, 1)), ids, torch.ones((2, 2)), [1, 2]
        )
        expected = x.to(torch.bfloat16) * 2

        def apply_ep(owner, **kwargs):
            self.assertIs(owner, method)
            self.assertIs(kwargs["x"], x)
            self.assertIs(kwargs["a1_scale"], dispatch.hidden_states_scale)
            torch.testing.assert_close(
                kwargs["topk_ids"], torch.tensor([[6, 7], [7, 0]])
            )
            return expected

        with (
            patch(
                "sglang.srt.layers.moe.ep_moe.layer._use_w4a8_contiguous_hipc", False
            ),
            patch(
                "sglang.srt.layers.moe.ep_moe.layer.get_moe_expert_parallel_rank",
                return_value=3,
            ),
            patch(
                "sglang.srt.layers.moe.ep_moe.layer.get_moe_expert_parallel_world_size",
                return_value=8,
            ),
            patch(
                "sglang.srt.layers.quantization.slimquant_w4a8_marlin."
                "SlimQuantW4A8Int8MarlinMoEMethod.apply_ep",
                new=apply_ep,
            ),
            patch(
                "deepgemm.m_grouped_w4a8_gemm_nt_contiguous_hipc", create=True
            ) as hipc,
        ):
            result = layer.run_moe_core(dispatch)
        hipc.assert_not_called()
        self.assertIsInstance(result, DeepEPNormalCombineInput)
        torch.testing.assert_close(result.hidden_states, expected)
        self.assertIs(result.topk_ids, ids)
        self.assertIs(result.topk_weights, dispatch.topk_weights)

    def test_reference_normal_switch_off_does_not_silently_drop_v41_clamp(self):
        layer = self.make_layer()
        method = layer.quant_method
        with (
            patch(
                "sglang.srt.layers.moe.ep_moe.layer._use_w4a8_contiguous_hipc", False
            ),
            patch(
                "sglang.srt.layers.quantization.slimquant_w4a8_marlin."
                "SlimQuantW4A8Int8MarlinMoEMethod.apply_ep"
            ) as legacy,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1"
            ):
                method.process_weights_after_loading(layer)
            with self.assertRaisesRegex(
                RuntimeError, "SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1"
            ):
                method.apply_ep(x=torch.ones((1, 32), dtype=torch.int8))
        legacy.assert_not_called()

    def test_reference_masked_switch_off_uses_legacy_kernel_and_ll_weights(self):
        layer = self.make_reference_layer(
            w13_weight=torch.empty((2, 32, 8), dtype=torch.int8),
            w2_weight=torch.empty((2, 16, 8), dtype=torch.int8),
            w13_weight_low_latency=torch.ones((2, 2, 2), dtype=torch.int32),
            w2_weight_low_latency=torch.ones((2, 2, 2), dtype=torch.int32),
            w13_weight_scale=torch.ones((2, 32, 1)),
            w2_weight_scale=torch.ones((2, 16, 1)),
            w4a8_padded_intermediate_size=128,
            w4a8_ll_padded_intermediate_size=16,
        )
        used_weights = []
        timeline = []

        def gemm(x, scale, weight, weight_scale, output, counts, expected_m):
            used_weights.append(weight)
            timeline.append("gemm")
            self.assertEqual(expected_m, 8)
            self.assertEqual(x.shape[-1], 16)
            output.fill_(4)

        def clamp_quant(input, limit, mask_m, expect_m):
            timeline.append("clamp")
            self.assertEqual(limit, 10.0)
            return torch.ones((2, 8, 16), dtype=torch.int8), torch.ones((2, 8, 1))

        layer.dispatcher.record_combine_input_ready_event.side_effect = lambda: (
            timeline.append("ready")
        )
        with (
            patch("sglang.srt.layers.moe.ep_moe.layer._use_w4a8_masked_hipc", False),
            patch.object(
                torch.ops.sglang,
                "m_grouped_w4a8_gemm_nt_masked",
                new=gemm,
                create=True,
            ),
            patch.object(
                torch.ops.sglang, "m_grouped_w4a8_gemm_nt_masked_hipc", create=True
            ) as hipc,
            patch("lightop.fuse_silu_mul_clamp_quant_ep", new=clamp_quant),
        ):
            dispatch = self.make_reference_dispatch_output()
            result = layer.run_moe_core(dispatch)
        hipc.assert_not_called()
        self.assertIs(used_weights[0], layer.w13_weight_low_latency)
        self.assertIs(used_weights[1], layer.w2_weight_low_latency)
        self.assertEqual(timeline, ["gemm", "clamp", "gemm", "ready"])
        self.assertIsInstance(result, DeepEPLLCombineInput)
        self.assertIs(result.topk_ids, dispatch.topk_ids)
        torch.testing.assert_close(
            result.hidden_states, torch.full((2, 8, 16), 4, dtype=torch.bfloat16)
        )

    def test_normal_dispatch_quantizes_before_capture_even_with_fp8_env(self):
        dispatcher = _DeepEPDispatcherImplNormal.__new__(_DeepEPDispatcherImplNormal)
        dispatcher.quant_config = {"hcu_w4a8_int8_dispatch": True}
        dispatcher.async_finish = False
        x = torch.ones((2, 16), dtype=torch.bfloat16)
        q = (x.to(torch.int8), torch.ones((2, 1)))
        topk = SimpleNamespace(
            topk_weights=torch.ones((2, 1)),
            topk_ids=torch.zeros((2, 1), dtype=torch.int32),
        )
        with (
            patch("sglang.srt.layers.moe.token_dispatcher.deepep._is_hcu", True),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.deepep._use_fp8_w8a8_moe", True
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.deepep.per_token_quant_int8",
                return_value=q,
            ),
        ):
            output = dispatcher.dispatch_a(x, topk)
        self.assertIs(output[0], q)
        self.assertEqual(output[1].dtype, torch.int64)

    def test_reference_normal_zero_counts_returns_bf16(self):
        layer = self.make_reference_layer(w13_weight_scale=torch.ones((2, 32, 1)))
        dispatch = DeepEPNormalDispatchOutput(
            torch.ones((3, 16), dtype=torch.int8),
            torch.ones((3, 1)),
            torch.zeros((3, 1), dtype=torch.int64),
            torch.ones((3, 1)),
            [0, 0],
        )
        result = layer.run_moe_core(dispatch)
        self.assertEqual(result.hidden_states.dtype, torch.bfloat16)
        self.assertEqual(result.hidden_states.count_nonzero(), 0)
        layer.dispatcher.record_combine_input_ready_event.assert_not_called()

    def test_low_latency_selects_int8_quant_type_independent_of_fp8_env(self):
        dispatcher = _DeepEPDispatcherImplLowLatency.__new__(
            _DeepEPDispatcherImplLowLatency
        )
        dispatcher.quant_config = {"hcu_w4a8_int8_dispatch": True}
        dispatcher.return_recv_hook = True
        dispatcher.num_max_dispatch_tokens_per_rank = 256
        dispatcher.num_experts = 2
        dispatcher.use_fp8 = True
        buffer = Mock()
        masked_m = torch.tensor([1, 1], dtype=torch.int32)
        x = torch.ones((1, 16), dtype=torch.bfloat16)
        ids = torch.zeros((1, 1), dtype=torch.int64)
        weights = torch.ones((1, 1))
        packed = (torch.ones((2, 8, 16), dtype=torch.int8), torch.ones((2, 8, 1)))
        buffer.low_latency_dispatch.return_value = (
            packed,
            masked_m,
            object(),
            object(),
            Mock(),
        )
        dispatcher._get_buffer = Mock(return_value=buffer)
        dispatcher._get_npu_mxfp_quantization_kwargs = Mock(return_value={})
        with (
            patch("sglang.srt.layers.moe.token_dispatcher.deepep._is_hcu", True),
            patch("sglang.srt.layers.moe.token_dispatcher.deepep.use_groupgemm", True),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.deepep._use_fp8_w8a8_moe", True
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.deepep._deepep_precompile_tp_barrier"
            ),
        ):
            result = dispatcher._dispatch_core(x, ids, weights)
        self.assertIs(result[0], packed)
        self.assertIs(result[1], masked_m)
        self.assertEqual(buffer.low_latency_dispatch.call_args.kwargs["quant_type"], 1)


if __name__ == "__main__":
    unittest.main()
