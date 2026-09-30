"""CPU regressions for HCU DSPARK unquantized DeepEP low latency."""

import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe.ep_moe import layer as ep_moe_layer
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
from sglang.srt.layers.moe.token_dispatcher import deepep
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPLLDispatchOutput,
    _DeepEPDispatcherImplLowLatency,
)
from sglang.srt.layers.moe.utils import DispatcherOutputDtype
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestHCUDSPARKUnquantizedDeepEP(CustomTestCase):
    def make_unquantized_method(self):
        return object.__new__(UnquantizedFusedMoEMethod)

    def test_global_quant_config_does_not_quantize_unquantized_draft_layer(self):
        dispatcher = Mock()

        with patch.object(ep_moe_layer, "_is_hcu", True):
            ep_moe_layer._configure_deepep_dispatcher_quantization(
                dispatcher,
                quant_config=object(),
                quant_method=self.make_unquantized_method(),
            )

        dispatcher.set_quant_config.assert_called_once_with(
            {"dispatcher_output_dtype": "bf16"}
        )

    def test_low_latency_uses_layer_quant_method_for_unquantized_draft(self):
        moe = DeepEPMoE.__new__(DeepEPMoE)
        torch.nn.Module.__init__(moe)
        moe.deprecate_flag = False
        moe.quant_config = object()
        moe.quant_method = self.make_unquantized_method()
        output = torch.empty((2, 4, 16), dtype=torch.bfloat16)
        moe.forward_unquantized_deepep_ll = Mock(return_value=output)

        topk_ids = torch.zeros((3, 2), dtype=torch.int64)
        topk_weights = torch.ones((3, 2), dtype=torch.float32)
        dispatch_output = DeepEPLLDispatchOutput(
            hidden_states=torch.empty((2, 4, 16), dtype=torch.bfloat16),
            hidden_states_scale=None,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            masked_m=torch.tensor([3, 2], dtype=torch.int32),
            expected_m=8,
        )

        with patch.object(ep_moe_layer, "_is_hcu", True):
            combine_input = moe.run_moe_core(dispatch_output)

        self.assertIsInstance(combine_input, DeepEPLLCombineInput)
        self.assertIs(combine_input.hidden_states, output)
        self.assertIs(combine_input.topk_ids, topk_ids)
        self.assertIs(combine_input.topk_weights, topk_weights)
        moe.forward_unquantized_deepep_ll.assert_called_once_with(dispatch_output)

    def test_bf16_low_latency_dispatch_uses_unquantized_wire_format(self):
        impl = _DeepEPDispatcherImplLowLatency.__new__(
            _DeepEPDispatcherImplLowLatency
        )
        impl.quant_config = {}
        impl.deepep_output_dtype = DispatcherOutputDtype.BF16
        impl.return_recv_hook = False
        impl.num_max_dispatch_tokens_per_rank = 64
        impl.num_experts = 128
        impl.use_fp8 = False

        packed_hidden = torch.empty((2, 4, 16), dtype=torch.bfloat16)
        packed_count = torch.zeros(2, dtype=torch.int32)
        handle = object()
        event = Mock()
        hook = Mock()
        buffer = Mock()
        buffer.low_latency_dispatch.return_value = (
            packed_hidden,
            packed_count,
            handle,
            event,
            hook,
        )
        impl._get_buffer = Mock(return_value=buffer)
        impl._get_npu_mxfp_quantization_kwargs = Mock(return_value={})

        hidden_states = torch.empty((1, 16), dtype=torch.bfloat16)
        topk_ids = torch.zeros((1, 2), dtype=torch.int64)
        topk_weights = torch.ones((1, 2), dtype=torch.float32)

        with (
            patch.object(deepep, "use_groupgemm", True),
            patch.object(deepep, "_use_fp8_w8a8_moe", False),
            patch.object(deepep, "_use_marlin_w16a16_moe", False),
            patch.object(deepep, "_use_marlin_w4a16_moe", False),
            patch.object(deepep, "_deepep_precompile_tp_barrier"),
        ):
            result = impl._dispatch_core(
                hidden_states,
                topk_ids,
                topk_weights,
            )

        call = buffer.low_latency_dispatch.call_args
        self.assertEqual(call.kwargs["quant_type"], 0)
        self.assertFalse(call.kwargs["fp8_round_scale"])
        self.assertTrue(call.kwargs["async_finish"])
        self.assertFalse(call.kwargs["return_recv_hook"])
        self.assertIs(result[0], packed_hidden)
        self.assertIs(result[1], packed_count)
        self.assertIs(result[2], event)
        self.assertIs(result[3], hook)
        self.assertIs(impl.handle, handle)


if __name__ == "__main__":
    unittest.main()
