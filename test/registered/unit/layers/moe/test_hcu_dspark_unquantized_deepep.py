"""CPU regressions for HCU DSPARK unquantized DeepEP low latency."""

import unittest
from types import SimpleNamespace
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


class _FakeLowLatencyBuffer:
    instances = []

    def __init__(self, group, nvl_bytes, rdma_bytes, **kwargs):
        self.group_size = group.size()
        self.num_nvl_bytes = nvl_bytes
        self.num_rdma_bytes = rdma_bytes
        self.low_latency_mode = kwargs["low_latency_mode"]
        self.num_qps_per_rank = kwargs["num_qps_per_rank"]
        self.cleans = []
        self.instances.append(self)

    @staticmethod
    def get_low_latency_rdma_size_hint(tokens, hidden, ranks, experts, num_topk=8):
        return tokens * hidden * experts * 4 + num_topk * 1024

    @staticmethod
    def get_dispatch_config(_ranks):
        return SimpleNamespace(
            get_nvl_buffer_size_hint=lambda *_args: 4096,
            get_rdma_buffer_size_hint=lambda *_args: 8192,
        )

    get_combine_config = get_dispatch_config

    def clean_low_latency_buffer(self, *layout):
        self.cleans.append(layout)


class TestHCUDSPARKLowLatencyBufferCapacity(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.resources = SimpleNamespace(buffers={})
        self.group = SimpleNamespace(size=lambda: 8)
        _FakeLowLatencyBuffer.instances = []
        config = SimpleNamespace(
            num_sms=32, normal_dispatch_config=None, normal_combine_config=None
        )
        for item in (
            patch.object(deepep, "get_resources", return_value=self.resources),
            patch.object(deepep, "_is_hcu", True),
            patch.object(deepep, "_is_npu", False),
            patch.object(deepep, "use_deepep", True),
            patch.object(deepep, "Buffer", _FakeLowLatencyBuffer),
            patch.object(deepep.DeepEPConfig, "get_instance", return_value=config),
            patch.object(deepep, "_set_nvshmem_qp_depth"),
            patch.object(deepep, "_is_mnnvl_fabric_supported", return_value=False),
            patch.object(deepep, "get_cuda_version", return_value=(12, 0)),
            patch.object(deepep, "is_tbo_enabled", return_value=True),
            patch.object(
                deepep.envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK,
                "get",
                return_value=128,
            ),
            patch.object(
                deepep._DeepEPDispatcherImplBase, "set_deepep_dispatcher_dtype"
            ),
            patch("torch.get_device_module", return_value=Mock()),
            patch(
                "torch.cuda.get_device_properties",
                return_value=SimpleNamespace(multi_processor_count=80),
            ),
        ):
            item.start()
            self.addCleanup(item.stop)

    def _dispatcher(self, experts, topk=6, mode=None):
        return _DeepEPDispatcherImplLowLatency(
            group=self.group,
            router_topk=topk,
            permute_fusion=False,
            num_experts=experts,
            num_local_experts=experts // 8,
            hidden_size=5120,
            params_dtype=torch.bfloat16,
            deepep_mode=mode or deepep.DeepEPMode.LOW_LATENCY,
            return_recv_hook=False,
        )

    def test_draft_first_use_reserves_target_capacity_and_qps(self):
        target = self._dispatcher(384)
        draft = self._dispatcher(128, topk=4)
        self.assertEqual(len(_FakeLowLatencyBuffer.instances), 0)
        buffer = draft._get_buffer()
        self.assertIs(target._get_buffer(), buffer)
        self.assertIs(draft._get_buffer(), buffer)
        self.assertEqual(len(_FakeLowLatencyBuffer.instances), 1)
        self.assertGreaterEqual(
            buffer.num_rdma_bytes,
            _FakeLowLatencyBuffer.get_low_latency_rdma_size_hint(128, 5120, 8, 384),
        )
        self.assertEqual(buffer.num_qps_per_rank, 48)
        self.assertEqual(
            buffer.cleans, [(128, 5120, 384), (128, 5120, 128)]
        )

    def test_target_first_use_reuses_same_buffer_for_draft(self):
        draft = self._dispatcher(128, topk=4)
        target = self._dispatcher(384)
        buffer = target._get_buffer()
        self.assertIs(draft._get_buffer(), buffer)
        self.assertEqual(len(_FakeLowLatencyBuffer.instances), 1)
        self.assertEqual(buffer.num_qps_per_rank, 48)
        self.assertEqual(buffer.cleans, [(128, 5120, 128)])

    def test_reservation_covers_actual_topk_above_native_default(self):
        self._dispatcher(384, topk=9)
        draft = self._dispatcher(128, topk=4)
        buffer = draft._get_buffer()
        self.assertGreaterEqual(
            buffer.num_rdma_bytes,
            _FakeLowLatencyBuffer.get_low_latency_rdma_size_hint(
                128, 5120, 8, 384, num_topk=9
            ),
        )

    def test_auto_mode_keeps_normal_allocation_and_largest_ll_layout(self):
        target = self._dispatcher(384, mode=deepep.DeepEPMode.AUTO)
        draft = self._dispatcher(128, mode=deepep.DeepEPMode.AUTO)
        buffer = draft._get_buffer()
        self.assertIs(target._get_buffer(), buffer)
        self.assertEqual(buffer.num_nvl_bytes, 4096)
        self.assertEqual(buffer.num_qps_per_rank, 48)
        self.assertGreaterEqual(
            buffer.num_rdma_bytes,
            _FakeLowLatencyBuffer.get_low_latency_rdma_size_hint(128, 5120, 8, 384),
        )

    def test_non_hcu_keeps_existing_buffer_compatibility_checks(self):
        with patch.object(deepep, "_is_hcu", False):
            target = self._dispatcher(384)
            draft = self._dispatcher(128, topk=4)
            buffer = draft._get_buffer()
            self.assertEqual(buffer.num_qps_per_rank, 16)
            self.assertEqual(
                buffer.num_rdma_bytes,
                _FakeLowLatencyBuffer.get_low_latency_rdma_size_hint(
                    128, 5120, 8, 128, num_topk=4
                ),
            )
            with self.assertRaisesRegex(RuntimeError, "expert count"):
                target._get_buffer()
            self.assertEqual(len(_FakeLowLatencyBuffer.instances), 1)

    def test_normal_mode_allocation_is_unchanged(self):
        buffer = deepep.DeepEPBuffer.get_deepep_buffer(
            self.group, 5120, 2, deepep.DeepEPMode.NORMAL
        )
        self.assertEqual(buffer.num_nvl_bytes, 4096)
        self.assertEqual(buffer.num_rdma_bytes, 8192)
        self.assertEqual(buffer.num_qps_per_rank, 32)
        self.assertFalse(buffer.low_latency_mode)

    def test_unregistered_larger_layout_still_errors_without_reallocation(self):
        draft = self._dispatcher(128, topk=4)
        buffer = draft._get_buffer()
        with self.assertRaisesRegex(RuntimeError, "shared buffer has"):
            deepep.DeepEPBuffer.get_deepep_buffer(
                self.group, 5120, 2, deepep.DeepEPMode.LOW_LATENCY, 128, 384, 6
            )
        self.assertEqual(len(_FakeLowLatencyBuffer.instances), 1)
        self.assertIs(deepep.DeepEPBuffer._state().buffer, buffer)
        self.assertEqual(buffer.cleans, [])

    def test_new_resource_state_does_not_retain_old_capacity(self):
        self._dispatcher(384)
        previous_state = deepep.DeepEPBuffer._state()
        self.resources.buffers.clear()
        draft = self._dispatcher(128, topk=4)
        buffer = draft._get_buffer()
        self.assertIsNot(deepep.DeepEPBuffer._state(), previous_state)
        self.assertEqual(buffer.num_qps_per_rank, 16)
        self.assertEqual(
            buffer.num_rdma_bytes,
            _FakeLowLatencyBuffer.get_low_latency_rdma_size_hint(128, 5120, 8, 128),
        )


if __name__ == "__main__":
    unittest.main()
