"""HCU CompressedTensors INT8 MoE runner/scale regression tests."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy

from sglang.srt.layers.moe import MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a8_int8_moe as int8_moe,
)
from sglang.test.test_utils import CustomTestCase


def _make_weight_quant():
    return SimpleNamespace(
        strategy=QuantizationStrategy.CHANNEL,
        dynamic=False,
    )


def _make_input_quant():
    return SimpleNamespace(
        strategy=QuantizationStrategy.TOKEN,
        dynamic=True,
    )


class TestCompressedTensorsW8A8Int8MoEHCU(CustomTestCase):
    @staticmethod
    def _make_layer():
        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(
            torch.zeros(512, 320, 2560, dtype=torch.int8), requires_grad=False
        )
        layer.w2_weight = torch.nn.Parameter(
            torch.zeros(512, 2560, 160, dtype=torch.int8), requires_grad=False
        )
        layer.w13_weight_scale = torch.nn.Parameter(
            torch.ones(512, 320, 1, dtype=torch.bfloat16), requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            torch.ones(512, 2560, 1, dtype=torch.bfloat16), requires_grad=False
        )
        layer.w13_input_scale = None
        layer.w2_input_scale = None
        return layer

    def test_normalize_weight_scales_casts_bf16_to_fp32(self):
        layer = self._make_layer()
        int8_moe.CompressedTensorsW8A8Int8MoE._normalize_weight_scales(layer)
        self.assertEqual(layer.w13_weight_scale.dtype, torch.float32)
        self.assertEqual(layer.w2_weight_scale.dtype, torch.float32)

    def test_create_moe_runner_auto_selects_aiter_on_hcu(self):
        scheme = int8_moe.CompressedTensorsW8A8Int8MoE(
            _make_weight_quant(), _make_input_quant()
        )
        layer = self._make_layer()
        config = MoeRunnerConfig(num_experts=512, num_local_experts=512)

        with mock.patch.object(int8_moe, "_is_hcu", True), mock.patch(
            "sglang.srt.layers.quantization.compressed_tensors.schemes."
            "compressed_tensors_w8a8_int8_moe.will_use_aiter_moe",
            return_value=True,
        ), mock.patch(
            "sglang.srt.layers.quantization.compressed_tensors.schemes."
            "compressed_tensors_w8a8_int8_moe.get_moe_runner_backend",
            return_value=MoeRunnerBackend.AUTO,
        ), mock.patch(
            "sglang.srt.layers.quantization.compressed_tensors.schemes."
            "compressed_tensors_w8a8_int8_moe.get_moe_a2a_backend",
            return_value=SimpleNamespace(supports_aiter=lambda: True),
        ), mock.patch(
            "sglang.srt.layers.quantization.compressed_tensors.schemes."
            "compressed_tensors_w8a8_int8_moe.MoeRunner"
        ) as moe_runner_cls:
            scheme.create_moe_runner(layer, config)
            moe_runner_cls.assert_called_once()
            self.assertEqual(
                moe_runner_cls.call_args.args[0], MoeRunnerBackend.AITER
            )

    def test_process_weights_after_loading_skips_shuffle_on_hcu_aiter(self):
        scheme = int8_moe.CompressedTensorsW8A8Int8MoE(
            _make_weight_quant(), _make_input_quant()
        )
        layer = self._make_layer()
        original_w13 = layer.w13_weight.data.clone()
        original_w2 = layer.w2_weight.data.clone()

        with mock.patch.object(int8_moe, "_is_hcu", True), mock.patch(
            "sglang.srt.layers.quantization.compressed_tensors.schemes."
            "compressed_tensors_w8a8_int8_moe.will_use_aiter_moe",
            return_value=True,
        ), mock.patch(
            "sglang.srt.layers.moe.moe_runner.aiter."
            "process_weights_after_loading_aiter_w8a8_int8"
        ) as init_aiter:
            scheme.process_weights_after_loading(layer)
            init_aiter.assert_called_once_with(layer)
            self.assertTrue(torch.equal(layer.w13_weight.data, original_w13))
            self.assertTrue(torch.equal(layer.w2_weight.data, original_w2))
            self.assertEqual(layer.w13_weight_scale.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
