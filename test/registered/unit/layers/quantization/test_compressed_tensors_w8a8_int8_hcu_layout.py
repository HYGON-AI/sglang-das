"""HCU W8A8 method-3 weight-layout regression tests."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from compressed_tensors.quantization import QuantizationStrategy
from torch.nn import Parameter

from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a8_int8 as w8a8_int8,
)
from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.test_utils import CustomTestCase

register_hcu_ci(est_time=5, suite="stage-b-test-1-hcu-small")


class TestCompressedTensorsW8A8Int8HCULayout(CustomTestCase):
    @staticmethod
    def _make_scheme():
        with mock.patch.dict(os.environ, {"W8A8_SUPPORT_METHODS": "3"}):
            return w8a8_int8.CompressedTensorsW8A8Int8(
                strategy=QuantizationStrategy.CHANNEL,
                is_static_input_scheme=False,
                input_symmetric=False,
            )

    @staticmethod
    def _make_layer(weight: torch.Tensor):
        layer = torch.nn.Module()
        layer.weight = Parameter(weight.clone(), requires_grad=False)
        layer.weight_scale = Parameter(
            torch.ones((weight.shape[0], 1), dtype=torch.float32),
            requires_grad=False,
        )
        layer.logical_widths = [weight.shape[0]]
        return layer

    def test_only_gfx928_selects_kme(self):
        cuda_device = SimpleNamespace(type="cuda")
        cases = {
            "gfx928:sramecc+:xnack-": True,
            "gfx936:sramecc+:xnack-": False,
            "gfx938:sramecc+:xnack-": False,
            "sm_90": False,
        }

        for arch, expected in cases.items():
            with self.subTest(arch=arch), mock.patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(gcnArchName=arch),
            ):
                self.assertEqual(w8a8_int8._use_kme_hipblaslt(cuda_device), expected)

        with mock.patch.object(torch.cuda, "get_device_properties") as get_props:
            self.assertFalse(w8a8_int8._use_kme_hipblaslt(SimpleNamespace(type="cpu")))
            get_props.assert_not_called()

    def test_method3_layout_and_azp_reduction_match(self):
        weight = torch.tensor(
            [[1, 2, 3, 4], [5, -1, 0, 2], [-3, 7, 1, -2]],
            dtype=torch.int8,
        )
        expected_azp = weight.sum(dim=1, keepdim=False, dtype=torch.int32).unsqueeze(0)

        for use_kme in (True, False):
            with self.subTest(use_kme=use_kme):
                layer = self._make_layer(weight)
                scheme = self._make_scheme()

                with mock.patch.object(
                    w8a8_int8, "_use_kme_hipblaslt", return_value=use_kme
                ), mock.patch.object(w8a8_int8.W8A8_TRITONJSON, "gen_model_json"):
                    scheme.process_weights_after_loading(layer)

                if use_kme:
                    self.assertEqual(layer.weight.shape, (4, 3))
                    self.assertEqual(layer.weight.stride(), (1, 4))
                    self.assertTrue(torch.equal(layer.weight, weight.t()))
                else:
                    self.assertEqual(layer.weight.shape, (3, 4))
                    self.assertTrue(layer.weight.is_contiguous())
                    self.assertTrue(torch.equal(layer.weight, weight))

                self.assertEqual(layer.azp_adj.shape, (1, 3))
                self.assertTrue(torch.equal(layer.azp_adj, expected_azp))


if __name__ == "__main__":
    unittest.main()
