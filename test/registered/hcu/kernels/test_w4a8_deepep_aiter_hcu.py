# Copyright (c) 2026 gencheng liu
# SPDX-License-Identifier: Apache-2.0

"""HCU DeepEP W4A8 MMAC checks against DeepGEMM."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.moe.w4a8_deepep_aiter import (
    w4a8_mmac_contiguous_out,
    w4a8_mmac_masked_out,
)
from sglang.srt.layers.quantization.slimquant_w4a8_marlin import (
    repack_and_shuffle_w4a8,
)
from sglang.srt.layers.quantization.w4a8_utils import (
    weight4bit_nt_kpack2_marlin2_qqq_from_packed_mem_efficient,
)
from sglang.test.ci.ci_register import register_hcu_ci

register_hcu_ci(
    est_time=600,
    suite="nightly-hcu-core-functional",
    nightly=True,
)


def _pack_int4(weight: torch.Tensor) -> torch.Tensor:
    unsigned = weight.to(torch.int16) & 0xF
    packed = (unsigned[..., 0::2] << 4) | unsigned[..., 1::2]
    return packed.to(torch.uint8).view(torch.int8)


def _make_weights(experts: int, n: int, k: int):
    logical = torch.randint(-8, 8, (experts, n, k), dtype=torch.int8, device="cuda")
    checkpoint = _pack_int4(logical)
    aiter_layout = repack_and_shuffle_w4a8(checkpoint.clone(), experts)
    deepgemm_layout = weight4bit_nt_kpack2_marlin2_qqq_from_packed_mem_efficient(
        checkpoint
    )
    scales = torch.rand((experts, n, 1), dtype=torch.float32, device="cuda") / 128
    return aiter_layout, deepgemm_layout, scales


class TestHCUW4A8DeepEPAiter(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260905)

    def test_contiguous_matches_masked_deepgemm(self):
        from deepgemm import m_grouped_w4a8_gemm_nt_masked

        experts, rows, n, k = 2, 256, 512, 2048
        total_rows = experts * rows
        weight, reference_weight, weight_scale = _make_weights(experts, n, k)
        activation = torch.randint(
            -127, 128, (total_rows, k), dtype=torch.int8, device="cuda"
        )
        activation_scale = (
            torch.rand((total_rows, 1), dtype=torch.float32, device="cuda") / 127
        )
        m_indices = torch.arange(experts, dtype=torch.int32, device="cuda")
        m_indices = m_indices.repeat_interleave(rows)
        workspace = torch.empty(
            total_rows + total_rows // 32 + 1,
            dtype=torch.int32,
            device="cuda",
        )
        actual = torch.empty((total_rows, n), dtype=torch.bfloat16, device="cuda")
        w4a8_mmac_contiguous_out(
            activation,
            activation_scale,
            weight,
            weight_scale,
            m_indices,
            workspace,
            actual,
        )

        reference = torch.empty((experts, rows, n), dtype=torch.bfloat16, device="cuda")
        masked_m = torch.full((experts,), rows, dtype=torch.int32, device="cuda")
        m_grouped_w4a8_gemm_nt_masked(
            (
                activation.view(experts, rows, k),
                activation_scale.view(experts, rows, 1),
            ),
            (reference_weight, weight_scale),
            reference,
            masked_m,
            rows,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(
            actual, reference.view(total_rows, n), rtol=1e-2, atol=0.25
        )

    def test_masked_matches_deepgemm_and_replays_graph(self):
        from deepgemm import m_grouped_w4a8_gemm_nt_masked

        experts, rows, n, k = 4, 32, 512, 2048
        counts = [0, 1, 17, 31]
        weight, reference_weight, weight_scale = _make_weights(experts, n, k)
        activation = torch.randint(
            -127,
            128,
            (experts, rows, k),
            dtype=torch.int8,
            device="cuda",
        )
        activation_scale = (
            torch.rand((experts, rows, 1), dtype=torch.float32, device="cuda") / 127
        )
        masked_m = torch.tensor(counts, dtype=torch.int32, device="cuda")
        metadata_rows = experts * rows
        workspace = torch.empty(
            metadata_rows + metadata_rows // 16 + 1,
            dtype=torch.int32,
            device="cuda",
        )
        actual = torch.empty((experts, rows, n), dtype=torch.bfloat16, device="cuda")

        def run_kernel():
            w4a8_mmac_masked_out(
                activation,
                activation_scale,
                weight,
                weight_scale,
                masked_m,
                workspace,
                actual,
                metadata_rows,
            )

        run_kernel()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_kernel()
        graph.replay()

        reference = torch.empty_like(actual)
        m_grouped_w4a8_gemm_nt_masked(
            (activation, activation_scale),
            (reference_weight, weight_scale),
            reference,
            masked_m,
            rows,
        )
        torch.cuda.synchronize()
        actual_valid = torch.cat(
            [actual[index, :count] for index, count in enumerate(counts) if count]
        )
        reference_valid = torch.cat(
            [reference[index, :count] for index, count in enumerate(counts) if count]
        )
        torch.testing.assert_close(actual_valid, reference_valid, rtol=1e-2, atol=0.25)

    def test_deepep_routes_through_quant_method_capabilities(self):
        from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
        from sglang.srt.layers.moe.token_dispatcher.deepep import (
            DeepEPLLDispatchOutput,
            DeepEPNormalDispatchOutput,
        )

        layer = DeepEPMoE.__new__(DeepEPMoE)
        torch.nn.Module.__init__(layer)
        layer.deprecate_flag = False
        layer.use_fp8_w8a8 = False
        layer.use_w4afp8 = False
        layer.use_w4a8_marlin = True
        layer.quant_config = object()

        normal_result = torch.tensor([1.0], device="cuda")
        low_latency_result = torch.tensor([2.0], device="cuda")
        calls = []

        def apply_normal(*, layer, dispatch_output):
            calls.append(("normal", layer, dispatch_output))
            return normal_result

        def apply_low_latency(*, layer, dispatch_output):
            calls.append(("low_latency", layer, dispatch_output))
            return low_latency_result

        layer.quant_method = SimpleNamespace(
            apply_deepep_normal=apply_normal,
            apply_deepep_low_latency=apply_low_latency,
        )
        topk_ids = torch.zeros((1, 1), dtype=torch.int64, device="cuda")
        topk_weights = torch.ones((1, 1), dtype=torch.float32, device="cuda")
        normal_dispatch = DeepEPNormalDispatchOutput(
            torch.empty((1, 1), device="cuda"),
            None,
            topk_ids,
            topk_weights,
            [0],
        )
        low_latency_dispatch = DeepEPLLDispatchOutput(
            torch.empty((1, 1, 1), device="cuda"),
            None,
            topk_ids,
            topk_weights,
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            1,
        )

        self.assertIs(layer.run_moe_core(normal_dispatch).hidden_states, normal_result)
        self.assertIs(
            layer.run_moe_core(low_latency_dispatch).hidden_states,
            low_latency_result,
        )
        self.assertEqual([call[0] for call in calls], ["normal", "low_latency"])
        self.assertTrue(all(call[1] is layer for call in calls))


if __name__ == "__main__":
    unittest.main()
