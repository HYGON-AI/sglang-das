# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""VL routing must dispatch the physical expert selected by EPLB."""

import unittest
from contextlib import ExitStack
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.moe import topk
from sglang.srt.multimodal.dsv41 import vl_routing

register_cpu_ci(est_time=3, stage="base-a", runner_config="cpu")


class TestVisionTopKEPLB(CustomTestCase):
    def _run_case(self, *, shared=0, padded=None, nextn=False, mapping=True,
                  hcu_gate=False, cuda_gate=False, idle=False, empty=False):
        experts = 384 if hcu_gate else 4
        rows = 0 if empty else 3
        logits = torch.linspace(0.1, 0.4, experts).repeat(rows, 1)
        text_bias = torch.zeros(experts)
        text_bias[:2] = torch.tensor([4.0, 3.0])
        image_bias = torch.zeros(experts)
        image_bias[2:4] = torch.tensor([4.0, 3.0])
        input_ids = None if idle else torch.tensor([1, 99, 2][:rows])
        config = topk.TopKConfig(
            top_k=2 + shared,
            scoring_func="sqrtsoftplus",
            sqrtsoftplus_log1p=True,
            num_fused_shared_experts=shared,
            routed_scaling_factor=1.5,
            apply_routed_scaling_factor_on_output=True,
            fused_shared_experts_scaling_factor=0.125 if shared else None,
        )
        moe = SimpleNamespace(
            topk=SimpleNamespace(topk_config=config), layer_id=17, is_nextn=nextn,
            gate=SimpleNamespace(e_score_correction_bias=text_bias,
                                 e_score_correction_bias_vl=image_bias),
            config=SimpleNamespace(image_token_id=99),
        )
        # Include redundant physical experts: the shared base is not E_logical.
        logical_to_physical = torch.arange(experts).roll(2) + 1
        info = ExpertLocationDispatchInfo(
            ep_dispatch_algorithm="static",
            partial_logical_to_rank_dispatch_physical_map=logical_to_physical,
            partial_logical_to_all_physical_map=logical_to_physical[:, None],
            partial_logical_to_all_physical_map_num_valid=torch.ones(
                experts, dtype=torch.int32),
            num_physical_experts=experts + 2,
        ) if mapping else None
        active_info = None if nextn else info
        count = None if padded is None else torch.tensor(padded, dtype=torch.int32)
        bias = text_bias if input_ids is None else torch.where(
            (input_ids == 99)[:, None], image_bias, text_bias)
        scores = F.softplus(logits).sqrt()
        logical_ids = (scores + bias).topk(2, dim=-1).indices
        routed_weights = scores.gather(1, logical_ids)
        routed_sum = routed_weights.sum(-1, keepdim=True)
        expected_weights = routed_weights / (routed_sum + topk._RENORMALIZE_SUM_EPSILON) * 1.5
        expected_ids = (
            logical_to_physical[logical_ids] if active_info is not None else logical_ids
        )
        if shared:
            expected_ids = torch.cat((expected_ids, torch.full(
                (rows, shared), active_info.num_physical_experts
                if active_info is not None else experts, dtype=torch.int64)), dim=1)
            expected_weights = torch.cat((expected_weights,
                (routed_sum / (routed_sum + topk._RENORMALIZE_SUM_EPSILON))
                .expand(rows, shared) * 0.125), dim=1)
        if padded is not None:
            expected_ids[padded:] = -1
            expected_weights[padded:] = 0
        recorder = Mock()
        native_gate = Mock()

        def gate(logits_arg, bias_arg, k, n_shared, renorm, scale, apply_scale):
            # Emulate only the kernel boundary; routing postprocessing is real.
            torch.testing.assert_close(bias_arg, bias)
            self.assertEqual((k, n_shared, renorm, scale, apply_scale),
                             (config.top_k, shared, True, 1.5, True))
            w = routed_weights / (routed_sum + topk._RENORMALIZE_SUM_EPSILON) * 1.5
            ids = logical_ids.int()
            if shared:
                ids = torch.cat((ids, torch.full((rows, shared), experts, dtype=torch.int32)), dim=1)
                w = torch.cat((w, (routed_sum / (routed_sum + topk._RENORMALIZE_SUM_EPSILON)).expand(rows, shared)), dim=1)
            return w.clone(), ids.clone()

        native_gate.side_effect = gate
        cuda_kernel = Mock()

        def cuda_route(logits_arg, bias_arg, **kwargs):
            self.assertIs(bias_arg, text_bias)
            packed = kwargs["packed_out"]
            if mapping and not nextn:
                self.assertIsNone(packed)
            else:
                self.assertIsNotNone(packed)
                packed.fill_(123)
            return gate(logits_arg, bias, kwargs["topk"],
                        kwargs["num_fused_shared_experts"], kwargs["renormalize"],
                        kwargs["routed_scaling_factor"],
                        kwargs["apply_routed_scaling_factor_on_output"])

        cuda_kernel.side_effect = cuda_route
        with ExitStack() as stack:
            for name, value in {"_is_hip": True, "_is_hcu": True,
                                "_is_cuda": False, "_use_aiter": False}.items():
                stack.enter_context(patch.object(topk, name, value))
            stack.enter_context(patch.object(topk, "_eplb_remap_enabled", return_value=True))
            stack.enter_context(patch.object(topk, "_can_fuse_padded_region", return_value=False))
            stack.enter_context(patch.object(topk, "_biased_grouped_topk_postprocess",
                                             side_effect=topk._topk_ids_postprocess_torch.__wrapped__))
            stack.enter_context(patch.object(topk, "get_moe_a2a_backend",
                return_value=SimpleNamespace(is_deepep=lambda: True)))
            stack.enter_context(patch.object(topk, "get_moe_runner_backend",
                return_value=SimpleNamespace(is_deep_gemm=lambda: True)))
            stack.enter_context(patch.object(topk.envs.SGLANG_AITER_MEGA_EPLB_PREFILL_ONLY,
                                             "get", return_value=False))
            capture = stack.enter_context(patch.object(topk, "capture_routed_experts_if_allowed"))
            stack.enter_context(patch.object(topk, "has_per_rank_fused_shared_slots", return_value=False))
            stack.enter_context(patch.object(vl_routing, "has_per_rank_fused_shared_slots", return_value=False))
            stack.enter_context(patch.object(vl_routing, "is_cuda", return_value=cuda_gate))
            stack.enter_context(patch.object(vl_routing, "is_hcu", return_value=hcu_gate))
            stack.enter_context(patch.object(vl_routing, "_use_lightop_sqrtsoftplus_v41_gate", True))
            init = stack.enter_context(patch.object(ExpertLocationDispatchInfo, "init_new", return_value=info))
            stack.enter_context(patch(
                "sglang.srt.eplb.expert_distribution.get_global_expert_distribution_recorder",
                return_value=recorder))
            # vl_routing imports this accessor directly after the fix.
            stack.enter_context(patch.object(vl_routing,
                "get_global_expert_distribution_recorder", return_value=recorder, create=True))
            if hcu_gate:
                stack.enter_context(patch.object(torch.ops.sglang,
                    "moe_fused_gate_sqrtsoftplus_v41_hcu", native_gate, create=True))
            if cuda_gate:
                kernel_module = ModuleType("sglang.kernels.ops.moe.moe_fused_gate")
                kernel_module.moe_fused_gate = cuda_kernel
                stack.enter_context(patch.dict("sys.modules", {
                    kernel_module.__name__: kernel_module}))
                stack.enter_context(patch(
                    "sglang.srt.layers.moe.utils.get_moe_runner_backend",
                    return_value=SimpleNamespace(is_flashinfer_mxfp4=lambda: True)))
            output = vl_routing.vision_topk(moe, logits, input_ids, count)

        torch.testing.assert_close(output.topk_ids.long(), expected_ids, atol=0, rtol=0)
        torch.testing.assert_close(output.topk_weights, expected_weights)
        self.assertIs(output.router_logits, logits)
        if nextn:
            init.assert_not_called()
        else:
            init.assert_called_once_with(layer_id=17)
        capture.assert_called_once()
        torch.testing.assert_close(capture.call_args.args[2].long(), logical_ids)
        recorder.on_select_experts.assert_called_once()
        recorded = recorder.on_select_experts.call_args.kwargs["topk_ids"]
        torch.testing.assert_close(recorded.long(), expected_ids[:, :2])
        self.assertEqual(native_gate.call_count, int(hcu_gate))
        self.assertEqual(cuda_kernel.call_count, int(cuda_gate))
        if cuda_gate:
            self.assertEqual(isinstance(output, topk.StandardTopKOutputPacked),
                             active_info is None)

    def test_mixed_text_and_image_bias_with_eplb(self):
        self._run_case()

    def test_eplb_disabled(self):
        self._run_case(mapping=False)

    def test_shared_expert_is_not_in_eplb_table(self):
        self._run_case(shared=1)

    def test_padding_and_idle_batch(self):
        for shared, padded, idle in [(0, 2, False), (1, 2, False), (1, 0, True)]:
            with self.subTest(shared=shared, padded=padded, idle=idle):
                self._run_case(shared=shared, padded=padded, idle=idle)

    def test_draft_does_not_use_target_mapping(self):
        self._run_case(nextn=True)

    def test_empty_batch(self):
        self._run_case(empty=True)

    def test_hcu_gate_keeps_per_token_bias_and_remaps(self):
        for shared in (0, 1):
            with self.subTest(shared=shared):
                self._run_case(hcu_gate=True, shared=shared, padded=2)

    def test_cuda_eplb_disables_stale_packed_ids(self):
        self._run_case(cuda_gate=True)

    def test_cuda_without_eplb_keeps_packed_output(self):
        self._run_case(cuda_gate=True, mapping=False)


if __name__ == "__main__":
    unittest.main()
