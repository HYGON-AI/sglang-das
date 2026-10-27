# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""DeepEP text batches must retain standard TopK's EPLB postprocessing."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import deepseek_v2

register_cpu_ci(est_time=3, stage="base-a", runner_config="cpu")


class TestDeepEPTextRouting(CustomTestCase):
    def _check_routing(self, mode, *, hip=True, images=False, vl_bias=True,
                       expect_vision=False):
        # Expert weights have moved: logical IDs must be remapped before dispatch.
        physical_to_logical = torch.tensor([2, 0, 3, 1])
        logical_to_physical = physical_to_logical.argsort()
        logical_ids = torch.tensor([[1, 3], [3, 1]], dtype=torch.int32)
        physical_ids = logical_to_physical[logical_ids.long()]
        weights = torch.tensor([[0.6, 0.4], [0.7, 0.3]])
        logits = torch.zeros((2, 4))
        standard_output = StandardTopKOutput(weights, physical_ids, logits)
        vision_output = StandardTopKOutput(weights, logical_ids, logits)
        topk = Mock(return_value=standard_output)
        experts = Mock(side_effect=lambda **kw: kw["topk_output"].topk_ids.float())
        experts.should_fuse_routed_scaling_factor_in_topk = True
        gate = Mock(return_value=logits)
        gate.e_score_correction_bias_vl = torch.zeros(4) if vl_bias else None
        hidden = torch.zeros((2, 2))
        batch = SimpleNamespace(
            forward_mode=mode,
            contains_image_inputs=lambda: images,
            moe_num_token_non_padded=lambda: torch.tensor(2, dtype=torch.int32),
        )
        info = object()
        input_ids = torch.tensor([42, 43])
        moe = SimpleNamespace(
            _fuse_shared_experts_inside_sbo=False,
            is_nextn=False,
            num_fused_shared_experts=0,
            alt_stream=None,
            gate=gate,
            topk=topk,
            experts=experts,
            layer_id=17,
            routed_scaling_factor=1.5,
            _forward_shared_experts=Mock(return_value=None),
        )
        with (
            patch.object(deepseek_v2, "_is_hip", hip),
            patch.object(deepseek_v2, "_use_aiter", False),
            patch.object(deepseek_v2, "vision_topk", return_value=vision_output) as vl,
            patch.object(deepseek_v2.ExpertLocationDispatchInfo, "init_new",
                         return_value=info),
            patch.object(
                deepseek_v2.envs.SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO,
                "get", return_value=False,
            ),
        ):
            actual = deepseek_v2.DeepseekV2MoE.forward_deepep(
                moe, hidden, batch, input_ids_global=input_ids
            )

        dispatched = experts.call_args.kwargs["topk_output"]
        if expect_vision:
            vl.assert_called_once()
            topk.assert_not_called()
            self.assertIs(dispatched, vision_output)
            torch.testing.assert_close(actual, logical_ids.float(), atol=0, rtol=0)
        else:
            vl.assert_not_called()
            topk.assert_called_once()
            self.assertIs(topk.call_args.kwargs["expert_location_dispatch_info"], info)
            self.assertIs(dispatched, standard_output)
            torch.testing.assert_close(actual, physical_ids.float(), atol=0, rtol=0)
            torch.testing.assert_close(
                physical_to_logical[dispatched.topk_ids.long()],
                logical_ids.long(), atol=0, rtol=0,
            )

    def test_hip_text_prefill_preserves_eplb_dispatch(self):
        self._check_routing(ForwardMode.EXTEND)

    def test_hip_decode_preserves_eplb_dispatch(self):
        # Decode can retain multimodal request state but its new tokens are text.
        self._check_routing(ForwardMode.DECODE, images=True)

    def test_hip_verify_preserves_eplb_dispatch(self):
        self._check_routing(ForwardMode.TARGET_VERIFY, images=True)

    def test_hip_image_extend_keeps_vision_routing(self):
        # Admission regression only; multimodal EPLB remapping is separate work.
        self._check_routing(ForwardMode.EXTEND, images=True, expect_vision=True)

    def test_non_hip_vision_admission_is_unchanged(self):
        self._check_routing(ForwardMode.EXTEND, hip=False, expect_vision=True)

    def test_model_without_vl_bias_uses_standard_topk(self):
        self._check_routing(ForwardMode.EXTEND, images=True, vl_bias=False)


if __name__ == "__main__":
    unittest.main()
