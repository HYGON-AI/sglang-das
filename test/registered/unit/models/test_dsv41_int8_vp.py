"""CPU guards for INT8 VP; device arithmetic has separate HCU checks."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization import w8a8_int8 as int8
from sglang.srt.layers.quantization import dsv41_channelwise_fp8 as fp8
from sglang.srt.speculative.draft_lm_head_vp import (
    DSparkLMHeadVocabParallel, channelwise_lm_head_scale,
)
from sglang.kernels.ops.gemm.dsv41_int8 import require_int8_vp_backend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestInt8VPGuards(unittest.TestCase):
    def test_unset_backend_accepts_blaslt_default(self):
        with patch.dict(os.environ):
            os.environ.pop("SGLANG_DSV41_INT8_VP", None)
            os.environ.pop("SGLANG_DSV41_INT8_PROJECTION_BACKEND", None)
            require_int8_vp_backend()

    def head(self):
        head = torch.nn.Linear(4, 8, bias=False)
        int8.install(head)
        return head

    def test_int8_scale_needs_no_extra_vp_switch(self):
        head = self.head()
        with patch.dict(os.environ):
            os.environ.pop("SGLANG_DSV41_INT8_VP", None)
            os.environ.pop("SGLANG_DSV41_INT8_PROJECTION_BACKEND", None)
            self.assertIs(channelwise_lm_head_scale(head), head.weight_scale)

    def test_rejects_non_blaslt_backend(self):
        with patch.dict(os.environ, {"SGLANG_DSV41_INT8_PROJECTION_BACKEND": "auto"}):
            with self.assertRaisesRegex(ValueError, "blaslt"):
                require_int8_vp_backend()

    def test_int8_scale_is_the_original_channel_scale(self):
        head = self.head()
        with patch.dict(os.environ, {"SGLANG_DSV41_INT8_PROJECTION_BACKEND": "blaslt"}):
            self.assertIs(channelwise_lm_head_scale(head), head.weight_scale)

    def test_existing_fp8_scale_contract_unchanged(self):
        scale = torch.ones(8, 1)
        head = SimpleNamespace(quant_method=fp8.ChannelwiseFp8Method(), weight_scale_inv=scale)
        self.assertIs(channelwise_lm_head_scale(head), scale)
        self.assertIsNone(channelwise_lm_head_scale(SimpleNamespace(quant_method=None)))

    def test_wrong_scale_dtype_rejected_before_collectives(self):
        with patch.dict(os.environ, {"SGLANG_DSV41_INT8_PROJECTION_BACKEND": "blaslt"}):
            with self.assertRaisesRegex(ValueError, "FP32"):
                DSparkLMHeadVocabParallel(full_weight=torch.ones(8, 4, dtype=torch.int8),
                    weight_scale=torch.ones(8, 1, dtype=torch.float16), vocab_size=8,
                    vp_size=8, max_rows_per_rank=4, local_world_size=8,
                    activation_dtype=torch.bfloat16)

    def test_release_full_weight_cache_preserves_other_projections(self):
        from sglang.kernels.ops.gemm import dsv41_int8 as service

        weight, other = torch.ones(8, 4, dtype=torch.int8), torch.ones(8, 4, dtype=torch.int8)
        packed = torch.ones(32, dtype=torch.int8)
        entries = {"head": dict(owner=weight, packed=packed), "other": dict(owner=other, packed=packed)}
        with patch.dict(service._packs, entries, clear=True):
            self.assertEqual(service.release_weight_cache(weight), 1)
            self.assertEqual(set(service._packs), {"other"})
            self.assertIs(service._packs["other"]["owner"], other)


if __name__ == "__main__":
    unittest.main()
