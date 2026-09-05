# Copyright (c) 2026 gencheng liu
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for cache selection and writes in the V4 compressor."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.dsv4.compressor_v2 import CompressorBackendMixin
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_hcu_ci

register_hcu_ci(est_time=30, suite="stage-a-test-1-hcu-small")


class TestCompressorDispatch(unittest.TestCase):
    def make_case(self, *, indexer=False, ratio=4, bf16=False, fp4=False):
        self.index_cache = torch.zeros(2, 64, 132, dtype=torch.uint8)
        self.unified_cache = torch.zeros(2, 64, 128, dtype=torch.bfloat16)
        self.extra_cache = torch.zeros(2, 64, 128, dtype=torch.uint8)
        self.out_loc = torch.tensor([4, 5])
        self.unified_loc = torch.tensor([6, 7])
        self.scores = torch.zeros(8, 512)
        self.state = torch.zeros(1, 4, 512)
        self.compress_pool = SimpleNamespace()
        pool = SimpleNamespace(
            get_index_k_with_scale_buffer=Mock(return_value=self.index_cache),
            get_index_k_page_size=Mock(return_value=64),
            get_unified_kv=Mock(return_value=self.unified_cache),
            get_extra_key_buffer=Mock(return_value=self.extra_cache),
            get_extra_key_page_size=Mock(return_value=32),
            is_bf16_attention_kv_cache=bf16,
            layer_mapping={3: (0, 0, self.compress_pool)},
        )
        self.compressor = SimpleNamespace(
            ratio=ratio,
            is_in_indexer=indexer,
            head_dim=128,
            rotate=True,
            compute_kv_score=Mock(return_value=self.scores),
            get_state_pool=Mock(
                return_value=SimpleNamespace(
                    kv_score_buffer=SimpleNamespace(kv_score=self.state)
                )
            ),
            ape=object(),
            norm=object(),
            freqs_cis=object(),
        )
        self.backend = SimpleNamespace(
            token_to_kv_pool=pool,
            enable_deepseek_v4_fp4_indexer=fp4,
            _get_out_loc=Mock(return_value=self.out_loc),
            _forward_compress_all_in_one=Mock(),
            forward_metadata=SimpleNamespace(
                core_metadata=SimpleNamespace(
                    unified=SimpleNamespace(**{f"c{ratio}_out_loc": self.unified_loc})
                )
            ),
        )

    def forward(self, *, unified=False, mode=ForwardMode.EXTEND, original_mode=None):
        batch = SimpleNamespace(forward_mode=mode, _original_forward_mode=original_mode)
        with patch(
            "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
            return_value=unified,
        ):
            CompressorBackendMixin.forward_unified(
                self.backend, self.scores, batch, 3, self.compressor
            )
        return batch

    def test_cache_layouts_and_forward_modes(self):
        for mode in (
            ForwardMode.EXTEND,
            ForwardMode.DECODE,
            ForwardMode.TARGET_VERIFY,
            ForwardMode.DRAFT_EXTEND_V2,
        ):
            for layout in ("indexer", "unified", "compressed"):
                for ratio in ((4,) if layout == "indexer" else (4, 128)):
                    with self.subTest(mode=mode, layout=layout, ratio=ratio):
                        self.make_case(indexer=layout == "indexer", ratio=ratio)
                        self.forward(unified=layout == "unified", mode=mode)
                        write = self.backend._forward_compress_all_in_one
                        write.assert_called_once()
                        args = write.call_args.kwargs
                        cache, loc, page_size = {
                            "indexer": (self.index_cache, self.out_loc, 64),
                            "unified": (self.unified_cache, self.unified_loc, 1),
                            "compressed": (self.extra_cache, self.out_loc, 32),
                        }[layout]
                        self.assertEqual(args["kv_cache"].data_ptr(), cache.data_ptr())
                        self.assertEqual(args["kv_cache"].dtype, torch.uint8)
                        self.assertIs(args["out_loc"], loc)
                        self.assertEqual(args["page_size"], page_size)
                        self.assertEqual(args["compress_ratio"], ratio)
                        self.assertEqual(args["head_dim"], 128)
                        self.assertTrue(args["rotate"])
                        self.assertEqual(args["is_indexer"], layout == "indexer")
                        self.assertEqual(args["bf16_store"], layout == "unified")
                        self.assertFalse(args["use_fp4_indexer"])
                        self.assertIs(args["kv_score_buffer"], self.state)
                        self.assertIs(args["kv_score_input"], self.scores)
                        self.assertIs(args["ape"], self.compressor.ape)
                        self.assertIs(args["norm"], self.compressor.norm)
                        self.assertIs(
                            args["freqs_cis_cache"], self.compressor.freqs_cis
                        )

    def test_indexer_takes_precedence_over_unified_and_bf16(self):
        for fp4 in (False, True):
            with self.subTest(fp4=fp4):
                self.make_case(indexer=True, bf16=True, fp4=fp4)
                self.forward(unified=True)
                write = self.backend._forward_compress_all_in_one
                write.assert_called_once()
                args = write.call_args.kwargs
                self.assertEqual(
                    args["kv_cache"].data_ptr(), self.index_cache.data_ptr()
                )
                self.assertIs(args["out_loc"], self.out_loc)
                self.assertEqual(args["use_fp4_indexer"], fp4)
                self.assertFalse(args["bf16_store"])
                self.backend.token_to_kv_pool.get_unified_kv.assert_not_called()

    def test_compressed_bf16_cache(self):
        self.make_case(bf16=True, fp4=True)
        self.forward()
        write = self.backend._forward_compress_all_in_one
        write.assert_called_once()
        self.assertTrue(write.call_args.kwargs["bf16_store"])
        self.assertFalse(write.call_args.kwargs["use_fp4_indexer"])

    def test_hisparse_address_translation(self):
        self.make_case()
        translated = torch.tensor([8, 9])
        self.compress_pool.translate_loc_to_hisparse_device = Mock()
        self.compress_pool._translate_loc_to_hisparse_device = Mock(
            return_value=translated
        )
        self.forward()
        translate = self.compress_pool._translate_loc_to_hisparse_device
        translate.assert_called_once_with(self.out_loc)
        write = self.backend._forward_compress_all_in_one
        write.assert_called_once()
        self.assertIs(write.call_args.kwargs["out_loc"], translated)

    def test_idle_does_not_compute_or_write(self):
        self.make_case()
        self.backend.online_c128_mtp = Mock()
        self.forward(mode=ForwardMode.IDLE)
        self.compressor.compute_kv_score.assert_not_called()
        self.compressor.get_state_pool.assert_not_called()
        self.backend._get_out_loc.assert_not_called()
        self.backend._forward_compress_all_in_one.assert_not_called()
        self.backend.online_c128_mtp.write_prefix_states.assert_not_called()

    def test_online_prefix_states_follow_cache_write(self):
        for layout in ("indexer", "unified", "compressed"):
            with self.subTest(layout=layout):
                self.make_case(indexer=layout == "indexer")
                calls = Mock()
                calls.attach_mock(self.backend._forward_compress_all_in_one, "cache")
                self.backend.online_c128_mtp = SimpleNamespace(
                    write_prefix_states=calls.prefix
                )
                self.forward(
                    unified=layout == "unified",
                    mode=ForwardMode.DECODE,
                    original_mode=ForwardMode.TARGET_VERIFY,
                )
                self.assertEqual(
                    [call[0] for call in calls.mock_calls], ["cache", "prefix"]
                )
                calls.prefix.assert_called_once_with(
                    layer_id=3,
                    compressor=self.compressor,
                    kv_score_input=self.scores,
                    logical_forward_mode=ForwardMode.TARGET_VERIFY,
                )


if __name__ == "__main__":
    unittest.main()
