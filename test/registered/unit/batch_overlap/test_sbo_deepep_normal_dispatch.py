"""Regression tests for DeepEP normal-mode SBO overlap arguments."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.batch_overlap.single_batch_overlap import SboFlags, compute_overlap_args
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class TestSboDeepEPNormalDispatch(CustomTestCase):
    @patch.object(
        SboFlags, "enable_combine_shared_two_stream_overlap", return_value=False
    )
    @patch.object(
        SboFlags, "enable_combine_down_gemm_two_stream_overlap", return_value=True
    )
    def test_2d_dispatch_skips_secondary_overlap(self, _down_enabled, _shared_enabled):
        with patch("torch.cuda.get_device_properties") as properties, patch(
            "torch.cuda.Event"
        ) as event:
            for tokens in (0, 4):
                with self.subTest(tokens=tokens):
                    dispatch_output = SimpleNamespace(
                        hidden_states=torch.empty((tokens, 16))
                    )
                    self.assertEqual(
                        compute_overlap_args(dispatch_output, alt_stream=None),
                        (None, None, {}),
                    )
            properties.assert_not_called()
            event.assert_not_called()

    @patch.object(
        SboFlags, "enable_combine_shared_two_stream_overlap", return_value=False
    )
    @patch.object(
        SboFlags, "enable_combine_down_gemm_two_stream_overlap", return_value=True
    )
    def test_3d_dispatch_keeps_secondary_overlap(self, _down_enabled, _shared_enabled):
        dispatch_output = SimpleNamespace(hidden_states=torch.empty((4, 129, 16)))
        stream = object()
        event = object()
        with patch(
            "torch.cuda.get_device_properties",
            return_value=SimpleNamespace(multi_processor_count=80),
        ), patch("torch.cuda.Event", return_value=event), patch(
            "sglang.srt.batch_overlap.single_batch_overlap.is_blackwell",
            return_value=False,
        ), patch(
            "sglang.srt.batch_overlap.single_batch_overlap.envs.SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS.is_set",
            return_value=False,
        ):
            combine, down, metadata = compute_overlap_args(dispatch_output, stream)
        self.assertTrue(combine.overlap)
        self.assertIs(combine.stream, stream)
        self.assertEqual(combine.num_sms, 3)
        self.assertEqual(combine.threshold, 77)
        self.assertIs(combine.wait_event, event)
        self.assertIs(down.start_event, event)
        self.assertIs(down.signal, combine.signal)
        self.assertEqual(down.num_sms, 77)
        self.assertEqual(tuple(down.signal.shape), (12,))
        self.assertEqual(down.signal.dtype, torch.int32)
        self.assertEqual(torch.count_nonzero(down.signal).item(), 0)
        self.assertEqual(metadata, {"compute_num_sms": 77})

    @patch.object(
        SboFlags, "enable_combine_shared_two_stream_overlap", return_value=False
    )
    @patch.object(
        SboFlags, "enable_combine_down_gemm_two_stream_overlap", return_value=False
    )
    def test_disabled_overlap_is_unchanged(self, _down_enabled, _shared_enabled):
        self.assertEqual(compute_overlap_args(None, None), (None, None, {}))


if __name__ == "__main__":
    unittest.main()
