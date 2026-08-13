"""Regression test for DeepEP normal-mode SBO overlap arguments."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.batch_overlap.single_batch_overlap import SboFlags, compute_overlap_args
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSboDeepEPNormalDispatch(CustomTestCase):
    @patch.object(
        SboFlags, "enable_combine_shared_two_stream_overlap", return_value=False
    )
    @patch.object(
        SboFlags, "enable_combine_down_gemm_two_stream_overlap", return_value=True
    )
    def test_2d_dispatch_skips_secondary_overlap(self, _down_enabled, _shared_enabled):
        dispatch_output = SimpleNamespace(hidden_states=torch.empty((4, 16)))

        self.assertEqual(
            compute_overlap_args(dispatch_output, alt_stream=None),
            (None, None, {}),
        )


if __name__ == "__main__":
    unittest.main()
