"""Unit tests for DSA forward-batch layout helpers."""

import unittest
from types import SimpleNamespace

from sglang.srt.layers.attention.dsa.forward_batch_utils import (
    effective_forward_mode,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSAForwardBatchUtils(CustomTestCase):
    def test_effective_mode_uses_original_mode_during_mlp_sync(self):
        current_mode = object()
        original_mode = object()
        forward_batch = SimpleNamespace(
            forward_mode=current_mode,
            _original_forward_mode=original_mode,
        )

        self.assertIs(effective_forward_mode(forward_batch), original_mode)

    def test_effective_mode_handles_explicit_none(self):
        current_mode = object()
        forward_batch = SimpleNamespace(
            forward_mode=current_mode,
            _original_forward_mode=None,
        )

        self.assertIs(effective_forward_mode(forward_batch), current_mode)

    def test_effective_mode_handles_missing_original_field(self):
        current_mode = object()
        forward_batch = SimpleNamespace(forward_mode=current_mode)

        self.assertIs(effective_forward_mode(forward_batch), current_mode)


if __name__ == "__main__":
    unittest.main()
