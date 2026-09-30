"""Unit tests for DSA forward-batch layout helpers."""

import unittest
from types import SimpleNamespace

from sglang.srt.layers.attention.dsa.forward_batch_utils import (
    effective_forward_mode,
    get_flashmla_kv_valid_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Mode:
    def __init__(self, kind):
        self.kind = kind

    def is_decode_or_idle(self):
        return self.kind in ("decode", "idle")

    def is_target_verify(self):
        return self.kind == "target_verify"

    def is_draft_extend_v2(self):
        return self.kind == "draft_extend_v2"


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

    def test_target_verify_uses_planned_five_of_six_rows(self):
        forward_batch = SimpleNamespace(
            forward_mode=_Mode("extend"),
            _original_forward_mode=_Mode("target_verify"),
            _original_batch_size=1,
            _original_num_tokens=5,
            forward_metadata_planned_bs=1,
            forward_metadata_planned_num_tokens=5,
        )

        self.assertEqual(get_flashmla_kv_valid_rows(forward_batch, 6), 5)

    def test_draft_extend_v2_uses_original_layout_without_preplan(self):
        forward_batch = SimpleNamespace(
            forward_mode=_Mode("extend"),
            _original_forward_mode=_Mode("draft_extend_v2"),
            _original_batch_size=1,
            _original_num_tokens=5,
            forward_metadata_planned_bs=None,
            forward_metadata_planned_num_tokens=None,
        )

        self.assertEqual(get_flashmla_kv_valid_rows(forward_batch, 6), 5)

    def test_decode_uses_request_rows_instead_of_transient_token_rows(self):
        forward_batch = SimpleNamespace(
            forward_mode=_Mode("extend"),
            _original_forward_mode=_Mode("decode"),
            _original_batch_size=5,
            _original_num_tokens=5,
            forward_metadata_planned_bs=5,
            forward_metadata_planned_num_tokens=25,
            spec_info=SimpleNamespace(num_tokens_per_req=1),
        )

        self.assertEqual(get_flashmla_kv_valid_rows(forward_batch, 6), 5)

    def test_inconsistent_plan_does_not_trim(self):
        forward_batch = SimpleNamespace(
            forward_mode=_Mode("extend"),
            _original_forward_mode=_Mode("target_verify"),
            _original_batch_size=1,
            _original_num_tokens=5,
            forward_metadata_planned_bs=2,
            forward_metadata_planned_num_tokens=5,
        )

        self.assertIsNone(get_flashmla_kv_valid_rows(forward_batch, 6))

    def test_unpadded_rows_do_not_trigger_repad(self):
        forward_batch = SimpleNamespace(
            forward_mode=_Mode("extend"),
            _original_forward_mode=_Mode("target_verify"),
            _original_batch_size=1,
            _original_num_tokens=5,
            forward_metadata_planned_bs=1,
            forward_metadata_planned_num_tokens=5,
        )

        self.assertIsNone(get_flashmla_kv_valid_rows(forward_batch, 5))


if __name__ == "__main__":
    unittest.main()
