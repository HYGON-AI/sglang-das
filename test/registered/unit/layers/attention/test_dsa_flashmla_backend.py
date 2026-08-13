"""Unit tests for platform-specific DSA FlashMLA routing and metadata."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.dsa.flashmla_backend import (
    DSAFlashMLAMetadata,
    can_fuse_flashmla_metadata,
    get_flashmla_op,
    refresh_flashmla_metadata,
    wrap_flashmla_metadata_result,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSAFlashMLABackend(CustomTestCase):
    def test_loader_routes_hcu_to_external_interface(self):
        expected_op = MagicMock()
        module = SimpleNamespace(flash_mla_sparse_fwd=expected_op)

        with patch(
            "sglang.srt.layers.attention.dsa.flashmla_backend.import_module",
            return_value=module,
        ) as import_mock:
            actual_op = get_flashmla_op("flash_mla_sparse_fwd", is_hcu=True)

        self.assertIs(actual_op, expected_op)
        import_mock.assert_called_once_with("flash_mla.flash_mla_interface")

    def test_loader_keeps_non_hcu_on_sgl_kernel(self):
        expected_op = MagicMock()
        module = SimpleNamespace(get_mla_metadata=expected_op)

        with patch(
            "sglang.srt.layers.attention.dsa.flashmla_backend.import_module",
            return_value=module,
        ) as import_mock:
            actual_op = get_flashmla_op("get_mla_metadata", is_hcu=False)

        self.assertIs(actual_op, expected_op)
        import_mock.assert_called_once_with("sgl_kernel.flash_mla")

    def test_hcu_metadata_object_is_wrapped_and_copied(self):
        backend_num_splits = torch.tensor([3], dtype=torch.int32)
        backend_metadata = SimpleNamespace(num_splits=backend_num_splits)
        source = wrap_flashmla_metadata_result((backend_metadata, None), is_hcu=True)

        destination = DSAFlashMLAMetadata(
            flashmla_metadata=SimpleNamespace(num_splits=None),
            num_splits=None,
        )
        destination.copy_(source)

        self.assertIs(destination.flashmla_metadata, backend_metadata)
        self.assertIs(destination.num_splits, backend_num_splits)
        self.assertFalse(can_fuse_flashmla_metadata(source, destination))

    def test_metadata_slice_accepts_uninitialized_hcu_scheduler(self):
        backend_metadata = SimpleNamespace(num_splits=None)
        metadata = wrap_flashmla_metadata_result((backend_metadata, None), is_hcu=True)

        sliced = metadata.slice(slice(0, 2))

        self.assertIs(sliced.flashmla_metadata, backend_metadata)
        self.assertIsNone(sliced.num_splits)

    def test_tensor_metadata_remains_fused_copy_eligible(self):
        source = DSAFlashMLAMetadata(
            flashmla_metadata=torch.tensor([1, 2], dtype=torch.int32),
            num_splits=torch.tensor([3, 4], dtype=torch.int32),
        )
        destination = DSAFlashMLAMetadata(
            flashmla_metadata=torch.zeros(2, dtype=torch.int32),
            num_splits=torch.zeros(2, dtype=torch.int32),
        )

        destination.copy_(source)

        self.assertTrue(
            torch.equal(destination.flashmla_metadata, source.flashmla_metadata)
        )
        self.assertTrue(torch.equal(destination.num_splits, source.num_splits))
        self.assertTrue(can_fuse_flashmla_metadata(source, destination))

    def test_hcu_refresh_returns_fresh_scheduler_object(self):
        captured_scheduler = SimpleNamespace(
            have_initialized=True,
            config="batch-6",
            tile_scheduler_metadata=object(),
            num_splits=object(),
        )
        fresh_scheduler = SimpleNamespace(
            have_initialized=False,
            config=None,
            tile_scheduler_metadata=None,
            num_splits=None,
        )
        destination = DSAFlashMLAMetadata(captured_scheduler, None)
        source = DSAFlashMLAMetadata(fresh_scheduler, None)

        refreshed = refresh_flashmla_metadata(
            destination,
            source,
            slice(0, 6),
            is_hcu=True,
        )

        self.assertIs(refreshed, source)
        self.assertIs(refreshed.flashmla_metadata, fresh_scheduler)
        self.assertIs(destination.flashmla_metadata, captured_scheduler)

    def test_tensor_refresh_keeps_destination_storage(self):
        destination = DSAFlashMLAMetadata(
            flashmla_metadata=torch.zeros(2, dtype=torch.int32),
            num_splits=torch.zeros(3, dtype=torch.int32),
        )
        source = DSAFlashMLAMetadata(
            flashmla_metadata=torch.tensor([1, 2], dtype=torch.int32),
            num_splits=torch.tensor([3, 4], dtype=torch.int32),
        )
        destination_metadata = destination.flashmla_metadata
        destination_splits = destination.num_splits

        refreshed = refresh_flashmla_metadata(
            destination,
            source,
            slice(0, 2),
            is_hcu=False,
        )

        self.assertIs(refreshed.flashmla_metadata, destination_metadata)
        self.assertEqual(refreshed.num_splits.data_ptr(), destination_splits.data_ptr())
        self.assertTrue(
            torch.equal(refreshed.flashmla_metadata, source.flashmla_metadata)
        )
        self.assertTrue(torch.equal(refreshed.num_splits, source.num_splits))

    def test_hcu_refresh_does_not_share_schedulers_across_steps(self):
        destination = DSAFlashMLAMetadata(SimpleNamespace(), None)
        first = DSAFlashMLAMetadata(SimpleNamespace(have_initialized=False), None)
        second = DSAFlashMLAMetadata(SimpleNamespace(have_initialized=False), None)

        first_refreshed = refresh_flashmla_metadata(
            destination, first, slice(0, 6), is_hcu=True
        )
        second_refreshed = refresh_flashmla_metadata(
            destination, second, slice(0, 6), is_hcu=True
        )

        self.assertIsNot(
            first_refreshed.flashmla_metadata,
            second_refreshed.flashmla_metadata,
        )


if __name__ == "__main__":
    unittest.main()
