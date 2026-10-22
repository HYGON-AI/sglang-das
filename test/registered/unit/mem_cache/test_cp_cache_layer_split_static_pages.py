import unittest

import torch

from sglang.srt.mem_cache.cp_cache_layer_split.staging import (
    contiguous_request_page_capacity,
    remap_indices_to_staging,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class TestCpCacheLayerSplitStaticPages(CustomTestCase):
    def test_contiguous_request_page_capacity(self):
        self.assertEqual(
            contiguous_request_page_capacity(32768 + 127, 1, 256, 1000), 131
        )
        self.assertEqual(contiguous_request_page_capacity(0, 4, 256, 1000), 1)
        self.assertEqual(contiguous_request_page_capacity(1000, 8, 64, 3), 3)

    def test_static_selection_padding_remap(self):
        active_mask = torch.tensor([1, 0, 1, 1, 0], dtype=torch.int32)
        selected_pages = torch.nonzero_static(
            active_mask, size=4, fill_value=-1
        ).flatten()
        indices = torch.tensor([1, 10, 13, -1], dtype=torch.int32)
        remap_fn = getattr(
            remap_indices_to_staging, "__wrapped__", remap_indices_to_staging
        )
        remapped = remap_fn(
            indices,
            selected_pages,
            page_size=4,
            max_pages=5,
            static_selected_pages=True,
        )
        self.assertTrue(torch.equal(remapped, torch.tensor([1, 6, 9, -1])))


if __name__ == "__main__":
    unittest.main()
