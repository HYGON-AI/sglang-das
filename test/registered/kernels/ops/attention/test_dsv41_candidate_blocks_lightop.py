"""HCU dense candidate-block selection: LightOp/legacy switch and mask ABI."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4 import lightop_indexer
from sglang.srt.layers.attention.dsv4 import candidate_indexer
from sglang.srt.utils import is_hcu
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# backend-specific: dsv41_select_candidate_blocks is a LightOp HCU native kernel.
register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")

FLAG = "SGLANG_USE_LIGHTOP_DSV41_CANDIDATE_BLOCKS"


def make_case(width, lengths, ties=False):
    generator = torch.Generator().manual_seed(width)
    # Unique block maxima let the legacy top-k and the stable LightOp selector
    # be compared exactly without prescribing the legacy tie order.
    logits = torch.stack(
        [torch.randperm((width + 7) // 8, generator=generator) for _ in lengths]
    ).float()
    if ties:
        logits.remainder_(3)
    logits = logits.repeat_interleave(8, dim=1)[:, :width]
    storage = torch.empty(len(lengths), (width + 3) // 4 * 4, device="cuda")
    storage[:, :width].copy_(logits)
    return storage[:, :width], torch.tensor(
        lengths, dtype=torch.int32, device="cuda"
    )


def reference(logits, lengths, topk):
    logits, lengths = logits.cpu(), lengths.cpu().tolist()
    rows, width = logits.shape
    mask = torch.zeros(rows, width, dtype=torch.bool)
    ids = torch.full((rows, topk), -1, dtype=torch.int32)
    for row, length in enumerate(lengths):
        length = min(length, width)
        nblocks = (length + 7) // 8
        if not nblocks:
            continue
        keys = logits[row, : (nblocks - 1) * 8].reshape(-1, 8).amax(1)
        keys = torch.cat((keys, torch.tensor([torch.inf])))
        kept = keys.argsort(descending=True, stable=True)[:topk].sort().values
        ids[row, : kept.numel()] = kept.int()
        for block in kept.tolist():
            mask[row, block * 8 : min((block + 1) * 8, width)] = True
    return mask, ids


@unittest.skipUnless(is_hcu() and torch.cuda.is_available(), "HCU kernels required")
class TestDsv41CandidateBlocksLightop(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        assert lightop_indexer.lightop_candidate_mask_available()

    def test_default_and_reference(self):
        for width, topk, lengths, ties in (
            (1024, 2048, [0, 1, 9, 1024], False),
            (1031, 64, [0, 9, 999, 1031], False),
            (20000, 2048, [9, 19997, 20000], False),
            (20000, 64, [1000, 20000], True),
            (131072, 2048, [65539, 131072], False),
        ):
            with self.subTest(width=width, topk=topk, ties=ties):
                logits, lengths = make_case(width, lengths, ties)
                expected_mask, expected_ids = reference(logits, lengths, topk)
                with patch.dict(os.environ):
                    os.environ.pop(FLAG, None)
                    with patch.object(
                        lightop_indexer,
                        "candidate_block_mask",
                        wraps=lightop_indexer.candidate_block_mask,
                    ) as native, patch.object(
                        candidate_indexer,
                        "_expand_blocks_kernel",
                        side_effect=AssertionError("fused mask must not expand"),
                    ), patch.object(
                        candidate_indexer,
                        "_hcu_topk_tables",
                        side_effect=AssertionError("fused mask needs no identity"),
                    ), patch.object(
                        lightop_indexer,
                        "candidate_block_table",
                        side_effect=AssertionError("fused mask needs no table operator"),
                    ):
                        mask, ids = candidate_indexer.select_candidate_blocks(
                            logits, lengths[:, None], topk, 8, return_block_ids=True
                        )
                        self.assertEqual(native.call_count, 1)
                torch.testing.assert_close(mask.cpu(), expected_mask, rtol=0, atol=0)
                torch.testing.assert_close(ids.cpu(), expected_ids, rtol=0, atol=0)

    def test_missing_package_wrapper_or_native_raises(self):
        logits, lengths = make_case(20000, [0, 9, 20000])
        symbol = SimpleNamespace(dsv41_select_candidate_blocks=object())
        available = lightop_indexer.lightop_candidate_mask_available.__wrapped__
        for modules in (
            None,
            (SimpleNamespace(), symbol),
            (symbol, SimpleNamespace()),
        ):
            with (
                self.subTest(modules=modules),
                patch.dict(os.environ, {FLAG: "1"}),
                patch.object(lightop_indexer, "_lightop_modules", return_value=modules),
                patch.object(
                    lightop_indexer,
                    "lightop_candidate_mask_available",
                    side_effect=available,
                ),
                patch.object(
                    candidate_indexer,
                    "_select_candidate_blocks_fused",
                    side_effect=AssertionError("missing LightOp must not use legacy"),
                ),
                patch.object(
                    candidate_indexer,
                    "_expand_blocks_kernel",
                    side_effect=AssertionError("missing LightOp must not expand"),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "dsv41_select_candidate_blocks"
                ):
                    candidate_indexer.select_candidate_blocks(
                        logits, lengths, 2048, 8, return_block_ids=True
                    )

    def test_legacy_switch(self):
        logits, lengths = make_case(20000, [1000, 19997, 20000])
        expected, _ = reference(logits, lengths, 2048)
        with patch.dict(os.environ, {FLAG: "0"}):
            legacy = candidate_indexer.select_candidate_blocks(
                logits, lengths, 2048, 8
            )
        with patch.dict(os.environ, {FLAG: "1"}):
            enabled = candidate_indexer.select_candidate_blocks(
                logits, lengths, 2048, 8
            )
        for mask in (legacy, enabled):
            torch.testing.assert_close(mask.cpu(), expected, rtol=0, atol=0)

    def test_unsupported_layout_raises(self):
        logits, lengths = make_case(20000, [1000, 20000])
        for scores in (logits.to(torch.bfloat16), logits[:, 1:], logits[:, ::2]):
            with self.subTest(
                shape=scores.shape, dtype=scores.dtype, stride=scores.stride()
            ):
                lens = lengths.clamp(max=scores.shape[1])
                with patch.dict(os.environ, {FLAG: "1"}), patch.object(
                    candidate_indexer, "_select_candidate_blocks_fused",
                    side_effect=AssertionError("unsupported ABI must not use legacy"),
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "dsv41_select_candidate_blocks"
                    ):
                        candidate_indexer.select_candidate_blocks(
                            scores, lens, 2048, 8
                        )

    def test_graph_replay_refreshes_lengths_and_padding(self):
        for width in (20000, 131072):
            with self.subTest(width=width), patch.dict(os.environ, {FLAG: "1"}):
                logits, lengths = make_case(width, [width, 100, 0])
                # Warm up allocations before capture; no identity table is needed.
                candidate_indexer.select_candidate_blocks(logits, lengths, 2048, 8)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    mask, ids = candidate_indexer.select_candidate_blocks(
                        logits, lengths, 2048, 8, return_block_ids=True
                    )
                for new_lengths in ([0, width, 9], [width - 3, 1, width]):
                    lengths.copy_(
                        torch.tensor(new_lengths, dtype=torch.int32, device="cuda")
                    )
                    graph.replay()
                    expected_mask, expected_ids = reference(logits, lengths, 2048)
                    torch.testing.assert_close(
                        mask.cpu(), expected_mask, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        ids.cpu(), expected_ids, rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
