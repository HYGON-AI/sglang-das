"""V4.1 query tiling contracts; real orchestration, CPU operator boundaries."""

import ast
import enum
import functools
import math
import os
import sys
import types
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[5]
ATTENTION = ROOT / "python/sglang/srt/layers/attention"
FLAG = "SGLANG_DSV41_PAGED_MQA_LOGITS_FREE_MEM_FRACTION"


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if len(nodes) != len(names):
        raise AssertionError(f"missing functions in {path}: {names}")
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.Module(body=[future, *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


class Mode(enum.Enum):
    EXTEND = 1
    DECODE = 2
    TARGET_VERIFY = 3


class Candidates:
    def __init__(self, request_masks=None, request_block_ids=None, **kwargs):
        self.request_masks = request_masks
        self.request_block_ids = request_block_ids


class TestDSV41MQABudget(CustomTestCase):
    def setUp(self):
        self.queries = []
        self.static = 0.8

        def memory(device, **kwargs):
            self.queries.append((device, kwargs))
            return 10_000, 100_000

        self.ns = dict(
            torch=torch,
            os=os,
            math=math,
            functools=functools,
            MQA_LOGITS_BYTES_PER_ELEM=4,
            MQA_LOGITS_TOTAL_MEM_FRACTION=0.3,
            MQA_LOGITS_STATIC_SKIP_ELEMS=8_000_000,
            mqa_logits_free_mem_bytes=memory,
            get_schedule=lambda: SimpleNamespace(mem_fraction_static=self.static),
        )
        load_functions(
            ATTENTION / "mqa_logits_utils.py",
            {
                "_dsv41_mqa_logits_budget_bytes",
                "dsv41_mqa_logits_rows_per_chunk",
                "mqa_logits_needs_budget_check",
                "mqa_logits_rows_per_chunk",
            },
            self.ns,
        )
        self.chunk = self.ns["dsv41_mqa_logits_rows_per_chunk"]

    def test_unset_is_independent_of_dsa_and_never_queries_device(self):
        with patch.dict(
            os.environ, {"SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION": "0.01"}
        ):
            os.environ.pop(FLAG, None)
            with patch.object(
                torch.cuda, "is_current_stream_capturing", side_effect=AssertionError
            ):
                self.assertIsNone(
                    self.chunk(num_rows=4096, num_cols=1_000_000, device_index=0)
                )
        self.assertEqual(self.queries, [])

    def test_invalid_fractions_fail_before_memory_query(self):
        for value in ("", "abc", "0", "-1", "1.1", "nan", "inf"):
            with self.subTest(value=value), patch.dict(os.environ, {FLAG: value}):
                with self.assertRaisesRegex(ValueError, FLAG):
                    self.chunk(num_rows=4096, num_cols=10000, device_index=0)
        self.assertEqual(self.queries, [])

    def test_capture_and_small_matrices_do_not_query_memory(self):
        with patch.dict(os.environ, {FLAG: "0.05"}):
            with patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=True
            ):
                self.assertIsNone(
                    self.chunk(num_rows=4096, num_cols=10000, device_index=0)
                )
            with patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=False
            ):
                self.assertIsNone(self.chunk(num_rows=10, num_cols=100, device_index=0))
        self.assertEqual(self.queries, [])

    def test_budget_cache_includes_fraction_static_fraction_and_device(self):
        budget = self.ns["_dsv41_mqa_logits_budget_bytes"]
        self.assertEqual(budget(0, 0.05, 0.8), 500)
        self.assertEqual(budget(0, 0.05, 0.8), 500)
        self.assertEqual(len(self.queries), 1)
        self.assertEqual(budget(0, 0.1, 0.8), 1000)
        self.assertEqual(budget(0, 0.05, 0.99), 50)
        self.assertEqual(budget(1, 0.05, 0.8), 500)
        self.assertEqual(budget(0, 0.05, None), 500)
        self.assertEqual(len(self.queries), 5)
        self.assertTrue(
            all(
                kwargs == {"include_reusable_cache": True} for _, kwargs in self.queries
            )
        )

    def test_native_physical_width_no_aiter_stride_padding(self):
        with (
            patch.dict(os.environ, {FLAG: "1"}),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        ):
            self.ns["_dsv41_mqa_logits_budget_bytes"] = lambda *args: 3 * 1004 * 4
            self.assertEqual(
                self.chunk(num_rows=10000, num_cols=1004, device_index=0), 3
            )
            self.ns["_dsv41_mqa_logits_budget_bytes"] = lambda *args: 1
            self.assertEqual(
                self.chunk(num_rows=10000, num_cols=1004, device_index=0), 1
            )
            self.ns["_dsv41_mqa_logits_budget_bytes"] = lambda *args: 10**12
            self.assertIsNone(self.chunk(num_rows=10000, num_cols=1004, device_index=0))


class TestDSV41MQARowChunking(CustomTestCase):
    def setUp(self):
        self.ns = dict(
            torch=torch,
            F=F,
            _is_hcu=True,
            CandidateMasks=Candidates,
            published_masks=lambda value: value,
            ceil_align=lambda n, a: (n + a - 1) // a * a,
        )
        load_functions(
            ATTENTION / "dsv4/candidate_indexer.py",
            {"select_candidate_blocks"},
            self.ns,
        )
        load_functions(
            ATTENTION / "deepseek_v4_backend.py",
            {
                "_prefill_candidate_mask",
                "_low_ratio_index_topk_chunked",
                "_publish_or_consume_candidates",
                "_low_ratio_index_topk_dense",
                "enter_late_layer_tail",
            },
            self.ns,
        )
        torch.manual_seed(51)
        self.lens = [11, 3, 9, 0]
        self.counts = [5, 0, 3, 1]
        self.q = torch.rand(9, 2, 128)
        self.k = torch.rand(23, 128)
        self.weights = torch.rand(9, 2)
        self.ks = torch.tensor([0] * 5 + [14] * 3 + [23], dtype=torch.int32)
        self.compress = torch.tensor([3, 5, 7, 9, 11, 2, 6, 9, 0], dtype=torch.int32)
        self.ke = self.ks + self.compress
        self.slots = torch.arange(23, dtype=torch.int64) * 3
        self.indexer = SimpleNamespace(
            index_topk=3,
            candidate_topk_blocks=2,
            candidate_block_size=2,
            uses_candidates=False,
            is_candidate_source=False,
        )
        self.owner = SimpleNamespace(
            forward_metadata=SimpleNamespace(candidate_metadata=None)
        )
        self.calls = []
        self.previous = None
        self.ns["_hcu_dense_fp4_mqa_logits"] = self.kernel
        self.ns["topk_transform_ragged_v2"] = self.topk
        self.ns["_hcu_candidate_logits_available"] = lambda: False
        self.ns["_TORCH_INDEXER_SCORE_BUDGET_BYTES"] = 1 << 30
        module = types.ModuleType("lightop.attention")
        module.dsv41_ragged_topk_postprocess = self.postprocess
        self.modules = patch.dict(sys.modules, {"lightop.attention": module})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def scores(self, q, weights, ks, ke, width):
        result = torch.full((q.shape[0], width), -torch.inf)
        for row in range(q.shape[0]):
            a, b = int(ks[row]), int(ke[row])
            values = torch.einsum("hd,kd->hk", q[row], self.k[a:b]).relu()
            result[row, : b - a] = (values * weights[row, :, None]).sum(0)
        return result

    def kernel(self, q, kv, weights, ks, ke, width, **kwargs):
        if self.previous is not None:
            self.assertIsNone(
                self.previous(), "previous logits/view survives next allocation"
            )
        self.assertIs(kv[0], self.k)
        self.calls.append(q[0].shape[0])
        result = self.scores(q[0], weights, ks, ke, width)
        blocks = kwargs.get("candidate_blocks")
        if blocks is not None:
            self.assertEqual(kwargs["candidate_query_lens"], [q[0].shape[0]])
            candidate = Candidates(request_block_ids=blocks)
            keep = self.ns["_prefill_candidate_mask"](
                candidate, 0, slice(None), width, self.indexer.candidate_block_size
            )
            result.masked_fill_(~keep, -torch.inf)
        self.previous = weakref.ref(result)
        return result

    @staticmethod
    def topk(logits, lens, out_offsets, out_indices):
        ranked = logits.topk(out_indices.shape[1], dim=-1).indices
        out_indices.copy_(ranked + out_offsets[:, None])

    @staticmethod
    def postprocess(logits, selected, ks, slots, pages, raw, masked):
        columns = selected.long() - ks[:, None]
        scores = logits.gather(1, columns)
        valid = scores > -torch.inf
        chosen = (
            selected.masked_fill(~valid, torch.iinfo(torch.int32).max).sort(-1).values
        )
        valid = chosen != torch.iinfo(torch.int32).max
        pages.copy_(
            torch.where(valid, slots[chosen.clamp_max(slots.numel() - 1).long()], -1)
        )
        if raw is not None:
            raw.copy_(torch.where(valid, chosen - ks[:, None], -1))

    def run_chunked(self, step, candidate_blocks=None, raw=True):
        self.previous = None
        pages = torch.full((12, 3), -1, dtype=torch.int32)
        indices = torch.full_like(pages, -1) if raw else None
        self.ns["_low_ratio_index_topk_chunked"](
            self.owner,
            self.indexer,
            self.q,
            (self.k, None),
            self.weights,
            self.ks,
            self.ke,
            self.compress,
            self.slots,
            self.lens,
            self.counts,
            pages,
            indices,
            candidate_blocks,
            step,
        )
        return pages, indices

    def expected(self, candidate=None):
        logits = self.scores(self.q, self.weights, self.ks, self.ke, 12)
        if candidate is not None:
            offset = 0
            for request, count in enumerate(self.counts):
                if self.lens[request] and count:
                    logits[offset : offset + count, : self.lens[request]].masked_fill_(
                        ~self.ns["_prefill_candidate_mask"](
                            candidate, request, slice(None), self.lens[request], 2
                        ),
                        -torch.inf,
                    )
                offset += count
        selected = torch.empty((9, 3), dtype=torch.int32)
        self.topk(logits, self.compress, self.ks, selected)
        pages = torch.full((12, 3), -1, dtype=torch.int32)
        raw = torch.full_like(pages, -1)
        self.postprocess(
            logits, selected, self.ks, self.slots, pages[:9], raw[:9], True
        )
        return pages, raw

    def test_ragged_cp_real_rows_offsets_empty_requests_and_padding(self):
        expected = self.expected()
        for step in (1, 2, 4, 20):
            with self.subTest(step=step):
                self.calls.clear()
                result = self.run_chunked(step)
                for got, want in zip(result, expected):
                    torch.testing.assert_close(got, want)
                self.assertLessEqual(max(self.calls), step)
                self.assertEqual(sum(self.calls), 8)  # zero-history row stays -1

    def test_source_keeps_only_compact_ids_and_releases_logits_views(self):
        self.indexer.is_candidate_source = True
        result = self.run_chunked(2)
        for got, want in zip(result, self.expected()):
            torch.testing.assert_close(got, want)
        candidate = self.owner.forward_metadata.candidate_metadata
        self.assertIsNone(candidate.request_masks)
        self.assertEqual(len(candidate.request_block_ids), len(self.counts))
        offset = 0
        for request, (count, lc) in enumerate(zip(self.counts, self.lens)):
            if count and lc:
                full = self.scores(self.q, self.weights, self.ks, self.ke, 12)
                expected = self.ns["select_candidate_blocks"](
                    full[offset : offset + count, :lc],
                    self.compress[offset : offset + count, None],
                    2,
                    2,
                )
                got = self.ns["_prefill_candidate_mask"](
                    candidate, request, slice(None), lc, 2
                )
                torch.testing.assert_close(got, expected)
            offset += count

    def test_consumers_support_compact_ids_and_original_masks(self):
        self.indexer.is_candidate_source = True
        self.run_chunked(2)
        candidate = self.owner.forward_metadata.candidate_metadata
        self.indexer.is_candidate_source = False
        self.indexer.uses_candidates = True
        expected = self.expected(candidate)
        for blocks in (None, candidate.request_block_ids):
            for got, want in zip(self.run_chunked(2, blocks), expected):
                torch.testing.assert_close(got, want)
        masks = [
            self.ns["_prefill_candidate_mask"](candidate, b, slice(None), lc, 2)
            if lc and count
            else torch.zeros(count, lc, dtype=torch.bool)
            for b, (lc, count) in enumerate(zip(self.lens, self.counts))
        ]
        self.owner.forward_metadata.candidate_metadata = Candidates(request_masks=masks)
        for got, want in zip(self.run_chunked(2), expected):
            torch.testing.assert_close(got, want)

    def test_compact_mask_handles_invalid_ids_partial_blocks_and_sliced_rows(self):
        candidate = Candidates(
            request_block_ids=[torch.tensor([[0, -1, 99], [2, 1, -1]])]
        )
        result = self.ns["_prefill_candidate_mask"](candidate, 0, slice(1, 2), 5, 2)
        torch.testing.assert_close(
            result, torch.tensor([[False, False, True, True, True]])
        )

    def test_no_raw_output(self):
        pages, raw = self.run_chunked(2, raw=False)
        self.assertIsNone(raw)
        torch.testing.assert_close(pages, self.expected()[0])

    def test_nonchunked_consumer_accepts_compact_candidate_metadata(self):
        self.indexer.is_candidate_source = True
        self.run_chunked(2)
        self.indexer.is_candidate_source = False
        self.indexer.uses_candidates = True
        candidate = self.owner.forward_metadata.candidate_metadata
        logits = self.scores(self.q, self.weights, self.ks, self.ke, 12)
        self.ns["_publish_or_consume_candidates"](
            self.owner,
            self.indexer,
            logits,
            self.compress,
            self.lens,
            self.counts,
            torch.zeros(0, 0, dtype=torch.bool),
        )
        expected = self.scores(self.q, self.weights, self.ks, self.ke, 12)
        offset = 0
        for request, (count, lc) in enumerate(zip(self.counts, self.lens)):
            if count and lc:
                keep = self.ns["_prefill_candidate_mask"](
                    candidate, request, slice(None), lc, 2
                )
                expected[offset : offset + count, :lc].masked_fill_(~keep, -torch.inf)
            offset += count
        torch.testing.assert_close(logits, expected)

    def test_dense_entry_dispatch_and_original_path_when_unset_or_decode_role(self):
        # Exercise the actual dense entry, including its EP/CP execution trim.
        q_lens = torch.tensor(self.counts, dtype=torch.int32)
        pages = torch.full((12, 3), -1, dtype=torch.int32)
        raw = torch.full_like(pages, -1)
        core = SimpleNamespace(
            sparse_page_indices=lambda _: pages, sparse_raw_indices=lambda _: raw
        )
        self.owner.forward_metadata = SimpleNamespace(
            core_metadata=core,
            candidate_metadata=None,
            low_ratio_query_lens=q_lens,
            low_ratio_dense_metadata={
                1: SimpleNamespace(
                    k_slots=self.slots,
                    ks=self.ks,
                    ke=self.ke,
                    compress_lens=self.compress,
                )
            },
        )
        self.owner.token_to_kv_pool = SimpleNamespace(
            get_low_ratio_index_k_fp4=lambda *args, **kwargs: (self.k, None)
        )
        # Native K scales are a tensor; use a CPU stand-in here.
        self.owner.token_to_kv_pool.get_low_ratio_index_k_fp4 = (
            lambda *args, **kwargs: (self.k, torch.ones(23))
        )
        self.owner._hcu_prefill_candidate_blocks = lambda *args: None
        self.owner._low_ratio_index_topk_chunked = types.MethodType(
            self.ns["_low_ratio_index_topk_chunked"], self.owner
        )
        self.indexer.queries = lambda q, freqs: q
        self.indexer.head_weights = lambda x: self.weights
        layer = SimpleNamespace(
            compress_ratio=1,
            indexer=self.indexer,
            layer_id=20,
            freqs_cis=torch.ones(12),
        )
        pos = torch.cat(
            (self.compress - 1, torch.ones(3, dtype=torch.int32))
        ).clamp_min(0)
        q_padded = F.pad(self.q, (0, 0, 0, 0, 0, 3))
        self.ns.update(
            DSV4Metadata=SimpleNamespace, _as_int_list=lambda v: v, ForwardMode=Mode
        )
        utils = types.ModuleType("sglang.srt.layers.attention.mqa_logits_utils")
        quant = types.ModuleType("sglang.kernels.ops.attention.dsv4.fp4_indexer")
        quant.quantize_fp4_indexer_tensor = lambda *args, **kwargs: None
        for role, mode, budget_rows, expected_calls in (
            ("prefill", Mode.EXTEND, 2, [2, 2, 1, 2, 1]),
            ("prefill", Mode.EXTEND, None, [9]),
            ("decode", Mode.EXTEND, 2, [9]),
            ("prefill", Mode.TARGET_VERIFY, 2, [9]),
        ):
            with self.subTest(role=role, mode=mode, budget_rows=budget_rows):
                budget_calls = []

                def budget(**kwargs):
                    budget_calls.append(kwargs)
                    return budget_rows

                utils.dsv41_mqa_logits_rows_per_chunk = budget
                self.ns["get_disagg"] = lambda: SimpleNamespace(
                    disaggregation_mode=role
                )
                batch = SimpleNamespace(seq_lens_cpu=self.lens, forward_mode=mode)
                self.calls.clear()
                self.previous = None
                with patch.dict(
                    sys.modules,
                    {
                        utils.__name__: utils,
                        quant.__name__: quant,
                    },
                ):
                    self.ns["_low_ratio_index_topk_dense"](
                        self.owner,
                        layer,
                        torch.ones(12, 1),
                        q_padded,
                        pos,
                        batch,
                        q_lens,
                        self.counts,
                    )
                self.assertEqual(self.calls, expected_calls)
                self.assertEqual(
                    len(budget_calls), int(role == "prefill" and mode == Mode.EXTEND)
                )
                for got, want in zip((pages, raw), self.expected()):
                    torch.testing.assert_close(got, want)

    def test_bounded_replay_tail_carries_compact_ids_including_empty_rank_rows(self):
        blocks = [
            torch.arange(10).view(5, 2),
            torch.empty(0, 2),
            torch.arange(6).view(3, 2),
        ]
        old = SimpleNamespace(
            candidate_metadata=Candidates(request_block_ids=blocks),
            core_attn_metadata=SimpleNamespace(),
        )
        tail = SimpleNamespace(cp_metadata=None, extend_seq_lens_cpu=[2, 0, 0])
        new = SimpleNamespace(
            late_layer_tail=tail, core_attn_metadata=SimpleNamespace(low_ratios=[])
        )
        owner = SimpleNamespace(
            forward_metadata=old,
            tail_forward_metadata=new,
            token_to_kv_pool=SimpleNamespace(request_window=None),
        )
        self.ns.update(
            get_local_dp_buffer_len=lambda: 10,
            is_cp_cache_layer_split_deepseek_v4_pool=lambda pool: False,
        )
        batch = SimpleNamespace(attn_cp_metadata=None)
        saved = self.ns["enter_late_layer_tail"](owner, batch)
        self.assertIs(saved[0], old)
        self.assertIs(owner.forward_metadata, new)
        self.assertIsNone(new.candidate_metadata.request_masks)
        for got, want in zip(
            new.candidate_metadata.request_block_ids,
            (blocks[0][-2:], blocks[1], blocks[2][:0]),
        ):
            torch.testing.assert_close(got, want)

    def test_sparse_consumer_accepts_block_only_source_and_preserves_padding(self):
        ns = dict(torch=torch, INVALID_BLOCK=torch.iinfo(torch.int32).max)
        load_functions(
            ATTENTION / "dsv4/prefill_sparse_lightop.py",
            {"consume_prefill_candidate_sparse"},
            ns,
        )
        retained = [torch.tensor([[0, 1, -1, -1], [1, 2, 3, -1]], dtype=torch.int32)]

        def native(q, weights, freqs, positions, **kwargs):
            self.assertIs(kwargs["request_block_ids"], retained)
            kwargs["out_page_indices"][:2, :512] = 7
            kwargs["out_raw_indices"][:2, :512] = 9

        ns.update(
            _lightop_functions=lambda: (None, None, None),
            _lightop_fused_consumer=lambda x: native,
        )
        pages = torch.zeros(4, 513, dtype=torch.int32)
        raw = torch.zeros_like(pages)
        indexer = SimpleNamespace(
            uses_candidates=True,
            is_candidate_source=False,
            candidate_block_size=8,
            index_head_dim=128,
            n_local_heads=32,
            candidate_topk_blocks=4,
            index_topk=512,
            head_weights_raw=lambda x: torch.ones(2, 32),
            wq_b=lambda q: (torch.zeros(2, 32 * 128), None),
            head_weight_scale=1.0,
            rope_head_dim=64,
        )
        pool = SimpleNamespace(
            get_index_k_page_size=lambda ratio: 8,
            get_index_k_with_scale_buffer=lambda layer: torch.zeros(
                1, 8 * 68, dtype=torch.uint8
            ),
        )
        handled = ns["consume_prefill_candidate_sparse"](
            pool=pool,
            indexer=indexer,
            layer_id=21,
            x=torch.zeros(2, 1),
            q_lora=torch.zeros(2, 1),
            freqs_cis=torch.zeros(20),
            positions=torch.tensor([8, 9]),
            req_to_token=torch.arange(20).view(1, 20),
            request_pool_indices=torch.tensor([0]),
            lc_per_request=[10],
            rows_per_request=[2],
            candidate_metadata=Candidates(request_block_ids=retained),
            compress_ratio=1,
            page_indices=pages,
            raw_indices=raw,
        )
        self.assertTrue(handled)
        self.assertTrue(torch.all(pages[:2, :512] == 7))
        self.assertTrue(torch.all(raw[:2, :512] == 9))
        self.assertTrue(torch.all(pages[2:] == -1))
        self.assertTrue(torch.all(raw[:, 512:] == -1))


if __name__ == "__main__":
    unittest.main()
