import os
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.deepseek_v4_hook import apply_deepseek_v4_defaults
from sglang.srt.environ import envs
from sglang.srt.layers.attention import deepseek_v4_backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeepseekV4SparsePrefillRouting(CustomTestCase):
    def _apply_hip_defaults(self):
        cfg = SimpleNamespace(
            dsv4_attn_backend="auto",
            max_running_requests=1,
            speculative_algorithm=None,
        )
        with (
            patch(
                "sglang.srt.arg_groups.deepseek_v4_hook.get_platform",
                return_value=SimpleNamespace(is_hip=True),
            ),
            patch(
                "sglang.srt.arg_groups.deepseek_v4_hook.resolving_view",
                return_value=cfg,
            ),
            patch("sglang.srt.arg_groups.deepseek_v4_hook.run_post_process_pass"),
        ):
            apply_deepseek_v4_defaults(SimpleNamespace(), "DeepseekV4ForCausalLM")

    def test_explicit_sparse_prefill_env_survives_hip_defaults(self):
        with envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.override(True):
            self._apply_hip_defaults()
            self.assertTrue(envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get())

    def test_hip_default_disables_sparse_prefill_when_env_is_unset(self):
        name = envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.name
        old_value = os.environ.pop(name, None)
        try:
            self._apply_hip_defaults()
            self.assertFalse(envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get())
        finally:
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value

    def test_v41_sparse_prefill_allows_cp(self):
        q = SimpleNamespace(shape=(128, 1, 8, 512))
        with (
            patch.object(deepseek_v4_backend, "_is_sm120", False),
            patch.object(
                deepseek_v4_backend, "dsa_use_prefill_cp", return_value=True
            ),
            envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.override(True),
        ):
            self.assertTrue(
                deepseek_v4_backend._should_use_sparse_prefill(
                    q, SimpleNamespace(), allow_cp=True
                )
            )

    def test_legacy_v4_sparse_prefill_still_rejects_cp(self):
        q = SimpleNamespace(shape=(128, 1, 8, 512))
        with (
            patch.object(deepseek_v4_backend, "_is_sm120", False),
            patch.object(
                deepseek_v4_backend, "dsa_use_prefill_cp", return_value=True
            ),
            envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.override(True),
        ):
            self.assertFalse(
                deepseek_v4_backend._should_use_sparse_prefill(
                    q, SimpleNamespace(), allow_cp=False
                )
            )

    def _indexer_fixture(self, lengths, *, tail=None):
        backend = deepseek_v4_backend.DeepseekV4AttnBackend.__new__(
            deepseek_v4_backend.DeepseekV4AttnBackend
        )
        backend.forward_metadata = SimpleNamespace(late_layer_tail=tail)
        lengths_tensor = torch.tensor(lengths, dtype=torch.int32)
        batch = SimpleNamespace(
            attn_cp_metadata=SimpleNamespace(total_seq_lens=sum(lengths)),
            extend_seq_lens=lengths_tensor,
            extend_seq_lens_cpu=lengths,
            req_pool_indices=torch.arange(len(lengths), dtype=torch.int32) + 7,
            positions=torch.arange(sum(lengths), dtype=torch.int32),
            forward_mode=SimpleNamespace(
                is_decode=lambda: False,
                is_target_verify=lambda: False,
                is_extend=lambda: True,
            ),
        )
        return backend, batch, SimpleNamespace(indexer=object(), compressor=None)

    def _dense_capture(self, batch, captured):
        def capture(layer, x, q_lora, pos, forward_batch, q_lens, q_lens_cpu):
            self.assertIs(forward_batch, batch)
            self.assertEqual(q_lens.tolist(), q_lens_cpu)
            self.assertEqual(sum(q_lens_cpu), pos.shape[0])
            self.assertEqual(x.shape[0], pos.shape[0])
            self.assertEqual(q_lora.shape[0], pos.shape[0])
            req = torch.repeat_interleave(
                batch.req_pool_indices.to(torch.int64),
                q_lens.to(torch.int64),
                output_size=pos.shape[0],
            )
            captured.update(
                q_lens=q_lens,
                q_lens_cpu=q_lens_cpu,
                x=x,
                q_lora=q_lora,
                pos=pos,
                request_positions=torch.stack((req, pos), dim=1),
            )

        return capture

    def _assert_cp_extend_route(self, lengths, rank, expected_lens, *, tail=None):
        backend, batch, layer = self._indexer_fixture(lengths, tail=tail)
        num_local = sum(expected_lens)
        x = torch.arange((num_local + 2) * 2).reshape(num_local + 2, 2)
        q_lora = torch.arange((num_local + 2) * 3).reshape(num_local + 2, 3)
        positions = torch.arange(num_local + 2, dtype=torch.int32) * 8 + rank
        captured = {}
        with (
            patch.object(
                deepseek_v4_backend,
                "get_parallel",
                return_value=SimpleNamespace(attn_cp_rank=rank, attn_cp_size=8),
            ),
            patch.object(backend, "_use_dense_fp4_prefill_indexer", return_value=True),
            patch.object(
                backend,
                "_low_ratio_index_topk_extend",
                wraps=backend._low_ratio_index_topk_extend,
            ) as extend,
            patch.object(
                backend,
                "_low_ratio_index_topk_dense",
                side_effect=self._dense_capture(batch, captured),
            ),
        ):
            backend._forward_low_ratio_sources_cp(
                layer=layer,
                x=x,
                q_lora=q_lora,
                positions=positions,
                forward_batch=batch,
                run_compressor=False,
                run_indexer=True,
            )
        extend.assert_called_once()
        self.assertEqual(captured["q_lens_cpu"], expected_lens)
        self.assertEqual(captured["q_lens"].dtype, torch.int32)
        torch.testing.assert_close(captured["x"], x[:num_local])
        torch.testing.assert_close(captured["q_lora"], q_lora[:num_local])
        torch.testing.assert_close(
            captured["pos"], positions[:num_local].to(torch.int64)
        )
        expected_req = torch.repeat_interleave(
            batch.req_pool_indices.to(torch.int64),
            torch.tensor(expected_lens),
            output_size=num_local,
        )
        torch.testing.assert_close(
            captured["request_positions"],
            torch.stack((expected_req, positions[:num_local].to(torch.int64)), dim=1),
        )

    def test_cp_extend_uses_local_rows_not_global_prompt_length(self):
        self._assert_cp_extend_route([4096], 0, [512])

    def test_cp_extend_preserves_unequal_request_and_zero_row_lengths(self):
        self._assert_cp_extend_route([1, 2, 13], 7, [0, 0, 2])

    def test_cp_extend_supports_rank_with_no_query_rows(self):
        self._assert_cp_extend_route([1, 2], 7, [0, 0])

    def test_cp_late_extend_uses_local_tail_lengths(self):
        tail = SimpleNamespace(
            local_lens_cpu=[1, 0, 2],
            extend_seq_lens=torch.tensor([8, 0, 16], dtype=torch.int32),
            extend_seq_lens_cpu=[8, 0, 16],
            req_global=torch.tensor([7, 9], dtype=torch.int64),
            pos_global=torch.tensor([30, 40], dtype=torch.int64),
        )
        self._assert_cp_extend_route([16, 8, 24], 0, [1, 0, 2], tail=tail)

    def test_non_cp_extend_preserves_batch_and_late_tail_lengths(self):
        tail = SimpleNamespace(
            extend_seq_lens=torch.tensor([2, 1], dtype=torch.int32),
            extend_seq_lens_cpu=[2, 1],
        )
        for active_tail in (None, tail):
            with self.subTest(late_tail=active_tail is not None):
                backend, batch, layer = self._indexer_fixture([3, 5], tail=active_tail)
                expected_tensor = (
                    batch.extend_seq_lens
                    if active_tail is None
                    else active_tail.extend_seq_lens
                )
                expected_lens = (
                    batch.extend_seq_lens_cpu
                    if active_tail is None
                    else active_tail.extend_seq_lens_cpu
                )
                rows = sum(expected_lens)
                x = torch.arange(rows * 2).reshape(rows, 2)
                q_lora = torch.arange(rows * 3).reshape(rows, 3)
                pos = torch.arange(rows, dtype=torch.int64)
                captured = {}
                with patch.object(
                    backend,
                    "_low_ratio_index_topk_dense",
                    side_effect=self._dense_capture(batch, captured),
                ):
                    backend._low_ratio_index_topk_extend(layer, x, q_lora, pos, batch)
                self.assertIs(captured["q_lens"], expected_tensor)
                self.assertEqual(captured["q_lens_cpu"], expected_lens)
                self.assertIs(captured["x"], x)
                self.assertIs(captured["q_lora"], q_lora)
                self.assertIs(captured["pos"], pos)

    def test_cp_torch_fallback_preserves_request_ownership(self):
        backend, batch, layer = self._indexer_fixture([1, 2, 13])
        x = torch.arange(8).reshape(4, 2)
        q_lora = torch.arange(12).reshape(4, 3)
        positions = torch.tensor([7, 15, -1, -1], dtype=torch.int32)
        with (
            patch.object(
                deepseek_v4_backend,
                "get_parallel",
                return_value=SimpleNamespace(attn_cp_rank=7, attn_cp_size=8),
            ),
            patch.object(backend, "_use_dense_fp4_prefill_indexer", return_value=False),
            patch.object(backend, "_low_ratio_index_topk_extend") as extend,
            patch.object(backend, "_low_ratio_index_topk_dense") as dense,
            patch.object(backend, "_low_ratio_index_topk_torch") as fallback,
        ):
            backend._forward_low_ratio_sources_cp(
                layer=layer,
                x=x,
                q_lora=q_lora,
                positions=positions,
                forward_batch=batch,
                run_compressor=False,
                run_indexer=True,
            )
        extend.assert_not_called()
        dense.assert_not_called()
        args, kwargs = fallback.call_args
        self.assertIs(args[0], layer)
        torch.testing.assert_close(args[1], x[:2])
        torch.testing.assert_close(args[2], q_lora[:2])
        torch.testing.assert_close(args[3], torch.tensor([9, 9], dtype=torch.int64))
        torch.testing.assert_close(args[4], torch.tensor([7, 15], dtype=torch.int64))
        torch.testing.assert_close(
            kwargs["req_order"], batch.req_pool_indices.to(torch.int64)
        )

    def _candidate_fixture(self, q_lens_cpu=None, lc_per_req=None):
        q_lens_cpu = [2, 0, 3] if q_lens_cpu is None else q_lens_cpu
        lc_per_req = [32768, 0, 65536] if lc_per_req is None else lc_per_req
        backend, _, _ = self._indexer_fixture(q_lens_cpu)
        indexer = SimpleNamespace(
            uses_candidates=True,
            is_candidate_source=False,
            candidate_block_size=8,
            candidate_topk_blocks=2048,
            n_heads=32,
            n_local_heads=32,
        )
        blocks = [
            (
                torch.arange(2048, dtype=torch.int32).repeat(rows, 1)
                if rows and lc
                else torch.empty(rows, 0, dtype=torch.int32)
            )
            for rows, lc in zip(q_lens_cpu, lc_per_req)
        ]
        backend.forward_metadata.candidate_metadata = (
            deepseek_v4_backend.CandidateMasks(request_block_ids=blocks)
        )
        return backend, indexer, lc_per_req, q_lens_cpu, blocks

    def _candidate_route(self, backend, indexer, lc_per_req, q_lens_cpu):
        with (
            patch.object(deepseek_v4_backend, "_is_hcu", True),
            patch.object(
                deepseek_v4_backend,
                "_hcu_candidate_logits_available",
                return_value=True,
            ),
            patch.dict(os.environ, {"SGLANG_HCU_OPT_DSV41_CANDIDATE_LOGITS": "1"}),
        ):
            return backend._hcu_prefill_candidate_blocks(
                indexer, lc_per_req, q_lens_cpu
            )

    def test_candidate_logits_returns_current_ragged_blocks_without_copy(self):
        backend, indexer, lc_per_req, q_lens_cpu, blocks = self._candidate_fixture()
        selected = self._candidate_route(backend, indexer, lc_per_req, q_lens_cpu)
        self.assertIs(selected, blocks)
        self.assertEqual([block.shape[0] for block in selected], [2, 0, 3])
        for actual, expected in zip(selected, blocks):
            self.assertIs(actual, expected)

    def test_candidate_logits_performance_cutoff_uses_compressed_k_length(self):
        for width, enabled in ((32768, False), (65535, False), (65536, True)):
            with self.subTest(width=width):
                backend, indexer, lc_per_req, q_lens_cpu, blocks = (
                    self._candidate_fixture(lc_per_req=[32768, 0, width])
                )
                selected = self._candidate_route(
                    backend, indexer, lc_per_req, q_lens_cpu
                )
                if enabled:
                    self.assertIs(selected, blocks)
                else:
                    self.assertIsNone(selected)

    def test_candidate_logits_defaults_off_and_requires_explicit_opt_in(self):
        backend, indexer, lc_per_req, q_lens_cpu, blocks = self._candidate_fixture()
        attention = ModuleType("lightop.attention")
        attention.fp8_fp4_mqa_logits = SimpleNamespace(supports_candidate_blocks=True)
        with (
            patch.object(deepseek_v4_backend, "_is_hcu", True),
            patch.dict("sys.modules", {"lightop.attention": attention}),
            patch.dict(os.environ),
        ):
            for flag in (None, "0", "invalid", "true", "2", ""):
                with self.subTest(flag=flag):
                    if flag is None:
                        os.environ.pop("SGLANG_HCU_OPT_DSV41_CANDIDATE_LOGITS", None)
                    else:
                        os.environ["SGLANG_HCU_OPT_DSV41_CANDIDATE_LOGITS"] = flag
                    self.assertIsNone(
                        backend._hcu_prefill_candidate_blocks(
                            indexer, lc_per_req, q_lens_cpu
                        )
                    )
            os.environ["SGLANG_HCU_OPT_DSV41_CANDIDATE_LOGITS"] = "1"
            self.assertIs(
                backend._hcu_prefill_candidate_blocks(indexer, lc_per_req, q_lens_cpu),
                blocks,
            )

    def test_candidate_logits_old_wheel_without_marker_uses_dense(self):
        backend, indexer, lc_per_req, q_lens_cpu, _ = self._candidate_fixture()
        attention = ModuleType("lightop.attention")
        attention.fp8_fp4_mqa_logits = SimpleNamespace()
        with (
            patch.object(deepseek_v4_backend, "_is_hcu", True),
            patch.dict("sys.modules", {"lightop.attention": attention}),
            patch.dict(os.environ, {"SGLANG_HCU_OPT_DSV41_CANDIDATE_LOGITS": "1"}),
        ):
            self.assertIsNone(
                backend._hcu_prefill_candidate_blocks(indexer, lc_per_req, q_lens_cpu)
            )

    def test_candidate_logits_preserves_dense_for_unsupported_routes(self):
        cases = (
            ("non_hcu", {}, False, True, [32768, 0, 65536]),
            ("no_capability", {}, True, False, [32768, 0, 65536]),
            (
                "no_candidates",
                {"uses_candidates": False},
                True,
                True,
                [32768, 0, 65536],
            ),
            ("publisher", {"is_candidate_source": True}, True, True, [32768, 0, 65536]),
            ("block_size", {"candidate_block_size": 16}, True, True, [32768, 0, 65536]),
            ("local_heads", {"n_local_heads": 16}, True, True, [32768, 0, 65536]),
            ("full_candidate_window", {}, True, True, [16384, 0, 8192]),
            ("short_context", {}, True, True, [4096, 0, 8192]),
        )
        for name, updates, is_hcu, available, lengths in cases:
            with self.subTest(case=name):
                backend, indexer, _, q_lens_cpu, _ = self._candidate_fixture()
                for key, value in updates.items():
                    setattr(indexer, key, value)
                with (
                    patch.object(deepseek_v4_backend, "_is_hcu", is_hcu),
                    patch.object(
                        deepseek_v4_backend,
                        "_hcu_candidate_logits_available",
                        return_value=available,
                    ),
                    patch.dict(
                        os.environ, {"SGLANG_HCU_OPT_DSV41_CANDIDATE_LOGITS": "1"}
                    ),
                ):
                    self.assertIsNone(
                        backend._hcu_prefill_candidate_blocks(
                            indexer, lengths, q_lens_cpu
                        )
                    )

    def test_candidate_logits_missing_or_invalid_block_metadata_uses_dense(self):
        backend, indexer, lc_per_req, q_lens_cpu, blocks = self._candidate_fixture()
        invalid_metadata = (
            None,
            SimpleNamespace(request_block_ids=blocks),
            deepseek_v4_backend.CandidateMasks(),
            deepseek_v4_backend.CandidateMasks(request_block_ids=blocks[:-1]),
        )
        for metadata in invalid_metadata:
            with self.subTest(metadata=type(metadata).__name__):
                backend.forward_metadata.candidate_metadata = metadata
                self.assertIsNone(
                    self._candidate_route(backend, indexer, lc_per_req, q_lens_cpu)
                )

    def test_candidate_logits_invalid_block_tensor_uses_dense(self):
        invalid_blocks = (
            torch.zeros(1, 2048, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
            torch.zeros(2, 2048, dtype=torch.int64),
            torch.zeros(2, 4096, dtype=torch.int32)[:, ::2],
        )
        for invalid in invalid_blocks:
            with self.subTest(shape=tuple(invalid.shape), dtype=invalid.dtype):
                backend, indexer, lc_per_req, q_lens_cpu, blocks = (
                    self._candidate_fixture()
                )
                blocks[0] = invalid
                self.assertIsNone(
                    self._candidate_route(backend, indexer, lc_per_req, q_lens_cpu)
                )

    def test_candidate_logits_accepts_empty_width_and_row_strided_views(self):
        backend, indexer, lc_per_req, q_lens_cpu, blocks = self._candidate_fixture(
            [2, 0, 3], [0, 32768, 65536]
        )
        self.assertEqual(tuple(blocks[0].shape), (2, 0))
        self.assertEqual(tuple(blocks[1].shape), (0, 0))
        blocks[2] = torch.arange(6 * 2048, dtype=torch.int32).reshape(6, 2048)[::2]
        selected = self._candidate_route(backend, indexer, lc_per_req, q_lens_cpu)
        self.assertIs(selected, blocks)
        self.assertIs(selected[2], blocks[2])
        self.assertEqual(selected[2].stride(), (4096, 1))

    def test_candidate_logits_does_not_reuse_previous_batch_blocks(self):
        backend, indexer, lc_per_req, q_lens_cpu, first_blocks = (
            self._candidate_fixture()
        )
        self.assertIs(
            self._candidate_route(backend, indexer, lc_per_req, q_lens_cpu),
            first_blocks,
        )
        second_blocks = [block.clone() for block in first_blocks]
        backend.forward_metadata.candidate_metadata = (
            deepseek_v4_backend.CandidateMasks(request_block_ids=second_blocks)
        )
        self.assertIs(
            self._candidate_route(backend, indexer, lc_per_req, q_lens_cpu),
            second_blocks,
        )
        backend.forward_metadata.candidate_metadata = None
        self.assertIsNone(
            self._candidate_route(backend, indexer, lc_per_req, q_lens_cpu)
        )

    def _consume_candidate_fixture(self):
        q_lens_cpu, lc_per_req = [2, 0, 1], [17, 0, 11]
        backend, indexer, _, _, _ = self._candidate_fixture(q_lens_cpu, lc_per_req)
        blocks = [
            torch.tensor([[0, 2], [1, 2]], dtype=torch.int32),
            torch.empty(0, 0, dtype=torch.int32),
            torch.tensor([[1, -1]], dtype=torch.int32),
        ]
        masks = [
            self._expand_candidate_blocks_reference(block, width)
            for block, width in zip(blocks, lc_per_req)
        ]
        backend.forward_metadata.candidate_metadata = (
            deepseek_v4_backend.CandidateMasks(
                request_masks=masks, request_block_ids=blocks
            )
        )
        compress_lens = torch.tensor([9, 17, 10], dtype=torch.int32)
        columns = torch.arange(20)[None, :]
        scores = torch.arange(60, dtype=torch.float32).reshape(3, 20)
        dense = torch.where(columns < compress_lens[:, None], scores, -torch.inf)
        candidate = dense.clone()
        start = 0
        for rows, width, mask in zip(q_lens_cpu, lc_per_req, masks):
            candidate[start : start + rows, :width] = torch.where(
                mask, candidate[start : start + rows, :width], -torch.inf
            )
            start += rows
        return (
            backend, indexer, lc_per_req, q_lens_cpu, compress_lens, dense, candidate
        )

    def test_candidate_consumer_skips_mask_and_preserves_topk_padding(self):
        backend, indexer, lengths, rows, lens, dense, candidate = (
            self._consume_candidate_fixture()
        )
        empty = torch.zeros(0, 0, dtype=torch.bool)
        backend._publish_or_consume_candidates(
            indexer, dense, lens, lengths, rows, empty
        )
        metadata = backend.forward_metadata.candidate_metadata
        with patch.object(torch.Tensor, "masked_fill_", side_effect=AssertionError):
            backend._publish_or_consume_candidates(
                indexer,
                candidate,
                lens,
                lengths,
                rows,
                empty,
                logits_already_masked=True,
            )
        self.assertIs(backend.forward_metadata.candidate_metadata, metadata)
        torch.testing.assert_close(candidate, dense)
        offsets = torch.tensor([0, 0, 17], dtype=torch.int32)
        selected = candidate.topk(12, dim=-1).indices.to(torch.int32) + offsets[:, None]
        actual = deepseek_v4_backend.mask_topk_scores(candidate, selected, offsets)
        expected = deepseek_v4_backend.mask_topk_scores(dense, selected, offsets)
        torch.testing.assert_close(actual, expected)
        self.assertEqual((actual == -1).sum(dim=1).tolist(), [4, 3, 10])

    def test_candidate_consumer_dense_fallback_still_masks_with_flag_one(self):
        for flag in (None, "0", "1"):
            with self.subTest(flag=flag), patch.dict(os.environ):
                if flag is None:
                    os.environ.pop("SGLANG_HCU_OPT_DSV41_CANDIDATE_LOGITS", None)
                else:
                    os.environ["SGLANG_HCU_OPT_DSV41_CANDIDATE_LOGITS"] = flag
                backend, indexer, lengths, rows, lens, dense, expected = (
                    self._consume_candidate_fixture()
                )
                backend._publish_or_consume_candidates(
                    indexer,
                    dense,
                    lens,
                    lengths,
                    rows,
                    torch.zeros(0, 0, dtype=torch.bool),
                )
                torch.testing.assert_close(dense, expected)
                self.assertTrue(torch.isneginf(dense[0, 8]))
                self.assertTrue(torch.isneginf(dense[1, :8]).all())

    def test_dense_indexer_propagates_candidate_guard_and_preserves_indices(self):
        backend, batch, _ = self._indexer_fixture([1, 0, 1])
        backend.forward_metadata.candidate_metadata = (
            deepseek_v4_backend.CandidateMasks(
                request_masks=[
                    torch.tensor([[True] * 8 + [False] * 3]),
                    torch.empty(0, 0, dtype=torch.bool),
                    torch.arange(65536)[None, :] >= 65528,
                ],
                request_block_ids=[
                    torch.tensor([[0]], dtype=torch.int32),
                    torch.empty(0, 0, dtype=torch.int32),
                    torch.tensor([[8191]], dtype=torch.int32),
                ],
            )
        )
        pages = torch.empty(2, 12, dtype=torch.int32)
        raw = torch.empty_like(pages)
        backend.forward_metadata.core_metadata = SimpleNamespace(
            sparse_page_indices=lambda ratio: pages,
            sparse_raw_indices=lambda ratio: raw,
        )
        backend.req_to_token = torch.arange(65536)[None, :].expand(10, -1)
        backend.token_to_kv_pool = SimpleNamespace(
            get_low_ratio_index_k_fp4=lambda layer, slots: (slots, slots)
        )
        indexer = SimpleNamespace(
            uses_candidates=True,
            is_candidate_source=False,
            candidate_block_size=8,
            candidate_topk_blocks=2048,
            n_local_heads=32,
            index_topk=12,
            queries=lambda q, freq: torch.zeros(2, 32, 128, dtype=torch.bfloat16),
            head_weights=lambda x: torch.ones(2, 32),
        )
        layer = SimpleNamespace(
            compress_ratio=1,
            layer_id=24,
            indexer=indexer,
            freqs_cis=torch.zeros(65536),
        )
        batch.seq_lens_cpu = [11, 0, 65536]
        pos = torch.tensor([1, 65535], dtype=torch.int64)
        results = []
        for flag, available in (("0", True), ("1", True), ("1", False)):

            def logits_boundary(q, k, weights, ks, ke, width, **kwargs):
                logits = torch.full((2, width), -torch.inf)
                logits[0, :2] = torch.arange(2, dtype=torch.float32)
                logits[1, :] = torch.arange(width, dtype=torch.float32)
                if kwargs["candidate_blocks"] is not None:
                    logits[0, 2:11] = -torch.inf
                    logits[1, :65528] = -torch.inf
                return logits

            with (
                patch.object(deepseek_v4_backend, "_is_hcu", True),
                patch.object(
                    deepseek_v4_backend,
                    "_hcu_candidate_logits_available",
                    return_value=available,
                ),
                patch.object(
                    deepseek_v4_backend,
                    "_hcu_dense_fp4_mqa_logits",
                    side_effect=logits_boundary,
                ),
                patch.object(
                    backend,
                    "_publish_or_consume_candidates",
                    wraps=backend._publish_or_consume_candidates,
                ) as publish,
                patch.dict(os.environ, {"SGLANG_HCU_OPT_DSV41_CANDIDATE_LOGITS": flag}),
            ):
                backend._low_ratio_index_topk_dense(
                    layer,
                    torch.zeros(2, 1),
                    torch.zeros(2, 1),
                    pos,
                    batch,
                    batch.extend_seq_lens,
                    batch.extend_seq_lens_cpu,
                )
            self.assertEqual(
                publish.call_args.kwargs["logits_already_masked"],
                flag == "1" and available,
            )
            results.append((pages.clone(), raw.clone()))
        for result in results[1:]:
            for actual, expected in zip(result, results[0]):
                torch.testing.assert_close(actual, expected)
        self.assertEqual(raw[0].tolist(), [0, 1] + [-1] * 10)
        self.assertEqual(raw[1].tolist(), list(range(65528, 65536)) + [-1] * 4)

    def _publish_candidate_fixture(
        self, lc_per_req, enabled, *, logits_already_masked=False
    ):
        q_lens_cpu = [2, 0, 1]
        backend, indexer, _, _, _ = self._candidate_fixture(q_lens_cpu, lc_per_req)
        indexer.layer_id = 20
        indexer.is_candidate_source = True
        indexer.uses_candidates = False
        width = (max(lc_per_req) + 3) // 4 * 4
        logits = torch.arange(3 * width, dtype=torch.float32).reshape(3, width)
        compress_lens = torch.tensor(
            [min(10, lc_per_req[0]), lc_per_req[0], lc_per_req[2]], dtype=torch.int32
        )
        with (
            patch.object(deepseek_v4_backend, "_is_hcu", True),
            patch.object(
                deepseek_v4_backend,
                "_hcu_candidate_logits_available",
                return_value=True,
            ),
            patch.dict(os.environ),
        ):
            flag = "1" if enabled is True else "0" if enabled is False else enabled
            if flag is None:
                os.environ.pop("SGLANG_HCU_OPT_DSV41_CANDIDATE_LOGITS", None)
            else:
                os.environ["SGLANG_HCU_OPT_DSV41_CANDIDATE_LOGITS"] = flag
            backend._publish_or_consume_candidates(
                indexer,
                logits,
                compress_lens,
                lc_per_req,
                q_lens_cpu,
                torch.zeros(0, 0, dtype=torch.bool),
                logits_already_masked=logits_already_masked,
            )
        return backend.forward_metadata.candidate_metadata, logits

    def test_candidate_source_never_skips_publication_or_causal_mask(self):
        expected, expected_logits = self._publish_candidate_fixture([17, 0, 11], False)
        candidate, logits = self._publish_candidate_fixture(
            [17, 0, 11], False, logits_already_masked=True
        )
        torch.testing.assert_close(logits, expected_logits)
        for mask, baseline in zip(candidate.request_masks, expected.request_masks):
            torch.testing.assert_close(mask, baseline)
        self.assertTrue(torch.isneginf(logits[0, 10:17]).all())

    def _expand_candidate_blocks_reference(self, blocks, width):
        mask = torch.zeros(blocks.shape[0], width, dtype=torch.bool)
        row_ids = torch.arange(blocks.shape[0])[:, None].expand_as(blocks)
        for offset in range(8):
            positions = blocks.to(torch.int64) * 8 + offset
            valid = (blocks >= 0) & (positions < width)
            mask[row_ids[valid], positions[valid]] = True
        return mask

    def test_candidate_publisher_blocks_match_masks_and_clip_partial_tail(self):
        lc_per_req = [65537, 0, 65539]
        candidate, logits = self._publish_candidate_fixture(lc_per_req, True)
        baseline, _ = self._publish_candidate_fixture(lc_per_req, False)
        self.assertIsNotNone(candidate.request_block_ids)
        self.assertIsNone(baseline.request_block_ids)
        for blocks, mask, expected in zip(
            candidate.request_block_ids,
            candidate.request_masks,
            baseline.request_masks,
        ):
            torch.testing.assert_close(mask, expected)
            torch.testing.assert_close(
                self._expand_candidate_blocks_reference(blocks, mask.shape[1]), mask
            )
        self.assertEqual(tuple(candidate.request_masks[1].shape), (0, 0))
        self.assertEqual(tuple(candidate.request_block_ids[1].shape), (0, 0))
        self.assertTrue(torch.isneginf(logits[0, 10:65537]).all())
        self.assertEqual((candidate.request_block_ids[0][0] >= 0).sum().item(), 2)
        last_block = (lc_per_req[2] - 1) // 8
        self.assertTrue((candidate.request_block_ids[2][0] == last_block).any())
        self.assertTrue(candidate.request_masks[2][0, -3:].all())

    def test_candidate_publisher_respects_enable_and_performance_cutoff(self):
        cases = (
            ([65537, 0, 65539], None, False),
            ([65537, 0, 65539], False, False),
            ([65537, 0, 65539], "invalid", False),
            ([65537, 0, 65539], "true", False),
            ([65537, 0, 65539], "2", False),
            ([65537, 0, 65539], "", False),
            ([4096, 0, 8192], True, False),
            ([32768, 0, 32768], True, False),
            ([32768, 0, 65535], True, False),
            ([32768, 0, 65536], True, True),
        )
        for lengths, enabled, retained in cases:
            with self.subTest(lengths=lengths, enabled=enabled):
                candidate, _ = self._publish_candidate_fixture(lengths, enabled)
                baseline, _ = self._publish_candidate_fixture(lengths, False)
                if retained:
                    self.assertIsNotNone(candidate.request_block_ids)
                    for blocks, mask in zip(
                        candidate.request_block_ids, candidate.request_masks
                    ):
                        torch.testing.assert_close(
                            self._expand_candidate_blocks_reference(
                                blocks, mask.shape[1]
                            ),
                            mask,
                        )
                else:
                    self.assertIsNone(candidate.request_block_ids)
                for mask, expected in zip(
                    candidate.request_masks, baseline.request_masks
                ):
                    torch.testing.assert_close(mask, expected)

    def test_candidate_publisher_empty_kv_keeps_query_shaped_block_placeholder(self):
        candidate, _ = self._publish_candidate_fixture([0, 0, 65539], True)
        self.assertEqual(tuple(candidate.request_masks[0].shape), (0, 0))
        self.assertEqual(tuple(candidate.request_block_ids[0].shape), (2, 0))
        self.assertEqual(tuple(candidate.request_block_ids[1].shape), (0, 0))

    def test_enter_late_layer_tail_crops_candidate_masks_and_blocks_together(self):
        prefix_lens = [2, 1, 4]
        tail_lens = [3, 2, 0]
        blocks = [
            torch.arange((prefix + tail) * 2, dtype=torch.int32).reshape(-1, 2)
            + request * 16
            for request, (prefix, tail) in enumerate(zip(prefix_lens, tail_lens))
        ]
        masks = [self._expand_candidate_blocks_reference(block, 384) for block in blocks]
        full_candidate = deepseek_v4_backend.CandidateMasks(
            request_masks=masks, request_block_ids=blocks
        )
        for use_cp in (False, True):
            with self.subTest(use_cp=use_cp):
                backend = deepseek_v4_backend.DeepseekV4AttnBackend.__new__(
                    deepseek_v4_backend.DeepseekV4AttnBackend
                )
                cp_metadata = (
                    SimpleNamespace(per_rank_actual_token=[5] * 8) if use_cp else None
                )
                tail = SimpleNamespace(
                    cp_metadata=cp_metadata,
                    local_lens_cpu=tail_lens,
                    # CP must select local rows, not these global tail lengths.
                    extend_seq_lens_cpu=[24, 16, 0] if use_cp else tail_lens,
                )
                full_metadata = SimpleNamespace(
                    candidate_metadata=full_candidate, core_attn_metadata=object()
                )
                backend.forward_metadata = full_metadata
                backend.tail_forward_metadata = SimpleNamespace(
                    late_layer_tail=tail,
                    candidate_metadata=None,
                    core_attn_metadata=SimpleNamespace(low_ratios=[]),
                )
                backend.token_to_kv_pool = SimpleNamespace(request_window=None)
                batch = SimpleNamespace(attn_cp_metadata=object())
                previous_cp = batch.attn_cp_metadata
                with (
                    patch.object(
                        deepseek_v4_backend, "get_local_dp_buffer_len", return_value=23
                    ),
                    patch.object(deepseek_v4_backend, "set_local_dp_buffer_len") as set_len,
                ):
                    saved = backend.enter_late_layer_tail(batch)

                self.assertIs(saved[0], full_metadata)
                self.assertIs(saved[1], previous_cp)
                self.assertEqual(saved[2], 23)
                self.assertIs(backend.forward_metadata, backend.tail_forward_metadata)
                cropped = backend.forward_metadata.candidate_metadata
                for request, prefix in enumerate(prefix_lens):
                    torch.testing.assert_close(
                        cropped.request_block_ids[request], blocks[request][prefix:]
                    )
                    torch.testing.assert_close(
                        cropped.request_masks[request], masks[request][prefix:]
                    )
                    torch.testing.assert_close(
                        self._expand_candidate_blocks_reference(
                            cropped.request_block_ids[request], 384
                        ),
                        cropped.request_masks[request],
                    )
                    self.assertEqual(
                        cropped.request_block_ids[request].shape[0], tail_lens[request]
                    )
                self.assertEqual(tuple(cropped.request_masks[2].shape), (0, 384))
                self.assertEqual(tuple(cropped.request_block_ids[2].shape), (0, 2))
                scores = torch.arange(5 * 384, dtype=torch.float32).reshape(5, 384)
                scores = torch.where(
                    torch.cat(cropped.request_masks), scores, -torch.inf
                )
                expected_scores = scores.clone()
                with patch.object(
                    torch.Tensor, "masked_fill_", side_effect=AssertionError
                ):
                    backend._publish_or_consume_candidates(
                        SimpleNamespace(is_candidate_source=False),
                        scores,
                        torch.full((5,), 384, dtype=torch.int32),
                        [384] * 3,
                        tail_lens,
                        torch.zeros(0, 0, dtype=torch.bool),
                        logits_already_masked=True,
                    )
                torch.testing.assert_close(scores, expected_scores)
                self.assertIs(full_metadata.candidate_metadata, full_candidate)
                if use_cp:
                    self.assertIs(batch.attn_cp_metadata, cp_metadata)
                    set_len.assert_called_once_with(40)
                else:
                    self.assertIs(batch.attn_cp_metadata, previous_cp)
                    set_len.assert_not_called()


if __name__ == "__main__":
    unittest.main()
