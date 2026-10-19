"""CPU control-flow regression: real CP gather/layout bodies, mocked collective.

This does not replace distributed service or model accuracy validation.
"""
import ast
import contextlib
from copy import copy
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ACTUAL = Path(os.environ.get('SWA_REFERENCE_SOURCE',
    str(Path(__file__).resolve().parents[4] / 'python' / 'sglang')))
SOURCE = Path(os.environ.get('SWA_TEST_SOURCE', str(ACTUAL)))


def load(path, name, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace[name]


class TestDecoderTailCP(CustomTestCase):
    def exercise(self, lengths, rank, compact=True, packed=True):
        total = sum(lengths)
        global_batch = SimpleNamespace(batch_size=len(lengths), positions=torch.arange(total),
            input_ids=torch.arange(total),
            forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
            extend_seq_lens_cpu=lengths, extend_seq_lens=torch.tensor(lengths))
        tail_layout = load(ACTUAL / 'kernels/ops/attention/dsv4_attn_metadata_kernels.py',
                           'late_layer_tail_layout', dict(torch=torch))
        global_indices, tail_lengths, _ = tail_layout(extend_lens_cpu=lengths,
            seq_lens_cpu=lengths, tail_len=128, device=torch.device('cpu'))
        layout = load(SOURCE / 'srt/layers/attention/deepseek_v4_backend.py',
            '_late_layer_tail_cp_layout', dict(torch=torch, get_parallel=lambda:
                SimpleNamespace(attn_cp_rank=rank, attn_cp_size=8),
                InterleaveContextParallelMetadata=SimpleNamespace), 'DeepseekV4AttnBackend')
        cp_tail = layout(None, global_batch, global_indices, torch.tensor(tail_lengths))
        full_len = (total + 7) // 8
        full_metadata = SimpleNamespace(total_seq_lens=total,
            per_rank_actual_token=[full_len] * 8,
            per_rank_logical_token=[len(range(r, total, 8)) for r in range(8)],
            gather_index=torch.tensor([i % 8 * full_len + i // 8 for i in range(total)]))
        global_batch.attn_cp_metadata = full_metadata
        indices = global_indices if compact else torch.arange(total)
        metadata = cp_tail['cp_metadata'] if compact else full_metadata
        physical = max(metadata.per_rank_actual_token)
        local_ids = indices[indices % 8 == rank]
        hidden = torch.zeros(physical, 1)
        hidden[:len(local_ids), 0] = local_ids.float()
        aux = hidden + 100000
        tail = SimpleNamespace(cp_metadata=metadata, global_token_indices=global_indices,
            extend_seq_lens_cpu=tail_lengths, extend_seq_lens=torch.tensor(tail_lengths))
        gather_number = [0]

        def collective(output, value):
            bias = 100000 if gather_number[0] else 0
            expected_local = local_ids.float() + bias
            torch.testing.assert_close(value[:len(local_ids), 0], expected_local)
            output.zero_()
            for owner in range(8):
                owner_ids = indices[indices % 8 == owner]
                output[owner * physical:owner * physical + len(owner_ids), 0] = owner_ids.float() + bias
            gather_number[0] += 1

        gather = load(ACTUAL / 'srt/layers/cp/interleave.py', '_gather_interleaved_tensor',
            dict(torch=torch, use_symmetric_memory=lambda *a, **k: contextlib.nullcontext(),
                 get_parallel=lambda: SimpleNamespace(attn_cp_group=None),
                 is_allocation_symmetric=lambda: False,
                 attn_cp_all_gather_into_tensor=collective), 'InterleaveCPStrategy')

        def gather_cp(value, batch, stream):
            return gather(SimpleNamespace(cp_rank=rank, cp_size=8), value, batch)

        def logits(ids, states, head, batch, auxiliary, **kwargs):
            torch.testing.assert_close(ids, indices)
            torch.testing.assert_close(states[:, 0], indices.float())
            for tensor in [auxiliary] if packed else auxiliary:
                torch.testing.assert_close(tensor[:, 0], indices.float() + 100000)
            expected_lengths = tail_lengths if compact else lengths
            self.assertEqual(batch.extend_seq_lens_cpu, expected_lengths)
            last_rows = torch.cumsum(torch.tensor(expected_lengths), 0) - 1
            return SimpleNamespace(hidden_states=auxiliary,
                selected_last_ids=ids[last_rows], hidden_states_token_indices=None)

        class Body:
            late_layer_start = 21 if compact else None

            def __call__(self, *args, **kwargs):
                return (hidden, hidden), aux if packed else [aux]

        model = SimpleNamespace(model=Body(), capture_aux_hidden_states=True,
            pp_group=SimpleNamespace(is_last_rank=True), lm_head=None,
            get_input_embeddings=lambda: lambda ids: ids[:, None].float(), logits_processor=logits)

        @contextlib.contextmanager
        def shard(embeddings, positions, batch, ids):
            yield embeddings[rank::8], positions[rank::8], ids[rank::8]

        run = load(SOURCE / 'srt/model_executor/runner/eager_runner.py', '_execute_extend_cp',
                   dict(torch=torch, copy=copy, cp_shard_model_inputs=shard,
                        cp_gather_after_forward=gather_cp), 'EagerRunner')
        module = ModuleType('sglang.srt.layers.logits_processor')
        module.LogitsMetadata = SimpleNamespace(from_forward_batch=lambda batch: copy(batch))
        runner = SimpleNamespace(model_runner=SimpleNamespace(model=model,
            attn_backend=SimpleNamespace(tail_forward_metadata=SimpleNamespace(late_layer_tail=tail))))
        with patch.dict('sys.modules', {module.__name__: module}), patch.object(torch.cuda, 'current_stream', return_value=None):
            result = run(runner, global_batch, {})
        expected_last = torch.cumsum(torch.tensor(lengths), 0) - 1
        torch.testing.assert_close(result.selected_last_ids, expected_last)
        if compact:
            torch.testing.assert_close(result.hidden_states_token_indices, global_indices)
        else:
            self.assertIsNone(result.hidden_states_token_indices)
        self.assertIs(global_batch.attn_cp_metadata, full_metadata)
        self.assertEqual(global_batch.extend_seq_lens_cpu, lengths)

    def test_compact_tail_and_packed_dspark(self):
        for lengths in ([32768], [4096], [17, 4097, 3], [1]):
            for rank in range(8):
                with self.subTest(lengths=lengths, rank=rank):
                    self.exercise(lengths, rank)

    def test_list_aux_capture(self):
        for rank in range(8):
            self.exercise([129, 7, 255], rank, packed=False)

    def test_disabled_unchanged(self):
        for rank in range(8):
            self.exercise([129, 7, 255], rank, compact=False)

    def test_hcu_tail_uses_window_floors(self):
        class Pool:
            swa_kv_pool = SimpleNamespace(page_size=128)

            def get_swa_key_buffer_radix(self, layer_id):
                return torch.zeros(1, 128 * 528, dtype=torch.uint8)

            def get_swa_key_bytes_per_token(self):
                return 528

        indices = torch.arange(128).repeat(16, 1)
        indices[:, :90] = -1
        lengths = torch.full((16,), 38, dtype=torch.int32)
        core = SimpleNamespace(swa_page_indices=indices, swa_topk_lengths=lengths,
                               get_flashmla_metadata=lambda ratio: None)
        mode = SimpleNamespace(is_decode_or_idle=lambda: False,
            is_target_verify=lambda: False, is_prefill=lambda **kwargs: True)
        namespace = dict(torch=torch, _is_hcu=True, DSV4AttnMetadata=SimpleNamespace,
            DeepSeekV4TokenToKVPool=Pool, is_cp_cache_layer_split_deepseek_v4_pool=lambda pool: False,
            _should_use_sparse_prefill=lambda *a, **k: True,
            envs=SimpleNamespace(SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=SimpleNamespace(get=lambda: True)))
        attention = load(SOURCE / 'srt/layers/attention/deepseek_v4_backend.py',
                         '_forward_attention', namespace, 'DeepseekV4AttnBackend')
        prefill = load(SOURCE / 'srt/layers/attention/deepseek_v4_backend.py',
                       '_forward_flash_mla_prefill', dict(), 'DeepseekV4AttnBackend')
        q = torch.zeros(16, 8, 512)

        def decode(**kwargs):
            # The existing page-index route must preserve the per-query floor.
            torch.testing.assert_close(kwargs['swa_page_indices'], indices)
            torch.testing.assert_close(kwargs['swa_topk_lengths'], lengths)
            visible = indices[0][indices[0] >= 0].float().mean()
            return torch.full_like(q, visible)

        for tail in (None, object()):
            backend = SimpleNamespace(mtp_enabled=False, trtllm_attn=False, is_dsv41=True,
                token_to_kv_pool=Pool(),
                forward_metadata=SimpleNamespace(core_attn_metadata=core, late_layer_tail=tail),
                _forward_prefill_sparse=lambda **kwargs: torch.full_like(q, -1),
                _forward_flash_mla_decode=decode)
            backend._forward_flash_mla_prefill = lambda **kwargs: prefill(backend, **kwargs)
            result = attention(backend, q, None, None,
                SimpleNamespace(layer_id=21, v_head_dim=512),
                SimpleNamespace(forward_mode=mode), 0, save_kv_cache=False,
                attn_sink=torch.zeros(8))
            expected = torch.full_like(q, -1 if tail is None else 108.5)
            torch.testing.assert_close(result, expected)


class TestDecoderReplayPadding(CustomTestCase):
    def check_padding(self, lengths, cp_active=False, replay=True, prefixes=None):
        prefixes = prefixes or [0] * len(lengths)
        seq_lengths = [p + n for p, n in zip(prefixes, lengths)]
        namespace = dict(torch=torch)
        kernel_path = ACTUAL / 'kernels/ops/attention/dsv4_attn_metadata_kernels.py'
        tail_layout = load(kernel_path, 'late_layer_tail_layout', namespace)
        indices, tail_lens, floor = tail_layout(
            extend_lens_cpu=lengths, seq_lens_cpu=seq_lengths,
            tail_len=128, device=torch.device('cpu'))
        namespace['ExpandPrefillCausallyResult'] = SimpleNamespace
        expand = load(kernel_path, 'expand_prefill_causally', namespace)
        namespace['ExpandPrefillCausally'] = SimpleNamespace(execute=expand)
        namespace['is_cp_active'] = lambda batch: cp_active
        backend_path = SOURCE / 'srt/layers/attention/deepseek_v4_backend.py'
        tail_rows = load(backend_path, '_tail_rows', namespace)
        physical = (sum(lengths) + 7) // 8 * 8
        out_loc = torch.nn.functional.pad(
            torch.arange(1, sum(lengths) + 1), (0, physical - sum(lengths)))
        out_loc = tail_rows(out_loc, token_indices=indices,
            contiguous_start=lengths[0] - tail_lens[0] if len(lengths) == 1 else None)
        tail_total = sum(tail_lens)
        metadata_rows = (tail_total + 7) // 8 * 8 if cp_active else len(out_loc)
        cp_metadata = SimpleNamespace(per_rank_actual_token=[metadata_rows // 8] * 8)
        forward_batch = SimpleNamespace(attn_cp_metadata=cp_metadata)
        expand_method = load(backend_path, 'expand_prefill_casually', namespace,
                             'DeepseekV4AttnBackend')
        prefill = load(backend_path, 'init_forward_metadata_prefill', namespace,
                       'DeepseekV4AttnBackend')

        class MetadataCaptured(Exception):
            pass

        def capture(**kwargs):
            self.assertEqual(kwargs['seq_lens_casual'].numel(), metadata_rows)
            actual_floor = kwargs['swa_replay_start']
            if replay:
                self.assertEqual(actual_floor.numel(), metadata_rows)
                torch.testing.assert_close(actual_floor[:tail_total], floor)
                self.assertEqual(torch.count_nonzero(actual_floor[tail_total:]).item(), 0)
                # The operation that failed in make_core_attn_metadata.
                visible = kwargs['seq_lens_casual'] - actual_floor
                expected = torch.cat([torch.arange(1, n + 1) for n in tail_lens])
                torch.testing.assert_close(visible[:tail_total], expected.to(visible.dtype))
                if metadata_rows == tail_total:
                    self.assertIs(actual_floor, floor)
            else:
                self.assertIsNone(actual_floor)
            raise MetadataCaptured

        backend = SimpleNamespace(req_to_token=None, make_core_attn_metadata=capture)
        backend.expand_prefill_casually = lambda **kw: expand_method(backend, **kw)
        tail_tensor = torch.tensor(tail_lens, dtype=torch.int32)
        with self.assertRaises(MetadataCaptured):
            prefill(backend, max_seq_len=max(seq_lengths),
                req_pool_indices=torch.arange(len(lengths), dtype=torch.int32),
                seq_lens=torch.tensor(seq_lengths, dtype=torch.int32),
                seq_lens_cpu=seq_lengths, out_cache_loc=out_loc,
                num_tokens=tail_total, extend_seq_lens=tail_tensor,
                extend_seq_lens_cpu=tail_lens,
                extend_start_loc=tail_tensor.cumsum(0) - tail_tensor,
                forward_batch=forward_batch,
                swa_replay_start=floor if replay else None)

    def test_short_hcu_fallback_and_non_aligned_tails(self):
        for length in (1, 4, 8, 9, 127, 128, 129, 131, 32768):
            with self.subTest(length=length):
                self.check_padding([length], prefixes=[37])

    def test_generic_cp_padding_is_preserved(self):
        for lengths in ([9], [131], [17, 4097, 3]):
            with self.subTest(lengths=lengths):
                self.check_padding(lengths, cp_active=True)

    def test_multiple_requests_preserve_distinct_window_floors(self):
        self.check_padding([5, 129, 3], prefixes=[0, 4096, 100])

    def test_without_replay_is_unchanged(self):
        self.check_padding([4], replay=False)


if __name__ == '__main__':
    unittest.main()
