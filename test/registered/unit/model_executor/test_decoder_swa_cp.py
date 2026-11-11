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
from sglang.srt.runtime_context import ForwardFlags

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
        cp_tail = layout(None, global_batch, global_indices, tail_lengths)
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
            request_window = None

            def get_swa_key_page_size(self):
                return self.swa_kv_pool.page_size

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


class TestLateLayerTailUpload(CustomTestCase):
    def layout(self, torch_module=torch):
        return load(SOURCE / 'kernels/ops/attention/dsv4_attn_metadata_kernels.py',
                    'late_layer_tail_layout', dict(torch=torch_module))

    def test_tail_rows_and_request_window_floors(self):
        cases = ([0], [1], [127], [128], [129], [0, 0], [0, 129, 7],
                 [1, 128, 32768], [65536, 131072, 1000000])
        layout = self.layout()
        for lengths in cases:
            for limit in (1, 2, 128):
                with self.subTest(lengths=lengths, limit=limit):
                    seq = [n + i * 4096 for i, n in enumerate(lengths)]
                    indices, tails, floors = layout(extend_lens_cpu=list(lengths),
                        seq_lens_cpu=seq, tail_len=limit, device=torch.device('cpu'))
                    rows, expected_floors, start = [], [], 0
                    for n, s in zip(lengths, seq):
                        t = min(n, limit)
                        rows.extend(range(start + n - t, start + n))
                        expected_floors.extend([s - t] * t)
                        start += n
                    self.assertEqual(tails, [min(n, limit) for n in lengths])
                    torch.testing.assert_close(indices, torch.tensor(rows, dtype=torch.int64))
                    torch.testing.assert_close(floors, torch.tensor(expected_floors, dtype=torch.int32))

    def test_hcu_lengths_upload_uses_private_pinned_nonblocking_copy(self):
        copies, allocations = [], []

        class HostTensor:
            def __init__(self, value):
                self.value = value

            def to(self, device, *, non_blocking=False):
                self_device.assertEqual(device, torch.device('cuda'))
                self_device.assertTrue(non_blocking)
                copies.append(self.value.clone())
                return self.value

        self_device = self

        def host_tensor(values, **kwargs):
            self.assertEqual(torch.device(kwargs.get('device', 'cpu')).type, 'cpu')
            self.assertTrue(kwargs.get('pin_memory', False))
            value = torch.tensor(values, dtype=kwargs.get('dtype'))
            allocations.append(value)
            return HostTensor(value)

        def cpu_arange(*args, **kwargs):
            kwargs['device'] = 'cpu'
            return torch.arange(*args, **kwargs)

        proxy = SimpleNamespace(tensor=host_tensor, arange=cpu_arange,
            cumsum=torch.cumsum, repeat_interleave=torch.repeat_interleave,
            int64=torch.int64, int32=torch.int32)
        layout = self.layout(proxy)
        for seq in ([4096, 8192, 32768], [4127, 8291, 32999]):
            indices, tails, floors = layout(extend_lens_cpu=[1, 129, 17],
                seq_lens_cpu=seq, tail_len=128, device=torch.device('cuda'))
            self.assertEqual(tails, [1, 128, 17])
            self.assertEqual(indices.numel(), 146)
            self.assertEqual(floors[0].item(), seq[0] - 1)
        self.assertEqual(len(copies), 2)
        self.assertEqual(copies[0].dtype, torch.int64)
        self.assertNotEqual(allocations[0].data_ptr(), allocations[1].data_ptr())
        torch.testing.assert_close(copies[0][2], torch.tensor([4096, 8192, 32768]))
        torch.testing.assert_close(copies[1][2], torch.tensor([4127, 8291, 32999]))


class TestLateLayerTailMetadataLengths(CustomTestCase):
    def test_tail_lengths_reuse_device_extend_lengths(self):
        kernel = ACTUAL / 'kernels/ops/attention/dsv4_attn_metadata_kernels.py'
        helper = load(kernel, 'late_layer_tail_layout', dict(torch=torch))
        backend_path = SOURCE / 'srt/layers/attention/deepseek_v4_backend.py'
        rows = load(backend_path, '_tail_rows', dict(torch=torch))

        def forbid_host_upload(*args, **kwargs):
            raise AssertionError('Tail lengths must reuse device extend lengths')

        modes = [(False, False), (True, False), (False, True)]
        lengths_cases = ([1], [0, 0], [1, 129, 0, 300], [32768] * 8)
        for lengths in lengths_cases:
            for dtype in (torch.int32, torch.int64):
                for legacy_cp, cp_active in modes:
                    with self.subTest(lengths=lengths, dtype=dtype,
                                      legacy_cp=legacy_cp, cp_active=cp_active):
                        expected = torch.tensor([min(n, 128) for n in lengths],
                                                dtype=torch.int32)
                        device_lengths = torch.tensor(lengths, dtype=dtype)
                        before = device_lengths.clone()
                        seq = torch.tensor([n + 4096 * (i + 1)
                                            for i, n in enumerate(lengths)], dtype=torch.int32)
                        batch = SimpleNamespace(extend_seq_lens_cpu=list(lengths),
                            extend_seq_lens=device_lengths, seq_lens_cpu=seq,
                            seq_lens=seq, out_cache_loc=torch.arange(sum(lengths)),
                            req_pool_indices=torch.arange(len(lengths), dtype=torch.int32))
                        cp_calls = []

                        class Captured(Exception):
                            pass

                        def cp_layout(forward_batch, indices, tail_lengths):
                            self.assertIs(forward_batch, batch)
                            self.assertEqual(tail_lengths, expected.tolist())
                            cp_calls.append(True)
                            return {'cp_metadata': 'cp-layout'}

                        def capture(**kwargs):
                            torch.testing.assert_close(kwargs['extend_seq_lens'], expected)
                            torch.testing.assert_close(kwargs['extend_start_loc'],
                                expected.cumsum(0) - expected)
                            self.assertEqual(kwargs['extend_seq_lens_cpu'], expected.tolist())
                            self.assertEqual(kwargs['num_tokens'], int(expected.sum()))
                            self.assertEqual(bool(cp_calls), legacy_cp or cp_active)
                            self.assertEqual(kwargs['cp_metadata'],
                                             'cp-layout' if legacy_cp or cp_active else None)
                            raise Captured

                        namespace = dict(torch=SimpleNamespace(tensor=forbid_host_upload,
                            int32=torch.int32, cumsum=torch.cumsum), SWA_WINDOW=128,
                            late_layer_tail_layout=helper, _tail_rows=rows,
                            is_cp_active=lambda batch: cp_active)
                        build = load(backend_path, '_build_late_layer_tail_metadata',
                                     namespace, 'DeepseekV4AttnBackend')
                        backend = SimpleNamespace(_late_layer_tail_cp_layout=cp_layout,
                                                  init_forward_metadata_prefill=capture)
                        with self.assertRaises(Captured):
                            build(backend, batch, legacy_cp=legacy_cp)
                        torch.testing.assert_close(device_lengths, before)


class TestLateLayerTailRequestIndices(CustomTestCase):
    def test_repeat_interleave_uses_logical_cpu_output_size(self):
        helper = load(ACTUAL / 'kernels/ops/attention/dsv4_attn_metadata_kernels.py',
                      'late_layer_tail_layout', dict(torch=torch))
        path = SOURCE / 'srt/layers/attention/deepseek_v4_backend.py'
        rows = load(path, '_tail_rows', dict(torch=torch))
        cases = ([0], [129], [0, 0], [1, 129, 7], [32768] * 8,
                 [3, 0, 129, 256, 1])
        for lengths in cases:
            for dtype in (torch.int32, torch.int64):
                for legacy_cp, cp_active in ((False, False), (True, False), (False, True)):
                    with self.subTest(lengths=lengths, dtype=dtype,
                                      legacy_cp=legacy_cp, cp_active=cp_active):
                        tails = [min(n, 128) for n in lengths]
                        request_ids = torch.arange(len(lengths), dtype=dtype) * 17 + 9
                        seq = torch.tensor([n + (i + 1) * 4096
                                            for i, n in enumerate(lengths)], dtype=torch.int32)
                        batch = SimpleNamespace(extend_seq_lens_cpu=list(lengths),
                            extend_seq_lens=torch.tensor(lengths, dtype=torch.int32),
                            seq_lens_cpu=seq, seq_lens=seq,
                            out_cache_loc=torch.arange(sum(lengths)),
                            req_pool_indices=request_ids)

                        class Captured(Exception):
                            pass

                        def repeat(input, repeats, **kwargs):
                            self.assertEqual(kwargs.get('output_size'), sum(tails))
                            self.assertIsInstance(kwargs['output_size'], int)
                            result = torch.repeat_interleave(input, repeats, **kwargs)
                            expected = torch.tensor([int(req) for req, n in
                                zip(request_ids, tails) for _ in range(n)], dtype=torch.int64)
                            torch.testing.assert_close(result, expected)
                            raise Captured

                        cp_metadata = SimpleNamespace(per_rank_actual_token=[
                            (sum(tails) + 7) // 8] * 8)
                        backend = SimpleNamespace(
                            _late_layer_tail_cp_layout=lambda *args: {'cp_metadata': cp_metadata},
                            init_forward_metadata_prefill=lambda **kwargs:
                                SimpleNamespace(core_attn_metadata=SimpleNamespace()),
                            token_to_kv_pool=SimpleNamespace(request_window=None,
                                translate_loc_from_full_to_swa=lambda value: value))
                        proxy = SimpleNamespace(int32=torch.int32, int64=torch.int64,
                            cumsum=torch.cumsum, repeat_interleave=repeat)
                        ns = dict(torch=proxy, SWA_WINDOW=128, _tail_rows=rows,
                            late_layer_tail_layout=helper, is_cp_active=lambda batch: cp_active)
                        build = load(path, '_build_late_layer_tail_metadata', ns,
                                     'DeepseekV4AttnBackend')
                        with self.assertRaises(Captured):
                            build(backend, batch, legacy_cp=legacy_cp)


class TestLateLayerTailCPCounts(CustomTestCase):
    def test_cpu_counts_match_token_ownership_without_gpu_readback(self):
        path = SOURCE / 'srt/layers/attention/deepseek_v4_backend.py'
        cases = ([0], [1], [7], [128], [129], [32768] * 8,
                 [1, 129, 0, 300, 17], [3, 0, 129, 256, 1])
        for lengths in cases:
            for tail_limit in (1, 7, 128):
                tails = [min(n, tail_limit) for n in lengths]
                indices_list = []
                offset = 0
                for length, tail in zip(lengths, tails):
                    offset += length
                    indices_list.extend(range(offset - tail, offset))
                indices = torch.tensor(indices_list, dtype=torch.int64)
                positions = torch.arange(sum(lengths) * 2, dtype=torch.int64)[::2]
                for size in (1, 2, 3, 8):
                    counts = [sum(i % size == r for i in indices_list)
                              for r in range(size)]
                    physical = max(counts)
                    slot = [0] * size
                    expected_gather = []
                    for i in indices_list:
                        rank = i % size
                        expected_gather.append(rank * physical + slot[rank])
                        slot[rank] += 1
                    for rank in range(size):
                        with self.subTest(lengths=lengths, tail_limit=tail_limit,
                                          size=size, rank=rank):
                            def forbid(*args, **kwargs):
                                raise AssertionError('CP tail must not use dynamic counts/readback')

                            def static_nonzero(value, *, size):
                                self.assertEqual(size, counts[rank])
                                return torch.nonzero_static(value, size=size)

                            proxy = SimpleNamespace(**{
                                name: getattr(torch, name) for name in
                                ('argsort', 'tensor', 'int64', 'empty_like', 'arange', 'cat')})
                            proxy.nonzero_static = static_nonzero
                            proxy.bincount = forbid
                            proxy.repeat_interleave = forbid
                            layout = load(path, '_late_layer_tail_cp_layout',
                                dict(torch=proxy, get_parallel=lambda:
                                     SimpleNamespace(attn_cp_rank=rank, attn_cp_size=size),
                                     InterleaveContextParallelMetadata=SimpleNamespace),
                                'DeepseekV4AttnBackend')
                            batch = SimpleNamespace(batch_size=len(lengths),
                                extend_seq_lens_cpu=list(lengths), positions=positions)
                            with patch.object(torch.Tensor, 'tolist', side_effect=forbid), \
                                 patch.object(torch.Tensor, 'item', side_effect=forbid):
                                out = layout(None, batch, indices, tails)
                            rows = [j for j, i in enumerate(indices_list) if i % size == rank]
                            expected_local_lens = []
                            cursor = 0
                            for tail in tails:
                                expected_local_lens.append(sum(cursor <= row < cursor + tail
                                                               for row in rows))
                                cursor += tail
                            pad = physical - len(rows)
                            pad_start = len(indices_list) + sum(physical - c for c in counts[:rank])
                            metadata = out['cp_metadata']
                            self.assertEqual(metadata.per_rank_logical_token, counts)
                            self.assertEqual(metadata.per_rank_actual_token, [physical] * size)
                            self.assertEqual(metadata.total_seq_lens, len(indices_list))
                            self.assertEqual(out['pad_rows'], pad)
                            self.assertEqual(out['local_lens_cpu'], expected_local_lens)
                            torch.testing.assert_close(out['local_lens'],
                                torch.tensor(expected_local_lens, dtype=torch.int64))
                            torch.testing.assert_close(metadata.gather_index,
                                torch.tensor(expected_gather, dtype=torch.int64))
                            torch.testing.assert_close(metadata.local_index,
                                torch.tensor(rows + list(range(pad_start, pad_start + pad)),
                                             dtype=torch.int64))
                            torch.testing.assert_close(out['local_token_indices'],
                                torch.tensor([(indices_list[j] - rank) // size for j in rows],
                                             dtype=torch.int64))
                            expected_positions = [int(positions[indices_list[j]]) for j in rows]
                            torch.testing.assert_close(out['local_positions'],
                                torch.tensor(expected_positions + [0] * pad, dtype=torch.int64))



class TestTailMoEDispatchScope(CustomTestCase):
    def exercise(self, *, enabled=True, hcu=True, cp=True, deepep=True,
                 tail_rows=(128,) * 8, fail=False):
        flags = ForwardFlags()
        seen = []
        output = torch.empty((1, 8))

        def mlp(*args, **kwargs):
            seen.append(flags.deepep_tail_max_tokens)
            if fail:
                raise RuntimeError("MLP failed")
            return output

        backend = SimpleNamespace(
            is_none=lambda: False, is_deepep=lambda: deepep,
            is_megamoe=lambda: not deepep, is_mori=lambda: False)
        tail = (SimpleNamespace(cp_metadata=SimpleNamespace(
            per_rank_actual_token=tail_rows)) if tail_rows is not None else None)
        namespace = dict(
            torch=torch, _is_hcu=hcu, dsa_use_prefill_cp=lambda batch: cp,
            is_cp_active=lambda batch: cp,
            get_parallel=lambda: SimpleNamespace(attn_dp_size=1, attn_tp_size=1),
            get_moe_a2a_backend=lambda: backend, _SHARED_EXPERT_LOCAL=False,
            envs=SimpleNamespace(
                SGLANG_DP_USE_REDUCE_SCATTER=SimpleNamespace(get=lambda: False),
                SGLANG_HCU_DSV41_TAIL_MOE_LOW_LATENCY=SimpleNamespace(get=lambda: enabled)),
            nullcontext=contextlib.nullcontext, get_forward=lambda: flags,
            get_attn_backend=lambda: SimpleNamespace(forward_metadata=SimpleNamespace(
                late_layer_tail=tail)))
        method = load(SOURCE / 'srt/models/deepseek_v4.py', '_run_moe_ffn_dp_sync',
                      namespace, 'DeepseekV4DecoderLayer')
        call = lambda: method(SimpleNamespace(mlp=mlp), output, SimpleNamespace(),
                              input_ids=None, input_ids_global=None)
        if fail:
            with self.assertRaisesRegex(RuntimeError, "MLP failed"):
                call()
        else:
            self.assertIs(call(), output)
        self.assertIsNone(flags.deepep_tail_max_tokens)
        self.assertFalse(flags.is_extend_in_batch)
        return seen

    def test_tail_uses_cp_wide_bound_not_local_tensor_rows(self):
        self.assertEqual(self.exercise(tail_rows=(16, 128, 48, 64, 0, 80, 96, 112)), [128])

    def test_full_prefill_and_other_paths_do_not_select_tail(self):
        for kwargs in (dict(enabled=False), dict(hcu=False), dict(cp=False),
                       dict(deepep=False), dict(tail_rows=None)):
            with self.subTest(**kwargs):
                self.assertEqual(self.exercise(**kwargs), [None])

    def test_tail_scope_restores_after_exception(self):
        self.assertEqual(self.exercise(fail=True), [128])




class TestHcuTailGraphLiveState(CustomTestCase):
    def runner(self):
        tree = ast.parse((SOURCE / 'srt/model_executor/runner/prefill_cuda_graph_runner.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                    and n.name == 'HcuDSV41TailGraphRunner')
        ns = dict(torch=torch)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                     '<tail graph>', 'exec'), ns)
        ll = SimpleNamespace(num_max_dispatch_tokens_per_rank=128,
                             quant_config={"hcu_w4a8_int8_dispatch": True},
                             _get_buffer=lambda: None)
        layer = SimpleNamespace(engram=None, self_attn=SimpleNamespace(attn_tp_size=1),
            mlp=SimpleNamespace(experts=SimpleNamespace(
                dispatcher=SimpleNamespace(_low_latency_dispatcher=ll))))
        model = SimpleNamespace(late_layer_start=0, end_layer=1, layers=[layer])
        with patch.object(torch.cuda, 'graph_pool_handle', return_value=(0, 0)):
            return ns['HcuDSV41TailGraphRunner'](model)

    def invoke(self, runner, *, rows=16, pre=None, cp=True):
        metadata = SimpleNamespace(per_rank_actual_token=[rows] * 8,
            per_rank_logical_token=[rows] * 8)
        tail = SimpleNamespace(cp_metadata=metadata if cp else None,
                               extend_seq_lens_cpu=[128])
        hidden = torch.ones(rows, 2)
        state = SimpleNamespace(streams=hidden, residual=hidden, pre=pre, input=None)
        batch = SimpleNamespace(request_id=object())
        import sys
        modules = {
            'sglang.srt.layers.moe.token_dispatcher.deepep': ModuleType('deepep'),
            'sglang.srt.models': ModuleType('models'),
        }
        modules['sglang.srt.layers.moe.token_dispatcher.deepep'].DeepEPBuffer = object()
        modules['sglang.srt.models'].deepseek_v4_mhc = SimpleNamespace(
            HcState=lambda residual, pre: SimpleNamespace(residual=residual, pre=pre))
        ids = torch.arange(rows)
        with patch.dict(sys.modules, modules), \
             patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False):
            return runner.run(state, ids, ids, ids, batch, tail, False, None)

    def test_warmup_and_oversized_layout_stay_eager(self):
        runner = self.runner()
        self.assertIsNone(self.invoke(runner))
        self.assertEqual(len(runner._warmed), 1)
        self.assertIsNone(self.invoke(runner, rows=129))
        self.assertEqual(len(runner._warmed), 1)
        self.assertIsNone(self.invoke(runner, cp=False))
        self.assertIsNone(runner.forward_batch)

    def test_capture_stub_does_not_write_attention_caches(self):
        fn = load(SOURCE / 'srt/models/deepseek_v4.py',
                  '_hcu_tail_graph_attention_stub', dict(torch=torch))
        x = torch.ones(4, 8)
        layer = SimpleNamespace(maybe_use_decode_attn_tp=lambda _: self.fail('cache write'))
        out = fn(None, layer, x, None, None, False)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)
        self.assertNotEqual(out.data_ptr(), x.data_ptr())

    def test_single_dispatcher_wrapper_is_supported(self):
        runner = self.runner()
        experts = runner.model.layers[0].mlp.experts
        experts.dispatcher = SimpleNamespace(_inners=[experts.dispatcher])
        self.assertIsNone(self.invoke(runner))
        self.assertEqual(len(runner._warmed), 1)

    def test_sbo_wrapper_stays_eager(self):
        runner = self.runner()
        experts = runner.model.layers[0].mlp.experts
        experts.dispatcher = SimpleNamespace(_inners=[experts.dispatcher] * 2)
        self.assertIsNone(self.invoke(runner))
        self.assertFalse(runner._warmed)

    def test_non_int8_dispatch_stays_eager(self):
        runner = self.runner()
        runner.model.layers[0].mlp.experts.dispatcher._low_latency_dispatcher.quant_config = {}
        self.assertIsNone(self.invoke(runner))
        self.assertFalse(runner._warmed)

    def test_graph_capacity_falls_back_before_capture(self):
        runner = self.runner()
        runner._entries = {n: object() for n in range(4)}
        self.assertIsNone(self.invoke(runner))
        self.assertFalse(runner._warmed)

    def test_exception_releases_live_batch_reference(self):
        runner = self.runner()
        self.invoke(runner)
        ll = runner.model.layers[0].mlp.experts.dispatcher._low_latency_dispatcher
        def fail():
            self.assertIsNotNone(runner.forward_batch)
            raise RuntimeError('layout prepare failed')
        ll._get_buffer = fail
        with self.assertRaisesRegex(RuntimeError, 'layout prepare failed'):
            self.invoke(runner)
        self.assertIsNone(runner.forward_batch)

    def test_attention_break_reads_each_live_batch(self):
        fn = load(SOURCE / 'srt/models/deepseek_v4.py',
                  '_hcu_tail_graph_attention', {})
        seen = []
        class Layer:
            def maybe_use_decode_attn_tp(self, batch):
                seen.append(('context', batch))
                return contextlib.nullcontext()
            def __call__(self, **kwargs):
                seen.append(('forward', kwargs['forward_batch']))
                return kwargs['x']
        runner = SimpleNamespace(forward_batch=object())
        first = runner.forward_batch
        x = torch.ones(2, 4)
        self.assertIs(fn(runner, Layer(), x, None, None, False), x)
        runner.forward_batch = object()
        fn(runner, Layer(), x, None, None, False)
        self.assertEqual(seen, [('context', first), ('forward', first),
                               ('context', runner.forward_batch),
                               ('forward', runner.forward_batch)])


if __name__ == '__main__':
    unittest.main()
