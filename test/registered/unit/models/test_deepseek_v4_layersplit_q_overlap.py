"""CPU dependency-order regression for HCU LayerSplit prefetch/Q overlap."""
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models import deepseek_v4 as model
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import env_gate
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _AttentionReached(Exception):
    pass


class TestLayerSplitQOverlap(CustomTestCase):
    def _run(self, *, hcu=True, v41=True, ratio=1, cp=True, split=True,
             graph=False, source=True, fused_a=False, empty=False):
        events, pending = [], {}
        rows = 0 if empty else 4
        x = torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8)
        positions = torch.arange(rows)
        expected_q = ((x + 1) * 2).reshape(rows, 2, 4)
        mode = SimpleNamespace(
            is_context_parallel_extend=lambda: split,
            is_decode_or_idle=lambda: False,
            is_target_verify=lambda: False,
            is_extend=lambda: True,
        )
        batch = SimpleNamespace(forward_mode=mode, positions=positions)
        pool = SimpleNamespace()

        def store(**kwargs):
            events.append("kv_store")
            pending["kv_ready"] = True

        pool.set_swa_key_buffer_radix_fused_norm_rope = store

        def sources(**kwargs):
            self.assertTrue(pending.get("kv_ready"))
            self.assertTrue(torch.equal(kwargs["q_lora"], (x + 1) * 2))
            events.append("sources")

        def fetch(kind):
            def run(*args):
                self.assertTrue(pending.get("kv_ready"))
                if source and ratio in (1, 2):
                    self.assertIn("sources", events)
                self.assertNotIn("prefetch_" + kind, events)
                events.append("prefetch_" + kind)
                pending[kind] = True
            return run

        def wait_swa(*args):
            self.assertTrue(pending.get("swa"))
            pending["swa"] = False
            events.append("wait_swa")

        def q_b(q, positions, out):
            events.append("q")
            result = q.reshape(rows, 2, 4).clone()
            if out is not None:
                out.copy_(result)
                result = out
            return result

        def attention(**kwargs):
            if split:
                self.assertFalse(pending["swa"])
                if ratio in (1, 2):
                    self.assertTrue(pending["extra"])
            torch.testing.assert_close(kwargs["q"], expected_q)
            events.append("attention")
            raise _AttentionReached

        backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(late_layer_tail=None),
            get_swa_out_cache_loc=lambda batch: torch.arange(rows),
            forward_low_ratio_sources=sources, forward=attention,
            store_cache=store, low_ratio_prefill_graph=False,
        )
        layer = SimpleNamespace(
            layer_id=8, is_dsv41=v41, compress_ratio=ratio,
            dsa_enable_prefill_cp=cp, alt_streams=None,
            fuse_wqa_wkv=fused_a, q_lora_rank=8, use_fused_qk_norm_rope=False,
            n_local_heads=2, head_dim=4, eps=1e-6,
            attn_mqa=SimpleNamespace(layer_id=8, v_head_dim=4),
            freqs_cis=torch.empty(0),
            kv_norm=SimpleNamespace(weight=torch.ones(4)),
            wq_a=lambda value: (value + 1, None),
            wqkv_a=lambda value: (torch.cat((value + 1, value[:, :4] * 3), dim=1), None),
            wkv=lambda value: (value[:, :4] * 3, None),
            _normalize_q_lora=lambda value: (value * 2, value * 2),
            _compute_q_b=q_b,
            _compute_kv_bf16=lambda value, positions, **kwargs: value[:, :4] * 3,
            _compute_kv_to_cache=lambda *args, **kwargs: store(),
            _kernel_num_heads=lambda rows: 2,
            _local_attn_sink=lambda heads: torch.zeros(heads),
            compressor=object() if source and ratio in (1, 2) else None,
            indexer=object() if source and ratio in (1, 2) else None,
        )
        layer._forward_prepare = lambda *args, **kwargs: model.MQALayer._forward_prepare(
            layer, *args, **kwargs
        )
        with ExitStack() as stack:
            for name, value in {
                "_is_hcu": hcu, "_is_cuda": False, "_is_hip": True, "_is_npu": False,
                "get_token_to_kv_pool": lambda: pool,
                "get_attn_backend": lambda: backend,
                "get_attn_tp_context": lambda: SimpleNamespace(input_scattered=False),
                "is_cp_cache_layer_split_deepseek_v4_pool": lambda pool: split,
                "dsa_use_prefill_cp": lambda batch: cp,
                "cp_materialize_global_token_order": lambda value, *args: value,
                "is_in_breakable_cuda_graph": lambda: graph,
                "bcg_deepseek_v4_low_ratio_sources": lambda *args: sources(q_lora=(x + 1) * 2),
                "bcg_deepseek_v4_attention_with_output": lambda q, *args: attention(q=q),
                "maybe_prefetch_cp_kv_swa": fetch("swa"),
                "maybe_prefetch_cp_kv_extra": fetch("extra"),
                "maybe_wait_cp_kv_swa_prefetch": wait_swa,
            }.items():
                stack.enter_context(patch.object(model, name, value))
            stack.enter_context(patch.object(env_gate, "is_unified_kv_triton", return_value=False))
            stack.enter_context(patch.object(env_gate, "is_unified_kv_fp8", return_value=False))
            stack.enter_context(patch.object(torch.cuda, "current_stream", return_value=object()))
            if empty and not split:
                self.assertIs(model.MQALayer.forward(layer, x, positions, batch), x)
            elif empty:
                # Empty ranks still participate in both fetches and the SWA wait.
                self.assertIs(model.MQALayer.forward(layer, x, positions, batch), x)
            else:
                with self.assertRaises(_AttentionReached):
                    model.MQALayer.forward(layer, x, positions, batch)
        return events

    def test_q_is_between_prefetch_and_wait_for_sources_and_consumers(self):
        for ratio in (1, 2):
            for source in (False, True):
                for fused_a in (False, True):
                    with self.subTest(ratio=ratio, source=source, fused_a=fused_a):
                        events = self._run(ratio=ratio, source=source, fused_a=fused_a)
                        expected = ["kv_store"] + (["sources"] if source else [])
                        expected += ["prefetch_swa", "prefetch_extra", "q", "wait_swa", "attention"]
                        self.assertEqual(events, expected)

    def test_other_paths_keep_q_before_kv_and_do_not_fetch_twice(self):
        for overrides in ({"hcu": False}, {"v41": False}, {"ratio": 0},
                          {"cp": False}, {"split": False}, {"graph": True}):
            with self.subTest(overrides=overrides):
                events = self._run(**overrides)
                self.assertEqual(events[:2], ["q", "kv_store"])
                self.assertEqual(events.count("q"), 1)
                self.assertEqual(events.count("prefetch_swa"), int(overrides.get("split", True)))
                self.assertEqual(events.count("prefetch_extra"), int(
                    overrides.get("split", True) and overrides.get("ratio", 1) in (1, 2)))

    def test_empty_rank_keeps_collective_participation(self):
        self.assertEqual(self._run(empty=True),
                         ["kv_store", "sources", "prefetch_swa", "prefetch_extra", "q", "wait_swa"])
        self.assertEqual(self._run(empty=True, split=False), [])


if __name__ == "__main__":
    unittest.main()
