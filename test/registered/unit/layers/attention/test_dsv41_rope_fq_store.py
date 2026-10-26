"""Prefill compression routing and LayerSplit ownership, without GPU imports."""

import ast
import enum
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[5]
FLAG = "SGLANG_HCU_OPT_DSV41_ROPE_FQ_STORE"


def load_method(path, name, namespace):
    nodes = [node for node in ast.walk(ast.parse(path.read_text()))
             if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(nodes) == 1, (path, name)
    future = ast.ImportFrom(module="__future__",
                            names=[ast.alias(name="annotations")], level=0)
    module = ast.Module(body=[future, *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class KVLayout(enum.Enum):
    V4 = 1
    V41 = 2
    V41_FP4 = 3


class MetadataOnlyHCU(torch.Tensor):
    """Real CPU values, with only the dispatch device predicate overridden."""

    @property
    def is_cuda(self):
        return True


class TestDSV41RopeFQStore(CustomTestCase):
    def setUp(self):
        self.latent = torch.arange(1024, dtype=torch.bfloat16).reshape(2, 512)
        self.latent = self.latent.as_subclass(MetadataOnlyHCU)
        self.freqs = torch.ones(2, 32, dtype=torch.complex64)
        self.slots = torch.tensor([1, 3])
        self.positions = torch.tensor([0, 1])
        self.writes = []
        self.rope_calls = []
        self.layout = KVLayout.V4
        self.owns = True

        def old_rope(x, freqs, dim, *, compressed_kv):
            self.rope_calls.append((x.clone(), freqs.clone(), dim, compressed_kv))
            return x + 2

        self.namespace = dict(torch=torch, os=os, KVLayout=KVLayout,
                              _is_hcu=True, _rope_fq4=old_rope)
        self.write = load_method(
            ROOT / "python/sglang/srt/layers/attention/deepseek_v4_backend.py",
            "_low_ratio_write_group", self.namespace)
        setter = load_method(
            ROOT / "python/sglang/srt/mem_cache/cp_cache_layer_split/deepseek_v4_pool.py",
            "set_extra_key_buffer_fused", dict(torch=torch))

        def store(layer, loc, values, valid_mask, freqs):
            self.writes.append((values, freqs, loc.clone()))

        item = SimpleNamespace(compress_layer_id=0,
                               compress_kv_pool=SimpleNamespace(set_key_buffer_fused=store))
        self.pool = SimpleNamespace(
            uniform_fp8=False, _unified_kv=False, dtype=torch.float8_e4m3fn,
            compression_ratios=[1], _extra_full_read_key="old-forward",
            get_extra_key_layout=lambda layer: self.layout,
            _owns_extra_key_layer_id=lambda layer: self.owns,
            _owner_compress_layer_item=lambda layer: item)
        self.pool.set_extra_key_buffer_fused = setter.__get__(self.pool)
        self.backend = SimpleNamespace(token_to_kv_pool=self.pool)
        self.layer = SimpleNamespace(layer_id=0, indexer=None, rope_head_dim=64,
                                     compressor=SimpleNamespace(finish=lambda x: self.latent),
                                     freqs_cis=self.freqs)

    def run_writer(self, **kwargs):
        self.write(self.backend, self.layer, None, self.slots, self.positions, **kwargs)

    def test_prefill_forwards_unrotated_latent_without_mutating_it(self):
        before = self.latent.clone()
        with patch.dict(os.environ, {FLAG: "1"}):
            self.run_writer(fuse_rope_fq_store=True)
        self.assertEqual(self.rope_calls, [])
        self.assertIs(self.writes[0][0], self.latent)
        torch.testing.assert_close(self.writes[0][1], self.freqs)
        torch.testing.assert_close(self.writes[0][2], self.slots)
        torch.testing.assert_close(self.latent, before)
        self.assertIsNone(self.pool._extra_full_read_key)

    def test_decode_and_disabled_flag_keep_old_round_trip(self):
        for flag, kwargs in (("1", {}), ("0", dict(fuse_rope_fq_store=True))):
            with self.subTest(flag=flag, kwargs=kwargs), patch.dict(os.environ, {FLAG: flag}):
                self.writes.clear()
                self.rope_calls.clear()
                self.run_writer(**kwargs)
                self.assertEqual(len(self.rope_calls), 1)
                self.assertIsNone(self.writes[0][1])
                torch.testing.assert_close(self.writes[0][0], self.latent + 2)

    def test_unsupported_pools_and_platform_keep_old_path(self):
        for target, name, value in ((self.pool, "uniform_fp8", True),
                                    (self.pool, "_unified_kv", True),
                                    (self.pool, "dtype", torch.bfloat16),
                                    (self.layer, "rope_head_dim", 32)):
            old = getattr(target, name)
            try:
                setattr(target, name, value)
                self.writes.clear()
                self.rope_calls.clear()
                with patch.dict(os.environ, {FLAG: "1"}):
                    self.run_writer(fuse_rope_fq_store=True)
                self.assertEqual(len(self.rope_calls), 1)
                self.assertIsNone(self.writes[0][1])
            finally:
                setattr(target, name, old)
        self.rope_calls.clear()
        self.namespace["_is_hcu"] = False
        with patch.dict(os.environ, {FLAG: "1"}):
            self.run_writer(fuse_rope_fq_store=True)
        self.assertEqual(len(self.rope_calls), 1)

    def test_nonowner_invalidates_staging_without_launching_store_or_rope(self):
        self.owns = False
        with patch.dict(os.environ, {FLAG: "1"}):
            self.run_writer(fuse_rope_fq_store=True)
        self.assertIsNone(self.pool._extra_full_read_key)
        self.assertEqual(self.writes, [])
        self.assertEqual(self.rope_calls, [])

    def test_index_keys_still_receive_pre_rope_values(self):
        seen = []
        self.layer.indexer = SimpleNamespace(
            owns_k=True, index_keys=lambda x, freqs: seen.append(x.clone()) or x)
        self.pool.set_index_k_fp4 = lambda **kwargs: None
        with patch.dict(os.environ, {FLAG: "1"}):
            self.run_writer(fuse_rope_fq_store=True)
        torch.testing.assert_close(seen[0], self.latent)
        self.assertIs(self.writes[0][0], self.latent)

    def test_physical_fp4_layout_is_unchanged(self):
        self.layout = KVLayout.V41_FP4
        with patch.dict(os.environ, {FLAG: "0"}):
            self.run_writer()
        self.assertEqual(self.rope_calls, [])
        self.assertIs(self.writes[0][0], self.latent)
        torch.testing.assert_close(self.writes[0][1], self.freqs)

    def test_precomputed_cp_norm_is_not_applied_twice(self):
        normalized = self.latent + 4
        self.layer.compressor.finish = Mock(side_effect=AssertionError("norm applied twice"))
        seen = []
        self.layer.indexer = SimpleNamespace(
            owns_k=True, index_keys=lambda x, freqs: seen.append(x.clone()) or x)
        self.pool.set_index_k_fp4 = lambda **kwargs: None
        with patch.dict(os.environ, {FLAG: "1"}):
            self.run_writer(fuse_rope_fq_store=True, normalized_latent=normalized)
        torch.testing.assert_close(seen[0], normalized)
        self.assertIs(self.writes[0][0], normalized)
        self.layer.compressor.finish.assert_not_called()


class TestDSV41CPPoolNorm(CustomTestCase):
    def test_prefill_cp_only_keeps_pairing_slots_and_normalized_values(self):
        for case in ("enabled", "flag_off", "non_cp", "decode", "other_platform", "bf16"):
            with self.subTest(case=case):
                self.check_case(case)

    def check_case(self, case):
        torch.manual_seed(7)
        dtype = torch.bfloat16 if case == "bf16" else torch.float32
        values = [torch.randn(4, 512).to(dtype).as_subclass(MetadataOnlyHCU) for _ in range(4)]
        previous, current, previous_score, score = values
        before = [v.clone() for v in values]
        weight = torch.rand(512) + .5
        eps = 1e-6
        def norm(pooled):
            x = pooled.bfloat16().float()
            return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight).bfloat16()
        def pool(kv, gate):
            return (kv * gate.softmax(1)).sum(1)
        expected = norm(pool(torch.stack((previous, current), 1), torch.stack((previous_score, score), 1)))
        calls, writes = [], []
        def fused(a, b, c, d, w, e):
            self.assertIs(w, weight)
            self.assertEqual(e, eps)
            calls.append(1)
            return norm(pool(torch.stack((a, b), 1), torch.stack((c, d), 1)))
        positions = torch.arange(4)
        requests = torch.tensor([1, 1, 2, 2])
        core = SimpleNamespace(raw_out_loc=torch.tensor([0, 4, 5, 6]),
                               c2_out_loc=torch.tensor([-1, 7, -1, 8]))
        def pairs(**kwargs):
            torch.testing.assert_close(kwargs["pad"], core.raw_out_loc == 0)
            torch.testing.assert_close(kwargs["req"], requests)
            return previous, previous_score
        def write(layer, pooled, slots, group_pos, **kwargs):
            latent = kwargs.get("normalized_latent")
            writes.append((norm(pooled) if latent is None else latent, slots, group_pos))
        backend = SimpleNamespace(forward_metadata=SimpleNamespace(core_metadata=core),
                                  _low_ratio_pair_partners=pairs, _low_ratio_write_group=write)
        layer = SimpleNamespace(layer_id=0, compress_ratio=2, compressor=SimpleNamespace(
            norm=SimpleNamespace(weight=weight, eps=eps), pool_pairs=pool))
        namespace = dict(torch=torch, os=os, _is_hcu=case != "other_platform",
                         get_parallel=lambda: SimpleNamespace(attn_cp_size=1 if case == "non_cp" else 8))
        compress = load_method(ROOT / "python/sglang/srt/layers/attention/deepseek_v4_backend.py",
                               "_low_ratio_compress_torch", namespace)
        flag = "SGLANG_HCU_OPT_DSV41_CP_POOL_NORM"
        with patch.dict(os.environ, {flag: "0" if case == "flag_off" else "1"}), patch.dict(
            sys.modules, {"lightop": SimpleNamespace(dsv41_pool_pairs_norm=fused)}
        ):
            compress(backend, layer, None, requests, positions, projected=(current, score),
                     fuse_rope_fq_store=case != "decode")
        self.assertEqual(len(calls), int(case == "enabled"))
        torch.testing.assert_close(writes[0][0], expected)
        torch.testing.assert_close(writes[0][1], torch.tensor([0, 7, 0, 8]))
        torch.testing.assert_close(writes[0][2], torch.tensor([0, 0, 2, 2]))
        for value, unchanged in zip(values, before):
            torch.testing.assert_close(value, unchanged)


if __name__ == "__main__":
    unittest.main()
