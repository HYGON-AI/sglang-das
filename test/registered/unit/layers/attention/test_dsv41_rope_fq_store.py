"""Prefill compression routing and LayerSplit ownership, without GPU imports."""

import ast
import enum
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
