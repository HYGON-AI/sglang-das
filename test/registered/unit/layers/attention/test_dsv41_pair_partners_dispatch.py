"""Pair-partner dispatch, with the native GPU boundary replaced on CPU."""

import ast
import dataclasses
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

FLAG = "SGLANG_HCU_OPT_DSV41_PAIR_PARTNERS"


def load_pair_method():
    # Avoid importing unrelated accelerator-only backend dependencies on CPU CI.
    root = Path(__file__).resolve().parents[5]
    backend = root / "python/sglang/srt/layers/attention/deepseek_v4_backend.py"
    state = root / "python/sglang/srt/mem_cache/deepseek_v4_compress_state.py"
    method = next(
        node for node in ast.walk(ast.parse(backend.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "_low_ratio_pair_partners"
    )
    state_ast = ast.parse(state.read_text())
    pair = next(node for node in state_ast.body if isinstance(node, ast.ClassDef) and node.name == "KVAndScore")
    pool = next(node for node in state_ast.body if isinstance(node, ast.ClassDef) and node.name == "CompressStatePool")
    pool.body = [node for node in pool.body if isinstance(node, ast.FunctionDef) and node.name in (
        "translate_from_req_position_to_state_loc", "get_state_by_state_loc", "set_state_by_state_loc"
    )]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = dict(torch=torch, os=os, dataclasses=dataclasses, _is_hcu=False)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, pair, pool, method], type_ignores=[])), str(backend), "exec"), namespace)
    return namespace


class TestDSV41PairPartnersDispatch(CustomTestCase):
    def setUp(self):
        self.ns = load_pair_method()
        self.state = self.ns["CompressStatePool"]()
        self.state.ring_size = 2
        self.packed = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        self.state.kv_score_buffer = self.ns["KVAndScore"](self.packed)
        self.owner = SimpleNamespace(token_to_kv_pool=SimpleNamespace(
            get_attention_compress_states=lambda _: self.state
        ))
        self.inputs = dict(layer_id=3, kv=torch.tensor([[1., 2.], [3., 4.], [5., 6.]]),
                           score=torch.tensor([[11., 12.], [13., 14.], [15., 16.]]),
                           req=torch.zeros(3, dtype=torch.int32), pos=torch.tensor([3, 4, 5]),
                           pad=torch.zeros(3, dtype=torch.bool))

    def run_method(self):
        return self.ns["_low_ratio_pair_partners"](self.owner, **self.inputs)

    def assert_torch_result(self, output):
        torch.testing.assert_close(output[0], torch.tensor([[0., 1.], [1., 2.], [3., 4.]]))
        torch.testing.assert_close(output[1], torch.tensor([[2., 3.], [11., 12.], [13., 14.]]))
        torch.testing.assert_close(self.packed[0], torch.tensor([3., 4., 13., 14.]))
        torch.testing.assert_close(self.packed[1], torch.tensor([5., 6., 15., 16.]))
        torch.testing.assert_close(self.packed[-1], torch.tensor([0., 0., -torch.inf, -torch.inf]))

    def test_non_hcu_uses_original_even_with_flag(self):
        with patch.dict(os.environ, {FLAG: "1"}), patch.dict("sys.modules", {"lightop.attention": None}):
            self.assert_torch_result(self.run_method())

    def test_hcu_flag_off_uses_original(self):
        self.ns["_is_hcu"] = True
        with patch.dict(os.environ, {FLAG: "0"}), patch.dict("sys.modules", {"lightop.attention": None}):
            self.assert_torch_result(self.run_method())

    def test_hcu_default_routes_to_fused(self):
        with patch.dict(os.environ):
            os.environ.pop(FLAG, None)
            self._assert_fused_result()

    def test_hcu_routes_packed_state_and_preserves_results(self):
        with patch.dict(os.environ, {FLAG: "1"}):
            self._assert_fused_result()

    def _assert_fused_result(self):
        self.ns["_is_hcu"] = True

        def native(kv, score, req, pos, pad, packed, ring):
            self.assertIs(packed, self.packed)
            self.assertIs(req, self.inputs["req"])
            self.assertIs(pos, self.inputs["pos"])
            self.assertIs(pad, self.inputs["pad"])
            self.assertEqual(ring, 2)
            first = packed[0].clone()
            out_kv = torch.cat((first[:2].unsqueeze(0), kv[:-1]))
            out_score = torch.cat((first[2:].unsqueeze(0), score[:-1]))
            packed[:2] = torch.cat((kv[1:], score[1:]), dim=1)
            packed[-1, :2] = 0
            packed[-1, 2:] = -torch.inf
            return out_kv, out_score

        with patch.dict("sys.modules", {
            "lightop.attention": SimpleNamespace(dsv41_pair_partners=native)
        }):
            self.assert_torch_result(self.run_method())

    def test_missing_native_is_not_silently_ignored(self):
        self.ns["_is_hcu"] = True
        before = self.packed.clone()
        for flag in (None, "1"):
            with self.subTest(flag=flag), patch.dict(os.environ), patch.dict("sys.modules", {"lightop.attention": SimpleNamespace()}):
                if flag is None:
                    os.environ.pop(FLAG, None)
                else:
                    os.environ[FLAG] = flag
                with self.assertRaises(ImportError):
                    self.run_method()
        torch.testing.assert_close(self.packed, before)

    def test_empty_batch_returns_inputs_without_import_or_state_update(self):
        self.ns["_is_hcu"] = True
        for key in ("kv", "score", "req", "pos", "pad"):
            self.inputs[key] = self.inputs[key][:0]
        before = self.packed.clone()
        with patch.dict(os.environ, {FLAG: "1"}), patch.dict("sys.modules", {"lightop.attention": None}):
            result = self.run_method()
        self.assertIs(result[0], self.inputs["kv"])
        self.assertIs(result[1], self.inputs["score"])
        torch.testing.assert_close(self.packed, before)


if __name__ == "__main__":
    unittest.main()
