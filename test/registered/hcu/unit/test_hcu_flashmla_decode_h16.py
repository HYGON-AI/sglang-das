# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Hygon modifications to this file are licensed under the Apache License,
# Version 2.0 (the "License"); you may not use these modifications except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import inspect
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).parents[4]
BACKEND_PATH = REPO_ROOT / "python/sglang/srt/layers/attention/dsa_backend.py"


class _Mode:
    def __init__(self, *, extend=False):
        self.extend = extend

    def is_extend_without_speculative(self):
        return self.extend


def _tree():
    return ast.parse(BACKEND_PATH.read_text())


def _top_level_function(name):
    return next(
        node
        for node in _tree().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _load_probe(*, hcu=True, metadata=None):
    nodes = [
        _top_level_function("_accepts_keyword"),
        _top_level_function("_hcu_flashmla_decode_h16_available"),
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    metadata = metadata or (lambda *, num_heads_q: num_heads_q)
    ops = {
        "flash_mla_with_kvcache": lambda **kwargs: None,
        "get_mla_metadata": metadata,
    }
    namespace = {
        "inspect": inspect,
        "_is_hcu": hcu,
        "get_flashmla_op": lambda name, **kwargs: ops[name],
    }
    exec(compile(module, str(BACKEND_PATH), "exec"), namespace)
    return namespace["_hcu_flashmla_decode_h16_available"]


def _load_method(name):
    backend = next(
        node
        for node in _tree().body
        if isinstance(node, ast.ClassDef) and node.name == "DeepseekSparseAttnBackend"
    )
    method = next(
        node
        for node in backend.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"ForwardMode": object}
    exec(compile(module, str(BACKEND_PATH), "exec"), namespace)
    return namespace[name]


class TestHcuFlashMlaDecodeH16(unittest.TestCase):
    def test_probe_accepts_explicit_head_abi(self):
        self.assertTrue(_load_probe()())

    def test_probe_rejects_missing_head_abi_or_non_hcu(self):
        self.assertFalse(_load_probe(metadata=lambda cache_seqlens: cache_seqlens)())
        self.assertFalse(_load_probe(hcu=False)())

    def test_prefill_keeps_h64_while_decode_uses_h16(self):
        select_mode = _load_method("_flashmla_kv_target_q_heads_for_mode")
        select_runtime = _load_method("_flashmla_kv_target_q_heads_for_runtime")
        owner = types.SimpleNamespace(
            _hcu_flashmla_decode_h16_enabled=True,
            flashmla_kv_num_q_heads=16,
        )
        self.assertEqual(select_mode(owner, _Mode(extend=True)), 64)
        self.assertEqual(select_mode(owner, _Mode(extend=False)), 16)
        self.assertEqual(select_runtime(owner, 8), 16)
        self.assertEqual(select_runtime(owner, 64), 64)
        self.assertEqual(select_runtime(owner, 128), 128)

    def test_h16_is_exact_and_default_off(self):
        source = BACKEND_PATH.read_text()
        for guard in (
            'os.getenv("SGLANG_HCU_FLASHMLA_DECODE_H16", "0") == "1"',
            "self.num_q_heads == 64",
            'self.dsa_decode_impl == "flashmla_kv"',
            "model_runner.server_args.tp_size == 8",
            'getattr(get_parallel(), "enable_cp_decode_attn_tp", False)',
            'arch.startswith("GlmMoeDsaForCausalLM")',
            "_hcu_flashmla_decode_h16_available()",
        ):
            with self.subTest(guard=guard):
                self.assertIn(guard, source)
        self.assertIn(
            "num_heads_q=self._flashmla_kv_target_q_heads_for_mode(", source
        )


if __name__ == "__main__":
    unittest.main()
