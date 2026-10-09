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
import types
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_hcu_ci
register_hcu_ci(est_time=5, suite="stage-b-test-1-hcu-small")


REPO_ROOT = Path(__file__).parents[4]
BACKEND_PATH = (
    REPO_ROOT / "python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py"
)


class _FakeScores:
    def __init__(self, *, dtype, ndim=2, row_stride=8, column_stride=1):
        self.dtype = dtype
        self.ndim = ndim
        self._strides = (row_stride, column_stride)

    def stride(self, dimension):
        return self._strides[dimension]


class TestHcuTopKV2RaggedStride(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(BACKEND_PATH.read_text())
        helper = next(
            node
            for node in cls.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_supports_topk_v2_ragged_layout"
        )
        module = ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[]))
        cls.float32 = object()
        namespace = {
            "torch": types.SimpleNamespace(Tensor=object, float32=cls.float32)
        }
        exec(compile(module, str(BACKEND_PATH), "exec"), namespace)
        cls.supports_layout = staticmethod(
            namespace["_supports_topk_v2_ragged_layout"]
        )

    def test_layout_contract_accepts_only_aligned_fp32_rows(self):
        self.assertTrue(self.supports_layout(_FakeScores(dtype=self.float32)))
        self.assertFalse(
            self.supports_layout(_FakeScores(dtype=self.float32, row_stride=7))
        )
        self.assertFalse(
            self.supports_layout(_FakeScores(dtype=self.float32, column_stride=2))
        )
        self.assertFalse(self.supports_layout(_FakeScores(dtype=object())))
        self.assertFalse(
            self.supports_layout(_FakeScores(dtype=self.float32, ndim=1))
        )

    def test_ragged_v2_dispatch_checks_the_layout_contract(self):
        backend = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DSATopKBackend"
        )
        transform = next(
            node
            for node in backend.body
            if isinstance(node, ast.FunctionDef) and node.name == "topk_transform"
        )
        ragged_v2_if = next(
            node
            for node in ast.walk(transform)
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "_topk_transform_v2_ragged"
                for child in ast.walk(node)
            )
        )
        layout_calls = [
            node
            for node in ast.walk(ragged_v2_if.test)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_supports_topk_v2_ragged_layout"
        ]
        self.assertEqual(len(layout_calls), 1)
        self.assertEqual(len(layout_calls[0].args), 1)
        self.assertIsInstance(layout_calls[0].args[0], ast.Name)
        self.assertEqual(layout_calls[0].args[0].id, "logits")

    def test_hcu_paged_fallback_keeps_ragged_v2_available(self):
        backend = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DSATopKBackend"
        )
        predicate = next(
            node
            for node in backend.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "should_use_topk_v2_paged"
        )
        module = ast.fix_missing_locations(
            ast.Module(body=[predicate], type_ignores=[])
        )
        switch = types.SimpleNamespace(get=lambda: False)
        namespace = {
            "_is_hcu": True,
            "envs": types.SimpleNamespace(SGLANG_HCU_TOPK_V2_PAGED=switch),
        }
        exec(compile(module, str(BACKEND_PATH), "exec"), namespace)
        should_use_paged = namespace["should_use_topk_v2_paged"]
        v2_backend = types.SimpleNamespace(should_use_topk_v2=lambda: True)
        self.assertFalse(should_use_paged(v2_backend))
        namespace["envs"].SGLANG_HCU_TOPK_V2_PAGED.get = lambda: True
        self.assertTrue(should_use_paged(v2_backend))
        namespace["_is_hcu"] = False
        self.assertTrue(should_use_paged(v2_backend))
        self.assertFalse(
            should_use_paged(types.SimpleNamespace(should_use_topk_v2=lambda: False))
        )


if __name__ == "__main__":
    unittest.main()
