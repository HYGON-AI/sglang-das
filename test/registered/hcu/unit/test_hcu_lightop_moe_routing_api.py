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
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_hcu_ci

register_hcu_ci(est_time=5, suite="stage-b-test-1-hcu-small")


REPO_ROOT = Path(__file__).parents[4]
ALIGN_SOURCE = (
    REPO_ROOT
    / "python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py"
)


class TestHcuLightOpMoeRoutingApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(ALIGN_SOURCE.read_text())

    def test_hcu_imports_the_public_moe_module(self):
        imports = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.ImportFrom) and node.module == "lightop"
        ]
        imported_names = {
            (alias.name, alias.asname) for node in imports for alias in node.names
        }
        self.assertIn(("moe", "op"), imported_names)
        self.assertNotIn(("op", None), imported_names)

    def test_standard_route_retains_triton_scratch_safety_fix(self):
        route = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "moe_align_block_size"
        )
        text = ast.unparse(route)
        self.assertIn("_hcu_moe_align_block_size_triton(", text)
        self.assertIn("sorted_ids.fill_(topk_ids.numel())", text)
        self.assertIn("expert_ids.masked_fill_(expert_ids == invalid_expert, -1)", text)

    def test_lightop_route_uses_public_output_buffer_abi(self):
        route = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "hcu_moe_align_block_size"
        )
        text = ast.unparse(route)
        self.assertIn("getattr(op, 'moe_align_block_size_out', None)", text)
        call = next(
            node
            for node in ast.walk(route)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "align_op"
        )
        self.assertEqual(
            [ast.unparse(arg) for arg in call.args],
            [
                "topk_ids",
                "num_experts",
                "block_size",
                "sorted_ids",
                "expert_ids",
                "num_tokens_post_pad",
                "expert_map if expert_mask is not None else None",
                "expert_mask",
                "None",
                "False",
                "True",
            ],
        )


if __name__ == "__main__":
    unittest.main()
