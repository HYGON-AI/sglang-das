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

    def test_both_routes_use_the_output_tensor_wrapper(self):
        calls = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_hcu_moe_align_block_size_out"
        ]
        self.assertEqual(len(calls), 2)
        for call in calls:
            keywords = {keyword.arg for keyword in call.keywords}
            self.assertTrue(
                {
                    "expert_map",
                    "expert_mask",
                    "num_local_tokens",
                    "is_ep",
                    "is_fuse_fill",
                }
                <= keywords
            )


if __name__ == "__main__":
    unittest.main()
