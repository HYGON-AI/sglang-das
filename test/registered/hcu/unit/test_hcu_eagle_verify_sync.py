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
EAGLE_UTILS_PATH = REPO_ROOT / "python/sglang/srt/speculative/eagle_utils.py"


def _function(tree, name):
    return next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


class TestHcuEagleVerifySync(unittest.TestCase):
    def test_verifier_outputs_share_one_int32_state_buffer(self):
        tree = ast.parse(EAGLE_UTILS_PATH.read_text())
        allocate_text = ast.unparse(
            _function(tree, "_allocate_eagle_verify_outputs")
        )
        self.assertEqual(allocate_text.count("torch.empty("), 1)
        self.assertIn("dtype=torch.int32", allocate_text)
        self.assertIn("predict = verify_state[:predict_size]", allocate_text)
        self.assertIn("accept_index = verify_state", allocate_text)
        self.assertIn(
            "num_correct_drafts = verify_state[predict_size + accept_index_size:]",
            allocate_text,
        )

    def test_sync_uses_full_tp_for_one_dp_and_two_dimensions_otherwise(self):
        tree = ast.parse(EAGLE_UTILS_PATH.read_text())
        sync_text = ast.unparse(_function(tree, "_sync_eagle_verify_outputs"))
        self.assertIn(
            "not is_dp_attention_enabled() or get_parallel().dp_size == 1",
            sync_text,
        )
        self.assertLess(
            sync_text.index("groups = (get_tp_group(),)"),
            sync_text.index(
                "groups = (get_parallel().attn_tp_group, "
                "get_parallel().attn_cp_group)"
            ),
        )
        self.assertIn(
            "groups = (get_parallel().attn_tp_group, "
            "get_parallel().attn_cp_group)",
            sync_text,
        )
        self.assertEqual(sync_text.count("group.broadcast(verify_state, src=0)"), 1)

    def test_sync_happens_after_all_verifier_paths(self):
        tree = ast.parse(EAGLE_UTILS_PATH.read_text())
        sample_text = ast.unparse(_function(tree, "eagle_sample"))
        self.assertEqual(sample_text.count("_sync_eagle_verify_outputs("), 1)
        self.assertLess(
            sample_text.index("verify_tree_greedy_func("),
            sample_text.index("_sync_eagle_verify_outputs("),
        )
        self.assertLess(
            sample_text.index("sampling_fn("),
            sample_text.index("_sync_eagle_verify_outputs("),
        )


if __name__ == "__main__":
    unittest.main()
