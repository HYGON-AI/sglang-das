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


REPO_ROOT = Path(__file__).parents[4]
EAGLE_UTILS_PATH = REPO_ROOT / "python/sglang/srt/speculative/eagle_utils.py"


class TestHcuEagleSamplingCompat(unittest.TestCase):
    def test_lightop_sampling_uses_supported_joint_filter_order(self):
        tree = ast.parse(EAGLE_UTILS_PATH.read_text())
        sample = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "sample_mtp_target_ids"
        )
        lightop_call = next(
            node
            for node in ast.walk(sample)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "lightop_top_k_top_p_sampling_from_probs"
        )
        keywords = {keyword.arg: keyword.value for keyword in lightop_call.keywords}

        self.assertEqual(ast.literal_eval(keywords["filter_apply_order"]), "joint")
        self.assertTrue(ast.literal_eval(keywords["deterministic"]))


if __name__ == "__main__":
    unittest.main()
