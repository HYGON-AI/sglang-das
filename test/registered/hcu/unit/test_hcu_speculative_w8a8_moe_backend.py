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
from typing import Optional


REPO_ROOT = Path(__file__).parents[4]
MOE_SOURCE = (
    REPO_ROOT
    / "python/sglang/srt/layers/quantization/compressed_tensors/schemes/"
    "compressed_tensors_w8a8_fp8_moe.py"
)


def _load_helper():
    tree = ast.parse(MOE_SOURCE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_should_use_hcu_fp8_w8a8_moe"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"Optional": Optional}
    exec(compile(module, str(MOE_SOURCE), "exec"), namespace)
    return namespace["_should_use_hcu_fp8_w8a8_moe"]


class TestHcuSpeculativeW8A8MoeBackend(unittest.TestCase):
    def test_eagle_draft_keeps_hcu_fast_path(self):
        helper = _load_helper()
        self.assertTrue(
            helper(
                enabled=True,
                in_speculative_a2a_scope=True,
                speculative_algorithm="EAGLE",
                use_deepep=False,
            )
        )

    def test_standalone_dspark_draft_keeps_canonical_weights(self):
        helper = _load_helper()
        self.assertFalse(
            helper(
                enabled=True,
                in_speculative_a2a_scope=True,
                speculative_algorithm="DSPARK",
                use_deepep=False,
            )
        )

    def test_deepep_dspark_and_target_keep_hcu_fast_path(self):
        helper = _load_helper()
        for in_speculative_scope, use_deepep in ((True, True), (False, False)):
            with self.subTest(
                in_speculative_scope=in_speculative_scope,
                use_deepep=use_deepep,
            ):
                self.assertTrue(
                    helper(
                        enabled=True,
                        in_speculative_a2a_scope=in_speculative_scope,
                        speculative_algorithm="DSPARK",
                        use_deepep=use_deepep,
                    )
                )

    def test_disabled_hcu_backend_stays_disabled(self):
        helper = _load_helper()
        self.assertFalse(
            helper(
                enabled=False,
                in_speculative_a2a_scope=True,
                speculative_algorithm="EAGLE",
                use_deepep=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
