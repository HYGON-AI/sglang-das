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
MODULE_PATH = (
    REPO_ROOT
    / "python/sglang/srt/distributed/device_communicators/triton_symm_mem_ag.py"
)


class TestHcuMultimemAllGatherGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(MODULE_PATH.read_text())
        cls.helper = next(
            node
            for node in cls.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_supports_multimem_all_gather"
        )

    def _supports(self, *, enabled: bool, cuda: bool) -> bool:
        module = ast.fix_missing_locations(
            ast.Module(body=[self.helper], type_ignores=[])
        )
        namespace = {"is_cuda": lambda: cuda}
        exec(compile(module, str(MODULE_PATH), "exec"), namespace)
        return namespace["_supports_multimem_all_gather"](enabled)

    def test_multimem_requires_both_enablement_and_cuda(self):
        self.assertTrue(self._supports(enabled=True, cuda=True))
        self.assertFalse(self._supports(enabled=False, cuda=True))
        self.assertFalse(self._supports(enabled=True, cuda=False))

    def test_gatherer_state_uses_platform_guard(self):
        gatherer = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MultimemAllGatherer"
        )
        constructor = next(
            node
            for node in gatherer.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        state_assignments = [
            node
            for node in ast.walk(constructor)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "_state"
                for target in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
            )
        ]
        guard_calls = [
            node
            for assignment in state_assignments
            for node in ast.walk(assignment)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_supports_multimem_all_gather"
        ]
        self.assertEqual(len(guard_calls), 1)
        self.assertEqual(len(guard_calls[0].args), 1)
        self.assertIsInstance(guard_calls[0].args[0], ast.Name)
        self.assertEqual(guard_calls[0].args[0].id, "enabled")


if __name__ == "__main__":
    unittest.main()
