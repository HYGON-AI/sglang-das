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
SERVER_ARGS_PATH = REPO_ROOT / "python/sglang/srt/server_args.py"
GRAPH_SETUP_PATH = (
    REPO_ROOT
    / "python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py"
)


def _load_method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


def _load_function(path, function_name, namespace):
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[function_name]


def _memory_budget_method(*, hcu=True, enabled=True, graph_limit=16):
    connector = types.SimpleNamespace(INSTANCE="instance")
    return _load_method(
        SERVER_ARGS_PATH,
        "ServerArgs",
        "_handle_hcu_flashmla_decode_h16_memory_budget",
        {
            "is_hcu": lambda: hcu,
            "get_bool_env_var": lambda name: enabled,
            "get_int_env_var": lambda name, default: graph_limit,
            "parse_connector_type": lambda path: "local",
            "ConnectorType": connector,
            "logger": types.SimpleNamespace(warning=lambda *args: None),
        },
    )


def _args(**overrides):
    decode = types.SimpleNamespace(
        max_bs=32,
        bs=[1, 2, 4, 8, 10, 12, 14, 16, 18, 20, 32],
    )
    values = {
        "enable_cp_decode_attn_tp": True,
        "tp_size": 8,
        "model_path": "glm",
        "max_running_requests": 128,
        "cuda_graph_config": types.SimpleNamespace(decode=decode),
        "_resolved": lambda: types.SimpleNamespace(attn_cp_size=8),
        "get_model_config": lambda: types.SimpleNamespace(
            hf_config=types.SimpleNamespace(
                architectures=["GlmMoeDsaForCausalLM"]
            )
        ),
    }
    values.update(overrides)
    owner = types.SimpleNamespace(**values)

    def declare(source, **fields):
        for name, value in fields.items():
            setattr(owner, name, value)

    owner._declare = declare
    return owner


class TestHcuFlashmlaDecodeH16Memory(unittest.TestCase):
    def test_exact_profile_clamps_graph_and_request_metadata(self):
        owner = _args()
        _memory_budget_method()(owner)
        self.assertEqual(owner.cuda_graph_config.decode.max_bs, 16)
        self.assertEqual(
            owner.cuda_graph_config.decode.bs,
            [1, 2, 4, 8, 10, 12, 14, 16],
        )
        self.assertEqual(owner.max_running_requests, 64)

    def test_validated_wider_limit_and_lower_user_limits_are_preserved(self):
        owner = _args(max_running_requests=32)
        owner.cuda_graph_config.decode.max_bs = 64
        owner.cuda_graph_config.decode.bs = [1, 2, 4, 8, 16, 32, 64]
        _memory_budget_method(graph_limit=64)(owner)
        self.assertEqual(owner.cuda_graph_config.decode.max_bs, 64)
        self.assertEqual(owner.cuda_graph_config.decode.bs[-1], 64)
        self.assertEqual(owner.max_running_requests, 32)

    def test_nonpositive_graph_limit_is_rejected(self):
        for graph_limit in (0, -1):
            with self.subTest(graph_limit=graph_limit), self.assertRaises(ValueError):
                _memory_budget_method(graph_limit=graph_limit)(_args())

    def test_default_off_and_topology_mismatches_are_unchanged(self):
        cases = (
            (_memory_budget_method(enabled=False), _args()),
            (_memory_budget_method(hcu=False), _args()),
            (_memory_budget_method(), _args(tp_size=4)),
            (_memory_budget_method(), _args(enable_cp_decode_attn_tp=False)),
            (
                _memory_budget_method(),
                _args(_resolved=lambda: types.SimpleNamespace(attn_cp_size=4)),
            ),
        )
        for method, owner in cases:
            with self.subTest(owner=owner):
                before = (
                    owner.cuda_graph_config.decode.max_bs,
                    list(owner.cuda_graph_config.decode.bs),
                    owner.max_running_requests,
                )
                method(owner)
                after = (
                    owner.cuda_graph_config.decode.max_bs,
                    owner.cuda_graph_config.decode.bs,
                    owner.max_running_requests,
                )
                self.assertEqual(after, before)

    def test_post_capture_release_requires_the_active_backend(self):
        calls = []
        platform = types.SimpleNamespace(
            synchronize=lambda: calls.append("synchronize"),
            empty_cache=lambda: calls.append("empty_cache"),
        )
        release = _load_function(
            GRAPH_SETUP_PATH,
            "release_hcu_flashmla_decode_h16_capture_cache",
            {
                "is_hcu": lambda: True,
                "get_bool_env_var": lambda name: True,
                "current_platform": platform,
                "log_info_on_rank0": lambda *args: calls.append("log"),
                "logger": object(),
            },
        )
        runner = types.SimpleNamespace(
            attn_backend=types.SimpleNamespace(
                _hcu_flashmla_decode_h16_enabled=True
            )
        )
        self.assertTrue(release(runner))
        self.assertEqual(calls, ["synchronize", "empty_cache", "log"])

        calls.clear()
        runner.attn_backend._hcu_flashmla_decode_h16_enabled = False
        self.assertFalse(release(runner))
        self.assertEqual(calls, [])

    def test_all_graph_capture_lifecycles_call_cleanup(self):
        paths = (
            "python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py",
            "python/sglang/srt/speculative/eagle_worker_v2.py",
            "python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py",
            "python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py",
        )
        for path in paths:
            with self.subTest(path=path):
                self.assertIn(
                    "release_hcu_flashmla_decode_h16_capture_cache(",
                    (REPO_ROOT / path).read_text(),
                )


if __name__ == "__main__":
    unittest.main()
