"""CPU contract tests for the real HYV4 config passes (hardware/config stubs)."""

import ast
import pathlib
import types
import unittest
from unittest.mock import Mock, patch

ROOT = pathlib.Path(__file__).resolve().parents[4]


class TestHcuHyv4SinkDispatch(unittest.TestCase):
    def run_passes(self, hcu, dtype, arch="HYV4ForCausalLM", **kw):
        path = ROOT / "python/sglang/srt/arg_groups/overrides.py"
        names = {"_dsa_kv_cache_dtype_default", "_dsa_split_backend_resolution"}
        nodes = [
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        for n in nodes:
            n.decorator_list = []
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *nodes,
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)
        hf = types.SimpleNamespace(architectures=[arch], learnable_sink=True)
        view = types.SimpleNamespace(
            kv_cache_dtype=dtype,
            dsa_prefill_backend=None,
            dsa_decode_backend=None,
            enable_hisparse=False,
            **{},
        )
        for key, value in kw.items():
            setattr(view, key, value)
        env = {
            "model_config_of": lambda v: types.SimpleNamespace(hf_config=hf),
            "get_platform": lambda: types.SimpleNamespace(
                is_npu=False, is_xpu=False, is_hip=hcu
            ),
            "is_hcu": lambda: hcu,
            "logger": Mock(),
            "_DEEPSEEK_FAMILY_ARCHS": {arch},
        }
        config = types.ModuleType("sglang.srt.configs.model_config")
        config.is_deepseek_dsa = lambda cfg: True
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(get_device_capability=lambda: (9, 0))
        with patch.dict(
            "sys.modules", {"sglang.srt.configs.model_config": config, "torch": torch}
        ):
            exec(compile(module, str(path), "exec"), env)
            dtype_result = env["_dsa_kv_cache_dtype_default"](view)
            for key, value in dtype_result.items():
                setattr(view, key, value)
            return dtype_result, env["_dsa_split_backend_resolution"](view)

    def test_hcu_fp8_retains_sink_capable_split_backends(self):
        for arch in ("HYV4ForCausalLM", "HYV4ForCausalLMNextN"):
            with self.subTest(arch=arch):
                self.assertEqual(
                    self.run_passes(True, "fp8_e4m3", arch),
                    (
                        {},
                        {
                            "dsa_prefill_backend": "flashmla_sparse",
                            "dsa_decode_backend": "flashmla_kv",
                        },
                    ),
                )

    def test_bfloat16_keeps_sparse_backend_on_both_platforms(self):
        for hcu in (False, True):
            self.assertEqual(
                self.run_passes(hcu, "bfloat16")[1],
                {
                    "dsa_prefill_backend": "flashmla_sparse",
                    "dsa_decode_backend": "flashmla_sparse",
                },
            )

    def test_non_hcu_fp8_and_other_arch_are_not_enabled(self):
        for hcu, arch in ((False, "HYV4ForCausalLM"), (True, "DeepseekV32ForCausalLM")):
            with self.assertRaises(ValueError):
                self.run_passes(hcu, "fp8_e4m3", arch)

    def test_incompatible_explicit_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_passes(True, "fp8_e4m3", dsa_decode_backend="trtllm")


if __name__ == "__main__":
    unittest.main()
