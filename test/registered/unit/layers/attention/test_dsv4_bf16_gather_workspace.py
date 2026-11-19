"""CPU regressions for capture-safe BF16 gather workspace views."""

import ast
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Literal, Optional, Tuple
from unittest.mock import Mock, patch

import torch


def workspace_methods():
    # Isolate the actual methods from hardware-dependent backend imports.
    root = Path(__file__).resolve().parents[5]
    source = root / "python/sglang/srt/layers/attention/deepseek_v4_backend.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4AttnBackend"
    )
    names = {
        "_allocate_dsv4_bf16_flashmla_workspace",
        "_get_dsv4_bf16_flashmla_workspace",
        "_prepare_dsv4_bf16_flashmla_inputs",
        "_try_prepare_dsv4_w4a8_bf16_inputs",
    }
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    scope = {
        "torch": torch,
        "Tuple": Tuple,
        "Optional": Optional,
        "ForwardMode": object,
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), scope)
    obj = SimpleNamespace(
        device="cpu",
        _dsv4_bf16_flashmla_decode=True,
        _dsv4_compact_gather_enabled=True,
        _dsv4_bf16_flashmla_workspaces={},
    )
    for name in names:
        setattr(obj, name, scope[name].__get__(obj))
    return obj


def model_family(config):
    root = Path(__file__).resolve().parents[5]
    source = root / "python/sglang/srt/layers/attention/deepseek_v4_backend.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_dsv4_bf16_gather_model_family"
    )
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    return scope[node.name](config)


def w4a8_config():
    return {
        "quant_method": "compressed-tensors",
        "config_groups": {
            "routed": {
                "targets": [r"re:.*ffn\.experts\.\d+\.w[123]$"],
                "weights": {
                    "type": "int",
                    "num_bits": 4,
                    "strategy": "channel",
                    "symmetric": True,
                },
                "input_activations": {
                    "type": "int",
                    "num_bits": 8,
                    "strategy": "token",
                    "dynamic": True,
                    "symmetric": True,
                },
            },
            "shared": {
                "targets": [r"re:.*shared_experts\.w[123]$"],
                "weights": {"type": "int", "num_bits": 8},
            },
            "embedding": {
                "targets": ["engram.embed"],
                "weights": {"type": "float", "num_bits": 8},
            },
        },
    }


class CudaLikeTensor(torch.Tensor):
    """CPU storage with CUDA eligibility, used only with spy gather operators."""

    @property
    def is_cuda(self):
        return True


class TestBF16GatherModelDispatch(unittest.TestCase):
    def test_forward_caller_keeps_fallback_lazy_and_original_kwargs(self):
        root = Path(__file__).resolve().parents[5]
        source = root / "python/sglang/srt/layers/attention/deepseek_v4_backend.py"
        cls = next(
            n
            for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4AttnBackend"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_forward_flash_mla_decode"
        )
        mode = SimpleNamespace(is_idle=lambda: False)
        scope = {
            "torch": torch,
            "Optional": Optional,
            "Literal": Literal,
            "ForwardBatch": object,
            "_get_logical_forward_mode": lambda batch: mode,
        }
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
            scope,
        )
        args = dict(
            q=torch.zeros((12, 1, 512)),
            swa_k_cache=torch.zeros((1, 256, 1, 584), dtype=torch.uint8),
            swa_page_indices=torch.zeros((12, 1, 128), dtype=torch.int32),
            swa_topk_lengths=torch.full((12,), 128, dtype=torch.int32),
            flashmla_metadata=None,
            attn_sink=torch.zeros(1),
            extra_k_cache=None,
            extra_indices=None,
            extra_topk_lengths=None,
            compress_ratio=0,
            layer_id=0,
            forward_batch=object(),
        )
        prepared = (
            args["swa_k_cache"],
            args["swa_page_indices"],
            args["swa_topk_lengths"],
            None,
            None,
            None,
        )
        for compact_result in (prepared, None):
            obj = SimpleNamespace(
                _dsv4_bf16_flashmla_decode=True,
                _try_prepare_dsv4_w4a8_bf16_inputs=Mock(return_value=compact_result),
                _prepare_dsv4_bf16_flashmla_inputs=Mock(return_value=prepared),
                _build_flash_mla_input_dict=Mock(return_value={}),
                _call_flash_mla_with_kvcache=Mock(
                    return_value=torch.zeros((12, 1, 512))
                ),
            )
            scope[method.name](obj, **args)
            obj._try_prepare_dsv4_w4a8_bf16_inputs.assert_called_once()
            if compact_result is None:
                obj._prepare_dsv4_bf16_flashmla_inputs.assert_called_once()
                self.assertNotIn(
                    "logical_forward_mode",
                    obj._prepare_dsv4_bf16_flashmla_inputs.call_args.kwargs,
                )
            else:
                obj._prepare_dsv4_bf16_flashmla_inputs.assert_not_called()

    def test_initialization_only_reserves_reuse_lengths_for_w4a8(self):
        root = Path(__file__).resolve().parents[5]
        source = root / "python/sglang/srt/layers/attention/deepseek_v4_backend.py"
        cls = next(
            n
            for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4AttnBackend"
        )
        init = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        names = {
            "_dsv4_bf16_gather_family",
            "_dsv4_compact_gather_enabled",
            "_dsv4_query_reuse_zero_lengths",
        }
        selected = [
            n
            for n in init.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Attribute) and t.attr in names for t in n.targets)
        ]
        self.assertEqual(len(selected), 3)
        code = compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec")
        for config, enabled in (
            (w4a8_config(), True),
            ({"quant_method": "fp8"}, False),
            ({}, False),
        ):
            obj = workspace_methods()
            obj._dsv4_lightop_bf16_gather = True
            scope = {
                "self": obj,
                "torch": torch,
                "model_runner": SimpleNamespace(
                    model_config=SimpleNamespace(hf_quant_config=config)
                ),
                "_dsv4_bf16_gather_model_family": model_family,
            }
            exec(code, scope)
            self.assertEqual(obj._dsv4_compact_gather_enabled, enabled)
            self.assertEqual(obj._dsv4_query_reuse_zero_lengths is not None, enabled)

    def test_matching_verify_shape_only_uses_compact_for_w4a8(self):
        for family, provide_api in (
            ("w4a8", True),
            ("w4a8", False),
            ("fp8", True),
            ("original", True),
        ):
            with self.subTest(family=family, provide_api=provide_api):
                obj = workspace_methods()
                obj._dsv4_compact_gather_enabled = family == "w4a8"
                obj._dsv4_lightop_bf16_gather = True
                obj.is_dspark_draft = False
                compact, plain = Mock(), Mock()
                obj._dsv4_lightop_kvcache_op = SimpleNamespace(
                    dsv4_gather_upconvert_dual_k_cache_paged=plain
                )
                package, api = ModuleType("lightop"), ModuleType("lightop.kvcache")
                package.__path__ = []
                package.kvcache = api
                if provide_api:
                    api.dsv4_gather_upconvert_compact_k_cache_paged = compact
                cache = torch.zeros((1, 256, 1, 584), dtype=torch.uint8).as_subclass(
                    CudaLikeTensor
                )
                mode = SimpleNamespace(
                    is_target_verify=lambda: True, is_draft_extend_v2=lambda: False
                )
                inputs = dict(
                    swa_k_cache=cache,
                    swa_indices=torch.zeros((12, 1, 128), dtype=torch.int32),
                    swa_topk_lengths=torch.full((12,), 128, dtype=torch.int32),
                    extra_k_cache=cache,
                    extra_indices=torch.zeros((12, 1, 512), dtype=torch.int32),
                    extra_topk_lengths=torch.full((12,), 512, dtype=torch.int32),
                )
                with (
                    patch.dict(
                        sys.modules, {"lightop": package, "lightop.kvcache": api}
                    ),
                    patch.object(
                        torch.cuda, "is_current_stream_capturing", return_value=False
                    ),
                ):
                    result = obj._try_prepare_dsv4_w4a8_bf16_inputs(
                        logical_forward_mode=mode, **inputs
                    )
                    if result is None:
                        obj._prepare_dsv4_bf16_flashmla_inputs(**inputs)
                if family == "w4a8" and provide_api:
                    compact.assert_called_once()
                    plain.assert_not_called()
                    self.assertEqual(compact.call_args.kwargs["reuse_group"], 6)
                else:
                    compact.assert_not_called()
                    plain.assert_called_once()

    def test_mixed_w4a8_checkpoint_is_not_misclassified_by_other_8bit_weights(self):
        self.assertEqual(model_family(w4a8_config()), "w4a8")

    def test_fp8_checkpoint_and_compressed_expert_group(self):
        self.assertEqual(model_family({"quant_method": "fp8"}), "fp8")
        # Read from the nmz26 DeepSeek-V4.1-Flash-1017-Channel-FP8-w8a8 config.
        self.assertEqual(
            model_family(
                {
                    "activation_scheme": "dynamic",
                    "is_checkpoint_fp8_serialized": True,
                    "is_per_channel": True,
                    "quant_method": "fp8",
                }
            ),
            "fp8",
        )
        config = w4a8_config()
        group = config["config_groups"]["routed"]
        group["weights"].update(type="float", num_bits=8)
        group["input_activations"]["type"] = "float"
        self.assertEqual(model_family(config), "fp8")

    def test_unknown_and_mixed_expert_layouts_keep_original(self):
        for config in (
            None,
            {},
            {"quant_method": "int8"},
            {"quant_method": "fp8", "routed_experts_quant_method": "mxfp4"},
            {"quant_method": "fp8", "routed_experts_quant_method": "unknown"},
            {"quant_method": "compressed-tensors", "config_groups": []},
        ):
            with self.subTest(config=config):
                self.assertEqual(model_family(config), "original")
        config = w4a8_config()
        extra = deepcopy(config["config_groups"]["routed"])
        extra["weights"].update(type="float", num_bits=8)
        extra["input_activations"]["type"] = "float"
        config["config_groups"]["fp8_experts"] = extra
        self.assertEqual(model_family(config), "original")

    def test_non_dynamic_or_asymmetric_w4_keeps_original(self):
        for field, value in (
            ("dynamic", False),
            ("symmetric", False),
            ("strategy", "tensor"),
            ("type", "float"),
        ):
            config = w4a8_config()
            config["config_groups"]["routed"]["input_activations"][field] = value
            self.assertEqual(model_family(config), "original")

    def test_fp8_workspace_keeps_its_original_exact_topk_layout(self):
        obj = workspace_methods()
        obj._dsv4_compact_gather_enabled = False
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            obj._get_dsv4_bf16_flashmla_workspace("swa", 2, 128)
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "before CUDA graph capture"):
                obj._get_dsv4_bf16_flashmla_workspace("swa", 2, 64)


class TestBF16GatherWorkspace(unittest.TestCase):
    def test_old_lightop_without_compact_api_keeps_single_gather(self):
        obj = workspace_methods()
        obj._dsv4_lightop_bf16_gather = True
        obj.is_dspark_draft = False
        gather = Mock()
        obj._dsv4_lightop_kvcache_op = SimpleNamespace(
            dsv4_gather_upconvert_k_cache_paged=gather
        )
        old_package = ModuleType("lightop")
        old_package.__path__ = []
        old_kvcache = ModuleType("lightop.kvcache")
        old_package.kvcache = old_kvcache
        indices = torch.zeros((2, 1, 8), dtype=torch.int32)
        lengths = torch.full((2,), 8, dtype=torch.int32)
        with patch.dict(
            sys.modules, {"lightop": old_package, "lightop.kvcache": old_kvcache}
        ):
            with patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=False
            ):
                result = obj._prepare_dsv4_bf16_flashmla_inputs(
                    swa_k_cache=torch.zeros((1, 256, 1, 584), dtype=torch.uint8),
                    swa_indices=indices,
                    swa_topk_lengths=lengths,
                    extra_k_cache=None,
                    extra_indices=None,
                    extra_topk_lengths=None,
                )
        gather.assert_called_once()
        self.assertEqual(result[0].shape, (2, 8, 1, 512))
        torch.testing.assert_close(result[2], lengths)

    def test_smaller_topk_keeps_reserved_graph_address_and_compact_stride(self):
        obj = workspace_methods()
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            original, original_ids = obj._get_dsv4_bf16_flashmla_workspace(
                "combined", 48, 1152
            )
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            for queries, topk in [(12, 640), (24, 768), (48, 1152), (6, 128)]:
                out, ids = obj._get_dsv4_bf16_flashmla_workspace(
                    "combined", queries, topk
                )
                self.assertEqual(out.data_ptr(), original.data_ptr())
                self.assertEqual(ids.data_ptr(), original_ids.data_ptr())
                self.assertEqual(out.shape, (queries, topk, 1, 512))
                self.assertEqual(ids.shape, (queries, 1, topk))
                self.assertTrue(out.is_contiguous())
                self.assertTrue(ids.is_contiguous())

    def test_growth_preserves_both_capacity_dimensions(self):
        obj = workspace_methods()
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            obj._get_dsv4_bf16_flashmla_workspace("combined", 48, 640)
            obj._get_dsv4_bf16_flashmla_workspace("combined", 12, 1152)
        out, _ = obj._dsv4_bf16_flashmla_workspaces["combined"]
        self.assertEqual(out.shape, (48, 1152, 1, 512))

    def test_missing_capacity_cannot_allocate_during_capture(self):
        obj = workspace_methods()
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "before CUDA graph capture"):
                obj._get_dsv4_bf16_flashmla_workspace("combined", 12, 640)


if __name__ == "__main__":
    unittest.main()
