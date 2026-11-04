"""CPU regression coverage for channelwise INT8 special projections."""

import unittest
import os
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization import w8a8_int8 as runtime
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestChannelwiseInt8(unittest.TestCase):
    def test_unset_backend_dispatches_to_blaslt(self):
        from sglang.kernels.ops.gemm import dsv41_int8 as dsv41_int8_blaslt

        with patch.dict(os.environ):
            os.environ.pop("SGLANG_DSV41_INT8_PROJECTION_BACKEND", None)
            with patch.object(dsv41_int8_blaslt, "project", return_value=torch.zeros(2, 1, 4)) as project:
                runtime.linear(torch.ones(2, 8), torch.ones(4, 8, dtype=torch.int8), torch.ones(4, 1))
            project.assert_called_once()

    def test_unset_backend_prepares_blaslt_weights(self):
        from sglang.kernels.ops.gemm import dsv41_int8 as dsv41_int8_blaslt

        layer = self.make_layer()
        self.load(layer)
        with patch.dict(os.environ):
            os.environ.pop("SGLANG_DSV41_INT8_PROJECTION_BACKEND", None)
            with patch.object(dsv41_int8_blaslt, "prepare_weight") as prepare:
                layer.quant_method.process_weights_after_loading(layer)
            prepare.assert_called_once_with(layer.weight)

    def test_blaslt_dispatch_preserves_requested_output_dtype(self):
        from sglang.kernels.ops.gemm import dsv41_int8 as dsv41_int8_blaslt

        w = torch.ones(4, 8, dtype=torch.int8)
        scale = torch.ones(4, 1)
        x = torch.ones(2, 8, dtype=torch.bfloat16)
        for dtype in (torch.bfloat16, torch.float32):
            with (
                patch.dict("os.environ", {"SGLANG_DSV41_INT8_PROJECTION_BACKEND": "blaslt"}),
                patch.object(dsv41_int8_blaslt, "project", return_value=torch.zeros(2, 1, 4, dtype=dtype)) as project,
            ):
                y = runtime.linear(x, w, scale, output_dtype=dtype)
            self.assertEqual(y.dtype, dtype)
            self.assertEqual(project.call_args.kwargs["output_dtype"], dtype)

    def test_default_lm_head_preserves_bf16_output(self):
        layer = self.make_layer()
        x = torch.ones(2, 8, dtype=torch.bfloat16)
        with patch.object(runtime, "linear", return_value=torch.zeros(2, 4, dtype=x.dtype)) as project:
            y = layer.quant_method.apply(layer, x)
        self.assertEqual(y.dtype, torch.bfloat16)
        self.assertEqual(project.call_args.kwargs["output_dtype"], torch.bfloat16)

    def test_draft_vp_uses_existing_size_without_extra_switch(self):
        from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark

        layer = self.make_layer()
        layer.tp_size = 1
        layer.org_vocab_size = 4
        draft = SimpleNamespace(_use_fp32_lm_head=False, lm_head=layer)
        with patch.dict(os.environ), patch("sglang.srt.speculative.draft_lm_head_vp.DSparkLMHeadVocabParallel") as vp:
            os.environ.pop("SGLANG_DSV41_INT8_VP", None)
            os.environ.pop("SGLANG_DSV41_INT8_PROJECTION_BACKEND", None)
            DeepseekV4ForCausalLMDSpark.enable_draft_lm_head_vp(
                draft, vp_size=8, max_rows_per_rank=64, local_world_size=8,
                activation_dtype=torch.bfloat16,
            )
            self.assertEqual(vp.call_args.kwargs["vp_size"], 8)
            self.assertIs(vp.call_args.kwargs["weight_scale"], layer.weight_scale)
            self.assertIs(draft.draft_lm_head_vp, vp.return_value)

    def make_layer(self):
        layer = torch.nn.Linear(8, 4, bias=False)
        runtime.install(layer)
        return layer

    def load(self, layer):
        w = torch.arange(32).reshape(4, 8).to(torch.int8)
        s = torch.arange(1, 5).float().reshape(4, 1)
        layer.weight.weight_loader(layer.weight, w)
        layer.weight_scale.weight_loader(layer.weight_scale, s)
        return w, s

    def test_config_selection_and_bf16_passthrough(self):
        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsConfig,
        )

        config = {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "ignore": [],
            "config_groups": {
                "special": {
                    "format": "int-quantized",
                    "targets": ["lm_head", "re:.*markov_head\\.head$", "re:.*wo_a$"],
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "strategy": "channel",
                        "symmetric": True,
                        "dynamic": False,
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "int",
                        "strategy": "token",
                        "symmetric": True,
                        "dynamic": True,
                    },
                }
            },
        }
        q = CompressedTensorsConfig.from_config(config)
        layer = torch.nn.Linear(8, 4, bias=False)
        with patch("sglang.srt.utils.is_hcu", return_value=True):
            for name in [
                "lm_head",
                "markov_head.head",
                "model.layers.0.self_attn.wo_a",
            ]:
                self.assertTrue(runtime.enabled(layer, q, name))
            self.assertFalse(runtime.enabled(layer, q, "other"))
            self.assertFalse(runtime.enabled(layer, None, "lm_head"))
        with patch("sglang.srt.utils.is_hcu", return_value=False):
            with self.assertRaisesRegex(ValueError, "requires HCU"):
                runtime.enabled(layer, q, "lm_head")

    def test_group_format_inherits_top_level_format(self):
        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsConfig,
        )

        config = CompressedTensorsConfig.from_config(
            {
                "quant_method": "compressed-tensors",
                "format": "int-quantized",
                "ignore": [],
                "config_groups": {
                    "head": {
                        "targets": ["lm_head"],
                        "weights": {
                            "num_bits": 8,
                            "type": "int",
                            "strategy": "channel",
                            "symmetric": True,
                            "dynamic": False,
                        },
                        "input_activations": {
                            "num_bits": 8,
                            "type": "int",
                            "strategy": "token",
                            "symmetric": True,
                            "dynamic": True,
                        },
                    }
                },
            }
        )
        with patch("sglang.srt.utils.is_hcu", return_value=True):
            self.assertTrue(runtime.enabled(torch.nn.Linear(8, 4), config, "lm_head"))

    def test_requires_weights_and_scale(self):
        layer = self.make_layer()
        with self.assertRaisesRegex(ValueError, "Missing"):
            layer.quant_method.process_weights_after_loading(layer)
        w, s = self.load(layer)
        layer.quant_method.process_weights_after_loading(layer)
        torch.testing.assert_close(layer.weight, w)
        torch.testing.assert_close(layer.weight_scale, s)

    def test_rejects_invalid_checkpoint_types(self):
        layer = self.make_layer()
        with self.assertRaises(ValueError):
            layer.weight.weight_loader(layer.weight, torch.ones(4, 8))
        for scale in [torch.ones(4), torch.ones(4, 1, dtype=torch.bfloat16)]:
            with self.assertRaises(ValueError):
                layer.weight_scale.weight_loader(layer.weight_scale, scale)

    def test_rejects_invalid_scales(self):
        for value in [float("nan"), float("inf"), -1, 0]:
            layer = self.make_layer()
            self.load(layer)
            layer.weight_scale.data[0] = value
            with self.assertRaises(ValueError):
                layer.quant_method.process_weights_after_loading(layer)

    def test_vocab_padding_keeps_matching_scale_slice(self):
        layer = torch.nn.Linear(8, 4, bias=False)
        layer.weight.output_dim = 0
        layer.shard_indices = SimpleNamespace(
            org_vocab_start_index=4, org_vocab_end_index=7
        )

        def loader(p, value):
            self.assertEqual(p.output_dim, 0)
            p.data[:3].copy_(value[4:7])
            p.data[3:].zero_()

        layer.weight.weight_loader = loader
        runtime.install(layer)
        w = torch.arange(56).reshape(7, 8).to(torch.int8)
        s = torch.arange(1, 8).float().reshape(7, 1)
        layer.weight.weight_loader(layer.weight, w)
        layer.weight_scale.weight_loader(layer.weight_scale, s)
        layer.quant_method.process_weights_after_loading(layer)
        torch.testing.assert_close(layer.weight[:3], w[4:7])
        torch.testing.assert_close(layer.weight_scale[:3], s[4:7])
        self.assertEqual(layer.weight_scale[3].item(), 0)

    def test_quantized_logits_honor_fp32_output(self):
        from sglang.srt.layers.logits_processor import LogitsProcessor

        layer = self.make_layer()
        x = torch.ones(2, 8, dtype=torch.bfloat16)

        def project(x, w, scale, *, output_dtype=None):
            self.assertEqual(w.dtype, torch.int8)
            self.assertEqual(output_dtype, torch.float32)
            return torch.zeros(x.shape[0], w.shape[0], dtype=output_dtype)

        processor = SimpleNamespace(use_fp32_lm_head=True)
        with patch.object(runtime, "linear", side_effect=project):
            y = LogitsProcessor._compute_lm_head(processor, x, layer)
        self.assertEqual(y.dtype, torch.float32)

    def test_markov_and_shared_draft_head_keep_quant_dispatch(self):
        from sglang.srt.models.deepseek_v4_dspark import (
            DeepseekV4ForCausalLMDSpark,
            DSparkV4MarkovHead,
        )

        head = DSparkV4MarkovHead.__new__(DSparkV4MarkovHead)
        torch.nn.Module.__init__(head)
        head.markov_w2 = self.make_layer()
        x = torch.ones(2, 8, dtype=torch.bfloat16)

        def project(x, w, scale, *, output_dtype=None):
            self.assertEqual(w.dtype, torch.int8)
            return torch.zeros(x.shape[0], w.shape[0], dtype=output_dtype or x.dtype)

        with patch.object(runtime, "linear", side_effect=project):
            for flag in (True, False):
                head._opt_markov_w2_bf16 = flag
                self.assertEqual(head.project_bias(x).dtype, torch.float32)
            draft = SimpleNamespace(
                lm_head=head.markov_w2,
                stages=[SimpleNamespace(norm=torch.nn.Identity())],
                _use_fp32_lm_head=True,
                _opt_markov_w2_tp_shard=True,
            )
            y = DeepseekV4ForCausalLMDSpark._logits_from_x_post_hc(draft, x)
            self.assertEqual(y.dtype, torch.float32)

    def test_markov_shard_slices_weight_and_scale_together(self):
        from sglang.srt.models.deepseek_v4_dspark import DSparkV4MarkovHead

        head = DSparkV4MarkovHead.__new__(DSparkV4MarkovHead)
        torch.nn.Module.__init__(head)
        head.markov_w2 = self.make_layer()
        self.load(head.markov_w2)
        head._tp_shard = SimpleNamespace(org_vocab_start=1, org_vocab_end=3)
        head._opt_markov_w2_bf16 = False
        head.vocab_size = 2
        head.get_prev_embeddings = lambda ids: torch.ones(len(ids), 8)
        head._vocab_gather = lambda x: x

        def project(x, w, scale, *, output_dtype=None):
            torch.testing.assert_close(scale, head.markov_w2.weight_scale[1:3])
            torch.testing.assert_close(w, head.markov_w2.weight[1:3])
            return torch.ones(x.shape[0], w.shape[0], dtype=output_dtype)

        with patch.object(runtime, "linear", side_effect=project):
            y = head._apply_step_logits_sharded(
                base_local=torch.zeros(2, 2), token_ids=torch.zeros(2, dtype=torch.long)
            )
        torch.testing.assert_close(y, torch.ones(2, 2))

    def test_empty_projection_has_correct_shape_and_dtype(self):
        w = torch.empty(0, 8, dtype=torch.int8)
        scale = torch.empty(0, 1)
        y = runtime.linear(torch.ones(2, 8), w, scale, output_dtype=torch.float32)
        self.assertEqual(y.shape, (2, 0))
        self.assertEqual(y.dtype, torch.float32)

    def test_markov_greedy_keeps_int8_scale_slice(self):
        from sglang.srt.models.deepseek_v4_dspark import DSparkV4MarkovHead

        head = DSparkV4MarkovHead.__new__(DSparkV4MarkovHead)
        torch.nn.Module.__init__(head)
        head.markov_w2 = self.make_layer()
        self.load(head.markov_w2)
        head._tp_shard = SimpleNamespace(org_vocab_start=1, org_vocab_end=3)
        head._is_dsv41 = True
        head._opt_markov_w2_bf16 = True
        head._shard_group = None
        head._vocab_gather = SimpleNamespace(gather_stacked=None)
        head.get_prev_embeddings = lambda ids: torch.ones(
            len(ids), 8, dtype=torch.bfloat16
        )

        class Base:
            is_cuda = True
            shape = (2, 3, 2)

            def __getitem__(self, index):
                return torch.zeros(2, 2)

        calls = []

        def project(x, w, scale, *, output_dtype=None):
            torch.testing.assert_close(scale, head.markov_w2.weight_scale[1:3])
            self.assertEqual(output_dtype, torch.bfloat16)
            calls.append(1)
            return torch.ones(2, 2, dtype=output_dtype)

        with (
            patch.object(runtime, "linear", side_effect=project),
            patch(
                "sglang.kernels.ops.speculative.dspark.sharded_greedy.sharded_greedy_step",
                return_value=torch.ones(2, dtype=torch.long),
            ),
        ):
            y = head.sample_block_greedy_fused(
                Base(), first_prev_tokens=torch.zeros(2, dtype=torch.long)
            )
        self.assertEqual(len(calls), 3)
        self.assertEqual(y.shape, (2, 3))

    def test_streaming_int8_immediate_and_bf16_unchanged(self):
        from sglang.srt.models.deepseek_v4 import _dequant_fp8_wo_a_streaming

        w = torch.ones(4, 8, dtype=torch.int8)
        scale = torch.ones(4, 1)
        consumed = []

        def source():
            yield "layers.0.attn.wo_a.weight", w
            consumed.append(True)
            yield "layers.0.attn.wo_a.weight_scale", scale

        stream = _dequant_fp8_wo_a_streaming(source())
        name, value = next(stream)
        self.assertEqual(name, "layers.0.attn.wo_a.weight")
        self.assertFalse(consumed)
        for items in [
            [
                ("layers.0.attn.wo_a.weight", w),
                ("layers.0.attn.wo_a.weight_scale", scale),
            ],
            [
                ("layers.0.attn.wo_a.weight_scale", scale),
                ("layers.0.attn.wo_a.weight", w),
            ],
        ]:
            out = dict(_dequant_fp8_wo_a_streaming(iter(items)))
            torch.testing.assert_close(out["layers.0.attn.wo_a.weight"], w)
            torch.testing.assert_close(out["layers.0.attn.wo_a.weight_scale"], scale)
        bf16 = w.to(torch.bfloat16)
        out = dict(
            _dequant_fp8_wo_a_streaming(iter([("layers.0.attn.wo_a.weight", bf16)]))
        )
        torch.testing.assert_close(out["layers.0.attn.wo_a.weight"], bf16)


class TestProjectionBackendSafety(unittest.TestCase):
    def test_default_backend_is_blaslt(self):
        with patch.dict(os.environ):
            os.environ.pop("SGLANG_DSV41_INT8_PROJECTION_BACKEND", None)
            self.assertEqual(runtime._projection_backend(), "blaslt")

    def test_explicit_blaslt_is_supported(self):
        with patch.dict(os.environ, {"SGLANG_DSV41_INT8_PROJECTION_BACKEND": "blaslt"}):
            self.assertEqual(runtime._projection_backend(), "blaslt")

    def test_legacy_and_unknown_backends_are_rejected(self):
        for mode in ("auto", "native", "reference", "unknown"):
            with self.subTest(mode=mode), patch.dict(os.environ, {"SGLANG_DSV41_INT8_PROJECTION_BACKEND": mode}):
                with self.assertRaisesRegex(ValueError, "backend"):
                    runtime.linear(torch.ones(2, 8), torch.ones(4, 8, dtype=torch.int8), torch.ones(4, 1))


if __name__ == "__main__":
    unittest.main()
