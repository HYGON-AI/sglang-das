"""CPU loader/dispatch guards; device arithmetic is covered by the HCU probe."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization import dsv41_channelwise_fp8 as runtime
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestChannelwiseRuntime(unittest.TestCase):
    def test_real_v41_config_normalization_is_supported(self):
        from sglang.srt.configs.deepseek_v41 import (
            _per_channel_fp8_as_compressed_tensors,
        )
        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsConfig,
        )

        config = CompressedTensorsConfig.from_config(
            _per_channel_fp8_as_compressed_tensors(
                {
                    "quant_method": "fp8",
                    "is_per_channel": True,
                    "activation_scheme": "dynamic",
                }
            )
        )
        with (
            patch.dict(os.environ, {"SGLANG_DSV41_CHANNELWISE_FP8_RUNTIME": "1"}),
            patch("sglang.srt.utils.is_hcu", return_value=True),
        ):
            self.assertTrue(runtime.enabled(config))

    def test_disabled_is_compatible_with_other_quantization(self):
        with patch.dict(os.environ, {"SGLANG_DSV41_CHANNELWISE_FP8_RUNTIME": "0"}):
            self.assertFalse(runtime.enabled(None))
            self.assertFalse(runtime.enabled(object()))

    def test_dspark_fp32_output_options_keep_fp8_dispatch(self):
        from sglang.srt.models.deepseek_v4_dspark import (
            DeepseekV4ForCausalLMDSpark,
            DSparkV4MarkovHead,
        )

        head = DSparkV4MarkovHead.__new__(DSparkV4MarkovHead)
        torch.nn.Module.__init__(head)
        head.markov_w2 = torch.nn.Linear(8, 4, bias=False)
        runtime.install(head.markov_w2)
        x = torch.ones(2, 8, dtype=torch.bfloat16)
        seen = []

        def project(x, weight, scale, *, output_dtype=None):
            seen.append(x.dtype)
            self.assertEqual(weight.dtype, torch.float8_e4m3fn)
            return torch.zeros(
                x.shape[0], weight.shape[0], dtype=output_dtype or x.dtype
            )

        with patch.object(runtime, "linear", side_effect=project):
            for flag in (True, False):
                head._opt_markov_w2_bf16 = flag
                head.project_bias(x)
            draft = SimpleNamespace(
                lm_head=head.markov_w2,
                stages=[SimpleNamespace(norm=torch.nn.Identity())],
                _use_fp32_lm_head=True,
                _opt_markov_w2_tp_shard=True,
            )
            logits = DeepseekV4ForCausalLMDSpark._logits_from_x_post_hc(draft, x)
        self.assertEqual(seen, [torch.bfloat16, torch.float32, torch.float32])
        self.assertEqual(logits.dtype, torch.float32)

    def test_target_fp32_logits_keep_fp8_weights_and_activation_dispatch(self):
        from sglang.srt.layers.logits_processor import LogitsProcessor

        head = torch.nn.Linear(8, 4, bias=False)
        runtime.install(head)
        x = torch.ones(2, 8, dtype=torch.bfloat16)
        processor = SimpleNamespace(use_fp32_lm_head=False)

        def project(a, weight, scale, *, output_dtype=None):
            self.assertEqual(a.dtype, torch.bfloat16)
            self.assertEqual(weight.dtype, torch.float8_e4m3fn)
            self.assertEqual(
                output_dtype,
                torch.float32 if processor.use_fp32_lm_head else torch.bfloat16,
            )
            return torch.zeros(a.shape[0], weight.shape[0], dtype=output_dtype)

        with patch.object(runtime, "linear", side_effect=project):
            for flag in (False, True):
                processor.use_fp32_lm_head = flag
                y = LogitsProcessor._compute_lm_head(processor, x, head)
                self.assertEqual(y.dtype, torch.float32 if flag else torch.bfloat16)

    def test_other_quantized_lm_heads_keep_existing_call_contract(self):
        from sglang.srt.layers.logits_processor import LogitsProcessor

        x = torch.ones(2, 8, dtype=torch.bfloat16)
        y = torch.zeros(2, 4, dtype=torch.bfloat16)
        calls = []

        def apply(layer, hidden, bias):
            self.assertIs(hidden, x)
            calls.append(layer)
            return y

        head = SimpleNamespace(
            weight=torch.empty(4, 8, dtype=torch.bfloat16),
            quant_method=SimpleNamespace(apply=apply),
        )
        processor = SimpleNamespace(use_fp32_lm_head=True)
        self.assertIs(LogitsProcessor._compute_lm_head(processor, x, head), y)
        self.assertEqual(calls, [head])

    def test_unquantized_lm_head_fp32_output_unchanged(self):
        from sglang.srt.layers.logits_processor import LogitsProcessor

        x = torch.randn(2, 8, dtype=torch.bfloat16)
        head = torch.nn.Linear(8, 4, bias=False).to(torch.bfloat16)
        processor = SimpleNamespace(use_fp32_lm_head=True)
        actual = LogitsProcessor._compute_lm_head(processor, x, head)
        torch.testing.assert_close(actual, x.float() @ head.weight.float().T)

    def test_lm_head_lora_rejects_fp8_before_allocating_wrapper_buffers(self):
        from sglang.srt.lora.layers import ParallelLMHeadWithLoRA

        head = torch.nn.Linear(8, 4, bias=False)
        runtime.install(head)
        head.embedding_dim = 8
        head.org_vocab_size = 4
        head.tp_size = 1
        with self.assertRaisesRegex(ValueError, "does not support lm_head LoRA"):
            ParallelLMHeadWithLoRA(head, lora_backend=None)

    def test_load_requires_both_fp8_weight_and_channel_scale(self):
        layer = torch.nn.Linear(8, 4, bias=False)
        runtime.install(layer)
        weight = torch.arange(32).reshape(4, 8).to(torch.float8_e4m3fn)
        scale = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
        with self.assertRaises(ValueError):
            layer.quant_method.process_weights_after_loading(layer)
        layer.weight.weight_loader(layer.weight, weight)
        with self.assertRaises(ValueError):
            layer.quant_method.process_weights_after_loading(layer)
        layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, scale)
        layer.quant_method.process_weights_after_loading(layer)
        torch.testing.assert_close(layer.weight.float(), weight.float())
        torch.testing.assert_close(layer.weight_scale_inv, scale)

    def test_rejects_non_fp8_or_block_scales(self):
        layer = torch.nn.Linear(8, 4, bias=False)
        runtime.install(layer)
        with self.assertRaises(ValueError):
            layer.weight.weight_loader(layer.weight, torch.ones(4, 8))
        with self.assertRaises(ValueError):
            layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, torch.ones(1))

    def test_weight_and_scale_use_same_tp_slice_and_padding(self):
        layer = torch.nn.Linear(8, 4, bias=False)
        layer.weight.output_dim = 0

        def loader(param, value):
            self.assertEqual(param.output_dim, 0)
            # Simulate a final vocabulary partition containing three real rows.
            param.data[:3].copy_(value[4:7])
            param.data[3:].zero_()

        layer.weight.weight_loader = loader
        runtime.install(layer)
        weight = torch.arange(56).reshape(7, 8).to(torch.float8_e4m3fn)
        scale = torch.arange(1, 8).float().reshape(7, 1)
        layer.weight.weight_loader(layer.weight, weight)
        layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, scale)
        torch.testing.assert_close(layer.weight[:3].float(), weight[4:7].float())
        torch.testing.assert_close(layer.weight_scale_inv[:3], scale[4:7])
        self.assertEqual(layer.weight_scale_inv[3].item(), 0)

    def test_grouped_keeps_group_scale_alignment(self):
        x = torch.randn(5, 2, 8)
        layer = SimpleNamespace(
            weight=torch.randn(6, 8), weight_scale_inv=torch.arange(6).reshape(6, 1)
        )

        def project(a, w, s):
            self.assertEqual(a.shape, (2, 5, 8))
            torch.testing.assert_close(s[:, :, 0], torch.arange(6).reshape(2, 3))
            return torch.matmul(a, w.transpose(-1, -2))

        with patch.object(runtime, "linear", side_effect=project):
            actual = runtime.grouped(x, layer, 2, 3)
        expected = torch.einsum("tgd,grd->tgr", x, layer.weight.reshape(2, 3, 8))
        torch.testing.assert_close(actual, expected)

    def test_fp32_batches_dispatch_separately(self):
        x = torch.randn(2, 3, 8)
        w = torch.randn(2, 4, 8).to(torch.float8_e4m3fn)
        scale = torch.ones(2, 4, 1)
        calls = []

        def quant(a):
            return a.to(torch.float8_e4m3fn), torch.ones(*a.shape[:-1], 1)

        def gemm(a, b, out):
            # Reproduce the backend restriction instead of silently accepting
            # the bad batched-FP32 dispatch which this regression guards.
            self.assertEqual(b[0].ndim, 2)
            calls.append(b[0].shape)
            out.copy_((a[0].float() * a[1]) @ (b[0].float() * b[1]).T)

        with patch.dict(
            sys.modules,
            {
                "deepgemm": SimpleNamespace(fp8_gemm=gemm),
                "lightop.quant": SimpleNamespace(per_token_quant_fp8=quant),
            },
        ):
            y = runtime.linear(x, w, scale)
        self.assertEqual(len(calls), 2)
        torch.testing.assert_close(
            y, x.to(torch.float8_e4m3fn).float() @ w.float().transpose(-1, -2)
        )


if __name__ == "__main__":
    unittest.main()
