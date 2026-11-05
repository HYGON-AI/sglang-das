"""Exercise the actual step_bias method without importing GPU-only model modules."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize("bf16", [False, True])
def test_int8_sharded_bias_returns_quantized_projection(bf16):
    root = next(p for p in Path(__file__).resolve().parents if (p / "python/sglang").is_dir())
    source = root / "python/sglang/srt/models/deepseek_v4_dspark.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DSparkV4MarkovHead")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "step_bias")
    method.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    class Int8Method:
        pass
    projected = torch.tensor([[1.25, 2.5]], dtype=torch.bfloat16 if bf16 else torch.float32)
    linear = Mock(return_value=projected)
    ns = dict(torch=torch, channelwise_int8=SimpleNamespace(ChannelwiseInt8Method=Int8Method, linear=linear), channelwise_fp8=SimpleNamespace(ChannelwiseFp8Method=type("Fp8Method", (), {})), F=SimpleNamespace(linear=Mock(side_effect=AssertionError("INT8 must not fall through to ordinary GEMM"))))
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), ns)
    weight = torch.arange(12, dtype=torch.int8).reshape(4, 3)
    scales = torch.tensor([1., 2., 3., 4.])
    latent = torch.ones(1, 3)
    owner = SimpleNamespace(get_prev_embeddings=lambda ids: latent, _tp_shard=SimpleNamespace(org_vocab_start=1, org_vocab_end=3), markov_w2=SimpleNamespace(weight=weight, weight_scale=scales, quant_method=Int8Method()), _opt_markov_w2_bf16=bf16)
    state = object()
    actual, returned_state = ns["step_bias"](owner, torch.tensor([0]), hidden_states=None, state=state)
    assert actual is projected and returned_state is state
    assert linear.call_count == 1
    torch.testing.assert_close(linear.call_args.args[1], weight[1:3])
    torch.testing.assert_close(linear.call_args.args[2], scales[1:3])
    assert linear.call_args.kwargs["output_dtype"] == projected.dtype
