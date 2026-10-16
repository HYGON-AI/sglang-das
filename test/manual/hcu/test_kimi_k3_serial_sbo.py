"""Native HCU precision gate for K3 serial SBO (requires an idle gfx938).

Run from the repository root with PYTHONPATH=python and one visible device:
  python test/manual/hcu/test_kimi_k3_serial_sbo.py --output /path/result.json

Checks the migrated project/MLP methods against the native strategy-3 path,
including changing inputs during graph replay. This is not a model-quality
evaluation or a multi-rank DeepEP integration test.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from lightop.quant import per_token_quant_int8
from torch import nn

from sglang.srt.layers import k3_sbo_gemm as helper
from sglang.srt.layers.quantization.slimquant_w4a8 import SlimQuantW4A8Int8LinearMethod
from sglang.srt.models import kimi_k3 as k3


class NativeLinear(nn.Module):
    """Use the actual target quantizer and BLASLt apply(), without model loading."""

    def __init__(self, n, k):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randint(-128, 128, (n, k), device="cuda", dtype=torch.int8),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty(n, 1, device="cuda", dtype=torch.float32).uniform_(
                0.0003, 0.007
            ),
            requires_grad=False,
        )
        self.quant_method = SlimQuantW4A8Int8LinearMethod.__new__(
            SlimQuantW4A8Int8LinearMethod
        )
        self.quant_method.w8a8_strategy = 3
        self.tp_size = 1
        self.bias = None
        self.gather_output = False
        self.input_is_parallel = True

    def forward(self, x, input_quant_args=None):
        return self.quant_method.apply(self, x, input_quant_args=input_quant_args), None


@torch.inference_mode()
def main(output):
    assert torch.version.hip, "HCU required"
    assert helper._supported_device(0), "This gate must exercise the gfx938 kernel"
    torch.manual_seed(20261015)
    rows = []
    result = dict(
        passed=False,
        torch=torch.__version__,
        arch=torch.cuda.get_device_properties(0).gcnArchName,
        hashes={
            str(Path(m.__file__)): hashlib.sha256(
                Path(m.__file__).read_bytes()
            ).hexdigest()
            for m in [helper, k3]
        },
        checks=rows,
    )

    def save():
        output.write_text(json.dumps(result, indent=2))

    def equal(actual, expected, **metadata):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        rows.append(dict(**metadata, bitwise=True))

    try:
        mlp = k3.KimiK3MLP.__new__(k3.KimiK3MLP)
        nn.Module.__init__(mlp)
        mlp.gate_up_proj = NativeLinear(12288, 7168)
        mlp.down_proj = NativeLinear(7168, 6144)
        mlp.act_fn = k3.SituAndMul(beta=4.0, linear_beta=25.0)
        mlp._dp_attention = False
        mlp._dense_attn_tp = False

        # Both real weight shapes, every small-M kernel specialization, and
        # native fallbacks covering the user's larger batch sizes.
        for name, linear in [("gate_up", mlp.gate_up_proj), ("down", mlp.down_proj)]:
            for m in list(range(1, 17)) + [17, 24, 32, 40, 48, 64, 80, 96, 128]:
                x = torch.randn(
                    m, linear.weight.shape[1], device="cuda", dtype=torch.bfloat16
                )
                assert helper._eligible(linear, x) == (m <= 16)
                for reuse in [False, True]:
                    qa = per_token_quant_int8(x) if reuse else None
                    equal(
                        helper.project(linear, x, qa)[0],
                        linear(x, qa)[0],
                        phase="projection",
                        projection=name,
                        tokens=m,
                        reuse=reuse,
                    )
            print(
                json.dumps(dict(phase="projection", projection=name, checks=len(rows))),
                flush=True,
            )
            save()

        # Independent INT64 sampled-channel reference, including maximum
        # magnitude INT8 products. The largest possible dot fits in INT32.
        for name, linear in [("gate_up", mlp.gate_up_proj), ("down", mlp.down_proj)]:
            channels = torch.tensor([0, 1, 31, 63, 64, 511, linear.weight.shape[0] - 1])
            for stress in ["random", "zero", "positive", "negative"]:
                x = torch.empty(
                    16, linear.weight.shape[1], device="cuda", dtype=torch.bfloat16
                )
                q = torch.randint(-128, 128, x.shape, device="cuda", dtype=torch.int8)
                scales = torch.empty(16, 1, device="cuda").uniform_(0.001, 0.5)
                if stress == "zero":
                    q.zero_()
                elif stress == "positive":
                    q.fill_(127)
                    linear.weight.fill_(127)
                elif stress == "negative":
                    q.fill_(-128)
                    linear.weight.fill_(-128)
                acc = (
                    q.cpu().to(torch.int64)
                    @ linear.weight[channels.to("cuda")].cpu().to(torch.int64).T
                )
                ref = (
                    acc.float()
                    * (scales.cpu() * linear.weight_scale[channels.to("cuda")].cpu().T)
                ).bfloat16()
                actual = helper.project(linear, x, (q, scales))[0]
                equal(
                    actual[:, channels.to("cuda")].cpu(),
                    ref,
                    phase="int64_reference",
                    projection=name,
                    stress=stress,
                )
                equal(
                    actual,
                    linear(x, (q, scales))[0],
                    phase="native_stress",
                    projection=name,
                    stress=stress,
                )
            linear.weight.random_(-128, 128)
        save()

        def split(x, qa):
            gate_up = mlp._serial_sbo_gate_up(x, qa)[0]
            return mlp._serial_sbo_down(gate_up)

        for fused in [False, True]:
            mlp._use_fused_situ_int8 = fused
            mlp.__dict__.pop("_fused_situ_int8_op", None)
            assert (mlp._fused_situ_int8_op is not None) == fused
            for m in [
                1,
                2,
                3,
                4,
                8,
                10,
                12,
                15,
                16,
                17,
                24,
                32,
                40,
                48,
                64,
                80,
                96,
                128,
            ]:
                for reuse in [False, True]:
                    x = torch.randn(m, 7168, device="cuda", dtype=torch.bfloat16)
                    qa = per_token_quant_int8(x) if reuse else None
                    for _ in range(3):
                        split(x, qa)
                        mlp(x, input_quant_args=qa)
                    equal(
                        split(x, qa),
                        mlp(x, input_quant_args=qa),
                        phase="mlp_eager",
                        fused=fused,
                        tokens=m,
                        reuse=reuse,
                    )
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = split(x, qa)
                    # Mutate graph inputs and explicit quantization buffers;
                    # comparing one unchanged replay would miss stale data.
                    for amplitude in [0.0, 0.001, 1.0, 1000.0]:
                        x.copy_(torch.randn_like(x) * amplitude)
                        if qa is not None:
                            new_q, new_scale = per_token_quant_int8(x)
                            qa[0].copy_(new_q)
                            qa[1].copy_(new_scale)
                        expected = mlp(x, input_quant_args=qa)
                        graph.replay()
                        equal(
                            captured,
                            expected,
                            phase="mlp_graph",
                            fused=fused,
                            tokens=m,
                            reuse=reuse,
                            amplitude=amplitude,
                        )
                    graph.reset()
                print(
                    json.dumps(
                        dict(phase="mlp", fused=fused, tokens=m, checks=len(rows))
                    ),
                    flush=True,
                )
                save()
        torch.cuda.synchronize()
        result["passed"] = True
    except Exception as exc:
        result["error"] = repr(exc)
        raise
    finally:
        save()
    print(
        json.dumps(dict(passed=True, checks=len(rows), output=str(output))), flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
