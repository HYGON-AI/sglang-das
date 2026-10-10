"""Numerical regression for large-group DeepGEMM activation fallback."""

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.moe.ep_moe_kernels import silu_and_mul_masked_post_quant_fwd
from sglang.test.ci.ci_register import register_hcu_ci

register_hcu_ci(est_time=30, suite="stage-b-test-1-hcu-small")


@pytest.mark.parametrize("width", [512, 768, 3072])
@pytest.mark.parametrize("limit", [None, 10.0])
def test_large_group_activation_matches_fp32_reference(width, limit):
    torch.manual_seed(108)
    x = (torch.randn((2, 4, 2 * width), device="cuda") * 20).bfloat16()
    masked = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    output = torch.empty((2, 4, width), device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.empty((2, 4, 1), device="cuda", dtype=torch.float32)
    silu_and_mul_masked_post_quant_fwd(
        x, output, scale, width, masked, swiglu_limit=limit, silu_mul_keep_fp32=True
    )
    gate, up = x.chunk(2, dim=-1)
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    ref = F.silu(gate.float()) * up.float()
    expected_scale = ref.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10) / 448
    for e, count in enumerate([3, 1]):
        torch.testing.assert_close(
            scale[e, :count], expected_scale[e, :count], rtol=1e-4, atol=1e-6
        )
        restored = output[e, :count].float() * scale[e, :count]
        error = (restored - ref[e, :count]).abs()
        # E4M3 has at most half a quantization bin of rounding error.
        assert torch.all(
            error <= ref[e, :count].abs() * 0.063 + scale[e, :count] * 0.01
        )


def test_deepgemm_large_group_retains_clamp(monkeypatch):
    from sglang.srt.layers.moe.moe_runner import deep_gemm as runner

    monkeypatch.setattr(runner.deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", False)
    x = torch.full((2, 4, 1024), 20.0, device="cuda", dtype=torch.bfloat16)
    masked = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    out, scale = runner._varlen_deep_gemm_silu_mul_quant(
        x, masked, 512, 1, swiglu_limit=10.0
    )
    torch.cuda.synchronize()
    expected = F.silu(torch.tensor(10.0, device="cuda")) * 10
    for e, count in enumerate([3, 1]):
        torch.testing.assert_close(
            out[e, :count].float() * scale[e, :count],
            expected.expand(count, 512),
            rtol=0.063,
            atol=0.01,
        )


def test_packed_scale_schedule_retains_clamp():
    torch.manual_seed(109)
    width = 2048
    x = (torch.randn((2, 4, 2 * width), device="cuda") * 20).bfloat16()
    masked = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    output = torch.empty((2, 4, width), device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.empty((2, 1, 4), device="cuda", dtype=torch.int32)
    silu_and_mul_masked_post_quant_fwd(
        x,
        output,
        scale,
        512,
        masked,
        scale_ue8m0=True,
        num_real_tokens=4,
        topk=1,
        swiglu_limit=10.0,
        silu_mul_keep_fp32=True,
    )
    gate, up = x.chunk(2, dim=-1)
    ref = F.silu(gate.clamp(max=10).float()) * up.clamp(-10, 10).float()
    ref = ref.reshape(2, 4, 4, 512)
    expected = ref.abs().amax(-1).clamp(min=1e-10) / 448
    expected = torch.exp2(torch.ceil(torch.log2(expected)))
    decoded = torch.stack(
        [
            (((scale[:, 0, :] >> (8 * i)) & 255) << 23).contiguous().view(torch.float32)
            for i in range(4)
        ],
        dim=-1,
    )
    for e, count in enumerate([3, 1]):
        torch.testing.assert_close(
            decoded[e, :count], expected[e, :count], rtol=0, atol=0
        )
        restored = (
            output[e, :count].float().reshape(count, 4, 512)
            * decoded[e, :count, :, None]
        )
        assert torch.all(
            (restored - ref[e, :count]).abs()
            <= ref[e, :count].abs() * 0.063 + decoded[e, :count, :, None] * 0.01
        )
