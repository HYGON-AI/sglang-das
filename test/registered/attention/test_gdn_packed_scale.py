import pytest
import torch

from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cuda_ci,
    register_hcu_ci,
)

register_amd_ci(est_time=60, stage="stage-b", runner_config="1-gpu-small-amd")
register_cuda_ci(est_time=60, stage="stage-b", runner_config="1-gpu-small")
register_hcu_ci(est_time=60, suite="stage-b-test-1-hcu-small")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA/HIP device")
def test_compiled_packed_decode_matches_eager_state_and_output():
    torch.manual_seed(42)
    bs, heads, dim = 4, 4, 128
    qkv = torch.randn(bs, 3 * heads * dim, device="cuda", dtype=torch.bfloat16) * 0.1
    a = torch.randn(bs, heads, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(bs, heads, device="cuda", dtype=torch.bfloat16) * 0.1
    A_log = torch.full((heads,), -1.0, device="cuda", dtype=torch.float32)
    dt_bias = torch.full((heads,), -1.0, device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(1, bs + 1, device="cuda", dtype=torch.int32)
    state = torch.randn(bs + 1, heads, dim, dim, device="cuda") * 0.01
    reference_state, compiled_state = state.clone(), state.clone()

    def decode(state, qkv, a, b, A_log, dt_bias, indices):
        out = torch.empty((bs, 1, heads, dim), device=state.device, dtype=qkv.dtype)
        fused_recurrent_gated_delta_rule_packed_decode(
            qkv, a, b, A_log, dt_bias, dim**-0.5, state, out, indices, True
        )
        return out

    compiled = torch.compile(decode, mode="default")
    for _ in range(2):
        expected = decode(reference_state, qkv, a, b, A_log, dt_bias, indices)
        actual = compiled(compiled_state, qkv, a, b, A_log, dt_bias, indices)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-3)
        torch.testing.assert_close(
            compiled_state, reference_state, rtol=1e-5, atol=1e-5
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
