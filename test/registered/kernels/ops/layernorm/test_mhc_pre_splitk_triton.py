import pytest
import torch

from sglang.kernels.ops.layernorm.mhc_pre_splitk_triton import (
    choose_split,
    is_supported_mhc_pre_splitk,
    mhc_pre_partial_splitk,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")


@pytest.mark.parametrize("variant", ["ieee", "split3"])
@pytest.mark.parametrize("num_tokens", [4096, 12288])
def test_mhc_pre_partial_splitk_matches_reference(num_tokens, variant):
    """Reducing the partials over the split axis must reproduce x @ fn.T."""
    if not torch.cuda.is_available():
        pytest.skip("Triton mHC kernel requires a GPU")

    torch.manual_seed(0)
    device = torch.device("cuda")
    hidden_size, hc_mult = 4096, 4
    hc_hidden_size = hc_mult * hidden_size
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult

    x = torch.randn(num_tokens, hc_hidden_size, device=device, dtype=torch.bfloat16) * 0.1
    fn = torch.randn(hc_mult3, hc_hidden_size, device=device, dtype=torch.float32) * 0.01

    out, sqrsum = mhc_pre_partial_splitk(x, fn, variant=variant)

    split = choose_split(hc_hidden_size)
    assert out.shape == (split, num_tokens, hc_mult3)
    assert sqrsum.shape == (split, num_tokens)

    ref_out = x.float() @ fn.t()
    ref_sqrsum = (x.float() ** 2).sum(-1)

    # split3 reconstructs the FP32 weight to ~24 mantissa bits from three BF16
    # terms, so it tracks the IEEE path up to the FP32 accumulation order.
    rtol = 2e-2 if variant == "split3" else 1e-3
    torch.testing.assert_close(out.sum(0), ref_out, rtol=rtol, atol=rtol)
    torch.testing.assert_close(sqrsum.sum(0), ref_sqrsum, rtol=1e-3, atol=1e-3)


def test_choose_split_and_support_predicate():
    assert choose_split(16384) == 8
    assert choose_split(1024) == 8
    assert choose_split(640) == 1  # only the unsplit path divides 640 into blocks
    assert choose_split(100) == 0  # not a multiple of the block size at all

    x = torch.empty(4, 4096, dtype=torch.bfloat16)
    fn = torch.empty(8, 4096, dtype=torch.float32)
    assert is_supported_mhc_pre_splitk(x, fn)
    # BF16 weights and N > 32 are outside the kernel's contract.
    assert not is_supported_mhc_pre_splitk(x, fn.to(torch.bfloat16))
    assert not is_supported_mhc_pre_splitk(x, torch.empty(33, 4096, dtype=torch.float32))
