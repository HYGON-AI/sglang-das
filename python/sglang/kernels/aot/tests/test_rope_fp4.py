# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Test: sgl_kernel.rope_fp4_fake_quant vs Python reference (HIP/DCU only).

Self-contained: reference implementations of rope_tail / fake_quant_fp4 /
fake_quant_compressed_kv are inlined so this file has no dependency on the
sglang source tree being in PYTHONPATH.
"""

import pytest
import torch

# Skip at collection time; pytestmark cannot guard module-level imports.
if not torch.cuda.is_available() or torch.version.hip is None:
    pytest.skip("requires HIP/DCU device", allow_module_level=True)

_sgl = pytest.importorskip("sgl_kernel", reason="sgl_kernel wheel not found")
if not hasattr(_sgl, "rope_fp4_fake_quant"):
    pytest.skip(
        "rope_fp4_fake_quant not in this sgl_kernel build "
        "(rebuild with setup_rocm.py after adding rope_fp4.hip)",
        allow_module_level=True,
    )
rope_fp4_fake_quant = _sgl.rope_fp4_fake_quant

DEVICE = "cuda"
ROPE_DIM = 64

# ---------------------------------------------------------------------------
# Reference implementations (pure Python/PyTorch, match dsv41_sparse.py)
# ---------------------------------------------------------------------------

def _rope_tail(x: torch.Tensor, freqs: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """Rotate the last rope_dim features of x with complex freqs [T, rope_dim//2]."""
    head, tail = x[..., :-rope_dim], x[..., -rope_dim:]
    tc = torch.view_as_complex(tail.float().unflatten(-1, (-1, 2)).contiguous())
    f = freqs.view(x.shape[0], *([1] * (x.ndim - 2)), rope_dim // 2)
    rotated = torch.view_as_real(tc * f).flatten(-2).to(x.dtype)
    return torch.cat([head, rotated], dim=-1)


def _round_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to FP4 E2M1 grid: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}."""
    ax = x.abs()
    step = torch.where(ax < 2.0, torch.full_like(ax, 0.5),
           torch.where(ax < 4.0, torch.full_like(ax, 1.0),
                                  torch.full_like(ax, 2.0)))
    return (ax / step).round() * step * x.sign()


def _fake_quant_fp4(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """UE8M0 per-block FP4 fake-quant (matches fake_quant_fp4 in torch_quant.py)."""
    orig = x
    x = x.float()
    B = x.shape[0]
    xr = x.reshape(B, -1, block)
    amax = xr.abs().amax(dim=-1, keepdim=True)
    # UE8M0: ceil to next power-of-2, floor = 6 * 2^-126
    FLOOR = 6.0 * (2.0 ** -126)
    a = torch.clamp(amax, min=FLOOR) / 6.0
    # Convert to power-of-2 by rounding the float exponent up.
    exp = torch.frexp(a)[1].float() - 1          # floor(log2(a))
    # If a is not exactly a power of 2, ceil to next.
    is_exact = (a == 2.0 ** exp)
    exp = torch.where(is_exact, exp, exp + 1)
    scale = (2.0 ** exp).reshape(B, -1, 1)        # [B, nblocks, 1]
    q = torch.clamp(xr / scale, -6.0, 6.0)
    q = _round_fp4_e2m1(q)
    return (q * scale).reshape_as(orig).to(orig.dtype)


def _fake_quant_compressed_kv(x: torch.Tensor, block: int = 16) -> torch.Tensor:
    """Per-block E4M3FN fake-quant (matches fake_quant_compressed_kv)."""
    orig = x
    x = x.float()
    B = x.shape[0]
    xr = x.reshape(B, -1, block)
    amax = xr.abs().amax(dim=-1, keepdim=True)
    # E4M3FN scale: clamp amax/6 to [2^-9, 448] then round to E4M3FN grid.
    s = torch.clamp(amax / 6.0, 1.953125e-3, 448.0)
    scale = s.to(torch.float8_e4m3fn).float().reshape(B, -1, 1)
    q = torch.clamp(xr / scale, -6.0, 6.0)
    q = _round_fp4_e2m1(q)
    return (q * scale).reshape_as(orig).to(orig.dtype)


def _ref(x, freqs, compressed_kv):
    rotated = _rope_tail(x, freqs, ROPE_DIM)
    if compressed_kv:
        return _fake_quant_compressed_kv(rotated)
    return _fake_quant_fp4(rotated)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B", [1, 4, 16, 128])
@pytest.mark.parametrize("D", [512, 128])
@pytest.mark.parametrize("compressed_kv", [False, True])
def test_rope_fp4_2d(B, D, compressed_kv):
    """2-D input [B, D] covering all four (D, compressed_kv) combos."""
    x = torch.randn(B, D, dtype=torch.bfloat16, device=DEVICE)
    freqs = torch.polar(
        torch.ones(B, ROPE_DIM // 2, device=DEVICE),
        torch.rand(B, ROPE_DIM // 2, device=DEVICE) * 6.28,
    ).to(torch.complex64)

    ref = _ref(x.clone(), freqs, compressed_kv)
    out = rope_fp4_fake_quant(x.clone(), freqs, ROPE_DIM, compressed_kv)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("B,H", [(4, 8), (16, 4)])
def test_rope_fp4_3d(B, H):
    """3-D input [B, H, D] — index-Q path where rows_per_token > 1."""
    D = 128
    x = torch.randn(B, H, D, dtype=torch.bfloat16, device=DEVICE)
    freqs = torch.polar(
        torch.ones(B, ROPE_DIM // 2, device=DEVICE),
        torch.rand(B, ROPE_DIM // 2, device=DEVICE) * 6.28,
    ).to(torch.complex64)

    ref = _ref(
        x.reshape(-1, D).clone(),
        freqs.repeat_interleave(H, dim=0),
        compressed_kv=False,
    )
    out = rope_fp4_fake_quant(x.clone(), freqs, ROPE_DIM, False)

    torch.testing.assert_close(out.reshape(-1, D), ref, rtol=1e-2, atol=1e-2)


def test_rope_fp4_unsupported_D():
    """Kernel must reject D values other than 128 and 512."""
    x = torch.randn(4, 256, dtype=torch.bfloat16, device=DEVICE)
    freqs = torch.polar(
        torch.ones(4, ROPE_DIM // 2, device=DEVICE),
        torch.zeros(4, ROPE_DIM // 2, device=DEVICE),
    ).to(torch.complex64)
    with pytest.raises(RuntimeError):
        rope_fp4_fake_quant(x, freqs, ROPE_DIM, False)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
