"""HCU-only numerics for ``diffusion.minimax_h3_rmsnorm_scale_shift``.

The fused kernel replaces the eager DiT block chain

    ``_modulate_scale_shift(norm(x), shift[idx], scale[idx])``

with a single launch. It is *not* bit-exact against that chain: the eager
``nn.RMSNorm`` and the kernel sum the squares in different orders, and the
kernel rounds the normalized value to bf16 one step earlier. The assertion
is therefore a tolerance, not ``torch.equal`` -- which is exactly why the
call site is behind ``SGLANG_MINIMAX_H3_FUSED_RMSNORM_ADALN`` and gated on
HCU. Needs real HCU hardware, so this lives under ``test/manual``.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.diffusion import (
    can_use_minimax_h3_rmsnorm_scale_shift,
    indexed_rmsnorm_scale_shift_bf16,
)
from sglang.srt.utils import is_hcu

HIDDEN = 5376
INDICES = 4


def _eager_reference(x, weight, shift, scale, indices, eps):
    normed = torch.nn.functional.rms_norm(x, (HIDDEN,), weight, eps)
    return normed * (1.0 + scale.index_select(0, indices)) + shift.index_select(
        0, indices
    )


@unittest.skipUnless(torch.cuda.is_available() and is_hcu(), "requires HCU hardware")
class TestMinimaxH3RmsnormScaleShift(unittest.TestCase):
    def _make_inputs(self, rows):
        device = torch.device("cuda:0")
        x = torch.randn(rows, HIDDEN, device=device, dtype=torch.bfloat16)
        weight = torch.randn(HIDDEN, device=device, dtype=torch.bfloat16) * 0.1 + 1
        shift = torch.randn(INDICES, HIDDEN, device=device, dtype=torch.bfloat16)
        scale = torch.randn(INDICES, HIDDEN, device=device, dtype=torch.bfloat16) * 0.1
        indices = torch.randint(0, INDICES, (rows,), device=device)
        return x, weight, shift, scale, indices

    def test_matches_eager_chain(self):
        torch.manual_seed(20261110)
        for rows in (1, 16, 4096):
            with self.subTest(rows=rows):
                x, weight, shift, scale, indices = self._make_inputs(rows)
                eps = 1e-6
                original = x.clone()

                self.assertTrue(
                    can_use_minimax_h3_rmsnorm_scale_shift(
                        x, weight, shift, scale, indices
                    )
                )

                with torch.inference_mode():
                    reference = _eager_reference(x, weight, shift, scale, indices, eps)
                    out = indexed_rmsnorm_scale_shift_bf16(
                        x, weight, shift, scale, indices, eps
                    )

                self.assertEqual(out.shape, x.shape)
                self.assertEqual(out.dtype, x.dtype)
                torch.testing.assert_close(out, reference, atol=0.05, rtol=0.02)
                # The kernel writes a fresh buffer; x must be untouched.
                self.assertTrue(torch.equal(x, original))

    def test_dit_projection_input_dispatch(self):
        from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
            MiniMaxH3DiTBlock,
        )

        x, weight, shift, scale, indices = self._make_inputs(rows=17)
        norm = torch.nn.RMSNorm(HIDDEN, eps=1e-6, device=x.device, dtype=x.dtype)
        block = SimpleNamespace(_fused_fp32_modulation=False)
        prepare = MiniMaxH3DiTBlock._prepare_projection_input
        flag = "SGLANG_MINIMAX_H3_FUSED_RMSNORM_ADALN"
        kernel = "sglang.kernels.ops.diffusion.indexed_rmsnorm_scale_shift_bf16"
        with torch.inference_mode():
            norm.weight.copy_(weight)
            with patch.dict("os.environ", {flag: "0"}):
                expected, affine = prepare(block, norm, x, shift, scale, indices)
                self.assertIsNone(affine)
            with (
                patch.dict("os.environ", {flag: "1"}),
                patch(kernel, wraps=indexed_rmsnorm_scale_shift_bf16) as fused,
            ):
                actual, affine = prepare(block, norm, x, shift, scale, indices)
                fused.assert_called_once()
                self.assertIsNone(affine)
                torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.02)

                # Compressed FP32 modulation goes through its own preparation.
                fused.reset_mock()
                block._fused_fp32_modulation = True
                prepare(block, norm, x, shift.float(), scale.float(), indices)
                fused.assert_not_called()

    def test_rejects_unsupported_inputs(self):
        x, weight, shift, scale, indices = self._make_inputs(8)

        self.assertTrue(
            can_use_minimax_h3_rmsnorm_scale_shift(x, weight, shift, scale, indices)
        )
        # 1-D x is not a [rows, hidden] activation.
        self.assertFalse(
            can_use_minimax_h3_rmsnorm_scale_shift(
                x[0], weight, shift, scale, indices[:1]
            )
        )
        # fp32 activations take the eager path.
        self.assertFalse(
            can_use_minimax_h3_rmsnorm_scale_shift(
                x.float(), weight, shift, scale, indices
            )
        )
        # A row/indices mismatch cannot be indexed.
        self.assertFalse(
            can_use_minimax_h3_rmsnorm_scale_shift(
                x, weight, shift, scale, indices[:-1]
            )
        )
        # A non-contiguous activation is not what the kernel strides for.
        wide = torch.randn(8, HIDDEN * 2, device=x.device, dtype=torch.bfloat16)
        self.assertFalse(
            can_use_minimax_h3_rmsnorm_scale_shift(
                wide[:, ::2], weight, shift, scale, indices
            )
        )


if __name__ == "__main__":
    unittest.main()
