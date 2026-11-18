"""HCU-only numerics for ``diffusion.minimax_h3_dit_qknorm_rope``.

The fused in-place kernel is compared against the split reference the DiT
attention block falls back to when the CUDA-JIT ``fused_inplace_qknorm_rope``
is unavailable (always the case on HCU): an affine ``nn.RMSNorm`` followed
by the HCU ``minimax_h3_rotary_embedding`` op.

Not bit-exact: the fused reduction and floating-point evaluation may differ
from the native split chain. Needs real
HCU hardware (``minimax_h3_rotary_embedding`` is HCU-only), so this lives
under ``test/manual`` rather than the CPU/CUDA-gated ``test/registered``
suite.
"""

import unittest

import torch

from sglang.kernels.ops.diffusion import (
    can_use_minimax_h3_dit_qknorm_rope,
    minimax_h3_dit_qknorm_rope,
)
from sglang.srt.utils import is_hcu

HEAD_DIM = 128
ROTARY_DIM = 96
HEADS = 56


def _make_cache(tokens: int, device: torch.device) -> torch.Tensor:
    inv = 10000.0 ** (-torch.arange(16, device=device, dtype=torch.float32) / 16)
    t = torch.arange(tokens, device=device, dtype=torch.float32)
    axes = torch.stack((t // 1024, (t // 32) % 32, t % 32), dim=1)
    half = (axes[:, :, None] * inv).reshape(tokens, 48)
    return torch.cat((half.cos(), half.sin()), dim=1).to(torch.bfloat16).contiguous()


def _split_reference(q, k, wq, wk, cache, positions, eps):
    from sgl_kernel import rotary_embedding  # noqa: F401  (registers the op namespace)

    q = torch.nn.functional.rms_norm(q, (HEAD_DIM,), wq, eps)
    k = torch.nn.functional.rms_norm(k, (HEAD_DIM,), wk, eps)
    torch.ops.sgl_kernel.minimax_h3_rotary_embedding(
        positions, q.view(q.shape[0], -1), k.view(k.shape[0], -1), HEAD_DIM, cache, True
    )
    return q, k


@unittest.skipUnless(torch.cuda.is_available() and is_hcu(), "requires HCU hardware")
class TestMinimaxH3DitQknormRope(unittest.TestCase):
    def test_matches_split_reference(self):
        torch.manual_seed(20261110)
        device = torch.device("cuda:0")
        for tokens in (16, 2048):
            with self.subTest(tokens=tokens):
                base = torch.randn(
                    tokens, HEADS * HEAD_DIM * 3, device=device, dtype=torch.bfloat16
                )
                wq = (
                    torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.1 + 1
                )
                wk = (
                    torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16) * 0.1 + 1
                )
                cache = _make_cache(tokens, device)
                positions = torch.arange(tokens, device=device, dtype=torch.long)
                eps = 1e-5

                q0, k0, v0 = base.split(HEADS * HEAD_DIM, dim=-1)
                q0 = q0.view(tokens, HEADS, HEAD_DIM)
                k0 = k0.view(tokens, HEADS, HEAD_DIM)
                v0 = v0.view(tokens, HEADS, HEAD_DIM)

                self.assertTrue(
                    can_use_minimax_h3_dit_qknorm_rope(
                        q0, k0, wq, wk, cache, eps, eps, positions=positions
                    )
                )

                with torch.inference_mode():
                    q_ref, k_ref = _split_reference(
                        q0.clone(), k0.clone(), wq, wk, cache, positions, eps
                    )

                    for variant in ("4wave", "2wave", "4heads"):
                        with self.subTest(variant=variant):
                            work = base.clone()
                            q, k, v = work.split(HEADS * HEAD_DIM, dim=-1)
                            q = q.view(tokens, HEADS, HEAD_DIM)
                            k = k.view(tokens, HEADS, HEAD_DIM)
                            v = v.view(tokens, HEADS, HEAD_DIM)
                            minimax_h3_dit_qknorm_rope(
                                q, k, wq, wk, cache, positions, eps, variant=variant
                            )

                            torch.testing.assert_close(q, q_ref, atol=0.04, rtol=0.02)
                            torch.testing.assert_close(k, k_ref, atol=0.04, rtol=0.02)
                            # V must be left untouched by the in-place kernel.
                            torch.testing.assert_close(v, v0, atol=0, rtol=0)

    def test_masks_partial_head_block_in_padded_qkv(self):
        device = torch.device("cuda:0")
        tokens, heads, padding = 17, 7, 16
        qkv_width = heads * 3 * HEAD_DIM
        work = torch.randn(
            tokens, qkv_width + padding, device=device, dtype=torch.bfloat16
        )
        qkv = work[:, :qkv_width].view(tokens, heads, 3, HEAD_DIM)
        q, k, v = qkv.unbind(2)
        original_v = v.clone()
        original_padding = work[:, qkv_width:].clone()
        wq = torch.randn(HEAD_DIM, device=device, dtype=q.dtype)
        wk = torch.randn_like(wq)
        cache = _make_cache(tokens, device)
        positions = torch.arange(tokens - 1, -1, -1, device=device, dtype=torch.int32)
        eps = 1e-5

        with torch.inference_mode():
            q_ref, k_ref = _split_reference(
                q.contiguous(), k.contiguous(), wq, wk, cache, positions.long(), eps
            )
            minimax_h3_dit_qknorm_rope(q, k, wq, wk, cache, positions, eps)

        torch.testing.assert_close(q, q_ref, atol=0.04, rtol=0.02)
        self.assertTrue(torch.equal(v, original_v))
        torch.testing.assert_close(k, k_ref, atol=0.04, rtol=0.02)
        self.assertTrue(torch.equal(work[:, qkv_width:], original_padding))

    def test_rejects_unsupported_inputs(self):
        device = torch.device("cuda:0")
        tokens = 8
        q = torch.randn(tokens, HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        wq = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
        wk = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
        cache = _make_cache(tokens, device)
        positions = torch.arange(tokens, device=device, dtype=torch.long)
        eps = 1e-5

        self.assertTrue(
            can_use_minimax_h3_dit_qknorm_rope(
                q, k, wq, wk, cache, eps, eps, positions=positions
            )
        )
        # Wrong dtype.
        self.assertFalse(
            can_use_minimax_h3_dit_qknorm_rope(
                q.float(), k.float(), wq, wk, cache, eps, eps, positions=positions
            )
        )
        # Mismatched eps.
        self.assertFalse(
            can_use_minimax_h3_dit_qknorm_rope(
                q, k, wq, wk, cache, eps, eps * 2, positions=positions
            )
        )
        # Wrong head_dim.
        q2 = torch.randn(tokens, HEADS, 64, device=device, dtype=torch.bfloat16)
        k2 = torch.randn_like(q2)
        self.assertFalse(
            can_use_minimax_h3_dit_qknorm_rope(
                q2, k2, wq[:64], wk[:64], cache, eps, eps, positions=positions
            )
        )


if __name__ == "__main__":
    unittest.main()
