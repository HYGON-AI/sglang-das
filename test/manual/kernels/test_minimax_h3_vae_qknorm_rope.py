"""HCU-only numerics for ``diffusion.minimax_h3_vae_qknorm_rope``.

The fused kernel is compared against the split reference the H3 video VAE
ViT attention module falls back to: an unaffine ``nn.RMSNorm`` followed by
the HCU ``minimax_h3_rotary_embedding`` op (``torch.ops.sgl_kernel``). Both
read the 3*head_dim-interleaved ``[B, S, H, 3*64]`` view produced by
chunking the fused ``to_qkv`` projection (head stride 192), not a
head-major QKV layout.

Not bit-exact: the fused reduction and floating-point evaluation may differ
from the native split chain. Needs real
HCU hardware (``minimax_h3_rotary_embedding`` is HCU-only), so this lives
under ``test/manual`` rather than the CPU/CUDA-gated ``test/registered``
suite.
"""

import unittest

import torch

from sglang.kernels.ops.diffusion import (
    can_use_minimax_h3_vae_qknorm_rope,
    minimax_h3_vae_qknorm_rope,
)
from sglang.srt.utils import is_hcu

HEAD_DIM = 64
ROTARY_DIM = 48
HEADS = 32


def _make_cache(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    angles = torch.randn((seq_len, ROTARY_DIM // 2), device="cuda", dtype=torch.float32)
    return torch.cat((angles.cos(), angles.sin()), dim=-1).to(dtype).contiguous()


def _split_reference(query, key, cache, positions, eps):
    from sgl_kernel import rotary_embedding  # noqa: F401  (registers the op namespace)

    norm = torch.nn.functional
    q = norm.rms_norm(query, (HEAD_DIM,), None, eps).contiguous()
    k = norm.rms_norm(key, (HEAD_DIM,), None, eps).contiguous()
    torch.ops.sgl_kernel.minimax_h3_rotary_embedding(
        positions, q.view(q.shape[1], -1), k.view(k.shape[1], -1), HEAD_DIM, cache, True
    )
    return q, k


@unittest.skipUnless(torch.cuda.is_available() and is_hcu(), "requires HCU hardware")
class TestMinimaxH3VaeQknormRope(unittest.TestCase):
    def test_matches_split_reference(self):
        torch.manual_seed(20261110)
        for seq_len in (16, 2048):
            with self.subTest(seq_len=seq_len):
                qkv = torch.randn(
                    (1, seq_len, HEADS, 3 * HEAD_DIM),
                    device="cuda",
                    dtype=torch.float16,
                )
                query, key, _ = qkv.chunk(3, dim=-1)
                cache = _make_cache(seq_len, query.dtype)
                positions = torch.arange(seq_len, device="cuda", dtype=torch.long)
                eps = 1e-5
                norm_q = torch.nn.RMSNorm(
                    HEAD_DIM, eps=eps, elementwise_affine=False
                ).cuda()
                norm_k = torch.nn.RMSNorm(
                    HEAD_DIM, eps=eps, elementwise_affine=False
                ).cuda()

                self.assertTrue(
                    can_use_minimax_h3_vae_qknorm_rope(
                        query, key, (None, None, cache, positions), norm_q, norm_k
                    )
                )

                before = qkv.clone()
                q_ref, k_ref = _split_reference(
                    query.clone(), key.clone(), cache, positions, eps
                )
                fused = minimax_h3_vae_qknorm_rope(query, key, cache, positions, eps)
                self.assertIsNotNone(fused)
                q_fused, k_fused = fused

                torch.testing.assert_close(q_fused, q_ref, atol=0.004, rtol=0.02)
                torch.testing.assert_close(k_fused, k_ref, atol=0.004, rtol=0.02)
                # The kernel must not mutate the source QKV view (V included).
                torch.testing.assert_close(qkv, before)

    def test_rejects_unsupported_inputs(self):
        norm_q = torch.nn.RMSNorm(HEAD_DIM, eps=1e-5, elementwise_affine=False).cuda()
        norm_k = torch.nn.RMSNorm(HEAD_DIM, eps=1e-5, elementwise_affine=False).cuda()
        affine_norm = torch.nn.RMSNorm(HEAD_DIM, elementwise_affine=True).cuda()
        seq_len = 8
        q = torch.randn(
            (1, seq_len, HEADS, HEAD_DIM), device="cuda", dtype=torch.float16
        )
        k = torch.randn_like(q)
        cache = _make_cache(seq_len, q.dtype)
        positions = torch.arange(seq_len, device="cuda", dtype=torch.long)
        rope = (None, None, cache, positions)

        self.assertTrue(can_use_minimax_h3_vae_qknorm_rope(q, k, rope, norm_q, norm_k))
        # Wrong dtype.
        self.assertFalse(
            can_use_minimax_h3_vae_qknorm_rope(
                q.float(), k.float(), rope, norm_q, norm_k
            )
        )
        # Affine norm weight present.
        self.assertFalse(
            can_use_minimax_h3_vae_qknorm_rope(q, k, rope, affine_norm, affine_norm)
        )
        # Batch size > 1.
        q2 = torch.randn(
            (2, seq_len, HEADS, HEAD_DIM), device="cuda", dtype=torch.float16
        )
        k2 = torch.randn_like(q2)
        self.assertFalse(
            can_use_minimax_h3_vae_qknorm_rope(q2, k2, rope, norm_q, norm_k)
        )
        # No rotary_pos_emb.
        self.assertFalse(can_use_minimax_h3_vae_qknorm_rope(q, k, None, norm_q, norm_k))


if __name__ == "__main__":
    unittest.main()
