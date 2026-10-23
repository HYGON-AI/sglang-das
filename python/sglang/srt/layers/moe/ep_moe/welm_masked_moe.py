"""Masked grouped bf16 MoE for DeepEP low-latency dispatch (HCU, unquantized).

The low-latency dispatch hands each rank its local experts as a dense
[num_local_experts, capacity, hidden] buffer with ``masked_m[e]`` valid rows
per expert. A dense bmm over that buffer costs num_local_experts * capacity
rows no matter how many tokens arrived; these kernels launch one program per
(expert, row block, column block) and exit immediately for row blocks past
``masked_m[e]``, so only the weights of experts that received tokens are read.

out[e, :m_e] = (silu(x W1^T) * (x W3^T)) W2^T; rows >= m_e are left unwritten.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gate_up_kernel(
    X, W13, H, MASKED_M,
    N, K,
    sxe, sxm, swe, swn, she, shm,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    e = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_e = tl.load(MASKED_M + e)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    for m0 in range(0, m_e, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        rmask = offs_m < m_e
        x_ptr = X + e * sxe + offs_m[:, None] * sxm + offs_k[None, :]
        g_ptr = W13 + e * swe + offs_n[None, :] * swn + offs_k[:, None]
        u_ptr = g_ptr + N * swn
        acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            x = tl.load(x_ptr, mask=rmask[:, None], other=0.0)
            acc_g += tl.dot(x, tl.load(g_ptr))
            acc_u += tl.dot(x, tl.load(u_ptr))
            x_ptr += BLOCK_K
            g_ptr += BLOCK_K
            u_ptr += BLOCK_K
        h = acc_g / (1.0 + tl.exp(-acc_g)) * acc_u
        tl.store(H + e * she + offs_m[:, None] * shm + offs_n[None, :],
                 h.to(H.dtype.element_ty), mask=rmask[:, None])


@triton.jit
def _down_kernel(
    H, W2, OUT, MASKED_M,
    N, K,
    she, shm, swe, swk, soe, som,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    e = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_e = tl.load(MASKED_M + e)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    for m0 in range(0, m_e, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        rmask = offs_m < m_e
        h_ptr = H + e * she + offs_m[:, None] * shm + offs_n[None, :]
        w_ptr = W2 + e * swe + offs_k[None, :] * swk + offs_n[:, None]
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for n in range(0, N, BLOCK_N):
            h = tl.load(h_ptr, mask=rmask[:, None], other=0.0)
            acc += tl.dot(h, tl.load(w_ptr))
            h_ptr += BLOCK_N
            w_ptr += BLOCK_N
        tl.store(OUT + e * soe + offs_m[:, None] * som + offs_k[None, :],
                 acc.to(OUT.dtype.element_ty), mask=rmask[:, None])


def _block_m(expected_m: int) -> int:
    return 16 if expected_m <= 16 else (32 if expected_m <= 64 else 64)


def welm_masked_moe(hidden_states, w13, w2, masked_m, expected_m: int = 16):
    """hidden_states [E, cap, K] bf16; w13 [E, 2N, K]; w2 [E, K, N];
    masked_m [E] int32. Returns out [E, cap, K]; rows past masked_m are undefined."""
    E, cap, K = hidden_states.shape
    N = w13.shape[1] // 2
    assert w13.shape == (E, 2 * N, K) and w2.shape == (E, K, N)
    assert hidden_states.stride(-1) == 1 and w13.stride(-1) == 1 and w2.stride(-1) == 1
    bm = _block_m(expected_m)
    out = torch.empty_like(hidden_states)  # rows past masked_m are never read by the LL combine
    h = torch.empty((E, cap, N), dtype=hidden_states.dtype, device=hidden_states.device)
    bn1, bk1 = 64, 64
    _gate_up_kernel[(E, N // bn1)](
        hidden_states, w13, h, masked_m, N, K,
        hidden_states.stride(0), hidden_states.stride(1), w13.stride(0), w13.stride(1),
        h.stride(0), h.stride(1),
        BLOCK_M=bm, BLOCK_N=bn1, BLOCK_K=bk1, num_warps=4,
    )
    bk2, bn2 = 64, 64
    _down_kernel[(E, K // bk2)](
        h, w2, out, masked_m, N, K,
        h.stride(0), h.stride(1), w2.stride(0), w2.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=bm, BLOCK_K=bk2, BLOCK_N=bn2, num_warps=4,
    )
    return out
