"""Triton implementation of grouped matmul (aligned with cann-bench golden).

Semantics match ``cann-bench/tasks/level3/grouped_matmul/golden.py``::

    for each expert g:
        y[start_g:end_g] = x[start_g:end_g] @ weight[g]  (+ bias[g])

where ``group_list`` is cumsum over M (last value == M).

Runs on CUDA / Hygon HCU via the ``cuda`` device (same path as other
``operation_triton`` kernels in this tree).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
import triton
import triton.language as tl

GroupList = Union[Sequence[int], torch.Tensor]


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("npu_grouped_matmul_triton requires a CUDA/HCU device")


def _to_device(t: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    if t is None:
        return None
    return t.detach().to(device=device).contiguous()


def _null_ptr(device: torch.device) -> torch.Tensor:
    return torch.empty(0, device=device, dtype=torch.float32)


def _pick_blocks(m: int, n: int, k: int):
    def _pow2_le(x, lo, hi):
        v = lo
        while v * 2 <= min(x, hi):
            v *= 2
        return v

    bm = _pow2_le(max(m, 1), 16, 64)
    bn = _pow2_le(max(n, 1), 16, 64)
    bk = _pow2_le(max(k, 1), 16, 64)
    return bm, bn, bk


def _parse_group_ends(group_list: GroupList, e: int, m: int) -> List[int]:
    if group_list is None:
        raise ValueError("group_list is required")
    if isinstance(group_list, torch.Tensor):
        ends = group_list.to(torch.int64).reshape(-1).tolist()
    else:
        ends = [int(v) for v in group_list]
    if len(ends) != e:
        raise ValueError(f"group_list length {len(ends)} != E {e}")
    if ends[-1] != m:
        raise ValueError(f"group_list last value {ends[-1]} must equal M {m}")
    prev = 0
    for i, cur in enumerate(ends):
        if cur < prev:
            raise ValueError(
                f"group_list must be non-decreasing; at {i}: {cur} < {prev}"
            )
        prev = cur
    return ends


@triton.jit
def _grouped_matmul_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    bias_ptr,
    group_ends_ptr,
    M,
    N,
    K,
    E,
    stride_xm,
    stride_xk,
    stride_we,
    stride_w0,
    stride_w1,
    stride_ym,
    stride_yn,
    stride_be,
    HAS_BIAS: tl.constexpr,
    TRANSPOSE_WEIGHT: tl.constexpr,
    IEEE_DOT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Per-group tiled GEMM.

    Grid: (E, cdiv(max_m_i, BLOCK_M), cdiv(N, BLOCK_N)).

    weight layout:
      - TRANSPOSE_WEIGHT=0: [E, K, N] → load B as [K, N] (stride_w0=K-axis, stride_w1=N-axis)
      - TRANSPOSE_WEIGHT=1: [E, N, K] → load B as [N, K] then use as A @ B^T
        (logical B for matmul is [K, N] via stride_w0=K-axis, stride_w1=N-axis)
    """
    g = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    if g >= E:
        return

    end = tl.load(group_ends_ptr + g)
    # cumsum: start = ends[g-1] (or 0 for g==0)
    start = tl.load(group_ends_ptr + g - 1, mask=g > 0, other=0)
    m_i = end - start
    if m_i <= 0:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Absolute token rows in the merged [M, K] / [M, N] tensors.
    row = start + offs_m

    a_ptrs = x_ptr + row[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_base = w_ptr + g * stride_we
    if TRANSPOSE_WEIGHT:
        # weight[g]: [N, K]; matmul needs [K, N] → index (k, n) as w[n, k]
        b_ptrs = w_base + offs_k[:, None] * stride_w1 + offs_n[None, :] * stride_w0
    else:
        # weight[g]: [K, N]
        b_ptrs = w_base + offs_k[:, None] * stride_w0 + offs_n[None, :] * stride_w1

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_remaining = K - k0
        a_mask = (offs_m[:, None] < m_i) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)
        if IEEE_DOT:
            acc = tl.dot(a, b, acc=acc, input_precision="ieee")
        else:
            acc = tl.dot(a, b, acc=acc)
        a_ptrs += BLOCK_K * stride_xk
        if TRANSPOSE_WEIGHT:
            b_ptrs += BLOCK_K * stride_w1
        else:
            b_ptrs += BLOCK_K * stride_w0

    if HAS_BIAS:
        bias = tl.load(bias_ptr + g * stride_be + offs_n, mask=offs_n < N, other=0.0).to(
            tl.float32
        )
        acc = acc + bias[None, :]

    c_ptrs = y_ptr + row[:, None] * stride_ym + offs_n[None, :] * stride_yn
    c_mask = (offs_m[:, None] < m_i) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def grouped_matmul_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    group_list: Optional[GroupList] = None,
    split_item: int = 0,
    transpose_weight: bool = False,
) -> List[torch.Tensor]:
    """Triton grouped matmul matching ``golden.grouped_matmul``.

    Args:
        x: ``[M, K]``
        weight: ``[E, K, N]`` if ``transpose_weight=False``, else ``[E, N, K]``
        bias: optional ``[E, N]``
        group_list: cumsum of length ``E``, last value == ``M``
        split_item: 0/1 → list of E tensors; 2/3 → single ``[M, N]`` in a list
        transpose_weight: whether weight last two dims need transpose for matmul

    Returns:
        ``List[Tensor]`` as described for ``split_item``.
        Floating ``x`` → output dtype ``x.dtype``; integer ``x`` → ``float32``
        accumulator (for dequant in ``GroupedMatmul``).
    """
    if x.dim() != 2:
        raise ValueError(f"x must be 2D [M, K], got shape {tuple(x.shape)}")
    if weight.dim() != 3:
        raise ValueError(
            f"weight must be 3D [E, K, N] or [E, N, K], got shape {tuple(weight.shape)}"
        )

    m, k = int(x.shape[0]), int(x.shape[1])
    e = int(weight.shape[0])
    if transpose_weight:
        if int(weight.shape[2]) != k:
            raise ValueError(
                f"K mismatch: x has {k}, weight (transposed) has {weight.shape[2]}"
            )
        n = int(weight.shape[1])
    else:
        if int(weight.shape[1]) != k:
            raise ValueError(f"K mismatch: x has {k}, weight has {weight.shape[1]}")
        n = int(weight.shape[2])

    ends = _parse_group_ends(group_list, e, m)
    starts = [0] + ends[:-1]
    max_mi = max((ends[g] - starts[g] for g in range(e)), default=0)

    device = _device()
    x_d = _to_device(x, device)
    w_d = _to_device(weight, device)
    bias_d = _to_device(bias, device)
    if bias_d is not None:
        if bias_d.dim() != 2 or bias_d.shape[0] != e or bias_d.shape[1] != n:
            raise ValueError(
                f"bias must be [E, N]=[{e}, {n}], got shape {tuple(bias_d.shape)}"
            )

    # Accumulate in float32 (same as golden's .float() path). Integer x/w
    # (W4A8 int8×int8) must keep the fp32 accumulator — casting to x.dtype
    # saturates at ±127 and breaks subsequent weight_scale * per_token_scale.
    y = torch.zeros((m, n), device=device, dtype=torch.float32)
    group_ends_t = torch.tensor(ends, device=device, dtype=torch.int64)

    if m > 0 and max_mi > 0 and n > 0 and k > 0:
        bm, bn, bk = _pick_blocks(max_mi, n, k)
        null = _null_ptr(device)
        # Contiguous weight strides
        # [E, K, N]: (K*N, N, 1) → stride_w0=N-axis-of-K, stride_w1=N-axis-of-N
        # [E, N, K]: (N*K, K, 1)
        if transpose_weight:
            stride_we = w_d.stride(0)
            stride_w0 = w_d.stride(1)  # N axis
            stride_w1 = w_d.stride(2)  # K axis
        else:
            stride_we = w_d.stride(0)
            stride_w0 = w_d.stride(1)  # K axis
            stride_w1 = w_d.stride(2)  # N axis

        grid = (e, triton.cdiv(max_mi, bm), triton.cdiv(n, bn))
        # fp32 inputs: IEEE matmul to match CPU BLAS. fp16/bf16: default
        # tensor-core path (matches golden's fp32-accum-then-cast closely enough).
        _grouped_matmul_kernel[grid](
            x_d,
            w_d,
            y,
            bias_d if bias_d is not None else null,
            group_ends_t,
            m,
            n,
            k,
            e,
            x_d.stride(0),
            x_d.stride(1),
            stride_we,
            stride_w0,
            stride_w1,
            y.stride(0),
            y.stride(1),
            bias_d.stride(0) if bias_d is not None else 0,
            HAS_BIAS=bias_d is not None,
            TRANSPOSE_WEIGHT=transpose_weight,
            IEEE_DOT=(x.dtype == torch.float32),
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
        )

    # Floating inputs: match x.dtype. Integer inputs: keep float32 accum so
    # GroupedMatmul can apply weight_scale * per_token_scale without saturation.
    if x.dtype.is_floating_point:
        y = y.to(x.dtype)

    if split_item in (0, 1):
        return [y[starts[g] : ends[g]] for g in range(e)]
    if split_item in (2, 3):
        return [y]
    raise ValueError(f"unsupported split_item={split_item}")

# ---------------------------------------------------------------------------
# Golden reference (same logic as cann-bench golden.grouped_matmul)
# ---------------------------------------------------------------------------
def grouped_matmul_golden(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    group_list=None,
    split_item: int = 0,
    transpose_weight: bool = False,
) -> List[torch.Tensor]:
    """CPU/device reference matching ``golden.grouped_matmul`` (fp32 accum).

    Integer ``x`` keeps float32 output (no cast-to-int8); float ``x`` casts back.
    """
    assert x.dim() == 2, "x must be 2D [M, K]"
    assert weight.dim() == 3, "weight must be 3D [E, K, N] or [E, N, K]"

    m, k = x.shape
    e = weight.shape[0]
    if transpose_weight:
        assert weight.shape[2] == k
        n = weight.shape[1]
    else:
        assert weight.shape[1] == k
        n = weight.shape[2]

    ends = _parse_group_ends(group_list, e, m)
    starts = [0] + ends[:-1]

    keep_fp32 = not x.dtype.is_floating_point
    y = torch.zeros(
        (m, n),
        dtype=torch.float32 if keep_fp32 else x.dtype,
        device=x.device,
    )
    x_f = x.float()
    for g in range(e):
        s, e_g = starts[g], ends[g]
        if s == e_g:
            continue
        w_g = weight[g].float()
        if transpose_weight:
            mm = torch.matmul(x_f[s:e_g], w_g.transpose(-2, -1))
        else:
            mm = torch.matmul(x_f[s:e_g], w_g)
        if bias is not None:
            mm = mm + bias[g].float().unsqueeze(0)
        y[s:e_g] = mm if keep_fp32 else mm.to(x.dtype)

    if split_item in (0, 1):
        return [y[starts[g] : ends[g]] for g in range(e)]
    return [y]

