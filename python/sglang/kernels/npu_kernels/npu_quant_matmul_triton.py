"""Triton implementation of ``torch_npu.npu_quant_matmul``.

Implements the formulas in ``torch_npu-npu_quant_matmul.md`` with a Triton
GEMM kernel (CUDA / Hygon HCU via the ``cuda`` device):

- no bias: ``out = x1 @ x2 * scale [* pertoken_scale] + offset``
- int32 bias: ``out = (x1 @ x2 + bias) * scale [* pertoken_scale] + offset``
- bf16/fp32 bias (no offset): ``out = x1 @ x2 * scale [* pertoken_scale] + bias``

``group_sizes`` / 2D group scale / int64 packed scale / int4 packing are not
supported in this reference path.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("npu_quant_matmul_triton requires a CUDA/HCU device")


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


@triton.jit
def _quant_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    scale_ptr,
    pertoken_ptr,
    bias_ptr,
    offset_ptr,
    HAS_SCALE: tl.constexpr,
    SCALE_SCALAR: tl.constexpr,  # scale has 1 element, broadcast to N
    HAS_PERTOKEN: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_OFFSET: tl.constexpr,
    INT32_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """C[M,N] = A[M,K] @ B[K,N] with quant epilogue (float32 accum)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_remaining = K - k0
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)
        acc = tl.dot(a, b, acc=acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS and INT32_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    if HAS_SCALE:
        if SCALE_SCALAR:
            scale = tl.load(scale_ptr).to(tl.float32)
            acc = acc * scale
        else:
            scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0).to(
                tl.float32
            )
            acc = acc * scale[None, :]

    if HAS_PERTOKEN:
        pt = tl.load(pertoken_ptr + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
        acc = acc * pt[:, None]

    if HAS_BIAS and (not INT32_BIAS):
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    if HAS_OFFSET:
        off = tl.load(offset_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc + off[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _prepare_1d_addend(
    t: torch.Tensor,
    n: int,
    name: str,
    *,
    allow_scalar: bool = True,
) -> torch.Tensor:
    """Normalize 1D bias/offset/scale to float32 length N (or 1)."""
    b = t.to(torch.float32).reshape(-1).contiguous()
    if b.numel() == n:
        return b
    if allow_scalar and b.numel() == 1:
        return b
    raise ValueError(f"{name} length must be 1 or N={n}, got {b.numel()}")


def _triton_quant_matmul_2d(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: torch.Tensor,
    *,
    offset: Optional[torch.Tensor],
    pertoken_scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run Triton quant matmul for contiguous 2D ``x1[M,K]``, ``x2[K,N]``."""
    m, k = x1.shape
    k2, n = x2.shape
    if k != k2:
        raise ValueError(f"K mismatch: x1 K={k}, x2 K={k2}")

    device = x1.device
    x1_c = x1.contiguous()
    x2_c = x2.contiguous()
    y = torch.empty((m, n), device=device, dtype=torch.float32)

    use_int32_bias = bias is not None and bias.dtype == torch.int32
    apply_offset = offset is not None and (bias is None or use_int32_bias)

    scale_f = _prepare_1d_addend(scale, n, "scale")
    scale_scalar = scale_f.numel() == 1

    pts_f = None
    if pertoken_scale is not None:
        pts_f = pertoken_scale.to(torch.float32).reshape(-1).contiguous()
        if pts_f.numel() != m:
            raise ValueError(
                f"pertoken_scale numel {pts_f.numel()} != M={m} for 2D path"
            )

    bias_f = None
    if bias is not None:
        if bias.ndim == 1:
            bias_f = _prepare_1d_addend(bias, n, "bias")
            if bias_f.numel() == 1:
                bias_f = bias_f.expand(n).contiguous()
        elif bias.ndim == 3:
            raise ValueError("3D bias requires batched path")
        else:
            raise ValueError(f"bias unsupported shape {tuple(bias.shape)}")

    off_f = None
    if apply_offset:
        if offset.ndim != 1:
            raise ValueError(f"offset must be 1D, got shape {tuple(offset.shape)}")
        off_f = _prepare_1d_addend(offset, n, "offset")
        if off_f.numel() == 1:
            off_f = off_f.expand(n).contiguous()

    null = _null_ptr(device)
    bm, bn, bk = _pick_blocks(m, n, k)
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
    _quant_matmul_kernel[grid](
        x1_c,
        x2_c,
        y,
        m,
        n,
        k,
        x1_c.stride(0),
        x1_c.stride(1),
        x2_c.stride(0),
        x2_c.stride(1),
        y.stride(0),
        y.stride(1),
        scale_f,
        pts_f if pts_f is not None else null,
        bias_f if bias_f is not None else null,
        off_f if off_f is not None else null,
        HAS_SCALE=True,
        SCALE_SCALAR=scale_scalar,
        HAS_PERTOKEN=pts_f is not None,
        HAS_BIAS=bias_f is not None,
        HAS_OFFSET=off_f is not None,
        INT32_BIAS=use_int32_bias,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
    )
    return y


def _cast_output(y: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    if out_dtype == y.dtype:
        return y
    if out_dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
        info = torch.iinfo(out_dtype)
        return y.round().clamp(info.min, info.max).to(out_dtype)
    return y.to(out_dtype)


def _normalize_output_dtype(
    output_dtype: Optional[object],
) -> torch.dtype:
    """Accept torch.dtype, str (\"float16\"/\"bfloat16\"/\"int8\"), or None→int8."""
    if output_dtype is None:
        return torch.int8
    if isinstance(output_dtype, torch.dtype):
        return output_dtype
    if isinstance(output_dtype, str):
        key = output_dtype.lower().strip()
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "int8": torch.int8,
            "int32": torch.int32,
        }
        if key not in mapping:
            raise ValueError(f"unsupported output_dtype string: {output_dtype!r}")
        return mapping[key]
    raise TypeError(f"output_dtype must be str/torch.dtype/None, got {type(output_dtype)}")


def npu_quant_matmul_triton(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: torch.Tensor,
    *,
    offset: Optional[torch.Tensor] = None,
    pertoken_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    output_dtype: Optional[object] = None,
    group_sizes: Optional[list] = None,
) -> torch.Tensor:
    """Triton reference for ``torch_npu.npu_quant_matmul``.

    Signature mirrors the NPU API. Tensors are moved to CUDA/HCU for compute;
    the result is returned on that device (caller may ``.cpu()``).
    ``output_dtype`` may be a ``torch.dtype`` or a string such as ``\"float16\"``.
    """
    if group_sizes is not None:
        raise NotImplementedError(
            "npu_quant_matmul_triton does not support group_sizes"
        )
    if scale.ndim == 2:
        raise NotImplementedError(
            "npu_quant_matmul_triton does not support 2D (group) scale"
        )
    if scale.dtype == torch.int64:
        raise NotImplementedError(
            "npu_quant_matmul_triton expects float scale; "
            "int64 packed scale is not supported"
        )
    if x1.ndim < 2 or x2.ndim < 2:
        raise ValueError(
            f"x1/x2 must be at least 2D, got x1.ndim={x1.ndim}, x2.ndim={x2.ndim}"
        )
    if x1.shape[-1] != x2.shape[-2]:
        raise ValueError(
            f"x1 K-dim {x1.shape[-1]} != x2 K-dim {x2.shape[-2]}"
        )
    if x1.dtype != torch.int8 or x2.dtype != torch.int8:
        raise TypeError(
            f"npu_quant_matmul_triton expects int8 x1/x2, got {x1.dtype}, {x2.dtype}"
        )
    if (
        bias is not None
        and bias.dtype != torch.int32
        and offset is not None
    ):
        raise ValueError(
            f"offset must be None when bias is {bias.dtype} "
            "(only int32 bias may combine with offset)"
        )

    output_dtype = _normalize_output_dtype(output_dtype)

    device = _device()
    x1 = _to_device(x1, device)
    x2 = _to_device(x2, device)
    scale = _to_device(scale, device)
    offset = _to_device(offset, device)
    pertoken_scale = _to_device(pertoken_scale, device)
    bias = _to_device(bias, device)

    # Broadcast leading batch dims like torch.matmul, then flatten to 2D GEMM
    # when x2 has no (or broadcastable unit) batch — covers dump & random 2D.
    batch1 = x1.shape[:-2]
    batch2 = x2.shape[:-2]
    batch = torch.broadcast_shapes(batch1, batch2)
    m, k = int(x1.shape[-2]), int(x1.shape[-1])
    n = int(x2.shape[-1])

    if pertoken_scale is not None:
        token_numel = x1.numel() // x1.shape[-1]
        if pertoken_scale.numel() != token_numel:
            raise ValueError(
                f"pertoken_scale numel {pertoken_scale.numel()} != "
                f"x1 token count {token_numel}"
            )

    # Expand to broadcasted batch, then flatten tokens for a single 2D launch
    # when bias is not 3D (or batch is empty).
    a = x1.broadcast_to(batch + (m, k)).contiguous()
    b = x2.broadcast_to(batch + (k, n)).contiguous()
    bsz = int(torch.tensor(batch).prod().item()) if batch else 1
    a2 = a.reshape(bsz * m, k)
    b2 = b.reshape(bsz, k, n) if bsz > 1 else b.reshape(k, n)

    if bias is not None and bias.ndim == 3:
        # Per-batch 3D bias: loop (doc single-op uses bias (31,1,16)).
        if len(batch) != 1 or bias.shape[0] != batch[0] or bias.shape[1] != 1:
            raise ValueError(
                f"bias 3D must be (batch,1,N) matching batch={batch}, "
                f"got {tuple(bias.shape)}"
            )
        outs = []
        pts = (
            pertoken_scale.reshape(bsz, m)
            if pertoken_scale is not None
            else None
        )
        for bi in range(bsz):
            x1_i = a2[bi * m : (bi + 1) * m]
            x2_i = b[bi] if bsz > 1 else b2
            pt_i = pts[bi] if pts is not None else None
            bias_i = bias[bi, 0, :]
            outs.append(
                _triton_quant_matmul_2d(
                    x1_i,
                    x2_i,
                    scale,
                    offset=offset,
                    pertoken_scale=pt_i,
                    bias=bias_i,
                )
            )
        out = torch.stack(outs, dim=0).reshape(batch + (m, n))
        return _cast_output(out, output_dtype)

    if bsz == 1:
        # Single matrix: x2 may still be [1,K,N] after broadcast — squeeze.
        x2_2d = b2 if b2.ndim == 2 else b2.reshape(k, n)
        pts = (
            pertoken_scale.reshape(m)
            if pertoken_scale is not None
            else None
        )
        out = _triton_quant_matmul_2d(
            a2.reshape(m, k),
            x2_2d,
            scale,
            offset=offset,
            pertoken_scale=pts,
            bias=bias,
        )
        return _cast_output(out.reshape(batch + (m, n)), output_dtype)

    # Shared 2D weight across batches (x2 originally 2D): one GEMM on flat x1.
    if len(batch2) == 0 or all(s == 1 for s in batch2):
        x2_2d = x2.reshape(k, n).contiguous()
        pts = (
            pertoken_scale.reshape(bsz * m)
            if pertoken_scale is not None
            else None
        )
        out = _triton_quant_matmul_2d(
            a2,
            x2_2d,
            scale,
            offset=offset,
            pertoken_scale=pts,
            bias=bias,
        )
        return _cast_output(out.reshape(batch + (m, n)), output_dtype)

    # General batched x2: per-batch launches with shared 1D bias/scale.
    outs = []
    pts = (
        pertoken_scale.reshape(bsz, m) if pertoken_scale is not None else None
    )
    for bi in range(bsz):
        outs.append(
            _triton_quant_matmul_2d(
                a2[bi * m : (bi + 1) * m],
                b[bi],
                scale,
                offset=offset,
                pertoken_scale=pts[bi] if pts is not None else None,
                bias=bias,
            )
        )
    out = torch.stack(outs, dim=0).reshape(batch + (m, n))
    return _cast_output(out, output_dtype)


# Alias matching torch_npu name
npu_quant_matmul = npu_quant_matmul_triton



def quant_matmul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: torch.Tensor,
    offset: Optional[torch.Tensor] = None,
    pertoken_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    output_dtype: Optional[str] = None,
) -> torch.Tensor:
    """
    量化矩阵乘法

    Args:
        x1: [..., m, k] int8 左矩阵
        x2: [..., k, n] int8 右矩阵
        scale: [t] 反量化 scale (float32 / bfloat16)
        offset: [t] 非对称量化偏移 (float32, t=1 或 n)，反量化时 out += offset；
                对称量化时为 None
        pertoken_scale: [m] per-token scale (float32)
        bias: [n] 或 [batch, 1, n] 偏置，int32 走 pre-scale，浮点走 post-scale
        output_dtype: 输出类型 "float16"（默认）或 "bfloat16"

    Returns:
        out: [..., m, n] float16 或 bfloat16
    """
    # 矩阵乘：int8/int32 输入直接走 fp64 matmul。
    # PR-001: int8 × int8 单乘积 ≤ 127² = 16129；K 个累加最大 |mm| = 16129·K，
    # 即便 K=65535 也只到 ~1.06e9，远小于 fp64 整数精确上界 2^53 (≈9e15)，
    # 故 fp64 matmul 对该量程是 *精确* 的（与 int64 累加逐位相等）。
    # 不用 int64 matmul：CPU 上 int64 GEMM 无 BLAS，朴素实现比 fp64 慢 ~1000x
    # （大 shape 单次可达 ~60s），会令评测在 golden 阶段超时。fp64 走 BLAS，
    # 精度不变而速度恢复。fp32 不可用（24-bit 尾数，K>1024 会溢出）。
    if x1.dtype in (torch.int8, torch.int32) and x2.dtype in (torch.int8, torch.int32):
        mm = torch.matmul(x1.double(), x2.double())
    else:
        # bf16/fp16 输入路径维持原 fp32 等效计算
        mm = torch.matmul(x1.float(), x2.float()).double()

    # int32 bias 在反量化前累加 (pre-scale)
    if bias is not None and bias.dtype == torch.int32:
        mm = mm + bias.double()

    # 反量化 scale
    y = mm * scale.double()

    # 非对称量化偏移 (zero-point 校正)：out = mm*scale + offset
    # offset 与 scale 配对（NPU 侧由 npu_trans_quant_param(scale, offset) 打包）；
    # 对称量化时 offset=None，此步为 no-op。
    if offset is not None:
        y = y + offset.double()

    # pertoken_scale 沿 m 维广播
    if pertoken_scale is not None:
        y = y * pertoken_scale.double().unsqueeze(-1)

    # 浮点 bias 在反量化后相加 (post-scale)
    if bias is not None and bias.dtype != torch.int32:
        y = y + bias.double()

    # 输出 dtype，默认 float16
    if output_dtype is None or output_dtype == "float16":
        return y.to(torch.float16)
    elif output_dtype == "bfloat16":
        return y.to(torch.bfloat16)
    else:
        raise ValueError(f"unsupported output_dtype: {output_dtype}")
