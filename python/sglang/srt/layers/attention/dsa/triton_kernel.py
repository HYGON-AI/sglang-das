from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


# Triton implementation
@triton.jit
def _act_quant_kernel(
    X_ptr,
    Y_ptr,
    S_ptr,
    M,
    N,
    group_size: tl.constexpr,
    round_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for activation quantization.

    Each block processes BLOCK_M rows and group_size columns.
    """
    # Get block IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # FP8 constants
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max

    # Calculate row and column offsets
    row_start = pid_m * BLOCK_M
    col_start = pid_n * group_size

    # Create offset arrays
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)

    # Mask for valid rows and columns
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    # Load input data
    x_ptrs = X_ptr + rows[:, None] * N + cols[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Compute absolute max along columns (group_size dimension) for each row
    x_abs = tl.abs(x)
    amax = tl.max(x_abs, axis=1)  # Shape: (BLOCK_M,)

    # Clamp amax to avoid division by zero
    amax = tl.maximum(amax, 1e-4)

    # Compute scale
    if round_scale:
        # Fast round scale using bit manipulation approximation
        # This is a simplified version - the exact bit manipulation is harder in Triton
        # Using log2 + ceil + pow2 as approximation
        log_val = tl.log2(amax * fp8_max_inv)
        log_ceil = tl.ceil(log_val)
        scale = tl.exp2(log_ceil)
    else:
        scale = amax * fp8_max_inv

    # Quantize: y = clamp(x / scale, fp8_min, fp8_max)
    scale_broadcast = scale[:, None]
    y = x / scale_broadcast
    y = tl.minimum(tl.maximum(y, fp8_min), fp8_max)

    # Store quantized output
    y_ptrs = Y_ptr + rows[:, None] * N + cols[None, :]
    tl.store(y_ptrs, y, mask=mask)

    # Store scales
    s_cols = pid_n
    s_ptrs = S_ptr + rows * (N // group_size) + s_cols
    s_mask = row_mask
    tl.store(s_ptrs, scale, mask=s_mask)


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization with Triton.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        scale_fmt (Optional[str], optional): The format of the scale. Default is None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"

    # Flatten all dims except last
    N = x.size(-1)
    x_flat = x.view(-1, N)
    M = x_flat.size(0)

    # Allocate output tensors
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    y_flat = y.view(-1, N)
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
    s_flat = s.view(-1, N // block_size)

    # Launch kernel
    BLOCK_M = 32
    BLOCK_N = block_size
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, block_size))
    round_scale = scale_fmt is not None

    _act_quant_kernel[grid](
        x_flat,
        y_flat,
        s_flat,
        M,
        N,
        group_size=block_size,
        round_scale=round_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_stages=0 if round_scale else 2,
    )

    return y, s


@triton.jit
def _get_valid_kv_indices_kernel(
    page_table_ptr,  # [bs, topk]
    kv_indptr_ptr,  # [bs + 1]
    kv_indices_ptr,  # [bs * topk] output buffer
    bs: tl.constexpr,
    topk: tl.constexpr,
):
    """
    Extract valid indices (non -1) from page_table into kv_indices.
    Each program handles one batch.
    """
    batch_id = tl.program_id(0)

    # Get the start position for this batch in kv_indices
    dst_start = tl.load(kv_indptr_ptr + batch_id)

    # Load all topk indices for this batch
    src_offset = batch_id * topk
    offsets = tl.arange(0, topk)
    indices = tl.load(page_table_ptr + src_offset + offsets)

    # Count valid indices and compact them
    mask = indices != -1

    # Use prefix sum to compute destination positions for valid elements
    # For each position, count how many valid elements are before it
    prefix_sum = tl.cumsum(mask.to(tl.int32), axis=0) - 1

    # Store valid indices to their compacted positions
    dst_positions = dst_start + prefix_sum
    tl.store(kv_indices_ptr + dst_positions, indices, mask=mask)


def get_valid_kv_indices(
    page_table_1: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    bs: int,
):
    """
    Extract valid indices from page_table_1 into kv_indices buffer.

    Args:
        page_table_1: [bs, topk] page table with -1 as invalid
        kv_indptr: [bs + 1] cumulative count of valid indices per batch
        kv_indices: [bs * topk] pre-allocated output buffer
        bs: batch size
    """
    topk = page_table_1.shape[1]
    grid = (bs,)
    _get_valid_kv_indices_kernel[grid](
        page_table_1,
        kv_indptr,
        kv_indices,
        bs,
        topk,
    )


@triton.jit
def _hadamard_transform_kernel(
    x_ptr,
    out_ptr,
    scale: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < dim
    row_offset = row * dim
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for k in tl.static_range(0, BLOCK_D):
        x_k = tl.load(x_ptr + row_offset + k, mask=k < dim, other=0.0).to(tl.float32)
        parity = tl.popcount(cols & k) & 1
        sign = tl.where(parity == 0, 1.0, -1.0)
        acc += x_k * sign

    tl.store(out_ptr + row_offset + cols, acc * scale, mask=mask)


def hadamard_transform_optimized(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    x: (..., dim) - dim must be a power of 2
    """
    x_shape = x.shape
    dim = x.shape[-1]
    assert (
        dim & (dim - 1)
    ) == 0, "Hidden size must be a power of 2 for Hadamard transform."

    x_2d = x.contiguous().view(-1, dim)
    out = torch.empty_like(x_2d)
    block_d = triton.next_power_of_2(dim)
    num_warps = min(max(block_d // 32, 1), 8)
    _hadamard_transform_kernel[(x_2d.shape[0],)](
        x_2d,
        out,
        float(scale),
        dim,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return out.view(*x_shape)


@triton.jit
def _fused_gate_scale_kernel(
    weights_ptr,
    q_scale_ptr,
    out_ptr,
    scale,
    M,
    K,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    w = tl.load(weights_ptr + pid_m)
    w_scaled = w * scale
    row_start_idx = pid_m * K
    for k_offset in range(0, K, BLOCK_K):
        cols = k_offset + tl.arange(0, BLOCK_K)
        mask = cols < K

        q_ptrs = q_scale_ptr + row_start_idx + cols
        out_ptrs = out_ptr + row_start_idx + cols
        q = tl.load(q_ptrs, mask=mask)
        out = w_scaled * q
        tl.store(out_ptrs, out, mask=mask)


def fused_get_logits_head_gate_triton(
    weights: torch.Tensor,
    q_scale: torch.Tensor,
    n_heads: int,
    softmax_scale: float,
) -> torch.Tensor:
    weights = weights.contiguous()
    q_scale = q_scale.contiguous()

    K = q_scale.size(-1)
    M = weights.numel()

    out_dtype = torch.promote_types(weights.dtype, q_scale.dtype)
    out = torch.empty_like(q_scale, dtype=out_dtype)

    scale = softmax_scale * (n_heads**-0.5)
    block_k = triton.next_power_of_2(K)
    if block_k > 1024:
        block_k = 1024

    grid = (M,)
    _fused_gate_scale_kernel[grid](
        weights,
        q_scale,
        out,
        scale,
        M,
        K,
        BLOCK_K=block_k,
    )
    return out
