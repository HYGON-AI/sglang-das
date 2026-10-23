from typing import Optional

import torch
import triton
import triton.language as tl
from torch import nn

try:
    from sglang.srt.custom_op import CustomOp
except ImportError:
    from sglang.kernels.fused_op import BaseFusedOp

    class CustomOp(BaseFusedOp):
        """Compatibility shim for the pre-BaseFusedOp WeLM operators."""

        def forward_native(self, *args, **kwargs):
            return self.forward_cuda(*args, **kwargs)

from sglang.kernels.ops.attention.rope import FusedSetKVBufferArg
from sglang.srt.layers.rotary_embedding import RotaryEmbedding


def _get_num_sms(multiplier: int = 1) -> int:
    return (
        torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        * multiplier
    )


def _router_matmul_get_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": block_m,
                "BLOCK_SIZE_N": block_n,
                "BLOCK_SIZE_K": block_k,
                "GROUP_SIZE_M": 8,
            },
            num_stages=num_stages,
            num_warps=8,
        )
        for block_m in [128]
        for block_n in [16, 32]
        for block_k in [64, 128]
        for num_stages in [3, 4]
    ]


@triton.jit
def _router_matmul_compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.autotune(configs=_router_matmul_get_configs(), key=["N", "K"], restore_value=["c_ptr"])
@triton.jit
def mmq_style_router_linear_kernel(  # pylint: disable=too-many-arguments,too-many-locals
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS,  # runtime: a constexpr here recompiles per row count
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        pid_m, pid_n = _router_matmul_compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = (
            c_ptr
            + stride_cm * offs_cm[:, None].to(tl.int64)
            + stride_cn * offs_cn[None, :].to(tl.int64)
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for ki in tl.range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (
                offs_am[:, None].to(tl.int64) * stride_am
                + offs_k[None, :].to(tl.int64) * stride_ak
            )
            b_ptrs = b_ptr + (
                offs_k[:, None].to(tl.int64) * stride_bk
                + offs_bn[None, :].to(tl.int64) * stride_bn
            )

            a = tl.load(
                a_ptrs,
                mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K,
                other=0.0,
            ).to(tl.float32)
            b = tl.load(
                b_ptrs,
                mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K,
                other=0.0,
            ).to(tl.float32)
            accumulator = tl.dot(a, b, accumulator)

        c = accumulator.to(c_ptr.dtype.element_ty)
        tl.store(c_ptrs, c, mask=c_mask)


def mmq_style_router_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2
    assert weight.dim() == 2
    assert x.shape[1] == weight.shape[1]
    assert x.dtype in (torch.bfloat16, torch.float16)
    weight = weight.to(x.dtype)
    weight_t = weight.t()
    output = torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=torch.float32)
    tokens, hidden_size = x.shape
    num_experts = weight.shape[0]
    num_sms = min(
        _get_num_sms(),
        triton.cdiv(tokens, 128) * triton.cdiv(num_experts, 16),
    )
    mmq_style_router_linear_kernel[(num_sms,)](
        x,
        weight_t,
        output,
        tokens,
        num_experts,
        hidden_size,
        x.stride(0),
        x.stride(1),
        weight_t.stride(0),
        weight_t.stride(1),
        output.stride(0),
        output.stride(1),
        NUM_SMS=num_sms,
    )
    return output


@triton.jit
def mmq_style_expert_bias_topk_kernel(
    scores_ptr,
    bias_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    M,
    N: tl.constexpr,
    score_stride_m,
    score_stride_n,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    scores = tl.load(
        scores_ptr + row * score_stride_m + offs * score_stride_n,
        mask=mask,
        other=-float("inf"),
    )
    scores = tl.where(scores == scores, scores, -float("inf"))
    bias = tl.load(bias_ptr + offs, mask=mask, other=0.0)
    routing_scores = tl.where(mask, scores + bias, -float("inf"))
    candidate_mask = mask

    elems_per_copy = 4
    copy_stride = 32 * elems_per_copy
    lane_idx = (offs % copy_stride) // elems_per_copy
    local_idx = (offs // copy_stride) * elems_per_copy + (offs % elems_per_copy)
    tie_rank = lane_idx * (N // 32) + local_idx
    invalid_rank = N + 1

    for k in tl.static_range(0, TOPK):
        max_routing_score = tl.max(routing_scores, axis=0)
        selected_rank = tl.min(
            tl.where(
                (routing_scores == max_routing_score) & candidate_mask,
                tie_rank,
                invalid_rank,
            ),
            axis=0,
        )
        selected_idx = tl.min(
            tl.where((tie_rank == selected_rank) & candidate_mask, offs, invalid_rank),
            axis=0,
        )
        selected_score = tl.max(
            tl.where((offs == selected_idx) & candidate_mask, scores, -float("inf")),
            axis=0,
        )
        tl.store(topk_weights_ptr + row * TOPK + k, selected_score)
        tl.store(topk_ids_ptr + row * TOPK + k, selected_idx)
        candidate_mask = candidate_mask & (offs != selected_idx)
        routing_scores = tl.where(
            candidate_mask, routing_scores, -float("inf")
        )


def mmq_style_expert_bias_topk(
    scores: torch.Tensor, expert_bias: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    assert scores.dim() == 2
    assert expert_bias.dim() == 1
    assert scores.shape[1] == expert_bias.shape[0]
    assert scores.dtype == torch.float32
    assert expert_bias.dtype == torch.float32
    assert scores.is_cuda and expert_bias.is_cuda

    num_tokens, num_experts = scores.shape
    assert num_experts % 128 == 0
    topk_weights = torch.empty(
        (num_tokens, topk), dtype=torch.float32, device=scores.device
    )
    topk_ids = torch.empty((num_tokens, topk), dtype=torch.int64, device=scores.device)
    mmq_style_expert_bias_topk_kernel[(num_tokens,)](
        scores,
        expert_bias,
        topk_weights,
        topk_ids,
        num_tokens,
        num_experts,
        scores.stride(0),
        scores.stride(1),
        TOPK=topk,
        BLOCK_SIZE=triton.next_power_of_2(num_experts),
    )
    return topk_weights, topk_ids


@triton.jit
def _do_rms_norm(hidden, gamma, cols: int, eps: tl.constexpr):
    hidden = hidden.to(gamma.dtype).to(tl.float32)
    inv_rms = tl.math.rsqrt(tl.sum(hidden * hidden, axis=-1) / cols + eps)
    out = hidden * inv_rms
    out *= gamma
    return out


@triton.jit
def _do_mmq_rms_norm(hidden, gamma, cols: int, eps: tl.constexpr):
    hidden = hidden.to(gamma.dtype)
    hidden = hidden.to(tl.float32)
    inv_rms = tl.math.rsqrt(tl.sum(hidden * hidden, axis=-1) / cols + eps)
    out = hidden * inv_rms
    out *= gamma
    return out, inv_rms


@triton.jit
def mmq_style_norm_after_attn_kernel(
    hidden_states_ptr: tl.tensor,
    residual_ptr: tl.tensor,
    onorm_gamma_ptr: tl.tensor,
    rnorm_gamma_ptr: tl.tensor,
    output_ptr: tl.tensor,
    residual_out_ptr: tl.tensor,
    fp32_out_ptr: tl.tensor,
    rows: int,
    cols: tl.constexpr,
    eps: float,
    NUM_SMS,  # runtime: a constexpr here recompiles per row count
    BLOCK_SIZE: tl.constexpr,  # pylint: disable=invalid-name
):
    cols_offsets = tl.arange(0, BLOCK_SIZE)
    mask = cols_offsets < cols
    onorm_gamma = tl.load(onorm_gamma_ptr + cols_offsets, mask=mask, other=0.0)
    rnorm_gamma = tl.load(rnorm_gamma_ptr + cols_offsets, mask=mask, other=0.0)
    output_dtype = output_ptr.dtype.element_ty

    for row_id in tl.range(tl.program_id(0), rows, NUM_SMS, num_stages=2):
        offsets = (row_id * cols + cols_offsets).to(tl.int64)
        hs = tl.load(hidden_states_ptr + offsets, mask=mask, other=0.0)
        onorm_out, _ = _do_mmq_rms_norm(hs, onorm_gamma, cols, eps)
        hs = onorm_out.to(hs.dtype)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
        hs += residual
        rnorm_out, _ = _do_mmq_rms_norm(hs, rnorm_gamma, cols, eps)
        tl.store(residual_out_ptr + offsets, hs, mask=mask)
        tl.store(fp32_out_ptr + offsets, rnorm_out, mask=mask)
        tl.store(output_ptr + offsets, rnorm_out.to(output_dtype), mask=mask)


def mmq_style_norm_after_attn(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    onorm_weight: torch.Tensor,
    rnorm_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert hidden_states.dim() == 2
    assert residual.dim() == 2
    assert hidden_states.shape == residual.shape
    hidden_states = hidden_states.contiguous()
    residual = residual.contiguous()
    onorm_weight = onorm_weight.contiguous()
    rnorm_weight = rnorm_weight.contiguous()
    output = torch.empty_like(hidden_states)
    residual_out = torch.empty_like(hidden_states, dtype=torch.float32)
    fp32_out = torch.empty_like(hidden_states, dtype=torch.float32)
    rows, cols = hidden_states.shape
    num_sms = min(rows, _get_num_sms(multiplier=8))
    block_size = triton.next_power_of_2(cols)
    mmq_style_norm_after_attn_kernel[(num_sms,)](
        hidden_states,
        residual,
        onorm_weight,
        rnorm_weight,
        output,
        residual_out,
        fp32_out,
        rows,
        cols,
        eps,
        num_sms,
        block_size,
    )
    return output, residual_out, fp32_out


@triton.jit
def rms_norm_kernel(  # pylint: disable=too-many-arguments,too-many-locals
    hidden_states_ptr: tl.tensor,
    reisdual_ptr: tl.tensor,
    gamma_ptr: tl.tensor,
    out_ptr: tl.tensor,
    out_residual_ptr: tl.tensor,
    out_copy_ptr: tl.tensor,
    rows: int,
    cols: tl.constexpr,
    eps: float,
    hidden_states_row_stride: int,
    hidden_states_num_kv: int,
    hidden_states_kv_stride: int,
    residual_row_stride: int,
    residual_after_layernorm: tl.constexpr,
    NUM_SMS,  # runtime: a constexpr here recompiles per row count
    BLOCK_SIZE: tl.constexpr,  # pylint: disable=invalid-name
):
    row_start = tl.program_id(0)
    cols_off = tl.arange(0, BLOCK_SIZE)
    mask = cols_off < cols
    gamma_shm = tl.load(gamma_ptr + cols_off, mask=mask, other=0.0)

    output_dtype = out_ptr.dtype.element_ty
    for row_id in tl.range(row_start, rows, NUM_SMS, num_stages=2):
        kv_idx = row_id // hidden_states_num_kv
        row_idx = row_id % hidden_states_num_kv
        kv_off = kv_idx * hidden_states_kv_stride
        h_offs = row_idx * hidden_states_row_stride + kv_off + cols_off
        #h_offs = (row_id * hidden_states_row_stride + cols_off).to(tl.int64)
        r_offs = (row_id * residual_row_stride + cols_off).to(tl.int64)
        h = tl.load(hidden_states_ptr + h_offs, mask=mask, other=0.0).to(tl.float32)
        if reisdual_ptr is not None:
            r = tl.load(reisdual_ptr + r_offs, mask=mask, other=0.0).to(tl.float32)
            h = h + r

        output_offs = (row_id * cols + cols_off).to(tl.int64)
        if not residual_after_layernorm and out_residual_ptr is not None:
            tl.store(
                out_residual_ptr + output_offs,
                h.to(reisdual_ptr.dtype.element_ty),
                mask=mask,
            )

        out = _do_rms_norm(h, gamma_shm, cols, eps)
        if out_copy_ptr is not None:
            tl.store(out_copy_ptr + output_offs, out, mask=mask)

        out = out.to(output_dtype)
        if residual_after_layernorm:
            tl.store(out_residual_ptr + output_offs, out, mask=mask)

        tl.store(out_ptr + output_offs, out, mask=mask)


class WelmV4FusedRMSNorm(CustomOp):
    def __init__(
        self, hidden_size: int, eps: float = 1e-6, weight_dtype: Optional = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=weight_dtype))
        self.num_sms = _get_num_sms(multiplier=8)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        residual_after_layernorm: bool = False,
        clone_fp32_out: bool = False,
        output_dtype: Optional[torch.dtype] = None,
    ):
        assert x.dim() in [2, 3]
        output = torch.empty_like(x, dtype=output_dtype or x.dtype)
        fp32_out = None
        out_residual = None
        if residual_after_layernorm:
            out_residual = torch.empty_like(x)
        elif residual is not None:
            out_residual = torch.empty_like(residual)
        if clone_fp32_out:
            fp32_out = torch.empty_like(x, dtype=torch.float32)
        cols = x.shape[-1]
        rows = x.numel() // cols

        if residual is not None:
            assert residual.is_contiguous()
            residual_row_stride = residual.stride(0)
        else:
            residual_row_stride = 0
        x_row_stride = x.stride(-2)
        x_num_kv = x.shape[-2]
        if x.dim() == 2:
            kv_stride = x.numel()
        else:
            kv_stride = x.stride(0)

        num_sms = min(rows, self.num_sms)
        block_size = triton.next_power_of_2(cols)
        rms_norm_kernel[(num_sms,)](
            x,
            residual,
            self.weight,
            output,
            out_residual,
            fp32_out,
            rows,
            cols,
            self.eps,
            x_row_stride,
            x_num_kv,
            kv_stride,
            residual_row_stride,
            residual_after_layernorm,
            num_sms,
            block_size,
        )
        if out_residual is None:
            out_residual = x

        if clone_fp32_out:
            return output, out_residual, fp32_out
        else:
            return output, out_residual


# The ops below are used inside the vision encoder of ``welmv4_vlm`` — they
# are NOT part of the LLM decoder path. They mirror the matching kernels in
# the upstream WeLMV4 training stack so the inference forward stays
# numerically aligned with training.
#
# If you touch these, keep them semantically equivalent to the upstream
# sources they were ported from.
# ---------------------------------------------------------------------------


# QuickGELU activation: ``out = x * sigmoid(1.702 * x)`` with fp32 compute
# and an implicit cast back to the input dtype on store.
_QUICK_GELU_SLOPE = 1.702


@triton.jit
def _welmv4_vision_quick_gelu_fwd_kernel(
    x_ptr: tl.tensor,
    out_ptr: tl.tensor,
    n_elements: int,
    slope: tl.constexpr,
    num_sms: tl.constexpr,
    block_size: tl.constexpr,
    num_stages: tl.constexpr = 2,
):
    pid = tl.program_id(axis=0)
    for block_start in tl.range(
        pid * block_size, n_elements, num_sms * block_size, num_stages=num_stages
    ):
        offsets = (block_start + tl.arange(0, block_size)).to(tl.int64)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0)
        x_fp32 = x.to(tl.float32)
        out = x_fp32 * tl.sigmoid(slope * x_fp32)
        tl.store(out_ptr + offsets, out, mask=mask)


def welmv4_vision_quick_gelu(x: torch.Tensor) -> torch.Tensor:
    """QuickGELU activation used inside ``WeLMV4VisionMLP``.

    Computes ``x * sigmoid(1.702 * x)`` with fp32 accumulation and casts
    back to the input dtype on store. Only used by the WeLMV4 vision
    encoder; do not call this from the LLM decoder.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return out
    # ``num_sms`` only controls how many CTAs launch — the ``tl.range``
    # stride in the kernel keeps the output correct for any value. 132 is
    # an upper bound that matches the H100 SM count.
    num_sms = 132
    block_size = 1024
    with torch.cuda.device(x.device):
        _welmv4_vision_quick_gelu_fwd_kernel[(num_sms,)](
            x_ptr=x,
            out_ptr=out,
            n_elements=n_elements,
            slope=_QUICK_GELU_SLOPE,
            num_sms=num_sms,
            block_size=block_size,
        )
    return out


@triton.jit


def mmq_style_shared_experts_add_residual_rms_norm_kernel(
    experts_output_ptr: tl.tensor,
    shared_output_ptr: tl.tensor,
    residual_ptr: tl.tensor,
    gamma_ptr: tl.tensor,
    out_ptr: tl.tensor,
    rows: int,
    cols: tl.constexpr,
    eps: float,
    NUM_SMS,  # runtime: a constexpr here recompiles per row count
    BLOCK_SIZE: tl.constexpr,  # pylint: disable=invalid-name
):
    row_start = tl.program_id(0)
    cols_off = tl.arange(0, BLOCK_SIZE)
    mask = cols_off < cols
    gamma_shm = tl.load(gamma_ptr + cols_off, mask=mask, other=0.0)
    output_dtype = out_ptr.dtype.element_ty

    for row_id in tl.range(row_start, rows, NUM_SMS, num_stages=2):
        offs = (row_id * cols + cols_off).to(tl.int64)
        experts_output = tl.load(
            experts_output_ptr + offs, mask=mask, other=0.0
        ).to(tl.float32)
        shared_output = tl.load(
            shared_output_ptr + offs, mask=mask, other=0.0
        ).to(tl.float32)
        residual = tl.load(residual_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        hidden_state = experts_output + shared_output
        hidden_state = hidden_state + residual
        out = _do_rms_norm(hidden_state, gamma_shm, cols, eps)
        tl.store(out_ptr + offs, out.to(output_dtype), mask=mask)


def mmq_style_shared_experts_add_residual_rms_norm(
    experts_output: torch.Tensor,
    shared_output: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    assert experts_output.dim() == 2
    assert shared_output.shape == experts_output.shape
    assert residual.shape == experts_output.shape
    assert gamma.dim() == 1 and gamma.shape[0] == experts_output.shape[1]
    assert experts_output.is_cuda and shared_output.is_cuda and residual.is_cuda
    assert experts_output.is_contiguous()
    assert shared_output.is_contiguous()
    assert residual.is_contiguous()

    output = torch.empty_like(experts_output)
    rows, cols = experts_output.shape
    num_sms = min(rows, _get_num_sms(multiplier=8))
    block_size = triton.next_power_of_2(cols)
    mmq_style_shared_experts_add_residual_rms_norm_kernel[(num_sms,)](
        experts_output,
        shared_output,
        residual,
        gamma,
        output,
        rows,
        cols,
        eps,
        num_sms,
        block_size,
    )
    return output


@triton.jit
def mmq_style_k_rms_norm_kernel(
    x_ptr: tl.tensor,
    gamma_ptr: tl.tensor,
    out_ptr: tl.tensor,
    tokens: int,
    kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    x_token_stride: int,
    out_token_stride: int,
    eps: float,
    NUM_SMS,  # runtime: a constexpr here recompiles per row count
    KV_HEADS_BLOCK: tl.constexpr,  # pylint: disable=invalid-name
):
    head_dim_offs = tl.arange(0, head_dim)
    kv_head_offs = tl.arange(0, KV_HEADS_BLOCK)
    block_offs = kv_head_offs[:, None] * head_dim + head_dim_offs[None, :]
    gamma = tl.load(gamma_ptr + head_dim_offs)

    for token_id in tl.range(tl.program_id(0), tokens, NUM_SMS, num_stages=4):
        in_offs = token_id * x_token_stride.to(tl.int64) + block_offs
        x = tl.load(
            x_ptr + in_offs, mask=kv_head_offs[:, None] < kv_heads, other=0.0
        ).to(tl.float32)
        inv_rms = tl.math.rsqrt(tl.sum(x * x, axis=-1) / head_dim + eps)
        out = x * inv_rms[:, None]
        out *= gamma[None, :]

        out_offs = token_id * out_token_stride.to(tl.int64) + block_offs
        tl.store(out_ptr + out_offs, out, mask=kv_head_offs[:, None] < kv_heads)


def mmq_style_k_rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float):
    assert x.dim() == 3
    assert x.is_contiguous()
    output = torch.empty_like(x)
    tokens, kv_heads, head_dim = x.shape
    num_sms = min(tokens, _get_num_sms())
    mmq_style_k_rms_norm_kernel[(num_sms,)](
        x,
        gamma,
        output,
        tokens,
        kv_heads,
        head_dim,
        x.stride(0),
        output.stride(0),
        eps,
        num_sms,
        triton.next_power_of_2(kv_heads),
    )
    return output


@triton.jit
def sigmoid_mul_kernel(
    x: tl.tensor,
    y: tl.tensor,
    rows: int,
    cols: tl.constexpr,
    y_row_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_SMS,  # runtime: a constexpr here recompiles per row count
):
    row_start = tl.program_id(0)
    col_off = tl.arange(0, BLOCK_SIZE)
    mask = col_off < cols
    for row_id in tl.range(row_start, rows, NUM_SMS, num_stages=4):
        y_off = row_id * y_row_stride + col_off
        y_data = tl.load(y + y_off, mask=mask, other=0.0)
        x_data = tl.load(x + row_id).to(tl.float32)
        out_data = tl.sigmoid(x_data).to(y.dtype.element_ty) * y_data
        tl.store(y + y_off, out_data, mask=mask)


# return sigmoid(x) * y
def inplace_sigmoid_mul(x: torch.Tensor, y: torch.Tensor):
    num_sms = _get_num_sms(multiplier=8)
    cols = y.shape[-1]
    rows = y.numel() // cols
    block_size = triton.next_power_of_2(cols)
    assert x.is_contiguous()
    sigmoid_mul_kernel[(num_sms,)](x, y, rows, cols, y.stride(-2), block_size, num_sms)


@triton.jit
def _rope(  # pylint: disable=too-many-arguments, too-many-locals
    data_ptr: tl.tensor,
    cos: tl.tensor,
    sin: tl.tensor,
    num_heads: tl.constexpr,
    num_heads_blocked: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
):
    half_rope_dim: tl.constexpr = rope_dim // 2
    num_head_offset = tl.arange(0, num_heads_blocked)
    rope_dim_offset = tl.arange(0, rope_dim)
    mask = num_head_offset[:, None] < num_heads
    x = tl.load(
        data_ptr + num_head_offset[:, None] * head_dim + rope_dim_offset[None, :],
        mask=mask,
    )
    x = x.reshape(num_heads_blocked, 2, half_rope_dim).trans(0, 2, 1)
    x1, x2 = x.split()
    x_out1 = x1 * cos - x2 * sin
    x_out2 = x1 * sin + x2 * cos
    x_out = tl.join(x_out1, x_out2).trans(0, 2, 1).reshape(num_heads_blocked, rope_dim)
    tl.store(
        data_ptr + num_head_offset[:, None] * head_dim + rope_dim_offset[None, :],
        x_out,
        mask=mask,
    )


@triton.jit
def _welmv4_inplace_rope_kernel(
    q_ptr: tl.tensor,
    k_ptr: tl.tensor,
    position_ptr: tl.tensor,
    cos_sin_cache_ptr: tl.tensor,
    last_index_ptr: tl.tensor,
    N: int,
    BS: int,
    q_token_stride: tl.constexpr,
    k_token_stride: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    num_sms,  # runtime: a constexpr here recompiles per token count
    num_stages: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_k_heads: tl.constexpr,
    num_q_heads_blocked: tl.constexpr,
    num_k_heads_blocked: tl.constexpr,
):
    half_rope_dim: tl.constexpr = rope_dim // 2
    cos_off = tl.arange(0, half_rope_dim)
    sin_off = tl.arange(half_rope_dim, rope_dim)
    for token_id in tl.range(tl.program_id(0), N, num_sms, num_stages=num_stages):
        position_id = tl.load(position_ptr + token_id)
        cos_sin_cache = tl.load(cos_sin_cache_ptr + position_id * rope_dim + cos_off)
        sin_sin_cache = tl.load(cos_sin_cache_ptr + position_id * rope_dim + sin_off)
        q_data_ptr = q_ptr + token_id * q_token_stride + head_dim - rope_dim
        k_data_ptr = k_ptr + token_id * k_token_stride + head_dim - rope_dim
        _rope(
            k_data_ptr,
            cos_sin_cache,
            sin_sin_cache,
            num_k_heads,
            num_k_heads_blocked,
            head_dim,
            rope_dim,
        )
        if last_index_ptr is not None:
            if token_id < BS:
                position_id = tl.load(last_index_ptr + token_id)
                position_id = tl.load(position_ptr + position_id)
                cos_sin_cache = tl.load(
                    cos_sin_cache_ptr + position_id * rope_dim + cos_off
                )
                sin_sin_cache = tl.load(
                    cos_sin_cache_ptr + position_id * rope_dim + sin_off
                )
                _rope(
                    q_data_ptr,
                    cos_sin_cache,
                    sin_sin_cache,
                    num_q_heads,
                    num_q_heads_blocked,
                    head_dim,
                    rope_dim,
                )
        else:
            _rope(
                q_data_ptr,
                cos_sin_cache,
                sin_sin_cache,
                num_q_heads,
                num_q_heads_blocked,
                head_dim,
                rope_dim,
            )


class WelmV4InplaceRotaryEmbedding(RotaryEmbedding):
    """WelmV4 rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )
        self.num_sms = _get_num_sms(multiplier=8)

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,
        last_index: Optional[torch.Tensor] = None,
    ):
        query = query.view(query.shape[0], -1, self.head_size)
        key = key.view(key.shape[0], -1, self.head_size)
        N = positions.shape[0]
        num_sms = min(N, self.num_sms)
        num_stages = 4
        BS = last_index.numel() if last_index is not None else 0
        _welmv4_inplace_rope_kernel[(num_sms,)](
            query,
            key,
            positions,
            self.cos_sin_cache,
            last_index,
            N,
            BS,
            query.stride(0),
            key.stride(0),
            self.head_size,
            self.rotary_dim,
            num_sms,
            num_stages,
            query.shape[-2],
            key.shape[-2],
            triton.next_power_of_2(query.shape[-2]),
            triton.next_power_of_2(key.shape[-2]),
        )
        return query, key

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        return s


# ---------------------------------------------------------------------------
# Vision-encoder RoPE (WeLMV4 VLM only)
#
# Half-rotate RoPE applied to the q and k slots of a packed
# ``(seq_len, q_heads + 2*kv_heads, head_dim)`` qkv tensor; the v slot is
# copied through unchanged. The kernel uses one fused FMA in fp32 and
# stores the result as bf16, which is more numerically stable than the
# eager ``mul -> rotate-half -> mul -> add -> cast`` sequence used by
# the generic vision RoPE in ``sglang.srt.layers.attention.vision``.
#
# The kernel works for any even ``head_dim`` (we pad the inner
# ``head_dim/2`` axis up to the next power of 2 and mask the unused tail),
# so it is not subject to the power-of-2 constraint of generic
# ``qkv_part_rope`` triton implementations. WeLMV4 ships with
# ``head_dim = 72`` so the masking is required, not optional.
# ---------------------------------------------------------------------------


@triton.jit
def _welmv4_qk_part_rope_kernel(  # pylint: disable=too-many-arguments,too-many-locals
    qkv_ptr: tl.tensor,
    cos_ptr: tl.tensor,
    sin_ptr: tl.tensor,
    qkv_out_ptr: tl.tensor,
    qkv_row_stride: tl.constexpr,
    cos_row_stride: tl.constexpr,
    sin_row_stride: tl.constexpr,
    seq_len: tl.int64,
    qk_rope_head_dim: tl.constexpr,
    q_heads: tl.constexpr,
    kv_heads: tl.constexpr,
    num_sms: tl.constexpr,
    num_stages: tl.constexpr,
    q_heads_pow2: tl.constexpr,  # heuristic
    kv_heads_pow2: tl.constexpr,  # heuristic
    half_rope_dim_pow2: tl.constexpr,  # heuristic; pad of qk_rope_head_dim//2
):
    """Apply the half-rotate RoPE formula to the q and k slots of a
    packed ``(seq_len, q_heads + 2*kv_heads, head_dim)`` qkv tensor.
    The v slot is copied through as-is. ``head_dim`` here equals
    ``qk_rope_head_dim`` — we don't support a non-zero
    ``qk_nope_head_dim`` because WeLMV4 doesn't need it (and the
    extra branching would just be dead code).

    The kernel is layout-equivalent to a generic qkv_part_rope triton
    kernel, with the head_dim power-of-2 constraint relaxed via masking.
    """
    half_rope_head_dim: tl.constexpr = qk_rope_head_dim // 2
    # 1D arange of length pow2, mask off the unused tail. half_rope_dim_pow2
    # is the next-power-of-2 of half_rope_head_dim (e.g. 64 for half=36).
    half_offs = tl.arange(0, half_rope_dim_pow2)
    half_mask = half_offs < half_rope_head_dim

    # Persistent CTA loop over seq tokens.
    for seq_id in tl.range(tl.program_id(0), seq_len, num_sms, num_stages=num_stages):
        seq64 = seq_id.to(tl.int64)
        cos = tl.load(
            cos_ptr + seq64 * cos_row_stride + half_offs, mask=half_mask, other=0.0
        )
        sin = tl.load(
            sin_ptr + seq64 * sin_row_stride + half_offs, mask=half_mask, other=0.0
        )

        token_offs = seq64 * qkv_row_stride

        # ---- q slot ----
        q_heads_offs = tl.arange(0, q_heads_pow2)
        q_head_mask = q_heads_offs < q_heads
        # offsets for the lo half (first half_rope_head_dim of each head)
        offs_q_lo = (
            token_offs + q_heads_offs[:, None] * qk_rope_head_dim + half_offs[None, :]
        )
        offs_q_hi = offs_q_lo + half_rope_head_dim
        mask_q_lo = q_head_mask[:, None] & half_mask[None, :]
        x_lo = tl.load(qkv_ptr + offs_q_lo, mask=mask_q_lo, other=0.0).to(tl.float32)
        x_hi = tl.load(qkv_ptr + offs_q_hi, mask=mask_q_lo, other=0.0).to(tl.float32)
        out_lo = x_lo * cos[None, :] - x_hi * sin[None, :]
        out_hi = x_lo * sin[None, :] + x_hi * cos[None, :]
        tl.store(qkv_out_ptr + offs_q_lo, out_lo, mask=mask_q_lo)
        tl.store(qkv_out_ptr + offs_q_hi, out_hi, mask=mask_q_lo)

        # ---- k slot (offset by q_heads * head_dim from start of token) ----
        kv_heads_offs = tl.arange(0, kv_heads_pow2)
        kv_head_mask = kv_heads_offs < kv_heads
        k_token_offs = token_offs + q_heads * qk_rope_head_dim
        offs_k_lo = (
            k_token_offs
            + kv_heads_offs[:, None] * qk_rope_head_dim
            + half_offs[None, :]
        )
        offs_k_hi = offs_k_lo + half_rope_head_dim
        mask_k_lo = kv_head_mask[:, None] & half_mask[None, :]
        x_lo = tl.load(qkv_ptr + offs_k_lo, mask=mask_k_lo, other=0.0).to(tl.float32)
        x_hi = tl.load(qkv_ptr + offs_k_hi, mask=mask_k_lo, other=0.0).to(tl.float32)
        out_lo = x_lo * cos[None, :] - x_hi * sin[None, :]
        out_hi = x_lo * sin[None, :] + x_hi * cos[None, :]
        tl.store(qkv_out_ptr + offs_k_lo, out_lo, mask=mask_k_lo)
        tl.store(qkv_out_ptr + offs_k_hi, out_hi, mask=mask_k_lo)

        # ---- v slot: copy through unchanged ----
        v_token_offs = k_token_offs + kv_heads * qk_rope_head_dim
        # Load full head_dim (rope+nope, but here qk_nope_head_dim=0 so
        # just head_dim) for each kv head; since head_dim isn't pow2 we
        # use two arange-pow2 chunks.
        offs_v_lo = (
            v_token_offs
            + kv_heads_offs[:, None] * qk_rope_head_dim
            + half_offs[None, :]
        )
        offs_v_hi = offs_v_lo + half_rope_head_dim
        v_lo = tl.load(qkv_ptr + offs_v_lo, mask=mask_k_lo, other=0.0)
        v_hi = tl.load(qkv_ptr + offs_v_hi, mask=mask_k_lo, other=0.0)
        tl.store(qkv_out_ptr + offs_v_lo, v_lo, mask=mask_k_lo)
        tl.store(qkv_out_ptr + offs_v_hi, v_hi, mask=mask_k_lo)


def welmv4_vision_qkv_part_rope(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    num_sms: int = 132,
) -> torch.Tensor:
    """Apply half-rotate RoPE to the q and k slots of ``qkv`` and copy the
    v slot through unchanged. Returns a new bf16 tensor of the same shape.

    Args:
        qkv: ``(seq_len, q_heads + 2*kv_heads, head_dim)`` bf16,
            contiguous. ``head_dim`` must be even but does NOT need to
            be a power of 2 (we mask).
        cos / sin: ``(seq_len, head_dim/2)`` fp32, contiguous.
        q_heads: number of q heads.
        kv_heads: number of k (== number of v) heads.

    Returns:
        ``qkv_out`` bf16, same shape as ``qkv``.
    """
    if qkv.dtype != torch.bfloat16:
        raise TypeError(f"qkv must be bf16, got {qkv.dtype}")
    if qkv.dim() != 3:
        raise ValueError(f"qkv must be 3D (seq, heads*3, head_dim), got {qkv.shape}")
    seq_len, total_heads, head_dim = qkv.shape
    if total_heads != q_heads + 2 * kv_heads:
        raise ValueError(
            f"total_heads={total_heads} != q_heads({q_heads}) + 2*kv_heads({kv_heads})"
        )
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    half = head_dim // 2
    if cos.shape != (seq_len, half) or sin.shape != (seq_len, half):
        raise ValueError(
            f"cos/sin must be (seq_len={seq_len}, head_dim/2={half}); "
            f"got cos={tuple(cos.shape)}, sin={tuple(sin.shape)}"
        )
    if cos.dtype != torch.float32:
        cos = cos.to(torch.float32)
    if sin.dtype != torch.float32:
        sin = sin.to(torch.float32)
    cos = cos.contiguous()
    sin = sin.contiguous()
    if not qkv.is_contiguous():
        qkv = qkv.contiguous()

    qkv_out = torch.empty_like(qkv)
    q_heads_pow2 = triton.next_power_of_2(q_heads)
    kv_heads_pow2 = triton.next_power_of_2(kv_heads)
    half_rope_dim_pow2 = triton.next_power_of_2(half)
    num_stages = 3

    with torch.cuda.device(qkv.device):
        _welmv4_qk_part_rope_kernel[(num_sms,)](
            qkv_ptr=qkv,
            cos_ptr=cos,
            sin_ptr=sin,
            qkv_out_ptr=qkv_out,
            qkv_row_stride=total_heads * head_dim,  # row stride in elements
            cos_row_stride=cos.stride(0),
            sin_row_stride=sin.stride(0),
            seq_len=seq_len,
            qk_rope_head_dim=head_dim,
            q_heads=q_heads,
            kv_heads=kv_heads,
            num_sms=num_sms,
            num_stages=num_stages,
            q_heads_pow2=q_heads_pow2,
            kv_heads_pow2=kv_heads_pow2,
            half_rope_dim_pow2=half_rope_dim_pow2,
        )
    return qkv_out


def welmv4_vision_apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply WeLMV4 vision-encoder RoPE to ``q`` and ``k``.

    Packs ``(q, k, v_pad)`` into the layout the triton kernel expects,
    runs ``welmv4_vision_qkv_part_rope``, and unpacks ``q_out`` and
    ``k_out``.

    Args:
        q: ``(seq_len, num_heads, head_dim)`` bf16. Contiguous; the
            function will make a contiguous copy if not.
        k: same shape as ``q``.
        cos / sin: either ``(seq_len, head_dim)`` (the SGLang vision
            encoder concatenates ``cos`` with itself before calling the
            applier) or ``(seq_len, head_dim/2)``. fp32 is preferred;
            bf16 is upcast inside ``welmv4_vision_qkv_part_rope``.

    Returns:
        ``(q_out, k_out)`` bf16, same shapes as the inputs.
    """
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise TypeError(
            f"welmv4_vision_apply_rope expects bf16 q/k; "
            f"got q.dtype={q.dtype}, k.dtype={k.dtype}"
        )
    if q.dim() != 3 or k.shape != q.shape:
        raise ValueError(
            f"welmv4_vision_apply_rope expects q,k of shape (seq, heads, head_dim) "
            f"with q.shape == k.shape; got q={tuple(q.shape)}, k={tuple(k.shape)}"
        )
    seq_len, num_heads, head_dim = q.shape

    # SGLang's vision encoder forward concatenates ``cos`` with itself
    # before passing it here (``cos = torch.cat([cos, cos], dim=-1)``).
    # The kernel only consumes the first ``head_dim/2`` columns; slice
    # them off so callers don't need to special-case.
    half = head_dim // 2
    if cos.shape[-1] == head_dim:
        cos = cos[..., :half]
        sin = sin[..., :half]

    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()

    v_pad = torch.zeros_like(q)
    qkv_in = torch.stack([q, k, v_pad], dim=1).flatten(1, 2).contiguous()
    qkv_out = welmv4_vision_qkv_part_rope(
        qkv=qkv_in,
        cos=cos,
        sin=sin,
        q_heads=num_heads,
        kv_heads=num_heads,
    )
    qkv_out = qkv_out.view(seq_len, 3, num_heads, head_dim)
    q_out = qkv_out[:, 0].contiguous()
    k_out = qkv_out[:, 1].contiguous()
    return q_out, k_out
