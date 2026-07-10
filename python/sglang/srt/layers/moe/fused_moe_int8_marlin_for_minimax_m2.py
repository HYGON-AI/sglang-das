import functools
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import vllm.envs as envs
from vllm import _custom_ops as ops
import torch
import triton
import triton.language as tl
import lmslim.envs as lsenvs

use_lightop = lsenvs.LMSLIM_USE_LIGHTOP
device_name = lsenvs.LMSLIM_GPU_NAME
num_cus = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
if use_lightop:
    from lightop import moe_gemm_marlin_w8a8, get_moe_cuda_marlin_config, fuse_silu_mul_quant
    from lightop import op as op

from lmslim.layers.gemm.int8_utils import (
   per_token_group_quant_int8, per_token_quant_int8)

import importlib.util as _iu, os as _os, sys as _sys, vllm
_mab_path = _os.path.join(
    _os.path.dirname(vllm.__file__),
    "model_executor", "layers", "fused_moe", "moe_align_block_size.py")
_mab_spec = _iu.spec_from_file_location("_vllm_moe_align_block_size_mod", _mab_path)
_mab_mod = _iu.module_from_spec(_mab_spec)
_sys.modules["_vllm_moe_align_block_size_mod"] = _mab_mod
_mab_spec.loader.exec_module(_mab_mod)
moe_align_block_size = _mab_mod.moe_align_block_size


@torch.compile
def moe_sum_reduce_torch_compile(x, out, routed_scaling_factor):
    torch.sum(x, dim=1, out=out)
    out.mul_(routed_scaling_factor)


@triton.jit
def _moe_sum_reduce_kernel(
        input_ptr,
        input_stride_0,
        input_stride_1,
        input_stride_2,
        output_ptr,
        output_stride_0,
        output_stride_1,
        token_num: int,
        topk_num: int,
        hidden_dim: int,
        routed_scaling_factor: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
        NUM_STAGE: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    token_start = token_block_id * BLOCK_M
    token_end = min((token_block_id + 1) * BLOCK_M, token_num)

    dim_start = dim_block_id * BLOCK_DIM
    dim_end = min((dim_block_id + 1) * BLOCK_DIM, hidden_dim)

    offs_dim = dim_start + tl.arange(0, BLOCK_DIM)

    for token_index in range(token_start, token_end):
        accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
        input_t_ptr = input_ptr + token_index * input_stride_0 + offs_dim
        for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
            tmp = tl.load(
                input_t_ptr + i * input_stride_1, mask=offs_dim < dim_end, other=0.0
            )
            accumulator += tmp
        accumulator = accumulator * routed_scaling_factor
        store_t_ptr = output_ptr + token_index * output_stride_0 + offs_dim
        tl.store(
            store_t_ptr,
            accumulator.to(input_ptr.dtype.element_ty),
            mask=offs_dim < dim_end,
        )


def moe_sum_reduce_triton(
        input: torch.Tensor, output: torch.Tensor, routed_scaling_factor: float
):
    assert input.is_contiguous()
    assert output.is_contiguous()

    token_num, topk_num, hidden_dim = input.shape
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim

    if token_num <= 32:
        BLOCK_M = 1
        BLOCK_DIM = 512
        NUM_STAGE = 2
        num_warps = 4

    elif token_num <= 128:
        BLOCK_M = 1
        BLOCK_DIM = 1024
        NUM_STAGE = 0
        num_warps = 2

    elif token_num <= 4096:
        BLOCK_M = 1
        BLOCK_DIM = 2048
        NUM_STAGE = 0
        num_warps = 2
    else:
        BLOCK_M = 1
        BLOCK_DIM = 2048
        NUM_STAGE = 2
        num_warps = 8

    grid = (
        triton.cdiv(token_num, BLOCK_M),
        triton.cdiv(hidden_dim, BLOCK_DIM),
    )

    _moe_sum_reduce_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        routed_scaling_factor=routed_scaling_factor,
        BLOCK_M=BLOCK_M,
        BLOCK_DIM=BLOCK_DIM,
        NUM_STAGE=NUM_STAGE,
        num_warps=num_warps,
    )
    return


def moe_reduce_dispatch(
        intermediate_cache3: torch.Tensor,
        out_hidden_states: torch.Tensor,
        begin_chunk_idx: int,
        end_chunk_idx: int,
        routed_scaling_factor: float,
        shared_output: Optional[torch.Tensor] = None,
):
    inter_cache_view = intermediate_cache3.view(*intermediate_cache3.shape)
    n = intermediate_cache3.shape[0]

    # 根据 n 大小选择不同的 reduce 实现
    # TODO: remove this assertion when callers pass different values
    assert routed_scaling_factor == 1.0, "routed_scaling_factor != 1.0 not yet supported in reduce path"
    if 1 <= n <= 4:
        moe_sum_reduce_torch_compile(inter_cache_view, out_hidden_states[begin_chunk_idx:end_chunk_idx], routed_scaling_factor)
    elif 4 < n <= 1024:
        moe_sum_reduce_triton(inter_cache_view, out_hidden_states[begin_chunk_idx:end_chunk_idx], routed_scaling_factor)
    elif 1024 < n <= 32768:
        ops.moe_sum_opt1(inter_cache_view, out_hidden_states[begin_chunk_idx:end_chunk_idx])
    else:
        ops.moe_sum(inter_cache_view, out_hidden_states[begin_chunk_idx:end_chunk_idx])

    # 根据 shared_output 是否存在决定怎么更新
    if shared_output is not None:
        out_hidden_states[begin_chunk_idx:end_chunk_idx].mul_(routed_scaling_factor).add_(shared_output[begin_chunk_idx:end_chunk_idx])
    else:
        out_hidden_states[begin_chunk_idx:end_chunk_idx].mul_(routed_scaling_factor)


def fused_experts_impl_int8_marlin(hidden_states: torch.Tensor,
                       w1: torch.Tensor,
                       w2: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       inplace: bool = False,
                       activation: str = "silu",
                       apply_router_weight_on_input: bool = False,
                       use_fp8_w8a8: bool = False,
                       use_int8_w8a8: bool = False,
                       use_int8_w8a16: bool = False,
                       use_int4_w4a16: bool = False,
                       per_channel_quant: bool = False,
                       global_num_experts: int = -1,
                       expert_map: Optional[torch.Tensor] = None,
                       w1_scale: Optional[torch.Tensor] = None,
                       w2_scale: Optional[torch.Tensor] = None,
                       w1_zp: Optional[torch.Tensor] = None,
                       w2_zp: Optional[torch.Tensor] = None,
                       a1_scale: Optional[torch.Tensor] = None,
                       a2_scale: Optional[torch.Tensor] = None,
                       block_shape: Optional[List[int]] = None,
                       use_nn_moe: Optional[bool] = False,
                       routed_scaling_factor: Optional[float] = 1.0,
                       shared_output: Optional[torch.Tensor] = None,
                       i_q: Optional[torch.Tensor] = None,
                       i_s: Optional[torch.Tensor] = None, **_):
    # Check constraints.
    assert use_int8_w8a8 is True and per_channel_quant is True , "Unsupport quant method"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]

    use_nn_moe=False

    num_tokens, K = hidden_states.shape
    E = w1.shape[0]
    N = w2.shape[1] * 64
    N2 = N * 2
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.shape[1]
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens, CHUNK_SIZE)
    if envs.VLLM_USE_GLOBAL_CACHE13:
        cache13 = torch.empty(
            (M * top_k_num * max(2 * N, K), ),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    else:
        cache13 = torch.empty(
            (M * top_k_num * max(2 * N, K), ),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
    )
    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1
    intermediate_cache1 = cache13[:M * top_k_num * N2].view(M, top_k_num, N2)
    intermediate_cache3 = cache13[:M * top_k_num * K].view(M, top_k_num, K)

    # This needs separate memory since it's used concurrently with cache1
    intermediate_cache2 = torch.empty((M * top_k_num, N),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[:tokens_in_chunk *
                                                      topk_ids.shape[1]]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
        try:
            config1, config2, status = get_moe_cuda_marlin_config(
                    E, tokens_in_chunk, N2, K, K, N, top_k_num, device_name, num_cus, compute_type
                    )
        except Exception as e:
            print(f"Warning: get_moe_cuda_config failed: {e}")
            status = False

        assert status, f'moe marlin unsupport this size E:{E}, N:{N}, K:{K}'
        if envs.USE_FUSED_RMS_QUANT and i_q is not None and i_s is not None:
            qcurr_hidden_states = i_q
            qa1_scale = i_s
        else:
            qcurr_hidden_states, qa1_scale = per_token_quant_int8(curr_hidden_states)

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_block_size(curr_topk_ids, config1['BLOCK_SIZE_M'],
                                    global_num_experts, expert_map))
        moe_gemm_marlin_w8a8(
            qcurr_hidden_states,
            w1,
            intermediate_cache1,
            qa1_scale,
            w1_scale,
            curr_topk_weights if apply_router_weight_on_input else None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k_num,
            config1)

        if activation == "silu":
            if use_lightop:
                qintermediate_cache2, qa2_scale = fuse_silu_mul_quant(intermediate_cache1.view(-1, N2))
            else:
                torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N2))
                qintermediate_cache2, qa2_scale = per_token_quant_int8(intermediate_cache2)

        elif activation == "gelu":
            torch.ops._C.gelu_and_mul(intermediate_cache2,
                                      intermediate_cache1.view(-1, N2))
            qintermediate_cache2, qa2_scale = per_token_quant_int8(intermediate_cache2)
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}")

        moe_gemm_marlin_w8a8(
            qintermediate_cache2,
            w2,
            intermediate_cache3,
            qa2_scale,
            w2_scale,
            curr_topk_weights if not apply_router_weight_on_input else None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            1,
            config2)

        if use_lightop and shared_output is not None:
            op.moe_sum(input=intermediate_cache3.view(*intermediate_cache3.shape),
                       output=out_hidden_states[begin_chunk_idx:end_chunk_idx],
                       bias=shared_output[begin_chunk_idx:end_chunk_idx],
                       expert_mask=None,
                       num_local_tokens=None,
                       factor=routed_scaling_factor)
        elif shared_output is not None:
            moe_reduce_dispatch(
                intermediate_cache3,
                out_hidden_states,
                begin_chunk_idx,
                end_chunk_idx,
                routed_scaling_factor,
                shared_output,
            )
        else:
            # TODO: remove this assertion when callers pass different values
            assert routed_scaling_factor == 1.0, "routed_scaling_factor != 1.0 not yet supported"
            moe_reduce_dispatch(
                intermediate_cache3,
                out_hidden_states,
                begin_chunk_idx,
                end_chunk_idx,
                routed_scaling_factor,
                None,
            )

    return out_hidden_states
def fused_experts_impl_int8_marlin_minimax_m2(
                       num_tokens,
                       K,
                       adevice,
                       adtype,
                       w1: torch.Tensor,
                       w2: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       inplace: bool = False,
                       activation: str = "silu",
                       apply_router_weight_on_input: bool = False,
                       use_fp8_w8a8: bool = False,
                       use_int8_w8a8: bool = False,
                       use_int8_w8a16: bool = False,
                       use_int4_w4a16: bool = False,
                       per_channel_quant: bool = False,
                       global_num_experts: int = -1,
                       expert_map: Optional[torch.Tensor] = None,
                       w1_scale: Optional[torch.Tensor] = None,
                       w2_scale: Optional[torch.Tensor] = None,
                       w1_zp: Optional[torch.Tensor] = None,
                       w2_zp: Optional[torch.Tensor] = None,
                       a1_scale: Optional[torch.Tensor] = None,
                       a2_scale: Optional[torch.Tensor] = None,
                       block_shape: Optional[List[int]] = None,
                       routed_scaling_factor: Optional[float] = 1.0,
                       shared_output: Optional[torch.Tensor] = None,
                       i_q: Optional[torch.Tensor] = None,
                       i_s: Optional[torch.Tensor] = None, **_):

    E = w1.shape[0]
    N = w2.shape[1] * 64
    N2 = N * 2
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.shape[1]
    CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens, CHUNK_SIZE)
    if envs.VLLM_USE_GLOBAL_CACHE13:
        from vllm.model_executor.layers.fused_moe.fused_moe import get_moe_cache
        cache13 = get_moe_cache(top_k_num, N2, K, device=adevice, dtype=adtype)
    else:
        cache13 = torch.empty(
            (M * top_k_num * max(2 * N, K), ),
            device=adevice,
            dtype=adtype,
        )
    intermediate_cache1 = cache13[:M * top_k_num * N2].view(M, top_k_num, N2)
    intermediate_cache3 = cache13[:M * top_k_num * K].view(M, top_k_num, K)

    intermediate_cache2 = torch.empty((M * top_k_num, N),
                                      device=adevice,
                                      dtype=adtype)

    if adtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif adtype == torch.float16:
        compute_type = tl.float16
    elif adtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {adtype}")

    out_hidden_states = torch.empty((num_tokens, K), device=adevice, dtype=adtype)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        qcurr_hidden_states = i_q[begin_chunk_idx:end_chunk_idx]
        qa1_scale = i_s[begin_chunk_idx:end_chunk_idx]

        tokens_in_chunk, _ = qcurr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[:tokens_in_chunk *
                                                      topk_ids.shape[1]]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
        try:
            config1, config2, status = get_moe_cuda_marlin_config(
                    E, tokens_in_chunk, N2, K, K, N, top_k_num, device_name, num_cus, compute_type
                    )
        except Exception as e:
            print(f"Warning: get_moe_cuda_config failed: {e}")
            status = False

        assert status, f'moe marlin unsupport this size E:{E}, N:{N}, K:{K}'

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_block_size(curr_topk_ids, config1['BLOCK_SIZE_M'],
                                    global_num_experts, expert_map))
        moe_gemm_marlin_w8a8(
            qcurr_hidden_states,
            w1,
            intermediate_cache1,
            qa1_scale,
            w1_scale,
            curr_topk_weights if apply_router_weight_on_input else None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k_num,
            config1)

        if activation == "silu":
            if use_lightop:
                qintermediate_cache2, qa2_scale = fuse_silu_mul_quant(intermediate_cache1.view(-1, N2))
            else:
                _silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N2))
                qintermediate_cache2, qa2_scale = per_token_quant_int8(intermediate_cache2)
        elif activation == "gelu":
            _gelu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N2))
            qintermediate_cache2, qa2_scale = per_token_quant_int8(intermediate_cache2)
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}")

        moe_gemm_marlin_w8a8(
            qintermediate_cache2,
            w2,
            intermediate_cache3,
            qa2_scale,
            w2_scale,
            curr_topk_weights if not apply_router_weight_on_input else None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            1,
            config2)

        if use_lightop and shared_output is not None:
            lightop.moe_sum(input=intermediate_cache3.view(*intermediate_cache3.shape),
                       output=out_hidden_states[begin_chunk_idx:end_chunk_idx],
                       bias=shared_output[begin_chunk_idx:end_chunk_idx],
                       expert_mask=None,
                       num_local_tokens=None,
                       factor=routed_scaling_factor)
        elif shared_output is not None:
            moe_reduce_dispatch(
                intermediate_cache3,
                out_hidden_states,
                begin_chunk_idx,
                end_chunk_idx,
                routed_scaling_factor,
                shared_output,
            )
        else:
            moe_reduce_dispatch(
                intermediate_cache3,
                out_hidden_states,
                begin_chunk_idx,
                end_chunk_idx,
                1.0,
                None,
            )

    return out_hidden_states
