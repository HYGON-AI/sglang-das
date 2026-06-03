# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/a6221a144af772fd1a68fe7e627935dc53e81738/vllm/model_executor/layers/fused_moe/fused_moe.py

"""Fused MoE kernel."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import triton
import torch.nn.functional as F
import triton.language as tl

from sglang.srt.environ import envs
from lightop import moe_gemm_marlin_w8a8_fp8, get_moe_cuda_marlin_config
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.utils import get_moe_padding_size
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_hip,
    is_musa,
    is_xpu,
    use_intel_xpu_backend,
)
from sglang.srt.utils.custom_op import register_custom_op
from sglang.srt.utils import direct_register_custom_op

from .fused_moe_triton_config import get_config_dtype_str, try_get_optimal_moe_config
from .fused_moe_triton_kernels import (
    act_and_mul_triton,
    invoke_fused_moe_kernel,
    moe_sum_reduce_triton,
    support_tensor_descriptor,
)
from .moe_align_block_size import moe_align_block_size
if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import StandardTopKOutput

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_xpu = is_xpu()
_use_sgl_xpu = use_intel_xpu_backend()
_is_musa = is_musa()
_use_lightop = get_bool_env_var("SGLANG_USE_LIGHTOP")
_use_aiter_moe = get_bool_env_var("SGLANG_ROCM_USE_AITER_MOE", default="true")


if _is_cuda:
    from sgl_kernel import moe_sum_reduce

    from sglang.jit_kernel.activation import gelu_and_mul, silu_and_mul
elif _is_cpu and _is_cpu_amx_available:
    pass
elif _is_hip:
    from sgl_kernel import gelu_and_mul, silu_and_mul
    print(f"YYYY: _use_lightop: {_use_lightop}")
    if _use_lightop:
        from lightop import op as ops
        from lightop import fuse_silu_and_mul, fuse_silu_mul_fp8_quant
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size as moe_align_block_size_lightop
    if _use_aiter:
        try:
            from aiter import moe_sum
        except ImportError:
            raise ImportError("aiter is required when SGLANG_USE_AITER is set to True")
    if _use_aiter_moe:
        try:
            import aiter
            from aiter.moe import (
                get_aiter_moe_config,
                aiter_moe,
                MoeSolutionType,
                MoeQuantType,
            )
        except ImportError:
            raise ImportError("aiter is required when SGLANG_ROCM_USE_AITER_MOE is set to True")
    # Note: vllm_ops is not needed for HIP when _use_aiter=False
    # because the code uses moe_sum_reduce_triton as fallback (line 619)
elif _is_xpu:
    from sgl_kernel import moe_sum_reduce, silu_and_mul
elif _is_musa:
    from sgl_kernel import moe_sum_reduce

    _silu_and_mul_musa = torch.nn.SwishGLU()

# Try to import vllm_ops for non-CUDA/HIP/XPU platforms
_has_vllm_ops = False
if not _is_cuda and not _is_hip and not _is_xpu:
    try:
        from vllm import _custom_ops as vllm_ops

        _has_vllm_ops = True
    except ImportError:
        # Fallback: vllm not available, will use native PyTorch implementations
        _has_vllm_ops = False
from vllm.platforms import current_platform
device_name = current_platform.get_device_name().replace(" ", "_")
num_cus= torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

padding_size = get_moe_padding_size(_use_aiter)


@register_custom_op(mutates_args=["hidden_states"])
def inplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    activation: int = 0,#0 silu 1 gelu
    is_gated: bool = True,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    filter_expert: bool = True,
    swiglu_limit: Optional[float] = None,
) -> None:
    if isinstance(activation, int):
        activation = "silu" if activation == 0 else "gelu"
    fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        b1,
        b2,
        True,
        activation,
        is_gated,
        apply_router_weight_on_input,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
        False,
        routed_scaling_factor,
        gemm1_alpha,
        gemm1_limit,
        filter_expert,
        swiglu_limit=swiglu_limit,
    )


def inplace_fused_experts_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    activation: int = 0,#0 silu 1 gelu
    is_gated: bool = True,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    filter_expert: bool = True,
) -> None:
    pass


direct_register_custom_op(
    op_name="inplace_fused_experts",
    op_func=inplace_fused_experts,
    mutates_args=["hidden_states"],
    fake_impl=inplace_fused_experts_fake,
)


@register_custom_op(out_shape="hidden_states")
def outplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    activation: int = 0,#0 silu 1 gelu
    is_gated: bool = True,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    filter_expert: bool = True,
    swiglu_limit: Optional[float] = None,
) -> torch.Tensor:
    if isinstance(activation, int):
        activation = "silu" if activation == 0 else "gelu"
    return fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        b1,
        b2,
        False,
        activation,
        is_gated,
        apply_router_weight_on_input,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
        no_combine=no_combine,
        routed_scaling_factor=routed_scaling_factor,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
        filter_expert=filter_expert,
        swiglu_limit=swiglu_limit,
    )


def outplace_fused_experts_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    activation: int = 0,#0 silu 1 gelu
    is_gated: bool = True,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    filter_expert: bool = True,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="outplace_fused_experts",
    op_func=outplace_fused_experts,
    mutates_args=[],
    fake_impl=outplace_fused_experts_fake,
)


def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_output: StandardTopKOutput,
    moe_runner_config: MoeRunnerConfig,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
):
    topk_weights, topk_ids, _ = topk_output
    filter_expert = (
        moe_runner_config.num_experts is None
        or moe_runner_config.num_experts != moe_runner_config.num_local_experts
    )
    act_id = (
        0 if (
            moe_runner_config.activation == 0
            or (isinstance(moe_runner_config.activation, str)
                and moe_runner_config.activation.lower() == "silu")
        ) else 1
    )
    if moe_runner_config.inplace:
        assert not moe_runner_config.no_combine, "no combine + inplace makes no sense"
        inplace_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            b1,
            b2,
            act_id,
            moe_runner_config.is_gated,
            moe_runner_config.apply_router_weight_on_input,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            w1_scale,
            w2_scale,
            w1_zp,
            w2_zp,
            a1_scale,
            a2_scale,
            block_shape,
            moe_runner_config.routed_scaling_factor,
            moe_runner_config.gemm1_alpha,
            moe_runner_config.gemm1_clamp_limit,
            filter_expert,
            swiglu_limit=moe_runner_config.swiglu_limit,
        )
        return hidden_states
    else:
        return outplace_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            b1,
            b2,
            act_id,
            moe_runner_config.is_gated,
            moe_runner_config.apply_router_weight_on_input,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            w1_scale,
            w2_scale,
            w1_zp,
            w2_zp,
            a1_scale,
            a2_scale,
            block_shape,
            no_combine=moe_runner_config.no_combine,
            routed_scaling_factor=moe_runner_config.routed_scaling_factor,
            gemm1_alpha=moe_runner_config.gemm1_alpha,
            gemm1_limit=moe_runner_config.gemm1_clamp_limit,
            filter_expert=filter_expert,
            swiglu_limit=moe_runner_config.swiglu_limit,
        )


@torch.compile
def moe_sum_reduce_torch_compile(x, out, routed_scaling_factor):
    torch.sum(x, dim=1, out=out)
    out.mul_(routed_scaling_factor)


@torch.compile
def _swiglu_silu_clamp_mul(x, gemm1_limit):
    gate, up = x.chunk(2, dim=-1)
    gate = F.silu(gate)
    gate = gate.clamp(min=None, max=gemm1_limit)
    up = up.clamp(min=-gemm1_limit, max=gemm1_limit)
    return gate * up


@torch.compile
def swiglu_gpt_oss_sigmoid_alpha(x, gemm1_alpha, gemm1_limit):
    # NOTE: This variant uses gemm1_alpha, unlike _swiglu_silu_clamp_mul.
    # At present, only GPT-OSS uses this variant.
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(min=None, max=gemm1_limit)
    up = up.clamp(min=-gemm1_limit, max=gemm1_limit)
    return gate * torch.sigmoid(gate * gemm1_alpha) * (up + 1)


@functools.lru_cache()
def _down_moe_use_tma():
    return support_tensor_descriptor()


def _shape_str(tensor: Optional[torch.Tensor]) -> str:
    return "None" if tensor is None else str(tuple(tensor.shape))


def fused_experts_impl_aiter(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: int = 0,#0 silu 1 gelu
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    routed_scaling_factor: Optional[float] = None,
    quant_type: Optional[MoeQuantType] = None,
):
    M, K = hidden_states.shape
    E, N1, _ = w1.shape
    _, N2, _ = w2.shape
    is_channelwise_w8a8 = quant_type == MoeQuantType.FP8_W8A8 and block_shape is None
    if not is_channelwise_w8a8 and (block_shape is None or len(block_shape) < 2):
        raise ValueError(
            "AITER MoE requires block_shape with two dimensions for this "
            "quantization mode, but got "
            f"{block_shape}. "
            f"M={M}, K={K}, E={E}, N1={N1}, N2={N2}, "
            f"top_k={topk_ids.shape[1]}, dtype={hidden_states.dtype}, "
            f"quant_type={quant_type}, "
            f"hidden_states_shape={_shape_str(hidden_states)}, "
            f"w1_shape={_shape_str(w1)}, w2_shape={_shape_str(w2)}, "
            f"w1_scale_shape={_shape_str(w1_scale)}, "
            f"w2_scale_shape={_shape_str(w2_scale)}, "
            f"a1_scale_shape={_shape_str(a1_scale)}, "
            f"a2_scale_shape={_shape_str(a2_scale)}"
        )
    block_size = 0 if is_channelwise_w8a8 else block_shape[1]
    status, moe_cfg = get_aiter_moe_config(
        M=M, E=E, N1=N1, N2=N2, K=K, top_k=topk_ids.shape[1], block_size=block_size, dtype=hidden_states.dtype, quant_type=quant_type,
    )
    if status:
        assert moe_cfg.solution_type is not None, \
            "status=True but solution_type is None"
        assert moe_cfg.config is not None, \
            "status=True but config is None"
        assert moe_cfg.solution_type in (
            MoeSolutionType.MOE_C,
            MoeSolutionType.ASM,
            MoeSolutionType.TRITON,
            MoeSolutionType.CK,
        ), f"Unexpected solution_type: {moe_cfg.solution_type}"
        assert moe_cfg.quant_type in (
            MoeQuantType.W4A16,
            MoeQuantType.FP8_W8A8,
        ), \
            f"Unexpected quant_type: {moe_cfg.quant_type}"
        dsv4_2604_submode = getattr(envs, "SGLANG_DSV4_2604_SUBMODE", None)
        is_2604b = (
            dsv4_2604_submode is not None
            and dsv4_2604_submode.get() == "2604B"
        )
        if is_2604b:
            deepseek_v4_moe_code_path_checker.observed += 1
        # print(
        #     f"[get_config_w4a16] M={M}, K={K}, N1={N1}, N2={N2}, E={E}, top_k={topk_ids.shape[1]}, block_size={block_shape[1]}, dtype={hidden_states.dtype} "
        #     f"solution={moe_cfg.solution_type}, "
        #     f"config keys={list(moe_cfg.config.keys())}"
        # )
    else:
        assert moe_cfg.solution_type is None, \
            "status=False but solution_type is not None"
        assert moe_cfg.config is None, \
            "status=False but config is not None"
        print(
            f"[get_config_aiter_moe] M={M}, K={K}, N1={N1}, N2={N2}, E={E}, top_k={topk_ids.shape[1]}, block_size={block_size}, dtype={hidden_states.dtype}, quant_type={quant_type} "
            f"no solution found (expected on unsupported configs)"
        )

    return aiter_moe(hidden_states, w1, w2, topk_weights, topk_ids, moe_cfg, inplace, activation, w1_scale, w2_scale, w1_zp, w2_zp, a1_scale, a2_scale, block_shape, E, None, routed_scaling_factor, output_dtype=hidden_states.dtype)

def _prepare_fused_moe_run(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: Optional[List[int]],
):
    """Resolve config, down_config, TMA flag, and aligned expert routing ids.

    Shared by ``fused_experts_impl`` and ``pre_permute_standard_to_triton`` so
    both paths compute alignment from the same source.
    """
    padded_size = padding_size
    if not (use_fp8_w8a8 or use_int8_w8a8) or block_shape is not None or _use_aiter:
        padded_size = 0

    num_tokens = hidden_states.shape[0]
    E = w1.shape[0]
    config_dtype = get_config_dtype_str(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        dtype=hidden_states.dtype,
    )

    config, (down_config, _) = try_get_optimal_moe_config(
        w1.shape,
        (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size),
        topk_ids.shape[1],
        config_dtype,
        num_tokens,
        block_shape=block_shape,
        per_channel_quant=per_channel_quant,
        return_down_config=True,
    )
    down_moe_use_tma = (
        _down_moe_use_tma()
        and down_config is not None
        and down_config.pop("USE_TMA", False)
    )

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], E
    )

    return (
        config,
        down_config,
        down_moe_use_tma,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    )


def _fused_moe_kernel_sequence(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: Dict[str, Any],
    down_config: Optional[Dict[str, Any]],
    down_moe_use_tma: bool,
    *,
    b1: Optional[torch.Tensor],
    b2: Optional[torch.Tensor],
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    w1_zp: Optional[torch.Tensor],
    w2_zp: Optional[torch.Tensor],
    a1_scale: Optional[torch.Tensor],
    a2_scale: Optional[torch.Tensor],
    block_shape: Optional[List[int]],
    activation: str,
    is_gated: bool,
    no_combine: bool,
    inplace: bool,
    apply_router_weight_on_input: bool,
    routed_scaling_factor: Optional[float],
    gemm1_alpha: Optional[float],
    gemm1_limit: Optional[float],
    filter_expert: bool,
    hooks: Optional[Any] = None,
    swiglu_limit: Optional[float] = None,
) -> torch.Tensor:
    """Run the MoE kernel/activation/kernel/combine sequence in a single shot.

    Inputs are already aligned and the block-size config is already resolved.
    Supports optional LoRA hooks that fire between the two kernels and before
    combine. Returns ``out_hidden_states``.
    """
    num_tokens = hidden_states.shape[0]
    E, N, _ = w1.shape
    topk = topk_ids.shape[1]
    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16

    padded_tokens = (
        min(num_tokens * topk, E + 1) * (config["BLOCK_SIZE_M"] - 1)
        if down_moe_use_tma
        else 0
    )
    total_tokens = num_tokens * topk + padded_tokens

    if no_combine:
        assert not inplace
        out_hidden_states = torch.empty(
            (num_tokens, topk, w2.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    elif inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    use_fused_moe_sum_all_reduce = (
        get_global_server_args().enable_fused_moe_sum_all_reduce
        and (not no_combine)
        and (topk > 2)
        and (not use_int8_w8a16)
        and (not use_int4_w4a16)
    )

    intermediate_cache1 = torch.empty(
        (total_tokens, N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    invoke_fused_moe_kernel(
        hidden_states,
        w1,
        b1,
        intermediate_cache1,
        a1_scale,
        w1_scale,
        w1_zp,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        apply_router_weight_on_input,
        topk,
        config,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
        c_sorted=down_moe_use_tma,
        filter_expert=filter_expert,
    )

    if hooks and hooks.after_gate_up:
        # Hooks expect intermediate_cache1 shaped (num_tokens, topk, N); the
        # underlying buffer is laid out as (total_tokens, N) where
        # total_tokens = num_tokens * topk (+ TMA padding). Slice off any
        # padding and reshape for the hook, which writes in-place on the view.
        hooks.after_gate_up(
            hidden_states,
            intermediate_cache1[: num_tokens * topk].view(num_tokens, topk, N),
            topk_weights,
            topk_ids,
        )

    intermediate_cache2 = torch.empty(
        (total_tokens, N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    # Activation function with multiplication
    if activation == "silu" and is_gated:
        # - gemm1_alpha != None: GPT-OSS-style swiglu(alpha, limit)
        # - gemm1_alpha == None and gemm1_limit != None: silu+clamp+mul(limit-only)
        # - swiglu_limit != None: DeepSeek V4 swiglu clamp + silu_and_mul (CUDA/HIP only)
        if gemm1_alpha is not None:
            assert gemm1_limit is not None
            intermediate_cache2 = swiglu_gpt_oss_sigmoid_alpha(
                intermediate_cache1.view(-1, N), gemm1_alpha, gemm1_limit
            )
        elif gemm1_limit is not None:
            intermediate_cache2 = _swiglu_silu_clamp_mul(
                intermediate_cache1.view(-1, N), gemm1_limit
            )
        elif swiglu_limit is not None:
            # DeepSeek V4: swiglu clamp before silu_and_mul.
            # Two paths gated by SGLANG_OPT_SWIGLU_CLAMP_FUSION:
            #   fusion=True: clamp fused into act_and_mul_triton or silu_and_mul_clamp
            #   fusion=False: explicit clamp_ on intermediate_cache1 (path checker)
            assert swiglu_limit == 10
            assert intermediate_cache1.shape == (total_tokens, N)
            assert _is_cuda or _is_hip, "DeepSeek V4 only supports CUDA/HIP downstream"

            swiglu_limit_for_triton: Optional[float] = None
            swiglu_limit_for_silu_and_mul_clamp: Optional[float] = None

            if envs.SGLANG_OPT_SWIGLU_CLAMP_FUSION.get():
                if filter_expert:
                    swiglu_limit_for_triton = swiglu_limit
                else:
                    assert (
                        _is_cuda
                    ), "fused silu_and_mul_clamp kernel is CUDA-only; HIP must disable SWIGLU_CLAMP_FUSION"
                    swiglu_limit_for_silu_and_mul_clamp = swiglu_limit
            else:
                half = N // 2
                intermediate_cache1[:, :half].clamp_(max=swiglu_limit)
                intermediate_cache1[:, half:].clamp_(
                    min=-swiglu_limit, max=swiglu_limit
                )

            if not filter_expert:
                if swiglu_limit_for_silu_and_mul_clamp is not None:
                    from sglang.jit_kernel.deepseek_v4 import silu_and_mul_clamp

                    silu_and_mul_clamp(
                        intermediate_cache1.view(-1, N),
                        intermediate_cache2,
                        swiglu_limit_for_silu_and_mul_clamp,
                    )
                else:
                    silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
            else:
                act_and_mul_triton(
                    intermediate_cache1.view(-1, N),
                    intermediate_cache2,
                    config,
                    topk_ids,
                    expert_ids,
                    down_moe_use_tma,
                    activation,
                    swiglu_limit=swiglu_limit_for_triton,
                )
        elif _is_cuda or _is_hip or _is_xpu:
            if filter_expert and _is_cuda:
                # HIP/XPU fall through to the unfiltered path: the down kernel
                # zeros filtered rows without reading their input.
                silu_and_mul(
                    intermediate_cache1.view(-1, N),
                    intermediate_cache2,
                    expert_ids=(expert_ids if down_moe_use_tma else topk_ids.view(-1)),
                    expert_step=(config["BLOCK_SIZE_M"] if down_moe_use_tma else 1),
                )
            else:
                silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
        elif _is_musa:
            intermediate_cache2 = _silu_and_mul_musa(intermediate_cache1.view(-1, N))
        else:
            if _has_vllm_ops:
                vllm_ops.silu_and_mul(
                    intermediate_cache2, intermediate_cache1.view(-1, N)
                )
            else:
                # Fallback: native PyTorch silu_and_mul
                x = intermediate_cache1.view(-1, N)
                d = x.shape[-1] // 2
                intermediate_cache2.copy_(F.silu(x[..., :d]) * x[..., d:])
    elif activation == "gelu" and is_gated:
        assert gemm1_alpha is None, "gemm1_alpha is not supported for gelu"
        assert gemm1_limit is None, "gemm1_limit is not supported for gelu"
        if _is_cuda or _is_hip:
            if filter_expert and _is_cuda:
                gelu_and_mul(
                    intermediate_cache1.view(-1, N),
                    intermediate_cache2,
                    expert_ids=(expert_ids if down_moe_use_tma else topk_ids.view(-1)),
                    expert_step=(config["BLOCK_SIZE_M"] if down_moe_use_tma else 1),
                )
            else:
                gelu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
        else:
            if _has_vllm_ops:
                vllm_ops.gelu_and_mul(
                    intermediate_cache2, intermediate_cache1.view(-1, N)
                )
            else:
                # Fallback: native PyTorch gelu_and_mul
                x = intermediate_cache1.view(-1, N)
                d = x.shape[-1] // 2
                intermediate_cache2.copy_(F.gelu(x[..., :d]) * x[..., d:])
    # Activation function without multiplication
    elif activation == "silu" and not is_gated:
        intermediate_cache2 = F.silu(intermediate_cache1.view(-1, N))
    elif activation == "gelu" and not is_gated:
        intermediate_cache2 = F.gelu(intermediate_cache1.view(-1, N))
    elif activation == "relu2" and not is_gated:
        intermediate_cache2 = torch.square(F.relu(intermediate_cache1.view(-1, N)))
    else:
        raise ValueError(f"Unsupported activation: {activation=}, with {is_gated=}")

    del intermediate_cache1

    intermediate_cache3 = torch.empty(
        (num_tokens, topk, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    # LoRA hooks force the second kernel to write to intermediate_cache3 so
    # hooks.after_down can inspect/modify it before reduction.
    _use_intermediate = not no_combine and (topk != 1 or hooks)

    out_slice = None
    if use_fused_moe_sum_all_reduce:
        out_slice = out_hidden_states
        out_slice.zero_()

    invoke_fused_moe_kernel(
        intermediate_cache2,
        w2,
        b2,
        (
            out_slice
            if use_fused_moe_sum_all_reduce
            else (
                intermediate_cache3
                if _use_intermediate
                else out_hidden_states.unsqueeze(0)
            )
        ),
        a2_scale,
        w2_scale,
        w2_zp,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        not apply_router_weight_on_input and not no_combine,
        1,
        down_config or config,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
        a_use_tma=down_moe_use_tma,
        b_use_tma=down_moe_use_tma,
        filter_expert=filter_expert,
        fuse_sum_all_reduce=use_fused_moe_sum_all_reduce,
        router_topk=topk,
    )

    if hooks and hooks.after_down:
        hooks.after_down(
            intermediate_cache2, intermediate_cache3, topk_weights, topk_ids
        )

    del intermediate_cache2

    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    if no_combine:
        pass
    elif _is_cuda or _is_musa:
        if use_fused_moe_sum_all_reduce:
            if routed_scaling_factor != 1.0:
                assert out_slice is not None
                out_slice.mul_(routed_scaling_factor)
        elif topk == 1 and routed_scaling_factor == 1.0 and not _use_intermediate:
            pass  # we wrote directly into out_hidden_states
        elif topk == 2 and routed_scaling_factor == 1.0:
            torch.add(
                intermediate_cache3[:, 0],
                intermediate_cache3[:, 1],
                out=out_hidden_states,
            ).squeeze(dim=1)
        else:
            # According to micro benchmark results, torch.compile can get better performance for small token.
            if num_tokens <= 32:
                moe_sum_reduce_torch_compile(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states,
                    routed_scaling_factor,
                )
            else:
                moe_sum_reduce(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states,
                    routed_scaling_factor,
                )
    elif _is_hip:
        if _use_aiter:
            moe_sum(
                intermediate_cache3.view(*intermediate_cache3.shape),
                out_hidden_states,
            )
        else:
            # According to micro benchmark results, torch.compile can get better performance for small token.
            if num_tokens <= 32:
                moe_sum_reduce_torch_compile(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states,
                    routed_scaling_factor,
                )
            else:
                moe_sum_reduce_triton(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states,
                    routed_scaling_factor,
                )
    elif _is_xpu:
        moe_sum_reduce(
            intermediate_cache3.view(*intermediate_cache3.shape),
            out_hidden_states,
            routed_scaling_factor,
        )
    else:
        if _has_vllm_ops:
            vllm_ops.moe_sum(
                intermediate_cache3.view(*intermediate_cache3.shape),
                out_hidden_states,
            )
        else:
            # Fallback: use triton moe_sum_reduce when vllm is not available
            moe_sum_reduce_triton(
                intermediate_cache3.view(*intermediate_cache3.shape),
                out_hidden_states,
                routed_scaling_factor,
            )

    del intermediate_cache3

    return out_hidden_states


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    inplace: bool = False,
    activation: int = 0,#0 silu 1 gelu
    is_gated: bool = True,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    filter_expert: bool = True,
    swiglu_limit: Optional[float] = None,
):
    if _use_aiter_moe and (use_int4_w4a16 or use_int8_w8a8 or use_fp8_w8a8) and hidden_states.dtype == torch.bfloat16:
        if use_int4_w4a16:
            quant_type = MoeQuantType.W4A16
        else:
            quant_type = MoeQuantType.FP8_W8A8
        return fused_experts_impl_aiter(hidden_states, w1, w2, topk_weights, topk_ids, inplace, activation, w1_scale, w2_scale, w1_zp, w2_zp, a1_scale, a2_scale, block_shape, routed_scaling_factor, quant_type)

    from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import dcu_moe_align_block_size
    if isinstance(activation, int):
        activation = "silu" if activation == 0 else "gelu"
    padded_size = padding_size
    if not (use_fp8_w8a8 or use_int8_w8a8) or block_shape is not None or _use_aiter:
        padded_size = 0

    # Check constraints.
    if use_int4_w4a16:
        assert hidden_states.shape[1] // 2 == w1.shape[2], "Hidden size mismatch"
    else:
        assert (
            hidden_states.shape[1] == w1.shape[2] - padded_size
        ), f"Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    (
        config,
        down_config,
        down_moe_use_tma,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    ) = _prepare_fused_moe_run(
        hidden_states,
        w1,
        w2,
        topk_ids,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
    )

    return _fused_moe_kernel_sequence(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        config,
        down_config,
        down_moe_use_tma,
        b1=b1,
        b2=b2,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        activation=activation,
        is_gated=is_gated,
        no_combine=no_combine,
        inplace=inplace,
        apply_router_weight_on_input=apply_router_weight_on_input,
        routed_scaling_factor=routed_scaling_factor,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
        filter_expert=filter_expert,
        hooks=None,
        swiglu_limit=swiglu_limit,
    )
    topk = topk_ids.shape[1]
    max_padded_tokens = (
        min(M * topk, E + 1) * (max_block_m - 1) if down_moe_use_tma else 0
    )
    total_tokens = M * topk + max_padded_tokens
    cache = torch.empty(
        total_tokens * max(N, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = cache[: M * topk * w2.shape[1]].view(
        (M, topk, w2.shape[1]),
    )

    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16

    if no_combine:
        assert not inplace
        out_hidden_states = torch.empty(
            (num_tokens, topk, w2.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    elif inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            config, (down_config, _) = get_config_func(tokens_in_chunk)
            down_moe_use_tma = (
                _down_moe_use_tma()
                and down_config is not None
                and down_config.pop("USE_TMA", False)
            )
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]

        padded_tokens = (
            min(tokens_in_chunk * topk, E + 1) * (config["BLOCK_SIZE_M"] - 1)
            if down_moe_use_tma
            else 0
        )
        total_tokens = tokens_in_chunk * topk + padded_tokens
        intermediate_cache1 = cache[: total_tokens * N].view(
            (total_tokens, N),
        )
        intermediate_cache2 = torch.empty(
            (total_tokens, N // 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        # use_fused_moe_sum_all_reduce = (
        #     get_global_server_args().enable_fused_moe_sum_all_reduce
        #     and (not no_combine)
        #     and (curr_topk_ids.shape[1] > 2)
        #     and (not use_int8_w8a16)
        #     and (not use_int4_w4a16)
        # )
        if _use_lightop:
            sorted_token_ids, expert_ids, num_tokens_post_padded = dcu_moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], E
            )
        else:
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], E
            )

        invoke_fused_moe_kernel(
            curr_hidden_states,
            w1,
            b1,
            intermediate_cache1,
            a1_scale,
            w1_scale,
            w1_zp,
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            topk_ids.shape[1],
            config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            c_sorted=down_moe_use_tma,
            filter_expert=filter_expert,
        )

        # Activation function with multiplication
        # print(f"activation: {activation} is_gated: {is_gated} gemm1_alpha: {gemm1_alpha} gemm1_limit: {gemm1_limit} filter_expert: {filter_expert} swiglu_limit: {swiglu_limit} _is_cuda or _is_hip or _is_xpu: {_is_cuda or _is_hip or _is_xpu}")
        if activation == "silu" and is_gated:
            # - gemm1_alpha != None: GPT-OSS-style swiglu(alpha, limit)
            # - gemm1_alpha == None and gemm1_limit != None: silu+clamp+mul(limit-only)
            if gemm1_alpha is not None:
                assert gemm1_limit is not None
                intermediate_cache2 = _swiglu_gpt_oss_sigmoid_alpha(
                    intermediate_cache1.view(-1, N), gemm1_alpha, gemm1_limit
                )
            elif gemm1_limit is not None:
                intermediate_cache2 = _swiglu_silu_clamp_mul(
                    intermediate_cache1.view(-1, N), gemm1_limit
                )
            # elif _use_lightop:
            elif False: #暂时关闭，开启后性能会很差
                fuse_silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
            # elif _is_cuda or _is_hip or _is_xpu:
            #     if not filter_expert:
            #         silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
            #     else:
            #         act_and_mul_triton(
            #             intermediate_cache1.view(-1, N),
            #             intermediate_cache2,
            #             config,
            #             curr_topk_ids,
            #             expert_ids,
            #             down_moe_use_tma,
            #             activation,
            #             swiglu_limit,
            #         )
            else:
                if _has_vllm_ops:
                    vllm_ops.silu_and_mul(
                        intermediate_cache2, intermediate_cache1.view(-1, N)
                    )
                else:
                    # Fallback: native PyTorch silu_and_mul
                    x = intermediate_cache1.view(-1, N)
                    d = x.shape[-1] // 2
                    intermediate_cache2.copy_(F.silu(x[..., :d]) * x[..., d:])
        elif activation == "gelu" and is_gated:
            assert gemm1_alpha is None, "gemm1_alpha is not supported for gelu"
            assert gemm1_limit is None, "gemm1_limit is not supported for gelu"
            if _is_cuda or _is_hip:
                if not filter_expert:
                    gelu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
                else:
                    act_and_mul_triton(
                        intermediate_cache1.view(-1, N),
                        intermediate_cache2,
                        config,
                        curr_topk_ids,
                        expert_ids,
                        down_moe_use_tma,
                        activation,
                    )
            else:
                if _has_vllm_ops:
                    vllm_ops.gelu_and_mul(
                        intermediate_cache2, intermediate_cache1.view(-1, N)
                    )
                else:
                    # Fallback: native PyTorch gelu_and_mul
                    x = intermediate_cache1.view(-1, N)
                    d = x.shape[-1] // 2
                    intermediate_cache2.copy_(F.gelu(x[..., :d]) * x[..., d:])
        # Activation function without multiplication
        elif activation == "silu" and not is_gated:
            intermediate_cache2 = F.silu(intermediate_cache1.view(-1, N))
        elif activation == "gelu" and not is_gated:
            intermediate_cache2 = F.gelu(intermediate_cache1.view(-1, N))
        elif activation == "relu2" and not is_gated:
            intermediate_cache2 = torch.square(F.relu(intermediate_cache1.view(-1, N)))
        else:
            raise ValueError(f"Unsupported activation: {activation=}, with {is_gated=}")

        out_slice = None
        if use_fused_moe_sum_all_reduce:
            out_slice = out_hidden_states[begin_chunk_idx:end_chunk_idx]
            out_slice.zero_()

        invoke_fused_moe_kernel(
            intermediate_cache2,
            w2,
            b2,
            (
                out_slice
                if use_fused_moe_sum_all_reduce
                else (
                    intermediate_cache3
                    if not no_combine and topk_ids.shape[1] != 1
                    else out_hidden_states[begin_chunk_idx:end_chunk_idx].unsqueeze(0)
                )
            ),
            a2_scale,
            w2_scale,
            w2_zp,
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            down_config or config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            a_use_tma=down_moe_use_tma,
            b_use_tma=down_moe_use_tma,
            filter_expert=filter_expert,
            fuse_sum_all_reduce=use_fused_moe_sum_all_reduce,
            router_topk=curr_topk_ids.shape[1],
        )

        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0

        if no_combine:
            pass
        elif _is_cuda:
            if use_fused_moe_sum_all_reduce:
                if routed_scaling_factor is None:
                    routed_scaling_factor = 1.0
                if routed_scaling_factor != 1.0:
                    assert out_slice is not None
                    out_slice.mul_(routed_scaling_factor)
            elif topk_ids.shape[1] == 1 and routed_scaling_factor == 1.0:
                pass  # we write directly into out_hidden_states
            elif topk_ids.shape[1] == 2 and routed_scaling_factor == 1.0:
                torch.add(
                    intermediate_cache3[:, 0],
                    intermediate_cache3[:, 1],
                    out=out_hidden_states[begin_chunk_idx:end_chunk_idx],
                ).squeeze(dim=1)
            else:
                # According to micro benchmark results, torch.compile can get better performance for small token.
                if tokens_in_chunk <= 32:
                    moe_sum_reduce_torch_compile(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        routed_scaling_factor,
                    )
                else:
                    moe_sum_reduce(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        routed_scaling_factor,
                    )

        elif _is_hip:
            if _use_aiter:
                moe_sum(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states[begin_chunk_idx:end_chunk_idx],
                )
            else:
                # According to micro benchmark results, torch.compile can get better performance for small token.
                if _use_lightop:
                        ops.moe_sum(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        factor=routed_scaling_factor,
                        expect_m=-1,
                    )
                elif tokens_in_chunk <= 32:
                    moe_sum_reduce_torch_compile(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        routed_scaling_factor,
                    )
                else:
                    moe_sum_reduce_triton(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        routed_scaling_factor,
                    )
        elif _is_xpu:
            moe_sum_reduce(
                intermediate_cache3.view(*intermediate_cache3.shape),
                out_hidden_states[begin_chunk_idx:end_chunk_idx],
                routed_scaling_factor,
            )
        else:
            if _has_vllm_ops:
                vllm_ops.moe_sum(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states[begin_chunk_idx:end_chunk_idx],
                )
            else:
                # Fallback: use triton moe_sum_reduce when vllm is not available
                moe_sum_reduce_triton(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states[begin_chunk_idx:end_chunk_idx],
                    routed_scaling_factor,
                )

    return out_hidden_states


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_output: StandardTopKOutput,
    moe_runner_config: MoeRunnerConfig = MoeRunnerConfig(),
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - topk_output (StandardTopKOutput): The top-k output of the experts.
    - moe_runner_config (MoeRunnerConfig): The configuration for the MoE runner.
    - b1 (Optional[torch.Tensor]): Optional bias for w1.
    - b2 (Optional[torch.Tensor]): Optional bias for w2.
    - use_fp8_w8a8 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int8_w8a8 (bool): If True, use int8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int8_w8a16 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int4_w4a16 (bool): If True, use matmul of int4 weight and bf16/fp16
        activation to compute the inner products for w1 and w2.
        Defaults to False.
    - w1_scale (Optional[torch.Tensor]): Optional scale to be used for
        w1.
    - w2_scale (Optional[torch.Tensor]): Optional scale to be used for
        w2.
    - a1_scale (Optional[torch.Tensor]): Optional scale to be used for
        a1.
    - a2_scale (Optional[torch.Tensor]): Optional scale to be used for
        a2.
    - block_shape: (Optional[List[int]]): Optional block size for block-wise
        quantization.
    - gemm1_alpha (Optional[float]): Optional gemm1_alpha for the activation
        function.
    - gemm1_limit (Optional[float]): Optional gemm1_limit for the swiglu activation
        function.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    if _use_sgl_xpu:
        topk_weight, topk_ids, _ = topk_output
        from sgl_kernel import fused_experts as sgl_fused_experts

        return sgl_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            b1=b1,
            b2=b2,
            use_fp8_w8a8=use_fp8_w8a8,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=w1_zp,
            w2_zp=w2_zp,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=block_shape,
        )

    return fused_experts(
        hidden_states,
        w1,
        w2,
        topk_output,
        moe_runner_config=moe_runner_config,
        b1=b1,
        b2=b2,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
    )

@triton.jit
def _per_token_quant_fp8(
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    stride_xq,
    N,
    BLOCK: tl.constexpr,
    fp8_min,
    fp8_max
):
    row_id = tl.program_id(0)

    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask,
                other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale_x = absmax / fp8_max
    x_q = tl.clamp(x / scale_x, min=fp8_min, max=fp8_max).to(xq_ptr.dtype.element_ty)
    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x)


def per_token_quant_fp8(x):
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    x_q = torch.empty_like(x, device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(x.shape[:-1] + (1, ),
                         device=x.device,
                         dtype=torch.float32)
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    finfo = torch.finfo(x_q.dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max
    #assert x.is_contiguous()
    _per_token_quant_fp8[(M, )](
        x,
        x_q,
        scales,
        stride_x=x.stride(-2),
        stride_xq=x_q.stride(-2),
        N=N,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
        fp8_min=fp8_min,
        fp8_max=fp8_max
    )
    return x_q, scales

def fused_moe_fp8_w8a8(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    global_num_experts: int = -1,
    origin_w1_shape: tuple = None,
    origin_w2_shape: tuple = None,
    expert_map: Optional[torch.Tensor] = None,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    num_bits: int = 8,
    inplace: bool = False,
    routed_scaling_factor: Optional[float] = None,
    bias: Optional[torch.Tensor] = None,
    hidden_states_fp8_input: Optional[torch.Tensor] = None,
    hidden_states_scale_fp8_input: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - w1_scale (torch.Tensor): Scale to be used for w1.
    - w2_scale (torch.Tensor): Scale to be used for w2.
    - g_idx1 (Optional[torch.Tensor]): The first set of act_order indices.
    - g_idx2 (Optional[torch.Tensor]): The second set of act_order indices.
    - sort_indices1 (Optional[torch.Tensor]): The first act_order input
        permutation.
    - sort_indices2 (Optional[torch.Tensor]): The second act_order input
        permutation.
    - topk_weights (torch.Tensor): Top-k weights.
    - topk_ids (torch.Tensor): Indices of topk-k elements.
    - w1_zeros (Optional[torch.Tensor]): Optional zero points to be used for w1.
    - w2_zeros (Optional[torch.Tensor]): Optional zero points to be used for w2.
    - num_bits (int): The number of bits in expert weights quantization.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    # from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import dcu_moe_align_block_size
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import dcu_moe_align_block_size

    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    # assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    if hidden_states_fp8_input is not None and hidden_states_scale_fp8_input is not None:
        hidden_states_fp8 = hidden_states_fp8_input
        hidden_states_scale_fp8 = hidden_states_scale_fp8_input
        if hidden_states_scale_fp8.dim() == hidden_states_fp8.dim() - 1:
            hidden_states_scale_fp8 = hidden_states_scale_fp8.unsqueeze(-1)
    else:
        hidden_states_fp8, hidden_states_scale_fp8 = per_token_quant_fp8(hidden_states)
    E = w1.shape[0]
    m, k = hidden_states.shape
    n1 = origin_w1_shape[1]
    k1 = origin_w1_shape[2]
    n2 = origin_w2_shape[1]
    k2 = origin_w2_shape[2]

    topk = topk_ids.shape[1]

    cuda_config1, cuda_config2, status = get_moe_cuda_marlin_config(
        E,
        m,
        n1,
        k1,
        n2,
        k2,
        topk,
        device_name,
        num_cus,
        hidden_states.dtype,
    )

    if "BLOCK_SIZE_M" in cuda_config1:
        block_size_m = cuda_config1["BLOCK_SIZE_M"]

        if global_num_experts == -1:
            global_num_experts = E

        sorted_token_ids, expert_ids, num_tokens_post_padded = dcu_moe_align_block_size(
            topk_ids, block_size_m, global_num_experts)

        # TODO: tune this further for specific models
        # intermediate_cache2 = torch.empty(
        #     (m * topk_ids.shape[1], n1 // 2),
        #     device=hidden_states.device,
        #     dtype=hidden_states.dtype,
        # )
        intermediate_cache13 = torch.empty(
            (m * topk_ids.shape[1] * max(n1, k),),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        intermediate_cache1 = intermediate_cache13[:m * topk_ids.shape[1] * n1]
        intermediate_cache1 = intermediate_cache1.view(-1, n1)
        intermediate_cache3 = intermediate_cache13[:m * topk_ids.shape[1] * k]
        intermediate_cache3 = intermediate_cache3.view(-1, k)

        intermediate_cache1 = moe_gemm_marlin_w8a8_fp8(
            hidden_states_fp8,
            w1,
            intermediate_cache1,
            hidden_states_scale_fp8,
            w1_scale,
            None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk,
            cuda_config1,
        )
        from lightop import fuse_silu_mul_fp8_quant

        fp8_cache2, fp8_cache2_scale = fuse_silu_mul_fp8_quant(intermediate_cache1, fp8type=0)

        intermediate_cache3 = moe_gemm_marlin_w8a8_fp8(
            fp8_cache2,
            w2,
            intermediate_cache3,
            fp8_cache2_scale,
            w2_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            1,
            cuda_config2,
        ).view(-1, topk, k)
        output = hidden_states if inplace else torch.empty_like(hidden_states)

        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0
        from lightop import op as ops # 报错缺少ops

        ops.moe_sum(
            intermediate_cache3,
            output,
            bias=bias,
            expert_mask=None,
            num_local_tokens=None,
            factor=routed_scaling_factor,
            expect_m=-1,
        )
        return output
    else:
        raise RuntimeError("No MoE implementation available. Please set: export SGLANG_ROCM_USE_AITER_MOE=true and export SGLANG_USE_FP8_W8A8_MOE=0")
