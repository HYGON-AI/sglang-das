# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Mega-MoE forward path and expert-weight prep shared by Deepseek V2/V4."""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.jit_kernel.deepseek_v4 import mega_moe_pre_dispatch
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.dp_attention import get_dp_global_num_tokens
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.utils import is_hcu

if TYPE_CHECKING:
    from deep_gemm import SymmBuffer

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE


_MEGA_MOE_SYMM_BUFFER: dict = {}
_MEGA_MOE_DG_ENV_APPLIED = False
_MEGA_MOE_HCU_W8A8_FUSED_PRE_DISPATCH: Optional[Any] = None
_MEGA_MOE_HCU_W8A8_FUSED_PRE_DISPATCH_CHECKED = False
_IS_HCU = is_hcu()

_HCU_MEGA_MOE_RUNTIME_DEEP_GEMM = "deep_gemm"
_HCU_MEGA_MOE_RUNTIME_MEGAMOE = "megamoe"
_HCU_MEGA_MOE_RUNTIMES = {
    _HCU_MEGA_MOE_RUNTIME_DEEP_GEMM,
    _HCU_MEGA_MOE_RUNTIME_MEGAMOE,
}

_MEGA_MOE_HCU_BACKEND_ENV = "MEGAMOE_HCU_BACKEND"
_MEGA_MOE_HCU_BACKEND_AUTO = "auto"
_MEGA_MOE_HCU_BACKEND_LL = "ll"
_MEGA_MOE_HCU_BACKEND_NORMAL = "normal"
_MEGA_MOE_HCU_NORMAL_LL_TOKEN_THRESHOLD_ENV = "MEGAMOE_HCU_NORMAL_LL_TOKEN_THRESHOLD"
_MEGA_MOE_HCU_NORMAL_LL_TOKEN_THRESHOLD = 512

logger = logging.getLogger(__name__)


def get_hcu_mega_moe_runtime() -> str:
    runtime = envs.SGLANG_HCU_MEGA_MOE_RUNTIME.get().strip().lower()
    if runtime not in _HCU_MEGA_MOE_RUNTIMES:
        raise ValueError(
            "SGLANG_HCU_MEGA_MOE_RUNTIME must be one of "
            f"{sorted(_HCU_MEGA_MOE_RUNTIMES)}, got {runtime!r}"
        )
    return runtime


def _is_standalone_megamoe_runtime() -> bool:
    return _IS_HCU and get_hcu_mega_moe_runtime() == _HCU_MEGA_MOE_RUNTIME_MEGAMOE


def _is_pd_prefill_instance() -> bool:
    from sglang.srt.server_args import get_global_server_args

    try:
        return get_global_server_args().disaggregation_mode == "prefill"
    except ValueError:
        return False

def _is_pd_decode_instance() -> bool:
    from sglang.srt.server_args import get_global_server_args

    try:
        return get_global_server_args().disaggregation_mode == "decode"
    except ValueError:
        return False

def _get_hcu_normal_ll_token_threshold() -> int:
    value = os.environ.get(_MEGA_MOE_HCU_NORMAL_LL_TOKEN_THRESHOLD_ENV)
    if value is None:
        return _MEGA_MOE_HCU_NORMAL_LL_TOKEN_THRESHOLD
    try:
        threshold = int(value)
    except ValueError:
        logger.warning(
            "Invalid %s=%s, fallback to %s",
            _MEGA_MOE_HCU_NORMAL_LL_TOKEN_THRESHOLD_ENV,
            value,
            _MEGA_MOE_HCU_NORMAL_LL_TOKEN_THRESHOLD,
        )
        return _MEGA_MOE_HCU_NORMAL_LL_TOKEN_THRESHOLD
    if threshold < 0:
        logger.warning(
            "Invalid %s=%s, fallback to %s",
            _MEGA_MOE_HCU_NORMAL_LL_TOKEN_THRESHOLD_ENV,
            value,
            _MEGA_MOE_HCU_NORMAL_LL_TOKEN_THRESHOLD,
        )
        return _MEGA_MOE_HCU_NORMAL_LL_TOKEN_THRESHOLD
    return threshold


def _select_hcu_megamoe_backend(selector_tokens: int) -> str:
    if selector_tokens < 0:
        raise ValueError("MegaMoE backend selector token count must be non-negative")
    if _is_pd_prefill_instance():
        return _MEGA_MOE_HCU_BACKEND_NORMAL
    if _is_pd_decode_instance():
        return _MEGA_MOE_HCU_BACKEND_LL

    mode = os.environ.get(_MEGA_MOE_HCU_BACKEND_ENV, _MEGA_MOE_HCU_BACKEND_AUTO)
    mode = mode.strip().lower()
    if mode == _MEGA_MOE_HCU_BACKEND_AUTO:
        return (
            _MEGA_MOE_HCU_BACKEND_LL
            if selector_tokens <= _get_hcu_normal_ll_token_threshold()
            else _MEGA_MOE_HCU_BACKEND_NORMAL
        )
    if mode in {_MEGA_MOE_HCU_BACKEND_LL, _MEGA_MOE_HCU_BACKEND_NORMAL}:
        return mode
    raise ValueError(
        f"{_MEGA_MOE_HCU_BACKEND_ENV} must be one of "
        f"{[_MEGA_MOE_HCU_BACKEND_AUTO, _MEGA_MOE_HCU_BACKEND_LL, _MEGA_MOE_HCU_BACKEND_NORMAL]}, "
        f"got {mode!r}"
    )


def _get_hcu_cuda_graph_max_tokens_per_rank(
    num_max_tokens_per_rank: int,
    selector_tokens: int,
) -> int:
    graph_tokens = max(
        _get_hcu_normal_ll_token_threshold(),
        int(selector_tokens),
        1,
    )
    if graph_tokens > num_max_tokens_per_rank:
        raise ValueError(
            "Standalone HCU MegaMoE CUDA graph requires "
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK to be at "
            f"least {graph_tokens}, got {num_max_tokens_per_rank}"
        )
    return graph_tokens


def _get_hcu_w8a8_fused_pre_dispatch():
    global _MEGA_MOE_HCU_W8A8_FUSED_PRE_DISPATCH
    global _MEGA_MOE_HCU_W8A8_FUSED_PRE_DISPATCH_CHECKED

    if _MEGA_MOE_HCU_W8A8_FUSED_PRE_DISPATCH_CHECKED:
        return _MEGA_MOE_HCU_W8A8_FUSED_PRE_DISPATCH

    try:
        import megamoe
    except Exception as exc:
        raise RuntimeError(
            "HCU W8A8 MegaMoE requires megamoe.mega_moe_pre_dispatch"
        ) from exc

    fused_pre_dispatch = getattr(megamoe, "mega_moe_pre_dispatch", None)
    if fused_pre_dispatch is None:
        raise RuntimeError(
            "HCU W8A8 MegaMoE requires megamoe.mega_moe_pre_dispatch; "
            "rebuild the megamoe package"
         )
    _MEGA_MOE_HCU_W8A8_FUSED_PRE_DISPATCH = fused_pre_dispatch
    _MEGA_MOE_HCU_W8A8_FUSED_PRE_DISPATCH_CHECKED = True
    return _MEGA_MOE_HCU_W8A8_FUSED_PRE_DISPATCH


def _prepare_standalone_megamoe_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    buf,
    num_tokens: int,
) -> None:
    fused_pre_dispatch = _get_hcu_w8a8_fused_pre_dispatch()
    quant_input = (
        hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
    )
    topk_ids_in = topk_ids if topk_ids.is_contiguous() else topk_ids.contiguous()
    if topk_weights.dtype not in (torch.float32, torch.bfloat16):
        topk_weights_in = topk_weights.to(buf.topk_weights.dtype)
    else:
        topk_weights_in = (
            topk_weights if topk_weights.is_contiguous() else topk_weights.contiguous()
        )
    fused_pre_dispatch(
        quant_input,
        topk_ids_in,
        topk_weights_in,
        buf.x,
        buf.x_sf,
        buf.topk_idx,
        buf.topk_weights,
        num_tokens=num_tokens,
    )

def _get_hcu_graph_capture_capacity_tokens(
    num_tokens: int,
    buf,
) -> int:
    capacity_tokens = max(int(num_tokens), 1)
    max_tokens = getattr(buf, "cuda_graph_max_tokens_per_rank", None)
    if max_tokens is not None and capacity_tokens > int(max_tokens):
        raise ValueError(
            "HCU MegaMoE CUDA graph capture capacity must be <= "
            f"cuda_graph_max_tokens_per_rank ({int(max_tokens)}), "
            f"got {capacity_tokens}"
        )
    return capacity_tokens

def _is_hcu_v4_pro_ll_masked_k1_shape(experts, w13, w2) -> bool:
    return (
        getattr(experts, "num_experts", None) == 384
        and getattr(experts, "top_k", None) == 6
        and getattr(experts, "num_fused_shared_experts", 0) == 0
        and getattr(experts, "hidden_size", None) == 7168
        and getattr(experts, "intermediate_size_per_partition", None) == 3072
        and w13.dim() == 3
        and w2.dim() == 3
        and tuple(w13.shape[1:]) == (6144, 7168)
        and tuple(w2.shape[1:]) == (7168, 3072)
    )

def set_mega_moe_cuda_graph_num_tokens(num_tokens: int) -> None:
    if not _is_standalone_megamoe_runtime():
        return
    for buf in _MEGA_MOE_SYMM_BUFFER.values():
        graph_num_tokens = getattr(buf, "cuda_graph_num_tokens", None)
        if graph_num_tokens is not None:
            graph_num_tokens.fill_(num_tokens)

def _apply_mega_moe_dg_env() -> None:
    """Forward sglang's FP4/MXF4 opt-in flags to DeepGEMM via env vars.

    DeepGEMM reads `DG_USE_FP4_ACTS` (and `DG_USE_MXF4_KIND`) at host-function
    call time — both `get_symm_buffer_for_mega_moe` and `fp8_fp4_mega_moe`.
    Forwarding once at first use is sufficient (these are static config
    flags, not per-request state) and matches the `setdefault` pattern so
    explicit `DG_USE_*` overrides from outside still win.
    """
    global _MEGA_MOE_DG_ENV_APPLIED
    if _MEGA_MOE_DG_ENV_APPLIED:
        return
    if envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS.get():
        os.environ.setdefault("DG_USE_FP4_ACTS", "1")
    if envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND.get():
        os.environ.setdefault("DG_USE_MXF4_KIND", "1")
    _MEGA_MOE_DG_ENV_APPLIED = True


def _get_mega_moe_symm_buffer(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    *,
    runtime: str,
    cuda_graph_max_tokens_per_rank: Optional[int] = None,
) -> SymmBuffer:
    if _IS_HCU and runtime == _HCU_MEGA_MOE_RUNTIME_MEGAMOE:
        import megamoe

        package_key = _HCU_MEGA_MOE_RUNTIME_MEGAMOE
        factory = megamoe.get_symm_buffer_for_mega_moe
    else:
        import deep_gemm

        _apply_mega_moe_dg_env()
        package_key = _HCU_MEGA_MOE_RUNTIME_DEEP_GEMM
        factory = deep_gemm.get_symm_buffer_for_mega_moe
        cuda_graph_max_tokens_per_rank = None

    key = (
        package_key,
        id(group),
        num_max_tokens_per_rank,
        cuda_graph_max_tokens_per_rank,
        num_experts,
        num_topk,
        hidden,
        intermediate_hidden,
    )
    buf = _MEGA_MOE_SYMM_BUFFER.get(key)
    if buf is None:
        kwargs = {}
        if cuda_graph_max_tokens_per_rank is not None:
            kwargs["cuda_graph_max_tokens_per_rank"] = cuda_graph_max_tokens_per_rank
        buf = factory(
            group,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            use_fp8_dispatch=True,
            activation="swiglu",
            **kwargs,
        )
        _MEGA_MOE_SYMM_BUFFER[key] = buf
    return buf


def should_use_mega_moe(moe: "DeepseekV2MoE", hidden_states: torch.Tensor) -> bool:
    if not get_moe_a2a_backend().is_megamoe():
        return False
    if not getattr(moe.experts, "_mega_moe_weights_built", False):
        return False
    if _IS_HCU:
        runtime = get_hcu_mega_moe_runtime()
        built_runtime = getattr(moe.experts, "_mega_moe_hcu_runtime", None)
        if built_runtime != runtime:
            raise RuntimeError(
                "HCU MegaMoE runtime changed after expert weights were built: "
                f"built={built_runtime!r}, current={runtime!r}. Restart the "
                "server after changing SGLANG_HCU_MEGA_MOE_RUNTIME."
            )
    if get_is_capture_mode():
        return not _IS_HCU or _is_standalone_megamoe_runtime()

    global_num_tokens = get_dp_global_num_tokens()
    if global_num_tokens:
        max_tokens_per_rank = max(global_num_tokens)
    else:
        max_tokens_per_rank = hidden_states.shape[0]
    cap = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    return max_tokens_per_rank <= cap


def forward_mega_moe(
    moe: "DeepseekV2MoE",
    hidden_states: torch.Tensor,
    forward_batch: Optional["ForwardBatch"] = None,
    input_ids_global: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]

    sbo_overlap_flag = (
        moe.alt_stream is not None
        and moe.num_fused_shared_experts == 0
        and num_tokens > 0
        and get_is_capture_mode()
    )

    if sbo_overlap_flag:
        current_stream = torch.cuda.current_stream()
        moe.alt_stream.wait_stream(current_stream)
        shared_output = moe._forward_shared_experts(hidden_states)
        mega_stream_ctx = torch.cuda.stream(moe.alt_stream)
    else:
        shared_output = moe._forward_shared_experts(hidden_states)
        mega_stream_ctx = nullcontext()

    with mega_stream_ctx:
        y = _run_mega_routed(
            moe, hidden_states, forward_batch, input_ids_global, num_tokens
        )

    if sbo_overlap_flag:
        current_stream.wait_stream(moe.alt_stream)

    if shared_output is not None:
        y.add_(shared_output)
    return y


def _run_mega_routed(
    moe: "DeepseekV2MoE",
    hidden_states: torch.Tensor,
    forward_batch: Optional["ForwardBatch"],
    input_ids_global: Optional[torch.Tensor],
    num_tokens: int,
) -> torch.Tensor:
    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    hidden_size = moe.config.hidden_size

    if num_tokens > 0:
        router_logits = moe.gate(hidden_states, forward_batch=forward_batch)
        topk_kwargs = {"input_ids": input_ids_global} if moe.is_hash else {}
        topk_output = moe.topk(
            hidden_states,
            router_logits,
            num_token_non_padded=(
                forward_batch.num_token_non_padded
                if forward_batch is not None
                else None
            ),
            expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                layer_id=moe.layer_id,
            ),
            **topk_kwargs,
        )
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights
    else:
        topk_ids = None
        topk_weights = None

    ep_group = get_moe_ep_group().device_group
    num_experts = moe.experts.num_experts
    top_k = moe.config.num_experts_per_tok + moe.num_fused_shared_experts
    intermediate_size = moe.config.moe_intermediate_size
    num_max_tokens_per_rank = (
        envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    )
    global_num_tokens = get_dp_global_num_tokens()
    dispatch_num_tokens = max(global_num_tokens) if global_num_tokens else num_tokens
    assert dispatch_num_tokens <= num_max_tokens_per_rank, (
        f"mega MoE: max_tokens_per_rank={dispatch_num_tokens} exceeds cap "
        f"SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="
        f"{num_max_tokens_per_rank}; raise the env var or shrink "
        f"cuda_graph_max_bs / chunked_prefill_size accordingly"
    )

    runtime = get_hcu_mega_moe_runtime() if _IS_HCU else _HCU_MEGA_MOE_RUNTIME_DEEP_GEMM
    cuda_graph_max_tokens_per_rank = (
        _get_hcu_cuda_graph_max_tokens_per_rank(
            num_max_tokens_per_rank,
            dispatch_num_tokens,
        )
        if _IS_HCU
        and runtime == _HCU_MEGA_MOE_RUNTIME_MEGAMOE
        and get_is_capture_mode()
        else None
    )
    buf = _get_mega_moe_symm_buffer(
        ep_group,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_topk=top_k,
        hidden=hidden_size,
        intermediate_hidden=intermediate_size,
        runtime=runtime,
        cuda_graph_max_tokens_per_rank=cuda_graph_max_tokens_per_rank,
    )

    if _IS_HCU:
        if runtime == _HCU_MEGA_MOE_RUNTIME_MEGAMOE:
            y = _run_standalone_hcu_w8a8_mega_moe(
                hidden_states=hidden_states,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                moe=moe,
                buf=buf,
                num_tokens=num_tokens,
                hidden_size=hidden_size,
                dispatch_num_tokens=dispatch_num_tokens,
            )
        else:
            y = _run_deep_gemm_hcu_w8a8_mega_moe(
                hidden_states=hidden_states,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                moe=moe,
                buf=buf,
                num_tokens=num_tokens,
                hidden_size=hidden_size,
                dispatch_num_tokens=dispatch_num_tokens,
            )
        if not moe.experts.should_fuse_routed_scaling_factor_in_topk:
            y.mul_(moe.routed_scaling_factor)
        return y

    import deep_gemm

    if num_tokens > 0:
        topk_ids_in = topk_ids.to(torch.int32)
        topk_weights_in = topk_weights.to(torch.float32)
    else:
        topk_ids_in = hidden_states.new_empty((0, top_k), dtype=torch.int32)
        topk_weights_in = hidden_states.new_empty((0, top_k), dtype=torch.float32)

    use_fp4_acts = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS.get()
    if use_fp4_acts:
        # FP4 path goes through DeepGEMM's mega_moe_pre_dispatch which
        # handles the E2M1 packing variant. The jit implementation
        # only emits FP8.
        deep_gemm.mega_moe_pre_dispatch(
            hidden_states,
            topk_ids_in,
            topk_weights_in,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            num_tokens=num_tokens,
            group_size=32,
            use_fp4_acts=True,
        )
    else:
        mega_moe_pre_dispatch(
            hidden_states,
            topk_ids_in,
            topk_weights_in,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            quant_group_size=32,
        )

    # Allocate at least one row so y has a non-null CUDA data_ptr;
    # the DeepGEMM tvm-ffi binding rejects nullptr in convert_to_torch_tensor().
    y = torch.empty(
        (max(num_tokens, 1), hidden_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    swiglu_limit = getattr(moe.config, "swiglu_limit", None)
    deep_gemm.fp8_fp4_mega_moe(
        y,
        moe.experts.mega_l1_weights,
        moe.experts.mega_l2_weights,
        buf,
        recipe=(1, 1, 32),
        activation="swiglu",
        activation_clamp=swiglu_limit,
        fast_math=True,
    )
    y = y[:num_tokens]

    if not moe.experts.should_fuse_routed_scaling_factor_in_topk:
        y.mul_(moe.routed_scaling_factor)
    return y


def _run_deep_gemm_hcu_w8a8_mega_moe(
    *,
    hidden_states: torch.Tensor,
    topk_ids: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    moe: "DeepseekV2MoE",
    buf,
    num_tokens: int,
    hidden_size: int,
    dispatch_num_tokens: int,
) -> torch.Tensor:
    import deep_gemm

    if num_tokens > 0:
        x_fp8, x_scale = deep_gemm.cast_to_fp8_channelwise(hidden_states)
        buf.x[:num_tokens].copy_(x_fp8)
        buf.x_sf[:num_tokens].copy_(x_scale)
        buf.topk_idx[:num_tokens].copy_(topk_ids.to(buf.topk_idx.dtype))
        buf.topk_weights[:num_tokens].copy_(topk_weights.to(buf.topk_weights.dtype))

    y = torch.empty(
        (num_tokens, hidden_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    deep_gemm.fp8_w8a8_mega_moe(
        y,
        moe.experts.mega_l1_weights,
        moe.experts.mega_l2_weights,
        buf,
        recipe=(1, 1, 32),
        activation="swiglu",
        activation_clamp=getattr(moe.config, "swiglu_limit", None),
        fast_math=True,
        dispatch_num_tokens=dispatch_num_tokens,
    )
    return y


def _run_standalone_hcu_w8a8_mega_moe(
    *,
    hidden_states: torch.Tensor,
    topk_ids: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    moe: "DeepseekV2MoE",
    buf,
    num_tokens: int,
    hidden_size: int,
    dispatch_num_tokens: int,
) -> torch.Tensor:
    import megamoe

    is_graph_capture = get_is_capture_mode()
    if num_tokens == 0 and dispatch_num_tokens == 0:
        return torch.empty(
            (0, hidden_size),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )

    if num_tokens > 0:
        _prepare_standalone_megamoe_inputs(
            hidden_states,
            topk_ids,
            topk_weights,
            buf,
            num_tokens,
        )

    graph_capacity_tokens = _get_hcu_graph_capture_capacity_tokens(num_tokens, buf) if is_graph_capture else None
    y = torch.empty(
        (
            max(
                num_tokens,
                graph_capacity_tokens if graph_capacity_tokens is not None else 1,
            ),
            hidden_size,
        ),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    api_kwargs = {"megamoe_backend": _select_hcu_megamoe_backend(dispatch_num_tokens)}
    if is_graph_capture:
        api_kwargs["graph"] = True
        api_kwargs["capacity_num_tokens"] = graph_capacity_tokens
    else:
        api_kwargs["capacity_num_tokens"] = dispatch_num_tokens

    megamoe.fp8_w8a8_mega_moe(
        y,
        moe.experts.mega_l1_weights,
        moe.experts.mega_l2_weights,
        buf,
        cumulative_local_expert_recv_stats=(
            None
            if is_graph_capture
            else getattr(moe.experts, "mega_moe_recv_stats", None)
        ),
        activation_clamp=getattr(moe.config, "swiglu_limit", None),
        fast_math=True,
        **api_kwargs,
    )
    return y[:num_tokens]


def build_mega_moe_experts_weights(experts) -> None:
    from deep_gemm import (
        transform_sf_into_required_layout,
        transform_weights_for_mega_moe,
    )
    from deep_gemm.mega import _interleave_l1_weights, _transpose_sf_for_utccp

    if getattr(experts, "_mega_moe_weights_built", False):
        return

    w13 = experts.w13_weight.data
    w13_sf_fp32 = experts.w13_weight_scale_inv.data
    w2 = experts.w2_weight.data
    w2_sf_fp32 = experts.w2_weight_scale_inv.data

    num_groups, n1, half_k1 = w13.shape
    k1 = half_k1 * 2
    _, n2, half_k2 = w2.shape
    k2 = half_k2 * 2

    w13_sf = transform_sf_into_required_layout(
        w13_sf_fp32,
        mn=n1,
        k=k1,
        recipe=(1, 32),
        num_groups=num_groups,
        disable_ue8m0_cast=False,
    )
    w2_sf = transform_sf_into_required_layout(
        w2_sf_fp32,
        mn=n2,
        k=k2,
        recipe=(1, 32),
        num_groups=num_groups,
        disable_ue8m0_cast=False,
    )

    if envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get():
        # Build the interleaved L1 weight + scale once; share the weight buffer
        # between `w13_weight.data` (normal deep-ep path) and `mega_l1_weights[0]`
        # (mega moe path). Mega moe additionally needs a UTCCP-transposed scale;
        # the deep-ep path consumes the non-transposed interleaved scale and a
        # swizzle-aware activation kernel. L2 weight is untouched by the mega
        # transform, so the existing `w2_weight.data` is shared directly.
        w13_interleaved, w13_sf_interleaved = _interleave_l1_weights((w13, w13_sf))
        w13_sf_utccp = _transpose_sf_for_utccp(w13_sf_interleaved)
        w2_sf_utccp = _transpose_sf_for_utccp(w2_sf)

        experts.w13_weight.data = w13_interleaved
        experts.w13_weight_scale_inv.data = w13_sf_interleaved
        experts.w2_weight_scale_inv.data = w2_sf
        experts.w13_weight_scale_inv.format_ue8m0 = True
        experts.w2_weight_scale_inv.format_ue8m0 = True

        experts.mega_l1_weights = (experts.w13_weight.data, w13_sf_utccp)
        experts.mega_l2_weights = (experts.w2_weight.data, w2_sf_utccp)
    else:
        l1_pair, l2_pair = transform_weights_for_mega_moe((w13, w13_sf), (w2, w2_sf))

        experts.mega_l1_weights = l1_pair
        experts.mega_l2_weights = l2_pair

    experts._mega_moe_weights_built = True


def _hcu_channelwise_scale(experts, names, rows: int, label: str) -> torch.Tensor:
    num_experts = int(experts.w13_weight.shape[0])
    for name in names:
        scale_param = getattr(experts, name, None)
        if scale_param is None:
            continue
        scale = scale_param.data if hasattr(scale_param, "data") else scale_param
        if scale.dim() == 3 and scale.shape == (num_experts, rows, 1):
            return scale.squeeze(-1).to(torch.float32).contiguous()
        if scale.dim() == 2 and scale.shape == (num_experts, rows):
            return scale.to(torch.float32).contiguous()
    raise ValueError(
        "HCU W8A8 MegaMoE requires channelwise FP32 scales shaped "
        f"[expert,row] for {label}; checked {', '.join(names)}"
    )


def build_hcu_w8a8_mega_moe_experts_weights(experts) -> None:
    runtime = get_hcu_mega_moe_runtime()

    if getattr(experts, "_mega_moe_weights_built", False):
        built_runtime = getattr(experts, "_mega_moe_hcu_runtime", None)
        if built_runtime != runtime:
            raise RuntimeError(
                "HCU MegaMoE expert weights were already built for "
                f"{built_runtime!r}, cannot reuse them with {runtime!r}"
            )
        return

    w13 = experts.w13_weight.data
    w2 = experts.w2_weight.data
    if w13.dim() != 3 or w2.dim() != 3 or w13.shape[0] != w2.shape[0]:
        raise ValueError("HCU W8A8 MegaMoE expects grouped 3D expert weights")
    if w13.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        raise ValueError("HCU W8A8 MegaMoE expects torch.float8_e4m3fn expert weights")

    num_experts, l1_rows, hidden = w13.shape
    _, l2_rows, intermediate = w2.shape
    if l1_rows != 2 * intermediate or l2_rows != hidden:
        raise ValueError("HCU W8A8 MegaMoE expects w13=[E,2I,H] and w2=[E,H,I]")

    w13_scale = _hcu_channelwise_scale(
        experts,
        ("w13_weight_scale", "w13_weight_scale1"),
        l1_rows,
        "w13",
    )
    w2_scale = _hcu_channelwise_scale(
        experts,
        ("w2_weight_scale", "w2_weight_scale1"),
        l2_rows,
        "w2",
    )

    if runtime == _HCU_MEGA_MOE_RUNTIME_DEEP_GEMM:
        if l1_rows % 16 != 0 or hidden % 16 != 0 or intermediate % 16 != 0:
            raise ValueError(
                "deep_gemm HCU W8A8 MegaMoE requires rows and K divisible by 16"
            )
        import deep_gemm

        experts.mega_l1_weights = (
            deep_gemm.weight8bit_nt_kpack2_marlin(w13.contiguous()),
            w13_scale,
        )
        experts.mega_l2_weights = (
            deep_gemm.weight8bit_nt_kpack2_marlin(w2.contiguous()),
            w2_scale,
        )
        experts._mega_moe_hcu_weight_layout = "marlin_kpack2"
    else:

        import megamoe

        if _is_pd_prefill_instance():
            experts.mega_l1_weights = {
                "normal": (
                    megamoe.flatten_pack5_weight_asm_normal(w13.contiguous()),
                    w13_scale,
                )
            }
            experts.mega_l2_weights = {
                "normal": (
                    megamoe.flatten_pack5_weight_asm_normal(w2.contiguous()),
                    w2_scale,
                )
            }
            experts._mega_moe_hcu_weight_layout = "normal"
        elif _is_pd_decode_instance() and _is_hcu_v4_pro_ll_masked_k1_shape(experts, w13, w2):
            experts.mega_l1_weights = {
                "ll_pro_masked": (
                    megamoe.weight8bit_nt_kpack2_marlin_masked(w13),
                    w13_scale,
                ),
            }
            experts.mega_l2_weights = {
                "unified": (megamoe.flatten_pack5_weight(w2), w2_scale),
            }
            experts._mega_moe_hcu_weight_layout = "ll_pro_masked"
        else:
            experts.mega_l1_weights = {
                "unified": (
                    megamoe.flatten_pack5_weight(w13.contiguous()),
                    w13_scale,
                )
            }
            experts.mega_l2_weights = {
                "unified": (
                    megamoe.flatten_pack5_weight(w2.contiguous()),
                    w2_scale,
                )
            }
            experts._mega_moe_hcu_weight_layout = "unified"

    experts._mega_moe_hcu_runtime = runtime
    experts._mega_moe_hcu_w8a8_weights = True
    experts._mega_moe_weights_built = True
