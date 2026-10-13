# Copyright 2026 Hygon Information Technology Co., Ltd.
#
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

import importlib
import logging
import os
import re
from types import MappingProxyType
from typing import Dict, Iterable, List, Mapping, Optional

import torch
import torch.nn.functional as F
import triton.language as tl
from torch.nn.parameter import Parameter

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.environ import envs
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe import (
    MoeRunner,
    MoeRunnerBackend,
    MoeRunnerConfig,
    get_moe_a2a_backend,
)
from sglang.srt.layers.moe.utils import (
    W4A8_TPMOE_BACKEND_AITER,
    W4A8_TPMOE_BACKEND_AUTO,
    W4A8_TPMOE_BACKEND_LIGHTOP,
    W4A8_TPMOE_BACKEND_TRITON,
    get_dspark_w4a8_tpmoe_backend_override,
    normalize_w4a8_tpmoe_backend,
)
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from sglang.srt.layers.quantization.slimquant_w4a8 import SlimQuantW4A8Int8LinearMethod
from sglang.srt.layers.quantization.w4a8_utils import w4a8_weight_repack_impl
from sglang.srt.utils import get_bool_env_var, set_weight_attrs
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
# from sglang.srt.layers.moe.token_dispatcher.base import CombineInput

logger = logging.getLogger(__name__)

W4A8_TPMOE_BACKEND_ENV = "SGLANG_W4A8_TPMOE_BACKEND"

_lmslim_w4a8_marlin_available = False
_lmslim_w4a8_triton_available = False
_aiter_w4a8_marlin_available = False
fused_experts_impl_w4a8_marlin = None
w4a8_triton = None
moe_align_block_size_lightop = None
get_moe_cache = None
per_token_quant_int8 = None
MoeQuantType = None
aiter_moe = None
get_aiter_moe_config = None
w4a8_moe_layout_shuffle_gemm2 = None
w4a8_moe_repack_shuffle = None


def _ensure_lightop_w4a8_marlin_available() -> None:
    global _lmslim_w4a8_marlin_available
    global fused_experts_impl_w4a8_marlin

    if _lmslim_w4a8_marlin_available:
        return
    try:
        from lightop.moe import (
            fused_experts_impl_w4a8_marlin as _fused_experts_impl_w4a8_marlin,
        )
        from lightop._lmslim_native.layers.fused_moe import (
            w4a8_marlin as _lightop_w4a8_marlin,
        )
    except Exception as e:
        raise RuntimeError(
            "lightop backend is selected for w4a8 tpmoe, but lightop is not available."
        ) from e

    _lightop_w4a8_marlin.moe_align_block_size_lightop = (
        _safe_lightop_ep_moe_align_block_size
    )
    fused_experts_impl_w4a8_marlin = _fused_experts_impl_w4a8_marlin
    _lmslim_w4a8_marlin_available = True


def _ensure_lightop_w4a8_triton_available() -> None:
    global _lmslim_w4a8_triton_available
    global w4a8_triton, moe_align_block_size_lightop, get_moe_cache, per_token_quant_int8

    if _lmslim_w4a8_triton_available:
        return
    try:
        from lightop._lmslim_native.layers.fused_moe import w4a8 as _w4a8_triton
        from lightop._lmslim_native.layers.fused_moe.w4a8_marlin import (
            moe_align_block_size_lightop as _moe_align_block_size_lightop,
        )
        from lightop._lmslim_native.vllm_compat.fused_moe_cache import (
            get_moe_cache as _get_moe_cache,
        )
        from lightop.quant import per_token_quant_int8 as _per_token_quant_int8
    except Exception as e:
        raise RuntimeError(
            "triton backend is selected for w4a8 tpmoe, but lightop triton "
            "kernels are not available."
        ) from e

    w4a8_triton = _w4a8_triton
    moe_align_block_size_lightop = _moe_align_block_size_lightop
    get_moe_cache = _get_moe_cache
    per_token_quant_int8 = _per_token_quant_int8
    _lmslim_w4a8_triton_available = True


def _ensure_aiter_w4a8_marlin_available() -> None:
    global _aiter_w4a8_marlin_available
    global MoeQuantType, aiter_moe, get_aiter_moe_config
    global w4a8_moe_layout_shuffle_gemm2
    global w4a8_moe_repack_shuffle

    if _aiter_w4a8_marlin_available:
        return
    try:
        from aiter.moe import MoeQuantType as _MoeQuantType
        from aiter.moe import aiter_moe as _aiter_moe
        from aiter.moe import get_aiter_moe_config as _get_aiter_moe_config
        from aiter.ops.shuffle import (
            w4a8_moe_layout_shuffle_gemm2 as _w4a8_moe_layout_shuffle_gemm2,
            w4a8_moe_repack_shuffle as _w4a8_moe_repack_shuffle,
        )
    except Exception as e:
        raise RuntimeError(
            "aiter backend is selected for w4a8 tpmoe, but aiter is not available."
        ) from e

    MoeQuantType = _MoeQuantType
    aiter_moe = _aiter_moe
    get_aiter_moe_config = _get_aiter_moe_config
    w4a8_moe_layout_shuffle_gemm2 = _w4a8_moe_layout_shuffle_gemm2
    w4a8_moe_repack_shuffle = _w4a8_moe_repack_shuffle
    _aiter_w4a8_marlin_available = True


def _aiter_w4a8_config_kwargs(**kwargs):
    """Force moe_c when SGLANG_FORCE_AITER_MOE_C=1 (tuned gfx936 configs)."""
    if envs.SGLANG_FORCE_AITER_MOE_C.get():
        from aiter.moe import MoeSolutionType

        kwargs["spec_sol_type"] = MoeSolutionType.MOE_C
    return kwargs


def _lookup_aiter_w4a8_config(**kwargs):
    """Look up the moe_c W4A8 config, avoiding the gfx936 M=4 outlier.

    The tuned table gives every small M ``MODE=374116`` except M=4, which is
    ``MODE=374216``. MTP target-verify forwards exactly 4 tokens, so it is the
    only path that selects that mode. Decode (M=1) and prefill (larger M) stay
    on 374116 and are accurate. Reuse the M=8 entry; its block size matches.
    """
    _ensure_aiter_w4a8_marlin_available()
    status, cfg = get_aiter_moe_config(**kwargs)
    if int(kwargs.get("M", -1)) != 4 or not status:
        return status, cfg
    alt_kwargs = dict(kwargs)
    alt_kwargs["M"] = 8
    status_alt, cfg_alt = get_aiter_moe_config(**alt_kwargs)
    if not status_alt:
        return status, cfg
    if not getattr(_lookup_aiter_w4a8_config, "_logged", False):
        logger.info(
            "[slimquant_w4a8_marlin] moe_c M=4 MODE=%s; using M=8 MODE=%s",
            (getattr(cfg, "config", None) or {}).get("MODE"),
            (getattr(cfg_alt, "config", None) or {}).get("MODE"),
        )
        _lookup_aiter_w4a8_config._logged = True
    return True, cfg_alt


_moe_c_align_patched = False
_moe_c_m4_mode_patched = False


def _patch_moe_c_m4_mode() -> None:
    """Replace the gfx936 M=4 moe_c mode with the neighboring entry.

    ``get_moe_configs_marlin`` picks the closest token-count key.
    ``E=128,N=640,K=2560`` uses MODE 374116 for every small M except M=4
    (MODE 374216). MTP verify is exactly 4 tokens, and ``moe_c_fused_experts``
    reloads that table itself, so the caller-side config is not enough.
    """
    global _moe_c_m4_mode_patched
    if _moe_c_m4_mode_patched:
        return
    import aiter.fused_moe_c as fmc

    original = fmc.get_moe_configs_marlin

    def _get_moe_configs_marlin(*args, **kwargs):
        configs = original(*args, **kwargs)
        if isinstance(configs, dict):
            good = configs.get(8)
            bad = configs.get(4)
            if (
                isinstance(good, dict)
                and isinstance(bad, dict)
                and bad.get("MODE") != good.get("MODE")
            ):
                fixed = dict(bad)
                fixed["MODE"] = good["MODE"]
                configs[4] = fixed
                if not getattr(_get_moe_configs_marlin, "_logged", False):
                    logger.info(
                        "[slimquant_w4a8_marlin] moe_c table M=4 MODE %s -> %s",
                        bad.get("MODE"),
                        good["MODE"],
                    )
                    _get_moe_configs_marlin._logged = True
        return configs

    fmc.get_moe_configs_marlin = _get_moe_configs_marlin
    _moe_c_m4_mode_patched = True


def _patch_moe_c_align_non_cub_to_triton() -> None:
    """Use Triton align when cub fused cannot launch.

    Cub ``fused_cub<E,8,E>`` only launches for E in {64,128,256} and only
    when launch threads == E. EP drop-sentinel uses E+1=129, so the wheel
    falls through to the legacy align kernel, which scrambles topk=10 EP
    token order and produces garbage without VMFault. Mirror the E>=224
    Triton path for every E cub cannot take.
    """
    global _moe_c_align_patched
    if _moe_c_align_patched:
        return
    import triton
    import aiter.fused_moe_c as fmc

    orig = fmc.moe_align_block_size

    def moe_align_block_size(topk_ids, block_size, num_experts, expert_map=None):
        if num_experts in (64, 128, 256):
            return orig(topk_ids, block_size, num_experts, expert_map)

        max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
        sorted_ids = torch.empty(
            (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
        )
        max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
        expert_ids = torch.empty(
            (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
        )
        num_tokens_post_pad = torch.empty(
            (1,), dtype=torch.int32, device=topk_ids.device
        )
        sorted_ids.fill_(topk_ids.numel())
        expert_ids.zero_()
        fmc.moe_align_block_size_triton(
            topk_ids,
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
        )
        if expert_map is not None:
            expert_ids = expert_map[expert_ids]
        return sorted_ids, expert_ids, num_tokens_post_pad

    fmc.moe_align_block_size = moe_align_block_size
    _moe_c_align_patched = True
    logger.info(
        "[slimquant_w4a8_marlin] patched moe_c align: non-cub E uses Triton"
    )


def _try_ensure_backend_available(ensure_backend) -> bool:
    try:
        ensure_backend()
        return True
    except RuntimeError as e:
        logger.info(str(e))
        return False


def _marlin_w4a8_tp_shapes_ok(
    hidden_size: int, intermediate_size_per_partition: int
) -> bool:
    """Marlin W4A8 repack tiles require K%32==0 and N%64==0 after unpack+T.

    Packed expert weights are [N, K/2]; ``w4a8_2_marlin_weight`` unpacks to
    [N, K] then transposes to [K, N]. For TP-sharded MoE:
      w13: N=2*intermediate_size_per_partition, K=hidden_size
      w2:  N=hidden_size, K=intermediate_size_per_partition
    e.g. Qwen3.8-Flash-Next (inter=640, TP=8) yields N_w13=160 / K_w2=80,
    which fails these constraints.
    """
    w13_n = 2 * intermediate_size_per_partition
    w13_k = hidden_size
    w2_n = hidden_size
    w2_k = intermediate_size_per_partition
    return (
        w13_k % 32 == 0
        and w13_n % 64 == 0
        and w2_k % 32 == 0
        and w2_n % 64 == 0
    )


def _resolve_w4a8_tpmoe_backend(
    requested_backend: str, *, env_name: str = W4A8_TPMOE_BACKEND_ENV
) -> str:
    backend = normalize_w4a8_tpmoe_backend(requested_backend, env_name=env_name)
    if backend == W4A8_TPMOE_BACKEND_AUTO:
        if _try_ensure_backend_available(_ensure_lightop_w4a8_marlin_available):
            return W4A8_TPMOE_BACKEND_LIGHTOP
        if _try_ensure_backend_available(_ensure_aiter_w4a8_marlin_available):
            return W4A8_TPMOE_BACKEND_AITER
        raise RuntimeError(
            "Neither lightop nor aiter backend is available for w4a8 tpmoe."
        )
    if backend == W4A8_TPMOE_BACKEND_LIGHTOP:
        _ensure_lightop_w4a8_marlin_available()
    elif backend == W4A8_TPMOE_BACKEND_TRITON:
        _ensure_lightop_w4a8_triton_available()
    else:
        _ensure_aiter_w4a8_marlin_available()
    return backend

_use_aiter_moe = get_bool_env_var("SGLANG_ROCM_USE_AITER_MOE", default="true")
_use_lightop_w4a8_marlin_moe = get_bool_env_var("SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE", default="true")
_use_int4_w4a8 = get_bool_env_var("SGLANG_USE_INT4_W4A8")

class MarlinMoeWorkspace:
    """
    Singleton manager for device-specific workspace buffers used by w4a8 Marlin-MoE.
    global_reduce_buffer will take 1.5MB * cus (about 120MB for BW200) memory in each device
    """

    _instances = {}

    def __new__(cls, device):
        if device not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[device] = instance
        return cls._instances[device]

    def __init__(self, device):
        if self._initialized:
            return
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.workspace = torch.zeros(
            500, dtype=torch.int, device=device, requires_grad=False
        )
        self.global_reduce_buffer = torch.zeros(
            sms * 6 * 128 * 512, dtype=torch.int, device=device, requires_grad=False
        )
        self._initialized = True

    def get_buffers(self):
        return self.workspace, self.global_reduce_buffer


def repack_and_shuffle_w4a8(weight_data, E):
    """Contiguous K-pack [E, N, K/2] → moe_c blocked + layout shuffle.

    Matches ``aiter.ops.shuffle.w4a8_moe_repack_shuffle`` / moe golden tests.
    """
    _ensure_aiter_w4a8_marlin_available()
    if w4a8_moe_repack_shuffle is not None:
        if not weight_data.is_contiguous():
            weight_data = weight_data.contiguous()
        return w4a8_moe_repack_shuffle(weight_data)

    from aiter.ops.shuffle import w4a8_moe_layout_shuffle_gemm2

    for i in range(E):
        expert = weight_data[i]
        n, k_half = expert.shape
        w_u8 = expert.to(torch.uint8)
        w_unpacked = torch.stack([(w_u8 >> 4) & 0x0F, w_u8 & 0x0F], dim=-1).view(n, -1)
        blocks = w_unpacked.view(n, -1, 8)
        packed = ((blocks[..., :4] << 4) | blocks[..., 4:]).view(n, k_half)
        shuffled = w4a8_moe_layout_shuffle_gemm2(packed).contiguous().reshape(n, k_half)
        weight_data[i].copy_(shuffled)
    return weight_data


def baseline_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    scales = scale_a * scale_b.T
    gemmout = torch.mm(a.to(dtype=torch.float32), b.to(dtype=torch.float32))
    output = (scales * gemmout).to(out_dtype)
    if bias is not None:
        output = output + bias
    return output.to(out_dtype)


def _get_w4a8_triton_chunk_size(
    cache13: torch.Tensor,
    *,
    top_k: int,
    n1: int,
    n2: int,
    num_tokens: int,
) -> int:
    requested_chunk_size = int(os.getenv("LMSLIM_FUSED_MOE_CHUNK_SIZE", "32768"))
    if requested_chunk_size <= 0:
        raise ValueError(
            "LMSLIM_FUSED_MOE_CHUNK_SIZE must be positive, "
            f"got {requested_chunk_size}."
        )
    cache_token_capacity = cache13.numel() // (top_k * max(n1, n2))
    if cache_token_capacity <= 0:
        raise RuntimeError(
            "W4A8 Triton MoE cache is too small: "
            f"cache_numel={cache13.numel()}, top_k={top_k}, n1={n1}, n2={n2}."
        )
    return min(requested_chunk_size, num_tokens, cache_token_capacity)


def _safe_lightop_ep_moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    expert_mask: Optional[torch.Tensor] = None,
    num_local_tokens: Optional[torch.Tensor] = None,
    pad_sorted_ids: bool = False,
    ep_size: int = 8,
    num_token: Optional[int] = None,
):
    if expert_map is None:
        raise ValueError("The safe LightOp EP align requires expert_map.")
    local_num_experts = num_experts // ep_size
    valid = (topk_ids >= 0) & (topk_ids < expert_map.numel())
    safe_ids = topk_ids.clamp(0, expert_map.numel() - 1).to(torch.long)
    local_ids = expert_map[safe_ids]
    local_ids = torch.where(
        valid & (local_ids >= 0) & (local_ids < local_num_experts),
        local_ids,
        torch.full_like(local_ids, -1),
    )
    if expert_mask is not None:
        local_ids = local_ids.masked_fill(expert_mask, -1)
    if num_local_tokens is not None:
        rows = torch.arange(topk_ids.shape[0], device=topk_ids.device)
        local_ids = local_ids.masked_fill(rows[:, None] >= num_local_tokens, -1)

    from sglang.srt.layers.moe.fused_moe_triton import (
        moe_align_block_size as sglang_moe_align_block_size,
    )

    return sglang_moe_align_block_size(local_ids, block_size, local_num_experts)


def _fit_w4a8_triton_block(block: int, dim: int, *, even: bool = False) -> int:
    """Shrink ``block`` so it divides ``dim`` (kernel has no K/N remainder mask).

    ``fused_moe_kernel_int4_w4a8`` loads B with ``BLOCK_SIZE_K // 2`` packed
    elements and no K-mask, so e.g. K=80 + BLOCK_SIZE_K=64 walks past the
    packed dim (40) and VMFaults on HCU. N uses ``% N`` wraparound without a
    store mask on B, so BLOCK_SIZE_N must also divide N for correctness.
    """
    b = min(int(block), int(dim))
    if even and b % 2:
        b -= 1
    while b > (2 if even else 1) and dim % b != 0:
        b //= 2
        if even and b % 2:
            b -= 1
    return max(b, 2 if even else 1)


def _adjust_w4a8_triton_block_config(
    config: dict, *, n: int, k: int
) -> dict:
    cfg = dict(config)
    cfg["BLOCK_SIZE_K"] = _fit_w4a8_triton_block(
        cfg.get("BLOCK_SIZE_K", 64), k, even=True
    )
    cfg["BLOCK_SIZE_N"] = _fit_w4a8_triton_block(
        cfg.get("BLOCK_SIZE_N", 64), n, even=False
    )
    return cfg


def fused_experts_impl_w4a8_triton(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    cache13: torch.Tensor,
    *,
    activation: str,
    apply_router_weight_on_input: bool,
    global_num_experts: int,
    expert_map: Optional[torch.Tensor],
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    routed_scaling_factor: float,
    num_local_tokens: Optional[torch.Tensor] = None,
    shared_output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run SlimQuant W4A8 Triton MoE GEMMs without Marlin repack."""
    assert hidden_states.ndim == 2 and hidden_states.is_contiguous()
    assert hidden_states.shape[1] == w1.shape[2] * 2
    assert topk_weights.shape == topk_ids.shape

    num_tokens = hidden_states.shape[0]
    if num_tokens == 0:
        return torch.empty_like(hidden_states)

    top_k = topk_ids.shape[1]
    n1 = w1.shape[1]
    n2 = w2.shape[1]
    if global_num_experts == -1:
        global_num_experts = w1.shape[0]
    expert_mask = None
    if expert_map is not None:
        # Mirrors the lightop marlin EP path: the align op expects an
        # (num_tokens, top_k) validity mask; all received slots are valid.
        expert_mask = torch.zeros(
            (num_tokens, top_k), dtype=torch.bool, device=hidden_states.device
        )
    chunk_size = _get_w4a8_triton_chunk_size(
        cache13,
        top_k=top_k,
        n1=n1,
        n2=n2,
        num_tokens=num_tokens,
    )
    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16
    output = torch.empty_like(hidden_states)

    for begin in range(0, num_tokens, chunk_size):
        end = min(begin + chunk_size, num_tokens)
        token_count = end - begin
        current_x = hidden_states[begin:end]
        current_ids = topk_ids[begin:end]
        current_weights = topk_weights[begin:end]
        cache1 = cache13[: token_count * top_k * n1].view(token_count, top_k, n1)
        cache3 = cache13[: token_count * top_k * n2].view(token_count, top_k, n2)

        config1, config2 = w4a8_triton.get_w8a8moe_json(
            token_count, w1.shape[0], n1, n2, n1 // 2
        )
        # GEMM1: A[K=n2/hidden], B[E,N=n1]; GEMM2: A[K=n1/2], B[E,N=n2].
        config1 = _adjust_w4a8_triton_block_config(config1, n=n1, k=n2)
        config2 = _adjust_w4a8_triton_block_config(config2, n=n2, k=n1 // 2)
        if expert_map is None:
            # StandardDispatcher apply path: topk_ids are LOCAL (0..E-1) with a
            # -1 sentinel for experts owned by other EP ranks. lightop's
            # vllm_compat.moe_align_block_size treats -1 as a large expert id and
            # faults with num_experts=256 or misbuckets the -1 entries with
            # num_experts=E. Route through sglang's align, which maps -1 to a
            # sentinel expert on HCU and maps the sentinel blocks back to -1 for
            # the GEMM kernel's filter-expert path.
            from sglang.srt.layers.moe.fused_moe_triton import (
                moe_align_block_size as sglang_moe_align_block_size,
            )

            sorted_ids, expert_ids, padded_count = sglang_moe_align_block_size(
                current_ids, config1["BLOCK_SIZE_M"], w1.shape[0]
            )
        else:
            # DeepEPMoE apply_ep path: topk_ids are GLOBAL and expert_map maps
            # global experts to local ids (-1 for experts on other ranks).
            # lightop's vllm_compat align drops expert_map in the lightop op
            # path, so use the lightop EP align from the marlin backend, which
            # passes the map to the op and aligns only this rank's tokens.
            sorted_ids, expert_ids, padded_count = moe_align_block_size_lightop(
                current_ids,
                config1["BLOCK_SIZE_M"],
                global_num_experts,
                expert_map=expert_map,
                expert_mask=expert_mask[begin:end],
                num_local_tokens=num_local_tokens,
                ep_size=global_num_experts // w1.shape[0],
            )
        qx, x_scale = per_token_quant_int8(current_x)
        w4a8_triton.invoke_fused_moe_kernel_w4a8(
            qx,
            w1,
            cache1,
            x_scale,
            w1_scale,
            None,
            current_weights,
            sorted_ids,
            expert_ids,
            padded_count,
            apply_router_weight_on_input,
            top_k,
            config1,
            compute_type=compute_type,
        )

        gate, up = cache1.chunk(2, dim=-1)
        if activation == "silu":
            activated = F.silu(gate) * up
        elif activation == "gelu":
            activated = F.gelu(gate) * up
        else:
            raise ValueError(f"Unsupported FusedMoE activation: {activation}")

        qactivated, activated_scale = per_token_quant_int8(
            activated.reshape(token_count * top_k, n1 // 2)
        )
        if expert_map is not None:
            # The EP align only schedules this rank's slots, so the second
            # GEMM never writes the rows of slots routed to other ranks.
            # cache1 is dead by now and cache3 overlaps its prefix, so it is
            # safe to zero cache3 here; the reduce below then sums only the
            # local contributions, matching the marlin backend's masked
            # op.moe_sum reduce.
            cache3.zero_()
        w4a8_triton.invoke_fused_moe_kernel_w4a8(
            qactivated,
            w2,
            cache3,
            activated_scale,
            w2_scale,
            None,
            current_weights,
            sorted_ids,
            expert_ids,
            padded_count,
            not apply_router_weight_on_input,
            1,
            config2,
            compute_type=compute_type,
        )
        reduced = cache3.sum(dim=1).mul_(routed_scaling_factor)
        if shared_output is not None:
            reduced.add_(shared_output[begin:end])
        output[begin:end].copy_(reduced)

    return output


_QWEN4_EXP_ARCHS = (
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpForCausalLMMTP",
)


def _hf_architectures(hf_config) -> list[str]:
    if hf_config is None:
        return []
    if isinstance(hf_config, dict):
        return list(hf_config.get("architectures") or [])
    return list(getattr(hf_config, "architectures", None) or [])


def _w8a8_glob_match(layer_name: str, pattern: str) -> bool:
    """Match checkpoint include globs such as ``model.language_model.layers.*...``.

    SGLang prefixes often omit the ``language_model`` segment, so both forms
    are tried. ``*`` matches a single dotted path component.
    """
    candidates = [pattern]
    if ".language_model." in pattern:
        candidates.append(pattern.replace(".language_model.", "."))
    for p in candidates:
        if p.startswith("re:"):
            if re.search(p[3:], layer_name):
                return True
            continue
        regex = "^" + re.escape(p).replace(r"\*", r"[^.]+") + "$"
        if re.match(regex, layer_name):
            return True
    return False


def _layer_in_w8a8_include(
    layer_name: Optional[str],
    include: Iterable[str] = tuple(),
    fused_mapping: Mapping[str, List[str]] = MappingProxyType({}),
) -> bool:
    """True if this Linear (or all fused shards) is listed in w8a8_include."""
    if layer_name is None or not include:
        return False
    include_list = list(include)
    proj_name = layer_name.split(".")[-1]
    if proj_name in fused_mapping:
        shard_names = [
            layer_name.replace(proj_name, shard_proj_name)
            for shard_proj_name in fused_mapping[proj_name]
        ]
        return all(
            any(_w8a8_glob_match(shard, pat) for pat in include_list)
            for shard in shard_names
        )
    return any(_w8a8_glob_match(layer_name, pat) for pat in include_list)


class SlimQuantW4A8Int8MarlinConfig(QuantizationConfig):
    """Config class for W4A8 Int8 Quantization.
    - Weight: static, per-channel, symmetric
    - Activation: dynamic, per-token, symmetric
    """

    def __init__(
        self,
        ignore: Optional[list[str]] = None,
        experts_only_linear: bool = False,
        w8a8_include: Optional[list[str]] = None,
        packed_modules_mapping: Optional[Dict[str, List[str]]] = None,
    ):
        super().__init__()
        self.ignore = ignore
        # Qwen3.8 Flash-Next ChannelWise W4A8 quantizes MoE experts only by
        # default; DeepSeek / Kimi still quantize dense Linear unless listed
        # in ignore. Mixed W4+W8 checkpoints also list dense W8 modules in
        # mixed_precision.w8a8_include.
        self.experts_only_linear = experts_only_linear
        self.w8a8_include = list(w8a8_include or [])
        self.packed_modules_mapping = packed_modules_mapping or {}

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_name(self) -> str:
        return "slimquant_w4a8_marlin"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, any]) -> "SlimQuantW4A8Int8MarlinConfig":
        archs = _hf_architectures(config.get("hf_config"))
        experts_only_linear = any(arch in _QWEN4_EXP_ARCHS for arch in archs)
        mixed = config.get("mixed_precision") or {}
        w8a8 = mixed.get("w8a8_include") or {}
        return cls(
            ignore=config.get("ignore"),
            experts_only_linear=experts_only_linear,
            w8a8_include=list(w8a8.get("modules") or []),
            packed_modules_mapping=config.get("packed_modules_mapping") or {},
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        if hf_quant_cfg.get("quant_method") == "slimquant_w4a8" and user_quant in (
            "slimquant_w4a8_marlin",
            "slimquant_marlin",
        ):
            return cls.get_name()
        return None

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        from sglang.srt.layers.moe.fused_moe_triton import (
            FusedMoE,
        )
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        if isinstance(layer, LinearBase):
            fused_mapping = getattr(self, "packed_modules_mapping", {}) or {}
            # Kimi-K3 INT4 (from mxfp4_to_int4.py) only quantizes the routed
            # experts; dense layers stay in BF16 and are listed in the
            # checkpoint's ignore list (native compressed-tensors style, with
            # "re:" regexes). Layers matched by the ignore list keep the
            # unquantized path; anything else goes through the W4A8 linear
            # method.
            if self.ignore and should_ignore_layer(
                layer_name=prefix,
                ignore=self.ignore,
                fused_mapping=fused_mapping,
            ):
                return UnquantizedLinearMethod()
            # Qwen3.8 Flash-Next: MoE experts are W4; denselinears are BF16
            # unless listed in mixed_precision.w8a8_include (W8A8). Without
            # that include list, keep Unquant so int8 weights are never run
            # through F.linear as bf16.
            if self.experts_only_linear:
                if _layer_in_w8a8_include(
                    prefix, self.w8a8_include, fused_mapping
                ):
                    return SlimQuantW4A8Int8LinearMethod(self)
                return UnquantizedLinearMethod()
            return SlimQuantW4A8Int8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

            fused_mapping = getattr(self, "packed_modules_mapping", {}) or {}
            # Qwen3.8 slimquant mixed W4/W8: compression_config.ignore lists
            # ``re:^mtp\\..*`` and mixed_precision.experts only covers
            # language_model layers. MTP experts stay BF16 fused
            # [E, 2N, K] (e.g. [512, 1280, 2560]); packing them as INT4
            # [E, 2N, K/2] makes EAGLE draft load copy_ 2560 into 1280.
            if self.ignore and should_ignore_layer(
                layer_name=prefix,
                ignore=self.ignore,
                fused_mapping=fused_mapping,
            ):
                logger.info(
                    "[slimquant_w4a8_marlin] FusedMoE %s is in ignore; "
                    "using UnquantizedFusedMoEMethod",
                    prefix,
                )
                return UnquantizedFusedMoEMethod(
                    getattr(layer, "use_triton_kernels", False),
                    getattr(layer, "use_flashinfer_trtllm_moe", False),
                    getattr(layer, "use_deep_gemm", False),
                )
            if get_moe_a2a_backend().is_megamoe():
                return SlimQuantW4A8Int8MarlinMoEMethod(self)
            dspark_backend_override = get_dspark_w4a8_tpmoe_backend_override()
            if dspark_backend_override is None:
                requested_backend = envs.SGLANG_W4A8_TPMOE_BACKEND.get()
                env_name = W4A8_TPMOE_BACKEND_ENV
            else:
                requested_backend = dspark_backend_override
                env_name = "SGLANG_DSPARK_FORCE_W4A8_TPMOE_BACKEND"
            resolved_backend = _resolve_w4a8_tpmoe_backend(
                requested_backend,
                env_name=env_name,
            )
            # Marlin lightop repack needs N%64 and K%32; fall back to Triton
            # when the TP shard shapes are incompatible (e.g. inter=640, TP=8).
            if resolved_backend == W4A8_TPMOE_BACKEND_LIGHTOP:
                hidden = getattr(layer, "hidden_size", None)
                inter = getattr(layer, "intermediate_size_per_partition", None)
                if (
                    hidden is not None
                    and inter is not None
                    and not _marlin_w4a8_tp_shapes_ok(hidden, inter)
                ):
                    logger.warning(
                        "[slimquant_w4a8_marlin] Marlin tile constraints not met "
                        "(hidden=%s, intermediate_per_tp=%s; need w13 N%%64==0 and "
                        "w2 K%%32==0). Falling back to Triton W4A8 MoE backend.",
                        hidden,
                        inter,
                    )
                    _ensure_lightop_w4a8_triton_available()
                    resolved_backend = W4A8_TPMOE_BACKEND_TRITON
            if resolved_backend == W4A8_TPMOE_BACKEND_AITER:
                selected_method = SlimQuantW4A8Int8AiterMoEMethod
            elif resolved_backend == W4A8_TPMOE_BACKEND_TRITON:
                selected_method = SlimQuantW4A8Int8TritonMoEMethod
            else:
                selected_method = SlimQuantW4A8Int8MarlinMoEMethod
            quant_method = selected_method(self)
            logger.info(
                "[slimquant_w4a8_marlin] selected_moe_method=%s "
                "requested_backend=%s resolved_backend=%s dspark_backend_override=%s",
                selected_method.__name__,
                requested_backend,
                resolved_backend,
                dspark_backend_override,
            )
            return quant_method
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class SlimQuantW4A8Int8MarlinMoEMethod:
    """MoE method for W4A8INT8 Marlin.
    Supports loading INT8 checkpoints with static weight scale and
    dynamic/static activation scale.
    Args:
        quant_config: The quantization config.
    """

    def __new__(cls, *args, **kwargs):

        if not hasattr(cls, "_initialized"):
            original_init = cls.__init__
            new_cls = type(
                cls.__name__,
                (FusedMoEMethodBase,),
                {
                    "__init__": original_init,
                    **{k: v for k, v in cls.__dict__.items() if k != "__dict__"},
                },
            )
            obj = super(new_cls, new_cls).__new__(new_cls)
            obj.__init__(*args, **kwargs)
            return obj
        return super().__new__(cls)

    def __init__(self, quant_config):
        from lightop.moe import fused_experts_impl_w4a8_marlin

        self.quant_config = quant_config
        self.use_deepep = get_moe_a2a_backend().is_deepep()
        self.fused_experts_impl_w4a8_marlin = fused_experts_impl_w4a8_marlin

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import (
            FusedMoeWeightScaleSupported,
        )

        tp_size = get_tensor_model_parallel_world_size()
        intermediate_size = intermediate_size_per_partition
        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size, hidden_size // 2, dtype=torch.int8
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, hidden_size, intermediate_size // 2, dtype=torch.int8
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, 2 * intermediate_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w13_input_scale = None
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = None
        layer.register_parameter("w2_input_scale", w2_input_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if get_moe_a2a_backend().is_megamoe():
            from sglang.srt.layers.moe.mega_moe import (
                build_hcu_w4a8_mega_moe_experts_weights,
            )

            build_hcu_w4a8_mega_moe_experts_weights(layer)
            return
        if not _use_lightop_w4a8_marlin_moe:
            if self.use_deepep:
                from deepgemm import pack_w4a8_moe_hipc_weight
                layer.w13_weight = Parameter(
                    pack_w4a8_moe_hipc_weight(layer.w13_weight.data),
                    requires_grad=False,
                )
                layer.w2_weight = Parameter(
                    pack_w4a8_moe_hipc_weight(layer.w2_weight.data),
                    requires_grad=False,
                )
                # Checkpoint stores scale/16. m_grouped_w4a8_gemm_nt_contiguous_hipc
                # applies that x16 internally, same as the masked marlin kernel.
                # Multiplying again here makes every expert GEMM 16x too large.
                scale_mul = 1.0
                layer.w13_weight_scale = Parameter(
                    layer.w13_weight_scale.data * scale_mul,
                    requires_grad=False,
                )
                layer.w2_weight_scale = Parameter(
                    layer.w2_weight_scale.data * scale_mul,
                    requires_grad=False,
                )
            else:
                # mxfp4_to_int4.py packs (even_k << 4) | odd_k with two's
                # complement nibbles and stores scale/16 (the legacy lightop
                # kernel applies the missing x16). The Triton W4A16 kernel instead
                # expects even k in the LOW nibble, offset-8 unsigned nibbles, and
                # the scale applied directly. Convert the checkpoint layout:
                #   - swap nibbles (even k -> low nibble)
                #   - flip bit 3 of every nibble (two's complement -> +8 offset)
                #   - restore the true per-channel scale (x16)
                def _to_triton_layout(w: torch.Tensor) -> torch.Tensor:
                    u = w.data.to(torch.uint8)
                    u = ((((u & 0x0F) << 4) | ((u >> 4) & 0x0F)) ^ 0x88).to(
                        torch.int8
                    )
                    return u.contiguous()

                if _use_aiter_moe:
                    E = layer.w13_weight.shape[0]
                    layer.w13_weight = Parameter(
                        repack_and_shuffle_w4a8(layer.w13_weight.data, E),
                        requires_grad=False,
                    )
                    layer.w2_weight = Parameter(
                        repack_and_shuffle_w4a8(layer.w2_weight.data, E),
                        requires_grad=False,
                    )
                    scale_mul = 1.0
                    layer.w13_weight_scale = Parameter(
                        layer.w13_weight_scale.data * scale_mul,
                        requires_grad=False,
                    )
                    layer.w2_weight_scale = Parameter(
                        layer.w2_weight_scale.data * scale_mul,
                        requires_grad=False,
                    )
                else:
                    layer.w13_weight = Parameter(
                        _to_triton_layout(layer.w13_weight), requires_grad=False
                    )
                    layer.w2_weight = Parameter(
                        _to_triton_layout(layer.w2_weight), requires_grad=False
                    )
                    scale_mul = 16.0
                    layer.w13_weight_scale = Parameter(
                        layer.w13_weight_scale.data * scale_mul,
                        requires_grad=False,
                    )
                    layer.w2_weight_scale = Parameter(
                        layer.w2_weight_scale.data * scale_mul,
                        requires_grad=False,
                    )
        else:
            # Legacy lightop path: repack into the Marlin W4A8 layout.
            layer.w13_weight = Parameter(
                w4a8_weight_repack_impl(
                    layer.w13_weight, use_deepep=self.use_deepep
                ),
                requires_grad=False,
            )
            layer.w2_weight = Parameter(
                w4a8_weight_repack_impl(layer.w2_weight, use_deepep=self.use_deepep),
                requires_grad=False,
            )
            layer.w13_weight_scale = Parameter(
                layer.w13_weight_scale.data, requires_grad=False
            )
            layer.w2_weight_scale = Parameter(
                layer.w2_weight_scale.data, requires_grad=False
            )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def _get_triton_quant_info(self, layer):
        """W4A16 quant info for the Triton MoE runner (same path the native
        MXFP4 model uses). Per-channel scales: block_shape[1] must be >= the K
        dimension of BOTH GEMMs (w13: hidden_size, w2: intermediate_size), so
        the kernel's group index stays 0 and the (E, N, 1) scale is read once
        per output channel.  ``SGLANG_USE_INT4_W4A8`` switches between W4A16
        and W4A8 (int8 x int8 tensor core), mirroring the MXFP4 flags."""

        if _use_aiter_moe:
            block_shape = None
        else:
            k_max = max(layer.w13_weight.shape[2], layer.w2_weight.shape[2]) * 2
            block_shape = [0, k_max]

        return TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            use_int4_w4a16=not _use_int4_w4a8,
            use_int4_w4a8=_use_int4_w4a8,
            per_channel_quant=True,
            block_shape=block_shape,
        )

    @torch._dynamo.disable()
    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output,
        i_q: Optional[torch.Tensor] = None,
        i_s: Optional[torch.Tensor] = None,
    ):
        if not _use_lightop_w4a8_marlin_moe:
            return self.runner.run(dispatch_output, self._get_triton_quant_info(layer))
        return self._apply_w4a8_marlin_lightop(layer, dispatch_output)

    @torch._dynamo.disable()
    def _apply_w4a8_marlin_lightop(
        self,
        layer: torch.nn.Module,
        dispatch_output,
    ):
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
        from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output
        x, topk_weights = apply_topk_weights_cpu(
            self.moe_runner_config.apply_router_weight_on_input, topk_weights, x
        )
        workspace, global_reduce_buffer = MarlinMoeWorkspace(x.device).get_buffers()
        routed_scaling_factor = (
            self.moe_runner_config.routed_scaling_factor
            if self.moe_runner_config.routed_scaling_factor is not None
            else 1.0
        )
        output = self.fused_experts_impl_w4a8_marlin(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            workspace=workspace,
            global_reduce_buffer=global_reduce_buffer,
            # Honor the caller's inplace setting: the fused shared-expert gate
            # re-reads hidden_states after the routed experts, so the routed
            # output must not alias the MoE input buffer.
            inplace=self.moe_runner_config.inplace,
            use_int4_w4a8=True,
            per_channel_quant=True,
            activation=layer.moe_runner_config.activation,
            expert_map=getattr(layer, "expert_map", None),
            apply_router_weight_on_input=self.moe_runner_config.apply_router_weight_on_input,
            global_num_experts=layer.moe_runner_config.num_experts,
            w1_scale=(layer.w13_weight_scale),
            w2_scale=(layer.w2_weight_scale),
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            use_nn_moe=False,
            routed_scaling_factor=routed_scaling_factor,
        )
        return StandardCombineInput(hidden_states=output)

    @torch._dynamo.disable()
    def apply_with_shared_output(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        activation: str = "silu",
        shared_output: Optional[torch.Tensor] = None,
        topk_output=None,
        i_q: Optional[torch.Tensor] = None,
        i_s: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not _use_lightop_w4a8_marlin_moe:
            from sglang.srt.layers.moe.token_dispatcher.standard import (
                StandardDispatchOutput,
            )

            dispatch_output = StandardDispatchOutput(
                hidden_states=x,
                hidden_states_scale=None,
                topk_output=topk_output,
            )
            output = self.runner.run(
                dispatch_output, self._get_triton_quant_info(layer)
            )
            return output.hidden_states
        return self._apply_with_shared_output_lightop(
            layer, x, activation, shared_output, topk_output, i_q, i_s
        )

    @torch._dynamo.disable()
    def _apply_with_shared_output_lightop(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        activation: str,
        shared_output: Optional[torch.Tensor],
        topk_output,
        i_q: Optional[torch.Tensor],
        i_s: Optional[torch.Tensor],
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        x, topk_weights = apply_topk_weights_cpu(
            self.moe_runner_config.apply_router_weight_on_input, topk_weights, x
        )
        workspace, global_reduce_buffer = MarlinMoeWorkspace(x.device).get_buffers()
        routed_scaling_factor = (
            self.moe_runner_config.routed_scaling_factor
            if self.moe_runner_config.routed_scaling_factor is not None
            else 1.0
        )
        return self.fused_experts_impl_w4a8_marlin(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            workspace=workspace,
            global_reduce_buffer=global_reduce_buffer,
            inplace=self.moe_runner_config.inplace,
            use_int4_w4a8=True,
            per_channel_quant=True,
            activation=activation,
            expert_map=getattr(layer, "expert_map", None),
            apply_router_weight_on_input=self.moe_runner_config.apply_router_weight_on_input,
            global_num_experts=layer.moe_runner_config.num_experts,
            w1_scale=(layer.w13_weight_scale),
            w2_scale=(layer.w2_weight_scale),
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            use_nn_moe=False,
            routed_scaling_factor=routed_scaling_factor,
            shared_output=shared_output,
            i_q=i_q,
            i_s=i_s,
        )

    def apply_ep(
        self,
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        use_nn_moe: Optional[bool] = False,
        num_local_tokens: Optional[torch.Tensor] = None,
        # config_select_bs: Optional[int] = None,
        routed_scaling_factor: Optional[float] = 1.0,
        shared_output: Optional[torch.Tensor] = None,
        # scales: Optional[torch.Tensor] = None,
        num_recv_tokens_per_expert: List = None,
        **_,
    ):
        workspace, global_reduce_buffer = MarlinMoeWorkspace(x.device).get_buffers()
        routed_scaling_factor = (
            1.0 if routed_scaling_factor is None else routed_scaling_factor
        )
        return self.fused_experts_impl_w4a8_marlin(
            x,
            w1,
            w2,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            workspace=workspace,
            global_reduce_buffer=global_reduce_buffer,
            inplace=self.moe_runner_config.inplace,
            use_int4_w4a8=True,
            per_channel_quant=True,
            activation=activation,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            use_nn_moe=use_nn_moe,
            shared_output=shared_output,
            routed_scaling_factor=float(routed_scaling_factor),
            # num_local_tokens=num_local_tokens,
            # config_select_bs=config_select_bs,
            # q_scales=scales
        )


class SlimQuantW4A8Int8TritonMoEMethod:
    """MoE method for W4A8INT8 Triton.

    Supports loading INT8 checkpoints with static weight scale and
    dynamic/static activation scale. Weights are used without Marlin repack;
    the GEMMs run through the lightop triton w4a8 fused-MoE kernels.
    Args:
        quant_config: The quantization config.
    """

    def __new__(cls, *args, **kwargs):

        if not hasattr(cls, "_initialized"):
            original_init = cls.__init__
            new_cls = type(
                cls.__name__,
                (FusedMoEMethodBase,),
                {
                    "__init__": original_init,
                    **{k: v for k, v in cls.__dict__.items() if k != "__dict__"},
                },
            )
            obj = super(new_cls, new_cls).__new__(new_cls)
            obj.__init__(*args, **kwargs)
            return obj
        return super().__new__(cls)

    def __init__(self, quant_config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import (
            FusedMoeWeightScaleSupported,
        )

        tp_size = get_tensor_model_parallel_world_size()
        intermediate_size = intermediate_size_per_partition
        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size, hidden_size // 2, dtype=torch.int8
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, hidden_size, intermediate_size // 2, dtype=torch.int8
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, 2 * intermediate_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w13_input_scale = None
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = None
        layer.register_parameter("w2_input_scale", w2_input_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight = Parameter(layer.w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(layer.w2_weight, requires_grad=False)
        layer.w13_weight_scale = Parameter(
            layer.w13_weight_scale.data, requires_grad=False
        )
        layer.w2_weight_scale = Parameter(
            layer.w2_weight_scale.data, requires_grad=False
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def _apply_triton(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        shared_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cache13 = get_moe_cache(
            topk_ids.shape[1],
            layer.w13_weight.shape[1],
            layer.w2_weight.shape[1],
            device=x.device,
            dtype=x.dtype,
        )
        routed_scaling_factor = (
            self.moe_runner_config.routed_scaling_factor
            if self.moe_runner_config.routed_scaling_factor is not None
            else 1.0
        )
        return fused_experts_impl_w4a8_triton(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            cache13,
            activation=activation,
            apply_router_weight_on_input=self.moe_runner_config.apply_router_weight_on_input,
            global_num_experts=self.moe_runner_config.num_experts,
            expert_map=getattr(layer, "expert_map", None),
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            routed_scaling_factor=routed_scaling_factor,
            shared_output=shared_output,
        )

    @torch._dynamo.disable()  # TODO: 性能优化需lmslim/lightop配合
    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output,
        i_q: Optional[torch.Tensor] = None,
        i_s: Optional[torch.Tensor] = None,
        # local_expert_mapping,
    ):
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

        topk_weights, topk_ids, _ = topk_output
        x, topk_weights = apply_topk_weights_cpu(
            self.moe_runner_config.apply_router_weight_on_input, topk_weights, x
        )
        output = self._apply_triton(
            layer,
            x,
            topk_weights,
            topk_ids,
            layer.moe_runner_config.activation,
            shared_output=None,
        )
        return StandardCombineInput(hidden_states=output)

    @torch._dynamo.disable()  # TODO: 性能优化需lmslim/lightop配合
    def apply_with_shared_output(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        activation: str = "silu",
        shared_output: Optional[torch.Tensor] = None,
        topk_output=None,
        i_q: Optional[torch.Tensor] = None,
        i_s: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        x, topk_weights = apply_topk_weights_cpu(
            self.moe_runner_config.apply_router_weight_on_input, topk_weights, x
        )
        if i_q is not None or i_s is not None:
            raise NotImplementedError(
                "pre-quantized activation input is not supported by the Triton W4A8 MoE path yet."
            )
        return self._apply_triton(
            layer,
            x,
            topk_weights,
            topk_ids,
            activation,
            shared_output=shared_output,
        )

    def apply_ep(
        self,
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        use_nn_moe: Optional[bool] = False,
        num_local_tokens: Optional[torch.Tensor] = None,
        # config_select_bs: Optional[int] = None,
        routed_scaling_factor: Optional[float] = 1.0,
        shared_output: Optional[torch.Tensor] = None,
        # scales: Optional[torch.Tensor] = None,
        num_recv_tokens_per_expert: List = None,
        **_,
    ):
        routed_scaling_factor = (
            1.0 if routed_scaling_factor is None else routed_scaling_factor
        )
        if shared_output is not None:
            raise NotImplementedError(
                "shared_output is not supported by apply_ep Triton W4A8 MoE path yet."
            )
        cache13 = get_moe_cache(
            topk_ids.shape[1],
            w1.shape[1],
            w2.shape[1],
            device=x.device,
            dtype=x.dtype,
        )
        return fused_experts_impl_w4a8_triton(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            cache13,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            routed_scaling_factor=float(routed_scaling_factor),
            num_local_tokens=num_local_tokens,
            shared_output=None,
        )


class SlimQuantW4A8Int8AiterMoEMethod:
    """MoE method for W4A8INT8 AITER."""

    def __new__(cls, *args, **kwargs):

        if not hasattr(cls, "_initialized"):
            original_init = cls.__init__
            new_cls = type(
                cls.__name__,
                (FusedMoEMethodBase,),
                {
                    "__init__": original_init,
                    **{k: v for k, v in cls.__dict__.items() if k != "__dict__"},
                },
            )
            obj = super(new_cls, new_cls).__new__(new_cls)
            obj.__init__(*args, **kwargs)
            return obj
        return super().__new__(cls)

    def __init__(self, quant_config):
        self.quant_config = quant_config
        self.use_deepep = get_moe_a2a_backend().is_deepep()
        self._ep_use_marlin = False
        _patch_moe_c_align_non_cub_to_triton()
        _patch_moe_c_m4_mode()

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import (
            FusedMoeWeightScaleSupported,
        )

        tp_size = get_tensor_model_parallel_world_size()
        intermediate_size = intermediate_size_per_partition
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size, hidden_size // 2, dtype=torch.int8
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, hidden_size, intermediate_size // 2, dtype=torch.int8
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, 2 * intermediate_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w13_input_scale = None
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = None
        layer.register_parameter("w2_input_scale", w2_input_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.use_deepep:
            # DeepEP grouped GEMM consumes the HIPC pack, not the Aiter TP
            # shuffle layout. The HIPC kernel applies the checkpoint's x16
            # internally, so the stored scale is passed through unchanged.
            from deepgemm import pack_w4a8_moe_hipc_weight

            layer.w13_weight = Parameter(
                pack_w4a8_moe_hipc_weight(layer.w13_weight.data),
                requires_grad=False,
            )
            layer.w2_weight = Parameter(
                pack_w4a8_moe_hipc_weight(layer.w2_weight.data),
                requires_grad=False,
            )
            scale_mul = 1.0
            layer.w13_weight_scale = Parameter(
                layer.w13_weight_scale.data * scale_mul, requires_grad=False
            )
            layer.w2_weight_scale = Parameter(
                layer.w2_weight_scale.data * scale_mul, requires_grad=False
            )
            self._ep_use_marlin = False
            return

        E = layer.w13_weight.shape[0]
        layer.w13_weight = Parameter(
            repack_and_shuffle_w4a8(layer.w13_weight.data, E), requires_grad=False
        )
        layer.w2_weight = Parameter(
            repack_and_shuffle_w4a8(layer.w2_weight.data, E), requires_grad=False
        )
        self._ep_use_marlin = False

        # Checkpoint already stores scale/16 (Triton restores with *16).
        # moe_c applies 16.0f * scale internally, so pass the stored scale as-is.
        # An extra /16 here collapses MoE (immediate EOS); do not divide again.
        layer.w13_weight_scale = Parameter(
            layer.w13_weight_scale.data, requires_grad=False
        )
        layer.w2_weight_scale = Parameter(
            layer.w2_weight_scale.data, requires_grad=False
        )
        if not getattr(SlimQuantW4A8Int8AiterMoEMethod, "_logged_aiter_scale", False):
            logger.info(
                "[slimquant_w4a8_marlin] aiter moe_c scale as-is "
                "w13_mean=%.6g w2_mean=%.6g",
                float(layer.w13_weight_scale.float().mean()),
                float(layer.w2_weight_scale.float().mean()),
            )
            SlimQuantW4A8Int8AiterMoEMethod._logged_aiter_scale = True

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    @torch._dynamo.disable()
    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output,
    ):
        from aiter.moe import MoeQuantType, aiter_moe, get_aiter_moe_config

        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_weights, topk_ids, _ = dispatch_output.topk_output
        # x, topk_weights = apply_topk_weights_cpu(
        #     self.moe_runner_config.apply_router_weight_on_input, topk_weights, x
        # )
        if x.shape[0] == 0:
            return StandardCombineInput(hidden_states=x)

        e = layer.w13_weight.size(0)
        k = x.size(-1)
        n1 = layer.w13_weight.size(1)
        n2 = layer.w2_weight.size(1)
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(torch.float32)
        from sglang.srt.distributed import get_moe_expert_parallel_world_size
        from sglang.srt.layers.moe.utils import get_moe_runner_backend

        # moe_c GEMM zeros remote blocks when expert_id==-1. Cub align cannot
        # index -1, so:
        # - Triton dispatcher already remapped to local ids with -1 drops:
        #   retarget drops to extra expert `e`, then map that id back to -1.
        # - AITER dispatcher keeps global ids: pass the real EP expert_map.
        ep_size = get_moe_expert_parallel_world_size()
        ids_are_local = not get_moe_runner_backend().is_aiter()
        if ep_size > 1 and ids_are_local:
            drop_id = e
            invalid = (topk_ids < 0) | (topk_ids >= e)
            topk_ids = topk_ids.masked_fill(invalid, drop_id)
            expert_map = self._drop_sentinel_expert_map(e, topk_ids.device)
            global_num_experts = e + 1
        elif ep_size > 1:
            global_E = self.moe_runner_config.num_experts or e
            expert_map = self._get_ep_expert_map(int(global_E), topk_ids.device)
            global_num_experts = int(global_E)
        else:
            expert_map = None
            global_num_experts = e
        topk = topk_ids.size(1)

        if x.dim() == 2:
            m = x.size(0)
        else:
            assert x.dim() == 3
            assert x.size(0) == e, f"{x.size(0)} == {e}"
            m = x.size(1)

        status, moe_config = _lookup_aiter_w4a8_config(
            **_aiter_w4a8_config_kwargs(
                M=m,
                E=e,
                N1=n1,
                N2=n2,
                K=k,
                top_k=topk,
                block_size=None,
                dtype=x.dtype,
                quant_type=MoeQuantType.W4A8,
            )
        )
        if not status:
            raise RuntimeError(
                "aiter backend did not find a valid w4a8 tpmoe config for "
                f"M={m}, E={e}, N1={n1}, N2={n2}, K={k}, topk={topk}, "
                f"dtype={x.dtype}."
            )
        if not getattr(self, "_logged_aiter_sol", False):
            logger.info(
                "[slimquant_w4a8_marlin] aiter moe solution_type=%s "
                "E=%s global_E=%s N1=%s N2=%s K=%s topk=%s expert_map=%s "
                "align_patch=%s",
                getattr(moe_config, "solution_type", None),
                e,
                global_num_experts,
                n1,
                n2,
                k,
                topk,
                None if expert_map is None else tuple(expert_map.shape),
                _moe_c_align_patched,
            )
            self._logged_aiter_sol = True

        output = aiter_moe(
            x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            moe_config=moe_config,
            activation="silu",
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            routed_scaling_factor=self.moe_runner_config.routed_scaling_factor,
        )
        return StandardCombineInput(hidden_states=output)

    def _drop_sentinel_expert_map(
        self,
        local_e: int,
        device: torch.device,
    ) -> torch.Tensor:
        cached = getattr(self, "_drop_sentinel_map", None)
        expected = local_e + 1
        if (
            cached is not None
            and cached.numel() == expected
            and cached.device == device
            and cached.dtype == torch.int32
        ):
            return cached
        expert_map = torch.arange(expected, device=device, dtype=torch.int32)
        expert_map[local_e] = -1
        self._drop_sentinel_map = expert_map
        return expert_map

    def _get_ep_expert_map(
        self,
        global_num_experts: int,
        device: torch.device,
    ) -> torch.Tensor:
        cached = getattr(self, "_ep_expert_map", None)
        if (
            cached is not None
            and cached.numel() == global_num_experts
            and cached.device == device
        ):
            return cached
        from sglang.srt.distributed import (
            get_moe_expert_parallel_rank,
            get_moe_expert_parallel_world_size,
        )
        from sglang.srt.layers.moe.fused_moe_triton.layer import determine_expert_map

        _, expert_map = determine_expert_map(
            get_moe_expert_parallel_world_size(),
            get_moe_expert_parallel_rank(),
            global_num_experts,
        )
        expert_map = expert_map.to(device=device, dtype=torch.int32)
        self._ep_expert_map = expert_map
        return expert_map

    @torch._dynamo.disable()
    def apply_ep(
        self,
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        use_nn_moe: Optional[bool] = False,
        num_local_tokens: Optional[torch.Tensor] = None,
        routed_scaling_factor: Optional[float] = 1.0,
        shared_output: Optional[torch.Tensor] = None,
        num_recv_tokens_per_expert: List = None,
        **_,
    ):
        if getattr(self, "_ep_use_marlin", False):
            # Weights were packed into the lightop Marlin layout. Honor the
            # DeepEP dummy-global-id ABI the same way SlimQuantW4A8Int8MarlinMoEMethod does.
            _ensure_lightop_w4a8_marlin_available()
            workspace, global_reduce_buffer = MarlinMoeWorkspace(x.device).get_buffers()
            routed_scaling_factor = (
                1.0 if routed_scaling_factor is None else routed_scaling_factor
            )
            return fused_experts_impl_w4a8_marlin(
                x,
                w1,
                w2,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                workspace=workspace,
                global_reduce_buffer=global_reduce_buffer,
                inplace=self.moe_runner_config.inplace,
                use_int4_w4a8=True,
                per_channel_quant=True,
                activation=activation,
                expert_map=expert_map,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                use_nn_moe=use_nn_moe,
                shared_output=shared_output,
                routed_scaling_factor=float(routed_scaling_factor),
            )

        # Non-DeepEP / TP path: aiter shuffled [E, N, K/2] weights.
        _ensure_aiter_w4a8_marlin_available()
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "apply_router_weight_on_input is not supported by AITER W4A8 apply_ep."
            )
        if shared_output is not None:
            raise NotImplementedError(
                "shared_output is not supported by AITER W4A8 apply_ep."
            )
        if x.shape[0] == 0:
            return x.bfloat16() if x.dtype != torch.bfloat16 else x

        e = w1.size(0)
        k = x.size(-1)
        n1 = w1.size(1)
        n2 = n1 // 2
        topk_ids = topk_ids.to(torch.int32)
        topk = topk_ids.size(1)
        if x.dim() == 2:
            m = x.size(0)
        else:
            assert x.dim() == 3
            assert x.size(0) == e, f"{x.size(0)} == {e}"
            m = x.size(1)
        orig_m = m
        scatter_idx = None

        if global_num_experts is None or global_num_experts < 0:
            global_num_experts = e
        if global_num_experts > e:
            from sglang.srt.distributed import (
                get_moe_expert_parallel_rank,
                get_moe_expert_parallel_world_size,
            )

            ep_rank = get_moe_expert_parallel_rank()
            rank_offset = ep_rank * (
                global_num_experts // get_moe_expert_parallel_world_size()
            )
            dummy = global_num_experts - 1 if ep_rank == 0 else 0
            local_ids = topk_ids - rank_offset
            invalid = (
                (topk_ids == dummy)
                | (topk_ids < 0)
                | (local_ids < 0)
                | (local_ids >= e)
            )
            valid = ~invalid
            if not valid.any():
                return torch.zeros(
                    orig_m, k, device=x.device, dtype=torch.bfloat16
                )
            # DeepEP pads unused top-k slots with a dummy global id. aiter_moe
            # has no skip mask, so compact to valid (token, expert) pairs with
            # topk=1 and scatter-add. Dummy traffic must not run as expert 0.
            scatter_idx, _ = torch.where(valid)
            x = x.index_select(0, scatter_idx)
            if a1_scale is not None:
                scale = a1_scale
                if scale.dim() == 1:
                    scale = scale.unsqueeze(-1)
                a1_scale = scale.index_select(0, scatter_idx)
            topk_ids = local_ids[valid].to(torch.int32).view(-1, 1)
            topk_weights = topk_weights[valid].view(-1, 1)
            m = x.size(0)
            topk = 1
            global_num_experts = e
            expert_map = None

        x = x.contiguous()
        topk_ids = topk_ids.contiguous()
        topk_weights = topk_weights.contiguous()
        if a1_scale is not None:
            a1_scale = a1_scale.contiguous()

        if x.dtype not in (torch.float16, torch.bfloat16):
            if a1_scale is None:
                raise RuntimeError(
                    "AITER W4A8 apply_ep received non-floating activations "
                    f"({x.dtype}) without a1_scale."
                )
            # Keep INT8 + per-token scale. Dequantizing to BF16 made aiter
            # pick the wrong W4A8 kernel and produced garbage logits.
            output_dtype = torch.bfloat16
            config_dtype = x.dtype
        else:
            output_dtype = x.dtype
            config_dtype = x.dtype
        status, moe_config = _lookup_aiter_w4a8_config(
            **_aiter_w4a8_config_kwargs(
                M=m,
                E=e,
                N1=n1,
                N2=n2,
                K=k,
                top_k=topk,
                block_size=None,
                dtype=config_dtype,
                quant_type=MoeQuantType.W4A8,
                activation=activation,
            )
        )
        if not status:
            raise RuntimeError(
                "aiter backend did not find a valid w4a8 tpmoe config for "
                f"M={m}, E={e}, N1={n1}, N2={n2}, K={k}, topk={topk}, "
                f"dtype={output_dtype}, activation={activation}."
            )

        routed_scaling_factor = (
            1.0 if routed_scaling_factor is None else routed_scaling_factor
        )
        output = aiter_moe(
            x,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights.to(torch.float32),
            topk_ids=topk_ids,
            moe_config=moe_config,
            activation=activation,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            routed_scaling_factor=float(routed_scaling_factor),
            output_dtype=output_dtype,
        )
        if scatter_idx is None:
            return output
        combined = torch.zeros(
            orig_m, k, device=output.device, dtype=output.dtype
        )
        combined.index_add_(0, scatter_idx, output)
        return combined
