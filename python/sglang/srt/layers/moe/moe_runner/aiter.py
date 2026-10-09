from __future__ import annotations

import csv
import functools
import inspect
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Union

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import get_bool_env_var, get_int_env_var

logger = logging.getLogger(__name__)

_AITER_UNIFIED_MOE_CONFIG_CACHE: dict[tuple, Any] = {}
_AITER_UNIFIED_MOE_FALLBACK_WARNINGS: set[tuple] = set()


@functools.cache
def _aiter_asm_has_exact_shape(
    arch: str,
    experts: int,
    intermediate_size: int,
    model_dim: int,
    top_k: int,
    m: int,
    input_dtype: str,
    csv_path: Optional[str] = None,
) -> bool:
    """Check that compact LL uses an exact row from AITER's ASM table."""
    if csv_path is None:
        from aiter.fused_moe_asm_wna16 import get_csv_path

        csv_path = get_csv_path("f8_w8a8_channel")
    path = os.path.abspath(csv_path)
    if not os.path.isfile(path):
        return False
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if (
                    row.get("arch") == arch
                    and row.get("quant_type") == "f8_w8a8_channel"
                    and row.get("indtype") == input_dtype
                    and int(row["token"]) == m
                    and int(row["inter_dim"]) == intermediate_size
                    and int(row["model_dim"]) == model_dim
                    and int(row["expert"]) == experts
                    and int(row["topk"]) == top_k
                ):
                    return True
    except (OSError, ValueError, KeyError):
        return False
    return False


@functools.cache
def _aiter_moe_c_has_exact_m(
    arch: str,
    experts: int,
    intermediate_size: int,
    quant_type: str,
    m: int,
    config_root: Optional[str] = None,
) -> Optional[bool]:
    """Return exact-M coverage when an installed AITER MoE-C table exists."""
    if config_root is None:
        from aiter.fused_moe_c import _moe_c_config_root

        config_root = _moe_c_config_root()
    category = "fp8_w8a8" if quant_type == "fp8_w8a8" else quant_type
    directory = os.path.join(config_root, arch, category)
    base = f"E={experts},N={intermediate_size},dtype={category}"
    top_path = os.path.join(directory, f"{base}.json")
    bottom_path = os.path.join(directory, f"{base},is_bottom=True.json")
    if not os.path.isfile(top_path) and not os.path.isfile(bottom_path):
        return None
    if not os.path.isfile(top_path) or not os.path.isfile(bottom_path):
        return False
    try:
        with open(top_path, encoding="utf-8") as file:
            top = json.load(file)
        with open(bottom_path, encoding="utf-8") as file:
            bottom = json.load(file)
    except (OSError, ValueError):
        return False
    key = str(m)
    return key in top and key in bottom


if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.base import CombineInput
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLDispatchOutput,
        DeepEPNormalDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.moriep import (
        MoriEPLLDispatchOutput,
        MoriEPNormalDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


class AiterQuantType(str, Enum):
    NONE = "No"
    PER_TOKEN = "per_Token"
    PER_128X128 = "per_128x128"
    PER_1X32 = "per_1x32"


@dataclass
class AiterMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    quant_type: AiterQuantType = AiterQuantType.NONE
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    a13_scale: Optional[torch.Tensor] = None
    a2_scale: Optional[torch.Tensor] = None
    b13: Optional[torch.Tensor] = None
    b2: Optional[torch.Tensor] = None
    expert_mask: Optional[torch.Tensor] = None
    doweight_stage1: bool = False
    hidden_pad: int = 0
    intermediate_pad: int = 0
    swiglu_limit: float = 0.0
    # PER_TOKEN alone cannot distinguish channel-INT8 from channel-FP8.
    use_int8_w8a8: bool = False
    use_fp8_w8a8: bool = False
    w8a8_no_shuffle: bool = False
    fused_moe_kwargs: Optional[dict[str, Any]] = None


def _prepare_aiter_w8a8_weights(layer: torch.nn.Module) -> None:
    """Keep checkpoint-layout weights; the unified runner owns any shuffle."""
    config = getattr(layer, "moe_runner_config", None)
    if getattr(layer, "apply_router_weight_on_input", False) or (
        config is not None and config.apply_router_weight_on_input
    ):
        raise RuntimeError(
            "AITER channel-W8A8 does not support apply_router_weight_on_input=True"
        )


def process_weights_after_loading_aiter_w8a8_int8(layer: torch.nn.Module) -> None:
    _prepare_aiter_w8a8_weights(layer)


def process_weights_after_loading_aiter_w8a8_fp8(layer: torch.nn.Module) -> None:
    _prepare_aiter_w8a8_weights(layer)


def _get_aiter_w8a8_quant_info(
    layer: torch.nn.Module, *, use_fp8: bool
) -> AiterMoeQuantInfo:
    dispatcher = getattr(layer, "dispatcher", None)
    expert_map = (
        getattr(dispatcher, "local_expert_mapping", None)
        if getattr(dispatcher, "expert_mask_gpu", None) is not None
        else None
    )
    if not use_fp8 and (
        layer.w13_weight.dtype != torch.int8 or layer.w2_weight.dtype != torch.int8
    ):
        raise ValueError("AITER INT8 W8A8 requires torch.int8 expert weights")
    return AiterMoeQuantInfo(
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        quant_type=AiterQuantType.PER_TOKEN,
        w13_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        a13_scale=getattr(layer, "w13_input_scale", None),
        a2_scale=getattr(layer, "w2_input_scale", None),
        expert_mask=expert_map,
        use_int8_w8a8=not use_fp8,
        use_fp8_w8a8=use_fp8,
        w8a8_no_shuffle=bool(
            not use_fp8 and getattr(layer, "_sglang_hcu_aiter_int8_noshuffle", False)
        ),
    )


def get_aiter_w8a8_int8_quant_info(layer: torch.nn.Module) -> AiterMoeQuantInfo:
    return _get_aiter_w8a8_quant_info(layer, use_fp8=False)


def get_aiter_w8a8_fp8_quant_info(layer: torch.nn.Module) -> AiterMoeQuantInfo:
    return _get_aiter_w8a8_quant_info(layer, use_fp8=True)


@dataclass
class AiterRunnerInput(RunnerInput):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor  # int32
    topk_weights: torch.Tensor  # float32
    # Effective activation quant_type (may differ from quant_info.quant_type
    # after the dispatch-aware decision in mori pre_permute).
    quant_type: AiterQuantType
    # Per-token activation scale produced by an EP dispatcher (mori). Falls
    # back to quant_info.a13_scale when None.
    a1_scale: Optional[torch.Tensor] = None
    # Mori-only fused_moe kwargs.
    num_local_tokens: Optional[torch.Tensor] = None
    output_dtype: Optional[torch.dtype] = None
    # True when a dispatcher adapter has already converted global routing to
    # dense local expert ids.  In that case the unified API must not apply the
    # dispatcher's global->local expert metadata a second time.
    uses_local_expert_ids: bool = False
    # ``None`` normally means that AITER may fall back to the quant method's
    # static activation scale.  DeepEP-LL compaction has already dequantized
    # FP8 input to BF16, so that fallback must be suppressed explicitly.
    input_is_dequantized: bool = False

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.AITER


@dataclass
class AiterRunnerOutput(RunnerOutput):
    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.AITER


_AITER_ACTIVATIONS = {
    "silu": "Silu",
    "swiglu": "Swiglu",
    "situ": "Situv2",
}


def _aiter_activation(activation: str):
    from aiter import ActivationType

    return getattr(ActivationType, _AITER_ACTIVATIONS.get(activation, "Gelu"))


def _aiter_quant_type(quant_type: AiterQuantType):
    from aiter import QuantType

    return getattr(QuantType, quant_type.value)


@functools.lru_cache(maxsize=None)
def _aiter_moe_supports_gemm1_activation_params(aiter_moe_fn: Any) -> bool:
    """Return whether an AITER MoE API accepts MiniMax alpha/limit."""
    try:
        parameters = inspect.signature(aiter_moe_fn).parameters
    except (TypeError, ValueError):
        return False
    return "gemm1_alpha" in parameters and "gemm1_limit" in parameters


def get_aiter_moe_activation_kwargs(
    runner_config: MoeRunnerConfig,
    aiter_moe_fn: Any,
) -> dict[str, float]:
    """Build optional activation arguments for an AITER MoE API.

    Ordinary models return an empty dictionary and retain AITER's default
    activation behavior. Models such as MiniMax-M3 set gemm1_alpha and
    gemm1_clamp_limit to request the GPT-OSS-style SwiGLU variant.
    """
    alpha = runner_config.gemm1_alpha
    limit = runner_config.gemm1_clamp_limit
    if alpha is None and limit is None:
        return {}

    if not _aiter_moe_supports_gemm1_activation_params(aiter_moe_fn):
        raise RuntimeError(
            "The selected model requires AITER gemm1_alpha/gemm1_limit support. "
            "Please upgrade AITER or use another MoE runner backend."
        )

    if alpha is not None:
        if limit is None:
            raise ValueError("gemm1_clamp_limit must be set when gemm1_alpha is set")
        if not runner_config.is_gated:
            raise ValueError("AITER gemm1_alpha requires a gated MoE activation")
        if runner_config.gemm1_beta not in (None, 1.0):
            raise ValueError("AITER gemm1_alpha currently supports only gemm1_beta=1.0")

        activation = str(runner_config.activation)
        if activation == "silu":
            if runner_config.gate_up_interleaved:
                raise ValueError(
                    "AITER activation='silu' with gemm1_alpha expects split "
                    "[gate..., up...] layout; set gate_up_interleaved=False"
                )
        elif activation == "swigluoai":
            if not runner_config.gate_up_interleaved:
                raise ValueError(
                    "AITER activation='swigluoai' expects interleaved "
                    "[gate0, up0, ...] layout"
                )
        else:
            raise ValueError(
                "AITER gemm1_alpha is supported only for activation='silu' "
                "(split layout) or activation='swigluoai' (interleaved layout)"
            )

    kwargs = {}
    if alpha is not None:
        kwargs["gemm1_alpha"] = float(alpha)
    if limit is not None:
        kwargs["gemm1_limit"] = float(limit)
    return kwargs


@functools.cache
def _aiter_fused_moe_supports_no_combine() -> bool:
    """Probe whether the installed aiter.fused_moe accepts a `no_combine` kwarg.

    Older wheels don't expose it, so feature-detect once and forward
    conditionally, matching the existing `**extra` conditional-kwarg pattern
    used for `num_local_tokens` / `dtype`.
    """
    from aiter.fused_moe import fused_moe

    return "no_combine" in inspect.signature(fused_moe).parameters


class AiterRunnerCore(MoeRunnerCore):
    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)
        self._unified_config_cache = _AITER_UNIFIED_MOE_CONFIG_CACHE
        self._unified_weight_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self._unified_scale_cache: dict[tuple, torch.Tensor] = {}
        # Dispatcher expert metadata is immutable for a loaded MoE layer.
        # Cache the unified API's normalized map so DeepEP's bool sink mask
        # does not allocate/rebuild one tensor per layer on every decode step.
        # Keep the source tensor in the value to prevent Python id reuse.
        self._unified_expert_map_cache: dict[
            tuple[int, int], tuple[torch.Tensor, int, torch.Tensor]
        ] = {}
        # AITER backends do not agree on the EP metadata contract: Triton and
        # MoE-C consume a global->local map, whereas the native ASM sorter
        # consumes a 0/1 global membership mask. Cache the latter separately
        # so conversion is not repeated on every forward pass.
        self._unified_asm_expert_mask_cache: dict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._warned_unified_fallbacks = _AITER_UNIFIED_MOE_FALLBACK_WARNINGS

    def _requires_unified_moe(self) -> bool:
        """Use the new API only for models needing its activation epilogue.

        Keeping ordinary models on ``aiter.fused_moe`` avoids changing their
        dispatcher-specific behavior while MiniMax-M3 gets alpha/limit support.
        """
        return self.config.activation != "situ" and (
            self.config.gemm1_alpha is not None
            or self.config.gemm1_clamp_limit is not None
        )

    @staticmethod
    def _unified_quant_params(
        quant_type: AiterQuantType,
    ) -> tuple[str, int, Optional[list[int]]]:
        from aiter.moe import MoeQuantType

        if quant_type == AiterQuantType.NONE:
            return MoeQuantType.W16A16, 0, None
        if quant_type == AiterQuantType.PER_TOKEN:
            # FP8 weights, dynamic per-token activations, channel-wise scales.
            return MoeQuantType.FP8_W8A8, 0, None
        if quant_type == AiterQuantType.PER_128X128:
            return MoeQuantType.FP8_W8A8, 128, [128, 128]
        raise NotImplementedError(
            "aiter.moe.aiter_moe does not support SGLang quant_type="
            f"{quant_type.value} for the MiniMax activation path"
        )

    def _get_unified_moe_config(
        self,
        runner_input: AiterRunnerInput,
        quant_info: AiterMoeQuantInfo,
        quant_type: str,
        block_size: int,
    ) -> Any:
        from aiter.moe import (
            AiterMoeConfig,
            MoeSolutionType,
            get_aiter_moe_config,
        )

        from sglang.srt.environ import envs

        w1 = quant_info.w13_weight
        w2 = quant_info.w2_weight
        top_k = runner_input.topk_ids.shape[-1]
        force_moe_c = envs.SGLANG_FORCE_AITER_MOE_C.get()
        cache_key = (
            runner_input.hidden_states.shape[0],
            w1.shape[0],
            w1.shape[1],
            w2.shape[1],
            w1.shape[2],
            top_k,
            block_size,
            runner_input.hidden_states.dtype,
            quant_type,
            self.config.activation,
            self.config.is_gated,
            runner_input.uses_local_expert_ids,
            force_moe_c,
            quant_info.w8a8_no_shuffle,
        )
        cached = self._unified_config_cache.get(cache_key)
        if cached is not None:
            return cached

        config_kwargs = dict(
            M=runner_input.hidden_states.shape[0],
            E=w1.shape[0],
            N1=w1.shape[1],
            N2=w2.shape[1],
            K=w1.shape[2],
            top_k=top_k,
            block_size=block_size,
            dtype=runner_input.hidden_states.dtype,
            quant_type=quant_type,
            activation=self.config.activation,
            gated=self.config.is_gated,
        )
        if force_moe_c:
            config_kwargs["spec_sol_type"] = MoeSolutionType.MOE_C
        if quant_info.w8a8_no_shuffle:
            # Match the scheme's probe: no-shuffle ASM, not a default
            # shuffled MoE-C config for the same dimensions.
            config_kwargs["spec_sol_type"] = MoeSolutionType.ASM
            config_kwargs["use_shuffle"] = 0
        status, moe_config = get_aiter_moe_config(**config_kwargs)
        if quant_info.w8a8_no_shuffle and status:
            status = (
                moe_config.solution_type == MoeSolutionType.ASM
                and not moe_config.need_shuffle
            )
        if runner_input.uses_local_expert_ids:
            from aiter.jit.utils.chip_info import get_gfx

            exact_compact_asm = (
                status
                and moe_config.solution_type == MoeSolutionType.ASM
                and _aiter_asm_has_exact_shape(
                    get_gfx(),
                    w1.shape[0],
                    w1.shape[1] // 2 if self.config.is_gated else w1.shape[1],
                    w2.shape[1],
                    top_k,
                    runner_input.hidden_states.shape[0],
                    str(runner_input.hidden_states.dtype),
                )
            )
        else:
            exact_compact_asm = True
        if not exact_compact_asm:
            # The compact DeepEP-LL representation uses local TopK=1 ids.
            # Only exact installed ASM rows are admitted: generic Triton is
            # the safe fallback, while MoE-C's nearest-M lookup and shuffled
            # weights have not passed this layout's service-level gate.
            status = False
        if status and moe_config.solution_type == MoeSolutionType.MOE_C:
            from aiter.jit.utils.chip_info import get_gfx

            exact_coverage = _aiter_moe_c_has_exact_m(
                get_gfx(),
                w1.shape[0],
                w1.shape[1] // 2 if self.config.is_gated else w1.shape[1],
                quant_type,
                runner_input.hidden_states.shape[0],
            )
            if exact_coverage is False:
                logger.warning(
                    "Installed AITER MoE-C config has no exact validated "
                    "entry for M=%s, E=%s, N=%s, quant=%s; rejecting nearest-M "
                    "MoE-C reuse and falling back to Triton.",
                    runner_input.hidden_states.shape[0],
                    w1.shape[0],
                    w1.shape[1] // 2 if self.config.is_gated else w1.shape[1],
                    quant_type,
                )
                status = False
        if not status:
            # The direct Triton path has safe heuristics even when the installed
            # tuned-config table has no MiniMax-M3/BW1100 entry.
            moe_config = AiterMoeConfig(
                quant_type=quant_type,
                solution_type=MoeSolutionType.TRITON,
                config={},
                need_shuffle=False,
            )
            warning_key = cache_key[1:9]
            if warning_key not in self._warned_unified_fallbacks:
                logger.warning(
                    "No tuned AITER MoE config for E=%s, N1=%s, N2=%s, "
                    "K=%s, top_k=%s, dtype=%s, quant=%s; falling back to "
                    "the unified AITER Triton implementation.",
                    w1.shape[0],
                    w1.shape[1],
                    w2.shape[1],
                    w1.shape[2],
                    top_k,
                    runner_input.hidden_states.dtype,
                    quant_type,
                )
                self._warned_unified_fallbacks.add(warning_key)
        else:
            logger.info(
                "AITER unified MoE selected: M=%s, E=%s, N1=%s, N2=%s, "
                "K=%s, top_k=%s, dtype=%s, quant=%s, solution=%s, "
                "need_shuffle=%s",
                runner_input.hidden_states.shape[0],
                w1.shape[0],
                w1.shape[1],
                w2.shape[1],
                w1.shape[2],
                top_k,
                runner_input.hidden_states.dtype,
                quant_type,
                moe_config.solution_type,
                moe_config.need_shuffle,
            )

        self._unified_config_cache[cache_key] = moe_config
        return moe_config

    def _get_unified_weights(
        self,
        quant_info: AiterMoeQuantInfo,
        moe_config: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w1 = quant_info.w13_weight
        w2 = quant_info.w2_weight
        if not moe_config.need_shuffle:
            return w1, w2

        cache_key = (
            id(w1),
            id(w2),
            moe_config.solution_type,
            moe_config.quant_type,
        )
        cached = self._unified_weight_cache.get(cache_key)
        if cached is not None:
            return cached

        from aiter.moe import aiter_moe_shfl_weight

        with torch.no_grad():
            if self._uses_moe_c_oai_interleaved_layout(moe_config):
                # The installed channel-FP8 MoE-C GEMM expects its generic
                # GEMM2 shuffle for both stages, while the swigluoai epilogue
                # consumes interleaved [gate0, up0, gate1, up1, ...] rows.
                # MiniMax stores split [gate..., up...] rows, so reorder W1
                # once before AITER's normal shuffle. W2 is unchanged.
                w1 = self._interleave_split_gate_up(w1)
                shuffled = aiter_moe_shfl_weight(w1, w2, moe_config)
            else:
                shuffled = aiter_moe_shfl_weight(w1, w2, moe_config)

        assert shuffled[0] is not None and shuffled[1] is not None
        weights = (shuffled[0], shuffled[1])
        self._unified_weight_cache[cache_key] = weights
        return weights

    @staticmethod
    def _interleave_split_gate_up(tensor: torch.Tensor) -> torch.Tensor:
        """Convert axis-1 [gate..., up...] data to [gate0, up0, ...]."""
        if tensor.ndim < 2 or tensor.shape[1] % 2:
            raise ValueError(
                "MiniMax MoE-C gate/up tensor must have an even axis-1, got "
                f"shape={tuple(tensor.shape)}"
            )
        half = tensor.shape[1] // 2
        tail = tensor.shape[2:]
        order = (0, 2, 1, *range(3, tensor.ndim + 1))
        return (
            tensor.reshape(tensor.shape[0], 2, half, *tail)
            .permute(order)
            .contiguous()
            .reshape_as(tensor)
        )

    def _uses_moe_c_oai_interleaved_layout(self, moe_config: Any) -> bool:
        return (
            moe_config.solution_type == "moe_c"
            and moe_config.quant_type in ("int8_w8a8", "fp8_w8a8")
            and self.config.gemm1_alpha is not None
            and self.config.gemm1_clamp_limit is not None
            and not self.config.gate_up_interleaved
        )

    def _get_unified_w1_scale(
        self,
        quant_info: AiterMoeQuantInfo,
        moe_config: Any,
    ) -> Optional[torch.Tensor]:
        scale = quant_info.w13_scale
        if scale is None or not self._uses_moe_c_oai_interleaved_layout(moe_config):
            return scale
        cache_key = (id(scale), moe_config.solution_type, moe_config.quant_type)
        cached = self._unified_scale_cache.get(cache_key)
        if cached is not None:
            return cached
        with torch.no_grad():
            interleaved = self._interleave_split_gate_up(scale)
        self._unified_scale_cache[cache_key] = interleaved
        return interleaved

    def _build_expert_map(
        self,
        expert_mask: Optional[torch.Tensor],
        num_local_experts: int,
    ) -> tuple[int, Optional[torch.Tensor]]:
        if expert_mask is None:
            return -1, None
        if expert_mask.ndim != 1:
            raise ValueError(
                "AITER expert metadata must be rank-1, got "
                f"shape={tuple(expert_mask.shape)}"
            )

        cache_key = (id(expert_mask), num_local_experts)
        cached = self._unified_expert_map_cache.get(cache_key)
        if cached is not None and cached[0] is expert_mask:
            return cached[1], cached[2]

        if expert_mask.dtype != torch.bool:
            if expert_mask.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                raise TypeError(
                    "AITER integer expert_map expected, got "
                    f"dtype={expert_mask.dtype}"
                )
            expert_map = (
                expert_mask
                if expert_mask.dtype == torch.int32
                else expert_mask.to(dtype=torch.int32)
            )
            # DeepEP's legacy dispatcher metadata is an unambiguous local
            # membership mask only in the [num_local_experts + sink] form.
            # Accept that exact shape while continuing to reject arbitrary
            # global 0/1 masks, which cannot encode a global->local mapping.
            if (
                expert_map.numel() == num_local_experts + 1
                and bool(torch.all((expert_map == 0) | (expert_map == 1)).item())
                and bool(torch.all(expert_map[:-1] == 1).item())
                and int(expert_map[-1].item()) == 0
            ):
                local_map = torch.full_like(expert_map, -1)
                local_map[:-1] = torch.arange(
                    num_local_experts,
                    dtype=torch.int32,
                    device=expert_map.device,
                )
                result = (expert_map.numel(), local_map)
                self._unified_expert_map_cache[cache_key] = (
                    expert_mask,
                    result[0],
                    result[1],
                )
                return result
            # A valid global->local map contains every local expert exactly
            # once and uses -1 for every remote expert. This deliberately
            # rejects legacy 0/1 membership masks, the source of a hard HCU VM
            # fault when E=128 metadata was passed with only E=16 weights.
            if bool(torch.any(expert_map < -1).item()):
                raise ValueError("AITER expert_map may only use -1 for remote experts")
            local_ids = expert_map[expert_map >= 0]
            expected = torch.arange(
                num_local_experts,
                dtype=torch.int32,
                device=expert_map.device,
            )
            if local_ids.numel() != num_local_experts or not bool(
                torch.equal(torch.sort(local_ids).values, expected)
            ):
                raise ValueError(
                    "AITER expert_map must map each local expert exactly once; "
                    f"local_entries={local_ids.numel()}, expected={num_local_experts}. "
                    "A 0/1 membership mask is not a valid unified-AITER map."
                )
            result = (expert_map.numel(), expert_map)
            self._unified_expert_map_cache[cache_key] = (
                expert_mask,
                result[0],
                result[1],
            )
            return result

        expert_map = torch.full(
            expert_mask.shape,
            -1,
            dtype=torch.int32,
            device=expert_mask.device,
        )
        local_experts = torch.nonzero(expert_mask, as_tuple=False).flatten()
        expert_map[local_experts] = torch.arange(
            local_experts.numel(), dtype=torch.int32, device=expert_mask.device
        )
        if local_experts.numel() != num_local_experts:
            raise ValueError(
                "AITER bool expert mask selects "
                f"{local_experts.numel()} experts, expected {num_local_experts}"
            )
        result = (expert_mask.numel(), expert_map)
        self._unified_expert_map_cache[cache_key] = (
            expert_mask,
            result[0],
            result[1],
        )
        return result

    def _run_unified_moe(
        self,
        runner_input: AiterRunnerInput,
        quant_info: AiterMoeQuantInfo,
    ) -> AiterRunnerOutput:
        from aiter.moe import MoeSolutionType, aiter_moe

        if self.config.no_combine:
            raise NotImplementedError(
                "aiter.moe.aiter_moe does not expose no_combine output"
            )
        if quant_info.b13 is not None or quant_info.b2 is not None:
            raise NotImplementedError(
                "aiter.moe.aiter_moe does not expose expert bias inputs"
            )
        if quant_info.doweight_stage1:
            raise NotImplementedError(
                "aiter.moe.aiter_moe does not expose doweight_stage1"
            )
        if quant_info.fused_moe_kwargs:
            raise NotImplementedError(
                "Extra legacy fused_moe kwargs cannot be passed to aiter_moe"
            )
        if quant_info.hidden_pad or quant_info.intermediate_pad:
            raise NotImplementedError(
                "Explicit hidden/intermediate padding is unsupported by aiter_moe"
            )

        quant_type, block_size, block_shape = self._unified_quant_params(
            runner_input.quant_type
        )
        if quant_info.use_int8_w8a8:
            from aiter.moe import MoeQuantType

            quant_type, block_size, block_shape = MoeQuantType.W8A8, 0, None
        moe_config = self._get_unified_moe_config(
            runner_input, quant_info, quant_type, block_size
        )
        w1, w2 = self._get_unified_weights(quant_info, moe_config)
        if runner_input.uses_local_expert_ids:
            global_num_experts, expert_map = -1, None
        else:
            global_num_experts, expert_map = self._build_expert_map(
                quant_info.expert_mask,
                w1.shape[0],
            )
        backend_expert_metadata = expert_map
        if expert_map is not None and moe_config.solution_type == getattr(
            MoeSolutionType, "ASM", "asm"
        ):
            cache_key = id(expert_map)
            cached_mask = self._unified_asm_expert_mask_cache.get(cache_key)
            if cached_mask is not None and cached_mask[0] is expert_map:
                backend_expert_metadata = cached_mask[1]
            else:
                backend_expert_metadata = (expert_map >= 0).to(dtype=torch.int32)
                self._unified_asm_expert_mask_cache[cache_key] = (
                    expert_map,
                    backend_expert_metadata,
                )
        a1_scale = (
            None
            if runner_input.input_is_dequantized
            else (
                runner_input.a1_scale
                if runner_input.a1_scale is not None
                else quant_info.a13_scale
            )
        )
        activation_kwargs = get_aiter_moe_activation_kwargs(self.config, aiter_moe)
        # DeepSeek-V4 requests ordinary SiLU with a clamp, not MiniMax's
        # alpha/beta activation. Preserve that epilogue when migrating W8A8.
        if (
            (quant_info.use_int8_w8a8 or quant_info.use_fp8_w8a8)
            and self.config.swiglu_limit is not None
            and "gemm1_limit" not in activation_kwargs
        ):
            if not _aiter_moe_supports_gemm1_activation_params(aiter_moe):
                raise RuntimeError("AITER channel-W8A8 requires gemm1_limit support")
            activation_kwargs["gemm1_limit"] = float(self.config.swiglu_limit)
        activation = (
            "swigluoai"
            if self._uses_moe_c_oai_interleaved_layout(moe_config)
            else self.config.activation
        )
        # In-place ASM writes the final MoE sum into hidden_states regardless
        # of ``output_dtype``.  DeepEP normal dispatch may provide FP8 input
        # while its combine kernel requires BF16 output, so that case must be
        # out-of-place.
        inplace = self.config.inplace and (
            runner_input.output_dtype is None
            or runner_input.output_dtype == runner_input.hidden_states.dtype
        )

        output = aiter_moe(
            hidden_states=runner_input.hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=runner_input.topk_weights,
            topk_ids=runner_input.topk_ids,
            moe_config=moe_config,
            inplace=inplace,
            activation=activation,
            w1_scale=self._get_unified_w1_scale(quant_info, moe_config),
            w2_scale=quant_info.w2_scale,
            a1_scale=a1_scale,
            a2_scale=quant_info.a2_scale,
            block_shape=block_shape,
            global_num_experts=global_num_experts,
            expert_map=backend_expert_metadata,
            routed_scaling_factor=self.config.routed_scaling_factor,
            output_dtype=runner_input.output_dtype,
            **activation_kwargs,
        )
        return AiterRunnerOutput(hidden_states=output)

    def run(
        self,
        runner_input: AiterRunnerInput,
        quant_info: AiterMoeQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> AiterRunnerOutput:
        if runner_input.hidden_states.shape[0] == 0:
            if self.config.no_combine:
                topk = runner_input.topk_ids.shape[-1]
                hidden_size = runner_input.hidden_states.shape[-1]
                return AiterRunnerOutput(
                    hidden_states=runner_input.hidden_states.new_empty(
                        (0, topk, hidden_size),
                        dtype=(
                            runner_input.output_dtype
                            or runner_input.hidden_states.dtype
                        ),
                    )
                )
            # Keep this fast path consistent with aiter_moe's output-dtype
            # contract. DeepEP normal dispatch can leave one EP rank empty;
            # returning its FP8 input unchanged while non-empty ranks return
            # BF16 makes the collective combine fail on mixed dtypes.
            hidden_states = runner_input.hidden_states
            if (
                runner_input.output_dtype is not None
                and hidden_states.dtype != runner_input.output_dtype
            ):
                hidden_states = hidden_states.to(runner_input.output_dtype)
            return AiterRunnerOutput(hidden_states=hidden_states)

        if (
            quant_info.use_int8_w8a8
            or quant_info.use_fp8_w8a8
            or self._requires_unified_moe()
        ):
            return self._run_unified_moe(runner_input, quant_info)

        if self.config.no_combine and not _aiter_fused_moe_supports_no_combine():
            raise NotImplementedError(
                "no_combine=True requested but the installed aiter.fused_moe does "
                "not accept a `no_combine` kwarg. Install an aiter build that "
                "supports fused_moe no_combine output."
            )

        from aiter.fused_moe import fused_moe

        from sglang.srt.environ import envs

        a1_scale = (
            runner_input.a1_scale
            if runner_input.a1_scale is not None
            else quant_info.a13_scale
        )

        extra: dict = {}
        if quant_info.fused_moe_kwargs:
            extra.update(quant_info.fused_moe_kwargs)
        # `situ` uses the release branch's beta/linear_beta mapping below.
        # Other gated activations (notably MiniMax-M3's split-layout SwiGLU)
        # use the explicit alpha/limit API when the installed AITER exposes it.
        if self.config.activation != "situ":
            extra.update(get_aiter_moe_activation_kwargs(self.config, fused_moe))
        if runner_input.num_local_tokens is not None:
            extra["num_local_tokens"] = runner_input.num_local_tokens
        if runner_input.output_dtype is not None:
            extra["dtype"] = runner_input.output_dtype
        if self.config.activation == "situ":
            from aiter.ops.flydsl.moe_common import GateMode

            extra["gate_mode"] = GateMode.SEPARATED.value
            if self.config.gemm1_alpha is not None:
                extra["beta"] = float(self.config.gemm1_alpha)
            if self.config.gemm1_clamp_limit is not None:
                extra["linear_beta"] = float(self.config.gemm1_clamp_limit)
        elif quant_info.swiglu_limit > 0:
            # GateMode is only needed for the gpt-oss MXFP4 swiglu_limit path.
            # Import lazily so models that don't use it (e.g. DeepSeek-V3 fp8,
            # swiglu_limit==0) still run on aiter builds where this module
            # lives elsewhere / is absent.
            from aiter.ops.flydsl.moe_common import GateMode

            # Default (INTERLEAVE) preserves the pre-fix behavior for paths
            # that prepare weights in the gate/up-interleaved layout. Set
            # `SGLANG_USE_AITER_MOE_GU_ITLV=0` to switch to SEPARATED, which
            # matches the layout produced by `Mxfp4MoEMethod` (gpt-oss
            # MXFP4) and the gptoss_fp4 tuned FlyDSL kernels.
            extra["gate_mode"] = (
                GateMode.INTERLEAVE.value
                if envs.SGLANG_USE_AITER_MOE_GU_ITLV.get()
                else GateMode.SEPARATED.value
            )
            extra["swiglu_limit"] = quant_info.swiglu_limit
        if self.config.no_combine:
            extra["no_combine"] = True

        output = fused_moe(
            hidden_states=runner_input.hidden_states,
            w1=quant_info.w13_weight,
            w2=quant_info.w2_weight,
            topk_weight=runner_input.topk_weights,
            topk_ids=runner_input.topk_ids,
            quant_type=_aiter_quant_type(runner_input.quant_type),
            activation=_aiter_activation(self.config.activation),
            w1_scale=quant_info.w13_scale,
            w2_scale=quant_info.w2_scale,
            a1_scale=a1_scale,
            a2_scale=quant_info.a2_scale,
            bias1=quant_info.b13,
            bias2=quant_info.b2,
            expert_mask=quant_info.expert_mask,
            doweight_stage1=quant_info.doweight_stage1,
            hidden_pad=quant_info.hidden_pad,
            intermediate_pad=quant_info.intermediate_pad,
            **extra,
        )
        return AiterRunnerOutput(hidden_states=output)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.AITER


# ---------------------------------------------------------------------------
# Pre-permute: dispatch_output -> AiterRunnerInput
# ---------------------------------------------------------------------------


@register_pre_permute("standard", "aiter")
def pre_permute_standard_to_aiter(
    dispatch_output: StandardDispatchOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AiterRunnerInput:
    hidden_states = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    topk_weights = topk_weights.to(torch.float32)

    if runner_config.apply_router_weight_on_input and not quant_info.doweight_stage1:
        # Pre-scale at the Python level for kernels that don't honor doweight_stage1.
        assert (
            topk_weights.dim() == 2 and topk_weights.shape[-1] == 1
        ), "apply_router_weight_on_input requires topk=1"
        hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
        topk_weights = torch.ones_like(topk_weights)

    return AiterRunnerInput(
        hidden_states=hidden_states,
        topk_ids=topk_ids.to(torch.int32),
        topk_weights=topk_weights,
        quant_type=quant_info.quant_type,
    )


def _is_mori_dispatch_output(dispatch_output: Any) -> bool:
    # MoriEP{Normal,LL}DispatchOutput carry the post-mori-permute origin_topk_*
    # tensors that the standard DeepEP outputs lack.
    return hasattr(dispatch_output, "origin_topk_ids")


def _resolve_mori_quant_type(
    dispatch_a1_dtype: torch.dtype,
    dispatch_scale: Optional[torch.Tensor],
    weight_quant: AiterQuantType,
) -> AiterQuantType:
    """Pick the activation quant_type for AITER when the dispatch path may have
    pre-quantized hidden_states. Mirrors the original MoriEPMoE.run_moe_core
    decision tree."""
    is_fp8_quant = weight_quant in (
        AiterQuantType.PER_128X128,
        AiterQuantType.PER_TOKEN,
    )
    is_w4a4 = weight_quant == AiterQuantType.PER_1X32
    is_fp4_dispatch = dispatch_a1_dtype == torch.float4_e2m1fn_x2
    has_dispatch_scale = dispatch_scale is not None

    if is_w4a4:
        # W4A4 weights always run as per_1x32; FP8 dispatch is upscaled to BF16
        # before this point so dispatch_scale won't conflict.
        return AiterQuantType.PER_1X32
    if is_fp8_quant:
        return weight_quant
    # BF16 weights: lift to the dispatch-side quant type when scales are provided.
    if has_dispatch_scale and is_fp4_dispatch:
        return AiterQuantType.PER_1X32
    if has_dispatch_scale and not is_fp4_dispatch:
        return AiterQuantType.PER_128X128
    return AiterQuantType.NONE


def _pre_permute_deepep_to_aiter(
    dispatch_output: Union[
        DeepEPNormalDispatchOutput,
        DeepEPLLDispatchOutput,
        MoriEPNormalDispatchOutput,
        MoriEPLLDispatchOutput,
    ],
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AiterRunnerInput:
    is_mori = _is_mori_dispatch_output(dispatch_output)

    use_compact_deepep_ll = (
        not is_mori
        and dispatch_output.format.is_deepep_ll()
        and (
            runner_config.gemm1_alpha is not None
            or runner_config.gemm1_clamp_limit is not None
        )
    )

    hidden_states = dispatch_output.hidden_states
    topk_ids = dispatch_output.topk_ids.to(torch.int32)
    topk_weights = dispatch_output.topk_weights.to(torch.float32)
    a1_scale: Optional[torch.Tensor] = None
    num_local_tokens: Optional[torch.Tensor] = None
    output_dtype: Optional[torch.dtype] = None
    quant_type = quant_info.quant_type

    if not is_mori and not dispatch_output.format.is_deepep_ll():
        # Normal DeepEP may dispatch FP8 activations.  Pass its per-token
        # dynamic scale to AITER and force the expert output back to BF16;
        # the ROCm DeepEP normal combine kernel cannot consume Float8.
        a1_scale = dispatch_output.hidden_states_scale
        output_dtype = torch.bfloat16

    if use_compact_deepep_ll:
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            compact_deepep_ll_for_aiter,
        )

        if runner_config.num_experts is None or runner_config.num_local_experts is None:
            raise RuntimeError("DeepEP-LL AITER compaction requires expert counts")
        if runner_config.num_experts % runner_config.num_local_experts != 0:
            raise RuntimeError(
                "DeepEP-LL AITER compaction requires num_experts divisible by "
                "num_local_experts"
            )
        ep_size = runner_config.num_experts // runner_config.num_local_experts
        # Every EP rank can receive assignments originating on *other*
        # attention-DP ranks, including when its own local input is empty.
        # ``global_dp_buffer_len * topk`` is the common strict upper bound for
        # the whole EP collective and stays graph-static.  The local bound is
        # retained for direct/unit callers where DP forward metadata has not
        # been initialized.  At MiniMax decode c16 this yields 16*4=64 rows,
        # versus DeepEP's E*capacity=16384 rows.
        from sglang.srt.layers.dp_attention import get_global_dp_buffer_len

        top_k = (
            dispatch_output.topk_ids.shape[-1]
            if dispatch_output.topk_ids.ndim > 1
            else 1
        )
        global_assignment_bound = get_global_dp_buffer_len() * top_k
        local_assignment_bound = dispatch_output.topk_ids.numel() * ep_size
        # A fully idle forward still needs one zero-weight sentinel because
        # AITER does not accept M=0.
        max_compact_tokens = max(1, global_assignment_bound, local_assignment_bound)
        (
            hidden_states,
            a1_scale,
            topk_ids,
            topk_weights,
            expert_offsets,
        ) = compact_deepep_ll_for_aiter(
            dispatch_output.hidden_states,
            dispatch_output.hidden_states_scale,
            dispatch_output.masked_m,
            max_compact_tokens,
            # Padding rows have zero routing weight, so expert 0 is a safe
            # in-range placeholder and avoids carrying global sink metadata
            # into a local-id AITER call.
            0,
        )
        running_state["aiter_deepep_ll_compact"] = True
        running_state["aiter_deepep_ll_masked_m"] = dispatch_output.masked_m
        running_state["aiter_deepep_ll_expert_offsets"] = expert_offsets
        running_state["aiter_deepep_ll_output_shape"] = (
            dispatch_output.hidden_states.shape
        )
        running_state["aiter_combine_topk_ids"] = dispatch_output.topk_ids
        running_state["aiter_combine_topk_weights"] = dispatch_output.topk_weights
        running_state["aiter_combine_is_mori"] = False
        logger.info_once(
            "DeepEP low-latency -> AITER compact adapter enabled: "
            "capacity_rows=%s, compact_rows=%s",
            dispatch_output.hidden_states.shape[0]
            * dispatch_output.hidden_states.shape[1],
            max_compact_tokens,
        )
        return AiterRunnerInput(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            quant_type=quant_type,
            # Preserve DeepEP's FP8+per-token-scale representation when
            # available. This removes an FP8->BF16->FP8 round trip before the
            # channel-FP8 AITER GEMM. Non-FP8 dispatcher inputs still arrive
            # here already dequantized and carry no scale.
            a1_scale=a1_scale,
            output_dtype=torch.bfloat16,
            uses_local_expert_ids=True,
            input_is_dequantized=a1_scale is None,
        )

    if is_mori:
        from sglang.kernels.ops.moe.rocm_moe_utils import upscale, upscale_mxfp4

        a1_scale = dispatch_output.hidden_states_scale
        num_local_tokens = dispatch_output.num_recv_tokens_per_expert
        output_dtype = dispatch_output.out_dtype

        # Truncate dispatch tensors to the configured cap; mori combine only
        # reads [0, totalRecvTokenNum), so the truncated result needs no
        # padding back.
        mori_max = get_int_env_var("SGLANG_MORI_MOE_MAX_INPUT_TOKENS", 0)
        if mori_max > 0:
            hidden_states = hidden_states[:mori_max]
            if a1_scale is not None:
                a1_scale = a1_scale[:mori_max]
            topk_ids = topk_ids[:mori_max]
            topk_weights = topk_weights[:mori_max]

        # Upscale dispatched activations when there is no AITER kernel for the
        # weight/activation dtype pair.
        weight_quant = quant_info.quant_type
        is_fp8_quant = weight_quant in (
            AiterQuantType.PER_128X128,
            AiterQuantType.PER_TOKEN,
        )
        is_w4a4 = weight_quant == AiterQuantType.PER_1X32
        is_fp4_dispatch = hidden_states.dtype == torch.float4_e2m1fn_x2

        # AITER fused_moe Clamped-SwiGLU is dispatched with
        # gate_mode=INTERLEAVE, for which AITER picks a bf16/fp8 `q_dtype_a`
        # Refer to https://github.com/ROCm/aiter/blob/a2617c366dc7271a1662ecda2023d19f6ccefcec/aiter/fused_moe.py#L406-L412
        swiglu_interleave = quant_info.swiglu_limit > 0 and get_bool_env_var(
            "SGLANG_USE_AITER_MOE_GU_ITLV", "true"
        )

        if is_w4a4 and a1_scale is not None and not is_fp4_dispatch:
            # W4A4 weights with FP8 dispatch: dequant FP8->BF16 first; the
            # FP4 per_1x32 path needs BF16 input.
            hidden_states = upscale(
                hidden_states, a1_scale, num_local_tokens, output_dtype
            )
            a1_scale = None
        elif is_w4a4 and is_fp4_dispatch and a1_scale is not None and swiglu_interleave:
            # W4A4 weights + FP4 dispatch on the clamped-SwiGLU/INTERLEAVE
            # path: AITER expects a bf16/fp8 activation here, not fp4x2.
            # Dequant FP4->BF16 and let fused_moe re-quantize internally.
            hidden_states = upscale_mxfp4(
                hidden_states, a1_scale, num_local_tokens, output_dtype
            )
            a1_scale = None
        elif is_fp8_quant and is_fp4_dispatch and a1_scale is not None:
            # FP8 weights + FP4 dispatch: no kernel for the fp4x2/fp8 pair;
            # dequant FP4->BF16 and let fused_moe re-quantize to FP8.
            hidden_states = upscale_mxfp4(
                hidden_states, a1_scale, num_local_tokens, output_dtype
            )
            a1_scale = None

        quant_type = _resolve_mori_quant_type(
            hidden_states.dtype, a1_scale, weight_quant
        )

        running_state["aiter_combine_topk_ids"] = dispatch_output.origin_topk_ids
        running_state["aiter_combine_topk_weights"] = (
            dispatch_output.origin_topk_weights
        )
    else:
        # DeepEP marks invalid topk slots with idx == -1; AITER cannot accept
        # negative ids, so reroute them to the sink slot at index
        # num_local_experts (masked off by quant_info.expert_mask which has
        # shape (num_local_experts + 1,)).
        topk_ids = torch.where(
            topk_ids == -1,
            torch.full_like(topk_ids, runner_config.num_local_experts),
            topk_ids,
        )
        running_state["aiter_combine_topk_ids"] = dispatch_output.topk_ids
        running_state["aiter_combine_topk_weights"] = dispatch_output.topk_weights

    running_state["aiter_combine_is_mori"] = is_mori

    return AiterRunnerInput(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        quant_type=quant_type,
        a1_scale=a1_scale,
        num_local_tokens=num_local_tokens,
        output_dtype=output_dtype,
    )


register_pre_permute("deepep_normal", "aiter")(_pre_permute_deepep_to_aiter)
register_pre_permute("deepep_ll", "aiter")(_pre_permute_deepep_to_aiter)


# ---------------------------------------------------------------------------
# Post-permute: AiterRunnerOutput -> CombineInput
# ---------------------------------------------------------------------------


@register_post_permute("aiter", "standard")
def post_permute_aiter_to_standard(
    runner_output: AiterRunnerOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(hidden_states=runner_output.hidden_states)


def _post_permute_aiter_to_deepep(
    runner_output: AiterRunnerOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
    is_normal: bool,
) -> CombineInput:
    hidden_states = runner_output.hidden_states
    if running_state.get("aiter_deepep_ll_compact"):
        if is_normal:
            raise RuntimeError("DeepEP-LL compact output cannot use normal combine")
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            scatter_aiter_to_deepep_ll,
        )

        hidden_states = scatter_aiter_to_deepep_ll(
            hidden_states,
            running_state["aiter_deepep_ll_masked_m"],
            running_state["aiter_deepep_ll_expert_offsets"],
            running_state["aiter_deepep_ll_output_shape"],
        )

    if running_state.get("aiter_combine_is_mori"):
        from sglang.srt.layers.moe.token_dispatcher.moriep import (
            MoriEPLLCombineInput,
            MoriEPNormalCombineInput,
        )

        cls = MoriEPNormalCombineInput if is_normal else MoriEPLLCombineInput
    else:
        from sglang.srt.layers.moe.token_dispatcher.deepep import (
            DeepEPLLCombineInput,
            DeepEPNormalCombineInput,
        )

        cls = DeepEPNormalCombineInput if is_normal else DeepEPLLCombineInput

    return cls(
        hidden_states=hidden_states,
        topk_ids=running_state["aiter_combine_topk_ids"],
        topk_weights=running_state["aiter_combine_topk_weights"],
    )


@register_post_permute("aiter", "deepep_normal")
def post_permute_aiter_to_deepep_normal(
    runner_output: AiterRunnerOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> CombineInput:
    return _post_permute_aiter_to_deepep(
        runner_output, quant_info, runner_config, running_state, is_normal=True
    )


@register_post_permute("aiter", "deepep_ll")
def post_permute_aiter_to_deepep_ll(
    runner_output: AiterRunnerOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> CombineInput:
    return _post_permute_aiter_to_deepep(
        runner_output, quant_info, runner_config, running_state, is_normal=False
    )
