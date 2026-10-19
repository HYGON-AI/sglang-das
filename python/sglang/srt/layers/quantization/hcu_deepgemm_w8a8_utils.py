from __future__ import annotations

import os

import torch

_INT8_DEEPGEMM_ASM_ENV = "SGLANG_INT8_DEEPGEMM_ASM"


def use_int8_deepgemm_asm() -> bool:
    return os.getenv(_INT8_DEEPGEMM_ASM_ENV, "0").lower() in ("1", "true")


def get_hcu_gfx_arch() -> str:
    if not torch.cuda.is_available():
        return ""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = getattr(props, "gcnArchName", props.name)
    return arch.split(":")[0].lower()


def is_hcu_gfx936() -> bool:
    return get_hcu_gfx_arch() == "gfx936"


def _register_runtime_buffer(
    layer: torch.nn.Module, name: str, value: torch.Tensor
) -> None:
    if name in layer._buffers:
        layer._buffers[name] = value
        layer._non_persistent_buffers_set.add(name)
        return

    if hasattr(layer, name):
        delattr(layer, name)
    layer.register_buffer(name, value, persistent=False)


def _check_int8_weight(weight: torch.Tensor) -> tuple[int, int, int]:
    if weight.dim() != 3:
        raise ValueError(
            f"DeepGEMM W8A8 INT8 weight must be [E, N, K], got {tuple(weight.shape)}"
        )
    if weight.element_size() != 1:
        raise ValueError(f"DeepGEMM W8A8 INT8 weight must be 8-bit, got {weight.dtype}")
    return int(weight.size(0)), int(weight.size(1)), int(weight.size(2))


def _pack_w8a8_int8_asm_masked_weight(weight: torch.Tensor) -> torch.Tensor:
    experts, n, k = _check_int8_weight(weight)
    if n % 16 != 0 or k % 64 != 0:
        raise ValueError(f"N={n}, K={k} must satisfy N%16==0 and K%64==0")
    return (
        weight.contiguous()
        .reshape(experts, n // 16, 16, k // 64, 4, 16)
        .permute(0, 3, 1, 4, 2, 5)
        .contiguous()
        .view(experts, n // 16, k * 16)
    )


def prepare_w8a8_int8_deepgemm_weights(layer: torch.nn.Module) -> None:
    if getattr(layer, "_w8a8_int8_deepgemm_repacked", False):
        return
    if not hasattr(layer, "w13_weight") or not hasattr(layer, "w2_weight"):
        return

    arch = get_hcu_gfx_arch()
    use_asm = use_int8_deepgemm_asm()
    need_contiguous_layout = True
    need_masked_layout = False
    from sglang.srt.layers.moe.utils import get_deepep_mode

    deepep_mode = get_deepep_mode()
    if deepep_mode.is_normal():
        from deepgemm import marlin_i8_contiguous_weight

        def pack_weight(weight: torch.Tensor) -> torch.Tensor:
            return marlin_i8_contiguous_weight(
                weight,
                shuffle_unique=0,
                low_memory=True,
            )
    elif use_asm:
        need_contiguous_layout = deepep_mode.enable_normal()
        need_masked_layout = deepep_mode.enable_low_latency()
        from deepgemm import marlin_i8_contiguous_weight

        def pack_weight(weight: torch.Tensor) -> torch.Tensor:
            return marlin_i8_contiguous_weight(
                weight,
                shuffle_unique=0,
                low_memory=True,
            )

        pack_masked_weight = _pack_w8a8_int8_asm_masked_weight
    elif arch == "gfx936":
        from deepgemm import pack_w8a8_gfx936_weight_to_w6 as pack_weight
    else:
        from deepgemm.m_group_gemm import (
            pack_int8_weight_enk_to_w6_low_latency as pack_weight,
        )

    w13_weight = layer.w13_weight
    w2_weight = layer.w2_weight

    layer._w8a8_int8_w13_weight_shape = tuple(w13_weight.shape)
    layer._w8a8_int8_w2_weight_shape = tuple(w2_weight.shape)

    with torch.no_grad():
        if need_contiguous_layout:
            w13_weight_deepgemm = pack_weight(w13_weight).detach()
            w2_weight_deepgemm = pack_weight(w2_weight).detach()
        if need_masked_layout:
            w13_weight_deepgemm_masked = pack_masked_weight(w13_weight).detach()
            w2_weight_deepgemm_masked = pack_masked_weight(w2_weight).detach()

    if need_contiguous_layout:
        _register_runtime_buffer(layer, "w13_weight_deepgemm", w13_weight_deepgemm)
        _register_runtime_buffer(layer, "w2_weight_deepgemm", w2_weight_deepgemm)
    if need_masked_layout:
        _register_runtime_buffer(
            layer, "w13_weight_deepgemm_masked", w13_weight_deepgemm_masked
        )
        _register_runtime_buffer(
            layer, "w2_weight_deepgemm_masked", w2_weight_deepgemm_masked
        )
    layer._w8a8_int8_deepgemm_repacked = True
    layer._w8a8_int8_deepgemm_arch = arch
    layer._w8a8_int8_deepgemm_asm = use_asm
    layer._w8a8_int8_deepgemm_has_contiguous = need_contiguous_layout
    layer._w8a8_int8_deepgemm_has_masked = need_masked_layout

    del layer.w13_weight
    del layer.w2_weight
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
