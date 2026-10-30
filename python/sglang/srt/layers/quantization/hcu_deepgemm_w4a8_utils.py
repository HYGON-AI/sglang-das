# Copyright 2026 Hygon Information Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0.
"""HIPC W4A8 packing and masked GEMMs ported from sglang-model's w4a8_int."""

import torch

from sglang.srt.layers.quantization.hcu_deepgemm_w8a8_utils import (
    _register_runtime_buffer,
)


def prepare_slimquant_w4a8_deepgemm_weights(layer: torch.nn.Module) -> None:
    if getattr(layer, "_w4a8_deepgemm_ready", False):
        return

    from deepgemm import pack_w4a8_moe_hipc_weight

    # normalize_expert_weight already made K-even the high nibble. Unlike
    # w4a8_int's loader, no nibble swap or zero-point conversion remains here.
    with torch.no_grad():
        w13 = pack_w4a8_moe_hipc_weight(layer.w13_weight).detach()
        w2 = pack_w4a8_moe_hipc_weight(layer.w2_weight).detach()
        # HIPC, like LightOp, promotes signed INT4 to INT8 as q * 16.
        if not getattr(layer, "_w4a8_scales_divided_by_16", False):
            layer.w13_weight_scale.div_(16.0)
            layer.w2_weight_scale.div_(16.0)
            layer._w4a8_scales_divided_by_16 = True

    _register_runtime_buffer(layer, "w13_weight_deepgemm", w13)
    _register_runtime_buffer(layer, "w2_weight_deepgemm", w2)
    del layer.w13_weight
    del layer.w2_weight
    layer._w4a8_deepgemm_ready = True


def apply_slimquant_w4a8_deepgemm_masked(
    layer, hidden_states, hidden_states_scale, masked_m, expected_m, moe_runner_config
):
    from deepgemm import m_grouped_w4a8_gemm_nt_masked_hipc
    from lightop import fuse_silu_mul_clamp_quant

    expected_m = min(hidden_states.shape[1], expected_m)
    gateup_output = torch.empty(
        (*hidden_states.shape[:2], layer.w13_weight_scale.shape[1]),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    m_grouped_w4a8_gemm_nt_masked_hipc(
        (hidden_states, hidden_states_scale),
        (layer.w13_weight_deepgemm, layer.w13_weight_scale),
        gateup_output,
        masked_m,
        expected_m,
    )
    if moe_runner_config.swiglu_limit is None:
        q_a2, q_a2_scale = torch.ops.sglang.fuse_silu_mul_quant_ep(
            gateup_output, masked_m
        )
    else:
        q_a2, q_a2_scale = fuse_silu_mul_clamp_quant(
            input=gateup_output,
            limit=moe_runner_config.swiglu_limit,
            mask_m=masked_m,
            expect_m=expected_m,
        )
    del gateup_output
    down_output = torch.empty(
        (*hidden_states.shape[:2], layer.w2_weight_scale.shape[1]),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    m_grouped_w4a8_gemm_nt_masked_hipc(
        (q_a2, q_a2_scale),
        (layer.w2_weight_deepgemm, layer.w2_weight_scale),
        down_output,
        masked_m,
        expected_m,
    )
    return down_output
