# Copyright (c) 2026 gencheng liu
# SPDX-License-Identifier: Apache-2.0

"""DeepEP glue for the HCU AITER-layout W4A8 grouped GEMM."""

from __future__ import annotations

import torch

from sglang.kernels.ops.moe import w4a8_deepep_aiter as _w4a8


@torch.no_grad()
@torch._dynamo.disable()
def forward_w4a8_deepep_normal(layer, dispatch_output):
    # Import lazily: this function is called after ep_moe.layer finished
    # importing, which avoids a module initialization cycle.
    from sglang.srt.layers.moe.ep_moe import layer as ep_layer

    (
        hidden_states,
        hidden_states_scale,
        topk_ids,
        topk_weights,
        num_recv_tokens_per_expert,
    ) = dispatch_output

    if num_recv_tokens_per_expert is None:
        return hidden_states.bfloat16()
    all_tokens = sum(num_recv_tokens_per_expert)
    if all_tokens <= 0:
        return hidden_states.bfloat16()
    if hidden_states.dtype != torch.int8 or hidden_states_scale is None:
        raise RuntimeError(
            "W4A8 DeepEP normal requires int8 dispatch activations and fp32 "
            "per-token scales"
        )
    if any(count % 256 for count in num_recv_tokens_per_expert):
        raise RuntimeError(
            "W4A8 MMAC requires DeepEP expert_alignment=256; received counts "
            f"{num_recv_tokens_per_expert}"
        )

    _, hidden_size = hidden_states.shape
    gate_up_size = layer.w13_weight.size(1)
    device = hidden_states.device
    a1 = torch.empty((all_tokens, hidden_size), dtype=torch.int8, device=device)
    a1_scale = torch.empty(
        (all_tokens, hidden_states_scale.shape[-1]),
        dtype=torch.float32,
        device=device,
    )

    if ep_layer.get_offloader().forbid_copy_engine_usage:
        counts_gpu = ep_layer.copy_list_to_gpu_no_ce(num_recv_tokens_per_expert)
    else:
        counts_gpu = torch.tensor(
            num_recv_tokens_per_expert,
            dtype=torch.int32,
            pin_memory=True,
            device="cpu",
        ).cuda(non_blocking=True)

    m_indices, output_index = ep_layer._ep_scatter_with_optional_lightop(
        hidden_states,
        hidden_states_scale,
        topk_ids,
        counts_gpu,
        a1,
        a1_scale,
        all_tokens,
        counts_are_aligned=True,
    )

    # The metadata workspace is tiny (about 4.125 bytes per expert-aligned
    # row) and can be reused by the two sequential GEMMs on the same stream.
    workspace = torch.empty(
        all_tokens + all_tokens // 32 + 1,
        dtype=torch.int32,
        device=device,
    )
    gate_up = torch.empty(
        (all_tokens, gate_up_size), dtype=torch.bfloat16, device=device
    )
    _w4a8.w4a8_mmac_contiguous_out(
        a1,
        a1_scale,
        layer.w13_weight,
        layer.w13_weight_scale,
        m_indices,
        workspace,
        gate_up,
    )
    del a1, a1_scale

    # DeepSeek-V4 clamps the two SwiGLU halves before activation.  Use the
    # LightOp fused clamp + SiLU + multiply + dynamic INT8 quantizer when the
    # model requests it, avoiding two standalone clamp launches.
    runner_config = getattr(layer, "moe_runner_config", None)
    swiglu_limit = getattr(runner_config, "swiglu_limit", None)
    if swiglu_limit is None:
        a2, a2_scale = ep_layer.fuse_silu_mul_quant(gate_up)
    else:
        from lightop.fuse_silu_mul_quant import fuse_silu_mul_clamp_quant

        a2, a2_scale = fuse_silu_mul_clamp_quant(gate_up, float(swiglu_limit))
    del gate_up
    down = torch.empty((all_tokens, hidden_size), dtype=torch.bfloat16, device=device)
    _w4a8.w4a8_mmac_contiguous_out(
        a2,
        a2_scale,
        layer.w2_weight,
        layer.w2_weight_scale,
        m_indices,
        workspace,
        down,
    )
    del a2, a2_scale, workspace

    # Zero initialization preserves the behavior for receive rows whose local
    # top-k entries are all invalid.
    gathered = torch.zeros_like(hidden_states, dtype=torch.bfloat16)
    ep_layer._ep_gather_with_optional_lightop(
        down, topk_ids, topk_weights, output_index, gathered
    )
    return gathered


@torch.no_grad()
@torch._dynamo.disable()
def forward_w4a8_deepep_low_latency(layer, dispatch_output):
    """Run W4A8 directly on DeepEP's masked [E,T,K] decode layout."""
    (
        hidden_states,
        hidden_states_scale,
        _topk_ids,
        _topk_weights,
        masked_m,
        expected_m,
    ) = dispatch_output

    if hidden_states.dtype != torch.int8 or hidden_states_scale is None:
        raise RuntimeError(
            "W4A8 DeepEP low-latency requires int8 dispatch activations "
            "and fp32 per-token scales"
        )
    if hidden_states.dim() != 3:
        raise RuntimeError(
            "W4A8 DeepEP low-latency expects hidden_states [E,T,K], got "
            f"{tuple(hidden_states.shape)}"
        )

    num_experts, rows_per_expert, hidden_size = hidden_states.shape
    total_rows = num_experts * rows_per_expert
    block_m = 16

    # expected_m is derived from the global token/expert ratio.  A safe launch
    # bound is all globally possible valid rows plus one partial BM tile for
    # each local expert.  The metadata kernel computes the exact compact list
    # from masked_m entirely on device, so this does not synchronize the host.
    runner_config = layer.moe_runner_config
    global_num_experts = runner_config.num_experts
    max_valid_rows = max(1, int(expected_m)) * global_num_experts
    metadata_rows = min(
        total_rows,
        max_valid_rows + num_experts * (block_m - 1),
    )
    metadata_rows = min(
        total_rows,
        ((metadata_rows + block_m - 1) // block_m) * block_m,
    )
    workspace = torch.empty(
        metadata_rows + metadata_rows // block_m + 1,
        dtype=torch.int32,
        device=hidden_states.device,
    )

    gate_up_size = layer.w13_weight.size(1)
    gate_up = torch.empty(
        (num_experts, rows_per_expert, gate_up_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    _w4a8.w4a8_mmac_masked_out(
        hidden_states,
        hidden_states_scale,
        layer.w13_weight,
        layer.w13_weight_scale,
        masked_m,
        workspace,
        gate_up,
        metadata_rows,
    )

    swiglu_limit = getattr(runner_config, "swiglu_limit", None)
    if swiglu_limit is None:
        from lightop.activation import fuse_silu_mul_quant_ep

        a2, a2_scale = fuse_silu_mul_quant_ep(
            gate_up, masked_m, expect_m=int(expected_m)
        )
    else:
        from lightop.fuse_silu_mul_quant import fuse_silu_mul_clamp_quant_ep

        a2, a2_scale = fuse_silu_mul_clamp_quant_ep(
            gate_up,
            float(swiglu_limit),
            masked_m,
            expect_m=int(expected_m),
        )
    del gate_up

    down = torch.empty(
        (num_experts, rows_per_expert, hidden_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    _w4a8.w4a8_mmac_masked_out(
        a2,
        a2_scale,
        layer.w2_weight,
        layer.w2_weight_scale,
        masked_m,
        workspace,
        down,
        metadata_rows,
    )
    return down
