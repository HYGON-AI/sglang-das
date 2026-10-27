from dataclasses import replace

import torch
import torch.nn.functional as F

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.moe.topk import (
    _RENORMALIZE_SUM_EPSILON,
    _use_lightop_sqrtsoftplus_v41_gate,
    StandardTopKOutput,
    StandardTopKOutputPacked,
    _mask_topk_ids_padded_region,
    _post_process_topk_ids,
    _zero_topk_weights_padded_region,
)
from sglang.srt.layers.moe.utils import has_per_rank_fused_shared_slots
from sglang.srt.utils import is_cuda, is_hcu


def _scale_fused_shared_weights(weights, num_fused_shared_experts, scaling_factor):
    # Standard EP replicates the fused shared expert on every rank and all-reduces,
    # so the shared columns carry a 1/ep_size factor.
    if num_fused_shared_experts and scaling_factor is not None:
        weights[:, -num_fused_shared_experts:] *= scaling_factor
    return weights


def _postprocess_vision_topk(
    moe,
    logits,
    weights,
    indices,
    num_token_non_padded,
    expert_location_dispatch_info,
    packed_topk=None,
):
    config = moe.topk.topk_config
    num_shared = config.num_fused_shared_experts
    # Gate kernels already append global shared slots. EPLB only maps routed
    # experts; passing shared IDs to its table would index past the table.
    routed_count = config.top_k - num_shared
    routed_ids, routed_weights, recorder_ids = _post_process_topk_ids(
        topk_ids=indices[:, :routed_count],
        topk_weights=weights[:, :routed_count],
        topk_config=(
            replace(config, top_k=routed_count, num_fused_shared_experts=0)
            if num_shared
            else config
        ),
        router_logits=logits,
        layer_id=moe.layer_id,
        num_token_non_padded=num_token_non_padded,
        expert_location_dispatch_info=expert_location_dispatch_info,
    )
    if recorder_ids is not None:
        get_global_expert_distribution_recorder().on_select_experts(
            topk_ids=recorder_ids
        )
    if num_shared:
        # Redundant routed experts move the global shared-slot base as well.
        shared_base = (
            expert_location_dispatch_info.num_physical_experts
            if expert_location_dispatch_info is not None
            else logits.shape[-1]
        )
        shared_ids = indices[:, routed_count:].to(routed_ids.dtype)
        shared_ids = shared_ids + (shared_base - logits.shape[-1])
        indices = torch.cat((routed_ids, shared_ids), dim=-1)
        weights = torch.cat((routed_weights, weights[:, routed_count:]), dim=-1)
    else:
        indices, weights = routed_ids, routed_weights
    if num_token_non_padded is not None:
        _mask_topk_ids_padded_region(indices, num_token_non_padded)
        _zero_topk_weights_padded_region(weights, num_token_non_padded)
    if packed_topk is not None:
        return StandardTopKOutputPacked(weights, indices, logits, packed_topk)
    return StandardTopKOutput(weights, indices, logits)


def vision_topk(moe, logits, input_ids, num_token_non_padded=None):
    config = moe.topk.topk_config
    num_fused_shared_experts = config.num_fused_shared_experts
    if num_fused_shared_experts:
        # VL gate kernels emit global shared slots, not per-rank shared slots.
        assert not has_per_rank_fused_shared_slots(num_fused_shared_experts), (
            "VL routing does not support per-rank fused shared slots"
        )
    expert_location_dispatch_info = (
        ExpertLocationDispatchInfo.init_new(layer_id=moe.layer_id)
        if not moe.is_nextn
        else None
    )

    # HCU lightop fast path: fused sqrtsoftplus gate with per-token 2D bias.
    # _use_lightop_sqrtsoftplus_v41_gate already implies HCU; is_hcu() is an
    # explicit guard so this block never runs on non-HCU builds regardless of
    # how the flag is set.
    if (
        is_hcu()
        and _use_lightop_sqrtsoftplus_v41_gate
        and logits.dtype == torch.float32
        and logits.is_contiguous()
        and logits.size(1) in (256, 384)  # kernel compiled only for these expert counts
        and 0 <= num_fused_shared_experts < config.top_k <= 16
        and config.routed_scaling_factor is not None
        and config.routed_scaling_factor > 0.0
        and moe.gate.e_score_correction_bias.dtype == torch.float32
        and (
            input_ids is None
            or moe.gate.e_score_correction_bias_vl.dtype == torch.float32
        )
    ):
        # input_ids is None only for idle DP padding-token forwards; pass 1D
        # bias directly so the kernel uses bias_row_stride=0 (broadcast).
        if input_ids is None:
            bias = moe.gate.e_score_correction_bias
        else:
            # Expand to [num_tokens, num_experts]; image tokens take the VL
            # bias, text tokens take the routed bias. .contiguous() is required
            # by the kernel's vectorized aligned load.
            bias = torch.where(
                (input_ids == moe.config.image_token_id)[:, None],
                moe.gate.e_score_correction_bias_vl,
                moe.gate.e_score_correction_bias,
            ).contiguous()
        weights, indices = torch.ops.sglang.moe_fused_gate_sqrtsoftplus_v41_hcu(
            logits,
            bias,
            config.top_k,
            num_fused_shared_experts,
            config.renormalize and config.top_k > 1,
            config.routed_scaling_factor,
            config.apply_routed_scaling_factor_on_output,
        )
        weights = _scale_fused_shared_weights(
            weights,
            num_fused_shared_experts,
            config.fused_shared_experts_scaling_factor,
        )
        return _postprocess_vision_topk(
            moe,
            logits,
            weights,
            indices,
            num_token_non_padded,
            expert_location_dispatch_info,
        )

    if is_cuda():
        from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
        from sglang.srt.layers.moe.utils import get_moe_runner_backend

        # Same admission as _fused_gate_emits_packed_ids: the shared-expert slots
        # rescaled below would rewrite weights after the router.
        packed_topk = None
        if (
            num_fused_shared_experts == 0
            and expert_location_dispatch_info is None
            and get_moe_runner_backend().is_flashinfer_mxfp4()
        ):
            packed_topk = torch.empty(
                (logits.shape[0], config.top_k), dtype=torch.int32, device=logits.device
            )

        weights, indices = moe_fused_gate(
            logits,
            moe.gate.e_score_correction_bias,
            topk=config.top_k,
            scoring_func="sqrtsoftplus",
            num_fused_shared_experts=num_fused_shared_experts,
            bias_alt=moe.gate.e_score_correction_bias_vl,
            input_ids=input_ids,
            bias_alt_token_id=moe.config.image_token_id,
            renormalize=config.renormalize and config.top_k > 1,
            renormalize_epsilon=_RENORMALIZE_SUM_EPSILON,
            routed_scaling_factor=config.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=config.apply_routed_scaling_factor_on_output,
            num_token_non_padded=num_token_non_padded,
            packed_out=packed_topk,
            sqrtsoftplus_log1p=True,
        )
        weights = _scale_fused_shared_weights(
            weights,
            num_fused_shared_experts,
            config.fused_shared_experts_scaling_factor,
        )
        return _postprocess_vision_topk(
            moe,
            logits,
            weights,
            indices,
            num_token_non_padded,
            expert_location_dispatch_info,
            packed_topk,
        )
    scores = F.softplus(logits.float()).sqrt()
    if input_ids is None:
        bias = moe.gate.e_score_correction_bias
    else:
        bias = torch.where(
            (input_ids == moe.config.image_token_id)[:, None],
            moe.gate.e_score_correction_bias_vl,
            moe.gate.e_score_correction_bias,
        )
    # The shared slots appended below use the same layout as biased_grouped_topk_gpu.
    topk_routed = config.top_k - num_fused_shared_experts
    indices = (scores + bias).topk(topk_routed, dim=-1).indices
    weights = scores.gather(-1, indices)
    routed_sum = weights.sum(-1, keepdim=True, dtype=torch.float32)
    if num_fused_shared_experts:
        shared_ids = logits.shape[-1] + torch.arange(
            num_fused_shared_experts, device=indices.device, dtype=indices.dtype
        )
        indices = torch.cat(
            [indices, shared_ids.expand(indices.shape[0], -1)],
            dim=-1,
        )
        weights = F.pad(weights, (0, num_fused_shared_experts))
        weights[:, topk_routed:] = routed_sum / config.routed_scaling_factor
    if config.renormalize and config.top_k > 1:
        weights = weights / (routed_sum + _RENORMALIZE_SUM_EPSILON)
    if config.apply_routed_scaling_factor_on_output:
        weights = weights * config.routed_scaling_factor
    weights = _scale_fused_shared_weights(
        weights, num_fused_shared_experts, config.fused_shared_experts_scaling_factor
    )
    weights, indices = weights.float(), indices.int()
    return _postprocess_vision_topk(
        moe,
        logits,
        weights,
        indices,
        num_token_non_padded,
        expert_location_dispatch_info,
    )
