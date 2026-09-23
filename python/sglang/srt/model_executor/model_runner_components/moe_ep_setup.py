from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.eplb.lplb_solver import (
    LoadAwareReplicaSolver,
    LPLBSolver,
    assert_lplb_supported_model,
    clear_global_lplb_solvers,
    set_global_lplb_solver,
)
from sglang.srt.layers.moe.hash_topk import HashTopK
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.utils import get_bool_env_var, is_hip, log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()


def prepare_moe_topk(
    *,
    model,
    model_config: ModelConfig,
    server_args: ServerArgs,
    moe_ep_size: int,
    moe_ep_rank: int,
) -> None:
    balancer_cls = None
    num_prepared = 0
    num_routed_experts = None
    for module in model.modules():
        if not isinstance(module, (TopK, HashTopK)):
            continue
        if not module.enable_waterfill or module.waterfill_balancer is not None:
            continue
        if num_routed_experts is None:
            num_routed_experts = getattr(
                model_config.hf_config, "n_routed_experts", None
            )
            if num_routed_experts is None:
                raise ValueError("Waterfill requires model config n_routed_experts.")
        if balancer_cls is None:
            from sglang.srt.layers.moe.waterfill import WaterfillBalancer

            balancer_cls = WaterfillBalancer
        # Static EPLB remaps TopK ids to physical expert ids before Waterfill.
        # Redundant experts therefore need to be included in the per-rank
        # expert count used for Waterfill's shared-expert slot remapping.
        num_physical_routed_experts = (
            num_routed_experts + server_args.ep_num_redundant_experts
        )
        if isinstance(module, TopK):
            routed_scaling_factor = module.topk_config.routed_scaling_factor
        else:
            routed_scaling_factor = module.routed_scaling_factor
        module.waterfill_balancer = balancer_cls(
            num_routed_experts=num_physical_routed_experts,
            world_size=moe_ep_size,
            rank=moe_ep_rank,
            layer_id=module.layer_id,
            routed_scaling_factor=(
                routed_scaling_factor if routed_scaling_factor is not None else 1.0
            ),
        )
        num_prepared += 1
    if num_prepared:
        log_info_on_rank0(logger, f"Prepared {num_prepared} Waterfill TopK modules.")


def init_lplb_solvers(*, model_config: ModelConfig) -> None:
    """Initialize per-layer LPLB solvers from current expert location metadata."""
    from sglang.srt.distributed import get_moe_ep_group
    from sglang.srt.runtime_context import get_exec

    dispatch_algorithm = get_exec().moe.ep_dispatch_algorithm
    # Gate: refuse the Hopper LP solver for non-DeepSeek MoE families whose
    # empty-token paths don't participate in its EP all-reduce. The ROCm
    # load-aware path explicitly wires empty MiniMax ranks through solve().
    architectures = getattr(model_config.hf_config, "architectures", None)
    if dispatch_algorithm == "lp" and architectures:
        assert_lplb_supported_model(architectures[0])

    metadata = get_global_expert_location_metadata()
    if metadata is None:
        return
    clear_global_lplb_solvers()
    ep_group = get_moe_ep_group()
    static_mass = None
    static_max_copies = None
    static_path = envs.SGLANG_EXPERIMENTAL_LPLB_STATIC_PROBS.get()
    if static_path:
        import torch

        if not torch.version.hip:
            raise NotImplementedError("Precomputed compact LP dispatch requires HIP")
        payload = torch.load(static_path, weights_only=True, map_location="cpu")
        if not torch.equal(
            payload["physical_to_logical_map"],
            metadata.physical_to_logical_map_cpu,
        ):
            raise ValueError(
                "Static LP artifact does not match the loaded expert layout"
            )
        if payload["physical_mass"].shape != metadata.physical_to_logical_map_cpu.shape:
            raise ValueError(
                "Static LP mass must match the physical expert layout shape"
            )
        static_mass = payload["physical_mass"].to(
            device=metadata.physical_to_logical_map.device, dtype=torch.float32
        )
        if not bool(torch.isfinite(static_mass).all() and (static_mass >= 0).all()):
            raise ValueError("Static LP mass must be finite and nonnegative")
        static_max_copies = int(
            metadata.logical_to_all_physical_map_num_valid.max().item()
        )
        logger.info(
            "Using precomputed LP replica probabilities from %s; replica table %d -> %d columns",
            static_path,
            metadata.logical_to_all_physical_map.shape[-1],
            static_max_copies,
        )
    for lid in range(metadata.num_layers):
        static_probabilities = None
        if static_mass is not None:
            replica_map = metadata.logical_to_all_physical_map[lid]
            static_probabilities = static_mass[lid][replica_map.clamp_min(0).long()]
            static_probabilities = static_probabilities * (replica_map >= 0)
        solver_cls = (
            LoadAwareReplicaSolver if dispatch_algorithm == "load_aware" else LPLBSolver
        )
        solver = solver_cls(
            phy2log=metadata.physical_to_logical_map[lid],
            log2phy=metadata.logical_to_all_physical_map[lid],
            num_gpus=metadata.ep_size,
            ep_group=ep_group,
            logical_to_all_physical_map_num_valid=(
                metadata.logical_to_all_physical_map_num_valid[lid]
            ),
            static_probabilities=static_probabilities,
            static_max_copies=static_max_copies,
        )
        set_global_lplb_solver(lid, solver)
    logger.info(
        "Initialized %s replica-dispatch solvers for %d layers",
        dispatch_algorithm,
        metadata.num_layers,
    )


def check_quantized_moe_compatibility(
    *,
    model_config: ModelConfig,
    tp_size: int,
    moe_ep_size: int,
    moe_dp_size: int,
) -> None:
    if (
        quantization_config := getattr(
            model_config.hf_config, "quantization_config", None
        )
    ) is not None and (
        weight_block_size := quantization_config.get("weight_block_size", None)
    ) is not None:
        weight_block_size_n = weight_block_size[0]

        if tp_size % moe_ep_size != 0:
            raise ValueError(
                f"tp_size {tp_size} must be divisible by ep_size {moe_ep_size}"
            )
        moe_tp_size = tp_size // moe_ep_size // moe_dp_size

        moe_intermediate_size = getattr(
            model_config.hf_text_config, "moe_intermediate_size", None
        )
        if moe_intermediate_size is None:
            return

        if moe_intermediate_size % moe_tp_size != 0:
            raise ValueError(
                f"moe_intermediate_size {moe_intermediate_size} must be divisible by moe_tp_size ({moe_tp_size}) which is tp_size ({tp_size}) divided by moe_ep_size ({moe_ep_size})."
            )

        if (
            not envs.SGLANG_SHARED_EXPERT_TP1.get()
            and (moe_intermediate_size // moe_tp_size) % weight_block_size_n != 0
            and not _use_aiter
        ):
            raise ValueError(
                f"For quantized MoE models, please make sure ({moe_intermediate_size=} / {moe_tp_size=}) % {weight_block_size_n=} == 0 "
                f"where moe_tp_size is equal to tp_size ({tp_size}) divided by ep_size ({moe_ep_size}). "
                f"You can fix this by setting arguments `--tp` and `--ep` correctly."
            )
