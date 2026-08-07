"""
Debug helper for comparing DeepEP vs TP-only INT8 MoE paths.

Usage:
1. Set env vars before launching:
   export SGLANG_DEBUG_MOE_LAYER_ID=0     # dump only this layer (0-indexed)
   export SGLANG_DEBUG_MOE_STEP_ID=0      # dump only this forward step

2. The dumps will be saved to /tmp/sglang_debug_moe/
   - tp/*.pt  : TP-only path dumps
   - deepep/*.pt : DeepEP path dumps

3. Compare with:
   python compare_dumps.py /tmp/sglang_debug_moe/tp /tmp/sglang_debug_moe/deepep
"""

import os
import torch
from typing import Optional

_DEBUG_DIR = "/tmp/sglang_debug_moe"
_LAYER_ID = int(os.environ.get("SGLANG_DEBUG_MOE_LAYER_ID", "-1"))
_STEP_ID = int(os.environ.get("SGLANG_DEBUG_MOE_STEP_ID", "-1"))

# Module-level step counter (incremented per forward call)
_step_counter: dict = {}
# Module-level layer counter maps layer_id -> step count
_layer_step_counter: dict = {}


def _dump_dir(path_selector: str) -> str:
    d = os.path.join(_DEBUG_DIR, path_selector)
    os.makedirs(d, exist_ok=True)
    return d


def _should_debug(layer_id: int, step_key: str = "global") -> bool:
    """Check if we should debug this layer at this step. Only rank 0 dumps."""
    # Only rank 0
    try:
        from sglang.srt.distributed import get_tensor_model_parallel_rank
        if get_tensor_model_parallel_rank() != 0:
            return False
    except Exception:
        pass
    if _LAYER_ID >= 0 and layer_id != _LAYER_ID:
        return False
    if _STEP_ID >= 0:
        cnt = _layer_step_counter.setdefault(f"{layer_id}_{step_key}", 0)
        _layer_step_counter[f"{layer_id}_{step_key}"] = cnt + 1
        if cnt != _STEP_ID:
            return False
    return True


def _save(tag: str, path_selector: str, **tensors):
    """Save named tensors to disk."""
    d = _dump_dir(path_selector)
    data = {}
    for name, t in tensors.items():
        if t is None:
            continue
        if isinstance(t, torch.Tensor):
            data[name] = t.detach().cpu().clone()
        elif isinstance(t, (list, tuple)):
            data[name] = [
                x.detach().cpu().clone() if isinstance(x, torch.Tensor) else x
                for x in t
            ]
        else:
            data[name] = t
    path = os.path.join(d, f"{tag}.pt")
    torch.save(data, path)
    print(f"[DEBUG_MOE] SAVED {tag} -> {path} (keys: {list(data.keys())})")


# Eagerly create the base dir at import time so we know the module is loaded
os.makedirs(_DEBUG_DIR, exist_ok=True)
print(f"[DEBUG_MOE] Module loaded. Dump dir: {_DEBUG_DIR}, "
      f"target_layer={_LAYER_ID}, target_step={_STEP_ID}")


# ===================== Weight dump (in process_weights_after_loading) =====================

def debug_weight_before_pack(
    layer_id: int,
    path_selector: str,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    expert_idx: int = 0,
):
    """Dump raw weight BEFORE any packing.
    Call this at the TOP of process_weights_after_loading, before the if-else.
    """
    if not _should_debug(layer_id, "weight"):
        return
    _save(
        "weight_before_pack",
        path_selector,
        w13_weight_e0=w13_weight[expert_idx].contiguous(),
        w2_weight_e0=w2_weight[expert_idx].contiguous(),
        w13_scale_e0=w13_scale[expert_idx].contiguous(),
        w2_scale_e0=w2_scale[expert_idx].contiguous(),
        w13_shape=torch.tensor(w13_weight.shape),
        w2_shape=torch.tensor(w2_weight.shape),
        layer_id=layer_id,
    )


def debug_weight_after_pack(
    layer_id: int,
    path_selector: str,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    expert_idx: int = 0,
):
    """Dump weight AFTER packing.
    Call this at the BOTTOM of process_weights_after_loading.
    """
    if not _should_debug(layer_id, "weight"):
        return
    _save(
        "weight_after_pack",
        path_selector,
        w13_weight_e0=w13_weight[expert_idx].contiguous(),
        w2_weight_e0=w2_weight[expert_idx].contiguous(),
        w13_scale_e0=w13_scale[expert_idx].contiguous(),
        w2_scale_e0=w2_scale[expert_idx].contiguous(),
        w13_shape=torch.tensor(w13_weight.shape),
        w2_shape=torch.tensor(w2_weight.shape),
        layer_id=layer_id,
    )


# ===================== Forward pass dumps =====================

def debug_tp_forward_in(
    layer_id: int,
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    global_num_experts: int,
):
    """Dump TP-only path inputs (before GEMM call).
    Place in CompressedTensorsW8A8Int8MarlinMoEMethod.apply().
    """
    if not _should_debug(layer_id, "forward"):
        return
    _save(
        "tp_forward_in",
        "tp",
        hidden_states=hidden_states[:16].contiguous(),  # first 16 tokens
        hidden_shape=torch.tensor(hidden_states.shape),
        w13_weight_shape=torch.tensor(w13_weight.shape),
        w2_weight_shape=torch.tensor(w2_weight.shape),
        w13_scale_e0=w13_scale[0].contiguous(),
        w2_scale_e0=w2_scale[0].contiguous(),
        topk_ids=topk_ids[:16].contiguous(),
        topk_weights=topk_weights[:16].contiguous(),
        global_num_experts=global_num_experts,
        layer_id=layer_id,
    )


def debug_tp_forward_out(
    layer_id: int,
    output: torch.Tensor,
):
    """Dump TP-only path output.
    """
    if not _should_debug(layer_id, "forward"):
        return
    _save(
        "tp_forward_out",
        "tp",
        output=output[:16].contiguous(),
        output_shape=torch.tensor(output.shape),
        layer_id=layer_id,
    )


def debug_deepep_forward_in(
    layer_id: int,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_recv_tokens_per_expert,
):
    """Dump DeepEP path inputs.
    Place in forward_groupgemm_w8a8_marlin_contiguous, after all_tokens check.
    """
    if not _should_debug(layer_id, "forward"):
        return
    _save(
        "deepep_forward_in",
        "deepep",
        hidden_states=hidden_states[:16].contiguous(),
        hidden_scale=hidden_states_scale[:16].contiguous() if hidden_states_scale is not None else None,
        hidden_shape=torch.tensor(hidden_states.shape),
        w13_weight_shape=torch.tensor(w13_weight.shape),
        w2_weight_shape=torch.tensor(w2_weight.shape),
        w13_scale_e0=w13_scale[0].contiguous(),
        w2_scale_e0=w2_scale[0].contiguous(),
        topk_ids=topk_ids[:16].contiguous(),
        topk_weights=topk_weights[:16].contiguous(),
        num_tokens_per_expert=num_recv_tokens_per_expert,
        layer_id=layer_id,
    )


def debug_deepep_forward_mid(
    layer_id: int,
    gateup_output: torch.Tensor,
    q_a2_all: torch.Tensor,
    q_a2_scale: torch.Tensor,
):
    """Dump DeepEP intermediate results (after GEMM1 + silu_quant).
    """
    if not _should_debug(layer_id, "forward"):
        return
    _save(
        "deepep_forward_mid",
        "deepep",
        gateup_output=gateup_output[:16].contiguous(),
        q_a2=(
            q_a2_all[:16].contiguous() if q_a2_all.dim() >= 2
            else q_a2_all[:64].contiguous()
        ),
        q_a2_scale=q_a2_scale[:16].contiguous() if q_a2_scale is not None else None,
        layer_id=layer_id,
    )


def debug_deepep_forward_out(
    layer_id: int,
    down_output: torch.Tensor,
    gather_out: torch.Tensor,
):
    """Dump DeepEP path outputs.
    """
    if not _should_debug(layer_id, "forward"):
        return
    _save(
        "deepep_forward_out",
        "deepep",
        down_output=down_output[:16].contiguous(),
        gather_out=gather_out[:16].contiguous(),
        layer_id=layer_id,
    )
