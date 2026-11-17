"""Generate a static EPLB placement and its offline LP replica dispatch weights."""

import argparse
import json
from pathlib import Path
from runpy import run_path

import numpy as np
import torch
from scipy.optimize import linprog


def generate_layout(
    counts, ep_size, redundant_experts, num_nodes, num_groups, hierarchical
):
    num_logical = counts.shape[1]
    if redundant_experts is None or redundant_experts < 0:
        raise ValueError("Layout generation requires nonnegative --redundant-experts")
    num_physical = num_logical + redundant_experts
    if num_physical % ep_size:
        raise ValueError("Physical experts must divide evenly among EP ranks")
    if num_nodes <= 0 or ep_size % num_nodes:
        raise ValueError("EP size must be divisible by a positive --num-nodes")
    if num_groups <= 0 or num_logical % num_groups:
        raise ValueError("Logical experts must be divisible by a positive --num-groups")
    if hierarchical and num_groups % num_nodes:
        raise ValueError("Hierarchical placement requires groups divisible by nodes")

    balance_counts = counts.clone()
    balance_counts[counts.sum(dim=1) == 0] = 1
    if not bool(torch.isfinite(balance_counts.float()).all()):
        raise ValueError("Expert counts overflowed the EPLB float32 input")
    source = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/srt/eplb/eplb_algorithms/deepseek.py"
    )
    # Load the repository's CPU algorithm without importing the serving runtime.
    rebalance_experts = run_path(str(source))["rebalance_experts"]
    mapping, _, _ = rebalance_experts(
        weight=balance_counts,
        num_replicas=num_physical,
        num_groups=num_groups,
        num_nodes=num_nodes,
        num_gpus=ep_size,
        enable_hierarchical=hierarchical,
    )
    return mapping


def prepare(
    layout_path: Path | None,
    counts_path: Path,
    ep_size: int,
    output: Path,
    *,
    layout_output: Path | None = None,
    redundant_experts: int | None = None,
    num_nodes: int = 1,
    num_groups: int = 1,
    hierarchical: bool = False,
):
    if ep_size <= 0:
        raise ValueError("EP size must be positive")
    if layout_path is not None and (
        layout_output is not None
        or redundant_experts is not None
        or num_nodes != 1
        or num_groups != 1
        or hierarchical
    ):
        raise ValueError("Layout generation options cannot be combined with --layout")
    if layout_path is None:
        layout_output = layout_output or output.with_name("static_mapping.pt")
    inputs = {counts_path.resolve()}
    if layout_path is not None:
        inputs.add(layout_path.resolve())
    outputs = [output.resolve(), output.with_suffix(".json").resolve()]
    if layout_output is not None:
        outputs.append(layout_output.resolve())
    if len(set(outputs)) != len(outputs) or inputs.intersection(outputs):
        raise ValueError(
            "Output files must be distinct from each other and input files"
        )

    observations = torch.load(counts_path, map_location="cpu", weights_only=True)
    counts = observations.get("logical_count", observations.get("logical_counts"))
    if not isinstance(counts, torch.Tensor) or counts.ndim not in (2, 3):
        raise ValueError(
            "logical_count must have shape [layers, experts] or [steps, layers, experts]"
        )
    counts = counts.double()
    if not bool(torch.isfinite(counts).all() and (counts >= 0).all()):
        raise ValueError("Expert counts must be finite and nonnegative")
    if counts.ndim == 3:
        counts = counts.sum(dim=0)
    if min(counts.shape) <= 0:
        raise ValueError("Expert count dimensions must be nonempty")
    if not bool(torch.isfinite(counts.sum(dim=1)).all()):
        raise ValueError("Aggregated expert counts overflowed")

    if layout_path is None:
        mapping = generate_layout(
            counts, ep_size, redundant_experts, num_nodes, num_groups, hierarchical
        )
    else:
        layout = torch.load(layout_path, map_location="cpu", weights_only=True)
        mapping = layout["physical_to_logical_map"]
    if not isinstance(mapping, torch.Tensor) or mapping.ndim != 2:
        raise ValueError("physical_to_logical_map must be a [layers, physical] tensor")
    if mapping.dtype not in (torch.int32, torch.int64):
        raise ValueError("physical_to_logical_map must contain integer expert IDs")
    num_layers, num_physical = mapping.shape
    num_logical = counts.shape[1]
    if min(num_layers, num_physical, num_logical) <= 0:
        raise ValueError("Expert layout and count dimensions must be nonempty")
    if counts.shape[0] != num_layers:
        raise ValueError("Expert counts and layout must have the same number of layers")
    if ep_size <= 0 or num_physical % ep_size:
        raise ValueError("Physical experts must divide evenly among positive EP ranks")
    if bool(((mapping < 0) | (mapping >= num_logical)).any()):
        raise ValueError("Physical layout contains an out-of-range logical expert ID")
    num_local = num_physical // ep_size
    physical_ids = np.arange(num_physical)
    upper = np.zeros((ep_size, num_physical + 1), dtype=np.float64)
    upper[physical_ids // num_local, physical_ids] = 1
    upper[:, -1] = -1
    objective = np.zeros(num_physical + 1)
    objective[-1] = 1
    rows, masses = [], []
    for layer, physical in enumerate(mapping):
        owners = physical.numpy()
        demand = counts[layer].numpy()
        demand = (
            demand / demand.sum()
            if demand.sum()
            else np.ones(num_logical) / num_logical
        )
        copies = np.bincount(owners, minlength=num_logical)
        if np.any(copies == 0):
            raise ValueError(
                f"Layer {layer} has a logical expert without any physical replica"
            )
        equality = np.zeros((num_logical, num_physical + 1), dtype=np.float64)
        equality[owners, physical_ids] = 1
        solution = linprog(
            objective,
            A_ub=upper,
            b_ub=np.zeros(ep_size),
            A_eq=equality,
            b_eq=demand,
            bounds=(0, None),
            method="highs",
        )
        if not solution.success:
            raise RuntimeError(f"LP failed at layer {layer}: {solution.message}")
        mass = solution.x[:-1]
        residual = float(np.max(np.abs(equality[:, :-1] @ mass - demand)))
        if not np.isfinite(mass).all() or np.any(mass < 0) or residual > 1e-7:
            raise RuntimeError(
                f"Invalid LP result at layer {layer}: residual={residual}"
            )
        uniform = demand[owners] / copies[owners]
        masses.append(torch.from_numpy(mass).float())
        rows.append(
            {
                "layer": layer,
                "max_rank_fraction_lp": float(
                    mass.reshape(ep_size, num_local).sum(axis=1).max()
                ),
                "max_rank_fraction_uniform_replicas": float(
                    uniform.reshape(ep_size, num_local).sum(axis=1).max()
                ),
                "equality_residual": residual,
            }
        )
    if layout_output is not None:
        layout_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"physical_to_logical_map": mapping}, layout_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"physical_to_logical_map": mapping, "physical_mass": torch.stack(masses)},
        output,
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "ep_size": ep_size,
                "method": "Offline exact LP over existing replicas; minimize maximum rank token mass.",
                "limitation": "Uses historical counts; does not optimize network locality or future workload drift.",
                "layers": rows,
            },
            indent=2,
        )
        + "\n"
    )
    result = {"layers": num_layers, "artifact": str(output)}
    if layout_output is not None:
        result["layout"] = str(layout_output)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    layout_options = parser.add_mutually_exclusive_group()
    layout_options.add_argument(
        "--layout",
        type=Path,
        help="Reuse an existing PT containing physical_to_logical_map",
    )
    layout_options.add_argument(
        "--layout-output",
        type=Path,
        help="Generated layout PT (default: static_mapping.pt beside --output)",
    )
    parser.add_argument(
        "--redundant-experts",
        type=int,
        help="Redundant experts per layer; required when generating a layout",
    )
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument(
        "--hierarchical", action="store_true", help="Use hierarchical EPLB placement"
    )
    parser.add_argument(
        "--counts",
        type=Path,
        required=True,
        help="EPLB recorder PT containing logical_count",
    )
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("static_lp_probabilities.pt")
    )
    args = parser.parse_args()
    if args.layout is None and args.redundant_experts is None:
        parser.error("--redundant-experts is required when --layout is omitted")
    torch.set_num_threads(4)
    prepare(
        args.layout,
        args.counts,
        args.ep_size,
        args.output,
        layout_output=args.layout_output,
        redundant_experts=args.redundant_experts,
        num_nodes=args.num_nodes,
        num_groups=args.num_groups,
        hierarchical=args.hierarchical,
    )


if __name__ == "__main__":
    main()
