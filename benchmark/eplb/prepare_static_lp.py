"""Precompute replica dispatch weights on an existing static EPLB placement."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linprog


def prepare(layout_path: Path, counts_path: Path, ep_size: int, output: Path):
    layout = torch.load(layout_path, map_location="cpu", weights_only=True)
    observations = torch.load(counts_path, map_location="cpu", weights_only=True)
    mapping = layout["physical_to_logical_map"]
    counts = observations.get("logical_count", observations.get("logical_counts"))
    if not isinstance(mapping, torch.Tensor) or mapping.ndim != 2:
        raise ValueError("physical_to_logical_map must be a [layers, physical] tensor")
    if mapping.dtype not in (torch.int32, torch.int64):
        raise ValueError("physical_to_logical_map must contain integer expert IDs")
    if not isinstance(counts, torch.Tensor) or counts.ndim not in (2, 3):
        raise ValueError(
            "logical_count must have shape [layers, experts] or [steps, layers, experts]"
        )
    counts = counts.double()
    if not bool(torch.isfinite(counts).all() and (counts >= 0).all()):
        raise ValueError("Expert counts must be finite and nonnegative")
    if counts.ndim == 3:
        counts = counts.sum(dim=0)
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
    if not bool(torch.isfinite(counts.sum(dim=1)).all()):
        raise ValueError("Aggregated expert counts overflowed")

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
    print(json.dumps({"layers": num_layers, "artifact": str(output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout",
        type=Path,
        required=True,
        help="PT containing physical_to_logical_map",
    )
    parser.add_argument(
        "--counts",
        type=Path,
        required=True,
        help="EPLB recorder PT containing logical_count",
    )
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    prepare(args.layout, args.counts, args.ep_size, args.output)


if __name__ == "__main__":
    main()
