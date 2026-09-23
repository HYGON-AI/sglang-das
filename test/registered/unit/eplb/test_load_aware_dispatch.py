from __future__ import annotations

import torch

from sglang.srt.eplb.load_aware_dispatch import (
    build_load_aware_probabilities_cpu_reference,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _expected_rank_loads(counts, mapping, probabilities, num_ranks):
    num_physical = int(mapping[mapping >= 0].max()) + 1
    physical_per_rank = num_physical // num_ranks
    loads = torch.zeros(num_ranks, dtype=torch.float32)
    for logical_id in range(mapping.shape[0]):
        for copy_index in range(mapping.shape[1]):
            physical_id = int(mapping[logical_id, copy_index])
            if physical_id < 0:
                continue
            rank = physical_id // physical_per_rank
            loads[rank] += counts[logical_id] * probabilities[logical_id, copy_index]
    return loads


def test_load_aware_probabilities_are_normalized_and_ignore_padding():
    # Four physical experts per rank. Logical 0 has two copies on rank 0 and
    # one on rank 1; the probability is split evenly between same-rank copies.
    mapping = torch.tensor(
        [
            [0, 1, 4],
            [2, -1, -1],
            [5, -1, -1],
            [6, 7, -1],
        ],
        dtype=torch.int64,
    )
    counts = torch.tensor([100, 40, 20, 0], dtype=torch.float32)
    probabilities = build_load_aware_probabilities_cpu_reference(
        counts, mapping, num_ranks=2
    )

    valid = mapping >= 0
    assert torch.allclose((probabilities * valid).sum(dim=1), torch.ones(4))
    assert torch.count_nonzero(probabilities[~valid]) == 0
    assert torch.allclose(probabilities[0, 0], probabilities[0, 1])


def test_load_aware_waterfill_reduces_the_slowest_rank():
    # Rank 0 owns the hot single-copy expert. The replicated hot expert can be
    # moved to rank 1, so water-filling should lower the critical rank load.
    mapping = torch.tensor(
        [
            [0, -1],
            [1, 4],
            [5, -1],
            [7, -1],
        ],
        dtype=torch.int64,
    )
    counts = torch.tensor([100, 100, 10, 10], dtype=torch.float32)
    probabilities = build_load_aware_probabilities_cpu_reference(
        counts, mapping, num_ranks=2
    )
    balanced_loads = _expected_rank_loads(counts, mapping, probabilities, 2)

    primary_probabilities = torch.zeros_like(probabilities)
    primary_probabilities[:, 0] = 1
    primary_loads = _expected_rank_loads(counts, mapping, primary_probabilities, 2)
    assert balanced_loads.max() < primary_loads.max()
    assert torch.allclose(balanced_loads, torch.tensor([110.0, 110.0]), atol=1e-3)


def test_largest_replicated_expert_is_balanced_first():
    mapping = torch.tensor(
        [
            [0, 4],
            [1, 5],
            [2, -1],
            [7, -1],
        ],
        dtype=torch.int64,
    )
    counts = torch.tensor([120, 20, 80, 0], dtype=torch.float32)
    probabilities = build_load_aware_probabilities_cpu_reference(
        counts, mapping, num_ranks=2
    )
    loads = _expected_rank_loads(counts, mapping, probabilities, 2)
    assert float(loads.max() - loads.min()) < 1e-3
