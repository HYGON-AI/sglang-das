# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Numerical reference checks for two-rank HCU collectives."""

import datetime
import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_hcu_ci

register_hcu_ci(
    est_time=180,
    suite="nightly-hcu-2",
    nightly=True,
)


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _collective_worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=120),
    )

    try:
        base = torch.arange(91, device=device, dtype=torch.float32).reshape(7, 13)

        reduced = base / 16 + rank
        dist.all_reduce(reduced)
        expected_reduced = base * (world_size / 16) + sum(range(world_size))
        torch.testing.assert_close(reduced, expected_reduced, rtol=0, atol=0)

        gathered_input = torch.full(
            (8,), rank + 0.25, device=device, dtype=torch.bfloat16
        )
        gathered = [torch.empty_like(gathered_input) for _ in range(world_size)]
        dist.all_gather(gathered, gathered_input)
        for source_rank, tensor in enumerate(gathered):
            expected = torch.full_like(tensor, source_rank + 0.25)
            torch.testing.assert_close(tensor, expected, rtol=0, atol=0)

        broadcast = torch.arange(16, device=device, dtype=torch.int32)
        if rank != 0:
            broadcast.zero_()
        dist.broadcast(broadcast, src=0)
        torch.testing.assert_close(
            broadcast,
            torch.arange(16, device=device, dtype=torch.int32),
            rtol=0,
            atol=0,
        )

        reduce_scatter_input = (
            torch.arange(world_size * 8, device=device, dtype=torch.float32)
            + rank * 100
        )
        reduce_scatter_output = torch.empty(8, device=device, dtype=torch.float32)
        dist.reduce_scatter_tensor(reduce_scatter_output, reduce_scatter_input)
        summed = (
            torch.arange(world_size * 8, device=device, dtype=torch.float32)
            * world_size
            + 100 * sum(range(world_size))
        )
        expected_scatter = summed[rank * 8 : (rank + 1) * 8]
        torch.testing.assert_close(
            reduce_scatter_output, expected_scatter, rtol=0, atol=0
        )
    finally:
        dist.destroy_process_group()


class TestBW1100CollectivesReferenceHCU(unittest.TestCase):
    def test_two_rank_collectives_match_reference(self):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest("requires two visible HCU devices")
        mp.spawn(
            _collective_worker,
            args=(2, _open_port()),
            nprocs=2,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
