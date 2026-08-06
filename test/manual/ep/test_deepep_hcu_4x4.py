"""Minimal HCU DeepEP low-latency reproduction for four nodes by four ranks."""

import os
import socket

import deep_ep
import torch
import torch.distributed as dist

NUM_MAX_DISPATCH_TOKENS_PER_RANK = 128
HIDDEN_SIZE = 6144
NUM_EXPERTS = 256
NUM_TOPK = 8
NUM_TEST_TOKENS = 4


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if world_size != 16 or local_world_size != 4:
        raise ValueError(
            "This reproduction requires world_size=16 and local_world_size=4; "
            f"got world_size={world_size}, local_world_size={local_world_size}."
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    group = dist.new_group(ranks=list(range(world_size)))
    print(
        f"topology rank={rank} node={socket.gethostname()} "
        f"local_rank={local_rank} device={torch.cuda.current_device()} "
        f"pid={os.getpid()}",
        flush=True,
    )

    rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
        NUM_MAX_DISPATCH_TOKENS_PER_RANK,
        HIDDEN_SIZE,
        world_size,
        NUM_EXPERTS,
        num_topk=NUM_TOPK,
    )
    print(f"stage=buffer_init_begin rank={rank} pid={os.getpid()}", flush=True)
    buffer = deep_ep.Buffer(
        group,
        num_nvl_bytes=0,
        num_rdma_bytes=rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=NUM_EXPERTS // world_size,
        allow_mnnvl=True,
        explicitly_destroy=True,
    )
    print(f"stage=buffer_init_end rank={rank} pid={os.getpid()}", flush=True)

    try:
        x = torch.full(
            (NUM_TEST_TOKENS, HIDDEN_SIZE),
            rank + 1,
            dtype=torch.bfloat16,
            device="cuda",
        )
        target_ranks = (torch.arange(NUM_TOPK, device="cuda") + rank) % world_size
        topk_idx = (target_ranks * (NUM_EXPERTS // world_size)).repeat(
            NUM_TEST_TOKENS, 1
        )
        topk_weights = torch.full(
            (NUM_TEST_TOKENS, NUM_TOPK),
            1.0 / NUM_TOPK,
            dtype=torch.float32,
            device="cuda",
        )

        print(f"stage=dispatch_begin rank={rank} pid={os.getpid()}", flush=True)
        recv_x, _, handle, _, _ = buffer.low_latency_dispatch(
            x,
            topk_idx,
            topk_weights,
            NUM_MAX_DISPATCH_TOKENS_PER_RANK,
            NUM_EXPERTS,
            quant_type=0,
        )
        print(f"stage=dispatch_end rank={rank} pid={os.getpid()}", flush=True)
        combined_x, _, _ = buffer.low_latency_combine(
            recv_x,
            topk_idx,
            topk_weights,
            handle,
        )
        torch.cuda.synchronize()

        if not torch.isfinite(combined_x).all():
            raise AssertionError(f"rank {rank} produced non-finite combine output")
        if not torch.allclose(combined_x, x, rtol=1e-3, atol=1e-3):
            max_diff = (combined_x.float() - x.float()).abs().max().item()
            raise AssertionError(f"rank {rank} combine mismatch: max_diff={max_diff}")
        print(f"PASS rank={rank}", flush=True)
    finally:
        buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
