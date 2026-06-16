import argparse
import importlib
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))


def import_pack_module():
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires an available GPU")

    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(
        ServerArgs(model_path="dummy", pack_paged_kv_to_varlen="on")
    )
    return importlib.import_module("sglang.srt.layers.attention.pack_paged_kv_to_varlen")


def measure(fn, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench_case(
    pack_module,
    batch_size,
    kv_len,
    hit_rate,
    page_size,
    num_heads,
    head_dim,
    warmups,
    iters,
):
    query_len = max(1, int(round(kv_len * (1.0 - hit_rate))))
    seq_lens = torch.full((batch_size,), kv_len, dtype=torch.int32)
    total_query_tokens = batch_size * query_len
    max_pages_per_seq = (kv_len + page_size - 1) // page_size
    num_pages = batch_size * max_pages_per_seq

    page_table = torch.arange(num_pages, dtype=torch.int32, device="cuda").reshape(
        batch_size, max_pages_per_seq
    )
    cu_seqlens_q = torch.arange(
        0,
        total_query_tokens + 1,
        step=query_len,
        dtype=torch.int32,
        device="cuda",
    )
    cu_seqlens_k = torch.nn.functional.pad(
        torch.cumsum(seq_lens.to(device="cuda"), dim=0, dtype=torch.int32), (1, 0)
    )
    q = torch.randn(
        total_query_tokens,
        num_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    key_cache = torch.randn(
        num_pages,
        num_heads,
        page_size,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    value_cache = torch.randn(
        num_pages,
        num_heads,
        head_dim,
        page_size,
        dtype=torch.float16,
        device="cuda",
    )
    seqused_k = seq_lens.to(device="cuda")
    softmax_scale = head_dim**-0.5

    from sglang.srt.layers.attention.flashattention_interface import (
        flash_attn_varlen_func,
        vllm_flash_attn_varlen_func,
    )

    def run_baseline():
        return vllm_flash_attn_varlen_func(
            q=q,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=query_len,
            seqused_k=seqused_k,
            max_seqlen_k=kv_len,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=(-1, -1),
            block_table=page_table,
            fa_version=2,
            q_descale=None,
            k_descale=None,
            v_descale=None,
        )

    def run_pack_only():
        return pack_module.pack_paged_kv_to_varlen(
            key_cache, value_cache, page_table, seq_lens, page_size
        )

    def run_packed():
        packed_k, packed_v = run_pack_only()
        return flash_attn_varlen_func(
            q=q,
            k=packed_k,
            v=packed_v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=query_len,
            max_seqlen_k=kv_len,
            softmax_scale=softmax_scale,
            causal=True,
        )

    for _ in range(warmups):
        run_baseline()
        run_packed()
    torch.cuda.synchronize()

    baseline_ms = measure(run_baseline, iters)
    packed_ms = measure(run_packed, iters)
    pack_only_ms = measure(run_pack_only, iters)
    return {
        "batch_size": batch_size,
        "kv_len": kv_len,
        "hit_rate": hit_rate,
        "query_len": query_len,
        "baseline_ms": baseline_ms,
        "packed_ms": packed_ms,
        "pack_only_ms": pack_only_ms,
        "speedup": baseline_ms / packed_ms,
    }


def bench_kernel_gap_case(
    pack_module,
    batch_size,
    q_len,
    total_kv_len,
    q_head,
    kv_head,
    page_size,
    head_dim,
    warmups,
    iters,
):
    from flash_attn import flash_attn_varlen_func, vllm_flash_attn_varlen_func

    q = torch.randn(
        batch_size, q_len, q_head, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    flat_k = torch.randn(
        batch_size, total_kv_len, kv_head, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    flat_v = torch.randn_like(flat_k)

    q_unpad = q.reshape(batch_size * q_len, q_head, head_dim)
    k_unpad = flat_k.reshape(batch_size * total_kv_len, kv_head, head_dim).contiguous()
    v_unpad = flat_v.reshape(batch_size * total_kv_len, kv_head, head_dim).contiguous()
    cu_seqlens_q = torch.arange(
        0,
        (batch_size + 1) * q_len,
        step=q_len,
        dtype=torch.int32,
        device="cuda",
    )
    cu_seqlens_k = torch.arange(
        0,
        (batch_size + 1) * total_kv_len,
        step=total_kv_len,
        dtype=torch.int32,
        device="cuda",
    )
    blocks_per_seq = total_kv_len // page_size
    num_blocks = batch_size * blocks_per_seq
    k_cache = (
        flat_k.reshape(num_blocks, page_size, kv_head, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    v_cache = (
        flat_v.reshape(num_blocks, page_size, kv_head, head_dim)
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    block_table = torch.arange(
        num_blocks, dtype=torch.int32, device="cuda"
    ).reshape(batch_size, blocks_per_seq)
    seqused_k = torch.full(
        (batch_size,), total_kv_len, dtype=torch.int32, device="cuda"
    )
    seq_lens_cpu = torch.full((batch_size,), total_kv_len, dtype=torch.int32)
    softmax_scale = head_dim**-0.5

    def run_ideal_varlen():
        return flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            q_len,
            total_kv_len,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=True,
        )

    def run_paged():
        return vllm_flash_attn_varlen_func(
            q=q_unpad,
            k=k_cache,
            v=v_cache,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=q_len,
            seqused_k=seqused_k,
            max_seqlen_k=total_kv_len,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_table,
            fa_version=2,
        )

    def run_pack_only():
        return pack_module.pack_paged_kv_to_varlen(
            k_cache, v_cache, block_table, seq_lens_cpu, page_size
        )

    def run_packed_total():
        packed_k, packed_v = run_pack_only()
        return flash_attn_varlen_func(
            q_unpad,
            packed_k,
            packed_v,
            cu_seqlens_q,
            cu_seqlens_k,
            q_len,
            total_kv_len,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=True,
        )

    for _ in range(warmups):
        run_ideal_varlen()
        run_paged()
        run_packed_total()
    torch.cuda.synchronize()

    ideal_ms = measure(run_ideal_varlen, iters)
    paged_ms = measure(run_paged, iters)
    pack_only_ms = measure(run_pack_only, iters)
    packed_total_ms = measure(run_packed_total, iters)
    return {
        "batch_size": batch_size,
        "q_len": q_len,
        "total_kv_len": total_kv_len,
        "q_head": q_head,
        "kv_head": kv_head,
        "ideal_varlen_ms": ideal_ms,
        "paged_ms": paged_ms,
        "pack_only_ms": pack_only_ms,
        "packed_total_ms": packed_total_ms,
        "ideal_speedup": paged_ms / ideal_ms,
        "packed_speedup": paged_ms / packed_total_ms,
        "pack_budget_ms": paged_ms - ideal_ms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--kernel-gap",
        action="store_true",
        help="Benchmark ideal varlen, paged varlen, and packed-total paths.",
    )
    args = parser.parse_args()

    pack_module = import_pack_module()
    if args.kernel_gap:
        cases = [
            (batch_size, q_len, total_kv_len, q_head, kv_head)
            for kv_head, q_head in ((1, 6), (2, 8), (4, 8), (8, 8))
            for batch_size in (1, 2, 4)
            for q_len, total_kv_len in ((12800, 51200), (16384, 65536))
        ]
        for batch_size, q_len, total_kv_len, q_head, kv_head in cases:
            result = bench_kernel_gap_case(
                pack_module,
                batch_size,
                q_len,
                total_kv_len,
                q_head,
                kv_head,
                args.page_size,
                args.head_dim,
                args.warmups,
                args.iters,
            )
            print(
                "bs={batch_size} qh={q_head} kvh={kv_head} q={q_len} "
                "total={total_kv_len} ideal_varlen_ms={ideal_varlen_ms:.3f} "
                "paged_ms={paged_ms:.3f} pack_only_ms={pack_only_ms:.3f} "
                "packed_total_ms={packed_total_ms:.3f} "
                "ideal_speedup={ideal_speedup:.3f} "
                "packed_speedup={packed_speedup:.3f} "
                "pack_budget_ms={pack_budget_ms:.3f}".format(**result)
            )
        return

    cases = [
        (bs, kv_len, hit_rate)
        for bs in (1, 2)
        for kv_len in (51200, 65536, 81920)
        for hit_rate in (0.70, 0.75, 0.90, 0.99)
    ]
    for batch_size, kv_len, hit_rate in cases:
        result = bench_case(
            pack_module,
            batch_size,
            kv_len,
            hit_rate,
            args.page_size,
            args.num_heads,
            args.head_dim,
            args.warmups,
            args.iters,
        )
        print(
            "bs={batch_size} kv={kv_len} hit={hit_rate:.2f} q={query_len} "
            "baseline_ms={baseline_ms:.3f} packed_ms={packed_ms:.3f} "
            "pack_only_ms={pack_only_ms:.3f} speedup={speedup:.3f}".format(
                **result
            )
        )


if __name__ == "__main__":
    main()
