"""JIT helpers for DSA kpool write-plan replay updates."""

from __future__ import annotations

import logging

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

logger = logging.getLogger(__name__)
_DUMMY_INT32_CACHE: dict[tuple[torch.device, int], torch.Tensor] = {}


def _dummy_int32_tensor(ref: torch.Tensor, size: int) -> torch.Tensor:
    key = (ref.device, size)
    dummy = _DUMMY_INT32_CACHE.get(key)
    if dummy is None:
        dummy = torch.empty((size,), dtype=torch.int32, device=ref.device)
        _DUMMY_INT32_CACHE[key] = dummy
    return dummy


@cache_once
def _jit_kpool_write_plan_module(num_draft_tokens: int, has_per_q: bool):
    args = make_cpp_args(num_draft_tokens, has_per_q)
    try:
        return load_jit(
            "kpool_write_plan",
            *args,
            cuda_files=["elementwise/kpool_write_plan.cuh"],
            cuda_wrappers=[
                (
                    "kpool_write_plan",
                    f"sglang::KPoolWritePlanKernel<{args}>::run",
                )
            ],
        )
    except Exception as e:
        logger.error(
            "Failed to compile JIT kpool write-plan "
            "(num_draft_tokens=%s, has_per_q=%s): %s",
            num_draft_tokens,
            has_per_q,
            e,
        )
        raise


@cache_once
def _jit_kpool_write_plan_multi_decode_module():
    try:
        return load_jit(
            "kpool_write_plan_multi_decode",
            cuda_files=["elementwise/kpool_write_plan.cuh"],
            cuda_wrappers=[
                (
                    "kpool_write_plan_multi_decode",
                    "sglang::KPoolWritePlanMultiDecodeKernel::run",
                )
            ],
        )
    except Exception as e:
        logger.error("Failed to compile JIT kpool write-plan multi decode: %s", e)
        raise


def kpool_write_plan_cuda(
    write_start: torch.Tensor,
    req_pool_indices: torch.Tensor,
    real_page_table: torch.Tensor,
    req_out: torch.Tensor,
    write_start_out: torch.Tensor,
    tail_logical_start_out: torch.Tensor,
    write_loc_out: torch.Tensor,
    pool_seqlens_per_q_out: torch.Tensor | None,
    seqlens_per_q_out: torch.Tensor | None,
    pool_size: int,
    num_draft_tokens: int,
    slots_per_page: int,
) -> None:
    has_per_q = pool_seqlens_per_q_out is not None
    assert has_per_q == (seqlens_per_q_out is not None), (
        "pool_seqlens_per_q_out and seqlens_per_q_out must be both set or both None"
    )

    module = _jit_kpool_write_plan_module(num_draft_tokens, has_per_q)
    per_q_size = max(1, write_start.shape[0] * num_draft_tokens)
    pool_seqlens_arg = (
        pool_seqlens_per_q_out
        if pool_seqlens_per_q_out is not None
        else _dummy_int32_tensor(write_start, per_q_size)
    )
    seqlens_arg = (
        seqlens_per_q_out
        if seqlens_per_q_out is not None
        else _dummy_int32_tensor(write_start, per_q_size)
    )
    module.kpool_write_plan(
        write_start,
        req_pool_indices,
        real_page_table,
        req_out,
        write_start_out,
        tail_logical_start_out,
        write_loc_out,
        pool_seqlens_arg,
        seqlens_arg,
        pool_size,
        slots_per_page,
    )


def kpool_write_plan_multi_decode_cuda(
    write_start: torch.Tensor,
    req_pool_indices: torch.Tensor,
    real_page_table: torch.Tensor,
    req_out0: torch.Tensor,
    write_start_out0: torch.Tensor,
    tail_logical_start_out0: torch.Tensor,
    write_loc_out0: torch.Tensor,
    req_out1: torch.Tensor,
    write_start_out1: torch.Tensor,
    tail_logical_start_out1: torch.Tensor,
    write_loc_out1: torch.Tensor,
    req_out2: torch.Tensor,
    write_start_out2: torch.Tensor,
    tail_logical_start_out2: torch.Tensor,
    write_loc_out2: torch.Tensor,
    req_out3: torch.Tensor,
    write_start_out3: torch.Tensor,
    tail_logical_start_out3: torch.Tensor,
    write_loc_out3: torch.Tensor,
    pool_size: int,
    slots_per_page: int,
) -> None:
    module = _jit_kpool_write_plan_multi_decode_module()
    module.kpool_write_plan_multi_decode(
        write_start,
        req_pool_indices,
        real_page_table,
        req_out0,
        write_start_out0,
        tail_logical_start_out0,
        write_loc_out0,
        req_out1,
        write_start_out1,
        tail_logical_start_out1,
        write_loc_out1,
        req_out2,
        write_start_out2,
        tail_logical_start_out2,
        write_loc_out2,
        req_out3,
        write_start_out3,
        tail_logical_start_out3,
        write_loc_out3,
        pool_size,
        slots_per_page,
    )
