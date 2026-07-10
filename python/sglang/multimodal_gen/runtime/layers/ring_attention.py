# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Self-contained ring attention implementation.

Replaces the dependency on torch.distributed.tensor.experimental._attention
with an embedded copy of _templated_ring_attention and all supporting classes.
Supports arbitrary seq_dim to work with any tensor layout (e.g. BSHD or BHSD).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import auto, Enum
from typing import Any, Optional, Protocol

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Enums and options
# ---------------------------------------------------------------------------

class _CausalBehavior(Enum):
    SKIP = None
    NOT_IS_CAUSAL = False
    IS_CAUSAL = True


class _RotateMethod(Enum):
    ALL_TO_ALL = auto()
    ALL_GATHER = auto()


@dataclass
class _ContextParallelOptions:
    convert_to_f32: bool = True
    enable_load_balance = True
    rotate_method: _RotateMethod = _RotateMethod.ALL_GATHER


_cp_options = _ContextParallelOptions()
_cp_options.enable_load_balance = False

def _is_causal_behavior(
    rank: int, world_size: int, i: int, is_causal: bool
) -> _CausalBehavior:
    if not is_causal:
        return _CausalBehavior.NOT_IS_CAUSAL

    if i == 0:
        return _CausalBehavior.IS_CAUSAL

    source_rank = (rank - i) % world_size
    if source_rank < rank or _cp_options.enable_load_balance:
        return _CausalBehavior.NOT_IS_CAUSAL
    else:
        return _CausalBehavior.SKIP


def _maybe_wait(tensor: torch.Tensor) -> torch.Tensor:
    """
    When tracing the code, the result tensor is not an AsyncCollectiveTensor,
    so we cannot call ``wait()``.
    """
    if isinstance(tensor, ft_c.AsyncCollectiveTensor):
        return tensor.wait()
    return tensor


def _partial_update(
    original: torch.Tensor,
    new: torch.Tensor,
    dim: int,
    n_chunks: int,
    idx: int,
    add: bool,
) -> torch.Tensor:
    chunks = list(original.chunk(n_chunks, dim=dim))
    assert chunks[idx].shape == new.shape, (original.shape, new.shape, idx)
    if add:
        chunks[idx] += new
    else:
        chunks[idx] = new
    return torch.cat(chunks, dim=dim)


# ---------------------------------------------------------------------------
# _SDPAMerger
# ---------------------------------------------------------------------------

class _SDPAMerger:
    """A class to help to merge the local SDPA result."""

    def __init__(self, convert_to_f32: bool, seq_dim: int):
        self._seq_dim = seq_dim
        self._out: Optional[torch.Tensor] = None
        self._lse: Optional[torch.Tensor] = None
        self._convert_to_f32 = convert_to_f32
        self._out_dtype = torch.float32
        self._lse_dtype = torch.float32

    def _merge_one(
        self, block_out: torch.Tensor, block_lse: torch.Tensor, partial: bool
    ) -> None:
        block_lse = block_lse.unsqueeze(dim=-1)
        if self._lse is None:
            self._lse = block_lse
            self._out = block_out
        else:
            ROUND_ROBIN_CYCLE = 2
            assert self._lse is not None
            assert self._out is not None
            lse = (
                self._lse.chunk(ROUND_ROBIN_CYCLE, dim=self._seq_dim)[1]
                if partial
                else self._lse
            )
            out = (
                self._out.chunk(ROUND_ROBIN_CYCLE, dim=self._seq_dim)[1]
                if partial
                else self._out
            )

            out = out - F.sigmoid(block_lse - lse) * (out - block_out)
            lse = lse - F.logsigmoid(lse - block_lse)
            if partial:
                self._lse = _partial_update(
                    self._lse,
                    lse,
                    dim=self._seq_dim,
                    n_chunks=ROUND_ROBIN_CYCLE,
                    idx=1,
                    add=False,
                )
                self._out = _partial_update(
                    self._out,
                    out,
                    dim=self._seq_dim,
                    n_chunks=ROUND_ROBIN_CYCLE,
                    idx=1,
                    add=False,
                )
            else:
                self._lse = lse
                self._out = out

    def step(self, out: torch.Tensor, lse: torch.Tensor, partial: bool) -> None:
        self._out_dtype = out.dtype
        self._lse_dtype = lse.dtype

        if self._convert_to_f32:
            out = out.to(torch.float32)
            lse = lse.to(torch.float32)

        self._merge_one(out, lse, partial)

    def results(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._out is not None
        assert self._lse is not None
        out, lse = self._out, self._lse.squeeze(-1)
        return out.to(self._out_dtype), lse.to(self._lse_dtype)


# ---------------------------------------------------------------------------
# _AttentionOp protocol
# ---------------------------------------------------------------------------

class _AttentionOp(Protocol):
    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, ...]: ...


# ---------------------------------------------------------------------------
# Ring rotater classes
# ---------------------------------------------------------------------------

class _RingRotater(ABC):
    @abstractmethod
    def __init__(self, pg: dist.ProcessGroup, seq_dim: int) -> None: ...

    @abstractmethod
    def exchange_buffers(self, curr_buffer: torch.Tensor) -> None: ...

    @abstractmethod
    def next_buffer(self) -> torch.Tensor: ...


class _AllToAllRotater(_RingRotater):
    """Use all_to_all to send the kv to the next rank"""

    def __init__(self, pg: dist.ProcessGroup, seq_dim: int) -> None:
        self._pg = pg
        self._seq_dim = seq_dim
        self._buffer: Optional[torch.Tensor] = None

    def exchange_buffers(self, curr_buffer: torch.Tensor) -> None:
        curr_buffer = curr_buffer.contiguous()
        size = dist.get_world_size(self._pg)
        dsts = list(range(1, size)) + [0]
        self._buffer = ft_c.permute_tensor(curr_buffer, dsts, self._pg)

    def next_buffer(self) -> torch.Tensor:
        assert self._buffer is not None
        return _maybe_wait(self._buffer)


class _AllGatherRotater(_RingRotater):
    """
    Allgather the kv and return the only the required kv.
    Only one communication will be done.
    """

    def __init__(self, pg: dist.ProcessGroup, seq_dim: int) -> None:
        self._pg = pg
        self._seq_dim = seq_dim
        self._aggregated_buffer: Optional[torch.Tensor] = None
        self._idx = 0

    def exchange_buffers(self, curr_buffer: torch.Tensor) -> None:
        self._idx += 1
        if self._aggregated_buffer is None:
            self._aggregated_buffer = ft_c.all_gather_tensor(
                curr_buffer.contiguous(), gather_dim=0, group=self._pg
            )

    def next_buffer(self) -> torch.Tensor:
        rank = dist.get_rank(self._pg)
        idx = rank - self._idx

        assert self._aggregated_buffer is not None
        self._aggregated_buffer = _maybe_wait(self._aggregated_buffer)
        return self._aggregated_buffer.chunk(dist.get_world_size(self._pg))[idx]


def _create_rotater(
    pg: dist.ProcessGroup, seq_dim: int, method: Optional[_RotateMethod] = None
) -> _RingRotater:
    if method is None:
        method = _cp_options.rotate_method

    if method == _RotateMethod.ALL_TO_ALL:
        return _AllToAllRotater(pg, seq_dim)
    elif method == _RotateMethod.ALL_GATHER:
        return _AllGatherRotater(pg, seq_dim)
    else:
        raise NotImplementedError(f"Unknown method {method}")


# ---------------------------------------------------------------------------
# _templated_ring_attention — main entry point
# ---------------------------------------------------------------------------

def _templated_ring_attention(
    group: dist.ProcessGroup,
    seq_dim: int,
    op: _AttentionOp,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    **kwargs: object,
) -> tuple[torch.Tensor, ...]:
    if is_causal and (query.size(seq_dim) != key.size(seq_dim)):
        raise NotImplementedError(
            "is_causal requires the same query and context sequence lengths"
        )
    if not is_causal and _cp_options.enable_load_balance:
        raise RuntimeError("Load balancing requires `is_causal=True`.")

    assert isinstance(group, dist.ProcessGroup), (
        "process group must be single dimension"
    )
    rank = dist.get_rank(group)
    size = dist.get_world_size(group)

    next_kv = None

    key = key.contiguous()
    value = value.contiguous()

    sdpa_merger = _SDPAMerger(_cp_options.convert_to_f32, seq_dim=seq_dim)

    rest: list[Any]
    out: torch.Tensor
    logsumexp: torch.Tensor

    rotater = _create_rotater(group, seq_dim, _RotateMethod.ALL_TO_ALL)

    for i in range(size):
        if i > 0:
            next_kv = rotater.next_buffer()
            key = next_kv[: key.numel()].reshape(key.shape)
            value = next_kv[key.numel() :].reshape(value.shape)

        if i < (size - 1):
            next_kv = torch.cat([key.flatten(), value.flatten()])
            next_kv = rotater.exchange_buffers(next_kv)

        is_causal_behavior = _is_causal_behavior(
            rank=rank, world_size=size, i=i, is_causal=is_causal
        )

        if is_causal_behavior == _CausalBehavior.SKIP:
            continue

        if i == 0 or (not _cp_options.enable_load_balance or not is_causal):
            q, k, v, partial = (query, key, value, False)
        elif i <= rank:
            ROUND_ROBIN_CYCLE = 2
            q, k, v, partial = (
                query,
                key.chunk(ROUND_ROBIN_CYCLE, dim=seq_dim)[0],
                value.chunk(ROUND_ROBIN_CYCLE, dim=seq_dim)[0],
                False,
            )
        else:
            q, k, v, partial = query.chunk(2, dim=seq_dim)[1], key, value, True

        out, logsumexp, *rest = op(
            q,
            k,
            v,
            is_causal=is_causal_behavior.value,
            **kwargs,
        )
        sdpa_merger.step(out, logsumexp, partial)

    return *sdpa_merger.results(), *rest


# ---------------------------------------------------------------------------
# Overlap-optimized ring attention with timing instrumentation
# ---------------------------------------------------------------------------

class _AllToAllRotaterOverlap:
    """Ring-shift rotater with true GPU-level compute-communication overlap.

    Uses a dedicated CUDA stream for NCCL communication so that KV exchange
    and attention compute run on different hardware queues concurrently.
    ``start_exchange`` launches the permute_tensor on ``comm_stream`` and
    returns without blocking the default stream.  ``finish_exchange`` re-joins
    the default stream after compute has finished.
    """

    def __init__(self, pg: dist.ProcessGroup, seq_dim: int) -> None:
        self._pg = pg
        self._comm_stream = torch.cuda.Stream()
        self._buffer: Optional[torch.Tensor] = None

    def start_exchange(self, kv_flat: torch.Tensor) -> None:
        kv_flat = kv_flat.contiguous()
        size = dist.get_world_size(self._pg)
        dsts = list(range(1, size)) + [0]

        # comm_stream must wait for kv_flat to be ready on default stream
        self._comm_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self._comm_stream):
            self._buffer = ft_c.permute_tensor(kv_flat, dsts, self._pg)

    def finish_exchange(self) -> torch.Tensor:
        # default stream waits for comm_stream to finish
        torch.cuda.current_stream().wait_stream(self._comm_stream)
        assert self._buffer is not None
        result = _maybe_wait(self._buffer)
        self._buffer = None
        return result


def _templated_ring_attention_overlap(
    group: dist.ProcessGroup,
    seq_dim: int,
    op: _AttentionOp,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    **kwargs: object,
) -> tuple[torch.Tensor, ...]:
    """Ring attention with compute-communication overlap and timing.

    The loop is restructured so that the async KV exchange (start_exchange)
    is launched BEFORE attention compute, and the wait (finish_exchange)
    happens AFTER compute.  This lets the NCCL communication stream and the
    default compute stream run concurrently.

    Timing is recorded with CUDA events and logged at the end.
    """
    if is_causal and (query.size(seq_dim) != key.size(seq_dim)):
        raise NotImplementedError(
            "is_causal requires the same query and context sequence lengths"
        )
    if not is_causal and _cp_options.enable_load_balance:
        raise RuntimeError("Load balancing requires `is_causal=True`.")

    assert isinstance(group, dist.ProcessGroup), (
        "process group must be single dimension"
    )
    rank = dist.get_rank(group)
    size = dist.get_world_size(group)

    key = key.contiguous()
    value = value.contiguous()

    sdpa_merger = _SDPAMerger(_cp_options.convert_to_f32, seq_dim=seq_dim)
    rotater = _AllToAllRotaterOverlap(group, seq_dim)

    rest: list[Any]
    out: torch.Tensor
    logsumexp: torch.Tensor

    for i in range(size):
        # ---- 1. Launch async KV exchange BEFORE compute ----
        if i < size - 1:
            kv_flat = torch.cat([key.flatten(), value.flatten()])
            rotater.start_exchange(kv_flat)

        # ---- 2. Attention compute (overlaps with exchange) ----
        is_causal_behavior = _is_causal_behavior(
            rank=rank, world_size=size, i=i, is_causal=is_causal
        )

        if is_causal_behavior != _CausalBehavior.SKIP:
            if i == 0 or (not _cp_options.enable_load_balance or not is_causal):
                q, k, v, partial = query, key, value, False
            elif i <= rank:
                q, k, v, partial = (
                    query,
                    key.chunk(2, dim=seq_dim)[0],
                    value.chunk(2, dim=seq_dim)[0],
                    False,
                )
            else:
                q, k, v, partial = (
                    query.chunk(2, dim=seq_dim)[1],
                    key,
                    value,
                    True,
                )

            out, logsumexp, *rest = op(
                q, k, v, is_causal=is_causal_behavior.value, **kwargs
            )
            sdpa_merger.step(out, logsumexp, partial)

        # ---- 3. Wait for exchange AFTER compute ----
        if i < size - 1:
            next_kv = rotater.finish_exchange()
            key = next_kv[: key.numel()].reshape(key.shape)
            value = next_kv[key.numel() :].reshape(value.shape)

    return *sdpa_merger.results(), *rest
