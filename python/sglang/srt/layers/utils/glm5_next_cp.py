from typing import List, Optional

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import (
    attn_cp_all_gather_into_tensor,
    attn_cp_reduce_scatter_tensor,
    is_allocation_symmetric,
)
from sglang.srt.layers.utils.cp_utils import (
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    prepare_context_parallel_metadata,
)
from sglang.srt.runtime_context import get_parallel


def prepare_glm5_next_context_parallel_metadata(
    kv_len, cp_rank, cp_size, forward_batch
):
    from sglang.srt.layers.attention.dsa.utils import (
        is_dsa_prefill_cp_round_robin_split,
    )

    metadata = prepare_context_parallel_metadata(
        kv_len,
        cp_rank,
        cp_size,
        forward_batch.seq_lens_cpu.tolist(),
        extend_seqs_len=forward_batch.extend_seq_lens_cpu,
        device=forward_batch.input_ids.device,
    )
    if is_dsa_prefill_cp_round_robin_split():
        # GLM-Next converts between plain KDA shards and round-robin NSA
        # shards. Its gathers need the padded input length and shard sizes,
        # which main's round-robin metadata no longer populates.
        total_len = int(kv_len)
        metadata.max_rank_len = [(total_len + cp_size - 1) // cp_size] * cp_size
        metadata.per_rank_actual_token = _plain_cp_chunk_sizes(total_len, cp_size)
        metadata.total_seq_lens = total_len
    return metadata


def _cp_size(cp_size: Optional[int] = None) -> int:
    return cp_size if cp_size is not None else get_parallel().attn_cp_size


def cp_plain_split(input_: torch.Tensor, cp_size: Optional[int] = None):
    """Split a full sequence into rank-contiguous CP chunks."""
    cp_size = _cp_size(cp_size)
    cp_rank = get_parallel().attn_cp_rank
    if cp_size == 1:
        return input_
    return input_.tensor_split(cp_size, dim=0)[cp_rank].contiguous()


def _metadata_total_len(forward_batch) -> Optional[int]:
    if forward_batch is None or forward_batch.attn_cp_metadata is None:
        return None
    total_len = forward_batch.attn_cp_metadata.total_seq_lens
    if total_len is None:
        return None
    if isinstance(total_len, torch.Tensor):
        return int(total_len.item())
    return int(total_len)


def _plain_cp_chunk_sizes(total_len: int, cp_size: int) -> List[int]:
    base = total_len // cp_size
    remainder = total_len % cp_size
    return [base + int(rank < remainder) for rank in range(cp_size)]


def cp_plain_all_gather(
    input_: torch.Tensor,
    cp_size: Optional[int] = None,
    forward_batch=None,
):
    """Gather rank-contiguous CP chunks back into natural sequence order."""
    cp_size = _cp_size(cp_size)
    if cp_size == 1:
        return input_

    group = get_parallel().attn_cp_group
    local_len = int(input_.shape[0])
    total_len = _metadata_total_len(forward_batch)
    if total_len is not None:
        sizes = _plain_cp_chunk_sizes(total_len, cp_size)
        max_len = max(sizes)
        if total_len % cp_size == 0:
            with use_symmetric_memory(
                get_parallel().attn_cp_group, disabled=not is_allocation_symmetric()
            ):
                gathered = input_.new_empty(max_len * cp_size, *input_.shape[1:])
            attn_cp_all_gather_into_tensor(gathered, input_)
            # Plain CP shards are rank-contiguous, so an equal-size all-gather
            # is already in natural token order.
            return gathered
        else:
            pad_shape = (max_len - local_len, *input_.shape[1:])
            padding = input_.new_zeros(pad_shape)
            gather_input = torch.cat([input_, padding], dim=0)
            with use_symmetric_memory(
                get_parallel().attn_cp_group, disabled=not is_allocation_symmetric()
            ):
                gathered = gather_input.new_empty(
                    max_len * cp_size, *gather_input.shape[1:]
                )
            attn_cp_all_gather_into_tensor(gathered, gather_input)
        chunks = gathered.tensor_split(cp_size, dim=0)
        return torch.cat([chunk[:size] for chunk, size in zip(chunks, sizes)], dim=0)

    sizes = group.all_gather_object(local_len)
    if all(size == sizes[0] for size in sizes):
        return group.all_gather(input_, dim=0)

    max_len = max(sizes)
    if local_len < max_len:
        pad_shape = (max_len - local_len, *input_.shape[1:])
        padding = input_.new_zeros(pad_shape)
        gather_input = torch.cat([input_, padding], dim=0)
    else:
        gather_input = input_

    gathered = group.all_gather(gather_input, dim=0)
    chunks = gathered.tensor_split(cp_size, dim=0)
    return torch.cat([chunk[:size] for chunk, size in zip(chunks, sizes)], dim=0)


def cp_plain_reduce_scatter(input_: torch.Tensor, cp_size: Optional[int] = None):
    """Sum full-sequence partial results across CP ranks and return this rank's chunk."""
    cp_size = _cp_size(cp_size)
    if cp_size == 1:
        return input_
    if input_.shape[0] % cp_size == 0:
        out_shape = (input_.shape[0] // cp_size, *input_.shape[1:])
        with use_symmetric_memory(
            get_parallel().attn_cp_group, disabled=not is_allocation_symmetric()
        ):
            output = input_.new_empty(out_shape)
        attn_cp_reduce_scatter_tensor(output, input_.contiguous())
        return output

    # Uneven CP chunks cannot use tensor reduce_scatter directly. Keep the
    # generic path for padded or irregular batches.
    reduced = get_parallel().attn_cp_group.all_reduce(input_)
    return cp_plain_split(reduced, cp_size)


def _cp_round_robin_all_gather_rerange(
    input_: torch.Tensor,
    forward_batch,
    cp_size: Optional[int] = None,
):
    """Gather round-robin CP shards and restore natural token order."""
    cp_size = _cp_size(cp_size)
    if cp_size == 1:
        return input_

    local_len = int(input_.shape[0])
    cp_meta = forward_batch.attn_cp_metadata
    sizes = cp_meta.per_rank_actual_token
    max_len = cp_meta.max_rank_len[0]
    total_len = int(cp_meta.total_seq_lens)

    if total_len % cp_size == 0:
        with use_symmetric_memory(
            get_parallel().attn_cp_group, disabled=not is_allocation_symmetric()
        ):
            gathered = input_.new_empty(max_len * cp_size, *input_.shape[1:])
        attn_cp_all_gather_into_tensor(gathered, input_)
        out_shape = gathered.shape
        return (
            gathered.view(cp_size, max_len, *out_shape[1:])
            .transpose(0, 1)
            .reshape(out_shape)
        )[:total_len]

    if local_len < max_len:
        pad_shape = (max_len - local_len, *input_.shape[1:])
        padding = input_.new_zeros(pad_shape)
        gather_input = torch.cat([input_, padding], dim=0)
    else:
        gather_input = input_

    with use_symmetric_memory(
        get_parallel().attn_cp_group, disabled=not is_allocation_symmetric()
    ):
        gathered = gather_input.new_empty(max_len * cp_size, *gather_input.shape[1:])
    attn_cp_all_gather_into_tensor(gathered, gather_input)
    token_idx = torch.arange(max_len, device=input_.device).unsqueeze(1)
    ranks = torch.arange(cp_size, device=input_.device).unsqueeze(0)
    sizes_tensor = torch.tensor(sizes, device=input_.device)
    valid = token_idx < sizes_tensor.unsqueeze(0)
    gather_idx = (ranks * max_len + token_idx).masked_select(valid)
    return gathered.index_select(0, gather_idx)


def cp_plain_to_scattered(
    input_: torch.Tensor, forward_batch, cp_size: Optional[int] = None
):
    """Convert this rank's plain CP chunk to the attention-scattered CP chunk."""
    cp_size = _cp_size(cp_size)
    if cp_size == 1:
        return input_

    from sglang.srt.layers.attention.dsa.utils import (
        is_dsa_prefill_cp_round_robin_split,
    )

    local_len = int(input_.shape[0])
    total_len = _metadata_total_len(forward_batch)
    if (
        is_dsa_prefill_cp_round_robin_split()
        and total_len is not None
        and total_len % (cp_size * cp_size) == 0
        and local_len == total_len // cp_size
    ):
        tail_shape = input_.shape[1:]
        send = (
            input_.view(local_len // cp_size, cp_size, *tail_shape)
            .transpose(0, 1)
            .contiguous()
        )
        recv = torch.empty_like(send)
        torch.distributed.all_to_all_single(
            recv, send, group=get_parallel().attn_cp_group.device_group
        )
        return recv.flatten(0, 1)

    gathered = cp_plain_all_gather(input_, cp_size, forward_batch)
    return cp_split_and_rebuild_data(forward_batch, gathered)


def cp_scattered_to_plain(
    input_: torch.Tensor, forward_batch, cp_size: Optional[int] = None
):
    """Convert this rank's attention-scattered CP chunk back to plain layout."""
    cp_size = _cp_size(cp_size)
    if cp_size == 1:
        return input_

    from sglang.srt.layers.attention.dsa.utils import (
        is_dsa_prefill_cp_round_robin_split,
    )

    local_len = int(input_.shape[0])
    total_len = _metadata_total_len(forward_batch)
    if (
        is_dsa_prefill_cp_round_robin_split()
        and total_len is not None
        and total_len % (cp_size * cp_size) == 0
        and local_len == total_len // cp_size
    ):
        tail_shape = input_.shape[1:]
        send = input_.view(cp_size, local_len // cp_size, *tail_shape)
        recv = torch.empty_like(send)
        torch.distributed.all_to_all_single(
            recv, send, group=get_parallel().attn_cp_group.device_group
        )
        return recv.transpose(0, 1).contiguous().view(local_len, *tail_shape)

    gathered = cp_all_gather_rerange_output(
        input_, cp_size, forward_batch, torch.cuda.current_stream()
    )
    return cp_plain_split(gathered, cp_size)
