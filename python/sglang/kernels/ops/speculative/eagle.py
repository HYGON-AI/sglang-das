import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_cpu, next_power_of_2

_is_cpu = is_cpu()

if _is_cpu:
    from sgl_kernel import fill_accept_out_cache_loc_cpu, fill_bonus_tokens_cpu


@triton.jit
def fill_bonus_tokens(
    accept_tokens,
    accept_lens,
    bonus_tokens_ptr,
    accept_stride: tl.constexpr,
):
    # NOTE: we cannot fuse any in-place operations of `accept_lens` inside this kernel
    # because this kernel reads accept_lens
    pid = tl.program_id(axis=0)
    # `accept_lens` includes the bonus token; the last accepted slot is at -1.
    accept_len = tl.load(accept_lens + pid)

    # accept_stride = per-req width of accept_tokens (= accept_index.shape[1]).
    bonus_token_idx = accept_stride * pid + accept_len - 1
    bonus_token = tl.load(accept_tokens + bonus_token_idx)
    tl.store(bonus_tokens_ptr + pid, bonus_token)


@triton.jit
def fill_bonus_tokens_from_predict(
    predict_ptr,
    accept_index_ptr,
    accept_lens_ptr,
    bonus_tokens_ptr,
    accept_index_stride: tl.constexpr,
):
    """Write each request's bonus directly from the flat prediction buffer."""
    pid = tl.program_id(axis=0).to(tl.int64)
    accept_len = tl.load(accept_lens_ptr + pid).to(tl.int64)
    index = tl.load(accept_index_ptr + pid * accept_index_stride + accept_len - 1).to(
        tl.int64
    )
    bonus_token = tl.load(predict_ptr + index)
    tl.store(bonus_tokens_ptr + pid, bonus_token)


def fill_bonus_tokens_func(
    accept_tokens: torch.Tensor,
    accept_lens: torch.Tensor,
    bonus_tokens: torch.Tensor,  # mutable
    accept_stride: int,
    batch_size: int,
):
    if _is_cpu:
        fill_bonus_tokens_cpu(
            accept_tokens,
            accept_lens,
            bonus_tokens,
            accept_stride,
        )
        return
    fill_bonus_tokens[(batch_size,)](
        accept_tokens,
        accept_lens,
        bonus_tokens,
        accept_stride,
    )


def fill_bonus_tokens_from_predict_func(
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    accept_lens: torch.Tensor,
    bonus_tokens: torch.Tensor,
    batch_size: int,
):
    """Fuse predict[accept_index] and bonus extraction for EAGLE verify."""
    if predict.ndim != 1:
        raise ValueError(f"predict must be 1D, got shape={tuple(predict.shape)}")
    if accept_index.ndim != 2:
        raise ValueError(
            "accept_index must be 2D, " f"got shape={tuple(accept_index.shape)}"
        )
    if accept_lens.ndim != 1 or accept_lens.shape[0] != batch_size:
        raise ValueError(
            "accept_lens must be 1D with batch_size entries, "
            f"got shape={tuple(accept_lens.shape)} batch_size={batch_size}"
        )
    if accept_index.shape[0] != batch_size:
        raise ValueError(
            "accept_index batch dimension must match batch_size, "
            f"got {accept_index.shape[0]} vs {batch_size}"
        )
    if bonus_tokens.ndim != 1 or bonus_tokens.shape[0] != batch_size:
        raise ValueError(
            "bonus_tokens must be 1D with batch_size entries, "
            f"got shape={tuple(bonus_tokens.shape)} batch_size={batch_size}"
        )
    if bonus_tokens.dtype != torch.int32:
        raise ValueError(f"bonus_tokens must be int32, got {bonus_tokens.dtype}")
    if bonus_tokens.device != predict.device:
        raise ValueError(
            f"bonus_tokens device {bonus_tokens.device} != predict device {predict.device}"
        )
    if not bonus_tokens.is_contiguous():
        raise ValueError("bonus_tokens must be contiguous")
    if batch_size == 0:
        return

    if _is_cpu:
        cols = (accept_lens.to(torch.long) - 1).clamp(min=0)
        indices = accept_index.gather(1, cols[:, None]).squeeze(1).to(torch.long)
        bonus_tokens.copy_(predict[indices])
        return

    # Triton is used only for CUDA/HIP tensors; preserve the existing path for
    # other accelerators whose backend may not support this kernel.
    if not predict.is_cuda:
        accept_tokens = predict[accept_index]
        fill_bonus_tokens_func(
            accept_tokens,
            accept_lens,
            bonus_tokens,
            accept_index.shape[1],
            batch_size,
        )
        return

    stream = torch.get_device_module(predict.device).current_stream()
    if not predict.is_contiguous():
        predict = predict.contiguous()
        predict.record_stream(stream)
    if not accept_index.is_contiguous():
        accept_index = accept_index.contiguous()
        accept_index.record_stream(stream)
    if not accept_lens.is_contiguous():
        accept_lens = accept_lens.contiguous()
        accept_lens.record_stream(stream)
    fill_bonus_tokens_from_predict[(batch_size,)](
        predict,
        accept_index,
        accept_lens,
        bonus_tokens,
        accept_index.shape[1],
    )


@triton.jit
def fill_accept_out_cache_loc(
    accept_index,
    out_cache_loc,
    accept_out_cache_loc,
    size_upper: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = tl.arange(0, size_upper)

    masks = (tl.load(accept_index + offset, offset < pid, other=-1) != -1).to(tl.int64)
    dst = tl.sum(masks)
    src = tl.load(accept_index + pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accept_out_cache_loc + dst, value)


def fill_accept_out_cache_loc_func(
    accept_index: torch.Tensor,
    out_cache_loc: torch.Tensor,
    accept_out_cache_loc: torch.Tensor,  # mutable
    size: int,
):
    if _is_cpu:
        fill_accept_out_cache_loc_cpu(
            accept_index,
            out_cache_loc,
            accept_out_cache_loc,
        )
        return
    fill_accept_out_cache_loc[(size,)](
        accept_index,
        out_cache_loc,
        accept_out_cache_loc,
        next_power_of_2(size),
    )
