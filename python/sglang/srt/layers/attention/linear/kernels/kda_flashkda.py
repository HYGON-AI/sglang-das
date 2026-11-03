from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)

# FlashKDA chunk size; the model passes raw token lengths and the kernel
# tiles them in 64-token chunks (also used to size intermediate states).
_FLASHKDA_CHUNK_SIZE = 64


@triton.jit
def _copy_indexed_kda_state(
    pool,
    packed,
    indices,
    STATE_SIZE: tl.constexpr,
    POOL_STRIDE: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    GATHER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(indices + row)
    valid = (slot >= 0) & (slot < NUM_SLOTS)
    if GATHER:
        value = tl.load(
            pool + slot * POOL_STRIDE + offsets,
            mask=valid & (offsets < STATE_SIZE),
            other=0,
        )
        tl.store(packed + row * STATE_SIZE + offsets, value, mask=offsets < STATE_SIZE)
    else:
        value = tl.load(
            packed + row * STATE_SIZE + offsets, mask=offsets < STATE_SIZE, other=0
        )
        tl.store(
            pool + slot * POOL_STRIDE + offsets,
            value,
            mask=valid & (offsets < STATE_SIZE),
        )


def _copy_kda_state(pool, packed, indices, *, gather):
    # DP padding uses -1: it has no committed state and must never alias
    # the final real slot through Python negative indexing.
    if (
        pool.ndim < 2
        or packed.shape != (indices.numel(), *pool.shape[1:])
        or indices.ndim != 1
        or indices.dtype not in (torch.int32, torch.int64)
        or not indices.is_contiguous()
        or not packed.is_contiguous()
        or pool.dtype != packed.dtype
        or pool.device != packed.device
        or pool.device != indices.device
    ):
        raise ValueError(
            "KDA state copy requires matching dense rows and contiguous indices"
        )
    if indices.numel() == 0 or packed.numel() == 0:
        return
    if not pool[0].is_contiguous():
        raise ValueError("KDA state pool rows must be contiguous")
    state_size = packed[0].numel()
    _copy_indexed_kda_state[(indices.numel(), triton.cdiv(state_size, 1024))](
        pool,
        packed,
        indices,
        state_size,
        pool.stride(0),
        pool.shape[0],
        gather,
        1024,
    )


def _load_flash_kda():
    """Import the optional ``flash_kda`` CUTLASS module."""
    try:
        import flash_kda
    except ImportError as e:
        raise ImportError(
            "The 'flashkda' KDA prefill backend requires the flash_kda module, "
            "which is not installed. Install it from source:\n"
            "    pip install git+https://github.com/MoonshotAI/FlashKDA.git"
        ) from e
    return flash_kda


def _triton_fallback(
    q,
    k,
    v,
    g,
    beta,
    ssm_states,
    cache_indices,
    query_start_loc,
    A_log=None,
    dt_bias=None,
    lower_bound=None,
    beta_is_raw=False,
    return_intermediate_states=False,
    track_state=None,
    track_chunk_idx=None,
):
    """Fall back to the Triton chunk_kda kernel (handles all preprocessing).

    `g` is the RAW gate; chunk_kda applies the gate activation internally when
    A_log is provided, so A_log/dt_bias/lower_bound must be threaded through too
    -- otherwise the fallback silently skips activation. chunk_kda updates the
    ssm state in-place via cache_indices and returns only the output tensor
    (or (output, h) when return_intermediate_states is set).
    """
    from sglang.kernels.ops.attention.fla.kda import chunk_kda

    return chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=ssm_states,
        initial_state_indices=cache_indices,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=query_start_loc,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        beta_is_raw=beta_is_raw,
        output_intermediate_states=return_intermediate_states,
        track_state=track_state,
        track_chunk_idx=track_chunk_idx,
    )


class FlashKDAKernel(LinearAttnKernelBase):
    """FlashKDA (MoonshotAI) fully-fused CUTLASS KDA prefill backend.

    Wraps the external ``flash_kda`` package (https://github.com/MoonshotAI/FlashKDA).

    FlashKDA fuses q/k L2 norm, beta sigmoid, and the KDA gate *inside* the
    kernel, so we pass RAW tensors plus ``A_log``/``dt_bias``/``lower_bound``.
    It is prefill-only, bf16, K == V == 128, HV == H (no GVA), and requires the
    safe (bounded) gate (``lower_bound`` set). Once this backend is selected
    via the prefill switch, every extend length runs on the fused kernel; only
    the correctness guards in ``_should_fall_back`` (unsafe gate, spec-decode
    draft-extend) route to the fallback.
    Requires an SM90+ GPU with the ``flash_kda`` package.
    """

    def __init__(self, fallback_kernel: Optional[LinearAttnKernelBase] = None):
        # Optional kernel serving the Triton fallback path. On HCU the generic
        # fla chunk_kda is unavailable; callers there wire the HcuKDAKernel in,
        # whose extend expects an ACTIVATED beta (beta_is_raw must stay False).
        self.fallback_kernel = fallback_kernel

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("FlashKDAKernel only supports prefill (extend)")

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        extend_seq_lens_cpu: Optional[list] = None,
        is_spec_decode: bool = False,
        beta_is_raw: bool = False,
        return_intermediate_states: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self._should_fall_back(
            lower_bound, is_spec_decode, query_start_loc, extend_seq_lens_cpu
        ):
            if self.fallback_kernel is not None:
                if beta_is_raw:
                    raise ValueError(
                        "flashkda fallback kernel expects an ACTIVATED beta; "
                        "raw-beta models are only supported with the generic "
                        "Triton fallback."
                    )
                # Returns plain o, or (o, h) when intermediate states were
                # requested -- the same contract as the fused path below.
                return self.fallback_kernel.extend(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    query_start_loc=query_start_loc,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    lower_bound=lower_bound,
                    return_intermediate_state=return_intermediate_states,
                )
            return _triton_fallback(
                q,
                k,
                v,
                g,
                beta,
                ssm_states,
                cache_indices,
                query_start_loc,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                beta_is_raw=beta_is_raw,
                return_intermediate_states=return_intermediate_states,
                track_state=kwargs.get("track_state"),
                track_chunk_idx=kwargs.get("track_chunk_idx"),
            )

        return self._flashkda_extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            beta_is_raw=beta_is_raw,
            return_intermediate_states=return_intermediate_states,
        )

    @staticmethod
    def _should_fall_back(
        lower_bound: Optional[float],
        is_spec_decode: bool,
        query_start_loc: torch.Tensor,
        extend_seq_lens_cpu: Optional[list],
    ) -> bool:
        """Whether to use the fallback kernel instead of the fused kernel.

        Only correctness guards remain -- the backend switch alone routes
        every prefill length through FlashKDA:
        - the fused kernel math requires the bounded (safe) gate; models with
          the unbounded gate (-exp(A_log)*softplus) leave lower_bound unset;
        - FlashKDA writes the committed recurrent state back in place, so it
          is unsafe for spec-decode draft-extend forwards (which must stay
          rollback-able). Those reach this backend through forward_extend, so
          gate them here rather than relying on the decode/target_verify stubs.
        Sequence lengths are no longer inspected (the parameters stay part of
        the signature because callers pass them).
        """
        return lower_bound is None or is_spec_decode

    def _flashkda_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        beta_is_raw: bool = False,
        return_intermediate_states: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        flash_kda = _load_flash_kda()

        # Input shapes (varlen, B == 1, matching chunk_kda's contract):
        #   q, k = [1, packed_seq, H, K]   v = [1, packed_seq, HV, V]
        #   g    = [1, packed_seq, HV, K]  beta = [1, packed_seq, H]
        # flash_kda wants these 4D tensors directly and RAW (it fuses l2norm /
        # beta sigmoid / gate activation in-kernel).
        num_heads = q.shape[2]
        head_dim = q.shape[3]
        scale = head_dim**-0.5

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()

        # FlashKDA applies sigmoid internally; invert only the already-activated
        # Kimi beta path.
        if not beta_is_raw:
            beta = torch.logit(beta.float().clamp_(1e-7, 1.0 - 1e-7))
        beta = beta.to(torch.bfloat16).contiguous()

        # flash_kda wants A_log [H] fp32 and dt_bias [H, K] fp32. The model
        # stores A_log as [1, 1, H, 1] and dt_bias as 1D [H*K], so reshape both.
        A_log = A_log.reshape(-1).float().contiguous()
        if dt_bias is not None:
            dt_bias = dt_bias.reshape(num_heads, -1).float().contiguous()

        # cu_seqlens must be int64 for flash_kda (FLA casts to long).
        cu_seqlens = query_start_loc.to(torch.int64)

        # flash_kda varlen state is [N, H, V, K] -- the SAME layout as sglang's
        # KDA pool, so no transpose is needed. Advanced indexing copies, so the
        # final state is written back in-place below (matching chunk_kda).
        initial_state = torch.empty(
            (cache_indices.numel(), *ssm_states.shape[1:]),
            device=ssm_states.device,
            dtype=ssm_states.dtype,
        )
        _copy_kda_state(ssm_states, initial_state, cache_indices, gather=True)

        out_buf = torch.empty_like(v)
        final_state = torch.empty_like(initial_state)
        intermediate_states = None
        if return_intermediate_states:
            sequence_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            num_intermediate_states = int(
                ((sequence_lengths + _FLASHKDA_CHUNK_SIZE - 1) // _FLASHKDA_CHUNK_SIZE)
                .sum()
                .item()
            )
            intermediate_states = initial_state.new_empty(
                (num_intermediate_states, num_heads, v.shape[-1], head_dim)
            )

        flash_kda.fwd(
            q,
            k,
            v,
            g,
            beta,
            scale,
            out_buf,
            A_log,
            dt_bias,
            lower_bound,
            initial_state=initial_state,
            final_state=final_state,
            cu_seqlens=cu_seqlens,
            intermediate_states=intermediate_states,
        )

        _copy_kda_state(ssm_states, final_state, cache_indices, gather=False)

        # FlashKDA returns per-64-token states as [chunks, H, V, K].
        # SGLang's tracking path consumes [1, chunks, H, V, K]. Match the
        # Triton kernel's contract: (o, h) only when intermediate states are
        # requested, plain o otherwise.
        h = (
            intermediate_states.unsqueeze(0)
            if intermediate_states is not None
            else None
        )
        if return_intermediate_states:
            return out_buf, h
        return out_buf
