from abc import ABC, abstractmethod
from typing import Tuple
from typing import Any, Optional
import torch

from sglang.srt.environ import envs

from sglang.kernels.npu_kernels.npu_grouped_matmul_triton import grouped_matmul_triton


def _unwrap(v: Any) -> Optional[torch.Tensor]:
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return v[0] if len(v) > 0 else None
    return v


def _to_cumsum(
    group_list: torch.Tensor, group_list_type: int
) -> torch.Tensor:
    """Triton grouped matmul expects cumsum (group_list_type=0)."""
    gl = group_list.to(torch.int64).reshape(-1)
    if group_list_type == 1:
        return torch.cumsum(gl, dim=0)
    return gl


def _scale_n_dim(scale: Optional[torch.Tensor]) -> Optional[int]:
    if scale is None:
        return None
    return int(scale.reshape(scale.shape[0], -1).shape[-1])


def _unpack_int4_along_last_dim(w_int8: torch.Tensor) -> torch.Tensor:
    """Inverse of ``NPUW4A4Int4MoEMethod._pack_int4`` along the last dim.

    Packing stores even-index int4 in the low nibble and odd-index int4 in the
    high nibble of each int8.
    """
    wu = w_int8.to(torch.int32) & 0xFF
    low = wu & 0x0F
    high = (wu >> 4) & 0x0F
    low = torch.where(low >= 8, low - 16, low).to(torch.int8)
    high = torch.where(high >= 8, high - 16, high).to(torch.int8)
    out = torch.empty(
        *w_int8.shape[:-1],
        w_int8.shape[-1] * 2,
        dtype=torch.int8,
        device=w_int8.device,
    )
    out[..., 0::2] = low
    out[..., 1::2] = high
    return out


def _prepare_triton_weight(
    weight: torch.Tensor, scale: Optional[torch.Tensor]
) -> torch.Tensor:
    """Unpack NPU W4A8 packed weights so Triton sees logical ``[E, K, N]``.

    ``NPUW4A8Int8MoEMethod.process_weights_after_loading`` does:
      [E, N_packed, K] int8 → transpose → [E, K, N_packed]
      → view int32 → [E, K, N_packed/4]

    where ``N_packed = logical_N / 2`` (two int4 values per int8). Triton has no
    int4-pack semantics, so restore int8 with logical N before the GEMM.
    """
    w = weight
    if w.dtype == torch.int32:
        # 4 int8 (8 int4) per int32 on the last dim.
        w = w.contiguous().view(torch.int8)

    scale_n = _scale_n_dim(scale)
    if (
        w.dtype == torch.int8
        and scale_n is not None
        and scale_n == w.shape[-1] * 2
    ):
        w = _unpack_int4_along_last_dim(w)
    return w


def _scale_to_float32(scale: torch.Tensor) -> torch.Tensor:
    """Restore float32 scales from NPU int64 bit-containers if needed."""
    if scale.dtype == torch.float32:
        return scale
    if scale.dtype in (torch.float16, torch.bfloat16):
        return scale.to(torch.float32)
    if scale.dtype == torch.int64:
        # Per-channel W4A8 path stores fp32 bits zero-extended into int64.
        return (scale & 0xFFFFFFFF).to(torch.int32).view(torch.float32)
    return scale.to(torch.float32)


class BaseMatmul(ABC):
    @abstractmethod
    def forward(
        self,
        layer: torch.nn.Module,
        weight_prefix: str,
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        output_dtype: torch.dtype,
        group_list_type: int,
        transposed: bool,
        **scale_args,
    ) -> torch.Tensor:
        pass


class GroupedMatmul(BaseMatmul):
    def forward(
        self,
        layer: torch.nn.Module,
        weight_prefix: str,
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        output_dtype: torch.dtype,
        group_list_type: int,
        transposed: bool,
        **scale_args,
    ) -> torch.Tensor:
        # Access the weight attribute directly from the layer
        weight = getattr(layer, f"{weight_prefix}_weight", None)
        if weight is None:
            raise AttributeError(
                f"Weight attribute '{weight_prefix}_weight' not found in layer"
            )

        device = hidden_states.device
        scale = _unwrap(scale_args.get("scale"))
        per_token_scale = _unwrap(scale_args.get("per_token_scale"))
        bias = _unwrap(scale_args.get("bias"))
        antiquant_scale = _unwrap(scale_args.get("antiquant_scale"))
        antiquant_offset = _unwrap(scale_args.get("antiquant_offset"))

        # Defense-in-depth for SGLANG_W4A8_MOE_SKIP_SCALE_BIAS: drop float
        # scale_bias even if a caller still passes it (int32 bias kept).
        if (
            envs.SGLANG_W4A8_MOE_SKIP_SCALE_BIAS.get()
            and bias is not None
            and bias.dtype != torch.int32
        ):
            global _SKIP_SCALE_BIAS_LOGGED
            if not _SKIP_SCALE_BIAS_LOGGED:
                logger.warning(
                    "SGLANG_W4A8_MOE_SKIP_SCALE_BIAS=1: skipping float "
                    "scale_bias in GroupedMatmul dequant (ablation)."
                )
                _SKIP_SCALE_BIAS_LOGGED = True
            bias = None

        w = weight
        if antiquant_scale is not None:
            # Formula 4: y = x @ ((w + offset) * antiquant_scale) + bias
            w_f = w.to(torch.float32)
            if antiquant_offset is not None:
                w_f = w_f + antiquant_offset.to(torch.float32)
            w = w_f * antiquant_scale.to(torch.float32)
        else:
            # W4A8 NPU packing is opaque to the Triton float/int GEMM.
            w = _prepare_triton_weight(w, scale)

        group_list = _to_cumsum(expert_tokens, group_list_type)
        has_quant = scale is not None or per_token_scale is not None
        # Float bias after scale (formula 3-2); int32 bias before scale (3-1).
        kernel_bias = None if has_quant else bias
        y = grouped_matmul_triton(
            hidden_states,
            w,
            bias=kernel_bias,
            group_list=group_list,
            split_item=2,
            # NPU ``transposed=True`` means weight is already in op layout
            # (no further transpose). Triton ``transpose_weight=True`` means
            # weight is [E, N, K] and needs an internal transpose.
            transpose_weight=not transposed,
        )[0]
        if has_quant:
            # Triton runs on CUDA/HCU; keep scale/bias on the same device as y.
            compute_device = y.device
            y = y.to(torch.float32)
            ends = group_list.detach().cpu().tolist()
            starts = [0] + ends[:-1]

            if bias is not None and bias.dtype == torch.int32:
                b = bias.to(device=compute_device, dtype=torch.float32)
                for e, (s, t) in enumerate(zip(starts, ends)):
                    if s < t:
                        y[s:t] = y[s:t] + b[e].reshape(1, -1)

            if scale is not None:
                sc = _scale_to_float32(scale).to(device=compute_device)
                for e, (s, t) in enumerate(zip(starts, ends)):
                    if s < t:
                        y[s:t] = y[s:t] * sc[e].reshape(1, -1)

            if per_token_scale is not None:
                y = y * per_token_scale.to(
                    device=compute_device, dtype=torch.float32
                ).reshape(-1, 1)

            if bias is not None and bias.dtype != torch.int32:
                b = bias.to(device=compute_device, dtype=torch.float32)
                for e, (s, t) in enumerate(zip(starts, ends)):
                    if s < t:
                        y[s:t] = y[s:t] + b[e].reshape(1, -1)

            y = y.to(output_dtype)
        elif output_dtype is not None and y.dtype != output_dtype:
            y = y.to(output_dtype)

        return y.to(device=device)


class GroupedMatmulSwigluQuant(BaseMatmul):
    """Grouped matmul with swiglu and requantisation fused into one kernel.

    Used for the gate/up projection (gmm1) of block-scaled MoE: the kernel emits
    activations already quantised for the following down projection, so the
    caller has no separate activation step. Unlike ``GroupedMatmul`` it returns
    ``(quantized_activations, block_scale)`` instead of a single tensor, and it
    takes no ``output_dtype`` — the output dtype is set through ``quant_dtype``
    in ``scale_args``.
    """

    def forward(
        self,
        layer: torch.nn.Module,
        weight_prefix: str,
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        output_dtype: torch.dtype = None,
        group_list_type: int = 1,
        transposed: bool = True,
        **scale_args,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = getattr(layer, f"{weight_prefix}_weight", None)
        if weight is None:
            raise AttributeError(
                f"Weight attribute '{weight_prefix}_weight' not found in layer"
            )
        # This op wants a cumulative group_list while the plain grouped matmul
        # keeps the COUNT form the dispatcher produces (group_list_type=1). The
        # asymmetry is intentional.
        group_list = expert_tokens.cumsum(0) if group_list_type == 1 else expert_tokens
        return torch.ops.npu.npu_grouped_matmul_swiglu_quant_v2(
            x=hidden_states,
            weight=[weight] if transposed else [weight.transpose(1, 2)],
            group_list=group_list,
            **scale_args,
        )
