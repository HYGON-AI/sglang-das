# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Inference-only WeLMV4 VLM model compatible with HF WeLMV4 VLM weights."""

from __future__ import annotations

import dataclasses
import os
from types import SimpleNamespace
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.welmv4_op import (
    welmv4_vision_apply_rope,
    welmv4_vision_quick_gelu,
)
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    get_embedding_and_mask,
)
from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.welmv4 import WeLMV4MoeForCausalLM
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, print_warning_once

# Maximum number of vision-encoder patch tokens processed in a single
# forward pass through the 27-layer vision transformer. **Disabled by
# default** to preserve the original "concat all images and forward once"
# behaviour byte-for-byte; set this env to a positive integer (e.g.
# ``262144`` = one 4096x4096 image) to opt in to per-image splitting,
# which is a numerically-equivalent transformation that bounds peak HBM
# for high-resolution / many-image requests. The same value is reused as
# the chunk size for the patch-embed unfold/GEMM step.
_VISION_MAX_PATCHES_ENV_VAR = "SGLANG_VLM_VISION_MAX_PATCHES"
_VISION_MAX_PATCHES_DISABLED = 0


def _read_positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        v = int(raw)
    except ValueError:
        return default
    return v if v > 0 else default


def _vision_max_patches() -> int:
    """Soft cap on the number of patch tokens processed per encoder pass.

    Returns ``0`` when the env is unset, signalling "no chunking; preserve
    the original concat-then-forward behaviour exactly". A positive value
    enables per-image splitting in the encoder loop and the patch-embed
    unfold/GEMM step.
    """
    return _read_positive_int_env(
        _VISION_MAX_PATCHES_ENV_VAR, _VISION_MAX_PATCHES_DISABLED
    )


def _vision_dp_tp_kwargs(use_data_parallel: bool) -> dict:
    """Linear kwargs that force ``tp_size=1`` when vision DP is enabled.

    Empty dict (i.e. inherit the global TP) when DP is off.
    """
    if use_data_parallel:
        return {"tp_size": 1, "tp_rank": 0}
    return {}


def _as_config(config):
    if isinstance(config, dict):
        return SimpleNamespace(**config)
    return config


def _patch_embed_forward_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    k_t: int,
    k_h: int,
    k_w: int,
    in_channels: int,
    hidden_size: int,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    x = x.view(-1, in_channels, k_t, k_h, k_w).to(dtype=weight.dtype)
    n = x.shape[0]
    weight2d = weight.reshape(hidden_size, -1)

    # The bias is added AFTER the GEMM, NOT through cuBLAS's fused fp32
    # epilogue. The WeLMV4 training stack computes the GEMM and the bias
    # add as two separate bf16 ops (``out = gemm(x, w); out = out + b``),
    # so each operation rounds to bf16 once. Using ``F.linear(x, w, b)``
    # here would fuse the bias add into the GEMM epilogue and round only
    # once in fp32, producing slightly different outputs. We mirror the
    # training-side two-round semantics so inference outputs stay aligned
    # with training.

    if chunk_size is None:
        chunk_size = _vision_max_patches()
    # ``chunk_size <= 0`` or ``chunk_size >= n`` => single-shot path
    # (preserves the original behaviour byte-for-byte).
    if chunk_size <= 0 or chunk_size >= n:
        unfolded = x.unfold(2, k_t, k_t).unfold(3, k_h, k_h).unfold(4, k_w, k_w)
        unfolded = unfolded.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(
            n, -1, in_channels * k_t * k_h * k_w
        )
        out = F.linear(unfolded, weight2d, None)
        if bias is not None:
            out = out + bias
        return out

    # Match the (n, 1, hidden_size) shape produced by the single-shot
    # path's reshape so callers see the same layout regardless of
    # ``chunk_size``.
    out = x.new_empty((n, 1, hidden_size))
    for s in range(0, n, chunk_size):
        e = min(n, s + chunk_size)
        seg = x[s:e]
        unfolded = seg.unfold(2, k_t, k_t).unfold(3, k_h, k_h).unfold(4, k_w, k_w)
        unfolded = unfolded.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(
            e - s, -1, in_channels * k_t * k_h * k_w
        )
        out_seg = F.linear(unfolded, weight2d, None)
        if bias is not None:
            out_seg = out_seg + bias
        out[s:e] = out_seg
    return out


class WeLMV4VisionPatchEmbed(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.in_channels = config.in_channels
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.hidden_size = config.hidden_size
        kernel_size = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.proj = nn.Conv3d(
            config.in_channels,
            config.hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        k_t, k_h, k_w = self.temporal_patch_size, self.patch_size, self.patch_size
        out = _patch_embed_forward_matmul(
            hidden_states,
            self.proj.weight,
            self.proj.bias,
            k_t,
            k_h,
            k_w,
            self.in_channels,
            self.hidden_size,
        )
        return out.view(-1, self.hidden_size)


class WeLMV4VisionPositionEmbedding(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.num_position_embeddings = config.num_position_embeddings
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        if self.num_grid_per_side**2 != config.num_position_embeddings:
            raise ValueError(
                "num_position_embeddings must be a perfect square, got "
                f"{config.num_position_embeddings}."
            )
        self.weight = nn.Parameter(
            torch.zeros(config.num_position_embeddings, config.hidden_size)
        )

    def _scale(self, x: torch.Tensor, size: int) -> torch.Tensor:
        if size == 1:
            return torch.zeros_like(x, dtype=torch.float32)
        # Use a 0-d fp32 tensor (rather than a Python int) as the divisor:
        # Torch dispatches scalar / tensor divisors to slightly different
        # CUDA kernels with different rounding behaviour, and the training
        # forward uses the tensor variant. Keeping the same dispatch here
        # preserves numerical alignment with training.
        denom = torch.tensor(
            float(size - 1),
            device=x.device,
            dtype=torch.float32,
        )
        return x.to(torch.float32) * (self.num_grid_per_side - 1) / denom

    def _build_for_image(
        self, num_frames: int, height: int, width: int
    ) -> torch.Tensor:
        device = self.weight.device
        dtype = self.weight.dtype
        h_axis = self._scale(torch.arange(height, device=device), height)
        w_axis = self._scale(torch.arange(width, device=device), width)

        h_floor = h_axis.floor().to(torch.int64)
        w_floor = w_axis.floor().to(torch.int64)
        h_ceil = (h_floor + 1).clamp(max=self.num_grid_per_side - 1)
        w_ceil = (w_floor + 1).clamp(max=self.num_grid_per_side - 1)
        # Keep the fractional part in fp32 through the bilinear outer
        # product; the cast to the embedding dtype happens once on the
        # stacked ``weights`` tensor below.
        dh = h_axis - h_floor.to(torch.float32)
        dw = w_axis - w_floor.to(torch.float32)

        base = h_floor * self.num_grid_per_side
        base_c = h_ceil * self.num_grid_per_side
        idx = torch.stack(
            [
                (base[:, None] + w_floor[None, :]).reshape(-1),
                (base[:, None] + w_ceil[None, :]).reshape(-1),
                (base_c[:, None] + w_floor[None, :]).reshape(-1),
                (base_c[:, None] + w_ceil[None, :]).reshape(-1),
            ],
            dim=0,
        )
        weights = torch.stack(
            [
                ((1 - dh)[:, None] * (1 - dw)[None, :]).reshape(-1),
                ((1 - dh)[:, None] * dw[None, :]).reshape(-1),
                (dh[:, None] * (1 - dw)[None, :]).reshape(-1),
                (dh[:, None] * dw[None, :]).reshape(-1),
            ],
            dim=0,
        ).to(dtype)
        pos = (F.embedding(idx, self.weight) * weights[:, :, None]).sum(dim=0)
        pos = pos.repeat(num_frames, 1)
        merge = self.spatial_merge_size
        return (
            pos.view(num_frames, height // merge, merge, width // merge, merge, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )

    def forward(self, grid_thw: torch.Tensor) -> torch.Tensor:
        out: List[torch.Tensor] = []
        for t, h, w in grid_thw.tolist():
            out.append(self._build_for_image(int(t), int(h), int(w)))
        return torch.cat(out, dim=0) if len(out) > 1 else out[0]


def _apply_welmv4_vision_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    x_shape: torch.Size,
) -> Tuple[torch.Tensor, torch.Tensor]:
    del x_shape
    cos, sin = position_embeddings
    return welmv4_vision_apply_rope(q, k, cos, sin)


class WeLMV4VisionAttention(VisionAttention):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            projection_size=config.hidden_size,
            use_qkv_parallel=True,
            qkv_bias=True,
            proj_bias=True,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=prefix,
            customized_position_embedding_applier=_apply_welmv4_vision_rope,
            use_data_parallel=use_data_parallel,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states.unsqueeze(0)
        hidden_states = super().forward(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        return hidden_states.squeeze(0)


class WeLMV4VisionMLP(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        tp_kwargs = _vision_dp_tp_kwargs(use_data_parallel)
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
            **tp_kwargs,
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
            **tp_kwargs,
        )
        self._quick_gelu = config.hidden_act == "quick_gelu"
        if config.hidden_act == "gelu_pytorch_tanh":
            self.act = nn.GELU(approximate="tanh")
        elif self._quick_gelu:
            self.act = nn.Sigmoid()
        else:
            self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(x)
        if self._quick_gelu:
            # Use the triton ``quick_gelu`` kernel: fp32 compute with a
            # cast back to the input dtype on store. Matches the
            # training-side activation byte-for-byte.
            hidden_states = welmv4_vision_quick_gelu(hidden_states)
        else:
            hidden_states = self.act(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class WeLMV4VisionBlock(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = WeLMV4VisionAttention(
            config,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            use_data_parallel=use_data_parallel,
        )
        self.mlp = WeLMV4VisionMLP(
            config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
            use_data_parallel=use_data_parallel,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class WeLMV4VisionEncoder(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        self.config = config
        self.patch_embed = WeLMV4VisionPatchEmbed(config)
        self.pos_embed = WeLMV4VisionPositionEmbedding(config)
        self.blocks = nn.ModuleList(
            [
                WeLMV4VisionBlock(
                    config,
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{i}", prefix),
                    use_data_parallel=use_data_parallel,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self):
        return self.patch_embed.proj.weight.device

    def _rope_pos_ids(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        merge = self.config.spatial_merge_size
        for t, h, w in grid_thw.tolist():
            hpos_ids = torch.arange(h, device=grid_thw.device).unsqueeze(1).expand(h, w)
            hpos_ids = (
                hpos_ids.reshape(h // merge, merge, w // merge, merge)
                .permute(0, 2, 1, 3)
                .flatten()
            )
            wpos_ids = torch.arange(w, device=grid_thw.device).unsqueeze(0).expand(h, w)
            wpos_ids = (
                wpos_ids.reshape(h // merge, merge, w // merge, merge)
                .permute(0, 2, 1, 3)
                .flatten()
            )
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        return torch.cat(pos_ids, dim=0)

    def _rope_cos_sin(
        self, grid_thw: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        rope_dim = head_dim // 2
        inv_freq = 1.0 / (
            10000.0
            ** (
                torch.arange(
                    0, rope_dim, 2, device=grid_thw.device, dtype=torch.float32
                )
                / rope_dim
            )
        )
        pos_ids = self._rope_pos_ids(grid_thw)
        max_grid = int(grid_thw[:, 1:].max().item())
        seq = torch.arange(max_grid, device=grid_thw.device, dtype=torch.float32)
        freqs = torch.outer(seq, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cos_hw = torch.cat([cos[pos_ids[:, 0]], cos[pos_ids[:, 1]]], dim=-1)
        sin_hw = torch.cat([sin[pos_ids[:, 0]], sin[pos_ids[:, 1]]], dim=-1)
        return cos_hw, sin_hw

    def _cu_seqlens(self, grid_thw: torch.Tensor) -> torch.Tensor:
        seq_lens: List[int] = []
        for t, h, w in grid_thw.tolist():
            seq_lens.extend([int(h) * int(w)] * int(t))
        seq_lens_t = torch.tensor(seq_lens, device=self.device, dtype=torch.int32)
        return F.pad(seq_lens_t.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0)

    @staticmethod
    def _split_groups_by_image(
        boundaries: List[int], max_patches: int
    ) -> List[Tuple[int, int]]:
        """Split ``cu_seqlens`` boundaries into encoder micro-batches.

        Returned ranges are ``(boundary_start, boundary_end)`` indices into
        ``boundaries`` such that each group covers a contiguous set of
        per-image segments and has ``boundaries[end] - boundaries[start]``
        patches; we never split inside a single image (which would break
        per-image self-attention).
        """
        groups: List[Tuple[int, int]] = []
        if len(boundaries) <= 1:
            return groups
        start = 0
        for i in range(1, len(boundaries)):
            # Try to keep this segment in the current group.
            if boundaries[i] - boundaries[start] <= max_patches or i - 1 == start:
                continue
            # The segment ending at ``i`` would push over the cap; close the
            # group at ``i - 1`` and start a new group from there.
            groups.append((start, i - 1))
            start = i - 1
        groups.append((start, len(boundaries) - 1))
        return groups

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        grid_thw = grid_thw.to(device=self.device)
        hidden_states = self.patch_embed(pixel_values)
        hidden_states = hidden_states + self.pos_embed(grid_thw).to(hidden_states.dtype)

        cu_seqlens = self._cu_seqlens(grid_thw)
        cos, sin = self._rope_cos_sin(grid_thw)
        # Keep cos/sin in fp32 here. The triton RoPE kernel
        # (``welmv4_vision_apply_rope``) reads them in fp32 directly;
        # casting to bf16 would lose precision on low-magnitude sin
        # values and add a measurable error to every block.
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)

        max_patches = _vision_max_patches()
        total = hidden_states.shape[0]
        # Cheap fast-path: small enough to fit in one shot, or the cap is
        # effectively unlimited. Preserves byte-for-byte the prior behaviour.
        if max_patches <= 0 or total <= max_patches:
            for block in self.blocks:
                hidden_states = block(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=(cos, sin),
                )
            return hidden_states

        # ``.tolist()`` triggers a one-shot GPU->CPU sync so we can drive
        # the per-image grouping in pure Python. This is a host-bound code
        # path (already does Python-level iteration over ``self.blocks``)
        # so the sync is effectively free; just be aware that this rules
        # the chunked branch out of any cuda-graph capture region.
        boundaries: List[int] = cu_seqlens.tolist()
        groups = self._split_groups_by_image(boundaries, max_patches)
        # A single image whose patch count itself exceeds the cap cannot
        # be split safely (self-attention is per-image). It will end up
        # alone in its own group via ``_split_groups_by_image``; warn so
        # the user can either lower input resolution or raise the cap.
        # Note: only triggers when the batch's total patch count > cap
        # (otherwise we take the fast-path above and never look at single
        # image sizes).
        max_single_image = max(
            boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)
        )
        if max_single_image > max_patches:
            print_warning_once(
                "WeLMV4 vision encoder: a single image has "
                f"{max_single_image} patches, exceeding "
                f"SGLANG_VLM_VISION_MAX_PATCHES={max_patches}. Self-attention "
                "is per-image, so this image is processed in one shot. "
                "Lower the input resolution (e.g. via processor max_pixels) "
                "or raise SGLANG_VLM_VISION_MAX_PATCHES if memory allows."
            )

        for g_start, g_end in groups:
            tok_start = boundaries[g_start]
            tok_end = boundaries[g_end]
            # ``cu_seqlens`` is already int32 on ``hidden_states.device``;
            # subtraction by a Python int preserves dtype/device, so no
            # extra ``.to()``/``.contiguous()`` is needed here.
            sub_cu = cu_seqlens[g_start : g_end + 1] - boundaries[g_start]
            sub_cos = cos[tok_start:tok_end]
            sub_sin = sin[tok_start:tok_end]
            h = hidden_states[tok_start:tok_end]
            for block in self.blocks:
                h = block(h, cu_seqlens=sub_cu, position_embeddings=(sub_cos, sub_sin))
            # In-place copy so we reuse the pre-allocated buffer rather
            # than holding both old and new activations in memory.
            hidden_states[tok_start:tok_end] = h
        return hidden_states


class WeLMV4VisionProjector(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        merge_hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.merge_hidden_size = merge_hidden_size
        self.ln_q = nn.LayerNorm(config.hidden_size, eps=1e-6)
        tp_kwargs = _vision_dp_tp_kwargs(use_data_parallel)
        self.mlp = nn.ModuleList(
            [
                ColumnParallelLinear(
                    merge_hidden_size,
                    merge_hidden_size,
                    bias=True,
                    quant_config=quant_config,
                    prefix=add_prefix("mlp.0", prefix),
                    **tp_kwargs,
                ),
                nn.GELU(),
                RowParallelLinear(
                    merge_hidden_size,
                    config.out_hidden_size,
                    bias=True,
                    quant_config=quant_config,
                    prefix=add_prefix("mlp.2", prefix),
                    **tp_kwargs,
                ),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.ln_q(hidden_states).view(-1, self.merge_hidden_size)
        hidden_states, _ = self.mlp[0](hidden_states)
        hidden_states = self.mlp[1](hidden_states)
        hidden_states, _ = self.mlp[2](hidden_states)
        return hidden_states


class WeLMV4VLMForConditionalGeneration(WeLMV4MoeForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config=None,
        prefix: str = "",
        draft_model_idx: Optional[int] = None,
    ):
        if not hasattr(config, "text_config"):
            raise ValueError("WeLMV4 VLM config must contain text_config.")
        text_config = _as_config(config.text_config)
        vision_config = _as_config(config.vision_config)

        encoder_only = getattr(config, "encoder_only", False)
        language_only = getattr(config, "language_only", False)
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder
        self.vlm_config = config
        self._is_nextn = (
            getattr(text_config, "architectures", [None])[0]
            == "WeLMV4MoeForCausalLMNextN"
        )

        if self._is_nextn:
            # Draft workers for the nested VLM use the language-only NextN
            # stack.  The target supplies multimodal embeddings during draft
            # extend, so constructing a second vision tower would waste HBM
            # and would incorrectly rerun vision on decode.
            from sglang.srt.distributed import get_pp_group
            from sglang.srt.layers.logits_processor import LogitsProcessor
            from sglang.srt.models.welmv4_nextn import WeLMV4ModelNextN

            nn.Module.__init__(self)
            self.pp_group = get_pp_group()
            self.config = text_config
            self.draft_model_idx = draft_model_idx
            self.model = WeLMV4ModelNextN(
                text_config,
                quant_config,
                prefix=add_prefix("model", prefix),
                draft_model_idx=draft_model_idx,
            )
            self.lm_head = None
            self.logits_processor = LogitsProcessor(text_config)
            self.capture_aux_hidden_states = False
            self.vision_encoder = None
            self.vision_projector = None
            self.post_init_after_load_weights(is_nextn=True)
            return

        if encoder_only:
            # In encoder_only mode, skip LLM instantiation entirely to save
            # GPU memory. Only the vision encoder + projector are needed.
            nn.Module.__init__(self)
            self.config = config
            self.vision_encoder = WeLMV4VisionEncoder(
                vision_config,
                quant_config=quant_config,
                prefix=add_prefix("vision_encoder", prefix),
                use_data_parallel=self.use_data_parallel,
            )
            self.vision_projector = WeLMV4VisionProjector(
                vision_config,
                quant_config=quant_config,
                prefix=add_prefix("vision_projector", prefix),
                use_data_parallel=self.use_data_parallel,
            )
            return

        super().__init__(text_config, quant_config=quant_config, prefix=prefix)
        self.config = config

        # In language_only mode the vision encoder runs on a separate server;
        # skip instantiating vision modules to save GPU memory.
        if not language_only:
            self.vision_encoder = WeLMV4VisionEncoder(
                vision_config,
                quant_config=quant_config,
                prefix=add_prefix("vision_encoder", prefix),
                use_data_parallel=self.use_data_parallel,
            )
            self.vision_projector = WeLMV4VisionProjector(
                vision_config,
                quant_config=quant_config,
                prefix=add_prefix("vision_projector", prefix),
                use_data_parallel=self.use_data_parallel,
            )
        else:
            self.vision_encoder = None
            self.vision_projector = None

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def should_apply_lora(self, module_name: str) -> bool:
        return not (
            module_name.startswith("vision_encoder")
            or module_name.startswith("vision_projector")
        )

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # Note: peak-memory control for high-resolution / many-image
        # batches is delegated to ``WeLMV4VisionEncoder.forward`` which
        # splits the concatenated input along image boundaries when the
        # total patch count exceeds ``SGLANG_VLM_VISION_MAX_PATCHES``.
        # That means we can keep the simple concat-then-forward path here
        # without giving up the OOM mitigation - and we keep the original
        # byte-for-byte numerics for callers that don't trigger chunking.
        pixel_values = torch.cat([item.feature for item in items], dim=0)
        image_grid_thw = torch.cat([item.image_grid_thw for item in items], dim=0)
        image_embeds = self.vision_encoder(pixel_values, image_grid_thw)
        return self.vision_projector(image_embeds)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        is_nextn = is_nextn or self._is_nextn
        if self._is_nextn and getattr(self, "draft_model_idx", None) is not None:
            from sglang.srt.models.welmv4_nextn import select_mtp_layer_weights

            weights = select_mtp_layer_weights(
                self.config, weights, self.draft_model_idx
            )
        params_dict = dict(self.named_parameters())
        language_only = getattr(self.vlm_config, "language_only", False)
        encoder_only = getattr(self.vlm_config, "encoder_only", False)
        # The base loader silently skips names it cannot place.  Vision
        # parameters left unloaded keep uninitialised memory and produce
        # garbage image embeddings, so track them and fail loudly instead.
        vision_params = {
            n for n in params_dict
            if n.startswith(("vision_encoder.", "vision_projector."))
        }
        loaded_vision: set = set()
        unmatched_vision: list = []

        def load_qkv_weights_and_forward_others():
            for name, loaded_weight in weights:
                # Skip vision weights entirely in language_only mode
                if language_only and (
                    "vision_encoder" in name or "vision_projector" in name
                ):
                    continue
                # Skip LLM weights entirely in encoder_only mode
                if encoder_only and not (
                    "vision_encoder" in name or "vision_projector" in name
                ):
                    continue
                if "vision_encoder" in name and "attn.qkv." in name:
                    qkv_name = name.replace("attn.qkv.", "attn.qkv_proj.")
                    param = params_dict.get(qkv_name)
                    if param is not None:
                        param.weight_loader(param, loaded_weight)
                        loaded_vision.add(qkv_name)
                    else:
                        unmatched_vision.append(name)
                    continue
                if name.startswith(("vision_encoder.", "vision_projector.")):
                    if name in params_dict:
                        loaded_vision.add(name)
                    else:
                        unmatched_vision.append(name)
                yield name, loaded_weight

        super().load_weights(load_qkv_weights_and_forward_others(), is_nextn=is_nextn)

        if vision_params and not language_only and not is_nextn:
            missing = sorted(vision_params - loaded_vision)
            if missing or unmatched_vision:
                raise RuntimeError(
                    "WeLMV4 VLM vision weights not loaded: "
                    f"{len(missing)} parameters never loaded (e.g. {missing[:8]}), "
                    f"{len(unmatched_vision)} checkpoint tensors unmatched "
                    f"(e.g. {unmatched_vision[:8]})."
                )

    def _image_token_id(self) -> int:
        image_token_id = getattr(self.config, "image_token_id", None)
        if image_token_id is None:
            raise ValueError(
                "config.image_token_id must be set for WeLMV4 VLM multimodal forward."
            )
        return int(image_token_id)

    def _collect_image_items(
        self, mm_inputs_list: List[MultimodalInputs]
    ) -> List[MultimodalDataItem]:
        image_items: List[MultimodalDataItem] = []
        for mm_inputs in mm_inputs_list:
            image_items.extend(item for item in mm_inputs.mm_items if item.is_image())
        return image_items

    def _logical_input_ids(
        self, input_ids: torch.Tensor, image_items: List[MultimodalDataItem]
    ) -> torch.Tensor:
        image_token_id = self._image_token_id()
        logical_input_ids = input_ids.clone()
        for item in image_items:
            if item.pad_value is not None:
                logical_input_ids[logical_input_ids == item.pad_value] = image_token_id
        vocab_size = self.model.embed_tokens.num_embeddings
        return logical_input_ids.clamp(min=0, max=vocab_size - 1)

    def _logical_forward_batch(
        self, forward_batch: ForwardBatch, image_items: List[MultimodalDataItem]
    ) -> ForwardBatch:
        # Image pad values are request hashes and therefore must not enter the
        # OE n-gram lookup.  Rewrite the batch-owned n-gram state in lock-step
        # with the logical input ids before running the language model.
        state_attr = "oe_context" if hasattr(forward_batch, "oe_context") else "n_gram_input_ids"
        oe_state = getattr(forward_batch, state_attr, None)
        if oe_state is None:
            return forward_batch

        pad_values = [
            item.pad_value for item in image_items if item.pad_value is not None
        ]
        if not pad_values:
            return forward_batch
        image_token_id = self._image_token_id()

        logical_grams: List[Optional[torch.Tensor]] = []
        for gram in oe_state.input_ids_grams:
            if gram is None:
                logical_grams.append(None)
                continue
            logical_gram = gram.clone()
            for pad_value in pad_values:
                logical_gram[logical_gram == pad_value] = image_token_id
            logical_grams.append(logical_gram)

        logical_buffer = oe_state.input_ids_buffer
        if logical_buffer is not None:
            logical_buffer = logical_buffer.clone()
            for pad_value in pad_values:
                logical_buffer[logical_buffer == pad_value] = image_token_id

        logical_state = dataclasses.replace(
            oe_state,
            input_ids_grams=logical_grams,
            input_ids_buffer=logical_buffer,
        )
        # Swap the state on the original batch instead of copying it: the
        # attention backend reads the batch from the forward context, so
        # per-forward fields the body sets (e.g. KV-mirror custom_last_index)
        # must land on that same object.  forward() restores the original.
        forward_batch._welm_vl_saved_oe_state = (state_attr, oe_state)
        setattr(forward_batch, state_attr, logical_state)
        return forward_batch

    def _get_image_embedding_and_mask(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        mm_inputs_list: List[MultimodalInputs],
        mm_input_indices: List[int],
        image_items: List[MultimodalDataItem],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        items_size = torch.zeros(len(mm_inputs_list) + 1, dtype=int)
        items_offsets = []
        for i, mm_inputs in enumerate(mm_inputs_list):
            mm_items = [item for item in mm_inputs.mm_items if item.is_image()]
            items_size[i + 1] = len(mm_items)
            offsets = []
            for item in mm_items:
                offsets.extend(item.offsets)
            items_offsets.append(offsets)

        if forward_batch.extend_prefix_lens_cpu is None:
            prefix_lens = [0] * len(mm_inputs_list)
        else:
            prefix_lens = [
                forward_batch.extend_prefix_lens_cpu[i] for i in mm_input_indices
            ]
        if forward_batch.extend_seq_lens_cpu is None:
            extend_lens = [input_ids.numel()] * len(mm_inputs_list)
        else:
            extend_lens = [
                forward_batch.extend_seq_lens_cpu[i] for i in mm_input_indices
            ]
        placeholder_tensor = torch.as_tensor(
            [item.pad_value for item in image_items], device=input_ids.device
        )
        # Third return value is the (possibly pruned) input_ids for EVS.
        # WeLMV4 VLM does not use EVS, so it is safe to discard.
        embedding, mask, _pruned_ids = get_embedding_and_mask(
            data_embedding_func=self.get_image_feature,
            embedding_items=image_items,
            placeholder_tensor=placeholder_tensor,
            input_ids=input_ids,
            items_size=torch.cumsum(items_size, dim=0).tolist(),
            prefix_length=prefix_lens,
            extend_length=extend_lens,
            items_offset_list=items_offsets,
        )
        return embedding, mask

    def _build_multimodal_input_embeds(
        self, input_ids: torch.Tensor, forward_batch: ForwardBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, ForwardBatch]:
        mm_inputs_with_indices = [
            (i, mm_input)
            for i, mm_input in enumerate(forward_batch.mm_inputs or [])
            if mm_input is not None
        ]
        mm_input_indices = [i for i, _ in mm_inputs_with_indices]
        mm_inputs_list = [mm_input for _, mm_input in mm_inputs_with_indices]
        image_items = self._collect_image_items(mm_inputs_list)
        logical_input_ids = self._logical_input_ids(input_ids, image_items)
        input_embeds = self.model.embed_tokens(logical_input_ids)
        logical_batch = self._logical_forward_batch(forward_batch, image_items)

        oe_state = getattr(logical_batch, "oe_context", None)
        if oe_state is None:
            oe_state = getattr(logical_batch, "n_gram_input_ids", None)
        if len(self.model.oe_grams) > 0 and oe_state is not None:
            input_embeds = self.model._compute_oe_embedding(
                logical_input_ids, logical_batch, input_embeds
            )

        image_embedding, image_mask = self._get_image_embedding_and_mask(
            input_ids=input_ids,
            forward_batch=forward_batch,
            mm_inputs_list=mm_inputs_list,
            mm_input_indices=mm_input_indices,
            image_items=image_items,
        )
        if image_embedding is not None and image_mask is not None:
            indices = torch.where(image_mask.squeeze(dim=-1))[0]
            input_embeds[indices] = image_embedding.to(
                device=input_embeds.device, dtype=input_embeds.dtype
            )

        if forward_batch.input_embeds is not None:
            forward_batch.input_embeds.copy_(input_embeds)
            input_embeds = forward_batch.input_embeds
        forward_batch.mm_inputs = None
        logical_batch.mm_inputs = None
        return input_embeds, logical_input_ids, logical_batch

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors=None,
    ) -> torch.Tensor:
        if self._is_nextn:
            # Share the language NextN forward (pre-norm hidden capture for
            # chained MTP steps) instead of the target causal-LM forward.
            from sglang.srt.models.welmv4_nextn import WeLMV4MoeForCausalLMNextN

            return WeLMV4MoeForCausalLMNextN.forward(
                self, input_ids, positions, forward_batch, input_embeds
            )
        has_image_inputs = (
            getattr(self.config, "image_token_id", None) is not None
            and forward_batch.forward_mode.is_extend_without_speculative()
            and forward_batch.contains_image_inputs()
        )
        if has_image_inputs and input_embeds is None:
            input_embeds, logical_input_ids, logical_batch = (
                self._build_multimodal_input_embeds(input_ids, forward_batch)
            )
            try:
                return super().forward(
                    input_ids=logical_input_ids,
                    positions=positions,
                    forward_batch=logical_batch,
                    input_embeds=input_embeds,
                    pp_proxy_tensors=pp_proxy_tensors,
                    skip_oe_fusion=True,
                )
            finally:
                saved = forward_batch.__dict__.pop("_welm_vl_saved_oe_state", None)
                if saved is not None:
                    setattr(forward_batch, saved[0], saved[1])

        return super().forward(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )


EntryClass = WeLMV4VLMForConditionalGeneration
