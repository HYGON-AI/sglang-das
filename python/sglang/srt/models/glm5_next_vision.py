# SPDX-License-Identifier: Apache-2.0
"""GLM5-Next vision configuration using main's vision attention interfaces."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.models.glm4v.configuration_glm4v import Glm4vVisionConfig

from sglang.srt.layers.attention.vision import (
    VisionAttention,
    VisionAttentionMetadata,
    prepare_vision_attention_metadata,
)
from sglang.srt.layers.layernorm import LayerNorm, RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.models.glm4v import Glm4vRMSNorm, Glm4vVisionPatchEmbed
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix, is_npu


@torch.compile
def swiglu_clamped(y: torch.Tensor, limit: float):
    gate, up = torch.chunk(y, 2, dim=-1)
    gate = torch.clamp(gate, max=limit)
    up = torch.clamp(up, min=-limit, max=limit)
    return F.silu(gate) * up


class Glm4vVisionMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
        swiglu_limit: float = float("inf"),
    ):
        super().__init__()
        self.tp_size = 1 if use_data_parallel else get_parallel().tp_size
        self.tp_rank = 0 if use_data_parallel else get_parallel().tp_rank
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=in_features,
            output_sizes=[hidden_features] * 2,  # [gate_proj, up_proj]
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )
        self.down_proj = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )
        self.act_fn = swiglu_clamped
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up, self.swiglu_limit)
        x, _ = self.down_proj(x)
        return x


class Glm4vVisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        num_heads: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        attn_qkv_bias: bool = True,
        num_dummy_heads: int = 0,
        rms_norm_eps: float = 1e-5,
        use_data_parallel: bool = False,
        proj_bias: bool = False,
        qk_normalization: bool = False,
        qk_normalization_by_head_size: bool = False,
        mlp_linear_bias: bool = False,
        swiglu_limit: float = float("inf"),
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=rms_norm_eps)
        self.norm2 = RMSNorm(dim, eps=rms_norm_eps)

        self.attn = VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            use_qkv_parallel=True,
            proj_bias=proj_bias,
            qkv_bias=attn_qkv_bias,
            qk_normalization=qk_normalization,
            qk_normalization_by_head_size=qk_normalization_by_head_size,
            layer_norm_eps=rms_norm_eps,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            num_dummy_heads=num_dummy_heads,
            use_data_parallel=use_data_parallel,
        )
        self.mlp = Glm4vVisionMLP(
            dim,
            intermediate_dim,
            bias=mlp_linear_bias,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
            use_data_parallel=use_data_parallel,
            swiglu_limit=swiglu_limit,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        forward_metadata: Optional[VisionAttentionMetadata] = None,
    ) -> torch.Tensor:
        S, B, H = x.shape
        # norm1: flatten to 2D -> [S*B, H], then reshape back
        x2d = x.reshape(-1, H)
        hidden_states = self.norm1(x2d).reshape(S, B, H)

        # Attention expects [B, S, H]
        hidden_states = rearrange(hidden_states, "s b h -> b s h")
        attn = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            forward_metadata=forward_metadata,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
        )
        attn = rearrange(attn, "b s h -> s b h")

        # norm2 with fused residual-add: also 2D
        attn2d = attn.reshape(-1, H)
        x_norm_2d, x_after_add_2d = self.norm2(x2d, residual=attn2d)
        x_norm = x_norm_2d.reshape(S, B, H)
        x_after_add = x_after_add_2d.reshape(S, B, H)

        # MLP and final residual
        mlp_out = self.mlp(x_norm)
        x = x_after_add + mlp_out
        return x


class Glm4vPatchMerger(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
        use_data_parallel: bool = False,
        layer_norm_eps: float = 1e-5,
        swiglu_limit: float = float("inf"),
    ) -> None:
        super().__init__()
        self.hidden_size = d_model
        tp_size = 1 if use_data_parallel else get_parallel().tp_size
        tp_rank = 0 if use_data_parallel else get_parallel().tp_rank
        self.proj = ReplicatedLinear(
            self.hidden_size,
            self.hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("proj", prefix),
        )
        self.post_projection_norm = LayerNorm(self.hidden_size, eps=layer_norm_eps)
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[context_dim] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        self.down_proj = RowParallelLinear(
            context_dim,
            self.hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        self.extra_activation_func = nn.GELU()
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor):
        x, _ = self.proj(x)
        x = self.extra_activation_func(self.post_projection_norm(x))
        gate_up, _ = self.gate_up_proj(x)
        x = swiglu_clamped(gate_up, self.swiglu_limit)
        x, _ = self.down_proj(x)
        return x


class Glm5NextVisionModel(nn.Module):
    def __init__(
        self,
        vision_config: Glm4vVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
        text_config=None,
        swiglu_limit: Optional[float] = None,
    ) -> None:
        super().__init__()

        patch_size = vision_config.patch_size
        temporal_patch_size = vision_config.temporal_patch_size
        in_channels = vision_config.in_channels
        depth = vision_config.depth
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads

        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.out_hidden_size = vision_config.out_hidden_size
        self.intermediate_dim = vision_config.intermediate_size
        self.use_data_parallel = use_data_parallel

        if swiglu_limit is None:
            swiglu_limit = getattr(vision_config, "swiglu_limit", None)
        if swiglu_limit is None and text_config is not None:
            swiglu_limit = getattr(text_config, "swiglu_limit", None)
        if swiglu_limit is None:
            swiglu_limit = float("inf")

        self.patch_embed = Glm4vVisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=self.hidden_size,
        )

        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = get_rope(
            head_size=head_dim,
            rotary_dim=head_dim // 2,
            max_position=8192,
            base=10000.0,
            is_neox_style=True,
        )

        self.blocks = nn.ModuleList(
            [
                Glm4vVisionBlock(
                    dim=self.hidden_size,
                    intermediate_dim=self.intermediate_dim,
                    num_heads=self.num_heads,
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{layer_idx}", prefix),
                    num_dummy_heads=vision_config.num_dummy_heads,
                    rms_norm_eps=vision_config.rms_norm_eps,
                    attn_qkv_bias=vision_config.attention_bias,
                    use_data_parallel=use_data_parallel,
                    proj_bias=True,
                    qk_normalization_by_head_size=True,
                    mlp_linear_bias=True,
                    swiglu_limit=swiglu_limit,
                )
                for layer_idx in range(depth)
            ]
        )

        merger_context_dim = getattr(
            vision_config, "projection_intermediate_size", None
        )
        if merger_context_dim is None:
            merger_context_dim = (
                text_config.intermediate_size
                if text_config is not None
                else vision_config.intermediate_size
            )

        self.merger = Glm4vPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=merger_context_dim,
            quant_config=quant_config,
            bias=False,
            prefix=add_prefix("merger", prefix),
            use_data_parallel=use_data_parallel,
            layer_norm_eps=vision_config.rms_norm_eps,
            swiglu_limit=swiglu_limit,
        )

        self.downsample = nn.Conv2d(
            in_channels=vision_config.hidden_size,
            out_channels=vision_config.out_hidden_size,
            kernel_size=vision_config.spatial_merge_size,
            stride=vision_config.spatial_merge_size,
        )
        self.post_layernorm = Glm4vRMSNorm(
            vision_config.hidden_size, eps=vision_config.rms_norm_eps
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(
        self, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            hpos_ids = (
                hpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            wpos_ids = (
                wpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0).to(self.device, non_blocking=True)
        max_grid_size = grid_thw[:, 1:].max()

        # Use pre-computed cos_sin_cache from RotaryEmbedding
        cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)

        cos_combined = cos[pos_ids].flatten(1)
        sin_combined = sin[pos_ids].flatten(1)
        return cos_combined, sin_combined, pos_ids

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        # patchify
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        # compute position embedding
        rotary_pos_emb_cos, rotary_pos_emb_sin, _ = self.rot_pos_emb(
            grid_thw
        )
        # compute cu_seqlens
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = int(seq_lens.max().item())

        rotary_pos_emb_cos = torch.cat([rotary_pos_emb_cos, rotary_pos_emb_cos], dim=-1)
        rotary_pos_emb_sin = torch.cat([rotary_pos_emb_sin, rotary_pos_emb_sin], dim=-1)

        # cu_seqlens must be on cpu because of npu_flash_attention_unpad operator restriction
        if is_npu():
            cu_seqlens = cu_seqlens.to("cpu")
        else:
            cu_seqlens = cu_seqlens.to(self.device, non_blocking=True)

        forward_metadata = prepare_vision_attention_metadata(
            cu_seqlens, device=self.device, max_seqlen=max_seqlen
        )

        # x.shape: (s, b, d) where b=1 for vision processing
        # transformers
        x = x.unsqueeze(1)
        for blk in self.blocks:
            x = blk(
                x,
                cu_seqlens=cu_seqlens,
                forward_metadata=forward_metadata,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
            )

        # adapter
        x = self.post_layernorm(x)
        x = x.view(-1, self.spatial_merge_size, self.spatial_merge_size, x.shape[-1])
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x).view(-1, self.out_hidden_size)
        x = self.merger(x)

        return x
