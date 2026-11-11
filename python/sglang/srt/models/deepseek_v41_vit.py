"""DeepSeek-V4.1 vision tower and aligner."""

import logging
from functools import lru_cache

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.vision import (
    VisionAttention,
    VisionAttentionMetadata,
    prepare_vision_attention_metadata,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.utils import is_hcu

logger = logging.getLogger(__name__)

_lightop_rms_norm = None
if is_hcu():
    try:
        # Older lightop wheels lack this entry; the native norm is used then.
        from lightop.norm import rms_norm_no_weight as _lightop_rms_norm
    except ImportError:
        logger.warning(
            "lightop.norm.rms_norm_no_weight unavailable; "
            "DeepSeek-V4.1 ViT RMSNorm stays native"
        )


@triton.jit
def _add_rmsnorm_kernel(
    x_ptr,
    residual_ptr,
    w_ptr,
    out_ptr,
    residual_out_ptr,
    n_cols,
    eps,
    BLOCK_N: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    x = tl.load(x_ptr + token * n_cols + cols, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + token * n_cols + cols, mask=mask, other=0.0)
    # Add in fp32 and round once: the separate bf16 add this replaces does the
    # same, so the norm below reads a bf16 sum rather than an fp32 partial one.
    summed = (x.to(tl.float32) + residual.to(tl.float32)).to(
        out_ptr.dtype.element_ty
    )
    tl.store(residual_out_ptr + token * n_cols + cols, summed, mask=mask)
    normed = summed.to(tl.float32)
    var = tl.sum(normed * normed, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        out_ptr + token * n_cols + cols,
        (normed * rstd * w).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


class _VisionRMSNorm(RMSNorm):
    """RMSNorm with an fp32 weight over bf16 activations.

    sglang's fused kernels require weight and input to share a dtype, so the
    fused path uses lightop's mixed-dtype entry (fp32 math, one final rounding,
    same as forward_native); everything else stays native.

    ``forward_add`` folds the residual add into the same launch. The round-to-
    bf16 of the sum happens before the variance, matching the reference
    ``norm(x + residual)`` order bit for bit.
    """

    def __init__(self, dim: int, fused: bool):
        super().__init__(dim, eps=1e-6, weight_dtype=torch.float32, force_native=True)
        self.use_lightop = fused and _lightop_rms_norm is not None
        self.use_add_kernel = fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (self.use_lightop and x.is_cuda):
            return super().forward(x)
        x = x.contiguous()
        out = torch.empty_like(x)
        _lightop_rms_norm(out, x, self.weight, self.variance_epsilon)
        return out

    def forward_add(
        self, x: torch.Tensor, residual: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(norm(x + residual), x + residual)`` in one launch and one pass.

        ``residual`` is None only for the first add of the tower, where there is
        nothing to sum yet and the sum is ``x`` itself.
        """
        if residual is None:
            return self.forward(x), x
        if not (self.use_add_kernel and x.is_cuda):
            summed = x + residual
            return self.forward(summed), summed
        x = x.contiguous()
        residual = residual.contiguous()
        out = torch.empty_like(x)
        residual_out = torch.empty_like(residual)
        _add_rmsnorm_kernel[(x.shape[0],)](
            x,
            residual,
            self.weight,
            out,
            residual_out,
            self.hidden_size,
            float(self.variance_epsilon),
            BLOCK_N=triton.next_power_of_2(self.hidden_size),
        )
        return out, residual_out


@lru_cache(8)
def get_vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


@triton.jit
def _rotate_heads(
    src,
    dst,
    src_stride_h,
    dst_stride_h,
    heads,
    dims,
    mask,
    cos,
    sin,
    HALF: tl.constexpr,
):
    x1 = tl.load(src + heads * src_stride_h + dims, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(src + heads * src_stride_h + dims + HALF, mask=mask, other=0.0).to(
        tl.float32
    )
    out_dtype = dst.dtype.element_ty
    tl.store(dst + heads * dst_stride_h + dims, (x1 * cos - x2 * sin).to(out_dtype), mask=mask)
    tl.store(
        dst + heads * dst_stride_h + dims + HALF,
        (x2 * cos + x1 * sin).to(out_dtype),
        mask=mask,
    )


@triton.jit
def _vision_rotary_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    cos_ptr,
    sin_ptr,
    q_stride_t,
    q_stride_h,
    k_stride_t,
    k_stride_h,
    q_out_stride_t,
    q_out_stride_h,
    k_out_stride_t,
    k_out_stride_h,
    cos_stride_t,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    heads = tl.arange(0, BLOCK_H)[:, None]
    dims = tl.arange(0, BLOCK_D)[None, :]
    dim_mask = dims < HALF
    cos = tl.load(cos_ptr + token * cos_stride_t + dims, mask=dim_mask, other=0.0)
    sin = tl.load(sin_ptr + token * cos_stride_t + dims, mask=dim_mask, other=0.0)
    _rotate_heads(
        q_ptr + token * q_stride_t,
        q_out_ptr + token * q_out_stride_t,
        q_stride_h,
        q_out_stride_h,
        heads,
        dims,
        (heads < NUM_Q_HEADS) & dim_mask,
        cos,
        sin,
        HALF,
    )
    _rotate_heads(
        k_ptr + token * k_stride_t,
        k_out_ptr + token * k_out_stride_t,
        k_stride_h,
        k_out_stride_h,
        heads,
        dims,
        (heads < NUM_K_HEADS) & dim_mask,
        cos,
        sin,
        HALF,
    )


def _apply_rotary_triton(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One launch for q and k; same fp32 math as ``apply_rotary``.

    q/k are [tokens, heads, head_dim]; cos/sin are fp32 [tokens, 1, head_dim // 2].
    """
    half = cos.shape[-1]
    assert q.dim() == 3 and k.dim() == 3, (q.shape, k.shape)
    assert q.shape[-1] == 2 * half and k.shape[-1] == 2 * half, (q.shape, cos.shape)
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and cos.stride(-1) == 1
    assert cos.stride() == sin.stride() and cos.dtype == torch.float32
    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    _vision_rotary_kernel[(q.shape[0],)](
        q,
        k,
        q_out,
        k_out,
        cos,
        sin,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        q_out.stride(0),
        q_out.stride(1),
        k_out.stride(0),
        k_out.stride(1),
        cos.stride(0),
        NUM_Q_HEADS=q.shape[1],
        NUM_K_HEADS=k.shape[1],
        HALF=half,
        BLOCK_H=triton.next_power_of_2(max(q.shape[1], k.shape[1])),
        BLOCK_D=triton.next_power_of_2(half),
    )
    return q_out, k_out


def apply_vision_rotary(q, k, position_embeddings, x_shape):
    # The reference pairs the two halves of each head, with FP32 arithmetic.
    cos, sin = position_embeddings
    return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)


def apply_vision_rotary_fused(q, k, position_embeddings, x_shape):
    cos, sin = position_embeddings
    if not q.is_cuda:
        return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
    return _apply_rotary_triton(q, k, cos, sin)


# Arbitrary cap on patches per packed ViT forward, to bound peak activation memory;
# a single image larger than this still runs alone.
MAX_PACKED_VIT_PATCHES = 65536


def split_vit_pack_groups(
    grids: list[tuple[int, int]], max_patches: int = MAX_PACKED_VIT_PATCHES
) -> list[range]:
    """Split images, in order, into contiguous groups of at most ``max_patches``."""
    groups = []
    start = 0
    group_patches = 0
    for index, (n_h, n_w) in enumerate(grids):
        num_patches = n_h * n_w
        if index > start and group_patches + num_patches > max_patches:
            groups.append(range(start, index))
            start = index
            group_patches = 0
        group_patches += num_patches
    groups.append(range(start, len(grids)))
    return groups


def packed_cu_seqlens(grids: list[tuple[int, int]]) -> list[int]:
    cu_seqlens = [0]
    for n_h, n_w in grids:
        cu_seqlens.append(cu_seqlens[-1] + n_h * n_w)
    return cu_seqlens


class PatchEmbed(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.proj = nn.Linear(3 * args.vision_patch_size**2, args.vision_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


class Attention(VisionAttention):
    def __init__(self, args, fused: bool):
        super().__init__(
            embed_dim=args.vision_dim,
            num_heads=args.vision_n_heads,
            projection_size=args.vision_dim,
            use_qkv_parallel=True,
            use_data_parallel=True,
            customized_position_embedding_applier=(
                apply_vision_rotary_fused if fused else apply_vision_rotary
            ),
            customized_position_embedding_applier_supports_strided_qkv=fused,
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        metadata: VisionAttentionMetadata,
    ) -> torch.Tensor:
        return (
            super()
            .forward(
                x,
                position_embeddings=(cos, sin),
                forward_metadata=metadata,
            )
            .squeeze(0)
        )


class MLP(nn.Module):
    def __init__(self, args, fused: bool):
        super().__init__()
        self.w1 = nn.Linear(args.vision_dim, 2 * args.vision_inter_dim, bias=False)
        self.w2 = nn.Linear(args.vision_inter_dim, args.vision_dim, bias=False)
        # w1 emits [gate | up], the layout SiluAndMul expects.
        self.act = SiluAndMul() if fused else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act is not None and x.is_cuda:
            return self.w2(self.act(self.w1(x)))
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, args, fused: bool):
        super().__init__()
        self.norm1 = _VisionRMSNorm(args.vision_dim, fused=fused)
        self.attn = Attention(args, fused=fused)
        self.norm2 = _VisionRMSNorm(args.vision_dim, fused=fused)
        self.mlp = MLP(args, fused=fused)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        metadata: VisionAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Consume the residual stream ``x + residual``; return the next one.

        ``residual`` is None for the first block only. The stream is threaded
        through both norms so each residual add costs no launch of its own.
        """
        normed, residual = self.norm1.forward_add(x, residual)
        attn_out = self.attn(normed, cos, sin, metadata)
        normed, residual = self.norm2.forward_add(attn_out, residual)
        return self.mlp(normed), residual


class ViT(nn.Module):
    """DeepSeek ViT: full bidirectional attention within each image, with 2D RoPE.

    Several images can share one forward: their patches are concatenated and
    attention is confined to each image's segment through varlen cu_seqlens.
    """

    def __init__(self, args):
        super().__init__()
        fused = envs.SGLANG_DSV41_VIT_FUSED_KERNELS.get()
        self.rope_dim = args.vision_dim // args.vision_n_heads // 2
        self.rope_theta = args.vision_rope_theta
        self.patch_embed = PatchEmbed(args)
        self.blocks = nn.ModuleList(
            [Block(args, fused=fused) for _ in range(args.vision_n_layers)]
        )
        self.norm = _VisionRMSNorm(args.vision_dim, fused=fused)
        self.patch_shape = (3, args.vision_patch_size, args.vision_patch_size)
        self.cuda_graph_enabled = envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get()
        self.cuda_graph_runner = None

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        return self.forward_packed(patches, grids=[(n_h, n_w)])

    def forward_packed(
        self, patches: torch.Tensor, grids: list[tuple[int, int]]
    ) -> torch.Tensor:
        """Encode images whose patches are concatenated in ``grids`` order.

        ``patches`` is [sum(n_h * n_w), ...] with trailing dimensions that
        flatten to one patch; returns [sum(n_h * n_w), vision_dim].
        With CUDA graphs on, the result may be a view of a graph-owned buffer
        that the next replay overwrites, so consume it before the next call.
        """
        assert patches.shape[0] == sum(h * w for h, w in grids), (patches.shape, grids)
        if self.cuda_graph_enabled and patches.is_cuda:
            if self.cuda_graph_runner is None:
                from sglang.srt.multimodal.deepseek_v41_vit_cuda_graph_runner import (
                    DeepseekV41ViTCudaGraphRunner,
                )

                self.cuda_graph_runner = DeepseekV41ViTCudaGraphRunner(self)
            return self.cuda_graph_runner.run(patches, grids)
        return self._forward_eager(patches, grids)

    def warmup_cuda_graphs(self) -> None:
        """Capture ViT CUDA graphs at server startup, before serving traffic.

        Called from ``post_load_weights`` so capture never runs concurrently
        with the GPU image preprocessing threads.
        """
        if not (self.cuda_graph_enabled and torch.cuda.is_available()):
            return
        try:
            if self.cuda_graph_runner is None:
                from sglang.srt.multimodal.deepseek_v41_vit_cuda_graph_runner import (
                    DeepseekV41ViTCudaGraphRunner,
                )

                self.cuda_graph_runner = DeepseekV41ViTCudaGraphRunner(self)
            self.cuda_graph_runner.warmup()
        except Exception:
            logger.exception(
                "DeepSeek-V4.1 ViT CUDA graph warmup failed; the vision tower "
                "runs eagerly"
            )
            self.cuda_graph_runner = None
            self.cuda_graph_enabled = False

    def _forward_eager(
        self, patches: torch.Tensor, grids: list[tuple[int, int]]
    ) -> torch.Tensor:
        cos, sin = self._packed_cos_sin(grids, device=patches.device)
        # Passing the known length avoids device-to-host length discovery per layer.
        metadata = prepare_vision_attention_metadata(
            torch.tensor(packed_cu_seqlens(grids), dtype=torch.int32),
            patches.device,
            max_seqlen=max(h * w for h, w in grids),
        )
        return self._forward_layers(patches, cos, sin, metadata)

    def _forward_layers(
        self,
        patches: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        metadata: VisionAttentionMetadata,
    ) -> torch.Tensor:
        # Device work only: no host sync or H2D copy, so it can be graph-captured.
        # ``residual`` is None before the first block: its first add is against
        # the patch embedding, which has nothing to fold into yet.
        x = self.patch_embed(patches)
        residual = None
        for block in self.blocks:
            x, residual = block(x, residual, cos, sin, metadata)
        # The last add is the one every block would have done for the next
        # layer, so it rides on the final norm instead of its own launch.
        return self.norm.forward_add(x, residual)[0]

    def _packed_cos_sin(
        self, grids: list[tuple[int, int]], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # RoPE positions restart at (0, 0) for every image.
        tables = [
            get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
            for n_h, n_w in grids
        ]
        if len(tables) == 1:
            cos, sin = tables[0]
        else:
            cos = torch.cat([c for c, _ in tables])
            sin = torch.cat([s for _, s in tables])
        return cos.to(device), sin.to(device)


class Aligner(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.downsample_ratio = args.vision_downsample_ratio
        in_dim = args.vision_dim * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, args.dim)
        self.w2 = nn.Linear(args.dim, args.dim)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        return self._project(self._downsample(x, n_h=n_h, n_w=n_w))

    def forward_packed(
        self, x: torch.Tensor, grids: list[tuple[int, int]]
    ) -> list[torch.Tensor]:
        """Align packed ViT output; returns one [aligned_tokens, dim] tensor per image."""
        per_image = x.split([n_h * n_w for n_h, n_w in grids])
        downsampled = [
            self._downsample(image, n_h=n_h, n_w=n_w)
            for image, (n_h, n_w) in zip(per_image, grids)
        ]
        projected = self._project(torch.cat(downsampled))
        return list(projected.split([d.shape[0] for d in downsampled]))

    def _downsample(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        return F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))
