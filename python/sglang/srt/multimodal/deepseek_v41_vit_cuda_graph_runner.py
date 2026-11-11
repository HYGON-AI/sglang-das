"""Bucketed CUDA graph runner for the DeepSeek-V4.1 vision tower.

Small images are launch-bound: a ~1500-patch forward issues ~1500 kernels and
the GPU idles between them. Images are padded up to a patch bucket so nearby
sizes replay one graph; the padding is a separate varlen segment, so it never
attends with real patches and every other ViT op is per-token.

Graphs are captured once at server startup (``warmup()``, driven from
``post_load_weights``) from SGLANG_DSV41_VIT_CUDA_GRAPH_WARMUP_PACKS. Serving
is replay-or-eager only: capturing on the serving thread raced the GPU image
preprocessing threads for the caching allocator.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.vision import prepare_vision_attention_metadata
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_memory_pool,
    graph_pool_capture_scope,
    graph_pool_replay_scope,
)
from sglang.srt.models.deepseek_v41_vit import packed_cu_seqlens

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v41_vit import ViT

logger = logging.getLogger(__name__)

# (num_images, padded_patches)
GraphKey = tuple[int, int]


def padded_patch_count(num_patches: int, bucket: int) -> int:
    """Round up to a bucket multiple, keeping at least one pad patch.

    The pad segment must be non-empty so the cu_seqlens length depends only on
    the image count, not on whether the pack happens to fill its bucket.
    """
    return (num_patches // bucket + 1) * bucket


class _CapturedVitGraph(msgspec.Struct, frozen=True):
    graph: Any
    patches: torch.Tensor  # [padded, *patch_shape], model dtype
    cos: torch.Tensor  # [padded, 1, rope_dim], fp32
    sin: torch.Tensor
    cu_seqlens: torch.Tensor  # [num_images + 2], int32; last segment is padding
    output: torch.Tensor  # [padded, vision_dim]


class DeepseekV41ViTCudaGraphRunner:
    def __init__(self, vit: ViT) -> None:
        self.vit = vit
        self.bucket = envs.SGLANG_DSV41_VIT_CUDA_GRAPH_BUCKET_PATCHES.get()
        self.max_patches = envs.SGLANG_DSV41_VIT_CUDA_GRAPH_MAX_PATCHES.get()
        self.capacity = envs.SGLANG_DSV41_VIT_CUDA_GRAPH_CACHE_CAPACITY.get()
        if min(self.bucket, self.max_patches, self.capacity) <= 0:
            raise ValueError(
                "SGLANG_DSV41_VIT_CUDA_GRAPH_* values must be positive, got "
                f"bucket={self.bucket} max_patches={self.max_patches} "
                f"capacity={self.capacity}"
            )
        self.graphs: dict[GraphKey, _CapturedVitGraph] = {}
        self._pool: Any = None
        self._capacity_logged = False
        logger.info(
            "DeepSeek-V4.1 ViT CUDA graph: bucket=%d max_patches=%d capacity=%d "
            "warmup_packs=%s",
            self.bucket,
            self.max_patches,
            self.capacity,
            envs.SGLANG_DSV41_VIT_CUDA_GRAPH_WARMUP_PACKS.get(),
        )

    def run(self, patches: torch.Tensor, grids: list[tuple[int, int]]) -> torch.Tensor:
        num_patches = patches.shape[0]
        padded = padded_patch_count(num_patches, self.bucket)
        entry = self.graphs.get((len(grids), padded))
        if entry is None:
            # Graphs are captured only during startup warmup; the serving
            # path is replay-or-eager so capture never races the GPU image
            # preprocessing threads for the caching allocator.
            return self.vit._forward_eager(patches, grids)
        self._fill_inputs(entry, patches, grids)
        with graph_pool_replay_scope():
            entry.graph.replay()
        return entry.output[:num_patches]

    def warmup(self) -> None:
        """Capture the warmup packs at server startup, before serving traffic.

        Lazy capture during serving raced the GPU image preprocessing threads
        for the caching allocator, which can hand a freshly allocated tensor a
        garbage pointer (VMFault on a fill kernel writing to a host-range
        address). Capturing here is race-free: no other thread touches CUDA.
        """
        weight = self.vit.patch_embed.proj.weight
        for key in _parse_warmup_packs(
            envs.SGLANG_DSV41_VIT_CUDA_GRAPH_WARMUP_PACKS.get()
        ):
            if key in self.graphs:
                continue
            if key[1] > self.max_patches:
                logger.warning(
                    "DeepSeek-V4.1 ViT CUDA graph warmup key=%s exceeds "
                    "max_patches=%d; skipped",
                    key,
                    self.max_patches,
                )
                continue
            if len(self.graphs) >= self.capacity:
                if not self._capacity_logged:
                    logger.warning(
                        "DeepSeek-V4.1 ViT CUDA graph cache reached capacity=%d; "
                        "remaining warmup shapes run eagerly",
                        self.capacity,
                    )
                    self._capacity_logged = True
                break
            patches = torch.zeros(
                (key[1] - 1, *self.vit.patch_shape), dtype=weight.dtype, device=weight.device
            )
            try:
                with graph_pool_capture_scope():
                    entry = self._capture(key, patches, _warmup_grids(key))
            except Exception:
                logger.exception(
                    "DeepSeek-V4.1 ViT CUDA graph warmup capture failed for "
                    "key=%s; this shape runs eagerly",
                    key,
                )
                continue
            self.graphs[key] = entry
        logger.info(
            "DeepSeek-V4.1 ViT CUDA graphs warmed up: %d captured", len(self.graphs)
        )


    # no_grad keeps activations off the autograd path: flash_attn otherwise
    # takes FlashAttnVarlenFunc.apply, whose cu_seqlens filtering (boolean
    # indexing) D2H-syncs and is illegal inside stream capture. Serving
    # forwards already run under no_grad, so replay parity is preserved.
    @torch.no_grad()
    def _capture(
        self, key: GraphKey, patches: torch.Tensor, grids: list[tuple[int, int]]
    ) -> _CapturedVitGraph:
        num_images, padded = key
        device = patches.device
        allocated_before = torch.cuda.memory_allocated(device)
        cos_dim = self.vit.rope_dim
        placeholder = _CapturedVitGraph(
            graph=None,
            patches=torch.zeros(
                (padded, *patches.shape[1:]), dtype=patches.dtype, device=device
            ),
            cos=torch.zeros((padded, 1, cos_dim), dtype=torch.float32, device=device),
            sin=torch.zeros((padded, 1, cos_dim), dtype=torch.float32, device=device),
            cu_seqlens=torch.zeros(num_images + 2, dtype=torch.int32, device=device),
            output=None,
        )
        self._fill_inputs(placeholder, patches, grids)

        def forward() -> torch.Tensor:
            # Built from the device buffer so seq_lens is recomputed on replay;
            # padded is the max segment length, an upper bound the kernels accept.
            metadata = prepare_vision_attention_metadata(
                placeholder.cu_seqlens, device, max_seqlen=padded
            )
            return self.vit._forward_layers(
                placeholder.patches, placeholder.cos, placeholder.sin, metadata
            )

        # Warm up outside capture so Triton JIT and lazy init never run inside it.
        forward()
        torch.cuda.synchronize(device)
        if self._pool is None:
            self._pool = get_or_create_global_graph_memory_pool(torch.cuda)
        graph = torch.cuda.CUDAGraph()
        # Captured at startup warmup, when no other thread touches CUDA, so the
        # default (global) capture error mode is safe and stricter.
        with torch.cuda.graph(graph, pool=self._pool):
            output = forward()
        graph.replay()
        logger.info(
            "Captured DeepSeek-V4.1 ViT CUDA graph: key=%s cache=%d/%d "
            "allocated_delta_mib=%.1f",
            key,
            len(self.graphs) + 1,
            self.capacity,
            (torch.cuda.memory_allocated(device) - allocated_before) / 2**20,
        )
        return msgspec.structs.replace(placeholder, graph=graph, output=output)

    def _fill_inputs(
        self,
        entry: _CapturedVitGraph,
        patches: torch.Tensor,
        grids: list[tuple[int, int]],
    ) -> None:
        # Pad rows keep stale values; they only reach the pad segment.
        num_patches = patches.shape[0]
        padded = entry.patches.shape[0]
        entry.patches[:num_patches].copy_(patches)
        cos, sin = self.vit._packed_cos_sin(grids, device=torch.device("cpu"))
        entry.cos[:num_patches].copy_(cos)
        entry.sin[:num_patches].copy_(sin)
        cu_seqlens = packed_cu_seqlens(grids) + [padded]
        entry.cu_seqlens.copy_(torch.tensor(cu_seqlens, dtype=torch.int32))


def _parse_warmup_packs(raw: str) -> list[GraphKey]:
    """Parse ``num_images:padded_patches`` entries into graph keys."""
    packs: list[GraphKey] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        images, _, padded = token.partition(":")
        try:
            key = (int(images), int(padded))
        except ValueError:
            raise ValueError(
                f"Invalid SGLANG_DSV41_VIT_CUDA_GRAPH_WARMUP_PACKS entry {token!r}; "
                "expected 'num_images:padded_patches'"
            ) from None
        if key[0] <= 0 or key[1] <= 0:
            raise ValueError(
                f"Invalid SGLANG_DSV41_VIT_CUDA_GRAPH_WARMUP_PACKS entry {token!r}; "
                "both values must be positive"
            )
        packs.append(key)
    return packs


def _warmup_grids(key: GraphKey) -> list[tuple[int, int]]:
    """Split a dummy pack of ``padded - 1`` patches into one row per image.

    ``padded - 1`` pads to exactly ``padded``; the grid geometry itself is
    arbitrary because every ViT op is per-token or varlen-segmented.
    """
    num_images, padded = key
    base, remainder = divmod(padded - 1, num_images)
    return [(1, base + (i < remainder)) for i in range(num_images)]
