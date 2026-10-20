"""KV pools carrying the QSA sparse-attention indexer caches.

``QSATokenToKVPool`` (compressed, Qwen4-Exp) adds the per-token BF16 index-key
state, its RoPE coordinates, and the paged compressed-K cache on top of the
hybrid full/linear KV pool. ``QwenDSATokenToKVPool`` (tokenwise,
Qwen3Next-DSA) adds only the flat per-token index-K cache.
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch

from sglang.srt.mem_cache.memory_pool import GB, HybridLinearKVPool, MambaPool
from sglang.srt.utils import is_hip

_IS_HIP = is_hip()

# Must stay in sync with _FP8_DECODE_ENV in
# sglang/srt/layers/attention/qsa/mqa.py.
_SGLANG_QSA_INDEXER_FP8_DECODE_ENV = "SGLANG_QSA_INDEXER_FP8_DECODE"
_FP8_DECODE_OFF_VALUES = ("0", "off", "false")


def _index_k_bytes(*, kv_heads: int, head_dim: int, dtype: torch.dtype) -> int:
    return kv_heads * head_dim * dtype.itemsize


def _scatter_rows(buffer: torch.Tensor, loc: torch.Tensor, values: torch.Tensor) -> None:
    """Write ``values[i]`` into ``buffer[loc[i]]``.

    On HIP, advanced indexing with duplicate destinations (QSA non-boundary
    rows all dump into reserved slot 0) can VMFault. Serialize those stores.
    """
    loc = loc.long().reshape(-1)
    values = values.to(buffer.dtype)
    if loc.numel() != values.shape[0]:
        raise ValueError(
            f"QSA scatter size mismatch: loc={loc.numel()}, values={values.shape[0]}"
        )
    if loc.numel() == 0:
        return
    max_idx = max(int(buffer.shape[0]) - 1, 0)
    loc = loc.clamp(0, max_idx)
    capturing = False
    if _IS_HIP:
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )

        capturing = get_is_capture_mode()
    # HIP advanced indexing with duplicate destinations can VMFault in eager
    # mode, so serialize those stores. The Python loop is illegal during CUDA
    # graph capture (hipErrorStreamCaptureUnsupported); use the vectorized
    # write while capturing.
    if (not _IS_HIP) or capturing:
        buffer[loc] = values
        return
    for i in range(loc.shape[0]):
        buffer[loc[i]] = values[i]


class QSATokenToKVPool(HybridLinearKVPool):
    """Hybrid KV pool with the minimal BF16 state required by simple QSA."""

    # DSV4-style compressed addressing: the full-KV allocator is paged with
    # the page a multiple of the compress ratio, so every compression
    # group's raw tokens are contiguous in one page and
    # ``compressed_slot = full_slot // compress_ratio`` is a stable,
    # collision-free mapping (any of the group's slots floor-divides to the
    # same value). The compressed cache therefore has exactly
    # ``full_slots // ratio`` slots, its lifecycle rides the full-KV
    # allocator and radix tree (page-granular sharing shares compressed
    # slots by arithmetic), and no ownership bookkeeping exists. Full slot 0
    # is the pools' reserved padding slot, so compressed slot 0 stays the
    # inert dump target for non-boundary rows.
    index_state_dtype = torch.bfloat16

    @classmethod
    def qsa_bytes_per_token(
        cls, *, kv_heads: int, head_dim: int, compress_ratio: int, num_layers: int
    ) -> int:
        """Per-token cost of the QSA index caches: the compressed keys only.

        Pre-compression state is a per-request ring of ``compress_ratio``
        slots (the pending group's members), not a per-token cache, so it
        does not price per token; its total is bounded by the request-slot
        count and stays outside this budget like the other per-request
        buffers.
        """
        index_k_bytes = _index_k_bytes(
            kv_heads=kv_heads, head_dim=head_dim, dtype=cls.index_state_dtype
        )
        return index_k_bytes // compress_ratio * num_layers


    def __init__(
        self,
        *,
        size: int,
        dtype: torch.dtype,
        page_size: int,
        head_num: int,
        head_dim: int,
        full_attention_layer_ids: List[int],
        device: str,
        mamba_pool: MambaPool,
        qsa_index_kv_heads: int,
        qsa_index_head_dim: int,
        qsa_compress_ratio: int,
        qsa_token_topk: int,
        num_request_slots: int,
        enable_memory_saver: bool = False,
        enable_kv_cache_copy: bool = False,
        start_layer: Optional[int] = None,
        full_kv_pool_class: Optional[type] = None,
        quant_method=None,
        post_capture_active: bool = False,
    ):
        if page_size <= 1 or page_size % qsa_compress_ratio != 0:
            raise ValueError(
                "compressed QSA requires a paged full-KV cache with the page "
                "a multiple of the compress ratio (compressed slots are "
                f"full_slot // ratio): page_size={page_size}, "
                f"ratio={qsa_compress_ratio}. With MambaRadixCache this "
                "needs the mamba extra-buffer strategy or "
                "--disable-radix-cache (see the Qwen4-Exp arg overrides)."
            )
        # The base __init__ computes mem_usage through the overridden
        # get_kv_size_bytes before the QSA buffers exist; give them empty
        # placeholders first and recompute mem_usage at the end.
        self.qsa_key_state_buffer_pool = []
        self.qsa_compressed_k_buffer_pool = []
        self.qsa_compressed_fp8_k_buffer_pool = []
        self.qsa_rope_position_buffer = torch.empty(0)
        super().__init__(
            size=size,
            dtype=dtype,
            page_size=page_size,
            head_num=head_num,
            head_dim=head_dim,
            full_attention_layer_ids=full_attention_layer_ids,
            device=device,
            mamba_pool=mamba_pool,
            enable_memory_saver=enable_memory_saver,
            enable_kv_cache_copy=enable_kv_cache_copy,
            use_mla=False,
            start_layer=start_layer,
            full_kv_pool_class=full_kv_pool_class,
            quant_method=quant_method,
            post_capture_active=post_capture_active,
        )
        if (
            min(
                qsa_index_kv_heads,
                qsa_index_head_dim,
                qsa_compress_ratio,
                qsa_token_topk,
            )
            <= 0
        ):
            raise ValueError("QSA cache configuration values must be positive")
        if qsa_token_topk % qsa_compress_ratio != 0:
            raise ValueError("qsa_token_topk must be divisible by qsa_compress_ratio")
        self.qsa_compress_ratio = int(qsa_compress_ratio)
        self.qsa_index_head_dim = int(qsa_index_head_dim)
        self.qsa_index_kv_heads = int(qsa_index_kv_heads)
        self.qsa_token_topk = int(qsa_token_topk)
        self.qsa_block_topk = self.qsa_token_topk // self.qsa_compress_ratio
        state_size = size + page_size
        # Compressed slots mirror the full-KV slot space 1:ratio; the "page"
        # seen by the scoring kernels is one full-KV page's worth of groups.
        self.qsa_compressed_page_size = page_size // self.qsa_compress_ratio
        self.qsa_compressed_capacity = -(state_size // -self.qsa_compress_ratio)
        # Pre-compression index-K state is a per-request RING, not a
        # per-token cache: once a group's compressed key is written, its raw
        # members are never read again, and page-granular prefix sharing
        # keeps every extend chunk group-aligned, so the only state that
        # must survive a forward is the pending group's members -- at most
        # ``ratio`` tokens per request, addressed as
        # ``req_pool_idx * ratio + position % ratio``. Request slot 0 is
        # never allocated, so ring rows [0, ratio) double as the inert dump
        # for tokens whose group already compressed in the same forward.
        if num_request_slots <= 0:
            raise ValueError(
                f"QSA pending ring needs request slots, got {num_request_slots}"
            )
        self.qsa_num_request_slots = int(num_request_slots)
        ring_slots = self.qsa_num_request_slots * self.qsa_compress_ratio
        self.qsa_key_state_buffer_pool = [
            torch.zeros(
                (ring_slots, self.qsa_index_kv_heads, self.qsa_index_head_dim),
                dtype=self.index_state_dtype,
                device=device,
            )
            for _ in full_attention_layer_ids
        ]
        # RoPE coordinates are layer-independent.  Keep the exact Qwen4-Exp MRoPE
        # position of every pending key so compression can rotate the pooled
        # key with the group's real starting coordinate.
        self.qsa_rope_position_buffer = torch.zeros(
            (ring_slots, 3), dtype=torch.int64, device=device
        )
        # One contiguous allocation behind per-layer views: every layer's
        # compressed pages are addressable from a single base pointer.
        self.qsa_compressed_flat = torch.zeros(
            (
                len(full_attention_layer_ids),
                self.qsa_compressed_capacity
                * self.qsa_index_kv_heads
                * self.qsa_index_head_dim,
            ),
            dtype=self.index_state_dtype,
            device=device,
        )
        self.qsa_compressed_k_buffer_pool = [
            self.qsa_compressed_flat[layer_offset].view(
                self.qsa_compressed_capacity,
                self.qsa_index_kv_heads,
                self.qsa_index_head_dim,
            )
            for layer_offset in range(len(full_attention_layer_ids))
        ]
        # FP8 shadow of the compressed-K cache, in the aiter packed layout the
        # gfx938 BoltOPs fp8 decode kernel consumes ([pages, page_size *
        # (head_dim + 4)] bytes: per page, page_size*head_dim K bytes followed
        # by page_size fp32 per-token scales). It is kept in sync incrementally
        # by update_qsa_compressed_fp8_k_buffer on every compressed-K write so
        # decode never repacks the whole pool per step.
        #
        # One extra scratch page per layer absorbs the QSA dump rows
        # (write_locs == 0): the inert bf16 dump slot would otherwise make the
        # vectorized scatter target duplicate destinations, which VMFaults on
        # HIP. Redirecting every dump row to the scratch page keeps all
        # destinations unique, so the update stays a single vectorized store
        # that is also CUDA-graph capture safe.
        self.qsa_compressed_fp8_k_buffer_pool = []
        if (
            _IS_HIP
            and self.qsa_index_kv_heads == 1
            and self.qsa_compressed_capacity % self.qsa_compressed_page_size == 0
            and os.environ.get(
                _SGLANG_QSA_INDEXER_FP8_DECODE_ENV, "1"
            ).strip().lower()
            not in _FP8_DECODE_OFF_VALUES
        ):
            pages = self.qsa_compressed_capacity // self.qsa_compressed_page_size
            if pages > 0:
                page_bytes = self.qsa_compressed_page_size * (
                    self.qsa_index_head_dim + 4
                )
                self.qsa_compressed_fp8_pages = pages
                self.qsa_compressed_fp8_flat = torch.zeros(
                    (len(full_attention_layer_ids), (pages + 1) * page_bytes),
                    dtype=torch.uint8,
                    device=device,
                )
                for layer_offset in range(len(full_attention_layer_ids)):
                    shadow = self.qsa_compressed_fp8_flat[layer_offset].view(
                        pages + 1, page_bytes
                    )
                    # Unwritten rows keep a zero K payload; the unit fp32
                    # scale keeps their logit contribution at zero, matching
                    # the zero rows of the bf16 pool.
                    shadow[
                        :, self.qsa_index_head_dim * self.qsa_compressed_page_size :
                    ].view(torch.float32).fill_(1.0)
                    self.qsa_compressed_fp8_k_buffer_pool.append(shadow)
        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB

    def get_qsa_key_state_buffer(self, layer_id: int) -> torch.Tensor:
        return self.qsa_key_state_buffer_pool[
            self._transfer_full_attention_id(layer_id)
        ]

    def set_qsa_key_state_buffer(
        self, layer_id: int, loc: torch.Tensor, token_k: torch.Tensor
    ) -> None:
        buffer = self.get_qsa_key_state_buffer(layer_id)
        _scatter_rows(buffer, loc, token_k)

    def set_qsa_rope_position_buffer(
        self, loc: torch.Tensor, positions: torch.Tensor
    ) -> None:
        positions = positions.long()
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        if positions.ndim != 2 or positions.shape[0] != 3:
            raise ValueError(
                f"QSA RoPE positions must be [tokens] or [3, tokens], got {positions.shape}"
            )
        _scatter_rows(
            self.qsa_rope_position_buffer, loc, positions.transpose(0, 1)
        )

    def get_qsa_rope_position_buffer(self, loc: torch.Tensor) -> torch.Tensor:
        return self.qsa_rope_position_buffer[loc.long()]

    def get_qsa_compressed_k_buffer(self, layer_id: int) -> torch.Tensor:
        return self.qsa_compressed_k_buffer_pool[
            self._transfer_full_attention_id(layer_id)
        ]

    def get_qsa_compressed_fp8_k_buffer(self, layer_id: int) -> Optional[torch.Tensor]:
        """Return the fp8 shadow cache view ``[pages, page_size, 1, D+4]``.

        The trailing scratch page is excluded. ``None`` when the fp8 decode
        shadow pool is unavailable (NVIDIA, kv_heads != 1, or disabled via
        SGLANG_QSA_INDEXER_FP8_DECODE).
        """
        if not self.qsa_compressed_fp8_k_buffer_pool:
            return None
        shadow = self.qsa_compressed_fp8_k_buffer_pool[
            self._transfer_full_attention_id(layer_id)
        ]
        return shadow[: self.qsa_compressed_fp8_pages].view(
            -1, self.qsa_compressed_page_size, 1, self.qsa_index_head_dim + 4
        )

    def update_qsa_compressed_fp8_k_buffer(
        self, layer_id: int, loc: torch.Tensor
    ) -> None:
        """Mirror the compressed-K rows at ``loc`` into the fp8 shadow pool.

        Reads back the rows just written to the bf16 pool (small, batch-sized
        gather), quantizes them to e4m3 with a per-row amax/448 scale, and
        scatters the bytes into the aiter-packed shadow layout. All ops are
        shape-static tensor math, so this is CUDA-graph capture safe.
        """
        if not self.qsa_compressed_fp8_k_buffer_pool:
            return
        shadow = self.qsa_compressed_fp8_k_buffer_pool[
            self._transfer_full_attention_id(layer_id)
        ]
        buffer = self.get_qsa_compressed_k_buffer(layer_id)
        rows = buffer.reshape(buffer.shape[0], -1)
        dim = rows.shape[1]
        page_size = self.qsa_compressed_page_size
        loc = loc.long().reshape(-1).clamp(0, max(rows.shape[0] - 1, 0))
        if loc.numel() == 0:
            return
        vals = rows[loc].float()
        amax = vals.abs().amax(dim=-1, keepdim=True)
        # Zero rows keep a unit scale (their fp8 payload is 0, so the
        # kernel's `reduced * scale` still yields a 0 logit).
        scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
        k8 = (vals / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(
            torch.uint8
        )
        s4 = scale.contiguous().view(torch.uint8)

        pages = self.qsa_compressed_fp8_pages
        # Redirect inert dump rows (write slot 0) to the scratch page; row 0
        # of the compressed pool is never allocated to a request, so nothing
        # reads it back through the page table. Unique destinations keep the
        # vectorized store VMFault-free on eager HIP.
        dest = torch.where(loc == 0, torch.full_like(loc, pages * page_size), loc)
        page = dest // page_size
        offset = dest % page_size
        k_view = shadow[:, : page_size * dim].view(pages + 1, page_size, dim)
        s_view = shadow[:, page_size * dim :].view(pages + 1, page_size, 4)
        k_view[page, offset] = k8
        s_view[page, offset] = s4

    def set_qsa_compressed_k_buffer(
        self, layer_id: int, loc: torch.Tensor, compressed_k: torch.Tensor
    ) -> None:
        buffer = self.get_qsa_compressed_k_buffer(layer_id)
        _scatter_rows(buffer, loc, compressed_k)
        self.update_qsa_compressed_fp8_k_buffer(layer_id, loc)

    def get_kv_size_bytes(self):
        k_size, v_size = super().get_kv_size_bytes()
        qsa_k_size = (
            sum(
                tensor.numel() * tensor.element_size()
                for tensor in self.qsa_key_state_buffer_pool
            )
            + sum(
                tensor.numel() * tensor.element_size()
                for tensor in self.qsa_compressed_k_buffer_pool
            )
            + sum(
                tensor.numel() for tensor in self.qsa_compressed_fp8_k_buffer_pool
            )
            + self.qsa_rope_position_buffer.numel() * 8
        )
        return k_size + qsa_k_size, v_size


class QwenDSATokenToKVPool(HybridLinearKVPool):
    """Hybrid KV pool carrying the per-token index-K cache of tokenwise QSA.

    Only the BF16 reference layout is ported: one flat
    ``[size + page_size, index_kv_heads, index_head_dim]`` buffer per
    full-attention (DSA) layer, addressed by raw KV slots.  The FP8
    deep_gemm layout is intentionally not ported yet and the caller-side
    fast paths fail loudly instead of silently degrading.
    """

    index_state_dtype = torch.bfloat16

    @classmethod
    def qsa_bytes_per_token(
        cls, *, kv_heads: int, head_dim: int, num_layers: int
    ) -> int:
        return (
            _index_k_bytes(
                kv_heads=kv_heads, head_dim=head_dim, dtype=cls.index_state_dtype
            )
            * num_layers
        )

    def __init__(
        self,
        *,
        size: int,
        dtype: torch.dtype,
        page_size: int,
        head_num: int,
        head_dim: int,
        full_attention_layer_ids: List[int],
        device: str,
        mamba_pool: MambaPool,
        qsa_index_kv_heads: int,
        qsa_index_head_dim: int,
        qsa_token_budget: int,
        enable_memory_saver: bool = False,
        enable_kv_cache_copy: bool = False,
        start_layer: Optional[int] = None,
        full_kv_pool_class: Optional[type] = None,
        quant_method=None,
        post_capture_active: bool = False,
    ):
        if page_size != 64:
            raise ValueError(
                "tokenwise QSA requires KV-cache page_size 64 for its paged "
                f"indexer buffer, got {page_size}"
            )
        self.dsa_index_k_buffer_pool = []
        super().__init__(
            size=size,
            dtype=dtype,
            page_size=page_size,
            head_num=head_num,
            head_dim=head_dim,
            full_attention_layer_ids=full_attention_layer_ids,
            device=device,
            mamba_pool=mamba_pool,
            enable_memory_saver=enable_memory_saver,
            enable_kv_cache_copy=enable_kv_cache_copy,
            use_mla=False,
            start_layer=start_layer,
            full_kv_pool_class=full_kv_pool_class,
            quant_method=quant_method,
            post_capture_active=post_capture_active,
        )
        if qsa_index_kv_heads != 1:
            raise ValueError(
                f"tokenwise QSA requires index_kv_heads = 1 (MQA), got "
                f"{qsa_index_kv_heads}"
            )
        if min(qsa_index_kv_heads, qsa_index_head_dim, qsa_token_budget) <= 0:
            raise ValueError("QSA cache configuration values must be positive")
        self.qsa_compress_ratio = 1
        self.qsa_index_kv_heads = int(qsa_index_kv_heads)
        self.qsa_index_head_dim = int(qsa_index_head_dim)
        self.qsa_token_topk = int(qsa_token_budget)
        self.qsa_block_topk = int(qsa_token_budget)
        state_size = size + page_size
        self.dsa_index_k_buffer_pool = [
            torch.zeros(
                (state_size, self.qsa_index_kv_heads, self.qsa_index_head_dim),
                dtype=self.index_state_dtype,
                device=device,
            )
            for _ in full_attention_layer_ids
        ]
        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB

    def set_dsa_index_k_buffer(
        self, layer_id: int, loc: torch.Tensor, index_k: torch.Tensor
    ) -> None:
        buffer = self.get_dsa_index_k_buffer(layer_id)
        buffer[loc.long()] = index_k.to(buffer.dtype)

    def get_dsa_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        return self.dsa_index_k_buffer_pool[self._transfer_full_attention_id(layer_id)]

    def get_kv_size_bytes(self):
        k_size, v_size = super().get_kv_size_bytes()
        dsa_k_size = sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.dsa_index_k_buffer_pool
        )
        return k_size + dsa_k_size, v_size
