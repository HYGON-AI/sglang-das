from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import IntEnum, auto
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, TypeAlias

import torch

from sglang.kernels.ops.attention.utils import (
    concat_mla_absorb_q_general,
    mla_quantize_and_rope_for_fp8,
    seqlens_expand_triton,
)
from sglang.srt.configs.model_config import (
    get_dsa_index_kpool,
    get_dsa_index_topk,
    is_deepseek_dsa,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.glm5_next.dequant_k_cache import (
    dequantize_k_cache_paged,
)
from sglang.srt.layers.attention.glm5_next.indexer import BaseIndexerMetadata
from sglang.srt.layers.attention.glm5_next.kpool.planner import (
    KPoolExtendPlan,
    KPoolWritePlan,
)
from sglang.srt.layers.attention.glm5_next.kpool.planner import (
    init_kpool_extend_metadata as _init_kpool_extend_metadata_impl,
)
from sglang.srt.layers.attention.glm5_next.kpool.planner import (
    init_kpool_write_plan as _init_kpool_write_plan_impl,
)
from sglang.srt.layers.attention.glm5_next.kpool.planner import (
    init_kpool_write_plan_capture as _init_kpool_write_plan_capture_impl,
)
from sglang.srt.layers.attention.glm5_next.kpool.planner import (
    init_pooled_paged_mqa_metadata as _init_pooled_paged_mqa_metadata_impl,
)
from sglang.srt.layers.attention.glm5_next.kpool.planner import (
    update_kpool_write_plan as _update_kpool_write_plan_impl,
)
from sglang.srt.layers.attention.glm5_next.kpool.planner import (
    update_kpool_write_plan_multi_decode as _update_kpool_write_plan_multi_decode_impl,
)
from sglang.srt.layers.attention.glm5_next.kpool.planner import (
    update_pooled_paged_mqa_metadata as _update_pooled_paged_mqa_metadata_impl,
)
from sglang.srt.layers.attention.glm5_next.mtp_precompute import (
    NativeSparseAttnBackendMTPPrecomputeMixin,
    PrecomputedMetadata,
    compute_cu_seqlens,
    fill_decode_page_table_gpu,
)
from sglang.srt.layers.attention.glm5_next.runtime import (
    get_glm5_next_runtime_args as get_global_server_args,
)
from sglang.srt.layers.attention.glm5_next.transform_index import (
    transform_index_page_table_decode,
    transform_index_page_table_prefill,
)
from sglang.srt.layers.attention.glm5_next.utils import (
    can_dsa_prefill_cp_round_robin_split,
    compute_dsa_seqlens,
    copy_cpu_values_to_device,
    dsa_cp_round_robin_split_data,
    dsa_cp_round_robin_split_q_seqs,
    dsa_prefill_has_history,
    dsa_use_prefill_cp,
    effective_forward_mode,
    is_dsa_enable_prefill_cp,
    is_dsa_prefill_cp_in_seq_split,
    pad_dsa_cache_seqlens,
    should_use_dsa_fused_topk,
    uses_logical_mtp_topk_indices,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_cuda, is_hcu, is_hip
from sglang.srt.utils.common import print_warning_once

logger = logging.getLogger(__name__)


def _disable_dsa_multi_replay_opt() -> bool:
    return os.getenv("SGLANG_DISABLE_DSA_MULTI_REPLAY_OPT", "0") == "1"


def _kpool_topk_sort_writeback_enabled() -> bool:
    value = os.getenv("SGLANG_KERNEL_KPOOL_TOPK_SORT_WRITEBACK")
    if value is None:
        value = os.getenv("SGL_KERNEL_KPOOL_TOPK_SORT_WRITEBACK")
    return value is not None and value != "" and value[0] != "0"


if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

_is_hip = is_hip()
_is_hcu = is_hcu()

if _is_hip:
    from sglang.srt.layers.attention.glm5_next.triton_kernel import get_valid_kv_indices


if _is_hip and not _is_hcu:
    try:
        from aiter import (  # noqa: F401
            flash_attn_varlen_func,
            mha_batch_prefill_func,
            paged_attention_ragged,
        )
        from aiter.mla import mla_decode_fwd, mla_prefill_fwd  # noqa: F401
    except ImportError:
        logger.warning(
            "aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device."
        )
from sglang.srt.layers.attention.flashattention_interface import flash_attn_with_kvcache


def _to_2d_context_lens(seqlens_32: torch.Tensor, batch_size: int) -> torch.Tensor:
    # Always normalize to (N_total, 1) layout, to avoid deadlock at deep_gemm.fp8_paged_mqa_logits
    if seqlens_32.dim() == 2:
        if seqlens_32.size(1) == 1:
            return seqlens_32
        # Fall through and re-flatten if the caller already gave us a (bs, next_n)
        # view ?we want (N_total, 1) regardless.
        seqlens_32 = seqlens_32.reshape(-1)
    return seqlens_32.contiguous().view(-1, 1)


@lru_cache(maxsize=1)
def _get_deep_gemm():
    try:
        import deep_gemm

        return deep_gemm
    except Exception:
        return None


# Reuse this workspace buffer across all DSA backend instances
global_workspace_buffer = None

# Control whether to use fused metadata copy kernel for cuda graph replay (default: enabled)
# Set SGLANG_USE_FUSED_METADATA_COPY=0 or false to disable
_USE_FUSED_METADATA_COPY = envs.SGLANG_USE_FUSED_METADATA_COPY.get() and not _is_hip


@dataclass(frozen=True)
class DSAFlashMLAMetadata:
    """Metadata only needed by FlashMLA"""

    flashmla_metadata: torch.Tensor
    num_splits: torch.Tensor

    def slice(self, sli):
        if self.num_splits:
            return DSAFlashMLAMetadata(
                flashmla_metadata=self.flashmla_metadata,
                num_splits=self.num_splits[sli],
            )
        else:
            return DSAFlashMLAMetadata(
                flashmla_metadata=self.flashmla_metadata,
                num_splits=self.num_splits,
            )

    def copy_(self, other: DSAFlashMLAMetadata):
        if _is_hcu:
            flashmla_metadata = other.flashmla_metadata
            if hasattr(flashmla_metadata, "flashmla_metadata"):
                flashmla_metadata = flashmla_metadata.flashmla_metadata
            if flashmla_metadata is None:
                object.__setattr__(self, "flashmla_metadata", None)
            elif hasattr(self.flashmla_metadata, "copy_"):
                self.flashmla_metadata.copy_(flashmla_metadata)
            else:
                object.__setattr__(self, "flashmla_metadata", flashmla_metadata)

            if other.num_splits is None:
                object.__setattr__(self, "num_splits", None)
            elif hasattr(self.num_splits, "copy_"):
                self.num_splits.copy_(other.num_splits)
            else:
                object.__setattr__(self, "num_splits", other.num_splits)
        else:
            self.flashmla_metadata.copy_(other.flashmla_metadata)
            self.num_splits.copy_(other.num_splits)


def _can_fuse_flashmla_metadata(
    *metadatas: Optional[DSAFlashMLAMetadata],
) -> bool:
    return all(
        metadata is not None
        and isinstance(metadata.flashmla_metadata, torch.Tensor)
        and isinstance(metadata.num_splits, torch.Tensor)
        for metadata in metadatas
    )


@dataclass(frozen=True)
class DSAMetadata:
    page_size: int

    # Sequence lengths for the forward batch
    cache_seqlens_int32: torch.Tensor
    # Maximum sequence length for query
    max_seq_len_q: int
    # Maximum sequence length for key
    max_seq_len_k: int
    # Cumulative sequence lengths for query
    cu_seqlens_q: torch.Tensor
    # Cumulative sequence lengths for key
    cu_seqlens_k: torch.Tensor
    # Page table, the index of KV Cache Tables/Blocks
    # this table is always with page_size = 1
    page_table_1: torch.Tensor

    # NOTE(dark): This will property be used in:
    # 1. dense decode/prefill, we use paged flash attention, need real_page_table
    # 2. sparse decode/prefill, indexer need real_page_table to compute the score
    real_page_table: torch.Tensor

    # DSA metadata (dsa prefill are expanded)
    dsa_cache_seqlens_int32: torch.Tensor  # this seqlens is clipped to `topk`
    dsa_cu_seqlens_q: torch.Tensor  # must be arange(0, len(dsa_cu_seqlens_k))
    dsa_cu_seqlens_k: torch.Tensor  # cumsum of `dsa_cache_seqlens_int32`
    dsa_extend_seq_lens_list: List[int]
    dsa_seqlens_expanded: torch.Tensor  # expanded, unclipped `seqlens`
    dsa_max_seqlen_q: Literal[1] = 1  # always 1 for decode, variable for extend

    flashmla_metadata: Optional[DSAFlashMLAMetadata] = None
    # DeepGEMM schedule metadata for paged MQA logits (decode/target_verify/draft_extend only).
    # Precomputed once per forward batch and reused across layers.
    paged_mqa_schedule_metadata: Optional[torch.Tensor] = None
    pooled_paged_mqa_schedule_metadata: Optional[torch.Tensor] = None
    pooled_cache_seqlens_int32: Optional[torch.Tensor] = None
    pooled_index_kpool: int = 1
    kpool_extend_plan: Optional[KPoolExtendPlan] = None
    kpool_write_plan: Optional[KPoolWritePlan] = None
    # The sum of sequence lengths for key, prefill only
    seq_lens_sum: Optional[int] = None
    # The flattened 1D page table with shape (seq_lens_sum,), prefill only
    # this table is always with page_size = 1
    page_table_1_flattened: Optional[torch.Tensor] = None
    # The offset of topk indices in ragged kv, prefill only
    # shape: (seq_lens_sum,)
    topk_indices_offset: Optional[torch.Tensor] = None

    # k_start and k_end in kv cache for each token.
    indexer_k_start_end: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    # seq lens for each batch.
    indexer_seq_lens_cpu: Optional[torch.Tensor] = None
    # seq lens for each batch.
    indexer_seq_lens: Optional[torch.Tensor] = None
    # batch index for each token.
    token_to_batch_idx: Optional[torch.Tensor] = None


@dataclass
class _KPoolForwardInputs:
    """Inputs that are only needed by the kpool metadata dispatcher."""

    full_real_page_table: Optional[torch.Tensor] = None
    full_seqlens_expanded: Optional[torch.Tensor] = None
    cp_overrides: dict = field(default_factory=dict)


class TopkTransformMethod(IntEnum):
    # Transform topk indices to indices to the page table (page_size = 1)
    PAGED = auto()
    # Transform topk indices to indices to ragged kv (non-paged)
    RAGGED = auto()


@torch.compile
def _compiled_cat(tensors: list[torch.Tensor], dim: int = -1) -> torch.Tensor:
    return torch.cat(tensors, dim=dim)


def _cat(tensors: list[torch.Tensor], dim: int = -1) -> torch.Tensor:
    """
    Concatenate two tensors along the last dimension.
    Use this function to concatenate q_nope and q_rope or k_nope and k_rope.
    """
    assert len(tensors) == 2

    qk_nope, qk_rope = tensors
    assert qk_nope.ndim == 3 and qk_rope.ndim == 3

    torch._dynamo.mark_dynamic(qk_nope, 0)
    torch._dynamo.mark_dynamic(qk_rope, 0)

    return _compiled_cat([qk_nope, qk_rope], dim=dim)


@dataclass(frozen=True)
class DSAIndexerMetadata(BaseIndexerMetadata):
    attn_metadata: DSAMetadata
    topk_transform_method: TopkTransformMethod
    paged_mqa_schedule_metadata: Optional[torch.Tensor] = None
    force_unfused_topk: bool = False

    def get_seqlens_int32(self) -> torch.Tensor:
        return self.attn_metadata.cache_seqlens_int32

    def get_page_table_64(self) -> torch.Tensor:
        return self.attn_metadata.real_page_table

    def get_page_table_1(self) -> torch.Tensor:
        return self.attn_metadata.page_table_1

    def get_seqlens_expanded(self) -> torch.Tensor:
        return self.attn_metadata.dsa_seqlens_expanded

    def get_cu_seqlens_k(self) -> torch.Tensor:
        return self.attn_metadata.cu_seqlens_k

    def get_indexer_kvcache_range(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.attn_metadata.indexer_k_start_end

    def get_indexer_seq_len(self) -> torch.Tensor:
        return self.attn_metadata.indexer_seq_lens

    def get_indexer_seq_len_cpu(self) -> torch.Tensor:
        return self.attn_metadata.indexer_seq_lens_cpu

    def get_dsa_extend_len_cpu(self) -> List[int]:
        return self.attn_metadata.dsa_extend_seq_lens_list

    def get_token_to_batch_idx(self) -> torch.Tensor:
        return self.attn_metadata.token_to_batch_idx

    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        ks: Optional[torch.Tensor] = None,
        cu_seqlens_q: torch.Tensor = None,
        ke_offset: torch.Tensor = None,
        batch_idx_list: List[int] = None,
        topk_indices_offset_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not _is_hcu:
            from sgl_kernel import (
                fast_topk_transform_fused,
                fast_topk_transform_ragged_fused,
                fast_topk_v2,
            )
        else:
            from lightop import (
                fast_topk_transform_fused,
                fast_topk_transform_ragged_fused,
            )
            from sgl_kernel import fast_topk_v2

        if topk_indices_offset_override is not None:
            cu_topk_indices_offset = topk_indices_offset_override
            cu_seqlens_q_topk = None
        elif cu_seqlens_q is not None:
            cu_seqlens_q = cu_seqlens_q.to(torch.int32)
            cu_seqlens_q_topk = compute_cu_seqlens(cu_seqlens_q)

            cu_topk_indices_offset = torch.repeat_interleave(
                cu_seqlens_q_topk[:-1],
                cu_seqlens_q,
                output_size=logits.shape[0],
            )
        else:
            cu_seqlens_q_topk = self.attn_metadata.cu_seqlens_q
            cu_topk_indices_offset = self.attn_metadata.topk_indices_offset
        if ke_offset is not None:
            seq_lens_topk = ke_offset
        else:
            seq_lens_topk = self.get_seqlens_expanded()
        if batch_idx_list is not None:
            page_table_size_1 = self.attn_metadata.page_table_1[batch_idx_list]
        else:
            page_table_size_1 = self.attn_metadata.page_table_1
        if (
            self.topk_transform_method == TopkTransformMethod.PAGED
            and page_table_size_1.shape[1] < logits.shape[1]
        ):
            seq_lens_topk = torch.clamp(seq_lens_topk, max=page_table_size_1.shape[1])

        if not envs.SGLANG_DSA_FUSE_TOPK.get() or self.force_unfused_topk:
            return fast_topk_v2(logits, seq_lens_topk, topk, row_starts=ks)
        elif self.topk_transform_method == TopkTransformMethod.PAGED:
            # NOTE(dark): if fused, we return a transformed page table directly
            return fast_topk_transform_fused(
                score=logits,
                lengths=seq_lens_topk,
                page_table_size_1=page_table_size_1,
                cu_seqlens_q=cu_seqlens_q_topk,
                topk=topk,
                row_starts=ks,
            )
        elif self.topk_transform_method == TopkTransformMethod.RAGGED:
            if cu_topk_indices_offset is None:
                raise RuntimeError(
                    "RAGGED topk_transform requires topk_indices_offset; "
                    "expected extend-without-speculative metadata."
                )
            return fast_topk_transform_ragged_fused(
                score=logits,
                lengths=seq_lens_topk,
                topk_indices_offset=cu_topk_indices_offset,
                topk=topk,
                row_starts=ks,
            )
        else:
            assert False, f"Unsupported {self.topk_transform_method = }"


_DSA_IMPL_T: TypeAlias = Literal[
    "flashmla_sparse", "flashmla_kv", "fa3", "tilelang", "trtllm"
]


class NativeSparseAttnBackend(
    NativeSparseAttnBackendMTPPrecomputeMixin, AttentionBackend
):
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
        seed_dsa_topk_from_draft_extend: bool = False,
    ):
        super().__init__()
        self.model_runner = model_runner
        self.hisparse_coordinator = model_runner.hisparse_coordinator
        self.forward_metadata: DSAMetadata
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.device = model_runner.device
        assert isinstance(model_runner.page_size, int)
        self.real_page_size = model_runner.page_size
        self.num_splits = (
            1 if get_global_server_args().enable_deterministic_inference else 0
        )
        self.use_dsa = is_deepseek_dsa(model_runner.model_config.hf_config)
        assert self.use_dsa, "DSA backend only supports DeepSeek DSA"
        self.dsa_kv_cache_store_fp8 = (
            model_runner.token_to_kv_pool.dsa_kv_cache_store_fp8
        )
        self.dsa_index_topk = get_dsa_index_topk(model_runner.model_config.hf_config)
        self.dsa_index_kpool = get_dsa_index_kpool(model_runner.model_config.hf_config)
        self.max_context_len = model_runner.model_config.context_len
        self.num_q_heads = (
            model_runner.model_config.num_attention_heads // get_parallel().attn_tp_size
        )
        self.kv_cache_dim = model_runner.token_to_kv_pool.kv_cache_dim
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        from sglang.srt.layers.attention.glm5_next import is_glm5_next_hcu

        self._is_glm5_next = is_glm5_next_hcu(model_runner.model_config.hf_config)

        assert model_runner.req_to_token_pool is not None
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        self.use_mha: bool = False
        self.dsa_prefill_impl: _DSA_IMPL_T = (
            get_global_server_args().dsa_prefill_backend
        )
        self.dsa_decode_impl: _DSA_IMPL_T = get_global_server_args().dsa_decode_backend
        if self.num_q_heads <= 64:
            self.flashmla_kv_num_q_heads = 64
        elif self.num_q_heads <= 128:
            self.flashmla_kv_num_q_heads = 128
        else:
            # Keep original head count if it exceeds current padded variants.
            self.flashmla_kv_num_q_heads = self.num_q_heads
        self.enable_auto_select_prefill_impl = self.dsa_prefill_impl == "flashmla_auto"

        self._lightop_decode_gather = None
        self._lightop_decode_gather_workspace = None
        self._lightop_decode_compact_indices = None
        self._lightop_decode_graph_workspaces = None
        # The packed physical row is 528 bytes for no-RoPE and 656 bytes
        # for RoPE: 512 FP8 latent values plus four FP32 scales, with an
        # extra 64 BF16 RoPE values only when qk_rope_head_dim is 64.
        # The LightOp destination keeps only the logical BF16 dimensions
        # needed by FlashMLA.
        self._lightop_decode_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        # KPool keeps ``topk`` selected history tokens and appends up to
        # ``kpool - 1`` uncompressed tail tokens.  HCU sparse FlashMLA pads
        # that logical width to a 64-token block with -1 sentinels.  The
        # gathered BF16 cache and its compact indices must use the exact same
        # width so the sparse softmax mask remains unchanged.
        self._lightop_decode_gather_logical_width = self.dsa_index_topk
        if self.dsa_index_kpool > 1:
            self._lightop_decode_gather_logical_width += self.dsa_index_kpool - 1
        self._lightop_decode_gather_width = self._lightop_decode_gather_logical_width
        if self.dsa_index_kpool > 1:
            topk_block_size = 64
            self._lightop_decode_gather_width = (
                (self._lightop_decode_gather_width + topk_block_size - 1)
                // topk_block_size
                * topk_block_size
            )
        lightop_decode_requested = envs.SGLANG_DSA_HCU_USE_LIGHTOP_DECODE_GATHER.get()
        lightop_decode_static_compatible = (
            _is_hcu
            and self._is_glm5_next
            and self.dsa_decode_impl == "flashmla_kv"
            and self.dsa_kv_cache_store_fp8
            and self.qk_rope_head_dim in (0, 64)
            and self.kv_lora_rank == 512
            and (
                self.kv_cache_dim == 656
                or (self.qk_rope_head_dim == 0 and self.kv_cache_dim == 528)
            )
            and self.dsa_index_topk == 2048
            and self.dsa_index_kpool in (1, 4, 16)
            and self.real_page_size == 64
        )
        if lightop_decode_requested and lightop_decode_static_compatible:
            try:
                from lightop import op as lightop_op

                self._lightop_decode_gather = getattr(
                    lightop_op,
                    "decode_gather_and_up_convert_with_indices",
                    None,
                )
            except (ImportError, AttributeError):
                self._lightop_decode_gather = None
            if self._lightop_decode_gather is None:
                logger.warning(
                    "LightOp decode gather was requested, but the installed "
                    "LightOp package does not expose "
                    "decode_gather_and_up_convert_with_indices"
                )
        elif lightop_decode_requested:
            logger.warning(
                "LightOp decode gather was requested but this DSA backend is "
                "not the supported GLM5-Next HCU FP8 FlashMLA configuration"
            )

        self._arange_buf = torch.arange(16384, device=self.device, dtype=torch.int32)

        if _is_hip:
            max_bs = model_runner.req_to_token_pool.size

            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )

            self.kv_indices = torch.zeros(
                max_bs * self.dsa_index_topk,
                dtype=torch.int32,
                device=self.device,
            )
            # Aiter mla_decode_fwd supports num_heads multiples of 16 in range [16, 128].
            # For models with fewer heads per GPU (e.g. GLM-5 64 heads / TP8 = 8), need to pad the heads to 16.
            self.need_pad_heads = self.num_q_heads < 16
            self.head_repeat_factor = (
                16 // self.num_q_heads if self.num_q_heads < 16 else 1
            )

        # Speculative decoding
        self.topk = get_global_server_args().speculative_eagle_topk or 0
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = (
            get_global_server_args().speculative_num_draft_tokens
        )
        self.speculative_step_id = speculative_step_id
        self.use_fused_topk = should_use_dsa_fused_topk(
            get_global_server_args(),
            seed_dsa_topk_from_draft_extend,
            model_runner.model_config,
        )
        self.max_graph_page_table_len = self.req_to_token.shape[1]
        self._real_page_col_indices = torch.arange(
            0,
            self.max_graph_page_table_len,
            self.real_page_size,
            device=self.device,
            dtype=torch.long,
        )

        self.device_capability = torch.cuda.get_device_capability()
        self.device_sm_major = self.device_capability[0]
        self.kv_cache_dtype = model_runner.kv_cache_dtype

        # Allocate global workspace buffer for TRT-LLM kernels (ragged attention on SM100/B200, or trtllm decode)
        if self.device_sm_major >= 10 or self.dsa_decode_impl == "trtllm":
            global global_workspace_buffer
            if global_workspace_buffer is None:
                global_workspace_buffer = torch.empty(
                    envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                    dtype=torch.uint8,
                    device=model_runner.device,
                )
            self.workspace_buffer = global_workspace_buffer
        else:
            self.workspace_buffer = None

    def get_device_int32_arange(self, l: int) -> torch.Tensor:
        if l > len(self._arange_buf):
            next_pow_of_2 = 1 << (l - 1).bit_length()
            self._arange_buf = torch.arange(
                next_pow_of_2, device=self.device, dtype=torch.int32
            )
        return self._arange_buf[:l]

    def _allocate_lightop_decode_workspaces(self, capacity: int) -> None:
        """Allocate contiguous buffers for BF16 sparse decode."""
        if self._lightop_decode_gather is None or capacity <= 0:
            return
        current_capacity = (
            0
            if self._lightop_decode_gather_workspace is None
            else self._lightop_decode_gather_workspace.shape[0]
        )
        if current_capacity >= capacity:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "LightOp decode gather workspace must be allocated before "
                "CUDA graph capture"
            )

        self._lightop_decode_gather_workspace = torch.empty(
            (
                capacity,
                self._lightop_decode_gather_width,
                self._lightop_decode_head_dim,
            ),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self._lightop_decode_compact_indices = torch.empty(
            (capacity, self._lightop_decode_gather_width),
            dtype=torch.int32,
            device=self.device,
        )

    def _get_lightop_decode_workspaces(
        self, num_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            torch.cuda.is_current_stream_capturing()
            and self._lightop_decode_graph_workspaces is not None
        ):
            workspace, compact_indices = self._lightop_decode_graph_workspaces
            if workspace.shape[0] < num_tokens:
                raise RuntimeError(
                    "LightOp decode gather exceeded its CUDA graph capacity"
                )
        else:
            self._allocate_lightop_decode_workspaces(num_tokens)
            workspace = self._lightop_decode_gather_workspace
            compact_indices = self._lightop_decode_compact_indices
        if workspace is None or compact_indices is None:
            raise RuntimeError("LightOp decode gather workspaces are unavailable")
        return (
            workspace[:num_tokens],
            compact_indices[:num_tokens],
        )

    def get_real_page_col_indices(self, max_seqlen_k: int) -> torch.Tensor:
        page_size = self.real_page_size
        num_cols = (max_seqlen_k + page_size - 1) // page_size
        if num_cols > len(self._real_page_col_indices):
            next_pow_of_2 = 1 << (max_seqlen_k - 1).bit_length()
            self._real_page_col_indices = torch.arange(
                0, next_pow_of_2, page_size, device=self.device, dtype=torch.long
            )
        return self._real_page_col_indices[:num_cols]

    def _transform_table_1_to_real(self, page_table: torch.Tensor) -> torch.Tensor:
        page_size = self.real_page_size
        if page_size == 1:
            return page_table
        # Keep the old gather-then-divide memory pattern, but cache the column
        # index as int64 so PyTorch does not insert an int32->int64 _to_copy.
        col_indices = self.get_real_page_col_indices(page_table.shape[1])
        return torch.index_select(page_table, 1, col_indices) // page_size

    # ---- Kpool metadata: delegated to dsa.kpool.planner ------------
    def _init_pooled_paged_mqa_metadata(
        self,
        metadata: DSAMetadata,
        seqlens_32: torch.Tensor,
        forward_mode: ForwardMode,
    ) -> DSAMetadata:
        return _init_pooled_paged_mqa_metadata_impl(
            metadata,
            seqlens_32,
            forward_mode,
            pool_size=self.dsa_index_kpool,
            real_page_size=self.real_page_size,
        )

    def _update_pooled_paged_mqa_metadata(
        self,
        metadata: DSAMetadata,
        seqlens_32: torch.Tensor,
        forward_mode: ForwardMode,
    ) -> None:
        _update_pooled_paged_mqa_metadata_impl(
            metadata,
            seqlens_32,
            forward_mode,
            pool_size=self.dsa_index_kpool,
            real_page_size=self.real_page_size,
        )

    def _build_kpool_metadata(
        self,
        metadata: DSAMetadata,
        forward_batch: ForwardBatch,
        topk_transform_method: TopkTransformMethod,
        kpool_inputs: _KPoolForwardInputs,
        cache_seqlens_int32: torch.Tensor,
        seqlens_expanded: torch.Tensor,
    ) -> DSAMetadata:
        mode = forward_batch.forward_mode
        if (is_cuda() or is_hcu()) and mode.is_decode_or_idle():
            metadata = self._init_pooled_paged_mqa_metadata(
                metadata=metadata,
                seqlens_32=cache_seqlens_int32,
                forward_mode=mode,
            )
            metadata = _init_kpool_write_plan_impl(
                metadata,
                forward_batch,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                real_page_table=metadata.real_page_table,
                num_draft_tokens=1,
                write_start=(forward_batch.seq_lens - 1).to(torch.int32),
            )
            return metadata
        if mode.is_extend_without_speculative():
            return _init_kpool_extend_metadata_impl(
                metadata,
                forward_batch,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                topk_transform_method=topk_transform_method,
                full_real_page_table=kpool_inputs.full_real_page_table,
                full_seqlens_expanded=kpool_inputs.full_seqlens_expanded,
                **kpool_inputs.cp_overrides,
            )
        if mode.is_target_verify() or mode.is_draft_extend_v2():
            if mode.is_target_verify():
                write_start = forward_batch.seq_lens.to(torch.int32)
                accept_length = None
            else:
                write_start = (
                    forward_batch.seq_lens - self.speculative_num_draft_tokens
                ).to(torch.int32)
                spec_info = forward_batch.spec_info
                # Reference project reads `spec_info.accept_length` which in
                # v2 already includes the bonus token. This project renamed it
                # per speculative-naming.md Rule 3: `num_accept_tokens` is the
                # bonus-inclusive count (= reference's accept_length in v2).
                # Missing this rename left accept_length=None every step, so
                # `plan.effective_n_per_batch` stayed at 0 and the fused
                # write-then-compress kernel never crossed a pool boundary,
                # leaving v2's compressed K out of the FP8 cache entirely.
                accept_length = (
                    spec_info.num_accept_tokens
                    if spec_info is not None
                    and getattr(spec_info, "num_accept_tokens", None) is not None
                    else None
                )
            return _init_kpool_write_plan_impl(
                metadata,
                forward_batch,
                pool_size=self.dsa_index_kpool,
                real_page_size=self.real_page_size,
                real_page_table=metadata.real_page_table,
                num_draft_tokens=self.speculative_num_draft_tokens,
                write_start=write_start,
                accept_length=accept_length,
            )
        return metadata

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        # A prefill worker can still see idle or non-CP forwards. Do not let a
        # compact scratch layout from the preceding CP batch leak into a path
        # whose page tables use physical locations.
        if not dsa_use_prefill_cp(forward_batch):
            configure_page_plan = getattr(
                self.token_to_kv_pool,
                "configure_main_kv_page_plan",
                None,
            )
            if configure_page_plan is not None:
                configure_page_plan(None, forward_batch)

        batch_size = forward_batch.batch_size
        device = forward_batch.seq_lens.device

        if forward_batch.forward_mode.is_target_verify():
            draft_token_num = self.speculative_num_draft_tokens
        else:
            draft_token_num = 0

        cache_seqlens_int32 = (
            forward_batch.seq_lens.to(torch.int32)
            if draft_token_num == 0
            else (forward_batch.seq_lens + draft_token_num).to(torch.int32)
        )
        cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)
        assert forward_batch.seq_lens_cpu is not None
        max_seqlen_k = int(forward_batch.seq_lens_cpu.max().item() + draft_token_num)
        # [b, max_seqlen_k]
        page_table = self.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :max_seqlen_k
        ]

        page_table_1_flattened = None
        topk_indices_offset = None

        # Centralized dispatch: decide all strategies for this batch
        self.set_dsa_prefill_impl(forward_batch)
        dsa_impl_for_batch = (
            self.dsa_decode_impl
            if (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend_v2()
            )
            else self.dsa_prefill_impl
        )
        use_flashmla_kv = (not self.use_mha) and dsa_impl_for_batch == "flashmla_kv"
        topk_transform_method = self.get_topk_transform_method(
            forward_batch.forward_mode
        )
        use_kpool = self.dsa_index_kpool > 1
        kpool_inputs = _KPoolForwardInputs()
        # Batch indices selected when cp enabled: After splitting multiple sequences,
        # a certain cp rank may not have some of these sequences.
        # We use bs_idx_cpu to mark which sequences are finally selected by the current cp rank,
        # a default value of None indicates that all sequences are selected.
        bs_idx_cpu = None
        # seq_len_cpu of selected sequences
        indexer_seq_lens_cpu = forward_batch.seq_lens_cpu
        indexer_seq_lens = forward_batch.seq_lens

        if forward_batch.forward_mode.is_decode_or_idle():
            extend_seq_lens_cpu = [1] * batch_size
            max_seqlen_q = 1
            cu_seqlens_q = self.get_device_int32_arange(batch_size + 1)
            seqlens_expanded = cache_seqlens_int32
        elif forward_batch.forward_mode.is_target_verify():
            max_seqlen_q = 1
            cu_seqlens_q = torch.arange(
                0,
                batch_size * self.speculative_num_draft_tokens + 1,
                1,
                dtype=torch.int32,
                device=device,
            )
            extend_seq_lens_cpu = [self.speculative_num_draft_tokens] * batch_size
            forward_batch.extend_seq_lens_cpu = extend_seq_lens_cpu

            seqlens_expanded = seqlens_expand_triton(
                copy_cpu_values_to_device(
                    extend_seq_lens_cpu, device, dtype=torch.int32
                ),
                cache_seqlens_int32,
                self.speculative_num_draft_tokens * batch_size,
                self.speculative_num_draft_tokens,
            )

            page_table = torch.repeat_interleave(
                page_table,
                repeats=self.speculative_num_draft_tokens,
                dim=0,
                output_size=batch_size * self.speculative_num_draft_tokens,
            )
        elif forward_batch.forward_mode.is_draft_extend_v2():
            assert (
                forward_batch.extend_seq_lens_cpu is not None
                and forward_batch.extend_seq_lens is not None
                and forward_batch.extend_prefix_lens_cpu is not None
            ), "All of them must not be None"

            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            assert forward_batch.extend_seq_lens is not None

            max_seqlen_q = 1
            cu_seqlens_q = torch.arange(
                0,
                forward_batch.extend_num_tokens + 1,
                1,
                dtype=torch.int32,
                device=device,
            )

            seqlens_expanded = seqlens_expand_triton(
                forward_batch.extend_seq_lens,
                cache_seqlens_int32,
                sum(extend_seq_lens_cpu),
                self.speculative_num_draft_tokens,
            )
            # DRAFT_EXTEND_V2: V2 worker pre-fills draft KV cache with ALL speculated
            # tokens upfront. All requests extend by the same fixed
            # (speculative_num_draft_tokens). Use scalar to avoid GPU sync.
            page_table = torch.repeat_interleave(
                page_table,
                repeats=self.speculative_num_draft_tokens,
                dim=0,
                output_size=batch_size * self.speculative_num_draft_tokens,
            )

            if use_kpool:
                kpool_inputs.full_real_page_table = self._transform_table_1_to_real(
                    page_table
                )
                kpool_inputs.full_seqlens_expanded = seqlens_expanded
        elif forward_batch.forward_mode.is_extend():
            assert (
                forward_batch.extend_seq_lens_cpu is not None
                and forward_batch.extend_seq_lens is not None
                and forward_batch.extend_prefix_lens_cpu is not None
            ), "All of them must not be None"
            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            assert forward_batch.extend_seq_lens is not None
            extend_seq_lens = forward_batch.extend_seq_lens

            seqlens_expanded = torch.cat(
                [
                    torch.arange(
                        kv_len - qo_len + 1,
                        kv_len + 1,
                        dtype=torch.int32,
                        device=device,
                    )
                    for qo_len, kv_len in zip(
                        forward_batch.extend_seq_lens_cpu,
                        forward_batch.seq_lens_cpu.tolist(),
                        strict=True,
                    )
                ]
            )

            if use_kpool:
                kpool_inputs.full_real_page_table = self._transform_table_1_to_real(
                    page_table
                )
                kpool_inputs.full_seqlens_expanded = seqlens_expanded

            if can_dsa_prefill_cp_round_robin_split(forward_batch):
                seqlens_expanded = dsa_cp_round_robin_split_data(seqlens_expanded)
                extend_seq_lens_cpu, extend_seq_lens, bs_idx_cpu, bs_idx = (
                    dsa_cp_round_robin_split_q_seqs(
                        extend_seq_lens_cpu, extend_seq_lens
                    )
                )
                indexer_seq_lens_cpu = indexer_seq_lens_cpu[bs_idx_cpu]
                indexer_seq_lens = indexer_seq_lens[bs_idx]
                cache_seqlens_int32 = cache_seqlens_int32[bs_idx]
                cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)
                max_seqlen_k = (
                    int(indexer_seq_lens_cpu.max().item() + draft_token_num)
                    if len(indexer_seq_lens_cpu) != 0
                    else 0
                )
                page_table = page_table[bs_idx, :max_seqlen_k]
                if use_kpool:
                    kpool_inputs.cp_overrides.update(
                        local_real_page_table=self._transform_table_1_to_real(
                            page_table
                        ),
                        local_seqlens_expanded=seqlens_expanded,
                        local_extend_seq_lens_cpu=extend_seq_lens_cpu,
                        local_seq_lens_cpu=indexer_seq_lens_cpu.tolist(),
                        local_req_pool_indices=forward_batch.req_pool_indices[bs_idx],
                    )

            if any(forward_batch.extend_prefix_lens_cpu) or bs_idx_cpu is not None:
                max_seqlen_q = (
                    max(extend_seq_lens_cpu) if len(extend_seq_lens_cpu) != 0 else 1
                )
                cu_seqlens_q = compute_cu_seqlens(extend_seq_lens.to(torch.int32))
            else:
                max_seqlen_q = max_seqlen_k
                cu_seqlens_q = cu_seqlens_k

            # Check if MHA FP8 dequantization is needed
            mha_dequantize_needed = (
                self.use_mha and self.token_to_kv_pool.dtype == torch.float8_e4m3fn
            )
            forward_batch.using_mha_one_shot_fp8_dequant = mha_dequantize_needed

            # page_table_1_flattened is only used when prefix sharing is enabled:
            has_prefix_sharing = any(forward_batch.extend_prefix_lens_cpu)
            if has_prefix_sharing and (
                topk_transform_method == TopkTransformMethod.RAGGED
                or mha_dequantize_needed
            ):
                page_table_1_flattened = torch.cat(
                    [
                        page_table[i, :kv_len]
                        for i, kv_len in enumerate(
                            indexer_seq_lens_cpu.tolist(),
                        )
                    ]
                )
                assert page_table_1_flattened.shape[0] == sum(indexer_seq_lens_cpu), (
                    f"{page_table_1_flattened.shape[0] = } must be the same as {sum(indexer_seq_lens_cpu) = }"
                )

                # Validate indices when logical tokens exceed physical capacity
                # This is likely to be triggered by PP with high kv reuse & parallelism
                kv_cache_capacity = (
                    self.token_to_kv_pool.size + self.token_to_kv_pool.page_size
                )
                if forward_batch.seq_lens_sum > kv_cache_capacity:
                    max_idx = page_table_1_flattened.max().item()
                    assert max_idx < kv_cache_capacity, (
                        f"Invalid page table index: max={max_idx}, "
                        f"kv_cache_capacity={kv_cache_capacity}"
                    )

            if topk_transform_method == TopkTransformMethod.RAGGED:
                topk_indices_offset = torch.repeat_interleave(
                    cu_seqlens_k[:-1],
                    extend_seq_lens,
                    output_size=sum(extend_seq_lens_cpu),
                )
        else:
            assert False, f"Unsupported {forward_batch.forward_mode = }"

        indexer_k_start_end, token_to_batch_idx = self._cal_indexer_k_start_end(
            forward_batch, bs_idx_cpu
        )
        # 1D, expanded seqlens (1D means cheap to compute, so always compute it)
        dsa_cache_seqlens_int32 = compute_dsa_seqlens(
            original_seq_lens=seqlens_expanded,
            dsa_index_topk=self.dsa_index_topk,
            index_kpool=self.dsa_index_kpool,
        )
        if use_flashmla_kv and _is_hcu and self.dsa_index_kpool > 1:
            # KPool adds a tail width of kpool-1. HCU FlashMLA sparse decode
            # requires params.topk (derived from cache_seqlens/metadata) to be
            # aligned to its TOPK_BLOCK_SIZE. The corresponding indices tensor
            # is padded with -1 in _forward_flashmla_kv.
            topk_block_size = 64
            dsa_cache_seqlens_int32 = (
                torch.div(
                    dsa_cache_seqlens_int32 + topk_block_size - 1,
                    topk_block_size,
                    rounding_mode="floor",
                )
                * topk_block_size
            ).to(torch.int32)
        dsa_cache_seqlens_int32 = pad_dsa_cache_seqlens(
            forward_batch, dsa_cache_seqlens_int32
        )
        dsa_cu_seqlens_k = compute_cu_seqlens(dsa_cache_seqlens_int32)
        dsa_cu_seqlens_q = self.get_device_int32_arange(len(dsa_cu_seqlens_k))

        paged_mqa_schedule_metadata = None
        # DeepGEMM paged MQA logits path needs a schedule metadata tensor.
        # Compute it once per forward batch and reuse it across layers.
        if (is_cuda() or _is_hcu) and (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            deep_gemm = _get_deep_gemm()
            if deep_gemm is not None:
                # NOTE: DeepGEMM paged path uses block_size=64.
                seqlens_32 = (
                    seqlens_expanded
                    if (
                        forward_batch.forward_mode.is_target_verify()
                        or forward_batch.forward_mode.is_draft_extend_v2()
                    )
                    else cache_seqlens_int32
                )
                seqlens_32_2d = _to_2d_context_lens(
                    seqlens_32, forward_batch.batch_size
                )
                paged_mqa_schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                    seqlens_32_2d, 64, deep_gemm.get_num_sms()
                )
        metadata = DSAMetadata(
            page_size=self.real_page_size,
            cache_seqlens_int32=cache_seqlens_int32,
            max_seq_len_q=max_seqlen_q,
            max_seq_len_k=max_seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seq_lens_sum=forward_batch.seq_lens_sum,
            page_table_1=page_table,
            page_table_1_flattened=page_table_1_flattened,
            flashmla_metadata=(
                self._compute_flashmla_metadata(
                    cache_seqlens=dsa_cache_seqlens_int32,
                    seq_len_q=1,
                )
                if use_flashmla_kv
                else None
            ),
            paged_mqa_schedule_metadata=paged_mqa_schedule_metadata,
            dsa_cache_seqlens_int32=dsa_cache_seqlens_int32,
            dsa_cu_seqlens_q=dsa_cu_seqlens_q,
            dsa_cu_seqlens_k=dsa_cu_seqlens_k,
            dsa_seqlens_expanded=seqlens_expanded,
            dsa_extend_seq_lens_list=self._get_dsa_page_table_transform_lens(
                forward_batch.forward_mode,
                extend_seq_lens_cpu,
                page_table.shape[0],
            ),
            real_page_table=self._transform_table_1_to_real(page_table),
            dsa_max_seqlen_q=1,
            topk_indices_offset=topk_indices_offset,
            indexer_k_start_end=indexer_k_start_end,
            indexer_seq_lens_cpu=indexer_seq_lens_cpu,
            indexer_seq_lens=indexer_seq_lens,
            token_to_batch_idx=token_to_batch_idx,
        )

        if use_kpool:
            metadata = self._build_kpool_metadata(
                metadata,
                forward_batch,
                topk_transform_method,
                kpool_inputs,
                cache_seqlens_int32,
                seqlens_expanded,
            )

        self.forward_metadata = metadata

    def _cal_indexer_k_start_end(
        self,
        forward_batch: ForwardBatch,
        bs_idx: Optional[List[int]] = None,
    ):
        if not forward_batch.forward_mode.is_extend_without_speculative():
            return None, None
        if forward_batch.batch_size == 0 or (bs_idx is not None and len(bs_idx) == 0):
            empty_t = torch.empty(0, dtype=torch.int32, device=self.device)
            return (empty_t, empty_t), empty_t

        # Suppose there are two requests, with extend_seq_len = [3, 2]
        # and seq_lens = [10, 4]
        # The logits matrix looks like this, with * representing the valid logits
        # and - representing the invalid logits:
        #
        #  ********--|----
        #  *********-|----
        #  **********|----
        #  ----------|***-
        #  ----------|****
        #
        # ks = [0, 0, 0, 10, 10]
        # ke = [8, 9, 10, 13, 14]
        ks_list = []
        ke_list = []
        token_to_batch_idx = []

        q_offset = 0
        k_offset = 0

        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )
        for i in range(forward_batch.batch_size):
            seq_len = forward_batch.seq_lens_cpu[i].item()
            assert isinstance(seq_len, int)
            extend_seq_len = forward_batch.extend_seq_lens_cpu[i]
            ks = torch.full(
                (extend_seq_len,), k_offset, dtype=torch.int32, device=self.device
            )
            kv_len = seq_len
            if forward_batch.forward_mode.is_target_verify():
                kv_len += self.speculative_num_draft_tokens
            seq_lens_expanded = torch.arange(
                kv_len - extend_seq_len + 1,
                kv_len + 1,
                dtype=torch.int32,
                device=self.device,
            )
            ke = ks + seq_lens_expanded
            ks_list.append(ks)
            ke_list.append(ke)

            # bi: The index within the selected batch bs_idx. Entries that were not selected are ignored.
            bi = bs_idx.index(i) if (bs_idx is not None and i in bs_idx) else i
            tb = torch.full(
                (extend_seq_len,), bi, dtype=torch.int32, device=self.device
            )
            token_to_batch_idx.append(tb)

            if bs_idx is None or i in bs_idx:  # skip batch not included in bs_idx
                q_offset += extend_seq_len
                k_offset += seq_len

        ks = torch.cat(ks_list, dim=0)
        ke = torch.cat(ke_list, dim=0)
        token_to_batch_idx = torch.cat(token_to_batch_idx, dim=0)
        if bs_idx is not None:
            assert can_dsa_prefill_cp_round_robin_split(forward_batch)
            ks = dsa_cp_round_robin_split_data(ks)
            ke = dsa_cp_round_robin_split_data(ke)
            token_to_batch_idx = dsa_cp_round_robin_split_data(token_to_batch_idx)
        return (ks, ke), token_to_batch_idx

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Initialize CUDA graph state for the attention backend.

        Args:
            max_bs (int): Maximum batch size to support in CUDA graphs

        This creates fixed-size tensors that will be reused during CUDA graph replay
        to avoid memory allocations.
        """
        self.decode_cuda_graph_metadata: Dict = {
            "cache_seqlens": torch.ones(
                max_num_tokens, dtype=torch.int32, device=self.device
            ),
            "cu_seqlens_q": torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "cu_seqlens_k": torch.zeros(
                max_bs + 1, dtype=torch.int32, device=self.device
            ),
            # fake page_table for sparse_prefill
            # Match req_to_token_pool width. Speculative decoding can carry
            # accepted bonus tokens into the next verify step before adding the
            # current draft-token window, so context_len + draft_tokens is not
            # always enough near the context boundary.
            "page_table": torch.zeros(
                max_num_tokens,
                self.max_graph_page_table_len,
                dtype=torch.int32,
                device=self.device,
            ),
            "flashmla_metadata": (
                self._compute_flashmla_metadata(
                    cache_seqlens=torch.ones(
                        max_num_tokens, dtype=torch.int32, device=self.device
                    ),
                    seq_len_q=1,
                )
                if self.dsa_decode_impl == "flashmla_kv"
                else None
            ),
            "target_verify_extend_seq_lens": copy_cpu_values_to_device(
                [self.speculative_num_draft_tokens or 0] * max_bs,
                self.device,
                dtype=torch.int32,
            ),
        }
        # max_num_tokens already includes speculative expansion.
        if self._lightop_decode_graph_workspaces is None:
            self._allocate_lightop_decode_workspaces(max_num_tokens)
            if self._lightop_decode_gather_workspace is not None:
                self._lightop_decode_graph_workspaces = (
                    self._lightop_decode_gather_workspace,
                    self._lightop_decode_compact_indices,
                )
        elif self._lightop_decode_graph_workspaces[0].shape[0] < max_num_tokens:
            raise RuntimeError("LightOp decode CUDA graph workspace cannot grow")

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        if in_capture:
            # The EAGLE draft capture batch has no input_ids until draft_forward.
            num_tokens = (
                forward_batch.batch_size * (self.topk or 1)
                if forward_batch.forward_mode.is_decode_or_idle()
                else forward_batch.input_ids.numel()
            )
            self._capture_cuda_graph_metadata(
                bs=forward_batch.batch_size,
                num_tokens=num_tokens,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                encoder_lens=forward_batch.encoder_lens,
                forward_mode=forward_batch.forward_mode,
                spec_info=forward_batch.spec_info,
            )
        else:
            self._replay_cuda_graph_metadata(
                bs=forward_batch.batch_size,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                seq_lens_sum=forward_batch.seq_lens_sum,
                encoder_lens=forward_batch.encoder_lens,
                forward_mode=forward_batch.forward_mode,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
                out_cache_loc=forward_batch.out_cache_loc,
                actual_forward_mode=getattr(forward_batch, "actual_forward_mode", None),
            )

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        # Like main's DSA backend, this backend has no tree-mask buffers to fix up.
        return None

    def _capture_cuda_graph_metadata(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        self.set_dsa_prefill_impl(forward_batch=None)

        """Initialize forward metadata for capturing CUDA graph."""
        if forward_mode.is_decode_or_idle():
            # Normal Decode
            # Get sequence information
            cache_seqlens_int32 = seq_lens.to(torch.int32)
            cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)

            # Use max context length for seq_len_k
            page_table_1 = self.decode_cuda_graph_metadata["page_table"][:bs, :]
            max_seqlen_q = 1
            max_seqlen_k = page_table_1.shape[1]

            # Precompute page table
            # Precompute cumulative sequence lengths

            # NOTE(dark): this is always arange, since we are decoding
            cu_seqlens_q = self.decode_cuda_graph_metadata["cu_seqlens_q"][: bs + 1]
            dsa_cache_seqlens_int32 = compute_dsa_seqlens(
                cache_seqlens_int32,
                dsa_index_topk=self.dsa_index_topk,
                index_kpool=self.dsa_index_kpool,
            )
            if (
                self.dsa_decode_impl == "flashmla_kv"
                and _is_hcu
                and self.dsa_index_kpool > 1
            ):
                topk_block_size = 64
                dsa_cache_seqlens_int32 = (
                    torch.div(
                        dsa_cache_seqlens_int32 + topk_block_size - 1,
                        topk_block_size,
                        rounding_mode="floor",
                    )
                    * topk_block_size
                ).to(torch.int32)

            seqlens_expanded = cache_seqlens_int32
            dsa_extend_seq_lens_list = [1] * num_tokens
            if self.dsa_decode_impl == "flashmla_kv":
                flashmla_metadata = self.decode_cuda_graph_metadata[
                    "flashmla_metadata"
                ].slice(slice(0, num_tokens + 1))
                flashmla_metadata.copy_(
                    self._compute_flashmla_metadata(
                        cache_seqlens=dsa_cache_seqlens_int32,
                        seq_len_q=1,
                    )
                )
            else:
                flashmla_metadata = None
        elif forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2():
            cache_seqlens_int32 = (seq_lens + self.speculative_num_draft_tokens).to(
                torch.int32
            )
            cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)
            max_seqlen_q = 1
            page_table_1 = self.decode_cuda_graph_metadata["page_table"][
                : bs * self.speculative_num_draft_tokens, :
            ]
            max_seqlen_k = page_table_1.shape[1]

            cu_seqlens_q = torch.arange(
                0,
                bs * self.speculative_num_draft_tokens + 1,
                1,
                dtype=torch.int32,
                device=self.device,
            )

            extend_seq_lens_cpu = [self.speculative_num_draft_tokens] * bs

            seqlens_int32_cpu = [
                self.speculative_num_draft_tokens + kv_len
                for kv_len in seq_lens.tolist()
            ]
            seqlens_expanded = torch.cat(
                [
                    torch.arange(
                        kv_len - qo_len + 1,
                        kv_len + 1,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    for qo_len, kv_len in zip(
                        extend_seq_lens_cpu,
                        seqlens_int32_cpu,
                        strict=True,
                    )
                ]
            )
            dsa_cache_seqlens_int32 = compute_dsa_seqlens(
                seqlens_expanded,
                dsa_index_topk=self.dsa_index_topk,
                index_kpool=self.dsa_index_kpool,
            )
            if (
                self.dsa_decode_impl == "flashmla_kv"
                and _is_hcu
                and self.dsa_index_kpool > 1
            ):
                topk_block_size = 64
                dsa_cache_seqlens_int32 = (
                    torch.div(
                        dsa_cache_seqlens_int32 + topk_block_size - 1,
                        topk_block_size,
                        rounding_mode="floor",
                    )
                    * topk_block_size
                ).to(torch.int32)
            dsa_extend_seq_lens_list = [1] * bs * self.speculative_num_draft_tokens

            if self.dsa_decode_impl == "flashmla_kv":
                flashmla_metadata = self.decode_cuda_graph_metadata[
                    "flashmla_metadata"
                ].slice(slice(0, bs * self.speculative_num_draft_tokens + 1))

                flashmla_metadata.copy_(
                    self._compute_flashmla_metadata(
                        cache_seqlens=dsa_cache_seqlens_int32,
                        seq_len_q=1,
                    )
                )
            else:
                flashmla_metadata = None

        dsa_cu_seqlens_k = compute_cu_seqlens(dsa_cache_seqlens_int32)
        dsa_cu_seqlens_q = self.get_device_int32_arange(len(dsa_cu_seqlens_k))
        real_page_table = self._transform_table_1_to_real(page_table_1)

        paged_mqa_schedule_metadata = None
        if (is_cuda() or _is_hcu) and (
            forward_mode.is_decode_or_idle()
            or forward_mode.is_target_verify()
            or forward_mode.is_draft_extend_v2()
        ):
            deep_gemm = _get_deep_gemm()
            if deep_gemm is not None:
                seqlens_32 = (
                    seqlens_expanded
                    if (
                        forward_mode.is_target_verify()
                        or forward_mode.is_draft_extend_v2()
                    )
                    else cache_seqlens_int32
                )
                seqlens_32_2d = _to_2d_context_lens(seqlens_32, bs)
                paged_mqa_schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                    seqlens_32_2d, 64, deep_gemm.get_num_sms()
                )

        metadata = DSAMetadata(
            page_size=self.real_page_size,
            cache_seqlens_int32=cache_seqlens_int32,
            max_seq_len_q=max_seqlen_q,
            max_seq_len_k=max_seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            page_table_1=page_table_1,
            flashmla_metadata=flashmla_metadata,
            paged_mqa_schedule_metadata=paged_mqa_schedule_metadata,
            dsa_cache_seqlens_int32=dsa_cache_seqlens_int32,
            dsa_cu_seqlens_q=dsa_cu_seqlens_q,
            dsa_cu_seqlens_k=dsa_cu_seqlens_k,
            dsa_seqlens_expanded=seqlens_expanded,
            real_page_table=real_page_table,
            dsa_extend_seq_lens_list=dsa_extend_seq_lens_list,
        )
        metadata = self._init_pooled_paged_mqa_metadata(
            metadata=metadata,
            seqlens_32=cache_seqlens_int32,
            forward_mode=forward_mode,
        )
        if self.dsa_index_kpool > 1:
            is_verify = forward_mode.is_target_verify()
            is_v2 = forward_mode.is_draft_extend_v2()
            is_ring_write = forward_mode.is_decode_or_idle() or is_verify or is_v2
            if is_ring_write:
                num_draft_tokens = (
                    1
                    if forward_mode.is_decode_or_idle()
                    else self.speculative_num_draft_tokens
                )
                metadata = _init_kpool_write_plan_capture_impl(
                    metadata,
                    max_bs=bs,
                    pool_size=self.dsa_index_kpool,
                    real_page_size=self.real_page_size,
                    real_page_table=real_page_table,
                    num_draft_tokens=num_draft_tokens,
                    device=self.device,
                    is_verify=is_verify or is_v2,
                    is_v2=is_v2,
                )
                write_start = seq_lens.to(torch.int32)
                if forward_mode.is_decode_or_idle():
                    write_start = write_start - 1
                elif is_v2:
                    write_start = write_start - self.speculative_num_draft_tokens
                _update_kpool_write_plan_impl(
                    metadata,
                    write_start=write_start,
                    req_pool_indices=req_pool_indices,
                    real_page_table=real_page_table,
                    pool_size=self.dsa_index_kpool,
                    real_page_size=self.real_page_size,
                    num_draft_tokens=num_draft_tokens,
                    forward_mode=forward_mode,
                )
        self.decode_cuda_graph_metadata[bs] = metadata
        self.forward_metadata = metadata

    def _replay_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        out_cache_loc: Optional[torch.Tensor] = None,
        actual_forward_mode: Optional[ForwardMode] = None,
    ):
        """Initialize forward metadata for replaying CUDA graph."""

        self.set_dsa_prefill_impl(forward_batch=None)

        seq_lens = seq_lens[:bs]
        if seq_lens_cpu is not None:
            seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]

        # Normal Decode
        metadata: DSAMetadata = self.decode_cuda_graph_metadata[bs]
        if forward_mode.is_decode_or_idle():
            # Normal Decode
            cache_seqlens = seq_lens.to(torch.int32)
            metadata.cache_seqlens_int32.copy_(cache_seqlens)
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32)
            )
            fill_decode_page_table_gpu(
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                metadata.page_table_1,
                bs,
            )
            page_indices = metadata.page_table_1
            dsa_cache_seqlens = compute_dsa_seqlens(
                cache_seqlens,
                dsa_index_topk=self.dsa_index_topk,
                index_kpool=self.dsa_index_kpool,
            )
            metadata.dsa_cache_seqlens_int32.copy_(dsa_cache_seqlens)
            seqlens_expanded = cache_seqlens
        elif forward_mode.is_target_verify():
            assert seq_lens_cpu is not None
            max_seqlen_k = int(
                seq_lens_cpu.max().item() + self.speculative_num_draft_tokens
            )

            cache_seqlens = (seq_lens + self.speculative_num_draft_tokens).to(
                torch.int32
            )
            metadata.cache_seqlens_int32.copy_(cache_seqlens)
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32)
            )
            page_indices = self.req_to_token[req_pool_indices, :max_seqlen_k]
            page_indices = torch.repeat_interleave(
                page_indices,
                repeats=self.speculative_num_draft_tokens,
                dim=0,
                output_size=bs * self.speculative_num_draft_tokens,
            )
            metadata.page_table_1[:, :max_seqlen_k].copy_(page_indices)
            seqlens_expanded = seqlens_expand_triton(
                self.decode_cuda_graph_metadata["target_verify_extend_seq_lens"][:bs],
                cache_seqlens,
                self.speculative_num_draft_tokens * bs,
                self.speculative_num_draft_tokens,
            )
            metadata.dsa_seqlens_expanded.copy_(seqlens_expanded)
            dsa_cache_seqlens = compute_dsa_seqlens(
                seqlens_expanded,
                self.dsa_index_topk,
                index_kpool=self.dsa_index_kpool,
            )
            metadata.dsa_cache_seqlens_int32.copy_(dsa_cache_seqlens)
        elif forward_mode.is_draft_extend_v2():
            assert seq_lens_cpu is not None
            max_seqlen_k = int(seq_lens_cpu.max().item())
            cache_seqlens = seq_lens.to(torch.int32)
            metadata.cache_seqlens_int32.copy_(cache_seqlens)
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32)
            )

            page_indices = self.req_to_token[req_pool_indices, :max_seqlen_k]

            extend_seq_lens = spec_info.num_accept_tokens[:bs]
            extend_num_tokens = bs * self.speculative_num_draft_tokens
            page_indices = torch.repeat_interleave(
                page_indices,
                repeats=self.speculative_num_draft_tokens,
                dim=0,
                output_size=extend_num_tokens,
            )
            metadata.page_table_1[: page_indices.shape[0], :max_seqlen_k].copy_(
                page_indices
            )

            seqlens_expanded = seqlens_expand_triton(
                extend_seq_lens,
                cache_seqlens,
                extend_num_tokens,
                self.speculative_num_draft_tokens,
            )
            metadata.dsa_seqlens_expanded[: seqlens_expanded.shape[0]].copy_(
                seqlens_expanded
            )
            dsa_cache_seqlens = compute_dsa_seqlens(
                seqlens_expanded,
                self.dsa_index_topk,
                index_kpool=self.dsa_index_kpool,
            )
            metadata.dsa_cache_seqlens_int32[: seqlens_expanded.shape[0]].copy_(
                dsa_cache_seqlens
            )

        # Pooled MQA buffers are graph-captured and backend-local. Refresh them
        # from the runtime cache lengths before replay so every DSA layer sees
        # the current batch rather than the capture batch's pooled lengths.
        self._update_pooled_paged_mqa_metadata(
            metadata=metadata,
            seqlens_32=metadata.cache_seqlens_int32,
            forward_mode=forward_mode,
        )

        # Update DeepGEMM paged MQA schedule metadata outside the captured graph.
        if (is_cuda() or _is_hcu) and (
            forward_mode.is_decode_or_idle()
            or forward_mode.is_target_verify()
            or forward_mode.is_draft_extend_v2()
        ):
            deep_gemm = _get_deep_gemm()
            if deep_gemm is not None:
                seqlens_32 = (
                    seqlens_expanded
                    if (
                        forward_mode.is_target_verify()
                        or forward_mode.is_draft_extend_v2()
                    )
                    else metadata.cache_seqlens_int32
                )
                seqlens_32_2d = _to_2d_context_lens(seqlens_32, bs)
                new_schedule = deep_gemm.get_paged_mqa_logits_metadata(
                    seqlens_32_2d, 64, deep_gemm.get_num_sms()
                )
                if metadata.paged_mqa_schedule_metadata is None:
                    object.__setattr__(
                        metadata, "paged_mqa_schedule_metadata", new_schedule
                    )
                else:
                    metadata.paged_mqa_schedule_metadata.copy_(new_schedule)
            else:
                object.__setattr__(metadata, "paged_mqa_schedule_metadata", None)

        # Keep graph-captured page-table tensor addresses stable while refreshing
        # runtime values before kpool write-plan or attention replay consumes them.
        assert self.real_page_size == metadata.page_size
        if self.real_page_size > 1:
            real_table = self._transform_table_1_to_real(page_indices)
            new_rows = real_table.shape[0]
            new_cols = real_table.shape[1]
            metadata.real_page_table[:new_rows, :new_cols].copy_(real_table)
        else:
            assert metadata.real_page_table is metadata.page_table_1

        # replay update kpool write plan
        if self.dsa_index_kpool > 1:
            is_verify = forward_mode.is_target_verify()
            is_v2 = forward_mode.is_draft_extend_v2()
            is_ring_write = forward_mode.is_decode_or_idle() or is_verify or is_v2
            if is_ring_write:
                real_page_table = metadata.real_page_table
                num_draft_tokens = (
                    1
                    if forward_mode.is_decode_or_idle()
                    else self.speculative_num_draft_tokens
                )
                write_start = seq_lens.to(torch.int32)
                if forward_mode.is_decode_or_idle():
                    write_start = write_start - 1
                elif is_v2:
                    write_start = write_start - self.speculative_num_draft_tokens
                # See dsa_backend.py:548 note: reference reads
                # `spec_info.accept_length` (bonus-inclusive in v2); this
                # project renamed it to `num_accept_tokens`. Without this
                # rename, replay leaves effective_n_per_batch at 0 and the
                # fused write-then-compress kernel never crosses a pool
                # boundary under CUDA graph.
                accept_length = (
                    spec_info.num_accept_tokens[:bs]
                    if is_v2
                    and spec_info is not None
                    and getattr(spec_info, "num_accept_tokens", None) is not None
                    else None
                )
                _update_kpool_write_plan_impl(
                    metadata,
                    write_start=write_start,
                    req_pool_indices=req_pool_indices,
                    real_page_table=real_page_table,
                    pool_size=self.dsa_index_kpool,
                    real_page_size=self.real_page_size,
                    num_draft_tokens=num_draft_tokens,
                    forward_mode=forward_mode,
                    accept_length=accept_length,
                )
        seqlens_expanded_size = seqlens_expanded.shape[0]
        assert (
            metadata.dsa_cache_seqlens_int32 is not None
            and metadata.dsa_cu_seqlens_k is not None
            and self.dsa_index_topk is not None
        )

        metadata.dsa_cu_seqlens_k[1 : 1 + seqlens_expanded_size].copy_(
            torch.cumsum(dsa_cache_seqlens, dim=0, dtype=torch.int32)
        )
        # NOTE(dark): (dsa-) cu_seqlens_q is always arange, no need to copy

        if self.dsa_decode_impl == "flashmla_kv":
            flashmla_metadata = metadata.flashmla_metadata.slice(
                slice(0, seqlens_expanded_size + 1)
            )
            flashmla_metadata.copy_(
                self._compute_flashmla_metadata(
                    cache_seqlens=dsa_cache_seqlens,
                    seq_len_q=1,
                )
            )

        self.forward_metadata = metadata

    def init_forward_metadata_replay_cuda_graph_from_precomputed(
        self,
        bs: int,
        precomputed: PrecomputedMetadata,
        forward_mode: ForwardMode,
        skip_kpool_write_plan_update: bool = False,
    ):
        """Fast path: copy precomputed metadata to this backend's metadata.

        This function only performs copy operations, no computation.

        Args:
            bs: Batch size
            precomputed: Precomputed metadata to copy from
            forward_mode: Forward mode
        """
        self.set_dsa_prefill_impl(forward_batch=None)

        metadata = self.decode_cuda_graph_metadata[bs]

        # Track whether fused kernel succeeded
        fused_kernel_succeeded = False

        # Use fused CUDA kernel for all copy operations
        if _USE_FUSED_METADATA_COPY:
            try:
                from sglang.kernels.ops.attention.fused_metadata_copy import (
                    fused_metadata_copy_cuda,
                )

                # Map forward_mode to integer enum
                if forward_mode.is_decode_or_idle():
                    mode_int = 0  # DECODE
                elif forward_mode.is_target_verify():
                    mode_int = 1  # TARGET_VERIFY
                else:
                    raise ValueError(f"Unsupported forward_mode: {forward_mode}")

                # Prepare FlashMLA tensors if needed
                flashmla_num_splits_src = None
                flashmla_num_splits_dst = None
                flashmla_metadata_src = None
                flashmla_metadata_dst = None
                if precomputed.flashmla_metadata is not None:
                    flashmla_num_splits_src = precomputed.flashmla_metadata.num_splits
                    flashmla_num_splits_dst = metadata.flashmla_metadata.num_splits
                    flashmla_metadata_src = (
                        precomputed.flashmla_metadata.flashmla_metadata
                    )
                    flashmla_metadata_dst = metadata.flashmla_metadata.flashmla_metadata

                # Call fused kernel
                fused_metadata_copy_cuda(
                    # Source tensors
                    precomputed.cache_seqlens,
                    precomputed.cu_seqlens_k,
                    precomputed.page_indices,
                    precomputed.dsa_cache_seqlens,
                    precomputed.seqlens_expanded,
                    precomputed.dsa_cu_seqlens_k,
                    precomputed.real_page_table,
                    flashmla_num_splits_src,
                    flashmla_metadata_src,
                    # Destination tensors
                    metadata.cache_seqlens_int32,
                    metadata.cu_seqlens_k,
                    metadata.page_table_1,
                    metadata.dsa_cache_seqlens_int32,
                    metadata.dsa_seqlens_expanded,
                    metadata.dsa_cu_seqlens_k,
                    (
                        metadata.real_page_table
                        if precomputed.real_page_table is not None
                        else None
                    ),
                    flashmla_num_splits_dst,
                    flashmla_metadata_dst,
                    # Parameters
                    mode_int,
                    bs,
                    precomputed.max_len,
                    precomputed.max_seqlen_k,
                    precomputed.seqlens_expanded_size,
                )

                # Successfully used fused kernel
                fused_kernel_succeeded = True

            except ImportError:
                print_warning_once(
                    "Fused metadata copy kernel not available, falling back to individual copies."
                )
            except Exception as e:
                print_warning_once(
                    f"Fused metadata copy kernel failed with error: {e}, falling back to individual copies."
                )

        # Fallback to individual copy operations if fused kernel disabled or failed
        if not fused_kernel_succeeded:
            # Copy basic seqlens
            metadata.cache_seqlens_int32.copy_(precomputed.cache_seqlens)
            metadata.cu_seqlens_k[1:].copy_(precomputed.cu_seqlens_k[1:])

            # Mode-specific copy logic
            if forward_mode.is_decode_or_idle():
                # Decode mode
                metadata.page_table_1[:, : precomputed.max_len].copy_(
                    precomputed.page_indices
                )
                metadata.dsa_cache_seqlens_int32.copy_(precomputed.dsa_cache_seqlens)
                # seqlens_expanded is same as cache_seqlens (already copied)

            elif forward_mode.is_target_verify():
                # Target verify mode
                metadata.page_table_1[:, : precomputed.max_seqlen_k].copy_(
                    precomputed.page_indices
                )
                metadata.dsa_seqlens_expanded.copy_(precomputed.seqlens_expanded)
                metadata.dsa_cache_seqlens_int32.copy_(precomputed.dsa_cache_seqlens)

            # Copy DSA cu_seqlens
            size = precomputed.seqlens_expanded_size
            metadata.dsa_cu_seqlens_k[1 : 1 + size].copy_(
                precomputed.dsa_cu_seqlens_k[1 : 1 + size]
            )

            # Copy real page table
            if precomputed.real_page_table is not None:
                rows, cols = precomputed.real_page_table.shape
                metadata.real_page_table[:rows, :cols].copy_(
                    precomputed.real_page_table
                )

            # Copy FlashMLA metadata in fallback path
            if precomputed.flashmla_metadata is not None:
                size = precomputed.seqlens_expanded_size
                flashmla_metadata = metadata.flashmla_metadata.slice(slice(0, size + 1))
                flashmla_metadata.copy_(precomputed.flashmla_metadata)

        # Refresh DeepGEMM paged MQA schedule metadata for the actual seqlens of
        # this replay (the captured graph holds stale data otherwise, which can
        # deadlock the kernel when the runtime work decomposition diverges from
        # the captured one).
        if is_cuda():
            deep_gemm = _get_deep_gemm()
            if deep_gemm is not None:
                if forward_mode.is_decode_or_idle():
                    seqlens_32 = metadata.cache_seqlens_int32
                else:
                    seqlens_32 = metadata.dsa_seqlens_expanded[
                        : precomputed.seqlens_expanded_size
                    ]
                seqlens_32_2d = _to_2d_context_lens(seqlens_32, bs)
                new_schedule = deep_gemm.get_paged_mqa_logits_metadata(
                    seqlens_32_2d, 64, deep_gemm.get_num_sms()
                )
                if metadata.paged_mqa_schedule_metadata is None:
                    object.__setattr__(
                        metadata, "paged_mqa_schedule_metadata", new_schedule
                    )
                else:
                    metadata.paged_mqa_schedule_metadata.copy_(new_schedule)

        # The fused metadata-copy kernels only copy the ordinary DSA fields.
        # Pooled MQA metadata and the KPool write plan are backend-local graph
        # buffers and must be refreshed from the runtime sequence lengths.
        self._update_pooled_paged_mqa_metadata(
            metadata=metadata,
            seqlens_32=metadata.cache_seqlens_int32,
            forward_mode=forward_mode,
        )
        if not skip_kpool_write_plan_update:
            self._update_kpool_write_plan_from_precomputed(
                metadata=metadata,
                precomputed=precomputed,
            )
        self.forward_metadata = metadata

    def _update_kpool_write_plan_from_precomputed(
        self,
        *,
        metadata: DSAMetadata,
        precomputed: PrecomputedMetadata,
    ) -> None:
        """Refresh the backend-local decode write plan on fast MTP replay."""
        if not (self.dsa_index_kpool > 1 and precomputed.req_pool_indices is not None):
            return

        bs = precomputed.req_pool_indices.shape[0]
        write_start = precomputed.cache_seqlens[:bs] - 1
        if write_start.dtype != torch.int32:
            write_start = write_start.to(torch.int32)
        _update_kpool_write_plan_impl(
            metadata,
            write_start=write_start,
            req_pool_indices=precomputed.req_pool_indices,
            real_page_table=metadata.real_page_table,
            pool_size=self.dsa_index_kpool,
            real_page_size=self.real_page_size,
            num_draft_tokens=1,
            forward_mode=ForwardMode.DECODE,
            accept_length=None,
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        causal = not layer.is_cross_attention
        metadata = self.forward_metadata
        assert causal, "DSA is causal only"

        forward_mode = effective_forward_mode(forward_batch)
        dsa_impl = (
            self.dsa_decode_impl
            if (
                forward_mode.is_decode_or_idle()
                or forward_mode.is_target_verify()
                or forward_mode.is_draft_extend_v2()
            )
            else self.dsa_prefill_impl
        )

        if q_rope is not None and q_rope.shape[-1] == 0:
            q_rope = None

        if dsa_impl == "trtllm" and not self.use_mha:
            return self._forward_trtllm(
                q,
                k,
                v,
                layer,
                forward_batch,
                metadata.dsa_cache_seqlens_int32,
                save_kv_cache,
                q_rope,
                k_rope,
                topk_indices,
                cos_sin_cache,
                is_neox,
                llama_4_scaling,
                is_prefill=not forward_mode.is_decode_or_idle(),
            )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                self.token_to_kv_pool.set_mla_kv_buffer(  # type: ignore
                    layer,
                    cache_loc,
                    k,
                    k_rope,
                )

        # Use MHA kernel if in MHA_ONE_SHOT mode
        if self.use_mha:
            assert k is not None and v is not None
            assert q_rope is None, "MHA_ONE_SHOT path should not pass q_rope"
            assert layer.tp_k_head_num == layer.tp_q_head_num > 1, (
                "MHA_ONE_SHOT requires dense multi-head config"
            )
            return self._forward_standard_mha(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
                metadata=metadata,
            )

        # Do absorbed multi-latent attention (MLA path)
        get_key_buffer_with_history = (
            getattr(
                self.token_to_kv_pool,
                "get_key_buffer_with_prefetch_history",
                None,
            )
            if _is_hcu and dsa_use_prefill_cp(forward_batch)
            else None
        )
        if get_key_buffer_with_history is not None:
            kv_cache = get_key_buffer_with_history(
                layer.layer_id,
                has_history=dsa_prefill_has_history(forward_batch),
            )
        else:
            kv_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)

        q_is_flashmla_padded = (
            _is_hcu
            and dsa_impl == "flashmla_sparse"
            and self.dsa_kv_cache_store_fp8
            and self.qk_rope_head_dim == 0
            and q_rope is None
            and q.ndim == 3
            and q.shape[1] == layer.tp_q_head_num
            and q.shape[2] == layer.head_dim + 64
            and q.is_contiguous()
        )
        if q_is_flashmla_padded:
            q_all = q
            q_nope = q_all[..., : layer.v_head_dim]
        elif q_rope is not None:
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
            q_all = None
        else:
            q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
            q_nope = q_all[:, :, : layer.v_head_dim]
            q_rope = (
                None
                if layer.head_dim == layer.v_head_dim
                else q_all[:, :, layer.v_head_dim :]
            )

        use_hcu_fp8_no_rope_512_prefill = (
            _is_hcu
            and dsa_impl == "flashmla_sparse"
            and self.dsa_kv_cache_store_fp8
            and self.qk_rope_head_dim == 0
            and q_rope is None
            and q_all is not None
            and q_all.shape[-1] == layer.v_head_dim
        )

        # Align topk_indices with q dimensions
        # This handles cases where q is padded (TP + partial DP attention)
        if topk_indices is not None:
            topk_indices = self._pad_topk_indices(topk_indices, q_nope.shape[0])

        # NOTE(dark): here, we use page size = 1
        topk_transform_method = self.get_topk_transform_method(forward_mode)
        if self._use_fused_topk(forward_batch):
            page_table_1 = topk_indices
        else:
            if topk_transform_method == TopkTransformMethod.RAGGED:
                topk_indices_offset = self._get_aligned_topk_indices_offset(
                    forward_batch,
                    metadata,
                    topk_indices.shape[0],
                )
                mask = topk_indices != -1
                topk_indices_offset = (
                    topk_indices_offset.unsqueeze(1)
                    if topk_indices_offset.ndim == 1
                    else topk_indices_offset
                )
                topk_indices = torch.where(
                    mask, topk_indices + topk_indices_offset, topk_indices
                )
            elif topk_transform_method == TopkTransformMethod.PAGED:
                if forward_mode.is_decode_or_idle():
                    page_table_1 = self._transform_decode_topk_indices(
                        forward_batch,
                        metadata.page_table_1,
                        topk_indices,
                    )
                else:
                    extend_lens_cpu = self._get_topk_transform_lens(
                        forward_batch,
                        metadata,
                        topk_indices.shape[0],
                    )
                    page_table_1 = self._transform_prefill_topk_indices(
                        forward_batch,
                        metadata.page_table_1,
                        topk_indices,
                        extend_lens_cpu,
                        page_table_row_indices=metadata.token_to_batch_idx,
                    )

        # todo hisparse: to cover more backends
        if self.hisparse_coordinator is not None:
            page_table_1 = self.token_to_kv_pool.translate_loc_to_hisparse_device(
                page_table_1
            )

        # Index-KV keeps the original physical metadata. Only the final Main-KV
        # consumer locations are translated into this batch's compact scratch.
        if topk_transform_method == TopkTransformMethod.PAGED:
            page_table_1 = self.token_to_kv_pool.translate_main_kv_loc_to_compact(
                page_table_1
            )

        if dsa_impl == "tilelang":
            if q_rope is not None:
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)
            return self._forward_tilelang(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif dsa_impl == "flashmla_sparse":
            if q_rope is not None:
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)

            if topk_transform_method == TopkTransformMethod.RAGGED:
                if any(forward_batch.extend_prefix_lens_cpu):
                    page_table_1_flattened = (
                        self.forward_metadata.page_table_1_flattened
                    )
                    assert page_table_1_flattened is not None
                    page_table_1_flattened = (
                        self.token_to_kv_pool.translate_main_kv_loc_to_compact(
                            page_table_1_flattened
                        )
                    )
                    kv_cache = dequantize_k_cache_paged(
                        kv_cache,
                        page_table_1_flattened,
                        target_dim_rope=(
                            0 if use_hcu_fp8_no_rope_512_prefill else None
                        ),
                    )
                else:
                    kv_cache = _cat([k, k_rope], dim=-1)
                page_table_1 = topk_indices

            return self._forward_flashmla_sparse(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
                layer_id=layer.layer_id,
                use_hcu_fp8_no_rope_512=use_hcu_fp8_no_rope_512_prefill,
            )
        elif dsa_impl == "flashmla_kv":
            if q_rope is not None:
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)
            return self._forward_flashmla_kv(
                q_all=q_all,
                kv_cache=kv_cache,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
                # TODO optimize args
                layer=layer,
                metadata=metadata,
                page_table_1=page_table_1,
                forward_batch=forward_batch,
            )
        elif dsa_impl == "fa3":
            return self._forward_fa3(
                q_rope=q_rope,
                kv_cache=kv_cache,
                v_head_dim=layer.v_head_dim,
                q_nope=q_nope,
                page_table=page_table_1,
                cache_seqlens=metadata.dsa_cache_seqlens_int32,
                cu_seqlens_q=metadata.dsa_cu_seqlens_q,
                cu_seqlens_k=metadata.dsa_cu_seqlens_k,
                max_seqlen_q=metadata.dsa_max_seqlen_q,
                sm_scale=layer.scaling,
                logit_cap=layer.logit_cap,
                page_size=1,
            )
        elif dsa_impl == "aiter":
            if q_rope is not None:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_aiter_extend(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                layer=layer,
            )
        else:
            raise ValueError(
                f"Unsupported {dsa_impl = } for forward_extend. Consider using an other attention backend."
            )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        causal = not layer.is_cross_attention
        metadata = self.forward_metadata
        assert causal, "DSA is causal only"

        if q_rope is not None and q_rope.shape[-1] == 0:
            q_rope = None

        if self.dsa_decode_impl == "trtllm":
            return self._forward_trtllm(
                q,
                k,
                v,
                layer,
                forward_batch,
                metadata.cache_seqlens_int32,
                save_kv_cache,
                q_rope,
                k_rope,
                topk_indices,
                cos_sin_cache,
                is_neox,
                llama_4_scaling,
            )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                self.token_to_kv_pool.set_mla_kv_buffer(  # type: ignore
                    layer,
                    cache_loc,
                    k,
                    k_rope,
                )

        # Do absorbed multi-latent attention
        kv_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        if q_rope is not None:
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
            # Caller passed split q_nope / q_rope; we'll need to concat below if
            # the chosen impl wants q_all.
            q_all = None
        else:
            # Caller passed already-concatenated q (q_all = q). Reuse it directly
            # via a zero-copy view; the impl-specific blocks below will skip the
            # otherwise redundant concat_mla_absorb_q_general call.
            q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
            q_nope = q_all[:, :, : layer.v_head_dim]
            q_rope = (
                None
                if layer.head_dim == layer.v_head_dim
                else q_all[:, :, layer.v_head_dim :]
            )

        # Align topk_indices with q dimensions
        if topk_indices is not None:
            topk_indices = self._pad_topk_indices(topk_indices, q_nope.shape[0])

        if self.hisparse_coordinator is not None:
            page_table_1 = self.hisparse_coordinator.swap_in_selected_pages(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                topk_indices,
                layer.layer_id,
            )
        elif self._use_fused_topk(forward_batch):
            page_table_1 = topk_indices
        else:
            page_table_1 = self._transform_decode_topk_indices(
                forward_batch,
                metadata.page_table_1,
                topk_indices,
            )

        if self.dsa_decode_impl == "flashmla_sparse":
            if q_rope is not None:
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)
            return self._forward_flashmla_sparse(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif self.dsa_decode_impl == "flashmla_kv":
            if q_rope is not None:
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)
            return self._forward_flashmla_kv(
                q_all=q_all,
                kv_cache=kv_cache,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
                # TODO optimize args
                layer=layer,
                metadata=metadata,
                page_table_1=page_table_1,
                forward_batch=forward_batch,
            )
        elif self.dsa_decode_impl == "tilelang":
            # Cat-skip (HIP-only): when caller passes q_rope=None on HIP, q_all
            # has already been set to a zero-copy view of q in the else branch
            # above and we can reuse it directly. The `not _is_hip` clause keeps
            # CUDA / MUSA paths byte-identical to pre-patch by always re-cat.
            if q_all is None or not _is_hip:
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)
            return self._forward_tilelang(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif self.dsa_decode_impl == "fa3":
            return self._forward_fa3(
                q_rope=q_rope,
                kv_cache=kv_cache,
                v_head_dim=layer.v_head_dim,
                q_nope=q_nope,
                page_table=page_table_1,
                cache_seqlens=metadata.dsa_cache_seqlens_int32,
                cu_seqlens_q=metadata.dsa_cu_seqlens_q,
                cu_seqlens_k=metadata.dsa_cu_seqlens_k,
                max_seqlen_q=metadata.dsa_max_seqlen_q,
                sm_scale=layer.scaling,
                logit_cap=layer.logit_cap,
                page_size=1,
            )
        elif self.dsa_decode_impl == "aiter":
            if q_all is None or not _is_hip:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_aiter(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                layer=layer,
                metadata=metadata,
                bs=forward_batch.batch_size,
            )

        else:
            assert False, f"Unsupported {self.dsa_decode_impl = }"

    def _forward_fa3(
        self,
        q_rope: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        q_nope: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        sm_scale: float,
        logit_cap: float,
        page_size: int,
    ) -> torch.Tensor:
        k_rope_cache = kv_cache[:, :, v_head_dim:]
        c_kv_cache = kv_cache[:, :, :v_head_dim]
        qk_rope_dim = k_rope_cache.shape[-1]
        k_rope_cache = k_rope_cache.view(-1, page_size, 1, qk_rope_dim)
        c_kv_cache = c_kv_cache.view(-1, page_size, 1, v_head_dim)
        k_rope_cache = k_rope_cache.to(q_rope.dtype)
        c_kv_cache = c_kv_cache.to(q_rope.dtype)
        o = flash_attn_with_kvcache(
            q=q_rope,
            k_cache=k_rope_cache,
            v_cache=c_kv_cache,
            qv=q_nope,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=sm_scale,
            causal=True,
            softcap=logit_cap,
            return_softmax_lse=False,
            num_splits=self.num_splits,
        )
        return o  # type: ignore

    def _forward_flashmla_sparse(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        page_table_1: torch.Tensor,
        sm_scale: float,
        layer_id: int = -1,
        use_hcu_fp8_no_rope_512: bool = False,
    ) -> torch.Tensor:
        use_aiter_sorted_indices = (
            _is_hcu
            and _kpool_topk_sort_writeback_enabled()
            and envs.SGLANG_DSA_KPOOL_AITER_TOPK.get()
            and self.dsa_index_kpool in (4, 16)
            and self.dsa_index_topk == 2048
        )
        if not _is_hcu:
            from sgl_kernel.flash_mla import flash_mla_sparse_fwd
        elif use_aiter_sorted_indices:
            import flash_mla.cuda as flash_mla_cuda
        else:
            from flash_mla.flash_mla_interface import flash_mla_sparse_fwd

        original_q_shape = tuple(q_all.shape)
        original_kv_shape = tuple(kv_cache.shape)

        # FlashMLA sparse kernel requires num_heads to be a multiple of 64 (Hopper) or 128 (Blackwell)
        # When using TP, num_heads might be smaller (e.g., 256//8=32)
        num_tokens, num_heads, head_dim = q_all.shape

        # Determine required padding based on GPU architecture (use cached value)
        required_padding = 128 if self.device_sm_major >= 10 else 64
        need_padding = num_heads % required_padding != 0

        hcu_padded_q_heads = False
        if _is_hcu and need_padding:
            assert required_padding % num_heads == 0, (
                f"num_heads {num_heads} cannot be padded to {required_padding}. "
                f"TP size may be too large for this model."
            )

            # HCU FlashMLA sparse prefill has the same head-count restriction as
            # decode; TP8 leaves GLM5-Next with only 4 local q heads, so pad to
            # the supported head variant and trim the output below.
            q_padded = q_all.new_zeros((num_tokens, required_padding, head_dim))
            q_padded[:, :num_heads, :] = q_all
            q_input = q_padded
            hcu_padded_q_heads = True
        elif _is_hcu:
            q_input = q_all
        elif need_padding:
            assert required_padding % num_heads == 0, (
                f"num_heads {num_heads} cannot be padded to {required_padding}. "
                f"TP size may be too large for this model."
            )

            # Pad q to required size
            q_padded = q_all.new_zeros((num_tokens, required_padding, head_dim))
            q_padded[:, :num_heads, :] = q_all
            q_input = q_padded
        else:
            q_input = q_all

        if (
            _is_hcu
            and self.dsa_kv_cache_store_fp8
            and self.qk_rope_head_dim == 0
            and not use_hcu_fp8_no_rope_512
        ):
            if q_input.shape[-1] == v_head_dim:
                q_padded = q_input.new_zeros(
                    *q_input.shape[:-1],
                    q_input.shape[-1] + 64,
                )
                q_padded[..., : q_input.shape[-1]] = q_input
                q_input = q_padded
            if kv_cache.shape[-1] == v_head_dim:
                kv_padded = kv_cache.new_zeros(
                    *kv_cache.shape[:-1],
                    kv_cache.shape[-1] + 64,
                )
                kv_padded[..., : kv_cache.shape[-1]] = kv_cache
                kv_cache = kv_padded

        if _is_hcu and self.dsa_index_kpool > 1:
            # HCU FlashMLA sparse prefill requires a stricter topk alignment than
            # the logical KPool width (topk + kpool - 1), so pad invalid pages.
            topk_block_size = 128
            topk_width = page_table_1.shape[-1]
            padded_topk_width = (
                (topk_width + topk_block_size - 1) // topk_block_size
            ) * topk_block_size
            if padded_topk_width != topk_width:
                page_table_1 = torch.nn.functional.pad(
                    page_table_1,
                    (0, padded_topk_width - topk_width),
                    value=-1,
                )

        # indices shape must be (s_q, h_kv=1, topk), keep h_kv=1 unchanged
        indices_input = page_table_1.unsqueeze(1)

        if use_aiter_sorted_indices:
            # Aiter's RAGGED KPool path already sorted the expanded token
            # indices during TopK writeback, so bypass FlashMLA's duplicate
            # Python-wrapper sort.
            o, _, _ = flash_mla_cuda.sparse_prefill_fwd(
                q_input,
                kv_cache,
                indices_input,
                sm_scale,
                v_head_dim,
                None,
                None,
            )
        else:
            o, _, _ = flash_mla_sparse_fwd(
                q=q_input,
                kv=kv_cache,
                indices=indices_input,
                sm_scale=sm_scale,
                d_v=v_head_dim,
            )

        # Trim output back to original num_heads if we padded
        if ((not _is_hcu) and need_padding) or hcu_padded_q_heads:
            o = o[:, :num_heads, :]

        return o

    def _forward_flashmla_kv(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        sm_scale: float,
        layer,
        metadata: DSAMetadata,
        page_table_1,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if not _is_hcu:
            from sgl_kernel.flash_mla import flash_mla_with_kvcache
        else:
            from flash_mla.flash_mla_interface import (
                flash_mla_sparse_fwd,
                flash_mla_with_kvcache,
            )

        cache_seqlens = metadata.dsa_cache_seqlens_int32
        assert metadata.flashmla_metadata is not None

        original_q_shape = tuple(q_all.shape)
        original_kv_shape = tuple(kv_cache.shape)

        # TODO the 2nd dim is seq_len_q, need to be >1 when MTP
        q_all = q_all.view(-1, 1, layer.tp_q_head_num, layer.head_dim)
        is_decode_family = (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        )
        # MLATokenToKVPool stores the packed cache as uint8 but exposes it as
        # the configured FP8 dtype.  LightOp consumes the packed bytes (rather
        # than PyTorch FP8 values), so reinterpret the OCP E4M3FN view without
        # copying.  Do not accept FNUZ: its bit encoding is different.
        lightop_kv_cache = None
        if kv_cache.is_contiguous():
            if kv_cache.dtype == torch.uint8:
                lightop_kv_cache = kv_cache
            elif kv_cache.dtype == torch.float8_e4m3fn:
                lightop_kv_cache = kv_cache.view(torch.uint8)
        use_lightop_decode_gather = (
            self._lightop_decode_gather is not None
            and is_decode_family
            and self.hisparse_coordinator is None
            and q_all.dtype == torch.bfloat16
            and q_all.shape[0] > 0
            and q_all.shape[-1] == self._lightop_decode_head_dim
            and v_head_dim == 512
            and lightop_kv_cache is not None
            and self.kv_cache_dim in (528, 656)
            and lightop_kv_cache.shape[-1] == self.kv_cache_dim
            and lightop_kv_cache.numel() % self.kv_cache_dim == 0
            and page_table_1.dim() == 2
            and page_table_1.dtype == torch.int32
            and page_table_1.is_contiguous()
            and page_table_1.shape[0] == q_all.shape[0]
            and page_table_1.shape[-1] == self._lightop_decode_gather_logical_width
            and cache_seqlens.dim() == 1
            and cache_seqlens.dtype == torch.int32
            and cache_seqlens.is_contiguous()
            and cache_seqlens.numel() > 0
            # DP attention pads q/page-table rows (for example 8 -> 48) while
            # retaining one cache length per real row.  In that layout the
            # per-row -1 indices, not an inferred repeated length, are the
            # sparse FlashMLA validity mask.
            and cache_seqlens.numel() <= q_all.shape[0]
        )
        use_hcu_fp8_no_rope_528_cache = (
            _is_hcu
            and self.dsa_kv_cache_store_fp8
            and self.qk_rope_head_dim == 0
            and self.kv_cache_dim == 528
        )
        if (
            use_hcu_fp8_no_rope_528_cache
            and q_all.shape[0] > 0
            and not use_lightop_decode_gather
        ):
            if is_decode_family:
                raise RuntimeError(
                    "HCU FP8 no-RoPE FlashMLA decode with 528-byte KV rows "
                    "requires the LightOp decode gather path"
                )
            raise RuntimeError(
                "HCU FP8 no-RoPE FlashMLA prefill with 528-byte KV rows "
                "must use flashmla_sparse"
            )
        # The installed gfx936 paged BF16 decode specialization is optimized
        # for d_qk=512.  FlashMLA's existing flat sparse BF16 entry supports
        # the RoPE-aware d_qk=576 layout.  LightOp has already compacted the
        # cache, so the latter can consume it without another gather.
        use_lightop_flat_sparse_fwd = (
            use_lightop_decode_gather and self.qk_rope_head_dim == 64
        )
        if (
            _is_hcu
            and self.dsa_kv_cache_store_fp8
            and self.qk_rope_head_dim == 0
            and self.kv_cache_dim == 656
            and not use_lightop_decode_gather
        ):
            q_padded = q_all.new_zeros(
                *q_all.shape[:-1],
                q_all.shape[-1] + 64,
            )
            q_padded[..., : q_all.shape[-1]] = q_all
            q_all = q_padded
        num_q_heads = q_all.shape[2]
        target_q_heads = self.flashmla_kv_num_q_heads
        if target_q_heads != num_q_heads:
            # Pad q heads to match FlashMLA decode supported head-count variants.
            q_input = q_all.new_zeros(
                q_all.shape[0], q_all.shape[1], target_q_heads, q_all.shape[3]
            )
            q_input[:, :, :num_q_heads, :] = q_all
        else:
            q_input = q_all

        kv_cache = kv_cache.view(-1, self.real_page_size, 1, self.kv_cache_dim)
        assert self.real_page_size == 64, "only page size 64 is supported"

        if _is_hcu and self.dsa_index_kpool > 1:
            # KPool appends up to kpool-1 uncompressed tail tokens after the
            # selected history tokens. HCU FlashMLA sparse decode additionally
            # requires the indices width to be a TOPK_BLOCK_SIZE multiple; pad
            # with -1 sentinels so the tail is preserved without adding work.
            topk_block_size = 64
            width = page_table_1.shape[-1]
            padded_width = (
                (width + topk_block_size - 1) // topk_block_size
            ) * topk_block_size
            if padded_width != width:
                page_table_1 = torch.nn.functional.pad(
                    page_table_1, (0, padded_width - width), value=-1
                )
        indices = page_table_1.unsqueeze(1)
        expected_indices_width = self.dsa_index_topk
        if self.dsa_index_kpool > 1:
            expected_indices_width = self.dsa_index_topk + self.dsa_index_kpool - 1
            if _is_hcu:
                topk_block_size = 64
                expected_indices_width = (
                    (expected_indices_width + topk_block_size - 1)
                    // topk_block_size
                    * topk_block_size
                )
        assert indices.shape[-1] == expected_indices_width, (
            "FlashMLA decode indices width mismatch: "
            f"got {indices.shape[-1]}, expected {expected_indices_width} "
            f"(topk={self.dsa_index_topk}, kpool={self.dsa_index_kpool})"
        )

        # DP/CP padding may append zero-length target-verify rows.  The CUDA
        # FlashMLA implementation tolerates them, but the current HCU kernel
        # can access invalid memory when cache_seqlens=0 and the corresponding
        # index row is all -1.  Follow the strip/re-pad convention used by the
        # sglang-zp KDA backend: run the kernel only for real tokens, recompute
        # its schedule for that exact batch, then restore the caller's shape.
        n_total = q_input.shape[0]
        n_valid = (
            int(sum(forward_batch.extend_seq_lens_cpu))
            if forward_batch.extend_seq_lens_cpu is not None
            else forward_batch.extend_num_tokens
        )
        if n_valid is None:
            n_valid = self._decode_dp_padding_num_valid(forward_batch, n_total)
        needs_repad = _is_hcu and n_valid is not None and 0 <= n_valid < n_total
        flashmla_metadata = metadata.flashmla_metadata
        if needs_repad:
            q_input = q_input[:n_valid]
            indices = indices[:n_valid]
            cache_seqlens = cache_seqlens[:n_valid]
            if n_valid > 0 and not use_lightop_flat_sparse_fwd:
                flashmla_metadata = self._compute_flashmla_metadata(
                    cache_seqlens=cache_seqlens,
                    seq_len_q=1,
                    is_fp8_kvcache=(
                        False
                        if use_lightop_decode_gather
                        else self.dsa_kv_cache_store_fp8
                    ),
                )

        flashmla_is_fp8_kvcache = self.dsa_kv_cache_store_fp8
        if use_lightop_decode_gather and (not needs_repad or n_valid > 0):
            gathered_kv, compact_indices = self._get_lightop_decode_workspaces(
                q_input.shape[0]
            )
            physical_token_ids = indices[:, 0, :]
            assert (
                physical_token_ids.shape[1]
                == gathered_kv.shape[1]
                == compact_indices.shape[1]
                == self._lightop_decode_gather_width
            ), "LightOp decode gather width mismatch after KPool padding"
            self._lightop_decode_gather(
                lightop_kv_cache,
                physical_token_ids,
                gathered_kv,
                cache_seqlens,
                compact_indices,
            )

            # FlashMLA must index the compact gathered cache, not the original
            # physical cache.  Preserve every -1 sentinel: replacing it with a
            # positive index to a zero row would change the softmax denominator.
            indices = compact_indices.unsqueeze(1)
            if use_lightop_flat_sparse_fwd:
                kv_cache = gathered_kv.view(-1, 1, self._lightop_decode_head_dim)
            else:
                kv_cache = gathered_kv.view(-1, 64, 1, self._lightop_decode_head_dim)
            flashmla_is_fp8_kvcache = False

        # HCU get_mla_metadata returns a lazy FlashMLASchedMeta whose config is
        # fixed by its first FlashMLA invocation. Normally every layer in this
        # metadata lifetime selects the same path; refresh defensively if an
        # earlier invocation initialized it for the opposite cache dtype.
        if not use_lightop_flat_sparse_fwd:
            hcu_sched_meta = flashmla_metadata.flashmla_metadata
            hcu_sched_config = getattr(hcu_sched_meta, "config", None)
            if (
                _is_hcu
                and getattr(hcu_sched_meta, "have_initialized", False)
                and hcu_sched_config is not None
                and hcu_sched_config.is_fp8_kvcache != flashmla_is_fp8_kvcache
            ):
                flashmla_metadata = self._compute_flashmla_metadata(
                    cache_seqlens=cache_seqlens,
                    seq_len_q=1,
                    is_fp8_kvcache=flashmla_is_fp8_kvcache,
                )

        if needs_repad and n_valid == 0:
            o = q_input.new_zeros((0, 1, target_q_heads, v_head_dim))
        elif use_lightop_flat_sparse_fwd:
            o, _, _ = flash_mla_sparse_fwd(
                q=q_input[:, 0],
                kv=kv_cache,
                indices=indices,
                sm_scale=sm_scale,
                d_v=v_head_dim,
            )
            o = o.unsqueeze(1)
        else:
            o, _ = flash_mla_with_kvcache(
                q=q_input,
                k_cache=kv_cache,
                cache_seqlens=cache_seqlens,
                head_dim_v=v_head_dim,
                tile_scheduler_metadata=flashmla_metadata.flashmla_metadata,
                num_splits=flashmla_metadata.num_splits,
                softmax_scale=sm_scale,
                indices=indices,
                # doc says it is not used, but if pass in None then error
                block_table=torch.empty(
                    (q_input.shape[0], 0), dtype=torch.int32, device=q_input.device
                ),
                is_fp8_kvcache=flashmla_is_fp8_kvcache,
            )

        if needs_repad:
            full_o = o.new_zeros((n_total, *o.shape[1:]))
            full_o[:n_valid] = o
            o = full_o

        if target_q_heads != num_q_heads:
            # Head padding leaves a gap of ``target_q_heads`` between token
            # rows.  A plain slice therefore returns a non-contiguous view
            # whose token stride still describes all padded heads.  The next
            # MLA stage feeds this tensor directly to HCU torch.bmm; materialize
            # the compact [tokens, 1, real_heads, d_v] layout first.
            o = o[:, :, :num_q_heads, :].contiguous()

        return o

    def _forward_standard_mha(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        metadata: DSAMetadata,
    ) -> torch.Tensor:
        """Standard MHA using FlashAttention varlen for MHA_ONE_SHOT mode."""
        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        k = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

        # MHA_ONE_SHOT: k/v include all tokens (prefix + current)
        cu_seqlens_q = metadata.cu_seqlens_q
        cu_seqlens_k = metadata.cu_seqlens_k
        max_seqlen_k = metadata.max_seq_len_k
        causal = True

        # Verify batch sizes match (length of cu_seqlens should be batch_size + 1)
        assert len(cu_seqlens_q) == len(cu_seqlens_k), (
            f"batch_size mismatch: cu_seqlens_q has {len(cu_seqlens_q) - 1} requests, "
            f"cu_seqlens_k has {len(cu_seqlens_k) - 1} requests"
        )

        # Use TRTLLm ragged attention for SM100 (Blackwell/B200) to avoid FA4 accuracy issues
        if self.device_sm_major >= 10:
            import flashinfer

            seq_lens = metadata.cache_seqlens_int32
            return flashinfer.prefill.trtllm_ragged_attention_deepseek(
                query=q,
                key=k,
                value=v,
                workspace_buffer=self.workspace_buffer,
                seq_lens=seq_lens,
                max_q_len=metadata.max_seq_len_q,
                max_kv_len=max_seqlen_k,
                bmm1_scale=layer.scaling,
                bmm2_scale=1.0,
                o_sf_scale=1.0,
                batch_size=forward_batch.batch_size,
                window_left=-1,
                cum_seq_lens_q=cu_seqlens_q,
                cum_seq_lens_kv=cu_seqlens_k,
                enable_pdl=False,
                is_causal=causal,
                return_lse=False,
                skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR.get(),
            )

        # Use FA3 for SM90 (Hopper/H200)
        return flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=metadata.max_seq_len_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=layer.scaling,
            causal=causal,
        )

    def _forward_tilelang(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        page_table_1: torch.Tensor,
        sm_scale: float,
    ) -> torch.Tensor:
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

        return tilelang_sparse_fwd(
            q=q_all,
            kv=kv_cache,
            indices=page_table_1.unsqueeze(1),
            sm_scale=sm_scale,
            d_v=v_head_dim,
        )

    def _forward_aiter(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        page_table_1: torch.Tensor,
        layer: RadixAttention,
        metadata: DSAMetadata,
        bs: int,
    ) -> torch.Tensor:
        q = q_all.reshape(-1, layer.tp_q_head_num * layer.head_dim)

        if layer.head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if self.need_pad_heads:
            q_kernel = q.view(
                -1, layer.tp_q_head_num, layer.head_dim
            ).repeat_interleave(self.head_repeat_factor, dim=1)
            o_kernel = q.new_empty(
                (
                    q.shape[0],
                    layer.tp_q_head_num * self.head_repeat_factor,
                    layer.v_head_dim,
                )
            )
        else:
            q_kernel = q.view(-1, layer.tp_q_head_num, layer.head_dim)
            o_kernel = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        kv_indptr = self.kv_indptr

        non_minus1_mask = page_table_1 != -1
        non_minus1_counts = non_minus1_mask.sum(dim=1)
        kv_indptr[1 : bs + 1] = torch.cumsum(non_minus1_counts, dim=0)

        kv_indices = self.kv_indices
        get_valid_kv_indices(page_table_1, kv_indptr, kv_indices, bs)

        mla_decode_fwd(
            q_kernel,
            kv_cache.view(-1, 1, 1, layer.head_dim),
            o_kernel,
            metadata.cu_seqlens_q,
            kv_indptr,
            kv_indices,
            metadata.cu_seqlens_q,
            metadata.max_seq_len_q,
            sm_scale=layer.scaling,
            logit_cap=layer.logit_cap,
        )

        if self.need_pad_heads:
            o = o_kernel[:, :: self.head_repeat_factor, :]

        return o

    def _forward_aiter_extend(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        page_table_1: torch.Tensor,
        layer: RadixAttention,
    ) -> torch.Tensor:
        num_tokens = q_all.shape[0]
        q = q_all.reshape(-1, layer.tp_q_head_num * layer.head_dim)

        if layer.head_dim != layer.v_head_dim:
            o = q.new_empty((num_tokens, layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if self.need_pad_heads:
            q_kernel = q.view(
                -1, layer.tp_q_head_num, layer.head_dim
            ).repeat_interleave(self.head_repeat_factor, dim=1)
            o_kernel = q.new_empty(
                (
                    num_tokens,
                    layer.tp_q_head_num * self.head_repeat_factor,
                    layer.v_head_dim,
                )
            )
        else:
            q_kernel = q.view(-1, layer.tp_q_head_num, layer.head_dim)
            o_kernel = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        non_minus1_mask = page_table_1 != -1
        non_minus1_counts = non_minus1_mask.sum(dim=1)

        kv_indptr = torch.zeros(num_tokens + 1, dtype=torch.int32, device=self.device)
        kv_indptr[1:] = torch.cumsum(non_minus1_counts, dim=0)

        # Allocate kv_indices with upper-bound size (num_tokens * topk)
        topk = page_table_1.shape[1]
        kv_indices = torch.zeros(
            num_tokens * topk, dtype=torch.int32, device=self.device
        )

        # Use get_valid_kv_indices kernel to extract valid indices
        get_valid_kv_indices(page_table_1, kv_indptr, kv_indices, num_tokens)

        # Build cu_seqlens_q for extend: each token is treated as seq_len_q=1
        cu_seqlens_q = torch.arange(
            0, num_tokens + 1, dtype=torch.int32, device=self.device
        )
        # TODO support more forward_mode
        mla_decode_fwd(
            q_kernel,
            kv_cache.view(-1, 1, 1, layer.head_dim),
            o_kernel,
            cu_seqlens_q,
            kv_indptr,
            kv_indices,
            cu_seqlens_q,
            1,  # max_seq_len_q = 1 for per-token attention
            sm_scale=layer.scaling,
            logit_cap=layer.logit_cap,
        )

        if self.need_pad_heads:
            o = o_kernel[:, :: self.head_repeat_factor, :]

        return o

    def _forward_trtllm(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        seq_lens: torch.Tensor,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
        is_prefill: bool = False,
    ) -> torch.Tensor:
        """Forward using TRT-LLM sparse MLA kernel."""
        import flashinfer.decode

        metadata = self.forward_metadata

        merge_query = q_rope is not None
        if self.kv_cache_dtype == torch.float8_e4m3fn:
            # For FP8 path, we quantize the query and rope parts and merge them into a single tensor
            # Note: rope application in deepseek_v2.py:forward_absorb_prepare is skipped for FP8 decode path of this trtllm_mla backend
            assert q_rope is not None, "For FP8 path q_rope should not be None."
            assert k_rope is not None, "For FP8 path k_rope should not be None."
            assert cos_sin_cache is not None, (
                "For FP8 path cos_sin_cache should not be None."
            )

            q, k, k_rope = mla_quantize_and_rope_for_fp8(
                q,
                q_rope,
                k.squeeze(1),
                k_rope.squeeze(1),
                forward_batch.positions,
                cos_sin_cache,
                is_neox,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
            )
            merge_query = False

            # Save KV cache if requested
        if save_kv_cache:
            assert k is not None and k_rope is not None, (
                "For populating trtllm_mla kv cache, both k_nope and k_rope should be not None."
            )
            cache_loc = (
                forward_batch.out_cache_loc
                if not layer.is_cross_attention
                else forward_batch.encoder_out_cache_loc
            )
            self.token_to_kv_pool.set_mla_kv_buffer(layer, cache_loc, k, k_rope)

        k_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_cache = k_cache.view(-1, self.real_page_size, self.kv_cache_dim).unsqueeze(1)

        if merge_query:
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope_reshaped = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
            q_all = concat_mla_absorb_q_general(q_nope, q_rope_reshaped)
        else:
            q_all = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        # Align topk_indices with q dimensions
        if topk_indices is not None:
            topk_indices = self._pad_topk_indices(topk_indices, q.shape[0])

        if self._use_fused_topk(forward_batch):
            page_table_1 = topk_indices
        elif is_prefill:
            extend_lens_cpu = self._get_topk_transform_lens(
                forward_batch,
                metadata,
                topk_indices.shape[0],
            )
            page_table_1 = self._transform_prefill_topk_indices(
                forward_batch,
                metadata.page_table_1,
                topk_indices,
                extend_lens_cpu,
                page_table_row_indices=metadata.token_to_batch_idx,
            )
        else:
            page_table_1 = self._transform_decode_topk_indices(
                forward_batch,
                metadata.page_table_1,
                topk_indices,
            )

        page_table_1 = self.token_to_kv_pool.translate_main_kv_loc_to_compact(
            page_table_1
        )

        q_scale = 1.0
        k_scale = (
            layer.k_scale_float
            if getattr(layer, "k_scale_float", None) is not None
            else 1.0
        )
        bmm1_scale = q_scale * k_scale * layer.scaling

        batch_size = page_table_1.shape[0]
        _, num_heads, head_dim = q_all.shape

        q = q_all.view(batch_size, 1, num_heads, head_dim)
        kv = kv_cache.view(-1, 1, self.real_page_size, self.kv_cache_dim)
        block_tables = page_table_1.unsqueeze(1)
        seq_lens = metadata.cache_seqlens_int32 if seq_lens is None else seq_lens

        out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=q,
            kv_cache=kv,
            workspace_buffer=self.workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=metadata.max_seq_len_k,
            sparse_mla_top_k=self.dsa_index_topk,
            bmm1_scale=bmm1_scale,
            backend="trtllm-gen",
            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
        )
        # Output: [batch, q_len=1, heads, v_dim] -> [batch, heads, v_dim]
        return out.squeeze(1)

    def _pad_topk_indices(
        self, topk_indices: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        current_tokens = topk_indices.shape[0]
        if current_tokens == num_tokens:
            return topk_indices

        assert current_tokens <= num_tokens, (
            f"topk_indices rows ({current_tokens}) > num_tokens ({num_tokens}); "
            "this indicates a mismatch between indexer output and q layout."
        )

        pad_size = num_tokens - current_tokens
        padding = torch.full(
            (pad_size, topk_indices.shape[1]),
            -1,
            dtype=topk_indices.dtype,
            device=topk_indices.device,
        )
        return torch.cat([topk_indices, padding], dim=0)

    @staticmethod
    def _decode_dp_padding_num_valid(
        forward_batch: ForwardBatch, num_rows: int
    ) -> Optional[int]:
        """Return real topk=1 decode rows when eager DP padding grew the batch."""
        if not effective_forward_mode(forward_batch).is_decode_or_idle():
            return None
        spec_info = getattr(forward_batch, "spec_info", None)
        if getattr(spec_info, "num_tokens_per_req", None) != 1:
            return None
        planned_rows = getattr(
            forward_batch, "forward_metadata_planned_num_tokens", None
        )
        planned_batch_size = getattr(forward_batch, "forward_metadata_planned_bs", None)
        original_batch_size = getattr(forward_batch, "_original_batch_size", None)
        if (
            isinstance(planned_rows, int)
            and 0 <= planned_rows < num_rows
            and planned_batch_size == planned_rows
            and original_batch_size == planned_rows
        ):
            return planned_rows
        return None

    def _transform_decode_topk_indices(
        self,
        forward_batch: ForwardBatch,
        page_table: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Localize logical decode indices while preserving DP padding rows.

        Multi-step eager draft metadata is planned before MLP-sync padding.
        When DP padding later grows the forward batch, ``page_table`` still
        contains only the planned request rows while ``topk_indices`` follows
        the padded q layout.  Transform the planned rows and keep synthetic
        rows invalid instead of indexing a non-existent request page table.
        """
        page_table_rows = page_table.shape[0]
        topk_rows = topk_indices.shape[0]
        if page_table_rows == topk_rows:
            return transform_index_page_table_decode(
                page_table=page_table,
                topk_indices=topk_indices,
                page_size=1,
            )

        num_valid = self._decode_dp_padding_num_valid(forward_batch, topk_rows)
        uses_logical_indices = uses_logical_mtp_topk_indices(forward_batch)
        assert (
            _is_hcu
            and self.dsa_decode_impl == "flashmla_kv"
            and uses_logical_indices
            and num_valid == page_table_rows
        ), (
            "decode page-table rows only may differ from top-k rows after "
            "topk=1 MLP-sync DP padding on the HCU flashmla_kv logical-index "
            "fallback: "
            f"page_table_rows={page_table_rows}, topk_rows={topk_rows}, "
            "planned_rows="
            f"{getattr(forward_batch, 'forward_metadata_planned_num_tokens', None)}, "
            "planned_batch_size="
            f"{getattr(forward_batch, 'forward_metadata_planned_bs', None)}, "
            f"original_batch_size={getattr(forward_batch, '_original_batch_size', None)}, "
            "num_tokens_per_req="
            f"{getattr(getattr(forward_batch, 'spec_info', None), 'num_tokens_per_req', None)}, "
            f"dsa_decode_impl={self.dsa_decode_impl}, uses_logical_indices={uses_logical_indices}"
        )

        return transform_index_page_table_decode(
            page_table=page_table,
            topk_indices=topk_indices,
            page_size=1,
        )

    def _transform_prefill_topk_indices(
        self,
        forward_batch: ForwardBatch,
        page_table: torch.Tensor,
        topk_indices: torch.Tensor,
        extend_lens_cpu: List[int],
        page_table_row_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Localize prefill indices while preserving synthetic padded q rows."""
        described_rows = sum(extend_lens_cpu)
        topk_rows = topk_indices.shape[0]
        if described_rows == topk_rows:
            return transform_index_page_table_prefill(
                page_table=page_table,
                topk_indices=topk_indices,
                extend_lens_cpu=extend_lens_cpu,
                page_size=1,
            )

        assert described_rows <= topk_rows, (
            "prefill page-table lengths describe more rows than top-k has: "
            f"described_rows={described_rows}, topk_rows={topk_rows}, "
            f"page_table_rows={page_table.shape[0]}, "
            f"forward_mode={forward_batch.forward_mode}, "
            f"effective_forward_mode={effective_forward_mode(forward_batch)}, "
            f"planned_rows={getattr(forward_batch, 'forward_metadata_planned_num_tokens', None)}, "
            f"planned_batch_size={getattr(forward_batch, 'forward_metadata_planned_bs', None)}"
        )

        localized = (
            transform_index_page_table_prefill(
                page_table=page_table,
                topk_indices=topk_indices[:described_rows],
                extend_lens_cpu=extend_lens_cpu,
                page_size=1,
            )
            if described_rows > 0
            else topk_indices.new_empty((0, *topk_indices.shape[1:]), dtype=torch.int32)
        )
        padding = localized.new_full(
            (topk_rows - described_rows, *topk_indices.shape[1:]), -1
        )
        return torch.cat([localized, padding], dim=0)

    def _pad_topk_indices_offset(
        self, topk_indices_offset: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        current_tokens = topk_indices_offset.shape[0]
        if current_tokens == num_tokens:
            return topk_indices_offset

        assert current_tokens <= num_tokens, (
            f"topk_indices_offset rows ({current_tokens}) > num_tokens "
            f"({num_tokens}); this indicates a mismatch between DSA metadata "
            "and q layout."
        )
        padding = torch.zeros(
            (num_tokens - current_tokens, *topk_indices_offset.shape[1:]),
            dtype=topk_indices_offset.dtype,
            device=topk_indices_offset.device,
        )
        return torch.cat([topk_indices_offset, padding], dim=0)

    def _get_aligned_topk_indices_offset(
        self,
        forward_batch: ForwardBatch,
        metadata: DSAMetadata,
        num_tokens: int,
    ) -> torch.Tensor:
        topk_indices_offset = metadata.topk_indices_offset
        assert topk_indices_offset is not None

        if dsa_use_prefill_cp(forward_batch) and is_dsa_prefill_cp_in_seq_split():
            # in-seq-split currently supports one request. Its indexer emits
            # only this rank's q rows, while metadata keeps the full-request
            # offset vector. Every local row shares the request's base offset.
            assert metadata.page_table_1.shape[0] == 1
            return topk_indices_offset[:1].expand(num_tokens)

        assert topk_indices_offset.shape[0] == metadata.dsa_seqlens_expanded.shape[0], (
            "topk_indices_offset must have one row per logical DSA "
            f"query token: {topk_indices_offset.shape[0]} != "
            f"{metadata.dsa_seqlens_expanded.shape[0]}"
        )
        # CP/DP alignment pads q and top-k to physical rows after the indexer
        # has emitted one offset per logical token. Extend offsets with neutral
        # zeros; matching padded top-k rows remain -1 under the caller's mask.
        return self._pad_topk_indices_offset(topk_indices_offset, num_tokens)

    @staticmethod
    def _get_topk_transform_lens(
        forward_batch: ForwardBatch,
        metadata: DSAMetadata,
        num_tokens: int,
    ) -> List[int]:
        if dsa_use_prefill_cp(forward_batch) and is_dsa_prefill_cp_in_seq_split():
            # in-seq-split is single-request today, so all rank-local q rows
            # map through the same request-local page-table row.
            assert metadata.page_table_1.shape[0] == 1
            return [num_tokens]

        assert metadata.dsa_extend_seq_lens_list is not None
        return metadata.dsa_extend_seq_lens_list

    @staticmethod
    def _get_dsa_page_table_transform_lens(
        forward_mode: ForwardMode,
        extend_seq_lens_cpu: List[int],
        page_table_rows: int,
    ) -> List[int]:
        if forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2():
            expanded_rows = sum(extend_seq_lens_cpu)
            assert expanded_rows == page_table_rows, (
                "Speculative page-table rows must equal the expanded request "
                f"lengths: rows={page_table_rows}, expanded_rows={expanded_rows}."
            )
            # Speculative modes repeat each request's page-table row once per
            # physical query token. The unfused PAGED transform must consume
            # that representation one row at a time. Keeping the original
            # per-request lengths here would describe the pre-repeat layout.
            return [1] * page_table_rows
        return extend_seq_lens_cpu

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for sequence length in CUDA graph."""
        return 1

    def set_dsa_prefill_impl(self, forward_batch: Optional[ForwardBatch] = None):
        """
        Decide all attention prefill dispatch strategies for this batch.
        """
        from sglang.srt.utils import get_device_sm, is_blackwell

        # Decide MHA vs MLA
        if forward_batch and forward_batch.forward_mode.is_extend_without_speculative():
            # Check if sequence meets criteria for MHA_ONE_SHOT
            assert forward_batch.seq_lens_cpu is not None
            max_kv_len = forward_batch.seq_lens_cpu.max().item()
            sum_seq_lens = sum(forward_batch.seq_lens_cpu)
            device_sm = get_device_sm()

            # Requirements: H200/B200, short sequences, supported dtype, fits in chunk
            self.use_mha = (
                (
                    device_sm == 90 or (device_sm >= 100 and device_sm < 110)
                )  # SM90/SM100 only
                and max_kv_len
                <= envs.SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.get()  # Short enough for MHA
                and self.token_to_kv_pool.dtype in [torch.bfloat16, torch.float8_e4m3fn]
                and sum_seq_lens
                <= forward_batch.get_max_chunk_capacity()  # Fits in chunk
                and (not is_dsa_enable_prefill_cp())  # CP not enabled
                and (self.hisparse_coordinator is None)
            )
        else:
            self.use_mha = False  # Decode/verify always use MLA

        # Set MLA implementation only if not using MHA
        if not self.use_mha and self.enable_auto_select_prefill_impl:
            if self.dsa_kv_cache_store_fp8:
                # The HCU FlashMLA paged FP8 entry does not support the compact
                # no-RoPE layout (q=512, packed KV row=528). Keep prefill on the
                # sparse BF16 path; decode uses LightOp to gather and dequantize.
                if _is_hcu and self.qk_rope_head_dim == 0 and self.kv_cache_dim == 528:
                    self.dsa_prefill_impl = "flashmla_sparse"
                    return
                if (
                    (is_blackwell() or _is_hcu)
                    and forward_batch is not None
                    and forward_batch.forward_mode == ForwardMode.EXTEND
                ):
                    total_kv_tokens = forward_batch.seq_lens_sum
                    total_q_tokens = forward_batch.extend_num_tokens
                    # Heuristic based on benchmarking flashmla_kv vs flashmla_sparse + dequantize_k_cache_paged
                    if total_kv_tokens < total_q_tokens * 512:
                        self.dsa_prefill_impl = "flashmla_sparse"
                        return
                self.dsa_prefill_impl = "flashmla_kv"
            else:
                # bf16 kv cache
                self.dsa_prefill_impl = "flashmla_sparse"

    def get_topk_transform_method(
        self, forward_mode: Optional[ForwardMode] = None
    ) -> TopkTransformMethod:
        """
        SGLANG_DSA_FUSE_TOPK controls whether to fuse the topk transform into the topk kernel.
        This method is used to select the topk transform method which can be fused or unfused.
        """
        if (
            # disable for MTP
            self.dsa_kv_cache_store_fp8
            and self.dsa_prefill_impl == "flashmla_sparse"
            and forward_mode
            in (
                ForwardMode.EXTEND,
                ForwardMode.MIXED,
                ForwardMode.SPLIT_PREFILL,
            )
        ):
            topk_transform_method = TopkTransformMethod.RAGGED
        else:
            topk_transform_method = TopkTransformMethod.PAGED
        return topk_transform_method

    def _force_unfused_topk(self, forward_batch: ForwardBatch) -> bool:
        forward_mode = effective_forward_mode(forward_batch)
        # A PD prefill draft backend emits request-relative logical positions
        # for the wire. Its decode counterpart remaps the seed once at batch
        # assembly, so logical MTP flags alone must not disable fused TopK.
        if not self.use_fused_topk:
            return True

        if self.hisparse_coordinator is not None and forward_mode.is_decode_or_idle():
            return True

        return (
            _is_hcu
            and self._is_glm5_next
            and self.dsa_index_kpool <= 1
            and (
                forward_mode.is_decode_or_idle()
                or forward_mode.is_target_verify()
                or forward_mode.is_draft_extend_v2()
            )
        )

    def _use_fused_topk(self, forward_batch: ForwardBatch) -> bool:
        return self.use_fused_topk and not self._force_unfused_topk(forward_batch)

    def get_indexer_metadata(
        self, layer_id: int, forward_batch: ForwardBatch
    ) -> DSAIndexerMetadata:
        forward_mode = effective_forward_mode(forward_batch)
        force_unfused = self._force_unfused_topk(forward_batch)
        return DSAIndexerMetadata(
            attn_metadata=self.forward_metadata,
            topk_transform_method=self.get_topk_transform_method(forward_mode),
            paged_mqa_schedule_metadata=self.forward_metadata.paged_mqa_schedule_metadata,
            force_unfused_topk=force_unfused,
        )

    def _compute_flashmla_metadata(
        self,
        cache_seqlens: torch.Tensor,
        seq_len_q: int,
        is_fp8_kvcache: Optional[bool] = None,
    ):
        if not _is_hcu:
            from sgl_kernel.flash_mla import get_mla_metadata
        else:
            from flash_mla.flash_mla_interface import get_mla_metadata

        num_heads_q = self.flashmla_kv_num_q_heads
        if is_fp8_kvcache is None:
            is_fp8_kvcache = self.dsa_kv_cache_store_fp8

        flashmla_metadata, num_splits = get_mla_metadata(
            cache_seqlens=cache_seqlens,
            # TODO doc says `num_q_tokens_per_q_seq * num_heads_q // num_heads_k`
            #      but the name looks like need seq_len_q?
            num_q_tokens_per_head_k=seq_len_q * num_heads_q // 1,
            num_heads_k=1,
            num_heads_q=num_heads_q,
            is_fp8_kvcache=is_fp8_kvcache,
            topk=self.dsa_index_topk,
        )
        return DSAFlashMLAMetadata(
            flashmla_metadata=flashmla_metadata,
            num_splits=flashmla_metadata.num_splits,
        )


class NativeSparseAttnMultiStepBackend:
    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
        seed_dsa_topk_from_draft_extend: bool = False,
    ):
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                NativeSparseAttnBackend(
                    model_runner,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                    seed_dsa_topk_from_draft_extend=seed_dsa_topk_from_draft_extend,
                )
            )
        self.dsa_index_kpool = self.attn_backends[0].dsa_index_kpool

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def get_cuda_graph_seq_len_fill_value(self):
        return self.attn_backends[0].get_cuda_graph_seq_len_fill_value()

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        if in_capture:
            from sglang.srt.model_executor.forward_batch_info import build_inner_fb_view

            inner_fb = build_inner_fb_view(
                forward_batch,
                bs=forward_batch.batch_size,
                forward_mode=ForwardMode.DECODE,
            )
            for backend in self.attn_backends:
                backend.init_forward_metadata_out_graph(inner_fb, in_capture=True)
        else:
            self._init_decode_replay_cuda_graph(forward_batch, forward_batch.batch_size)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        for backend in self.attn_backends:
            backend.init_forward_metadata_in_graph(forward_batch)

    def _init_decode_replay_cuda_graph(self, forward_batch: ForwardBatch, bs: int):
        if envs.SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA.get():
            # Precompute metadata once (shared across all backends)
            precomputed = self.attn_backends[0]._precompute_replay_metadata(
                bs=bs,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

            # Use multi-backend fused copy when we have 3 or more backends
            # This is 3x faster than calling the single-backend copy 3 times
            if self.speculative_num_steps > 3 and not _disable_dsa_multi_replay_opt():
                try:
                    from sglang.kernels.ops.attention.fused_metadata_copy import (
                        fused_metadata_copy_multi_cuda,
                    )

                    metadata0 = self.attn_backends[0].decode_cuda_graph_metadata[bs]
                    metadata1 = self.attn_backends[1].decode_cuda_graph_metadata[bs]
                    metadata2 = self.attn_backends[2].decode_cuda_graph_metadata[bs]

                    # Set dsa_prefill_impl for first 3 backends (required by the method)
                    for i in range(3):
                        self.attn_backends[i].set_dsa_prefill_impl(forward_batch=None)

                    # Prepare FlashMLA tensors if they are plain tensors. HCU
                    # FlashMLA metadata can be backend objects, which the
                    # fused metadata-copy kernel cannot accept as optional
                    # tensor arguments.
                    fused_flashmla_metadata = _can_fuse_flashmla_metadata(
                        precomputed.flashmla_metadata,
                        metadata0.flashmla_metadata,
                        metadata1.flashmla_metadata,
                        metadata2.flashmla_metadata,
                    )
                    flashmla_num_splits_src = None
                    flashmla_metadata_src = None
                    flashmla_num_splits_dst0 = None
                    flashmla_num_splits_dst1 = None
                    flashmla_num_splits_dst2 = None
                    flashmla_metadata_dst0 = None
                    flashmla_metadata_dst1 = None
                    flashmla_metadata_dst2 = None

                    if fused_flashmla_metadata:
                        flashmla_num_splits_src = (
                            precomputed.flashmla_metadata.num_splits
                        )
                        flashmla_metadata_src = (
                            precomputed.flashmla_metadata.flashmla_metadata
                        )
                        flashmla_num_splits_dst0 = (
                            metadata0.flashmla_metadata.num_splits
                        )
                        flashmla_num_splits_dst1 = (
                            metadata1.flashmla_metadata.num_splits
                        )
                        flashmla_num_splits_dst2 = (
                            metadata2.flashmla_metadata.num_splits
                        )
                        flashmla_metadata_dst0 = (
                            metadata0.flashmla_metadata.flashmla_metadata
                        )
                        flashmla_metadata_dst1 = (
                            metadata1.flashmla_metadata.flashmla_metadata
                        )
                        flashmla_metadata_dst2 = (
                            metadata2.flashmla_metadata.flashmla_metadata
                        )

                    # Call the multi-backend fused kernel for first 3 backends
                    fused_metadata_copy_multi_cuda(
                        # Source tensors
                        precomputed.cache_seqlens,
                        precomputed.cu_seqlens_k,
                        precomputed.page_indices,
                        precomputed.dsa_cache_seqlens,
                        precomputed.dsa_cu_seqlens_k,
                        precomputed.real_page_table,
                        flashmla_num_splits_src,
                        flashmla_metadata_src,
                        # Destination tensors for backend 0
                        metadata0.cache_seqlens_int32,
                        metadata0.cu_seqlens_k,
                        metadata0.page_table_1,
                        metadata0.dsa_cache_seqlens_int32,
                        metadata0.dsa_cu_seqlens_k,
                        (
                            metadata0.real_page_table
                            if precomputed.real_page_table is not None
                            else None
                        ),
                        flashmla_num_splits_dst0,
                        flashmla_metadata_dst0,
                        # Destination tensors for backend 1
                        metadata1.cache_seqlens_int32,
                        metadata1.cu_seqlens_k,
                        metadata1.page_table_1,
                        metadata1.dsa_cache_seqlens_int32,
                        metadata1.dsa_cu_seqlens_k,
                        (
                            metadata1.real_page_table
                            if precomputed.real_page_table is not None
                            else None
                        ),
                        flashmla_num_splits_dst1,
                        flashmla_metadata_dst1,
                        # Destination tensors for backend 2
                        metadata2.cache_seqlens_int32,
                        metadata2.cu_seqlens_k,
                        metadata2.page_table_1,
                        metadata2.dsa_cache_seqlens_int32,
                        metadata2.dsa_cu_seqlens_k,
                        (
                            metadata2.real_page_table
                            if precomputed.real_page_table is not None
                            else None
                        ),
                        flashmla_num_splits_dst2,
                        flashmla_metadata_dst2,
                        # Parameters
                        bs,
                        precomputed.max_len,
                        precomputed.seqlens_expanded_size,
                    )

                    if (
                        precomputed.flashmla_metadata is not None
                        and not fused_flashmla_metadata
                    ):
                        size = precomputed.seqlens_expanded_size
                        for metadata in (metadata0, metadata1, metadata2):
                            flashmla_metadata = metadata.flashmla_metadata.slice(
                                slice(0, size + 1)
                            )
                            flashmla_metadata.copy_(precomputed.flashmla_metadata)

                    # The multi-copy kernel only handles dense DSA fields.
                    # Refresh backend-local KPool state for the fused
                    # destinations just like the single-backend replay path.
                    kpool_multi_updated = (
                        self.attn_backends[0].dsa_index_kpool > 1
                        and precomputed.req_pool_indices is not None
                    )
                    if kpool_multi_updated:
                        metadata3 = (
                            self.attn_backends[3].decode_cuda_graph_metadata[bs]
                            if self.speculative_num_steps > 4
                            else None
                        )
                        write_start = precomputed.cache_seqlens[:bs] - 1
                        if write_start.dtype != torch.int32:
                            write_start = write_start.to(torch.int32)
                        real_page_table = (
                            precomputed.real_page_table
                            if precomputed.real_page_table is not None
                            else metadata0.real_page_table
                        )
                        _update_kpool_write_plan_multi_decode_impl(
                            metadata0,
                            metadata1,
                            metadata2,
                            metadata3,
                            write_start=write_start,
                            req_pool_indices=precomputed.req_pool_indices,
                            real_page_table=real_page_table,
                            pool_size=self.attn_backends[0].dsa_index_kpool,
                            real_page_size=self.attn_backends[0].real_page_size,
                        )
                    for i in range(3):
                        backend = self.attn_backends[i]
                        backend_metadata = backend.decode_cuda_graph_metadata[bs]
                        backend._update_pooled_paged_mqa_metadata(
                            metadata=backend_metadata,
                            seqlens_32=backend_metadata.cache_seqlens_int32,
                            forward_mode=ForwardMode.DECODE,
                        )
                        if not kpool_multi_updated:
                            backend._update_kpool_write_plan_from_precomputed(
                                metadata=backend_metadata,
                                precomputed=precomputed,
                            )
                        backend.forward_metadata = backend_metadata

                    # Copy remaining backends one by one (if > 3 backends)
                    for i in range(3, self.speculative_num_steps - 1):
                        self.attn_backends[
                            i
                        ].init_forward_metadata_replay_cuda_graph_from_precomputed(
                            bs=bs,
                            precomputed=precomputed,
                            forward_mode=ForwardMode.DECODE,
                            skip_kpool_write_plan_update=(
                                kpool_multi_updated and i == 3
                            ),
                        )
                except (ImportError, Exception) as e:
                    # Fallback to loop if multi-backend kernel not available or fails
                    if isinstance(e, ImportError):
                        print_warning_once(
                            "Multi-backend fused metadata copy kernel not available, falling back to loop."
                        )
                    else:
                        print_warning_once(
                            f"Multi-backend fused metadata copy kernel failed with error: {e}, falling back to loop."
                        )
                    for i in range(self.speculative_num_steps - 1):
                        self.attn_backends[
                            i
                        ].init_forward_metadata_replay_cuda_graph_from_precomputed(
                            bs=bs,
                            precomputed=precomputed,
                            forward_mode=ForwardMode.DECODE,
                        )
            else:
                # Less than 3 backends: copy to each backend individually
                for i in range(self.speculative_num_steps - 1):
                    self.attn_backends[
                        i
                    ].init_forward_metadata_replay_cuda_graph_from_precomputed(
                        bs=bs,
                        precomputed=precomputed,
                        forward_mode=ForwardMode.DECODE,
                    )
        else:
            # Fallback: compute metadata separately for each backend
            for i in range(self.speculative_num_steps - 1):
                self.attn_backends[i]._replay_cuda_graph_metadata(
                    bs=bs,
                    req_pool_indices=forward_batch.req_pool_indices,
                    seq_lens=forward_batch.seq_lens,
                    seq_lens_sum=forward_batch.seq_lens_sum,
                    encoder_lens=None,
                    forward_mode=ForwardMode.DECODE,
                    spec_info=forward_batch.spec_info,
                    seq_lens_cpu=forward_batch.seq_lens_cpu,
                    out_cache_loc=None,
                )
