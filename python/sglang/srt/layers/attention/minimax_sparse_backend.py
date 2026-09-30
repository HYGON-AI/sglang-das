from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.configs.model_config import (
    get_minimax_sparse_attention_config,
    get_minimax_sparse_disable_value_layer_ids,
    get_minimax_sparse_layer_ids,
    get_minimax_sparse_score_type,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    SharedReadEnds,
)
from sglang.srt.layers.cp.base import get_cp_strategy
from sglang.srt.layers.cp.utils import is_cp_v2_active
from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import m3_fp8_attn_gemm_enabled
from sglang.srt.utils import is_npu

if is_npu():
    from sglang.kernels.ops.attention.minimax_sparse.common.index import (
        topk_index_reduce,
    )

# Adaptive block_size_q thresholds (cut K-cache traffic; affects only the serial loop).
_BSQ_THRESHOLD_64 = 4096  # max_seqlen_k >= 4K  -> BSQ=64
_BSQ_THRESHOLD_32 = 1024  # max_seqlen_k >= 1K  -> BSQ=32
_BSQ_THRESHOLD_16 = 512  # max_seqlen_k >= 512 -> BSQ=16
# BSQ<=64 is UB-safe for the prefill indexer (Q tile up to 8KB at BSQ=64).


def _native_indexer_enabled() -> bool:
    # Native AscendC packed indexer switch (default off).
    return envs.SGLANG_MINIMAX_NPU_NATIVE_INDEXER.get()


def _native_attn_enabled() -> bool:
    # Native AscendC sparse MAIN-attention switch (default off).
    return envs.SGLANG_MINIMAX_NPU_NATIVE_ATTN.get()


if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)
_minimax_sparse_cp_logged = False


def _kv_cache_to_bnsd(
    k_cache: torch.Tensor, v_cache: torch.Tensor, page_size: int
) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """Reshape NHD slot-major KV caches to BNSD [pages, page_size, heads, dim].

    Already-paged 4D inputs pass through unchanged.
    """
    if k_cache.dim() == 4:
        num_pages, _, num_kv_heads, head_dim = k_cache.shape
        return k_cache, v_cache, num_pages, num_kv_heads, head_dim
    num_pages = k_cache.shape[0] // page_size
    num_kv_heads = k_cache.shape[1]
    head_dim = k_cache.shape[2]
    return (
        k_cache.view(num_pages, page_size, num_kv_heads, head_dim),
        v_cache.view(num_pages, page_size, num_kv_heads, head_dim),
        num_pages,
        num_kv_heads,
        head_dim,
    )


def _idx_cache_to_bnsd(
    idx_k_cache: torch.Tensor,
    idx_v_cache: Optional[torch.Tensor],
    page_size: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int]:
    """Reshape NHD slot-major index caches to BNSD; already-paged 4D passes through."""
    if idx_k_cache.dim() == 4:
        return idx_k_cache, idx_v_cache, idx_k_cache.shape[2], idx_k_cache.shape[3]
    num_pages = idx_k_cache.shape[0] // page_size
    idx_kv_heads = idx_k_cache.shape[1]
    idx_dim = idx_k_cache.shape[2]
    idx_v_bnsd = (
        None
        if idx_v_cache is None
        else idx_v_cache.view(num_pages, page_size, idx_kv_heads, idx_dim)
    )
    return (
        idx_k_cache.view(num_pages, page_size, idx_kv_heads, idx_dim),
        idx_v_bnsd,
        idx_kv_heads,
        idx_dim,
    )


def _quant_q_fp8(q: torch.Tensor, q_scale: Optional[float]) -> torch.Tensor:
    # Same convention as the KV pools: the fp8 tensor stores value/scale and
    # the attention kernels multiply the logits back by the scale (None = unit).
    if q_scale is not None:
        q = q / q_scale
    return q.to(torch.float8_e4m3fn)


class MiniMaxSparseAttnBackend(AttentionBackend):
    def __init__(self, runner: ModelRunner):
        assert isinstance(runner.token_to_kv_pool, MiniMaxSparseKVPool)
        self.is_npu = is_npu()
        self.kv_pool = runner.token_to_kv_pool
        self.token_to_kv_pool = runner.token_to_kv_pool  # alias for TboAttnBackend
        self.req_to_token_pool = runner.req_to_token_pool  # pool obj for TboAttnBackend
        self.req_to_token = runner.req_to_token_pool.req_to_token
        self.max_context_len = int(runner.model_config.context_len)
        # Per-forward cache for the native decode block table (rebuilt each forward).
        self._native_decode_bt: dict = {}
        # Reuse the packed and restored query-sharded CP payload buffers across
        # sparse layers. The forward stream orders each layer before reuse.
        self._query_sharded_cp_pack_buffers: dict[tuple, torch.Tensor] = {}
        self._query_sharded_cp_restore_buffers: dict[
            tuple, tuple[torch.Tensor, ...]
        ] = {}
        # FlashMLA index metadata is invariant across MiniMax sparse layers but
        # not across scheduler forwards. It is cleared by
        # init_forward_metadata_out_graph and rebuilt once per ragged CP segment.
        self._flash_mla_prefill_index_meta: dict[str, SimpleNamespace] = {}
        self._flash_mla_prefill_safe_pages: Optional[torch.Tensor] = None
        self.cache_flash_mla_prefill_metadata = os.environ.get(
            "SGLANG_MINIMAX_CACHE_PREFILL_INDEX_METADATA", "0"
        ) in ("1", "true", "True")
        self.fp8_attn_gemm = m3_fp8_attn_gemm_enabled(runner.server_args)
        self.triton_fp8_main_attn_gemm = (
            envs.SGLANG_ENABLE_M3_TRITON_FP8_ATTN_GEMM.get()
        )
        if self.triton_fp8_main_attn_gemm:
            arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
            if not arch.startswith("gfx938"):
                raise RuntimeError(
                    "SGLANG_ENABLE_M3_TRITON_FP8_ATTN_GEMM is currently "
                    f"validated only on gfx938, got {arch or 'non-ROCm device'}"
                )
            if runner.server_args.attention_backend != "triton":
                raise RuntimeError(
                    "SGLANG_ENABLE_M3_TRITON_FP8_ATTN_GEMM requires "
                    "--attention-backend triton"
                )
            if runner.server_args.kv_cache_dtype != "fp8_e4m3":
                raise RuntimeError(
                    "SGLANG_ENABLE_M3_TRITON_FP8_ATTN_GEMM requires "
                    "--kv-cache-dtype fp8_e4m3"
                )
            if self.kv_pool.main_pool.dtype != torch.float8_e4m3fn:
                raise RuntimeError(
                    "gfx938 Triton FP8 attention requires an e4m3fn main KV "
                    f"pool, got {self.kv_pool.main_pool.dtype}"
                )
            if (
                self.kv_pool.index_k_pool is None
                or self.kv_pool.index_k_pool.dtype != runner.dtype
            ):
                raise RuntimeError(
                    "gfx938 Triton FP8 main attention keeps index-K in the "
                    "model dtype so sparse Top-K selection remains unchanged"
                )
        self.main_fp8_attn_gemm = self.fp8_attn_gemm or self.triton_fp8_main_attn_gemm
        if self.fp8_attn_gemm:
            assert self.kv_pool.main_pool.dtype == torch.float8_e4m3fn, (
                "fp8 attn-GEMM mode requires an fp8_e4m3fn main KV pool, got "
                f"{self.kv_pool.main_pool.dtype}"
            )

        hf_config = runner.model_config.hf_config
        sparse_cfg = get_minimax_sparse_attention_config(hf_config)
        self.idx_head_dim = sparse_cfg["sparse_index_dim"]
        self.dense_layer_ids, self.sparse_layer_ids = get_minimax_sparse_layer_ids(
            sparse_cfg
        )
        self.disable_value_layer_ids: set[int] = set(
            get_minimax_sparse_disable_value_layer_ids(sparse_cfg)
        )
        self.score_type: str = get_minimax_sparse_score_type(sparse_cfg)

        # Plain Python int so it is safe inside CUDA graphs (no .item() at graph time).
        self._max_seqlen_q: int = 1
        self._max_seqlen_k: int = 1

        # NPU: per-forward cached metadata for the triton paths (rebuilt each forward).
        self._prefill_meta: Optional[SimpleNamespace] = None
        self._extend_meta: Optional[SimpleNamespace] = None
        self._extend_meta_key: Optional[int] = None
        self._decode_seq_lens_i32_cg: dict[int, torch.Tensor] = {}
        self._verify_meta_cg: dict[tuple, SimpleNamespace] = {}
        self._target_verify_meta_cache: dict[tuple, SimpleNamespace] = {}
        self._target_verify_meta_cache_enabled = True

        # Experimental query-sharded prefill CP.  Each rank computes only its
        # zigzag query slices, while Main K/V and Index K/V for the new chunk
        # are materialized on every CP rank.  This preserves the existing
        # exact local Top-K algorithm and avoids a distributed Top-K in the
        # first implementation.
        self.enable_query_sharded_cp = os.environ.get(
            "SGLANG_OPT_USE_MINIMAX_QUERY_SHARDED_CP", "0"
        ) in ("1", "true", "True")
        # Share one selected sparse-KV block set across a small group of
        # adjacent prefill queries.  The original GPU default (1) launches one
        # program per query and is prohibitively slow for MiniMax-M3's 16K
        # extend-on-128K workload.  Keep the source default conservative, but
        # expose the same query blocking already used by the NPU path so ROCm
        # deployments can opt into the throughput-oriented mode.
        self.block_size_q = int(
            os.environ.get("SGLANG_MINIMAX_PREFILL_BLOCK_SIZE_Q", "1")
        )
        if self.block_size_q not in (1, 2, 4, 8, 16, 32, 64):
            raise ValueError(
                "SGLANG_MINIMAX_PREFILL_BLOCK_SIZE_Q must be one of "
                "1,2,4,8,16,32,64, got "
                f"{self.block_size_q}"
            )
        self.block_size_q_min_query_len = int(
            os.environ.get("SGLANG_MINIMAX_PREFILL_BLOCK_SIZE_Q_MIN_QUERY_LEN", "4096")
        )
        if self.block_size_q_min_query_len < 0:
            raise ValueError(
                "SGLANG_MINIMAX_PREFILL_BLOCK_SIZE_Q_MIN_QUERY_LEN must be >= 0, "
                f"got {self.block_size_q_min_query_len}"
            )
        self.block_size_q_max_query_len = int(
            os.environ.get("SGLANG_MINIMAX_PREFILL_BLOCK_SIZE_Q_MAX_QUERY_LEN", "0")
        )
        if self.block_size_q_max_query_len < 0:
            raise ValueError(
                "SGLANG_MINIMAX_PREFILL_BLOCK_SIZE_Q_MAX_QUERY_LEN must be >= 0, "
                f"got {self.block_size_q_max_query_len}"
            )
        self._last_gpu_prefill_block_size_q: Optional[int] = None
        if self.block_size_q > 1:
            logger.warning(
                "MiniMax sparse prefill is using approximate query blocking: "
                "block_size_q=%d. Adjacent queries share one sparse top-k block "
                "selection when max_query_len >= %d; shorter prefills stay on the "
                "accuracy-safe block_size_q=1 path. The blocked path is not "
                "numerically equivalent to Q1, so run the full model accuracy gate "
                "before using it in production.",
                self.block_size_q,
                self.block_size_q_min_query_len,
            )
        self.block_size_k = sparse_cfg["sparse_block_size"]
        if "sparse_init_block" in sparse_cfg:
            self.init_blocks = sparse_cfg["sparse_init_block"]
        else:
            init_tokens = sparse_cfg["sparse_init_tokens"]
            self.init_blocks = (
                init_tokens + self.block_size_k - 1
            ) // self.block_size_k
        if "sparse_local_block" in sparse_cfg:
            self.local_blocks = sparse_cfg["sparse_local_block"]
        else:
            local_tokens = sparse_cfg["sparse_local_tokens"]
            self.local_blocks = (
                local_tokens + self.block_size_k - 1
            ) // self.block_size_k + 1
        self.topk_blocks = sparse_cfg["sparse_topk_blocks"]

        # MSA (fmha_sm100) is SM100-only; fall back to the Triton sparse path when
        # the kernel is unavailable or its constraints don't hold.
        if self.is_npu:
            self.use_msa = False
            # Prime the native sparse op probe before cuda-graph capture.
            from sgl_kernel_npu.attention.gqa_share_sparse_attention import (
                _get_native_sparse_op,
            )

            self._native_sparse_ok = _get_native_sparse_op() is not None
        else:
            self._native_sparse_ok = False
            from sglang.srt.layers.attention.minimax_sparse_ops.msa import (
                msa_available,
            )

            # MSA (fmha_sm100) runs bf16, or uniform fp8_e4m3 under fp8 attn-GEMM mode
            # (which also casts q to fp8). An fp8 main KV cache WITHOUT the flag
            # would pair a bf16 q with fp8 K/V — unsupported by fmha_sm100's
            # uniform-dtype kernels — so it stays on the Triton sparse path (which
            # dequants fp8 on load). e5m2 is never allowed into MSA (fmha_sm100's
            # variant lookup would silently dispatch the e4m3 kernel).
            _main_kv_is_fp8 = self.kv_pool.main_pool.dtype in (
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            )
            _msa_fp8_ok = (
                self.fp8_attn_gemm
                and self.kv_pool.main_pool.dtype == torch.float8_e4m3fn
            )
            self.use_msa = (
                not envs.SGLANG_DISABLE_MSA.get()
                and msa_available()
                and self.block_size_k == 128
                and self.kv_pool.page_size == self.block_size_k
                and self.topk_blocks in (4, 8, 16, 32)
                and (not _main_kv_is_fp8 or _msa_fp8_ok)
            )
            if (
                not self.use_msa
                and not envs.SGLANG_DISABLE_MSA.get()
                and msa_available()
                and self.block_size_k == 128
                and self.kv_pool.page_size != self.block_size_k
            ):
                logger.warning(
                    "MiniMax-M3 MSA decode disabled: page_size=%d != sparse block size "
                    "%d. Pass --page-size 128 (with an attention backend that allows it, "
                    "e.g. fa4 or trtllm_mha) to enable the faster MSA kernel; falling "
                    "back to the Triton sparse path.",
                    self.kv_pool.page_size,
                    self.block_size_k,
                )

        # Explicit MiniMax-only FlashMLA backend. Stage-3 is independently
        # selectable; the indexer gate below additionally replaces Stage-1/2.
        self.use_flash_mla_gfx938 = envs.SGLANG_OPT_USE_MINIMAX_FLASH_MLA_GFX938.get()
        self.use_flash_mla_gfx938_decode = (
            self.use_flash_mla_gfx938
            and envs.SGLANG_OPT_USE_MINIMAX_FLASH_MLA_GFX938_DECODE.get()
        )
        self.use_flash_mla_gfx938_indexer = (
            self.use_flash_mla_gfx938
            and envs.SGLANG_OPT_USE_MINIMAX_FLASH_MLA_GFX938_INDEXER.get()
        )
        if self.use_flash_mla_gfx938:
            if self.is_npu:
                raise RuntimeError(
                    "SGLANG_OPT_USE_MINIMAX_FLASH_MLA_GFX938 requires ROCm gfx938"
                )
            from sglang.srt.layers.attention.minimax_sparse_ops.flash_mla_gfx938 import (
                flash_mla_gfx938_available,
            )

            local_q_heads = (
                runner.model_config.num_attention_heads // get_parallel().attn_tp_size
            )
            local_kv_heads = self.kv_pool.main_pool.head_num
            contract_errors = []
            if not flash_mla_gfx938_available():
                contract_errors.append(
                    "installed flash_mla lacks usable gfx938 MSA128 kernels"
                )
            if self.fp8_attn_gemm:
                contract_errors.append("FP8 attention GEMM must be disabled")
            if self.kv_pool.main_pool.dtype != torch.bfloat16:
                contract_errors.append(
                    f"KV cache must be BF16, got {self.kv_pool.main_pool.dtype}"
                )
            if self.block_size_k != 128 or self.kv_pool.page_size != 128:
                contract_errors.append(
                    "sparse block size and KV page size must both be 128, got "
                    f"block={self.block_size_k}, page={self.kv_pool.page_size}"
                )
            if self.topk_blocks != 16:
                contract_errors.append(
                    f"sparse TopK must be 16, got {self.topk_blocks}"
                )
            if (local_q_heads, local_kv_heads) != (64, 4):
                contract_errors.append(
                    "local heads must be Q64/KV4 (attention TP1), got "
                    f"Q{local_q_heads}/KV{local_kv_heads}"
                )
            if self.use_flash_mla_gfx938_indexer:
                if self.score_type != "max":
                    contract_errors.append(
                        f"FlashMLA Stage-1/2 requires score_type=max, got {self.score_type}"
                    )
                if set(self.sparse_layer_ids) != self.disable_value_layer_ids:
                    contract_errors.append(
                        "FlashMLA Stage-1/2 requires Index Value disabled on "
                        "every sparse layer"
                    )
                if (
                    get_parallel().attn_cp_size > 1
                    and os.environ.get("SGLANG_MINIMAX_COMBINE_CP_SEGMENTS", "0") == "1"
                ):
                    contract_errors.append(
                        "FlashMLA Stage-1/2 does not support combined CP segments; "
                        "set SGLANG_MINIMAX_COMBINE_CP_SEGMENTS=0"
                    )
            if getattr(runner.server_args, "enable_two_batch_overlap", False):
                contract_errors.append("two-batch overlap is not supported yet")
            if contract_errors:
                raise RuntimeError(
                    "gfx938 FlashMLA MSA128 contract mismatch: "
                    + "; ".join(contract_errors)
                )
            if self.use_msa:
                logger.info(
                    "[MiniMaxSparse] gfx938 FlashMLA explicitly replaces the "
                    "fmha_sm100 MSA Stage-3 backend"
                )
            self.use_msa = False

        self._msa_dec_meta = None
        if self.use_msa:
            self.num_q_heads = (
                runner.model_config.num_attention_heads // get_parallel().attn_tp_size
            )
            self.num_kv_heads = self.kv_pool.main_pool.head_num
            self._msa_nb_max = (
                self.max_context_len + self.block_size_k - 1
            ) // self.block_size_k
            self._msa_cg: dict[int, tuple] = {}

        self._flash_mla_page_table = None
        self._flash_mla_page_tables: dict[tuple, torch.Tensor] = {}
        self._flash_mla_sched_meta: dict[tuple[int, int], object] = {}
        self._flash_mla_prefill_indices: dict[tuple, torch.Tensor] = {}
        self._flash_mla_decode_indices: dict[tuple, torch.Tensor] = {}
        self._flash_mla_nb_max = (
            self.req_to_token.shape[1] + self.block_size_k - 1
        ) // self.block_size_k

        self.page_size = self.kv_pool.page_size
        self.use_dense_sparse_decode = (
            (not self.is_npu)
            and not self.use_flash_mla_gfx938
            and envs.SGLANG_OPT_USE_MINIMAX_DENSE_SPARSE_DECODE.get()
            and self.block_size_k % self.page_size == 0
            # _dense_sparse_main_decode calls trtllm decode with a bf16 q and
            # unit bmm scales — no fp8 handling yet (follow-up).
            and not self.fp8_attn_gemm
        )
        from sglang.srt.model_executor.cuda_graph_config import (
            Backend,
            Phase,
            check_cuda_graph_backend,
        )

        _sa = getattr(runner, "server_args", None)
        self.speculative_num_draft_tokens = getattr(
            _sa, "speculative_num_draft_tokens", None
        )
        self._target_verify_meta_cache_enabled = not bool(
            getattr(_sa, "enable_two_batch_overlap", False)
        )
        _decode_cuda_graph = not check_cuda_graph_backend(
            Phase.DECODE, Backend.DISABLED
        )
        self._flash_mla_decode_current = False
        self._use_msa_decode = self.use_msa and (
            not _decode_cuda_graph or envs.SGLANG_OPT_USE_MSA_DECODE_UNDER_GRAPH.get()
        )

        # MSA + spec decode + cuda graph crashes mid-capture: TARGET_VERIFY batches
        # route to forward_extend, dereferencing absent extend metadata. Fail at startup.
        if (
            self.use_msa
            and _decode_cuda_graph
            and getattr(_sa, "speculative_algorithm", None) is not None
        ):
            raise NotImplementedError(
                "MiniMax-M3 MSA attention does not support speculative decoding under "
                "CUDA graph. Use --disable-cuda-graph, set SGLANG_DISABLE_MSA=1, or "
                "disable speculative decoding."
            )
        self._msa_owns_decode = self._use_msa_decode and not (
            self.use_dense_sparse_decode and self.kv_pool.main_pool.head_num == 1
        )
        self.dense_backend: Optional[AttentionBackend] = None
        _main_attn_name = (
            "flash_mla_gfx938"
            if self.use_flash_mla_gfx938
            else ("MSA" if self.use_msa else "triton")
        )
        logger.info(
            f"[MiniMaxSparse] Backend initialized "
            f"(score_type={self.score_type!r}, "
            f"main_attn={_main_attn_name}, "
            f"msa_decode={self._use_msa_decode}, "
            f"flash_mla_decode={self.use_flash_mla_gfx938_decode}, "
            f"flash_mla_indexer={self.use_flash_mla_gfx938_indexer}, "
            f"msa_owns_decode={self._msa_owns_decode}, "
            f"decode_cuda_graph={_decode_cuda_graph}, "
            f"prefill_block_q={self.block_size_q}, "
            f"prefill_block_q_min_query_len={self.block_size_q_min_query_len}, "
            f"fp8_attn_gemm={self.fp8_attn_gemm}, "
            f"triton_fp8_main_attn_gemm={self.triton_fp8_main_attn_gemm}, "
            f"npu_native_attn={'on' if (self._native_sparse_ok and _native_attn_enabled()) else 'off'}, "
            f"disable_value_layers={sorted(self.disable_value_layer_ids)})"
        )
        if self.fp8_attn_gemm and self.use_msa:
            logger.info(
                "[MiniMaxSparse] fp8 MSA active: the first forward may "
                "JIT-compile fmha_sm100 fp8 kernel variants (cold cache can "
                "take minutes; compiles serialize across TP ranks)."
            )

    def _select_gpu_prefill_block_size_q(self) -> int:
        """Use exact Q1 for short prompts and blocked Q for long extends.

        The 128K-prefix + 16K-extend target benefits heavily from sharing one
        sparse block selection across adjacent queries. Applying that same
        approximation to short accuracy prompts is unnecessary and can move
        model-level scores. A threshold of zero disables this adaptive fallback.
        """
        selected = self.block_size_q
        if (
            selected > 1
            and self.block_size_q_min_query_len > 0
            and int(self._max_seqlen_q) < self.block_size_q_min_query_len
        ):
            selected = 1
        if (
            selected > 1
            and self.block_size_q_max_query_len > 0
            and int(self._max_seqlen_q) > self.block_size_q_max_query_len
        ):
            selected = 1
        if selected != self._last_gpu_prefill_block_size_q:
            logger.info(
                "[MiniMaxSparse] selected GPU prefill block-Q=%d "
                "(configured=%d, max_query_len=%d, fast_path_min_query_len=%d, "
                "fast_path_max_query_len=%d)",
                selected,
                self.block_size_q,
                int(self._max_seqlen_q),
                self.block_size_q_min_query_len,
                self.block_size_q_max_query_len,
            )
            self._last_gpu_prefill_block_size_q = selected
        return selected

    @staticmethod
    def _choose_decode_score_max_chunks(batch_size: int) -> int:
        """Score chunk count per graph bucket.

        bs=1 uses 16 chunks; larger buckets keep 32. Verify has its own tuning.
        """
        return 16 if int(batch_size) == 1 else 32

    @staticmethod
    def _choose_block_size_q(max_seqlen_k: int) -> int:
        """Pick block_size_q from max KV length (MINIMAX_NPU_PREFILL_BSQ overrides)."""
        _forced = os.environ.get("MINIMAX_NPU_PREFILL_BSQ")
        if _forced:
            try:
                _v = int(_forced)
                if _v > 0:
                    return _v
            except ValueError:
                pass
        if max_seqlen_k >= _BSQ_THRESHOLD_64:
            return 64
        if max_seqlen_k >= _BSQ_THRESHOLD_32:
            return 32
        if max_seqlen_k >= _BSQ_THRESHOLD_16:
            return 16
        return 1

    # ------------------------------------------------------------------
    # Delegation helpers
    # ------------------------------------------------------------------

    def _get_target_verify_meta(
        self, forward_batch: ForwardBatch, ndt: int
    ) -> Optional[SimpleNamespace]:
        """Reuse TARGET_VERIFY request/length tensors across sparse layers."""
        if not self._target_verify_meta_cache_enabled:
            return None
        bs = int(forward_batch.seq_lens.shape[0])
        ndt = int(ndt)
        if bs <= 0 or ndt <= 0:
            return None

        device = forward_batch.seq_lens.device
        key = (device.type, device.index, bs, ndt)
        meta = self._target_verify_meta_cache.get(key)
        if meta is None:
            # Graph setup runs before the backend starts the actual CUDA graph
            # capture.  If this helper is nevertheless reached from inside a
            # capture, leave allocation to the legacy graph-safe fallback.
            try:
                if torch.cuda.is_current_stream_capturing():
                    return None
            except Exception:
                pass
            try:
                offsets = torch.arange(1, ndt + 1, device=device, dtype=torch.long)
                prefix_lens = torch.empty(bs, device=device, dtype=torch.long)
                req_matrix = torch.empty((bs, ndt), device=device, dtype=torch.long)
                seq_matrix = torch.empty_like(req_matrix)
            except RuntimeError:
                try:
                    capturing = torch.cuda.is_current_stream_capturing()
                except Exception:
                    capturing = False
                if capturing:
                    return None
                raise
            meta = SimpleNamespace(
                offsets=offsets,
                prefix_lens=prefix_lens,
                req_matrix=req_matrix,
                seq_matrix=seq_matrix,
                per_query_req=req_matrix.reshape(-1),
                per_query_seq_lens=seq_matrix.reshape(-1),
            )
            self._target_verify_meta_cache[key] = meta

        meta.prefix_lens.copy_(forward_batch.seq_lens)
        meta.req_matrix.copy_(
            forward_batch.req_pool_indices.reshape(bs, 1).expand(bs, ndt)
        )
        torch.add(
            meta.prefix_lens[:, None],
            meta.offsets[None, :],
            out=meta.seq_matrix,
        )
        return meta

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        # getattr covers replay views lacking extend_seq_lens_cpu and TARGET_VERIFY.
        self._msa_dec_meta = None
        self._flash_mla_page_table = None
        self._flash_mla_prefill_index_meta.clear()
        self._flash_mla_prefill_safe_pages = None
        if (
            not self.is_npu
            and forward_batch.forward_mode.is_target_verify()
            and self.speculative_num_draft_tokens
        ):
            self._get_target_verify_meta(
                forward_batch, int(self.speculative_num_draft_tokens)
            )
        if self.is_npu:
            # Invalidate cached prefill/extend metadata; rebuilt on first sparse layer.
            self._prefill_meta = None
            self._extend_meta = None
            self._extend_meta_key = None
        extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_lens is not None:
            self._max_seqlen_q = int(max(extend_lens))
        elif forward_batch.forward_mode.is_target_verify():
            # EAGLE TARGET_VERIFY does not populate extend_seq_lens_cpu.  The
            # packed query still contains one fixed-size verify block per
            # request, so using the decode default (1) leaves all but the first
            # verify row uncomputed.  This is especially damaging under graph
            # replay: stale/zero logits make every draft miss and can corrupt
            # the fallback target token.
            self._max_seqlen_q = int(self.speculative_num_draft_tokens or 1)
        else:
            self._max_seqlen_q = 1
        if in_capture and (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
        ):
            # Capture uses tiny dummy seq_lens; bound by full context so replay
            # (longer sequences) does not miss KV blocks.
            self._max_seqlen_k = self.max_context_len
        else:
            self._max_seqlen_k = int(forward_batch.seq_lens_cpu.max().item())

        if self.use_flash_mla_gfx938:
            is_decode_like = (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
            )
            self._flash_mla_decode_current = (
                self.use_flash_mla_gfx938_decode and is_decode_like
            )
            if (
                forward_batch.forward_mode.is_target_verify()
                and self._flash_mla_decode_current
            ):
                ndt = int(self.speculative_num_draft_tokens or 0)
                verify_meta = self._get_target_verify_meta(forward_batch, ndt)
                if verify_meta is None:
                    raise RuntimeError(
                        "gfx938 FlashMLA TARGET_VERIFY requires reusable packed "
                        "request metadata"
                    )
                # Native FlashMLA target verify keeps speculative positions in
                # the s_q dimension. The page mapping is request-invariant;
                # per-position sequence bounds are passed separately below.
                self._prepare_flash_mla_page_table(
                    forward_batch.req_pool_indices,
                    verify_meta.seq_matrix[:, -1],
                )
            else:
                self._prepare_flash_mla_page_table(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                )

        # Build plan + page table eager (outside capture) so captured forward_decode
        # runs only device-side ops; host-side code can't be captured.
        if self._msa_owns_decode and forward_batch.forward_mode.is_decode_or_idle():
            self._prepare_msa_decode_meta(forward_batch)

        # ---- REPLAY-FRESH native verify block_table ----
        if (
            self.is_npu
            and forward_batch.forward_mode.is_target_verify()
            and self.speculative_num_draft_tokens
        ):
            _ndt = self.speculative_num_draft_tokens
            _bs = forward_batch.seq_lens.shape[0]
            _key = (_bs, int(_ndt))
            _vmeta = self._verify_meta_cg.get(_key)
            if _vmeta is not None:
                _vmeta.per_query_req.copy_(
                    forward_batch.req_pool_indices.long().repeat_interleave(int(_ndt))
                )
                _prefix = (forward_batch.seq_lens.to(torch.long) - int(_ndt)).clamp(
                    min=0
                )
                _offs = torch.arange(
                    1,
                    int(_ndt) + 1,
                    device=forward_batch.seq_lens.device,
                    dtype=torch.long,
                )
                _vmeta.per_query_seq_lens.copy_(
                    (_prefix.unsqueeze(1) + _offs.unsqueeze(0))
                    .reshape(-1)
                    .to(torch.int32)
                )
                _mb = self.req_to_token.shape[1] // self.page_size
                _bt_cols = (
                    torch.arange(
                        _mb, device=_vmeta.per_query_req.device, dtype=torch.long
                    )
                    * self.page_size
                ).clamp(max=self.req_to_token.shape[1] - 1)
                _vmeta.native_bt = (
                    self.req_to_token[_vmeta.per_query_req][:, _bt_cols]
                    // self.page_size
                ).to(torch.int32)

    def _prepare_msa_decode_meta(self, forward_batch: ForwardBatch):
        """Refresh the persistent per-batch-size MSA decode plan + page table in place."""
        from sglang.srt.layers.attention.minimax_sparse_ops.msa import (
            build_msa_decode_cg_plan,
            update_msa_decode_cg_meta,
        )

        bs = forward_batch.seq_lens.shape[0]
        if bs == 0:
            return
        entry = self._msa_cg.get(bs)
        if entry is None:
            device = forward_batch.seq_lens.device
            plan = build_msa_decode_cg_plan(
                self.num_q_heads,
                self.num_kv_heads,
                self.block_size_k,
                self.topk_blocks,
                bs,
                device=device,
                is_fp8=self.fp8_attn_gemm,
            )
            kv_indices_buf = torch.zeros(
                bs * self._msa_nb_max, dtype=torch.int32, device=device
            )
            entry = (plan, kv_indices_buf)
            self._msa_cg[bs] = entry
        plan, kv_indices_buf = entry
        update_msa_decode_cg_meta(
            plan,
            kv_indices_buf,
            self.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            self.block_size_k,
            self.topk_blocks,
            self.num_q_heads,
            self.num_kv_heads,
        )
        self._msa_dec_meta = (kv_indices_buf, plan)

    def _prepare_flash_mla_page_table(
        self, slot_ids: torch.Tensor, seq_lens: torch.Tensor
    ) -> None:
        """Refresh a stable logical-page to physical-page table for this batch."""
        from sglang.srt.layers.attention.minimax_sparse_ops.flash_mla_gfx938 import (
            update_flash_mla_page_table,
        )

        rows = int(slot_ids.numel())
        if rows == 0:
            return
        device = slot_ids.device
        key = (device.type, device.index, rows)
        table = self._flash_mla_page_tables.get(key)
        if table is None:
            table = torch.empty(
                (rows, self._flash_mla_nb_max),
                dtype=torch.int32,
                device=device,
            )
            self._flash_mla_page_tables[key] = table
        update_flash_mla_page_table(
            table,
            self.req_to_token,
            slot_ids,
            seq_lens,
            self.block_size_k,
        )
        self._flash_mla_page_table = table

    def _get_flash_mla_sched_meta(
        self, layer_id: int, rows: int, verify_group_size: int = 1
    ):
        """Keep one initialized FlashMLA scheduler object per layer/graph shape."""
        verify_group_size = int(verify_group_size)
        if verify_group_size <= 0 or int(rows) % verify_group_size:
            raise ValueError(
                "FlashMLA scheduler rows must be divisible by "
                f"verify_group_size, got {rows} and {verify_group_size}"
            )
        key = (int(layer_id), int(rows) // verify_group_size, verify_group_size)
        meta = self._flash_mla_sched_meta.get(key)
        if meta is None:
            from sglang.srt.layers.attention.minimax_sparse_ops.flash_mla_gfx938 import (
                new_flash_mla_decode_metadata,
            )

            meta = new_flash_mla_decode_metadata()
            self._flash_mla_sched_meta[key] = meta
        return meta

    def _get_flash_mla_indices_output(
        self,
        rows: int,
        device: torch.device,
        *,
        decode: bool,
        verify_group_size: int = 1,
    ) -> torch.Tensor:
        """Reuse the page-expanded Top16 workspace across sparse layers."""
        rows = int(rows)
        if decode:
            verify_group_size = int(verify_group_size)
            if verify_group_size <= 0 or rows % verify_group_size:
                raise ValueError(
                    "FlashMLA workspace rows must be divisible by "
                    f"verify_group_size, got {rows} and {verify_group_size}"
                )
            batch = rows // verify_group_size
            key = (device.type, device.index, batch, verify_group_size)
            output = self._flash_mla_decode_indices.get(key)
            if output is None:
                output = torch.empty(
                    (batch, verify_group_size, 4, 16 * self.block_size_k),
                    dtype=torch.int32,
                    device=device,
                )
                self._flash_mla_decode_indices[key] = output
            return output

        stream_id = int(torch.cuda.current_stream(device).cuda_stream)
        key = (device.type, device.index, stream_id)
        output = self._flash_mla_prefill_indices.get(key)
        if output is None or output.shape[0] < rows:
            output = torch.empty(
                (rows, 4, 16 * self.block_size_k),
                dtype=torch.int32,
                device=device,
            )
            self._flash_mla_prefill_indices[key] = output
        return output[:rows]

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        if not self.is_npu:
            return
        # Layer-invariant decode/verify metadata as captured ops (re-read at replay).
        fm = forward_batch.forward_mode
        if fm.is_target_verify():
            ndt = self.speculative_num_draft_tokens
            if ndt:
                prefix = (forward_batch.seq_lens.to(torch.long) - int(ndt)).clamp(min=0)
                offsets = torch.arange(
                    1,
                    int(ndt) + 1,
                    device=forward_batch.seq_lens.device,
                    dtype=torch.long,
                )
                per_query_seq_lens = (
                    (prefix.unsqueeze(1) + offsets.unsqueeze(0))
                    .reshape(-1)
                    .to(torch.int32)
                )
                per_query_req = forward_batch.req_pool_indices.long().repeat_interleave(
                    int(ndt)
                )
                # Captured block_table for the native verify op (re-runs at replay).
                _mb = self.req_to_token.shape[1] // self.page_size
                _bt_cols = (
                    torch.arange(_mb, device=per_query_req.device, dtype=torch.long)
                    * self.page_size
                ).clamp(max=self.req_to_token.shape[1] - 1)
                _native_bt = (
                    self.req_to_token[per_query_req][:, _bt_cols] // self.page_size
                ).to(torch.int32)
                self._verify_meta_cg[(forward_batch.seq_lens.shape[0], int(ndt))] = (
                    SimpleNamespace(
                        per_query_seq_lens=per_query_seq_lens,
                        per_query_req=per_query_req,
                        native_bt=_native_bt,
                    )
                )
        elif fm.is_decode_or_idle():
            self._decode_seq_lens_i32_cg[forward_batch.seq_lens.shape[0]] = (
                forward_batch.seq_lens.to(torch.int32)
            )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def _merge_sparse_blocks(
        self,
        topk_blocks: torch.Tensor,
        query_positions: torch.Tensor,
        num_blocks: int,
    ) -> torch.Tensor:
        """Append forced init/local blocks to top-k block ids and deduplicate."""
        total = self.topk_blocks + self.init_blocks + self.local_blocks
        if self.init_blocks <= 0 and self.local_blocks <= 0:
            return topk_blocks

        block_size = self.block_size_k
        q_len = query_positions.shape[0]
        num_idx_heads = topk_blocks.shape[1]
        qcol = query_positions[:, None, None]

        if self.init_blocks == 0 and self.local_blocks == 1:
            local = (query_positions // block_size).clamp(
                min=0, max=max(num_blocks - 1, 0)
            )
            local = (
                local.to(topk_blocks.dtype)
                .view(q_len, 1, 1)
                .expand(-1, num_idx_heads, -1)
            )
            valid_topk = (topk_blocks >= 0) & (topk_blocks < num_blocks)
            valid_topk = valid_topk & (topk_blocks * block_size <= qcol)
            local_duplicate = ((topk_blocks == local) & valid_topk).any(
                dim=-1, keepdim=True
            )
            valid_local = (local >= 0) & (local < num_blocks)
            valid_local = valid_local & (local * block_size <= qcol) & ~local_duplicate
            return torch.cat(
                [
                    torch.where(
                        valid_topk, topk_blocks, torch.full_like(topk_blocks, -1)
                    ),
                    torch.where(valid_local, local, torch.full_like(local, -1)),
                ],
                dim=-1,
            )

        forced_parts = []
        if self.init_blocks > 0:
            forced_parts.append(
                torch.arange(
                    self.init_blocks,
                    device=topk_blocks.device,
                    dtype=topk_blocks.dtype,
                )
                .view(1, 1, -1)
                .expand(q_len, num_idx_heads, -1)
            )
        if self.local_blocks > 0:
            offsets = torch.arange(
                self.local_blocks,
                device=topk_blocks.device,
                dtype=query_positions.dtype,
            )
            block_ids = query_positions // block_size
            first = (block_ids - self.local_blocks + 1).clamp(min=0)
            forced_parts.append(
                (first[:, None] + offsets[None, :])
                .to(topk_blocks.dtype)
                .view(q_len, 1, -1)
                .expand(-1, num_idx_heads, -1)
            )

        forced = torch.cat(forced_parts, dim=-1)
        candidates = torch.cat([forced, topk_blocks], dim=-1)
        valid = (candidates >= 0) & (candidates < num_blocks)
        valid = valid & (candidates * block_size <= qcol)
        invalid_value = torch.full_like(candidates, num_blocks)
        sorted_candidates = torch.sort(
            torch.where(valid, candidates, invalid_value), dim=-1
        ).values
        sorted_valid = sorted_candidates < num_blocks
        previous = torch.cat(
            [
                torch.full_like(sorted_candidates[..., :1], -1),
                sorted_candidates[:, :, :-1],
            ],
            dim=-1,
        )
        keep = sorted_valid & (sorted_candidates != previous)
        ranks = torch.cumsum(keep.to(torch.int32), dim=-1) - 1
        output = torch.full(
            (q_len, num_idx_heads, total + 1),
            -1,
            dtype=topk_blocks.dtype,
            device=topk_blocks.device,
        )
        overflow_rank = torch.full_like(ranks, total)
        scatter_index = torch.where(keep & (ranks < total), ranks, overflow_rank).long()
        scatter_src = torch.where(keep, sorted_candidates, -1)
        output.scatter_(2, scatter_index, scatter_src)
        return output[:, :, :total]

    def _prepare_npu_triton_topk_idx(
        self,
        topk_idx: torch.Tensor,
        seq_lens: torch.Tensor,
        num_idx_heads: int,
        num_kv_heads: int,
        max_blocks: int,
    ) -> torch.Tensor:
        """Prepare NPU triton top-k ids in the GQA kernel layout. MiniMax-M3 (TP=16) emits it directly, skipping transpose+append+dedup."""
        if (
            self.init_blocks == 0
            and self.local_blocks == 1
            and num_idx_heads == num_kv_heads
            and topk_idx.shape[0] == num_kv_heads
            and topk_idx.dtype == torch.int32
            and topk_idx.is_contiguous()
            and seq_lens.is_contiguous()
        ):
            # Fused prefill topk already appended the causal local block ([..., topk+1]); decode/verify still need the append.
            if topk_idx.shape[2] == self.topk_blocks + 1:
                return topk_idx
            from sgl_kernel_npu.indexer.flash_block_score_decode import (
                append_local_block_to_topk_idx,
            )

            return append_local_block_to_topk_idx(
                topk_idx, seq_lens, self.block_size_k, max_blocks
            )

        if num_idx_heads > num_kv_heads:
            idx_group_size = num_idx_heads // num_kv_heads
            topk_idx = topk_index_reduce(
                topk_idx.view(num_kv_heads, idx_group_size, -1, self.topk_blocks),
                dim=1,
            )

        topk_2d = topk_idx.permute(1, 0, 2).contiguous()
        query_positions = (seq_lens.to(torch.long) - 1).clamp(min=0)
        topk_merged = self._merge_sparse_blocks(topk_2d, query_positions, max_blocks)
        return topk_merged.permute(1, 0, 2).contiguous()

    def _build_native_block_table(
        self, req_indices: torch.Tensor, max_blocks: int, device
    ) -> torch.Tensor:
        """Logical->physical page table for the native sparse main op."""
        blk_cols = (
            torch.arange(max_blocks, device=device, dtype=torch.long) * self.page_size
        ).clamp(max=self.req_to_token.shape[1] - 1)
        return (self.req_to_token[req_indices][:, blk_cols] // self.page_size).to(
            torch.int32
        )

    def _forward_npu_triton_decode(
        self,
        q: torch.Tensor,  # [B, num_q_heads, head_dim]
        k_cache: torch.Tensor,  # [num_slots, num_kv_heads, head_dim] (NHD)
        v_cache: torch.Tensor,  # [num_slots, num_kv_heads, head_dim]
        idx_q: torch.Tensor,  # [B, num_idx_heads, idx_dim]
        idx_k_cache: torch.Tensor,  # [num_slots, idx_kv_heads, idx_dim]
        idx_v_cache: Optional[
            torch.Tensor
        ],  # [num_slots, idx_kv_heads, idx_dim] or None
        forward_batch: ForwardBatch,
    ):
        """NPU decode via the ported triton kernels (BNSD paged).
        NHD paged KV reshapes to [pages, block_size, H, D]; block table from
        req_to_token.
        """
        from sgl_kernel_npu.attention.gqa_share_sparse_attention import (
            flash_decode_bnsd_with_gqa_share_sparse,
        )
        from sgl_kernel_npu.indexer.flash_block_score_decode import (
            flash_decode_bnsd_with_topk_idx,
        )

        page_size = self.page_size  # == block_size_k
        num_q_heads = q.shape[1]
        head_dim = q.shape[2]
        num_idx_heads = idx_q.shape[1]
        idx_dim = idx_q.shape[2]

        k_bnsd, v_bnsd, num_pages, num_kv_heads, head_dim = _kv_cache_to_bnsd(
            k_cache, v_cache, page_size
        )
        idx_k_bnsd, idx_v_bnsd, idx_kv_heads, idx_dim = _idx_cache_to_bnsd(
            idx_k_cache, idx_v_cache, page_size
        )

        # int32 seq_lens is layer-invariant: read the per-bs buffer built once
        # per forward (captured op), with an inline eager fallback.
        bs = forward_batch.seq_lens.shape[0]
        seq_lens = self._decode_seq_lens_i32_cg.get(bs)
        if seq_lens is None:
            seq_lens = forward_batch.seq_lens.to(torch.int32)
        max_seqlen = (
            int(self._max_seqlen_k)
            if self._max_seqlen_k
            else int(seq_lens.max().item())
        )
        max_blocks = (max_seqlen + page_size - 1) // page_size
        disable_index_value = idx_v_cache is None
        # Native main op takes a logical->physical block table, hoisted to
        # once per forward (cached by id(forward_batch)); triton falls back to req_to_token.
        _native_main_kwargs = None
        if self._native_sparse_ok and _native_attn_enabled():
            try:
                _fb_id = id(forward_batch)
                _bt = self._native_decode_bt.get(_fb_id)
                if _bt is None or _bt.shape[0] != q.shape[0]:
                    _bt = self._build_native_block_table(
                        forward_batch.req_pool_indices.long(), max_blocks, q.device
                    )
                    self._native_decode_bt = {_fb_id: _bt}  # single-entry: drop stale
                _native_main_kwargs = {"block_table": _bt}
            except Exception:
                _native_main_kwargs = None
        if disable_index_value:
            page_source_kwargs = dict(
                block_table=None,
                req_to_token=self.req_to_token,
                req_pool_indices=forward_batch.req_pool_indices,
                max_num_blocks=max_blocks,
                num_pages=num_pages,
                sanitize_page_ids=False,
            )
        else:
            # Legacy score+index-value contract for non-MiniMax-M3 sparse layouts.
            req_idx = forward_batch.req_pool_indices.long()
            max_cols = self.req_to_token.shape[1]
            blk_cols = (
                torch.arange(max_blocks, device=q.device, dtype=torch.long) * page_size
            ).clamp(max=max_cols - 1)
            token_slots = self.req_to_token[req_idx][:, blk_cols]
            page_source_kwargs = dict(
                block_table=(token_slots // page_size).to(torch.int32)
            )

        # 1) indexer: score idx_k + index attention + topk (init/local=0;
        # forced blocks are re-appended by _prepare_npu_triton_topk_idx).
        idx_o, topk_idx = flash_decode_bnsd_with_topk_idx(
            q=idx_q,
            sink=None,
            k_cache_bnsd=idx_k_bnsd,
            v_cache_bnsd=idx_v_bnsd,
            **page_source_kwargs,
            seq_lens=seq_lens,
            max_seqlen=max_seqlen,
            block_size=page_size,
            topk=self.topk_blocks,
            init_blocks=0,
            local_blocks=0,
            sm_scale=idx_dim**-0.5,
            score_type=self.score_type,
            disable_index_value=disable_index_value,
            runtime_fill_only=True,
            score_max_chunks=self._choose_decode_score_max_chunks(bs),
            fused_append_local=True,
            use_native=_native_indexer_enabled(),
        )

        # 2) Reduce heads and append forced blocks.
        topk_idx = self._prepare_npu_triton_topk_idx(
            topk_idx, seq_lens, num_idx_heads, num_kv_heads, max_blocks
        )

        # 4) Main sparse attention; native op uses the cached block table override.
        _main_kwargs = (
            {**page_source_kwargs, **_native_main_kwargs}
            if _native_main_kwargs is not None
            else page_source_kwargs
        )
        o = flash_decode_bnsd_with_gqa_share_sparse(
            q=q,
            sink=None,
            k_cache_bnsd=k_bnsd,
            v_cache_bnsd=v_bnsd,
            **_main_kwargs,
            seq_lens=seq_lens,
            block_size=page_size,
            topk_idx=topk_idx,
            sm_scale=head_dim**-0.5,
            use_native=_native_attn_enabled(),
        )

        return idx_o, o

    def _forward_npu_triton_verify(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        idx_q: torch.Tensor,
        idx_k_cache: torch.Tensor,
        idx_v_cache: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        prefix_lens: torch.Tensor,
    ):
        """Capture-safe sparse attention for TARGET_VERIFY.
        ndt queries per request, each causal (j attends KV[0:prefix+j+1]). Flatten
        to per-query rows, reuse the decode kernels (device ops only, no .item()).
        """
        from sgl_kernel_npu.attention.gqa_share_sparse_attention import (
            flash_decode_bnsd_with_gqa_share_sparse,
        )
        from sgl_kernel_npu.indexer.flash_block_score_decode import (
            flash_decode_bnsd_with_topk_idx,
        )

        page_size = self.page_size  # == block_size_k
        num_q_heads = q.shape[1]
        head_dim = q.shape[2]
        num_idx_heads = idx_q.shape[1]
        idx_dim = idx_q.shape[2]
        num_tokens = q.shape[0]
        bs = forward_batch.seq_lens.shape[0]
        ndt = num_tokens // max(bs, 1)

        k_bnsd, v_bnsd, num_pages, num_kv_heads, head_dim = _kv_cache_to_bnsd(
            k_cache, v_cache, page_size
        )
        idx_k_bnsd, idx_v_bnsd, idx_kv_heads, idx_dim = _idx_cache_to_bnsd(
            idx_k_cache, idx_v_cache, page_size
        )

        # Per-query causal seq_lens + req are layer-invariant, built once per
        # forward as captured ops; inline fallback for the eager path.
        vmeta = self._verify_meta_cg.get((bs, ndt))
        if vmeta is None:
            prefix = (forward_batch.seq_lens.to(torch.long) - int(ndt)).clamp(min=0)
            offsets = torch.arange(1, int(ndt) + 1, device=q.device, dtype=torch.long)
            per_query_seq_lens = (
                (prefix.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1).to(torch.int32)
            )
            per_query_req = forward_batch.req_pool_indices.long().repeat_interleave(
                int(ndt)
            )
        else:
            per_query_seq_lens = vmeta.per_query_seq_lens
            per_query_req = vmeta.per_query_req

        # ``max_seqlen`` comes from the capture-safe ``_max_seqlen_k`` (host-derived
        # in init_forward_metadata_out_graph) so no device->host sync here.
        max_seqlen = (
            int(self._max_seqlen_k)
            if self._max_seqlen_k
            else int(per_query_seq_lens.max().item())
        )
        max_blocks = (max_seqlen + page_size - 1) // page_size
        disable_index_value = idx_v_cache is None
        # Native verify-main: per-query block_table. CUDA-graph path uses the
        # captured vmeta.native_bt (refreshed on replay); eager builds it per call.
        _native_main_kwargs = None
        if self._native_sparse_ok and _native_attn_enabled():
            try:
                _bt = (
                    vmeta.native_bt
                    if (
                        vmeta is not None
                        and getattr(vmeta, "native_bt", None) is not None
                    )
                    else None
                )
                if _bt is None:
                    _bt = self._build_native_block_table(
                        per_query_req.long(), max_blocks, q.device
                    )
                _native_main_kwargs = {"block_table": _bt}
            except Exception:
                _native_main_kwargs = None
        if disable_index_value:
            # Keep verify's page-id range guard in the direct-map kernel.
            page_source_kwargs = dict(
                block_table=None,
                req_to_token=self.req_to_token,
                req_pool_indices=per_query_req,
                max_num_blocks=max_blocks,
                num_pages=num_pages,
                sanitize_page_ids=True,
            )
        else:
            max_cols = self.req_to_token.shape[1]
            blk_cols = (
                torch.arange(max_blocks, device=q.device, dtype=torch.long) * page_size
            ).clamp(max=max_cols - 1)
            token_slots = self.req_to_token[per_query_req][:, blk_cols]
            block_table = (token_slots // page_size).to(torch.int32)
            block_table = block_table.clamp(min=0, max=num_pages - 1)
            page_source_kwargs = dict(block_table=block_table)

        # 1) indexer: score idx_k + index attention + topk (init/local=0).
        # Pack each request's ndt draft queries into the gqa row dim.
        pack_verify = (
            disable_index_value and int(ndt) > 1 and num_idx_heads == idx_kv_heads
        )
        if pack_verify:
            idx_q_score = idx_q.reshape(bs, ndt * num_idx_heads, idx_dim)
            if num_idx_heads == 1:
                # Row order == flat query order (request-major), so the
                # per-query lengths double as the packed per-row lengths.
                score_seq_lens = per_query_seq_lens
            else:
                score_seq_lens = (
                    per_query_seq_lens.view(bs, ndt, 1)
                    .expand(bs, ndt, num_idx_heads)
                    .reshape(-1)
                )
            score_page_source_kwargs = dict(
                block_table=None,
                req_to_token=self.req_to_token,
                req_pool_indices=forward_batch.req_pool_indices,
                max_num_blocks=max_blocks,
                num_pages=num_pages,
                sanitize_page_ids=True,
            )
        else:
            idx_q_score = idx_q
            score_seq_lens = per_query_seq_lens
            score_page_source_kwargs = page_source_kwargs
        idx_o, topk_idx = flash_decode_bnsd_with_topk_idx(
            q=idx_q_score,
            sink=None,
            k_cache_bnsd=idx_k_bnsd,
            v_cache_bnsd=idx_v_bnsd,
            **score_page_source_kwargs,
            seq_lens=score_seq_lens,
            max_seqlen=max_seqlen,
            block_size=page_size,
            topk=self.topk_blocks,
            init_blocks=0,
            local_blocks=0,
            sm_scale=idx_dim**-0.5,
            score_type=self.score_type,
            disable_index_value=disable_index_value,
            packed_seq_lens=pack_verify,
            # 64-chunk graph for long contexts; runtime uses 16 chunks while
            # <=256 blocks to cut short-context score work. Runtime direct-fill
            # removes register TopK maintenance.
            score_blocks_per_chunk=8 if pack_verify else 16,
            score_max_chunks=64 if pack_verify else 32,
            runtime_fill_only=pack_verify,
            runtime_score_short_max_blocks=256 if pack_verify else 0,
            runtime_score_short_chunks=16 if pack_verify else 0,
            fused_append_local=True,
            use_native=_native_indexer_enabled(),
        )
        if pack_verify:
            # [ndt*H, bs, K] -> [H, bs*ndt, K] (request-major rows).
            k_last = topk_idx.shape[-1]
            topk_idx = (
                topk_idx.view(ndt, num_idx_heads, bs, k_last)
                .permute(1, 2, 0, 3)
                .reshape(num_idx_heads, bs * ndt, k_last)
                .contiguous()
            )

        # 2) Reduce heads and append forced blocks in the GQA kernel layout.
        topk_idx = self._prepare_npu_triton_topk_idx(
            topk_idx,
            per_query_seq_lens,
            num_idx_heads,
            num_kv_heads,
            max_blocks,
        )

        # 4) Main sparse attention; native op uses the cached block table override.
        _vmain_kwargs = (
            {**page_source_kwargs, **_native_main_kwargs}
            if _native_main_kwargs is not None
            else page_source_kwargs
        )
        o = flash_decode_bnsd_with_gqa_share_sparse(
            q=q,
            sink=None,
            k_cache_bnsd=k_bnsd,
            v_cache_bnsd=v_bnsd,
            **_vmain_kwargs,
            seq_lens=per_query_seq_lens,
            block_size=page_size,
            topk_idx=topk_idx,
            sm_scale=head_dim**-0.5,
            use_native=_native_attn_enabled(),
        )
        return idx_o, o

    def _build_prefill_meta(
        self,
        forward_batch: ForwardBatch,
        cu_seqlens: torch.Tensor,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        device,
        page_size: int,
        num_pages: int,
        total_q: int,
    ) -> SimpleNamespace:
        """Build layer-invariant prefill metadata once per forward.
        Depends only on batch shape + req_to_token (invariant across layers).
        per_query_req is the direct-page-lookup map, kept live (no per-query table).
        """
        seq_lens_l = seq_lens.to(device=device, dtype=torch.long)
        prefix_lens_l = prefix_lens.to(device=device, dtype=torch.long)
        cu_q = cu_seqlens.to(device=device, dtype=torch.long)
        extend_lens = (seq_lens_l - prefix_lens_l).clamp(min=0)  # [bs]
        per_query_req = forward_batch.req_pool_indices.long().repeat_interleave(
            extend_lens
        )  # [total_q]
        # Query j of request r sits at position prefix_r + j and causally attends to
        # KV[0 : prefix_r + j + 1], so its seq_len = prefix_r + j + 1.
        per_query_prefix = prefix_lens_l.repeat_interleave(extend_lens)  # [total_q]
        per_query_within = torch.arange(
            total_q, device=device, dtype=torch.long
        ) - cu_q[:-1].repeat_interleave(
            extend_lens
        )  # 0-indexed within each request
        per_query_seq_lens = (per_query_prefix + per_query_within + 1).to(torch.int32)

        max_seqlen = (
            int(self._max_seqlen_k)
            if self._max_seqlen_k
            else int(per_query_seq_lens.max().item())
        )
        max_blocks = (max_seqlen + page_size - 1) // page_size
        block_size_q = self._choose_block_size_q(max_seqlen)

        # Score-path qblock mappings (layer-invariant), built once per forward.
        from sgl_kernel_npu.indexer.flash_block_score_prefill import (
            _build_qblock_mappings as _build_score_qblock_mappings,
        )

        qblock_mappings = _build_score_qblock_mappings(
            cu_seqlens,
            seq_lens,
            self.req_to_token,
            forward_batch.req_pool_indices,
            block_size_q,
            page_size,
            max_blocks,
            device,
        )

        # FIA prep workspace (layer-invariant shape, reused across layers).
        topk1 = self.topk_blocks + 1
        fia_block_table_ws = torch.empty(
            (total_q, topk1), dtype=torch.int32, device=device
        )
        fia_actual_kvlen_ws = torch.empty((total_q,), dtype=torch.int32, device=device)

        return SimpleNamespace(
            per_query_req=per_query_req,
            # Pre-cast int32 for the FIA prep kernel (avoids a per-layer cast).
            per_query_req_i32=per_query_req.to(torch.int32),
            per_query_seq_lens=per_query_seq_lens,
            max_seqlen=max_seqlen,
            max_blocks=max_blocks,
            block_size_q=block_size_q,
            qblock_mappings=qblock_mappings,
            fia_block_table_ws=fia_block_table_ws,
            fia_actual_kvlen_ws=fia_actual_kvlen_ws,
        )

    def _forward_npu_triton_prefill(
        self,
        q: torch.Tensor,  # [total_extend_tokens, num_q_heads, head_dim]
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        idx_q: torch.Tensor,  # [total_extend_tokens, num_idx_heads, idx_dim]
        idx_k_cache: torch.Tensor,
        idx_v_cache: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        cu_seqlens: torch.Tensor,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        # Prefill main-attention launch tuning (decode path is unaffected).
        main_num_warps: int = 4,
        main_num_stages: int = 2,
        # Fuse this many selected blocks per loop step of the main kernel. Same
        # block set per query -> same math, only online-softmax regrouping. Use
        # num_stages=1 when >1 (larger K/V tiles pressure the UB).
        main_blocks_per_step: int = 1,
    ):
        """NPU block-sparse PREFILL via the ported triton decode kernels.
        Generalizes verify to variable per-request extend lengths: each token becomes
        a per-query row with a causal seq_len; decode kernels attend selected blocks.
        """
        from sgl_kernel_npu.attention.gqa_share_sparse_attention import (
            flash_decode_bnsd_with_gqa_share_sparse,
        )

        page_size = self.page_size  # == block_size_k
        num_q_heads = q.shape[1]
        head_dim = q.shape[2]
        num_idx_heads = idx_q.shape[1]
        idx_dim = idx_q.shape[2]
        total_q = q.shape[0]

        k_bnsd, v_bnsd, num_pages, num_kv_heads, head_dim = _kv_cache_to_bnsd(
            k_cache, v_cache, page_size
        )
        idx_k_bnsd, idx_v_bnsd, idx_kv_heads, idx_dim = _idx_cache_to_bnsd(
            idx_k_cache, idx_v_cache, page_size
        )

        # Layer-invariant metadata: built once per forward (first layer builds).
        meta = self._prefill_meta
        if meta is None:
            # Graph setup runs before the backend starts the actual CUDA graph
            # capture.  If this helper is nevertheless reached from inside a
            # capture, leave allocation to the legacy graph-safe fallback.
            try:
                if torch.cuda.is_current_stream_capturing():
                    return None
            except Exception:
                pass
            meta = self._build_prefill_meta(
                forward_batch,
                cu_seqlens,
                seq_lens,
                prefix_lens,
                q.device,
                page_size,
                num_pages,
                total_q,
            )
            self._prefill_meta = meta
        per_query_seq_lens = meta.per_query_seq_lens
        max_seqlen = meta.max_seqlen
        max_blocks = meta.max_blocks
        block_size_q = meta.block_size_q
        per_query_req = meta.per_query_req

        disable_index_value = idx_v_cache is None

        # 1) indexer: score idx_k + index attention + topk (init/local=0).
        # Batched varlen indexer tiles queries into block_size_q blocks and
        # scores every query-block x kv-block in one 2D dot. Fused topk +
        # causal-local append yields [..., topk+1]; the prepare helper skips
        # the duplicate append.
        from sgl_kernel_npu.indexer.flash_block_score_prefill import (
            flash_prefill_bnsd_indexer,
        )
        from sgl_kernel_npu.indexer.flash_block_score_prefill import (
            flash_prefill_bnsd_with_topk_idx as _flash_prefill_score_topk,
        )

        topk_per_query_seq_lens = per_query_seq_lens

        if disable_index_value:
            idx_o = None
            topk_idx = _flash_prefill_score_topk(
                idx_q,
                idx_k_bnsd,
                cu_seqlens,
                seq_lens,
                self.req_to_token,
                forward_batch.req_pool_indices,
                block_size_q,
                page_size,
                self.topk_blocks,
                idx_dim**-0.5,
                self.score_type,
                qblock_mappings=meta.qblock_mappings,
                per_query_seq_lens=topk_per_query_seq_lens,
            )
        else:
            idx_o, topk_idx = flash_prefill_bnsd_indexer(
                idx_q,
                idx_k_bnsd,
                idx_v_bnsd,
                cu_seqlens,
                seq_lens,
                self.req_to_token,
                forward_batch.req_pool_indices,
                block_size_q,
                page_size,
                self.topk_blocks,
                idx_dim**-0.5,
                self.score_type,
                qblock_mappings=meta.qblock_mappings,
                per_query_seq_lens=topk_per_query_seq_lens,
            )

        # 2) Reduce heads and append forced blocks in the GQA kernel layout.
        topk_idx = self._prepare_npu_triton_topk_idx(
            topk_idx,
            per_query_seq_lens,
            num_idx_heads,
            num_kv_heads,
            max_blocks,
        )
        # No range/dtype guard needed: _prepare_npu_triton_topk_idx emits {-1} U
        # [0, max_blocks-1] as int32 on both paths, and the main kernel masks
        # logical_block < 0 and sanitizes physical ids to [0, num_pages-1].

        # 4) main sparse attention over the selected blocks.
        # BPS>1 fuses blocks per step of the decode-main kernel.
        main_bps = int(
            os.environ.get(
                "SGLANG_MINIMAX_NPU_PREFILL_MAIN_BPS", str(main_blocks_per_step)
            )
        )
        main_ns = main_num_stages if main_bps == 1 else min(main_num_stages, 1)

        def _decode_main():
            # Use the request-token map directly in the decode-main kernel.  This
            # avoids materializing a [total_q, max_blocks] page table for every
            # sparse layer and keeps the per-query mapping live for graph replay.
            return flash_decode_bnsd_with_gqa_share_sparse(
                q=q,
                sink=None,
                k_cache_bnsd=k_bnsd,
                v_cache_bnsd=v_bnsd,
                block_table=None,
                req_to_token=self.req_to_token,
                req_pool_indices=per_query_req,
                max_num_blocks=max_blocks,
                num_pages=num_pages,
                sanitize_page_ids=True,
                seq_lens=per_query_seq_lens,
                block_size=page_size,
                topk_idx=topk_idx,
                sm_scale=head_dim**-0.5,
                topk_blocks_per_step=main_bps,
                num_warps=main_num_warps,
                num_stages=main_ns,
            )

        def _fia_main():
            # Native Ascend FA (FIA) with a per-query custom block_table.
            from sgl_kernel_npu.attention.fia_blockq_attention import (
                flash_prefill_bnsd_blockq_sparse_fia,
            )

            return flash_prefill_bnsd_blockq_sparse_fia(
                q=q,
                k_cache_bnsd=k_bnsd,
                v_cache_bnsd=v_bnsd,
                topk_idx=topk_idx,
                seq_lens=per_query_seq_lens,
                per_query_req=meta.per_query_req_i32,
                req_to_token=self.req_to_token,
                block_size=page_size,
                sm_scale=head_dim**-0.5,
                num_pages=num_pages,
                max_num_blocks=max_blocks,
                block_table_out=meta.fia_block_table_ws,
                actual_kvlen_out=meta.fia_actual_kvlen_ws,
            )

        use_fia = envs.SGLANG_MINIMAX_NPU_PREFILL_FIA.get() and num_kv_heads == 1
        o = _fia_main() if use_fia else _decode_main()

        return idx_o, o

    @staticmethod
    def _is_sparse_kv_cached_by_fusion(
        forward_batch: ForwardBatch, layer_id: int
    ) -> bool:
        layer_ids = forward_batch.minimax_m3_precached_sparse_layers
        return layer_ids is not None and layer_id in layer_ids

    def forward(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if forward_batch.forward_mode.is_idle():
            idx_q = kwargs.get("idx_q")
            num_idx_heads = idx_q.shape[1]
            disable_value = layer.layer_id in self.disable_value_layer_ids
            idx_out: Optional[torch.Tensor] = (
                None
                if disable_value
                else q.new_zeros(q.shape[0], num_idx_heads * self.idx_head_dim)
            )
            out = q.new_zeros(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
            return idx_out, out
        else:
            return super().forward(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

    def _resolve_extend_meta(self, forward_batch: ForwardBatch, q: torch.Tensor):
        """Return (cu_seqlens, seq_lens, prefix_lens); NPU caches per-forward casts."""
        # TARGET_VERIFY may omit extend_seq_lens (seq_lens=prefix+draft) on
        # both NPU and GPU speculative paths. Reconstruct the per-sequence
        # draft lengths and prefix lengths needed by sparse prefill attention.
        if (
            forward_batch.forward_mode.is_target_verify()
            and forward_batch.extend_seq_lens is None
        ):
            _bs = forward_batch.seq_lens.shape[0]
            _ndt = self.speculative_num_draft_tokens or (q.shape[0] // max(_bs, 1))
            forward_batch.extend_seq_lens = torch.full(
                (_bs,),
                int(_ndt),
                dtype=torch.int32,
                device=forward_batch.seq_lens.device,
            )
            forward_batch.extend_seq_lens_cpu = [int(_ndt)] * _bs
            if forward_batch.extend_prefix_lens is None:
                forward_batch.extend_prefix_lens = (
                    forward_batch.seq_lens.to(torch.int32) - int(_ndt)
                ).clamp(min=0)

        # NPU cache hit (same forward_batch).
        if (
            self.is_npu
            and self._extend_meta_key == id(forward_batch)
            and self._extend_meta is not None
        ):
            m = self._extend_meta
            return m.cu_seqlens, m.seq_lens, m.prefix_lens

        cu_seqlens = torch.cat(
            [
                torch.zeros(
                    1, dtype=torch.int32, device=forward_batch.extend_seq_lens.device
                ),
                forward_batch.extend_seq_lens.to(torch.int32).cumsum(0).to(torch.int32),
            ]
        )
        seq_lens = forward_batch.seq_lens.to(torch.int32)
        if forward_batch.extend_prefix_lens is not None:
            prefix_lens = forward_batch.extend_prefix_lens.to(torch.int32)
        else:
            prefix_lens = torch.zeros_like(seq_lens)

        # NPU cache write.
        if self.is_npu:
            self._extend_meta = SimpleNamespace(
                cu_seqlens=cu_seqlens, seq_lens=seq_lens, prefix_lens=prefix_lens
            )
            self._extend_meta_key = id(forward_batch)
        return cu_seqlens, seq_lens, prefix_lens

    def _get_flash_mla_prefill_index_meta(
        self,
        cache_key: str,
        cu_seqlens: torch.Tensor,
        seq_lens: torch.Tensor,
        q_lens: torch.Tensor,
        q_lens_cpu,
        num_queries: int,
        device: torch.device,
    ) -> SimpleNamespace:
        """Return forward-local prefix/K-end metadata for one ragged segment."""
        if not (
            self.cache_flash_mla_prefill_metadata and self.use_flash_mla_gfx938_indexer
        ):
            return SimpleNamespace(prefix_lens=seq_lens - q_lens, k_end=None)

        meta = self._flash_mla_prefill_index_meta.get(cache_key)
        if meta is not None:
            if meta.num_queries != int(num_queries) or meta.q_lens_cpu != tuple(
                int(x) for x in q_lens_cpu
            ):
                raise RuntimeError(
                    "MiniMax FlashMLA forward-local metadata cache shape changed "
                    f"for {cache_key}: cached rows={meta.num_queries}, "
                    f"current rows={num_queries}"
                )
            return meta

        from sglang.srt.layers.attention.minimax_sparse_ops.flash_mla_gfx938 import (
            build_flash_mla_sparse_prefill_k_end,
        )

        prefix_lens = seq_lens - q_lens
        k_end = build_flash_mla_sparse_prefill_k_end(
            cu_seqlens,
            prefix_lens,
            q_lens,
            num_queries,
            device,
        )
        meta = SimpleNamespace(
            prefix_lens=prefix_lens,
            k_end=k_end,
            num_queries=int(num_queries),
            q_lens_cpu=tuple(int(x) for x in q_lens_cpu),
        )
        self._flash_mla_prefill_index_meta[cache_key] = meta
        return meta

    def _materialize_query_sharded_cp_kv(
        self,
        forward_batch: ForwardBatch,
        k: torch.Tensor,
        v: torch.Tensor,
        idx_k: Optional[torch.Tensor],
        idx_v: Optional[torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """All-gather one CP-local MiniMax KV payload in a single collective."""
        strategy = get_cp_strategy()
        if strategy is None or strategy.name != "zigzag":
            raise RuntimeError(
                "MiniMax query-sharded CP requires CP-v2 with --cp-strategy zigzag"
            )

        if idx_v is not None and idx_k is None:
            raise RuntimeError("MiniMax index V cannot be materialized without index K")
        has_idx_k = idx_k is not None
        parts = [k, v]
        if idx_k is not None:
            parts.append(idx_k)
        if idx_v is not None:
            parts.append(idx_v)
        if len({part.dtype for part in parts}) != 1:
            raise RuntimeError(
                "MiniMax query-sharded CP requires Main/Index projection tensors "
                "to share a dtype before KV-cache storage"
            )

        shapes = [tuple(part.shape[1:]) for part in parts]
        # Derive row widths from shape rather than indexing row zero. Reuse the
        # packed workspace and fill it in place to remove one allocation per
        # sparse layer while preserving the single all-gather.
        local_rows = int(k.shape[0])
        if any(int(part.shape[0]) != local_rows for part in parts):
            raise RuntimeError(
                "MiniMax query-sharded CP payloads must have matching row counts"
            )
        flat_parts = [part.flatten(1) for part in parts]
        widths = [flat.shape[1] for flat in flat_parts]
        total_width = sum(widths)
        pack_key = (k.device.index, k.dtype, tuple(widths))
        packed = self._query_sharded_cp_pack_buffers.get(pack_key)
        if packed is None or packed.shape[0] < local_rows:
            capacity = max(local_rows, 2 * packed.shape[0] if packed is not None else 0)
            packed = torch.empty(
                (capacity, total_width), dtype=k.dtype, device=k.device
            )
            self._query_sharded_cp_pack_buffers[pack_key] = packed
        # Use one output-backed cat launch instead of one copy launch per
        # projection. The destination is a persistent workspace, so this does
        # not add an allocation to the layer hot path.
        torch.cat(flat_parts, dim=1, out=packed[:local_rows])
        # This result is consumed immediately by the current-stream KV stores,
        # so MiniMax can opt into the explicitly ephemeral reusable workspace.
        packed_full = strategy.gather_kv_cache_reusable(
            packed[:local_rows], forward_batch, torch.cuda.current_stream()
        )
        return self._restore_query_sharded_cp_kv(
            packed_full,
            shapes,
            widths,
            has_idx_k=has_idx_k,
            has_idx_v=idx_v is not None,
        )

    def _restore_query_sharded_cp_kv(
        self,
        packed_full: torch.Tensor,
        shapes: list[tuple],
        widths: list[int],
        has_idx_k: bool,
        has_idx_v: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        full_rows = int(packed_full.shape[0])

        # In the normal BF16 KV-cache path, packed_full has a contiguous inner
        # dimension and an arbitrary row stride. The store kernels already
        # carry row strides, so expose each payload as a view and avoid four
        # full-size device-to-device restore copies. Quantized pools retain
        # the old contiguous restore path because their quantizers may mutate
        # or reinterpret the source tensor.
        index_pool = None
        if has_idx_k:
            index_pool = (
                self.kv_pool.index_kv_pool if has_idx_v else self.kv_pool.index_k_pool
            )
        # ``MHATokenToKOnlyPool`` is an unquantized raw-storage pool and does
        # not expose the quant-method based property used by the regular MHA
        # pool.  Fall back to its storage dtype contract instead of assuming
        # every index-pool implementation has ``is_quantized_kv_cache``.
        index_pool_is_quantized = bool(
            index_pool is not None
            and (
                getattr(index_pool, "is_quantized_kv_cache")
                if hasattr(index_pool, "is_quantized_kv_cache")
                else index_pool.store_dtype != index_pool.dtype
            )
        )
        zero_copy_restore = (
            not self.kv_pool.main_pool.is_quantized_kv_cache
            and not index_pool_is_quantized
        )
        if zero_copy_restore:
            views = []
            offset = 0
            for width, shape in zip(widths, shapes):
                views.append(
                    packed_full[:, offset : offset + width].view(full_rows, *shape)
                )
                offset += width
            return (
                views[0],
                views[1],
                (views[2] if has_idx_k else None),
                (views[3] if has_idx_v else None),
            )

        restore_key = (
            packed_full.device.index,
            packed_full.dtype,
            tuple(shapes),
        )
        restore_buffers = self._query_sharded_cp_restore_buffers.get(restore_key)
        if restore_buffers is None or restore_buffers[0].shape[0] < full_rows:
            old_capacity = (
                int(restore_buffers[0].shape[0]) if restore_buffers is not None else 0
            )
            capacity = max(full_rows, 2 * old_capacity)
            restore_buffers = tuple(
                torch.empty(
                    (capacity, *shape),
                    dtype=packed_full.dtype,
                    device=packed_full.device,
                )
                for shape in shapes
            )
            self._query_sharded_cp_restore_buffers[restore_key] = restore_buffers
        restored = tuple(buf[:full_rows] for buf in restore_buffers)
        offset = 0
        for dst, width, shape in zip(restored, widths, shapes):
            dst.copy_(
                packed_full[:, offset : offset + width].reshape(full_rows, *shape)
            )
            offset += width
        return (
            restored[0],
            restored[1],
            (restored[2] if has_idx_k else None),
            (restored[3] if has_idx_v else None),
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        *,
        idx_q: Optional[torch.Tensor],
        idx_k: Optional[torch.Tensor],
        idx_v: Optional[torch.Tensor],
    ):
        disable_value = layer.layer_id in self.disable_value_layer_ids
        if idx_q is None or idx_k is None:
            raise RuntimeError("MiniMax sparse prefill requires index Q and K")
        cp_active = is_cp_v2_active(forward_batch)
        if cp_active:
            if not self.enable_query_sharded_cp:
                raise RuntimeError(
                    "MiniMax prefill CP is experimental; set "
                    "SGLANG_OPT_USE_MINIMAX_QUERY_SHARDED_CP=1 to enable the "
                    "query-sharded full-KV path"
                )
            if self.is_npu:
                raise RuntimeError(
                    "MiniMax query-sharded CP is currently implemented only for CUDA/HIP"
                )
            k, v, idx_k, idx_v = self._materialize_query_sharded_cp_kv(
                forward_batch, k, v, idx_k, idx_v
            )
            global _minimax_sparse_cp_logged
            if not _minimax_sparse_cp_logged:
                logger.info(
                    "MiniMax query-sharded sparse PCP selected: strategy=zigzag, "
                    "cp_size=%s, local_q=%s, materialized_kv=%s",
                    get_cp_strategy().cp_size,
                    q.shape[0],
                    k.shape[0],
                )
                _minimax_sparse_cp_logged = True
        kv_cached_by_fusion = self._is_sparse_kv_cached_by_fusion(
            forward_batch, layer.layer_id
        )
        # A fused QKNorm/RoPE projection can only have stored this CP rank's
        # local token rows. Query-sharded MiniMax CP needs every rank to own
        # the complete new-chunk sparse KV, so overwrite that partial write
        # with the globally materialized tensors.
        store_loc = forward_batch.out_cache_loc
        if cp_active or not kv_cached_by_fusion:
            if idx_k is None:
                self.kv_pool.set_kv_buffer(
                    layer,
                    store_loc,
                    k,
                    v,
                    layer.k_scale_float,
                    layer.v_scale_float,
                )
            else:
                self.kv_pool.set_fused_kv_index_buffer(
                    layer,
                    store_loc,
                    k,
                    v,
                    idx_k,
                    None if disable_value else idx_v,
                    layer.k_scale_float,
                    layer.v_scale_float,
                    layer.idx_k_scale_float,
                    layer.idx_v_scale_float,
                )
        k_cache, v_cache = self.kv_pool.get_kv_buffer(layer.layer_id)
        if idx_k is None:
            idx_k_cache = None
            idx_v_cache = None
        elif disable_value:
            idx_k_cache = self.kv_pool.get_index_k_buffer(layer.layer_id)
            idx_v_cache = None
        else:
            idx_k_cache, idx_v_cache = self.kv_pool.get_index_kv_buffer(layer.layer_id)

        if not self.is_npu and forward_batch.forward_mode.is_target_verify():
            # EAGLE top-k=1 verification is a packed batch of short causal
            # chains.  Running it through the long-prefill path is both
            # wasteful and unsafe under ROCm graph capture (AITER Top-K is not
            # captureable, while the generic Triton prefill Top-K uses static
            # long-sequence workspaces).  Treat every verify position as a
            # decode query with its own causal sequence length and reuse the
            # graph-safe MiniMax decode score/main-attention kernels.
            bs = int(forward_batch.seq_lens.shape[0])
            spec_info = forward_batch.spec_info
            ndt = int(
                getattr(spec_info, "num_tokens_per_req", 0)
                or self.speculative_num_draft_tokens
                or 0
            )
            original_num_tokens = q.shape[0]
            logical_num_tokens = bs * ndt
            if bs <= 0 or ndt <= 0 or logical_num_tokens > original_num_tokens:
                raise RuntimeError(
                    "MiniMax TARGET_VERIFY expects a uniform packed query "
                    "layout before DP residual padding, got "
                    f"q_rows={original_num_tokens}, batch={bs}, width={ndt}."
                )
            if self.speculative_num_draft_tokens is not None and ndt != int(
                self.speculative_num_draft_tokens
            ):
                raise RuntimeError(
                    "MiniMax TARGET_VERIFY query width does not match "
                    "--speculative-num-draft-tokens: "
                    f"packed={ndt}, configured={self.speculative_num_draft_tokens}."
                )

            # DP MLP synchronization aligns token rows to the attention TP
            # width.  The aligned count need not be divisible by the packed
            # verification width (for example 16 rows for 5 * 3 logical
            # rows), so the final rows are padding rather than another query
            # position.  Sparse verify metadata intentionally describes only
            # complete request groups; trim that residual here and restore it
            # after attention for the rank-coupled MLP collectives.
            if logical_num_tokens < original_num_tokens:
                q = q[:logical_num_tokens]
                idx_q = idx_q[:logical_num_tokens]

            meta = self._get_target_verify_meta(forward_batch, ndt)
            if meta is None:
                per_query_req = forward_batch.req_pool_indices.long().repeat_interleave(
                    ndt
                )
                # Spec-v2 keeps batch.seq_lens at the length *before* this
                # verify iteration; draft cache rows begin at exactly that
                # offset. Therefore causal lengths are prefix+1 .. prefix+ndt.
                prefix_lens = forward_batch.seq_lens.to(torch.long)
                offsets = torch.arange(
                    1,
                    ndt + 1,
                    device=forward_batch.seq_lens.device,
                    dtype=torch.long,
                )
                per_query_seq_lens = (prefix_lens[:, None] + offsets[None, :]).reshape(
                    -1
                )
            else:
                per_query_req = meta.per_query_req
                per_query_seq_lens = meta.per_query_seq_lens

            if self.main_fp8_attn_gemm:
                q = _quant_q_fp8(q, layer.q_scale_float)
            if self.fp8_attn_gemm:
                idx_q = _quant_q_fp8(idx_q, layer.idx_q_scale_float)

            from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
                minimax_sparse_decode,
            )

            idx_o, o = minimax_sparse_decode(
                q,
                None,
                k_cache,
                v_cache,
                idx_q,
                None,
                idx_k_cache,
                idx_v_cache,
                self.req_to_token,
                per_query_req,
                per_query_seq_lens,
                self._max_seqlen_k,
                1,
                self.block_size_k,
                self.topk_blocks,
                self.init_blocks,
                self.local_blocks,
                score_type=self.score_type,
                disable_index_value=disable_value,
                page_size=self.page_size,
                use_msa=False,
                use_flash_mla_gfx938=self._flash_mla_decode_current,
                use_flash_mla_gfx938_indexer=(
                    self.use_flash_mla_gfx938_indexer and self._flash_mla_decode_current
                ),
                flash_mla_page_table=self._flash_mla_page_table,
                flash_mla_sched_meta=(
                    self._get_flash_mla_sched_meta(
                        layer.layer_id, logical_num_tokens, ndt
                    )
                    if self._flash_mla_decode_current
                    else None
                ),
                flash_mla_indices_output=(
                    self._get_flash_mla_indices_output(
                        logical_num_tokens,
                        q.device,
                        decode=True,
                        verify_group_size=ndt,
                    )
                    if self._flash_mla_decode_current
                    else None
                ),
                q_scale=layer.q_scale_float,
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
                idx_q_scale=layer.idx_q_scale_float,
                idx_k_scale=layer.idx_k_scale_float,
                idx_v_scale=layer.idx_v_scale_float,
                verify_group_size=ndt,
            )
            o = o.reshape(logical_num_tokens, -1).contiguous()
            if idx_o is not None:
                idx_o = idx_o.reshape(logical_num_tokens, -1).contiguous()
            if logical_num_tokens < original_num_tokens:
                pad_len = original_num_tokens - logical_num_tokens
                o = torch.cat([o, o.new_zeros(pad_len, o.shape[1])], dim=0)
                if idx_o is not None:
                    idx_o = torch.cat(
                        [idx_o, idx_o.new_zeros(pad_len, idx_o.shape[1])], dim=0
                    )
            return idx_o, o

        cu_seqlens, seq_lens, prefix_lens = self._resolve_extend_meta(forward_batch, q)
        # DP attention pads q beyond real tokens; trim (CPU list avoids a sync).
        if cp_active:
            cp_meta = forward_batch.attn_cp_metadata
            # CP-v2 rewrites per_rank_actual_token to the collective-aligned
            # physical row count.  Sparse attention metadata describes only
            # logical queries, so trim padding with the preserved logical list.
            per_rank_tokens = (
                cp_meta.per_rank_logical_token or cp_meta.per_rank_actual_token
            )
            actual_num_tokens = int(per_rank_tokens[get_cp_strategy().cp_rank])
        elif forward_batch.extend_seq_lens_cpu is not None:
            actual_num_tokens = int(sum(forward_batch.extend_seq_lens_cpu))
        else:
            actual_num_tokens = int(cu_seqlens[-1].item())
        original_num_tokens = q.shape[0]
        if actual_num_tokens < original_num_tokens:
            q = q[:actual_num_tokens]
            if idx_q is not None:
                idx_q = idx_q[:actual_num_tokens]

        if self.is_npu:
            if forward_batch.forward_mode.is_target_verify():
                # TARGET_VERIFY runs under cuda-graph capture; use the
                # capture-safe verify path (decode kernels, no .item()).
                idx_o, o = self._forward_npu_triton_verify(
                    q,
                    k_cache,
                    v_cache,
                    idx_q,
                    idx_k_cache,
                    idx_v_cache,
                    forward_batch,
                    prefix_lens,
                )
            else:
                idx_o, o = self._forward_npu_triton_prefill(
                    q,
                    k_cache,
                    v_cache,
                    idx_q,
                    idx_k_cache,
                    idx_v_cache,
                    forward_batch,
                    cu_seqlens,
                    seq_lens,
                    prefix_lens,
                )
        else:
            # fp8 attention GEMMs: quantize q/idx_q after the KV store (which
            # reads the bf16 k/v) and the DP trim.
            if self.main_fp8_attn_gemm:
                q = _quant_q_fp8(q, layer.q_scale_float)
            if self.fp8_attn_gemm:
                if idx_q is not None:
                    idx_q = _quant_q_fp8(idx_q, layer.idx_q_scale_float)

            # GPU (CUDA/ROCm) sparse path; imported here so NPU never touches it.
            from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
                minimax_sparse_prefill,
            )

            effective_block_size_q = self._select_gpu_prefill_block_size_q()
            # Query-sharded CP rewrites tensor shapes and backend max lengths
            # to rank-local values.  The scheduler's CPU list remains in the
            # original request domain (128K seed versus 16K cache-hit extend),
            # which is the domain used by exact-grouped-main policy bounds.
            logical_max_seqlen_q = max(
                forward_batch.extend_seq_lens_cpu or [self._max_seqlen_q]
            )
            flash_mla_safe_pages = None
            if (
                self.cache_flash_mla_prefill_metadata
                and self.use_flash_mla_gfx938_indexer
            ):
                if self._flash_mla_prefill_safe_pages is None:
                    self._flash_mla_prefill_safe_pages = (
                        self._flash_mla_page_table.clamp_min(0).reshape(-1).contiguous()
                    )
                flash_mla_safe_pages = self._flash_mla_prefill_safe_pages

            def run_sparse_segment(
                segment_q,
                segment_idx_q,
                segment_cu_seqlens,
                segment_seq_lens,
                segment_prefix_lens,
                segment_q_lens_cpu,
                segment_max_q,
                segment_max_k,
                segment_slot_ids=None,
                flash_mla_prefill_k_end=None,
            ):
                return minimax_sparse_prefill(
                    segment_q,
                    k_cache,
                    v_cache,
                    None,
                    segment_idx_q,
                    idx_k_cache,
                    idx_v_cache,
                    None,
                    self.req_to_token,
                    (
                        forward_batch.req_pool_indices
                        if segment_slot_ids is None
                        else segment_slot_ids
                    ),
                    segment_cu_seqlens,
                    segment_seq_lens,
                    segment_prefix_lens,
                    segment_max_q,
                    segment_max_k,
                    effective_block_size_q,
                    self.block_size_k,
                    self.topk_blocks,
                    self.init_blocks,
                    self.local_blocks,
                    score_type=self.score_type,
                    disable_index_value=disable_value,
                    use_msa=self.use_msa,
                    use_flash_mla_gfx938=self.use_flash_mla_gfx938,
                    use_flash_mla_gfx938_indexer=(self.use_flash_mla_gfx938_indexer),
                    flash_mla_page_table=self._flash_mla_page_table,
                    flash_mla_indices_output=(
                        self._get_flash_mla_indices_output(
                            segment_q.shape[0],
                            segment_q.device,
                            decode=False,
                        )
                        if self.use_flash_mla_gfx938
                        else None
                    ),
                    flash_mla_prefill_k_end=flash_mla_prefill_k_end,
                    flash_mla_safe_pages=flash_mla_safe_pages,
                    seqlens_cpu=segment_q_lens_cpu,
                    q_scale=layer.q_scale_float,
                    k_scale=layer.k_scale_float,
                    v_scale=layer.v_scale_float,
                    idx_q_scale=layer.idx_q_scale_float,
                    idx_k_scale=layer.idx_k_scale_float,
                    idx_v_scale=layer.idx_v_scale_float,
                    logical_max_seqlen_q=logical_max_seqlen_q,
                )

            if cp_active:
                cp_meta = forward_batch.attn_cp_metadata
                split_at = cp_meta.total_q_prev_tokens
                if os.environ.get("SGLANG_MINIMAX_COMBINE_CP_SEGMENTS", "0") == "1":
                    # The local tensor is already laid out as all early zigzag
                    # slices followed by all late slices.  Treat those slices
                    # as one 2*batch ragged launch: attention programs remain
                    # request-independent, while score/Top-K/union/main launch
                    # count and the two output cat kernels are halved.
                    combined_cu = torch.cat(
                        [
                            cp_meta.cu_seqlens_q_prev_tensor,
                            cp_meta.cu_seqlens_q_next_tensor[1:] + split_at,
                        ]
                    )
                    combined_seq_lens = torch.cat(
                        [cp_meta.kv_len_prev_tensor, cp_meta.kv_len_next_tensor]
                    )
                    combined_q_lens = torch.cat(
                        [
                            cp_meta.actual_seq_q_prev_tensor,
                            cp_meta.actual_seq_q_next_tensor,
                        ]
                    )
                    combined_slots = torch.cat(
                        [
                            forward_batch.req_pool_indices,
                            forward_batch.req_pool_indices,
                        ]
                    )
                    combined_q_lens_cpu = (
                        cp_meta.actual_seq_q_prev_list + cp_meta.actual_seq_q_next_list
                    )
                    combined_meta = self._get_flash_mla_prefill_index_meta(
                        "cp_combined",
                        combined_cu,
                        combined_seq_lens,
                        combined_q_lens,
                        combined_q_lens_cpu,
                        actual_num_tokens,
                        q.device,
                    )
                    idx_o, o = run_sparse_segment(
                        q[:actual_num_tokens],
                        (None if idx_q is None else idx_q[:actual_num_tokens]),
                        combined_cu,
                        combined_seq_lens,
                        combined_meta.prefix_lens,
                        combined_q_lens_cpu,
                        max(
                            cp_meta.max_seqlen_q_prev,
                            cp_meta.max_seqlen_q_next,
                        ),
                        max(
                            cp_meta.kv_len_prev_list + cp_meta.kv_len_next_list,
                            default=0,
                        ),
                        combined_slots,
                        combined_meta.k_end,
                        "cp_combined",
                    )
                else:
                    segment_specs = (
                        (
                            q[:split_at],
                            None if idx_q is None else idx_q[:split_at],
                            cp_meta.cu_seqlens_q_prev_tensor,
                            cp_meta.kv_len_prev_tensor,
                            cp_meta.actual_seq_q_prev_tensor,
                            cp_meta.actual_seq_q_prev_list,
                            cp_meta.max_seqlen_q_prev,
                            cp_meta.kv_len_prev_list,
                        ),
                        (
                            q[split_at:actual_num_tokens],
                            (
                                None
                                if idx_q is None
                                else idx_q[split_at:actual_num_tokens]
                            ),
                            cp_meta.cu_seqlens_q_next_tensor,
                            cp_meta.kv_len_next_tensor,
                            cp_meta.actual_seq_q_next_tensor,
                            cp_meta.actual_seq_q_next_list,
                            cp_meta.max_seqlen_q_next,
                            cp_meta.kv_len_next_list,
                        ),
                    )

                    def launch_segment(spec, cache_key):
                        (
                            segment_q,
                            segment_idx_q,
                            segment_cu,
                            segment_kv_lens,
                            segment_q_lens,
                            q_lens_cpu,
                            max_q,
                            kv_lens_cpu,
                        ) = spec
                        segment_meta = self._get_flash_mla_prefill_index_meta(
                            cache_key,
                            segment_cu,
                            segment_kv_lens,
                            segment_q_lens,
                            q_lens_cpu,
                            segment_q.shape[0],
                            segment_q.device,
                        )
                        return run_sparse_segment(
                            segment_q,
                            segment_idx_q,
                            segment_cu,
                            segment_kv_lens,
                            segment_meta.prefix_lens,
                            q_lens_cpu,
                            max_q,
                            max(kv_lens_cpu, default=0),
                            flash_mla_prefill_k_end=segment_meta.k_end,
                        )

                    segment_outputs = []
                    if (
                        os.environ.get("SGLANG_MINIMAX_PARALLEL_CP_SEGMENTS", "0")
                        == "1"
                    ):
                        # Keep the two zigzag segments as independent ragged
                        # batches (the exact numerical contract), but submit
                        # them to separate streams.  A single batch-1 score
                        # grid under-fills gfx938; concurrent grids recover
                        # occupancy without changing sequence metadata.
                        current_stream = torch.cuda.current_stream(q.device)
                        streams = getattr(self, "_minimax_cp_segment_streams", None)
                        if streams is None:
                            streams = tuple(
                                torch.cuda.Stream(device=q.device) for _ in range(2)
                            )
                            self._minimax_cp_segment_streams = streams
                        for stream in streams:
                            stream.wait_stream(current_stream)
                        for segment_id, (spec, stream) in enumerate(
                            zip(segment_specs, streams)
                        ):
                            with torch.cuda.stream(stream):
                                segment_outputs.append(
                                    launch_segment(spec, f"cp_segment_{segment_id}")
                                )
                        for stream in streams:
                            current_stream.wait_stream(stream)
                        # The segment outputs are allocated on their auxiliary
                        # streams but consumed by the concatenation below on
                        # the current stream.  Waiting establishes execution
                        # order; record_stream additionally prevents the CUDA
                        # caching allocator from recycling an auxiliary-stream
                        # allocation before the current-stream consumer has
                        # finished reading it.  This matters under the high
                        # allocator pressure of FlashMLA's score/index buffers.
                        for idx_part, out_part in segment_outputs:
                            out_part.record_stream(current_stream)
                            if idx_part is not None:
                                idx_part.record_stream(current_stream)
                    else:
                        segment_outputs = [
                            launch_segment(spec, f"cp_segment_{segment_id}")
                            for segment_id, spec in enumerate(segment_specs)
                        ]
                    idx_parts, out_parts = zip(*segment_outputs)
                    idx_o = (
                        None
                        if any(part is None for part in idx_parts)
                        else torch.cat(idx_parts, dim=0)
                    )
                    o = torch.cat(out_parts, dim=0)
            else:
                full_q_lens = (
                    forward_batch.extend_seq_lens
                    if forward_batch.extend_seq_lens is not None
                    else seq_lens - prefix_lens
                )
                full_meta = self._get_flash_mla_prefill_index_meta(
                    "full",
                    cu_seqlens,
                    seq_lens,
                    full_q_lens,
                    forward_batch.extend_seq_lens_cpu or [q.shape[0]],
                    q.shape[0],
                    q.device,
                )
                idx_o, o = run_sparse_segment(
                    q,
                    idx_q,
                    cu_seqlens,
                    seq_lens,
                    full_meta.prefix_lens,
                    forward_batch.extend_seq_lens_cpu or [q.shape[0]],
                    self._max_seqlen_q,
                    self._max_seqlen_k,
                    flash_mla_prefill_k_end=full_meta.k_end,
                )
        if actual_num_tokens < original_num_tokens:
            pad_len = original_num_tokens - actual_num_tokens
            o = torch.cat([o, o.new_zeros(pad_len, *o.shape[1:])], dim=0)
            if idx_o is not None:
                idx_o = torch.cat(
                    [idx_o, idx_o.new_zeros(pad_len, *idx_o.shape[1:])], dim=0
                )

        return (
            (
                None
                if idx_o is None
                else idx_o.reshape(original_num_tokens, -1).contiguous()
            ),
            o.reshape(original_num_tokens, -1).contiguous(),
        )

    def _dense_sparse_main_decode(
        self,
        q: torch.Tensor,
        page_table: torch.Tensor,
        real_seq_lens: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

        if isinstance(self.dense_backend, TRTLLMHAAttnBackend):
            import flashinfer

            ps = self.page_size
            nkv = 1
            head_dim = q.size(-1)
            # [max_slots, nkv, D] -> [num_pages, page_size, nkv, D]
            #                     -> [num_pages, nkv, page_size, D] (HND, trtllm default)
            kc = k_cache.view(-1, ps, nkv, head_dim).permute(0, 2, 1, 3)
            vc = v_cache.view(-1, ps, nkv, head_dim).permute(0, 2, 1, 3)
            return flashinfer.decode.trtllm_batch_decode_with_kv_cache(  # type: ignore
                query=q.contiguous(),
                kv_cache=(kc, vc),
                workspace_buffer=self.dense_backend.workspace_buffer,
                block_tables=page_table,
                seq_lens=real_seq_lens,
                max_seq_len=self.topk_blocks * self.block_size_k,
                bmm1_scale=layer.scaling,
                bmm2_scale=1.0,
            )
        raise NotImplementedError(
            "dense sparse decode currently supports trtllm_mha only (fa3 is TODO)"
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        *,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
        **kwargs,
    ):
        assert len(kwargs) == 0
        disable_value = layer.layer_id in self.disable_value_layer_ids
        store_loc = forward_batch.out_cache_loc
        self.kv_pool.set_fused_kv_index_buffer(
            layer,
            store_loc,
            k,
            v,
            idx_k,
            None if disable_value else idx_v,
            layer.k_scale_float,
            layer.v_scale_float,
            layer.idx_k_scale_float,
            layer.idx_v_scale_float,
        )
        k_cache, v_cache = self.kv_pool.get_kv_buffer(layer.layer_id)
        if disable_value:
            idx_k_cache = self.kv_pool.get_index_k_buffer(layer.layer_id)
            idx_v_cache = None
        else:
            idx_k_cache, idx_v_cache = self.kv_pool.get_index_kv_buffer(layer.layer_id)

        attn_fn = None
        if self.use_dense_sparse_decode and k_cache.shape[1] == 1:

            def attn_fn(main_q, page_table, real_seq_lens):
                return self._dense_sparse_main_decode(
                    main_q,
                    page_table,
                    real_seq_lens,
                    k_cache,
                    v_cache,
                    layer,
                    forward_batch,
                )

        msa_kv_indices = msa_plan = None
        if self._use_msa_decode and attn_fn is None:
            if self._msa_dec_meta is not None:
                msa_kv_indices, msa_plan = self._msa_dec_meta
            elif q.shape[0] > 0:
                # Rebuilding the plan inline would run host-side code inside
                # CUDA-graph capture; fail loudly instead.
                raise RuntimeError(
                    "MSA decode metadata missing: init_forward_metadata_out_graph "
                    "did not prepare the plan for this forward (gate mismatch)."
                )

        if self.is_npu:
            idx_o, o = self._forward_npu_triton_decode(
                q,
                k_cache,
                v_cache,
                idx_q,
                idx_k_cache,
                idx_v_cache,
                forward_batch,
            )
        else:
            # fp8 attn-GEMM: quantize q/idx_q after the KV store (reads bf16 k/v).
            if self.main_fp8_attn_gemm:
                q = _quant_q_fp8(q, layer.q_scale_float)
            if self.fp8_attn_gemm:
                idx_q = _quant_q_fp8(idx_q, layer.idx_q_scale_float)

            # GPU (CUDA/ROCm) sparse path; imported here so NPU never touches it.
            from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
                minimax_sparse_decode,
            )

            idx_o, o = minimax_sparse_decode(
                q,
                None,
                k_cache,
                v_cache,
                idx_q,
                None,
                idx_k_cache,
                idx_v_cache,
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                self._max_seqlen_k,
                1,
                self.block_size_k,
                self.topk_blocks,
                self.init_blocks,
                self.local_blocks,
                score_type=self.score_type,
                disable_index_value=disable_value,
                dense_main_attn_fn=attn_fn,
                page_size=self.page_size,
                use_msa=self._use_msa_decode,
                use_flash_mla_gfx938=self._flash_mla_decode_current,
                use_flash_mla_gfx938_indexer=(
                    self.use_flash_mla_gfx938_indexer and self._flash_mla_decode_current
                ),
                flash_mla_page_table=self._flash_mla_page_table,
                flash_mla_sched_meta=(
                    self._get_flash_mla_sched_meta(layer.layer_id, q.shape[0])
                    if self._flash_mla_decode_current
                    else None
                ),
                flash_mla_indices_output=(
                    self._get_flash_mla_indices_output(
                        q.shape[0], q.device, decode=True
                    )
                    if self._flash_mla_decode_current
                    else None
                ),
                msa_kv_indices=msa_kv_indices,
                msa_plan=msa_plan,
                q_scale=layer.q_scale_float,
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
                idx_q_scale=layer.idx_q_scale_float,
                idx_k_scale=layer.idx_k_scale_float,
                idx_v_scale=layer.idx_v_scale_float,
            )
        return (
            None if idx_o is None else idx_o.reshape(q.shape[0], -1).contiguous(),
            o.reshape(q.shape[0], -1).contiguous(),
        )


class MiniMaxHybridAttnBackend(AttentionBackend):
    """Combines a dense backend and a sparse backend, routing by call site."""

    def __init__(
        self,
        dense_backend: AttentionBackend,
        sparse_backend: MiniMaxSparseAttnBackend,
        sparse_layer_ids: list[int],
    ):
        self.dense = dense_backend
        self.sparse = sparse_backend
        self.sparse_layer_ids = sparse_layer_ids
        # CP strategies obtain the active KV pool through get_attn_backend().
        # Expose the shared runner pool on this outer hybrid wrapper just like
        # HybridAttnBackend does.  Silently forwarding different child pools
        # would make CP materialize dense-layer KV into the wrong cache.
        dense_kv_pool = getattr(dense_backend, "token_to_kv_pool", None)
        sparse_kv_pool = sparse_backend.token_to_kv_pool
        if dense_kv_pool is not None and dense_kv_pool is not sparse_kv_pool:
            raise RuntimeError(
                "MiniMax dense and sparse attention backends must share the "
                "same token_to_kv_pool"
            )
        self.token_to_kv_pool = sparse_kv_pool
        self.req_to_token_pool = getattr(dense_backend, "req_to_token_pool", None)
        # Let the sparse decode reuse the dense paged backend (page table + workspace).
        self.sparse.dense_backend = dense_backend
        self.extend_dummy_seqs_capped_by_req_pool = getattr(
            dense_backend, "extend_dummy_seqs_capped_by_req_pool", False
        ) or getattr(sparse_backend, "extend_dummy_seqs_capped_by_req_pool", False)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # delegate so the dense (FlashInfer) backend keeps its own eager init.
        self.sparse.init_forward_metadata(forward_batch)
        self.dense.init_forward_metadata(forward_batch)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        self.sparse.init_forward_metadata_out_graph(forward_batch, in_capture)
        self.dense.init_forward_metadata_out_graph(forward_batch, in_capture)

    def shared_read_ends(self, fm: ForwardMode) -> SharedReadEnds:
        return SharedReadEnds.max_of(
            b.shared_read_ends(fm) for b in (self.sparse, self.dense)
        )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        self.sparse.init_forward_metadata_in_graph(forward_batch)
        self.dense.init_forward_metadata_in_graph(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.dense.init_cuda_graph_state(max_bs, max_num_tokens)
        self.sparse.init_cuda_graph_state(max_bs, max_num_tokens)

    def get_cuda_graph_seq_len_fill_value(self):
        return self.sparse.get_cuda_graph_seq_len_fill_value()

    def get_verify_buffers_to_fill_after_draft(self):
        # EAGLE3 verify buffer interface: the dense (ascend) backend owns the
        # tree-mask/position buffers consumed by the verify forward. The base
        # AttentionBackend raises NotImplementedError, so delegate to dense.
        return self.dense.get_verify_buffers_to_fill_after_draft()

    def update_verify_buffers_to_fill_after_draft(self, spec_info, cuda_graph_bs=None):
        return self.dense.update_verify_buffers_to_fill_after_draft(
            spec_info, cuda_graph_bs
        )

    def forward(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if layer.layer_id in self.sparse_layer_ids:
            return self.sparse.forward(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

        # DP attention pads q to an even length but flashinfer builds qo_indptr from
        # extend_seq_lens, so padded q.shape[0] != qo_indptr[-1] and paged-prefill
        # raises. Trim q and re-pad output; k/v stay untrimmed so KV-cache writes
        # align with out_cache_loc.
        mode = forward_batch.forward_mode
        if mode.is_extend() and forward_batch.extend_seq_lens_cpu is not None:
            actual_num_tokens = int(sum(forward_batch.extend_seq_lens_cpu))
            original_num_tokens = q.shape[0]
            if actual_num_tokens < original_num_tokens:
                o = self.dense.forward(
                    q[:actual_num_tokens],
                    k,
                    v,
                    layer,
                    forward_batch,
                    save_kv_cache,
                    **kwargs,
                )
                pad_len = original_num_tokens - actual_num_tokens
                return torch.cat([o, o.new_zeros(pad_len, *o.shape[1:])], dim=0)

        return self.dense.forward(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward_extend(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if layer.layer_id in self.sparse_layer_ids:
            return self.sparse.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        else:
            return self.dense.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

    def forward_decode(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if layer.layer_id in self.sparse_layer_ids:
            return self.sparse.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        else:
            return self.dense.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
