"""DeepSeek V4.1 wrappers around the LayerSplit pool API."""

from sglang.srt.mem_cache.cp_cache_layer_split.deepseek_v4_pool import (
    CpCacheLayerSplitDeepSeekV4TokenToKVPool,
)
from sglang.srt.model_executor.forward_context import get_attn_backend


def is_cp_cache_layer_split_deepseek_v4_pool(pool) -> bool:
    return isinstance(pool, CpCacheLayerSplitDeepSeekV4TokenToKVPool)


def _should_sync_cp_cache_layer_split(pool, forward_batch) -> bool:
    if not is_cp_cache_layer_split_deepseek_v4_pool(pool):
        return False
    # Even short extend/warmup batches without attn_cp_metadata still read the
    # layer-sharded cache. All CP ranks participate in the same extend step.
    return (
        forward_batch is not None
        and forward_batch.forward_mode.is_context_parallel_extend()
    )


def maybe_prepare_cp_cache_layer_split_forward(pool, forward_batch) -> None:
    if is_cp_cache_layer_split_deepseek_v4_pool(pool):
        pool.prepare_forward(
            require_prefetched_reads=forward_batch.forward_mode.is_context_parallel_extend()
        )


def _get_core_attn_metadata():
    attn_backend = get_attn_backend()
    metadata = getattr(attn_backend, "forward_metadata", None)
    core_metadata = getattr(metadata, "core_attn_metadata", None)
    if core_metadata is None:
        raise RuntimeError("CP Cache LayerSplit requires DSV4 core attention metadata")
    return core_metadata


def maybe_prefetch_cp_kv_swa(pool, layer_id: int, forward_batch=None) -> None:
    if _should_sync_cp_cache_layer_split(pool, forward_batch):
        metadata = get_attn_backend().forward_metadata
        cache = getattr(metadata, "sparse_prefill_cache", None)
        # Sparse prefill dequantizes the full gathered SWA span, not only the
        # per-query attention indices.
        pool.prefetch_swa_layer(
            layer_id,
            _get_core_attn_metadata().swa_page_indices,
            read_indices=cache.swa_token_ids if cache is not None else None,
            num_reqs=cache.num_reqs if cache is not None else None,
        )


def maybe_wait_cp_kv_swa_prefetch(pool, layer_id: int, forward_batch=None) -> None:
    if _should_sync_cp_cache_layer_split(pool, forward_batch):
        pool.wait_swa_prefetch(layer_id)


def maybe_prefetch_cp_kv_extra(pool, layer_id: int, forward_batch=None) -> None:
    if _should_sync_cp_cache_layer_split(pool, forward_batch):
        ratio = pool.compression_ratios[layer_id]
        if ratio in (1, 2):
            metadata = get_attn_backend().forward_metadata
            cache = getattr(metadata, "sparse_prefill_cache", None)
            core = _get_core_attn_metadata()
            read_indices = None
            if cache is not None:
                # The workspace dequantizes every compressed position before
                # TopK selects rows; broadcasting only TopK pages is unsafe.
                read_indices = cache.ensure_compressed(
                    ratio, core.page_table, pool.kv_pools[ratio].page_size
                ).flat_token_ids
            pool.prefetch_low_ratio_extra_layer(
                layer_id,
                core.sparse_page_indices(ratio),
                read_indices=read_indices,
                num_reqs=cache.num_reqs if cache is not None else None,
            )
