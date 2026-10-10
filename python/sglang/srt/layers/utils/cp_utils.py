"""Import-only shims for deprecated NPU/MUSA backends awaiting CP refactoring.

The legacy CP algorithms have been removed. These names keep the retained
NPU/MUSA attention backends importable for non-CP inference; calling them fails.
"""


def _deprecated_platform_cp():
    raise ValueError(
        "Prefill CP on NPU/MUSA is deprecated; CP support will be refactored soon."
    )


def cp_all_gather_rerange_output(input_tensor, cp_size, forward_batch, stream):
    _deprecated_platform_cp()


def cp_all_gather_rerange_kv_cache(input_tensor, cp_size, forward_batch, stream):
    _deprecated_platform_cp()


def cp_allgather_and_save_kv_cache(forward_batch, layer, k, v, cp_size, swa_loc=None):
    _deprecated_platform_cp()

from sglang.srt.utils import is_hcu

_is_hcu = is_hcu()
if _is_hcu:
    from sglang.srt.layers.utils.hcu_cp_utils import (
        ContextParallelMetadata,
        is_prefill_context_parallel_enabled,
        is_mla_prefill_cp_enabled,
        mla_use_prefill_cp,
        can_cp_split,
        cp_split_and_rebuild_data,
        cp_split_and_rebuild_position,
        cp_round_robin_input_ids,
        cp_all_gather_reorganized_into_tensor,
        cp_all_gather_reorganized_into_tensor_kv_cache,
        cp_all_gather_rerange_launch,
        cp_all_gather_rerange_finish,
        cp_all_gather_rerange_output,
        cp_all_gather_rerange_kv_cache,
        cp_allgather_and_save_kv_cache,
        prepare_context_parallel_metadata,
    )
