"""HCU GLM state registration using main's per-component transfer metadata."""

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool


def append_hcu_dsa_state(kv_args, token_pool, draft_pool, num_hidden_layers):
    from sglang.srt.disaggregation.utils import append_state_component

    def unwrap(pool):
        return pool.full_kv_pool if isinstance(pool, HybridLinearKVPool) else pool

    target = unwrap(token_pool)
    draft = unwrap(draft_pool)
    pools = [(target, token_pool.get_kv_layer_ids())]
    if draft is not None:
        if not getattr(draft, "is_hcu_glm5_next_pool", False):
            raise ValueError(
                "HCU GLM target and draft require the same physical KPool layout"
            )
        # Draft owns one attention layer. IDs stay stable when a CP rank has
        # no draft buffer; both PD peers reserve the same band above target.
        pools.append(
            (
                draft,
                [
                    num_hidden_layers + i - draft.start_layer
                    for i in draft.get_kv_layer_ids()
                ],
            )
        )

    index_ptrs, index_lens, index_items, index_ids = [], [], [], []
    tail_ptrs, tail_lens, tail_items, tail_ids = [], [], [], []
    has_tail = False
    for pool, ids in pools:
        ptrs, lens, items = pool.get_state_buf_infos()
        if len(ptrs) != len(ids):
            raise ValueError("HCU indexer buffer IDs do not match its owned layers")
        index_ptrs.extend(ptrs)
        index_lens.extend(lens)
        index_items.extend(items)
        index_ids.extend(ids)
        has_tail |= pool.kpool_use_compress
        if not pool.kpool_use_compress:
            continue
        owned = [i - pool.start_layer for i in pool.get_kv_layer_ids()]
        ptrs, lens, items = pool.get_compress_tail_buf_infos()
        selected = owned + [pool.layer_num + i for i in owned]
        tail_ptrs.extend(ptrs[i] for i in selected)
        tail_lens.extend(lens[i] for i in selected)
        tail_items.extend(items[i] for i in selected)
        tail_ids.extend(ids + ids)
    append_state_component(
        kv_args,
        StateType.DSA,
        index_ptrs,
        index_lens,
        index_items,
        layer_ids=index_ids,
    )
    if has_tail:
        # Preserve component order even on a rank that owns no tail buffers.
        append_state_component(
            kv_args,
            StateType.DSA_TAIL,
            tail_ptrs,
            tail_lens,
            tail_items,
            layer_ids=tail_ids,
        )
