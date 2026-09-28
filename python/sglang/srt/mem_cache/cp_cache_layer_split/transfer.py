"""Optional V4 transfer descriptors; legacy wire fields keep their positions."""

import json


def match_transfer_entries(src_layout, dst_layout, src_sizes, dst_sizes):
    """Match complete (family, global layer) identities, checking the page ABI."""
    if len(src_layout) != len(src_sizes) or len(dst_layout) != len(dst_sizes):
        raise RuntimeError(
            "LayerSplit PD descriptors/sizes are missing or incomplete; update both P and D"
        )
    src_keys = [tuple(x) for x in src_layout]
    dst_keys = [tuple(x) for x in dst_layout]
    if len(set(src_keys)) != len(src_keys) or len(set(dst_keys)) != len(dst_keys):
        raise RuntimeError("Duplicate LayerSplit PD transfer descriptor")
    destinations = {key: i for i, key in enumerate(dst_keys)}
    pairs = []
    for i, key in enumerate(src_keys):
        if key not in destinations:
            raise RuntimeError(f"Decode is missing LayerSplit PD cache {key}")
        j = destinations[key]
        if src_sizes[i] != dst_sizes[j]:
            raise RuntimeError(
                f"LayerSplit PD page ABI mismatch for {key}: P={src_sizes[i]}, D={dst_sizes[j]}"
            )
        pairs.append((i, j))
    return pairs


def configure_v4_transfer(kv_args, pool, draft_pool):
    from sglang.srt.disaggregation.base.conn import StateType
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    if not isinstance(pool, DeepSeekV4TokenToKVPool) or pool._unified_kv:
        return
    kv_args.cp_cache_layer_split = bool(
        getattr(pool, "requires_descriptor_matched_transfer", False)
    )
    kv_layout = pool.get_kv_transfer_layout()
    if draft_pool is not None:
        kv_layout += [
            ("draft_" + kind, layer)
            for kind, layer in draft_pool.get_kv_transfer_layout()
        ]
    states = []
    swa_seen = False
    for state_type, ptrs in zip(kv_args.state_types, kv_args.state_data_ptrs):
        if state_type == StateType.SWA and not swa_seen:
            layout = pool.get_state_transfer_layout()
            swa_seen = True
        elif state_type == StateType.SWA and draft_pool is not None:
            layout = [
                ("draft_" + kind, layer)
                for kind, layer in draft_pool.get_state_transfer_layout()
            ]
        elif state_type == StateType.C128_STATE:
            layout = pool.get_c128_state_transfer_layout()
        else:
            layout = []
        if layout and len(layout) != len(ptrs):
            raise RuntimeError(
                "V4 state transfer descriptors do not match allocated buffers"
            )
        states.append(layout)
    if len(kv_layout) != len(kv_args.kv_data_ptrs):
        raise RuntimeError("V4 KV transfer descriptors do not match allocated buffers")
    kv_args.v4_transfer_metadata = dict(
        version=1, kv=kv_layout, states=states, kv_sizes=kv_args.kv_item_lens
    )


def encode_v4_transfer_metadata(kv_args):
    metadata = getattr(kv_args, "v4_transfer_metadata", None)
    return json.dumps(metadata, separators=(",", ":")).encode() if metadata else b""


def decode_v4_transfer_metadata(msg):
    if len(msg) <= 20 or not msg[20]:
        return {}
    data = json.loads(msg[20])
    if data.get("version") != 1:
        raise RuntimeError("Unsupported V4 LayerSplit PD descriptor version")
    return data
