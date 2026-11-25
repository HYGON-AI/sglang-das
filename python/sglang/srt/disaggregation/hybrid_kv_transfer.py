"""Byte planner for the PD KV transfer of a hybrid MLA + MHA model (Kimi-K3 + DSPARK).

The target model keeps one flat MLA latent per token; the DSPARK draft keeps an
ordinary MHA KV cache whose heads are split over attention TP. When prefill and
decode use different attention TP sizes, a draft page holds a different number
of bytes on each peer and each draft head lives on exactly one decode rank, so
the per-entry whole-page copy the Mooncake sender uses is not enough. This
module describes every registered buffer (``attach_specs``), checks the peers
before any address is handed out (``peer_problem``, ``validate_peer``), carries
the descriptions in the optional registration frame 21, and turns them into
byte-copy blocks (``plan`` / ``send``).

Standard library only at import time, no GPU allocations. Gated by
SGLANG_K3_HYBRID_KV_TRANSFER_V1 on BOTH peers. Limited to ordinary (DCP=1) FP8
pools whose page size and dtype match on both sides.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

FLAG = "SGLANG_K3_HYBRID_KV_TRANSFER_V1"
# Named by the decode side in the bootstrap handshake and reported in frame 21 as
# "protocol". Informational: the peers are matched by the buffer descriptions
# themselves (validate_peer), not by this string.
PROTOCOL_ID = "k3-hybrid-kv-v1"


def enabled():
    value = os.getenv(FLAG, "0")
    if value not in ("0", "1"):
        raise RuntimeError(f"{FLAG} must be 0 or 1")
    return value == "1"


@dataclass(frozen=True)
class Entry:
    family: str
    layer: int
    kind: str
    base: int
    span: int
    page_size: int
    page_bytes: int
    layout: str
    heads: int
    head_start: int
    total_heads: int
    dim: int
    element_bytes: int
    dtype: str

    @property
    def key(self):
        return self.family, self.layer, self.kind

    def validate(self):
        numbers = (
            self.base,
            self.span,
            self.page_size,
            self.page_bytes,
            self.heads,
            self.total_heads,
            self.dim,
            self.element_bytes,
        )
        if any(type(x) is not int or x <= 0 for x in numbers):
            raise ValueError(f"Invalid positive integer geometry for {self.key}")
        if type(self.layer) is not int or self.layer < 0:
            raise ValueError("Invalid layer id")
        if type(self.head_start) is not int or self.head_start < 0:
            raise ValueError("Invalid head start")
        if self.head_start + self.heads > self.total_heads:
            raise ValueError(f"Invalid head range for {self.key}")
        if self.layout not in ("flat", "bhtd", "bhdt", "nhd"):
            raise ValueError(f"Unsupported layout: {self.layout}")
        if self.family not in ("main_mla", "draft_mha"):
            raise ValueError(f"Unsupported family: {self.family}")
        if self.family == "main_mla" and (self.kind != "kv" or self.layout != "flat"):
            raise ValueError("MLA must have a flat kv entry")
        if self.family == "draft_mha" and (
            self.kind not in ("k", "v") or self.layout == "flat"
        ):
            raise ValueError("Draft MHA needs explicit K/V layout")
        if (
            self.page_bytes
            != self.page_size * self.heads * self.dim * self.element_bytes
        ):
            raise ValueError(f"Page ABI mismatch for {self.key}")
        if self.span % self.page_bytes:
            raise ValueError(f"Unaligned registered span for {self.key}")
        if self.dtype not in ("fp8_e4m3", "fp8_e4m3fn", "fp8_e4m3fnuz"):
            raise ValueError(
                f"Hybrid KV transfer requires FP8 E4M3 storage, got {self.dtype}"
            )
        if self.element_bytes != 1:
            raise ValueError("Hybrid KV transfer requires one-byte KV storage")


def _entries(items):
    result = [item if isinstance(item, Entry) else Entry(**item) for item in items]
    for item in result:
        item.validate()
    if len({item.key for item in result}) != len(result):
        raise ValueError("Duplicate (family, layer, K/V) transfer entries")
    return result


def encode_specs(specs):
    if specs is None:
        return b""
    decode_specs(json.dumps(specs).encode())
    return json.dumps(specs, separators=(",", ":")).encode()


def decode_specs(data):
    if not data:
        return None
    if len(data) > 262144:
        raise ValueError("Hybrid KV metadata exceeds 256 KiB")
    specs = json.loads(data)
    if not isinstance(specs, dict):
        raise ValueError("Hybrid KV metadata must be an object")
    if specs.get("version") != 1 or not isinstance(specs.get("entries"), list):
        raise ValueError("Unsupported hybrid KV metadata version")
    if len(specs["entries"]) > 256:
        raise ValueError("Too many hybrid KV entries")
    _entries(specs["entries"])
    topology = specs.get("topology")
    if topology is not None:
        if not isinstance(topology, dict):
            raise ValueError("Hybrid KV topology metadata must be an object")
        tp = topology.get("attn_tp")
        rank = topology.get("attn_rank")
        if (
            type(tp) is not int
            or tp <= 0
            or type(rank) is not int
            or not 0 <= rank < tp
        ):
            raise ValueError("Invalid hybrid KV attention TP topology")
    return specs


def _head_range(total, tp, rank):
    if not 0 <= rank < tp:
        raise ValueError("Attention TP rank is outside its group")
    if total >= tp and total % tp == 0:
        return rank * (total // tp), total // tp
    if tp > total and tp % total == 0:
        return rank // (tp // total), 1
    raise ValueError("KV heads and attention TP do not have an integral mapping")


def _mha_layout(tensor, pool, kind, page_size):
    shape = tuple(int(x) for x in tensor.shape)
    dim = int(pool.head_dim if kind == "k" else pool.v_head_dim)
    heads = int(pool.head_num)
    if not tensor.is_contiguous():
        raise ValueError("Non-contiguous draft storage is not supported")
    if len(shape) == 3 and shape[1:] == (heads, dim):
        return "nhd"
    if len(shape) != 4:
        raise ValueError(f"Unsupported draft KV tensor shape: {shape}")
    if getattr(pool, "use_hnd", False):
        if shape[1:] == (heads, page_size, dim):
            return "bhtd"
    else:
        # Resolve the platform switch too: when page_size == head_dim the
        # shape alone cannot distinguish HCU V [B,H,D,T] from [B,H,T,D].
        from sglang.srt.utils import is_hcu

        if is_hcu() and os.getenv("SGLANG_KV_LAYOUT_HCU_FA", "true").lower() in (
            "1",
            "true",
        ):
            expected = (
                (heads, page_size, dim) if kind == "k" else (heads, dim, page_size)
            )
            if shape[1:] == expected:
                return "bhtd" if kind == "k" else "bhdt"
    raise ValueError(f"Unknown 4D draft KV layout: kind={kind}, shape={shape}")


def attach_specs(args, main_pool, draft_pool, *, scheduler, is_decode=False):
    """Describe the buffers one rank registers. Nothing is read from the
    scheduler unless the switch is on for a hybrid MLA model."""
    if not enabled() or not getattr(args, "is_hybrid_mla_backend", False):
        return
    full = getattr(main_pool, "full_kv_pool", None)
    if full is None or not getattr(main_pool, "use_mla", False):
        raise ValueError(
            "Hybrid KV transfer requires the main pool to be backed by MLA"
        )
    from sglang.srt.runtime_context import get_parallel

    if get_parallel().attn_dcp_size != 1:
        raise ValueError("Hybrid KV transfer requires DCP=1")
    if is_decode and draft_pool is None:
        raise ValueError("Decode is missing the DSPARK draft KV pool")
    tp, rank = scheduler.ps.attn_tp_size, scheduler.ps.attn_tp_rank
    draft_path = scheduler.server_args.speculative_draft_model_path
    model_layers = scheduler.model_config.num_hidden_layers
    ptrs, spans, sizes = args.kv_data_ptrs, args.kv_data_lens, args.kv_item_lens
    if not (len(ptrs) == len(spans) == len(sizes) == len(args.kv_layer_ids)):
        raise ValueError("KV geometry metadata length mismatch")
    page = int(args.page_size)
    dtype = str(args.kv_cache_dtype_str)
    by_pointer = {}
    main_ptrs, _, _ = main_pool.get_contiguous_buf_infos()
    for ptr in main_ptrs:
        by_pointer[int(ptr)] = ("main_mla", "kv", "flat", 1, 0, 1, None, None)
    if draft_pool is not None:
        config = json.loads((Path(draft_path) / "config.json").read_text())
        config = config.get("text_config", config)
        total = int(config["num_key_value_heads"])
        start, heads = _head_range(total, tp, rank)
        if int(draft_pool.head_num) != heads:
            raise ValueError("Draft pool heads disagree with attention TP/config")
        for kind, buffers in (("k", draft_pool.k_buffer), ("v", draft_pool.v_buffer)):
            for layer, tensor in enumerate(buffers):
                ptr = int(tensor.data_ptr())
                if ptr in by_pointer:
                    raise ValueError("Aliased main/draft buffers")
                by_pointer[ptr] = (
                    "draft_mha",
                    kind,
                    _mha_layout(tensor, draft_pool, kind, page),
                    heads,
                    start,
                    total,
                    int(tensor.element_size()),
                    model_layers + layer,
                )
    entries = []
    for ptr, span, size, layer in zip(ptrs, spans, sizes, args.kv_layer_ids):
        if int(ptr) not in by_pointer:
            raise ValueError("Unrecognized hybrid KV registration pointer")
        family, kind, layout, heads, start, total, element, expected_layer = by_pointer[
            int(ptr)
        ]
        if expected_layer is not None and int(layer) != expected_layer:
            raise ValueError(
                "Draft K/V layer IDs disagree; apply the draft-layer-ID fix first"
            )
        if family == "main_mla":
            element = int(full.kv_buffer[0].element_size())
        dim = int(size) // (page * heads * element)
        if family == "draft_mha":
            expected_dim = int(
                draft_pool.head_dim if kind == "k" else draft_pool.v_head_dim
            )
            if dim != expected_dim:
                raise ValueError("Draft item length disagrees with its physical tensor")
        entries.append(
            Entry(
                family,
                int(layer),
                kind,
                int(ptr),
                int(span),
                page,
                int(size),
                layout,
                heads,
                start,
                total,
                dim,
                element,
                dtype,
            )
        )
    _entries(entries)
    args.hybrid_kv_specs = {
        "version": 1,
        "protocol": PROTOCOL_ID,
        "topology": {"attn_tp": int(tp), "attn_rank": int(rank)},
        "entries": [asdict(e) for e in entries],
    }


def peer_problem(protocol, src_cp, dst_cp, dst_pp, src_tp, dst_tp):
    """Why a decode worker with the switch on cannot take KV from this prefill
    worker, or None when it can. Runs on the decode side as soon as the bootstrap
    server has answered, before any write address is handed out."""
    if protocol != PROTOCOL_ID:
        return (
            f"the prefill worker does not report {PROTOCOL_ID}; "
            f"set {FLAG} on every prefill and decode process"
        )
    if src_cp != 1 or dst_cp != 1 or dst_pp != 1:
        return "the transfer needs CP1 on both peers and a single decode PP stage"
    if not src_tp or dst_tp < src_tp or dst_tp % src_tp:
        return "the prefill attention TP must divide the decode attention TP"
    return None


def validate_peer(src_specs, dst_specs, dst_ptrs, dst_layer_ids):
    if dst_specs is not None and dst_specs.get("invalid"):
        raise ValueError(f"Invalid peer hybrid geometry: {dst_specs['invalid']}")
    if dst_specs is None:
        raise ValueError("Hybrid KV peer has no v1 geometry; update both P and D")
    if (
        src_specs is None
        or src_specs.get("version") != 1
        or dst_specs.get("version") != 1
    ):
        raise ValueError("Missing or unsupported local/remote hybrid geometry")
    for label, specs in (("source", src_specs), ("destination", dst_specs)):
        topology = specs.get("topology")
        if topology is not None:
            if not isinstance(topology, dict):
                raise ValueError(f"Invalid {label} hybrid KV topology object")
            tp = topology.get("attn_tp")
            rank = topology.get("attn_rank")
            if (
                type(tp) is not int
                or tp <= 0
                or type(rank) is not int
                or not 0 <= rank < tp
            ):
                raise ValueError(f"Invalid {label} hybrid KV attention TP topology")
    source, target = _entries(src_specs["entries"]), _entries(dst_specs["entries"])
    for specs, entries in ((src_specs, source), (dst_specs, target)):
        topology = specs.get("topology")
        if topology is not None:
            for e in entries:
                if e.family == "draft_mha" and (e.head_start, e.heads) != _head_range(
                    e.total_heads, topology["attn_tp"], topology["attn_rank"]
                ):
                    raise ValueError(
                        "Draft head ownership disagrees with registered TP/rank"
                    )
    if len(target) != len(dst_ptrs) or len(target) != len(dst_layer_ids):
        raise ValueError("Hybrid metadata does not match peer registration length")
    if [e.base for e in target] != list(dst_ptrs) or [e.layer for e in target] != list(
        dst_layer_ids
    ):
        raise ValueError(
            "Hybrid pointer/layer metadata disagrees with peer registration"
        )
    destinations = {e.key: e for e in target}
    pairs = []
    for src in source:
        dst = destinations.get(src.key)
        if dst is None:
            raise ValueError(f"Peer is missing entry {src.key}")
        if (src.page_size, src.dtype, src.element_bytes, src.dim, src.total_heads) != (
            dst.page_size,
            dst.dtype,
            dst.element_bytes,
            dst.dim,
            dst.total_heads,
        ):
            raise ValueError(f"Hybrid entry ABI mismatch: {src.key}")
        if src.layout != dst.layout:
            raise ValueError(
                f"Hybrid KV transfer does not convert physical layouts: {src.key}"
            )
        pairs.append((src, dst))
    return pairs


def decode_registration(data):
    """Keep malformed optional metadata inside the normal registration-failure
    handling, rather than terminating the shared bootstrap thread."""
    try:
        return decode_specs(data)
    except (TypeError, ValueError, KeyError) as error:
        return {"invalid": str(error)}


def _append(blocks, src, dst, size):
    if (
        blocks
        and blocks[-1][0] + blocks[-1][2] == src
        and blocks[-1][1] + blocks[-1][2] == dst
    ):
        old_src, old_dst, old_size = blocks[-1]
        blocks[-1] = (old_src, old_dst, old_size + size)
    else:
        blocks.append((src, dst, size))


def plan(pairs, src_pages, dst_pages):
    """Produce byte-copy blocks; no transfers occur until every entry passes."""
    src_pages, dst_pages = list(src_pages), list(dst_pages)
    if len(src_pages) != len(dst_pages):
        raise ValueError("Source/destination page counts differ")
    if len(set(map(int, dst_pages))) != len(dst_pages):
        raise ValueError("Duplicate destination pages")
    blocks = []
    for src, dst in pairs:
        src.validate()
        dst.validate()
        first = max(src.head_start, dst.head_start)
        last = min(src.head_start + src.heads, dst.head_start + dst.heads)
        for src_page, dst_page in zip(src_pages, dst_pages):
            src_page, dst_page = int(src_page), int(dst_page)
            if not (
                0 <= src_page < src.span // src.page_bytes
                and 0 <= dst_page < dst.span // dst.page_bytes
            ):
                raise ValueError(
                    f"Hybrid KV page outside registered span: entry={src.key} "
                    f"src_page={src_page}, dst_page={dst_page}"
                )
            if first >= last:
                continue  # Another P rank owns the requested draft KV heads.
            sbase = src.base + src_page * src.page_bytes
            dbase = dst.base + dst_page * dst.page_bytes
            width = src.dim * src.element_bytes
            sh, dh, count = first - src.head_start, first - dst.head_start, last - first
            if src.layout == "flat":
                _append(blocks, sbase, dbase, src.page_bytes)
            elif src.layout in ("bhtd", "bhdt"):
                # HCU K [B,H,T,D] and V [B,H,D,T] each have a whole head
                # contiguous inside the page. Never reinterpret V as K.
                head_bytes = src.page_size * width
                _append(
                    blocks,
                    sbase + sh * head_bytes,
                    dbase + dh * head_bytes,
                    count * head_bytes,
                )
            else:
                for token in range(src.page_size):
                    _append(
                        blocks,
                        sbase + (token * src.heads + sh) * width,
                        dbase + (token * dst.heads + dh) * width,
                        count * width,
                    )
    return blocks


def send(manager, session, src_pages, dst_pages, dst_ptrs, dst_layer_ids):
    peer = manager.decode_kv_args_table[session]
    if peer.dst_dcp_size != 1:
        raise ValueError("Hybrid KV transfer requires destination DCP=1")
    pairs = validate_peer(
        getattr(manager.kv_args, "hybrid_kv_specs", None),
        peer.hybrid_kv_specs,
        dst_ptrs,
        dst_layer_ids,
    )
    blocks = plan(pairs, src_pages, dst_pages)
    # Bound each engine call without allocating another GPU pool or changing
    # request status/lifetime handling. Caller receives the real error code.
    for start in range(0, len(blocks), 4096):
        rc = manager._transfer_data(session, blocks[start : start + 4096])
        if rc != 0:
            return rc
    return 0
