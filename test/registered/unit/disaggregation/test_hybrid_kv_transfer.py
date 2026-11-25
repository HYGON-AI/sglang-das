"""Unit tests for the hybrid MLA/MHA PD KV byte planner (Kimi-K3 + DSPARK), no GPU.

The planner is pure Python over byte geometry, so these tests need no GPU.
They pin the two things it fixes when the prefill and decode attention TP
sizes differ for a model whose draft keeps an ordinary MHA KV cache:

* a draft page holds a different number of bytes on each peer, so the
  destination has to be addressed with its own page size;
* a draft KV head lives on exactly one decode rank, so each prefill rank
  sends only the heads that destination owns.

The numbers are those of the Kimi-K3 + DSPARK deployment: 16 draft KV heads
of width 64, FP8 storage, page size 64, prefill attention TP 4 and decode
attention TP 8.
"""

import dataclasses
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation import hybrid_kv_transfer as hkt
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE = 64
DRAFT_HEADS = 16
DRAFT_DIM = 64
MLA_BYTES_PER_TOKEN = 576
MODEL_LAYERS = 93
DRAFT_LAYERS = 5
# A registered span that is a whole number of main (36864 B) and draft pages.
SPAN = 147456 * 576
MAIN_BASE = 1 << 30
SRC_BASE = 1 << 31
DST_BASE = 1 << 32
PREFILL_HEADS = DRAFT_HEADS // 4
DECODE_HEADS = DRAFT_HEADS // 8
HEAD_BYTES = PAGE * DRAFT_DIM
DEFAULT = object()
FLAG = hkt.FLAG


def entry(family, layer, kind, base, **overrides):
    """An Entry for one registered transfer buffer, with real page geometry."""
    dim = DRAFT_DIM if family == "draft_mha" else MLA_BYTES_PER_TOKEN
    values = dict(
        family=family,
        layer=layer,
        kind=kind,
        base=base,
        span=SPAN,
        page_size=PAGE,
        page_bytes=0,
        layout="flat" if family == "main_mla" else "bhtd",
        heads=1,
        head_start=0,
        total_heads=1,
        dim=dim,
        element_bytes=1,
        dtype="fp8_e4m3",
    )
    values.update(overrides)
    values["page_bytes"] = (
        PAGE * values["heads"] * values["dim"] * values["element_bytes"]
    )
    return hkt.Entry(**values)


def main_entry(layer, base, **overrides):
    return entry("main_mla", layer, "kv", base, **overrides)


def prefill_draft(layer, kind, rank, base=SRC_BASE, **overrides):
    """The draft entry prefill attention rank ``rank`` registers: 4 local heads."""
    values = dict(
        heads=PREFILL_HEADS, head_start=rank * PREFILL_HEADS, total_heads=DRAFT_HEADS
    )
    values.update(overrides)
    return entry("draft_mha", layer, kind, base, **values)


def decode_draft(layer, kind, rank, base=DST_BASE, **overrides):
    """The draft entry decode attention rank ``rank`` registers: 2 local heads."""
    values = dict(
        heads=DECODE_HEADS, head_start=rank * DECODE_HEADS, total_heads=DRAFT_HEADS
    )
    values.update(overrides)
    return entry("draft_mha", layer, kind, base, **values)


def draft_byte(global_head, token, dim):
    return ((global_head * 7 + token * 3 + dim) % 256) & 0xFF


class Memory:
    """Bytes of the first pages of one registered buffer."""

    def __init__(self, base, page_bytes, pages=24, fill=0xEE):
        self.base = base
        self.page_bytes = page_bytes
        self.data = bytearray([fill]) * (page_bytes * pages)

    def fill_draft_page(self, page, head_start, heads):
        """Write a [heads, page, dim] page whose bytes identify head/token/dim."""
        offset = page * self.page_bytes
        for head in range(heads):
            for token in range(PAGE):
                self.data[offset : offset + DRAFT_DIM] = bytes(
                    draft_byte(head_start + head, token, dim)
                    for dim in range(DRAFT_DIM)
                )
                offset += DRAFT_DIM

    def draft_head(self, page, local_head):
        offset = page * self.page_bytes + local_head * HEAD_BYTES
        return bytes(self.data[offset : offset + HEAD_BYTES])


def apply(blocks, source, target):
    for src, dst, size in blocks:
        s, d = src - source.base, dst - target.base
        target.data[d : d + size] = source.data[s : s + size]


def specs(entries, tp, rank):
    return {
        "version": 1,
        "protocol": hkt.PROTOCOL_ID,
        "topology": None if tp is None else {"attn_tp": tp, "attn_rank": rank},
        "entries": [dataclasses.asdict(e) for e in entries],
    }


@contextmanager
def flag(value):
    with patch.dict(os.environ, {FLAG: value}):
        yield


@contextmanager
def runtime(attn_dcp_size=1):
    runtime = types.ModuleType("sglang.srt.runtime_context")
    runtime.get_parallel = lambda: SimpleNamespace(attn_dcp_size=attn_dcp_size)
    with patch.dict(sys.modules, {"sglang.srt.runtime_context": runtime}):
        yield


class FakeTensor:
    def __init__(self, shape, pointer, element_size=1):
        self.shape = shape
        self._pointer = pointer
        self._element_size = element_size

    def is_contiguous(self):
        return True

    def data_ptr(self):
        return self._pointer

    def element_size(self):
        return self._element_size


def registration(main_layers, local_heads, draft_ids=None):
    """KVArgs plus the two pools one attention rank registers."""
    main_ptrs = [MAIN_BASE + i * SPAN for i in range(len(main_layers))]
    main_item = PAGE * MLA_BYTES_PER_TOKEN
    main_pool = SimpleNamespace(
        use_mla=True,
        full_kv_pool=SimpleNamespace(kv_buffer=[FakeTensor((), 0)]),
        get_contiguous_buf_infos=lambda: (
            main_ptrs,
            [SPAN] * len(main_layers),
            [main_item] * len(main_layers),
        ),
    )
    draft_item = PAGE * local_heads * DRAFT_DIM
    pointers = {
        kind: [
            SRC_BASE + (DRAFT_LAYERS * i + layer) * SPAN
            for layer in range(DRAFT_LAYERS)
        ]
        for i, kind in enumerate(("k", "v"))
    }
    buffers = {
        kind: [
            FakeTensor((SPAN // draft_item + PAGE, local_heads, PAGE, DRAFT_DIM), ptr)
            for ptr in pointers[kind]
        ]
        for kind in ("k", "v")
    }
    draft_pool = SimpleNamespace(
        head_num=local_heads,
        head_dim=DRAFT_DIM,
        v_head_dim=DRAFT_DIM,
        use_hnd=True,
        k_buffer=buffers["k"],
        v_buffer=buffers["v"],
        get_contiguous_buf_infos=lambda: (
            pointers["k"] + pointers["v"],
            [SPAN] * (2 * DRAFT_LAYERS),
            [draft_item] * (2 * DRAFT_LAYERS),
        ),
    )
    if draft_ids is None:
        # one id per draft layer, repeated for K and V, as build_kv_layer_ids emits them
        draft_ids = [MODEL_LAYERS + i for i in range(DRAFT_LAYERS)] * 2
    args = SimpleNamespace(
        kv_data_ptrs=main_ptrs + pointers["k"] + pointers["v"],
        kv_data_lens=[SPAN] * (len(main_layers) + 2 * DRAFT_LAYERS),
        kv_item_lens=[main_item] * len(main_layers) + [draft_item] * (2 * DRAFT_LAYERS),
        kv_layer_ids=list(main_layers) + list(draft_ids),
        page_size=PAGE,
        kv_cache_dtype_str="fp8_e4m3",
        is_hybrid_mla_backend=True,
    )
    return args, main_pool, draft_pool


def scheduler_of(tp, rank, draft_path):
    """What attach_specs reads from the scheduler once the switch is on."""
    return SimpleNamespace(
        ps=SimpleNamespace(attn_tp_size=tp, attn_tp_rank=rank),
        server_args=SimpleNamespace(speculative_draft_model_path=draft_path),
        model_config=SimpleNamespace(num_hidden_layers=MODEL_LAYERS),
    )


class Untouched:
    """Stands in for the scheduler: any attribute read fails the test."""

    def __getattr__(self, name):
        raise AssertionError(
            f"the scheduler was read ({name}) although nothing is attached"
        )


class HeadRangeTest(CustomTestCase):
    def test_heads_split_evenly(self):
        self.assertEqual(hkt._head_range(DRAFT_HEADS, 4, 3), (12, 4))
        self.assertEqual(hkt._head_range(DRAFT_HEADS, 8, 5), (10, 2))
        self.assertEqual(hkt._head_range(DRAFT_HEADS, 16, 0), (0, 1))

    def test_heads_replicate_when_ranks_outnumber_them(self):
        self.assertEqual(hkt._head_range(8, 16, 1), (0, 1))
        self.assertEqual(hkt._head_range(8, 16, 2), (1, 1))

    def test_non_integral_split_is_rejected(self):
        with self.assertRaises(ValueError):
            hkt._head_range(DRAFT_HEADS, 5, 0)
        with self.assertRaises(ValueError):
            hkt._head_range(DRAFT_HEADS, 4, 4)


class FlagTest(CustomTestCase):
    def test_off_by_default(self):
        with patch.dict(os.environ):
            os.environ.pop(FLAG, None)
            self.assertFalse(hkt.enabled())

    def test_values(self):
        with flag("1"):
            self.assertTrue(hkt.enabled())
        with flag("0"):
            self.assertFalse(hkt.enabled())
        with flag("2"):
            with self.assertRaises(RuntimeError):
                hkt.enabled()


def handshake(**overrides):
    """The decode-side handshake check for the Kimi-K3 deployment (P 4 -> D 8)."""
    values = dict(
        protocol=hkt.PROTOCOL_ID, src_cp=1, dst_cp=1, dst_pp=1, src_tp=4, dst_tp=8
    )
    values.update(overrides)
    return hkt.peer_problem(**values)


class PeerProblemTest(CustomTestCase):
    def test_supported_layouts_pass(self):
        for src_tp, dst_tp in ((4, 8), (8, 8), (1, 16), (4, 16)):
            with self.subTest(src_tp=src_tp, dst_tp=dst_tp):
                self.assertIsNone(handshake(src_tp=src_tp, dst_tp=dst_tp))

    def test_peer_without_the_protocol_names_the_switch(self):
        for protocol in (None, "", "k3-hybrid-kv-v2"):
            with self.subTest(protocol=protocol):
                problem = handshake(protocol=protocol)
                self.assertIn(FLAG, problem)
                self.assertIn("every prefill and decode process", problem)

    def test_parallel_layouts_outside_the_plan_are_refused(self):
        self.assertIn("CP1", handshake(src_cp=2))
        self.assertIn("CP1", handshake(dst_cp=2))
        self.assertIn("decode PP", handshake(dst_pp=2))
        for src_tp, dst_tp in ((0, 8), (None, 8), (4, 2), (4, 6)):
            with self.subTest(src_tp=src_tp, dst_tp=dst_tp):
                self.assertIn("attention TP", handshake(src_tp=src_tp, dst_tp=dst_tp))


class EntryTest(CustomTestCase):
    def test_page_bytes_must_match_geometry(self):
        prefill_draft(MODEL_LAYERS, "k", 0).validate()
        wrong = dataclasses.asdict(prefill_draft(MODEL_LAYERS, "k", 0))
        wrong["page_bytes"] = PAGE * DECODE_HEADS * DRAFT_DIM
        with self.assertRaisesRegex(ValueError, "Page ABI mismatch"):
            hkt.Entry(**wrong).validate()

    def test_head_range_must_stay_inside_the_model(self):
        with self.assertRaisesRegex(ValueError, "Invalid head range"):
            prefill_draft(MODEL_LAYERS, "k", 3, head_start=14).validate()

    def test_main_is_flat_and_draft_is_not(self):
        with self.assertRaises(ValueError):
            main_entry(0, MAIN_BASE, layout="bhtd").validate()
        with self.assertRaises(ValueError):
            prefill_draft(0, "k", 0, layout="flat").validate()

    def test_duplicate_entries_are_rejected(self):
        item = prefill_draft(0, "k", 0)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            hkt._entries([item, item])


class PlanTest(CustomTestCase):
    def test_main_entry_copies_whole_pages_and_merges_runs(self):
        page_bytes = PAGE * MLA_BYTES_PER_TOKEN
        blocks = hkt.plan(
            [(main_entry(7, SRC_BASE), main_entry(7, DST_BASE))],
            [10, 11, 12],
            [7, 8, 9],
        )
        self.assertEqual(
            blocks,
            [(SRC_BASE + 10 * page_bytes, DST_BASE + 7 * page_bytes, 3 * page_bytes)],
        )

    def test_each_prefill_rank_sends_only_the_heads_the_decode_rank_owns(self):
        for p_rank in range(4):
            for d_rank in range(8):
                with self.subTest(prefill_rank=p_rank, decode_rank=d_rank):
                    src = prefill_draft(MODEL_LAYERS, "k", p_rank)
                    dst = decode_draft(MODEL_LAYERS, "k", d_rank)
                    source = Memory(SRC_BASE, src.page_bytes)
                    target = Memory(DST_BASE, dst.page_bytes)
                    for page in (10, 11, 12):
                        source.fill_draft_page(page, src.head_start, src.heads)

                    blocks = hkt.plan([(src, dst)], [10, 11, 12], [7, 8, 9])
                    apply(blocks, source, target)

                    owned = set(range(dst.head_start, dst.head_start + dst.heads))
                    owned &= set(range(src.head_start, src.head_start + src.heads))
                    self.assertEqual(len(blocks), 3 if owned else 0)
                    for src_page, dst_page in zip((10, 11, 12), (7, 8, 9)):
                        for local_head in range(dst.heads):
                            global_head = dst.head_start + local_head
                            want = [0xEE] * HEAD_BYTES
                            if global_head in owned:
                                want = [
                                    draft_byte(global_head, token, dim)
                                    for token in range(PAGE)
                                    for dim in range(DRAFT_DIM)
                                ]
                            self.assertEqual(
                                list(target.draft_head(dst_page, local_head)), want
                            )

    def test_destination_is_addressed_with_its_own_page_size(self):
        # The stock sender addressed the destination with the source page size
        # (16384 here), doubling the stride of the 8192-byte decode pages.
        pair = (prefill_draft(MODEL_LAYERS, "k", 1), decode_draft(MODEL_LAYERS, "k", 2))
        blocks = hkt.plan([pair], [10, 11], [7, 8])
        self.assertEqual(
            [block[1] for block in blocks],
            [DST_BASE + 7 * pair[1].page_bytes, DST_BASE + 8 * pair[1].page_bytes],
        )
        self.assertEqual(
            [block[2] for block in blocks], [2 * HEAD_BYTES, 2 * HEAD_BYTES]
        )

    def test_head_slice_starts_inside_the_page_for_the_second_owner(self):
        pair = (prefill_draft(MODEL_LAYERS, "k", 1), decode_draft(MODEL_LAYERS, "k", 3))
        (block,) = hkt.plan([pair], [10], [7])
        self.assertEqual(
            block,
            (
                SRC_BASE + 10 * pair[0].page_bytes + 2 * HEAD_BYTES,
                DST_BASE + 7 * pair[1].page_bytes,
                2 * HEAD_BYTES,
            ),
        )

    def test_non_owner_pairs_produce_no_blocks(self):
        pair = (prefill_draft(0, "k", 0), decode_draft(0, "k", 4))
        self.assertEqual(hkt.plan([pair], [1], [1]), [])

    def test_token_major_layout_slices_every_token(self):
        src = prefill_draft(0, "k", 0, layout="nhd")
        dst = decode_draft(0, "k", 0, layout="nhd")
        blocks = hkt.plan([(src, dst)], [0], [0])
        self.assertEqual(len(blocks), PAGE)
        self.assertEqual(blocks[0], (SRC_BASE, DST_BASE, DECODE_HEADS * DRAFT_DIM))
        self.assertEqual(
            blocks[1],
            (
                SRC_BASE + PREFILL_HEADS * DRAFT_DIM,
                DST_BASE + DECODE_HEADS * DRAFT_DIM,
                DECODE_HEADS * DRAFT_DIM,
            ),
        )

    def test_transposed_value_layout_keeps_heads_contiguous_like_keys(self):
        keys = hkt.plan(
            [(prefill_draft(0, "k", 1), decode_draft(0, "k", 2))], [10], [7]
        )
        values = hkt.plan(
            [
                (
                    prefill_draft(0, "v", 1, layout="bhdt"),
                    decode_draft(0, "v", 2, layout="bhdt"),
                )
            ],
            [10],
            [7],
        )
        self.assertEqual(keys, values)

    def test_page_outside_the_registered_span_is_rejected(self):
        pair = (prefill_draft(0, "k", 0), decode_draft(0, "k", 0))
        with self.assertRaisesRegex(ValueError, "outside registered span"):
            hkt.plan([pair], [SPAN // pair[0].page_bytes], [0])
        with self.assertRaisesRegex(ValueError, "outside registered span"):
            hkt.plan([pair], [0], [SPAN // pair[1].page_bytes])

    def test_inconsistent_page_lists_are_rejected(self):
        pair = (prefill_draft(0, "k", 0), decode_draft(0, "k", 0))
        with self.assertRaisesRegex(ValueError, "page counts"):
            hkt.plan([pair], [1, 2], [1])
        with self.assertRaisesRegex(ValueError, "Duplicate destination"):
            hkt.plan([pair], [1, 2], [1, 1])


class ValidatePeerTest(CustomTestCase):
    def setUp(self):
        self.src = [
            main_entry(4, SRC_BASE),
            prefill_draft(MODEL_LAYERS, "k", 1),
            prefill_draft(MODEL_LAYERS, "v", 1),
        ]
        self.dst = [
            main_entry(4, DST_BASE),
            decode_draft(MODEL_LAYERS, "k", 2),
            decode_draft(MODEL_LAYERS, "v", 2),
        ]
        self.src_specs = specs(self.src, 4, 1)

    def validate(self, dst, dst_specs=DEFAULT):
        if dst_specs is DEFAULT:
            dst_specs = specs(dst, 8, 2)
        return hkt.validate_peer(
            self.src_specs,
            dst_specs,
            [e.base for e in dst],
            [e.layer for e in dst],
        )

    def test_entries_are_paired_by_layer(self):
        self.assertEqual(self.validate(self.dst), list(zip(self.src, self.dst)))

    def test_peer_without_the_protocol_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "update both P and D"):
            self.validate(self.dst, dst_specs=None)

    def test_malformed_peer_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid peer hybrid geometry"):
            self.validate(self.dst, dst_specs={"invalid": "boom"})

    def test_missing_local_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing or unsupported"):
            hkt.validate_peer(
                None,
                specs(self.dst, 8, 2),
                [e.base for e in self.dst],
                [e.layer for e in self.dst],
            )

    def test_head_ownership_must_match_the_registered_rank(self):
        with self.assertRaisesRegex(ValueError, "head ownership disagrees"):
            self.validate(self.dst, dst_specs=specs(self.dst, 8, 3))

    def test_topology_values_are_checked_on_both_sides(self):
        dst_specs = specs(self.dst, 8, 2)
        bad_topologies = (
            {"attn_tp": 0, "attn_rank": 0},
            {"attn_tp": 8, "attn_rank": 8},
            {"attn_tp": 8},
        )
        for topology in bad_topologies:
            with self.subTest(topology=topology):
                with self.assertRaisesRegex(ValueError, "topology"):
                    hkt.validate_peer(
                        {**self.src_specs, "topology": topology},
                        dst_specs,
                        [e.base for e in self.dst],
                        [e.layer for e in self.dst],
                    )

    def test_entry_abi_mismatch_is_rejected(self):
        dst = list(self.dst)
        dst[2] = decode_draft(MODEL_LAYERS, "v", 2, total_heads=2 * DRAFT_HEADS)
        with self.assertRaisesRegex(ValueError, "ABI mismatch"):
            self.validate(dst, dst_specs=specs(dst, None, None))

    def test_layout_conversion_is_rejected(self):
        dst = list(self.dst)
        dst[1] = decode_draft(MODEL_LAYERS, "k", 2, layout="nhd")
        with self.assertRaisesRegex(ValueError, "does not convert physical layouts"):
            self.validate(dst, dst_specs=specs(dst, None, None))

    def test_missing_peer_entry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Peer is missing entry"):
            self.validate(self.dst[:2])

    def test_peer_registration_must_match_its_geometry(self):
        dst_specs = specs(self.dst, 8, 2)
        with self.assertRaisesRegex(ValueError, "disagrees with peer registration"):
            hkt.validate_peer(
                self.src_specs,
                dst_specs,
                [e.base + 8 for e in self.dst],
                [e.layer for e in self.dst],
            )


class WireTest(CustomTestCase):
    def setUp(self):
        self.specs = specs(
            [main_entry(4, SRC_BASE), prefill_draft(MODEL_LAYERS, "k", 1)], 4, 1
        )

    def test_absent_metadata_is_an_empty_frame(self):
        self.assertEqual(hkt.encode_specs(None), b"")
        self.assertIsNone(hkt.decode_specs(b""))
        self.assertIsNone(hkt.decode_registration(b""))

    def test_round_trip(self):
        self.assertEqual(hkt.decode_specs(hkt.encode_specs(self.specs)), self.specs)
        self.assertEqual(
            hkt.decode_registration(hkt.encode_specs(self.specs)), self.specs
        )

    def test_bad_frames_stay_in_the_registration_failure_path(self):
        bad_frames = (
            b"not json",
            b"[]",
            b'{"version": 2, "entries": []}',
            b"x" * 262145,
        )
        for data in bad_frames:
            with self.subTest(data=data[:16]):
                self.assertIn("invalid", hkt.decode_registration(data))

    def test_bad_geometry_is_reported_not_raised(self):
        broken = {
            "version": 1,
            "entries": [{"family": "draft_mha", "layer": 0, "kind": "k", "base": 1}],
        }
        self.assertIn("invalid", hkt.decode_registration(json.dumps(broken).encode()))


class SendTest(CustomTestCase):
    def setUp(self):
        self.src = [
            main_entry(4, SRC_BASE),
            prefill_draft(MODEL_LAYERS, "k", 1),
            prefill_draft(MODEL_LAYERS, "v", 1),
        ]
        self.dst = [
            main_entry(4, DST_BASE),
            decode_draft(MODEL_LAYERS, "k", 2),
            decode_draft(MODEL_LAYERS, "v", 2),
        ]
        self.dst_ptrs = [e.base for e in self.dst]
        self.dst_layers = [e.layer for e in self.dst]

    def manager(self, rc=0, dcp_size=1, with_peer=True):
        peer = SimpleNamespace(
            dst_dcp_size=dcp_size,
            hybrid_kv_specs=specs(self.dst, 8, 2) if with_peer else None,
        )
        transfer = Mock(return_value=rc)
        return (
            SimpleNamespace(
                decode_kv_args_table={"session": peer},
                kv_args=SimpleNamespace(hybrid_kv_specs=specs(self.src, 4, 1)),
                _transfer_data=transfer,
            ),
            transfer,
        )

    def test_all_entries_go_in_one_engine_call(self):
        manager, transfer = self.manager()
        self.assertEqual(
            hkt.send(
                manager, "session", [10, 11], [7, 8], self.dst_ptrs, self.dst_layers
            ),
            0,
        )
        (blocks,) = transfer.call_args.args[1:]
        # main: one merged run of 2 pages; draft K and V: one head slice per page
        self.assertEqual(
            [size for _, _, size in blocks],
            [2 * PAGE * MLA_BYTES_PER_TOKEN] + [2 * HEAD_BYTES] * 4,
        )

    def test_engine_calls_are_bounded(self):
        manager, transfer = self.manager()
        pages = list(range(2200))  # 1 main block + 2200 K blocks + 2200 V blocks
        self.assertEqual(
            hkt.send(manager, "session", pages, pages, self.dst_ptrs, self.dst_layers),
            0,
        )
        self.assertEqual(
            [len(call.args[1]) for call in transfer.call_args_list], [4096, 305]
        )

    def test_engine_error_stops_the_transfer_and_is_returned(self):
        manager, transfer = self.manager(rc=-7)
        pages = list(range(2200))
        self.assertEqual(
            hkt.send(manager, "session", pages, pages, self.dst_ptrs, self.dst_layers),
            -7,
        )
        self.assertEqual(transfer.call_count, 1)

    def test_peer_without_the_protocol_is_rejected(self):
        manager, transfer = self.manager(with_peer=False)
        with self.assertRaisesRegex(ValueError, "update both P and D"):
            hkt.send(manager, "session", [1], [1], self.dst_ptrs, self.dst_layers)
        transfer.assert_not_called()

    def test_context_parallel_decode_is_rejected(self):
        manager, transfer = self.manager(dcp_size=2)
        with self.assertRaisesRegex(ValueError, "destination DCP=1"):
            hkt.send(manager, "session", [1], [1], self.dst_ptrs, self.dst_layers)
        transfer.assert_not_called()


class AttachSpecsTest(CustomTestCase):
    def setUp(self):
        self.draft_path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.draft_path, ignore_errors=True)
        with open(os.path.join(self.draft_path, "config.json"), "w") as fh:
            json.dump({"num_key_value_heads": DRAFT_HEADS}, fh)

    def attach(
        self,
        args,
        main_pool,
        draft_pool,
        is_decode,
        tp=None,
        rank=None,
        dcp=1,
        **kwargs,
    ):
        """attach_specs for attention rank `rank` of a group of `tp` under context
        parallel size `dcp`, or for the scheduler object passed as `scheduler=`."""
        kwargs.setdefault("scheduler", scheduler_of(tp, rank, self.draft_path))
        with runtime(attn_dcp_size=dcp):
            hkt.attach_specs(args, main_pool, draft_pool, is_decode=is_decode, **kwargs)
        return args

    def test_disabled_leaves_the_registration_untouched(self):
        # Nothing is read from the scheduler either: a deployment without the
        # switch registers exactly as before.
        args, main_pool, draft_pool = registration([4], PREFILL_HEADS)
        with flag("0"):
            self.attach(
                args, main_pool, draft_pool, is_decode=False, scheduler=Untouched()
            )
        self.assertFalse(hasattr(args, "hybrid_kv_specs"))

    def test_non_hybrid_models_are_untouched(self):
        args, main_pool, draft_pool = registration([4], DECODE_HEADS)
        args.is_hybrid_mla_backend = False
        with flag("1"):
            self.attach(
                args, main_pool, draft_pool, is_decode=True, scheduler=Untouched()
            )
        self.assertFalse(hasattr(args, "hybrid_kv_specs"))

    def test_every_registered_buffer_is_described(self):
        args, main_pool, draft_pool = registration([4], PREFILL_HEADS)
        with flag("1"):
            self.attach(args, main_pool, draft_pool, is_decode=False, tp=4, rank=1)
        described = args.hybrid_kv_specs
        self.assertEqual(described["topology"], {"attn_tp": 4, "attn_rank": 1})
        self.assertEqual(
            [(e["family"], e["layer"], e["kind"]) for e in described["entries"]],
            [("main_mla", 4, "kv")]
            + [
                ("draft_mha", MODEL_LAYERS + i, kind)
                for kind in ("k", "v")
                for i in range(DRAFT_LAYERS)
            ],
        )
        main, key = described["entries"][:2]
        self.assertEqual(
            (main["layout"], main["dim"], main["page_bytes"]),
            ("flat", MLA_BYTES_PER_TOKEN, PAGE * MLA_BYTES_PER_TOKEN),
        )
        self.assertEqual(
            (key["layout"], key["heads"], key["head_start"], key["total_heads"]),
            ("bhtd", PREFILL_HEADS, PREFILL_HEADS, DRAFT_HEADS),
        )

    def test_prefill_and_decode_descriptions_agree_on_what_to_send(self):
        prefill = registration([4], PREFILL_HEADS)
        owner = registration([4], DECODE_HEADS)
        other = registration([4], DECODE_HEADS)
        with flag("1"):
            self.attach(*prefill, is_decode=False, tp=4, rank=1)
            self.attach(*owner, is_decode=True, tp=8, rank=3)
            self.attach(*other, is_decode=True, tp=8, rank=4)

        def planned(decode):
            described = decode.hybrid_kv_specs["entries"]
            pairs = hkt.validate_peer(
                prefill[0].hybrid_kv_specs,
                decode.hybrid_kv_specs,
                [e["base"] for e in described],
                [e["layer"] for e in described],
            )
            return hkt.plan(pairs, [10], [7])

        key = prefill[0].hybrid_kv_specs["entries"][1]
        key_owner = owner[0].hybrid_kv_specs["entries"][1]
        # decode rank 3 owns draft heads 6-7, which live on prefill rank 1 (heads 4-7)
        self.assertIn(
            (
                key["base"] + 10 * key["page_bytes"] + 2 * HEAD_BYTES,
                key_owner["base"] + 7 * key_owner["page_bytes"],
                2 * HEAD_BYTES,
            ),
            planned(owner[0]),
        )
        # decode rank 4 owns heads 8-9: only the main entry is sent to it
        self.assertEqual(
            [size for _, _, size in planned(other[0])],
            [PAGE * MLA_BYTES_PER_TOKEN],
        )

    def test_per_entry_draft_layer_ids_are_rejected(self):
        # The stock decode registration numbered draft K and V buffers separately,
        # so they no longer matched prefill's one-id-per-layer convention.
        stock_ids = [MODEL_LAYERS + i for i in range(2 * DRAFT_LAYERS)]
        args, main_pool, draft_pool = registration([4], DECODE_HEADS, stock_ids)
        with flag("1"):
            with self.assertRaisesRegex(ValueError, "layer IDs disagree"):
                self.attach(args, main_pool, draft_pool, is_decode=True, tp=8, rank=0)

    def test_decode_needs_the_draft_pool(self):
        args, main_pool, _ = registration([4], DECODE_HEADS)
        with flag("1"):
            with self.assertRaisesRegex(ValueError, "missing the DSPARK draft KV pool"):
                self.attach(args, main_pool, None, is_decode=True, tp=8, rank=0)

    def test_prefill_rank_without_the_draft_pool_describes_the_main_entry(self):
        # Pipeline stages that only accumulate context features own no draft KV.
        args, main_pool, _ = registration([4], PREFILL_HEADS)
        for name in ("kv_data_ptrs", "kv_data_lens", "kv_item_lens", "kv_layer_ids"):
            setattr(args, name, getattr(args, name)[:1])
        with flag("1"):
            self.attach(args, main_pool, None, is_decode=False, tp=4, rank=0)
        self.assertEqual(
            [e["family"] for e in args.hybrid_kv_specs["entries"]], ["main_mla"]
        )

    def test_unsupported_setups_are_rejected(self):
        args, main_pool, draft_pool = registration([4], DECODE_HEADS)
        with flag("1"):
            with self.assertRaisesRegex(ValueError, "backed by MLA"):
                self.attach(
                    args,
                    SimpleNamespace(**{**vars(main_pool), "use_mla": False}),
                    draft_pool,
                    is_decode=True,
                    tp=8,
                    rank=0,
                )
            with self.assertRaisesRegex(ValueError, "Draft pool heads disagree"):
                self.attach(args, main_pool, draft_pool, is_decode=True, tp=4, rank=0)
            unknown_pointer = SimpleNamespace(
                **{**vars(args), "kv_data_ptrs": args.kv_data_ptrs[:-1] + [123]}
            )
            with self.assertRaisesRegex(
                ValueError, "Unrecognized hybrid KV registration"
            ):
                self.attach(
                    unknown_pointer, main_pool, draft_pool, is_decode=True, tp=8, rank=0
                )
            with self.assertRaisesRegex(ValueError, "DCP=1"):
                self.attach(
                    args, main_pool, draft_pool, is_decode=True, tp=8, rank=0, dcp=2
                )


if __name__ == "__main__":
    unittest.main()
