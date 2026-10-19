"""Dependency-free wire/config checks; these are not device or accuracy tests.

Run with: python test/manual/dsv4/test_rebase_contracts.py
Load the pure protocol module directly, without initializing the serving runtime.
"""

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "v41_transfer_contract",
    ROOT / "python/sglang/srt/mem_cache/cp_cache_layer_split/transfer.py",
)
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


class TestV41RebaseContracts(unittest.TestCase):
    def test_old_main_registration_has_no_layer_split_descriptor(self):
        message = [b""] * 22
        # Main's format/type payloads are binary, not LayerSplit JSON.
        message[20], message[21] = b"\xff\x01", b"\x02\x03"
        self.assertEqual(transfer.decode_v4_transfer_metadata(message), {})

    def test_descriptor_round_trip_is_appended_after_main_fields(self):
        data = dict(version=1, kv=[["c1_kv", 20]], states=[], kv_sizes=[65536])
        payload = transfer.encode_v4_transfer_metadata(
            SimpleNamespace(v4_transfer_metadata=data)
        )
        message = [b""] * 22 + [payload]
        message[20], message[21] = b"main-formats", b"main-types"
        self.assertEqual(transfer.decode_v4_transfer_metadata(message), data)
        self.assertEqual(message[20:22], [b"main-formats", b"main-types"])

    def test_empty_descriptor_is_optional(self):
        self.assertEqual(transfer.encode_v4_transfer_metadata(SimpleNamespace()), b"")
        self.assertEqual(transfer.decode_v4_transfer_metadata([b""] * 23), {})

    def test_unknown_descriptor_version_is_rejected(self):
        message = [b""] * 22 + [json.dumps(dict(version=2)).encode()]
        with self.assertRaisesRegex(RuntimeError, "version"):
            transfer.decode_v4_transfer_metadata(message)

    def test_owner_shard_matches_reordered_full_decode_pool(self):
        self.assertEqual(
            transfer.match_transfer_entries(
                [("c1_kv", 20), ("c2_indexer", 2)],
                [("swa", 0), ("c2_indexer", 2), ("c1_kv", 20)],
                [64, 16],
                [32, 16, 64],
            ),
            [(0, 2), (1, 1)],
        )

    def test_mismatched_page_size_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "page ABI mismatch"):
            transfer.match_transfer_entries([("kv", 1)], [("kv", 1)], [64], [32])

    def test_missing_layer_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "missing"):
            transfer.match_transfer_entries([("kv", 1)], [("kv", 2)], [64], [64])

    def test_duplicate_layer_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            transfer.match_transfer_entries(
                [("kv", 1)] * 2, [("kv", 1)], [64, 64], [64]
            )

    def test_positional_server_arg_fields_have_no_duplicates(self):
        path = ROOT / "python/sglang/srt/arg_groups/field_order.py"
        module = ast.parse(path.read_text())
        fields = next(
            ast.literal_eval(node.value)
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "POSITIONAL_FIELD_ORDER"
                for target in node.targets
            )
        )
        self.assertEqual(len(fields), len(set(fields)))
        self.assertIn("enable_cp_cache_layer_split", fields)


if __name__ == "__main__":
    unittest.main(verbosity=2)
