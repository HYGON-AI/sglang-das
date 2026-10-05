# Copyright 2026 Hygon Information Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0.

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


SOURCE = (
    Path(__file__).parents[4]
    / "python/sglang/srt/layers/attention/dsa/utils.py"
)


def _load_admission_functions():
    tree = ast.parse(SOURCE.read_text())
    names = {"can_dsa_prefill_cp_round_robin_split", "can_dsa_cp_split"}
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "effective_forward_mode": lambda batch: SimpleNamespace(
            is_context_parallel_extend=lambda: True
        ),
        "get_parallel": lambda: SimpleNamespace(attn_cp_size=8),
        "is_dsa_prefill_cp_round_robin_split": lambda: True,
        "is_dsa_enable_prefill_cp": lambda: True,
        "_can_dsa_cp_v2_split_uneven": lambda batch: getattr(
            batch, "cp_v2_active", False
        ),
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


class TestHcuDsaCpAlignment(unittest.TestCase):
    def test_cp_v2_interleave_shards_unaligned_indexer_metadata(self):
        functions = _load_admission_functions()
        for tokens in (12, 17):
            batch = SimpleNamespace(extend_seq_lens_cpu=[tokens], cp_v2_active=True)
            with self.subTest(tokens=tokens):
                self.assertTrue(
                    functions["can_dsa_prefill_cp_round_robin_split"](batch)
                )

    def test_misaligned_prefill_falls_back_in_both_admission_paths(self):
        functions = _load_admission_functions()
        for tokens in (12, 17):
            batch = SimpleNamespace(extend_seq_lens_cpu=[tokens])
            with self.subTest(tokens=tokens):
                self.assertFalse(
                    functions["can_dsa_prefill_cp_round_robin_split"](batch)
                )
                self.assertFalse(
                    functions["can_dsa_cp_split"](tokens, 8, True, batch)
                )

    def test_aligned_prefill_retains_cp_and_padded_mismatch_falls_back(self):
        functions = _load_admission_functions()
        aligned = SimpleNamespace(extend_seq_lens_cpu=[16])
        self.assertTrue(functions["can_dsa_prefill_cp_round_robin_split"](aligned))
        self.assertTrue(functions["can_dsa_cp_split"](16, 8, True, aligned))

        unaligned_real_tokens = SimpleNamespace(extend_seq_lens_cpu=[12])
        self.assertFalse(
            functions["can_dsa_cp_split"](16, 8, True, unaligned_real_tokens)
        )


if __name__ == "__main__":
    unittest.main()
