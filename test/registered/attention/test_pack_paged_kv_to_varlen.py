import importlib
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))


def _import_pack_module():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("Test requires an available GPU")

    return importlib.import_module("sglang.srt.layers.attention.pack_paged_kv_to_varlen")


class TestPackPagedKVToVarlen(unittest.TestCase):
    def test_pack_paged_kv_to_varlen_multi_batch(self):
        pack_module = _import_pack_module()

        page_size = 4
        num_pages = 16
        num_heads = 2
        head_dim = 3
        seq_lens = torch.tensor([5, 9, 12], dtype=torch.int32)
        page_table = torch.tensor(
            [
                [3, 1, 0],
                [8, 2, 5],
                [6, 7, 4],
            ],
            dtype=torch.int32,
            device="cuda",
        )

        key_tokens = torch.arange(
            num_pages * page_size * num_heads * head_dim,
            dtype=torch.float16,
            device="cuda",
        ).reshape(num_pages, page_size, num_heads, head_dim)
        value_tokens = -key_tokens

        key_cache = key_tokens.permute(0, 2, 1, 3).contiguous()
        value_cache = value_tokens.permute(0, 2, 3, 1).contiguous()

        packed_k, packed_v = pack_module.pack_paged_kv_to_varlen(
            key_cache,
            value_cache,
            page_table,
            seq_lens,
            page_size,
        )

        expected_k = []
        expected_v = []
        for batch_idx, seq_len in enumerate(seq_lens.tolist()):
            page_count = (seq_len + page_size - 1) // page_size
            pages = page_table[batch_idx, :page_count].to(torch.long)
            expected_k.append(
                key_tokens.index_select(0, pages).reshape(
                    -1, num_heads, head_dim
                )[:seq_len]
            )
            expected_v.append(
                value_tokens.index_select(0, pages).reshape(
                    -1, num_heads, head_dim
                )[:seq_len]
            )

        self.assertTrue(torch.equal(packed_k, torch.cat(expected_k, dim=0)))
        self.assertTrue(torch.equal(packed_v, torch.cat(expected_v, dim=0)))


if __name__ == "__main__":
    unittest.main()
