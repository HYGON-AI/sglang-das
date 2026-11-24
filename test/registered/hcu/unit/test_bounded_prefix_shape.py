import types
import unittest

from sglang.srt.model_executor.runner.hcu_bounded_prefix_graph import (
    _PRE_TAIL_BUCKETS,
    _PRE_TAIL_MIN_ROWS,
    _TAIL_BUCKETS,
    _TAIL_MIN_ROWS,
    _admit_prefix_bucket,
    _layers_topk_specs_ready,
    _pick_row_bucket,
)
from sglang.test.ci.ci_register import register_hcu_ci

register_hcu_ci(est_time=8, suite="stage-b-test-1-hcu-small")


class TestBoundedPrefixShape(unittest.TestCase):
    def test_pre_tail_exact_buckets_across_input_lengths(self):
        # 4K (32/req), 32K (64/req), 64K (96/req), 128K (192/req)
        for live_rows, expected in [
            (32, 32),
            (64, 64),
            (96, 96),
            (192, 192),
            (256, 256),
            (384, 384),
            (512, 512),
        ]:
            self.assertEqual(
                _admit_prefix_bucket(
                    live_rows,
                    [32256],
                    _PRE_TAIL_BUCKETS,
                    min_rows=_PRE_TAIL_MIN_ROWS,
                    max_rows=512,
                ),
                expected,
            )

    def test_tail_buckets_and_rounding(self):
        self.assertEqual(
            _admit_prefix_bucket(
                16, [32256], _TAIL_BUCKETS, min_rows=_TAIL_MIN_ROWS, max_rows=192
            ),
            16,
        )
        self.assertEqual(
            _admit_prefix_bucket(
                176, [32256] * 11, _TAIL_BUCKETS, min_rows=_TAIL_MIN_ROWS, max_rows=192
            ),
            192,
        )
        self.assertIsNone(
            _admit_prefix_bucket(
                240, [32256] * 15, _TAIL_BUCKETS, min_rows=_TAIL_MIN_ROWS, max_rows=192
            )
        )

    def test_cold_and_tiny_warmup_batches_stay_eager(self):
        self.assertIsNone(
            _admit_prefix_bucket(
                64, [0], _PRE_TAIL_BUCKETS, min_rows=_PRE_TAIL_MIN_ROWS, max_rows=512
            )
        )
        self.assertIsNone(
            _admit_prefix_bucket(
                64, [], _PRE_TAIL_BUCKETS, min_rows=_PRE_TAIL_MIN_ROWS, max_rows=512
            )
        )
        self.assertIsNone(
            _admit_prefix_bucket(
                2, [16], _PRE_TAIL_BUCKETS, min_rows=_PRE_TAIL_MIN_ROWS, max_rows=512
            )
        )
        self.assertIsNone(
            _admit_prefix_bucket(
                2, [16], _TAIL_BUCKETS, min_rows=_TAIL_MIN_ROWS, max_rows=192
            )
        )

    def test_compute_saturated_batches_and_custom_cap(self):
        self.assertIsNone(
            _pick_row_bucket(960, _PRE_TAIL_BUCKETS, max_rows=512, min_rows=32)
        )
        self.assertIsNone(
            _pick_row_bucket(384, _PRE_TAIL_BUCKETS, max_rows=256, min_rows=32)
        )
        self.assertEqual(
            _pick_row_bucket(240, _PRE_TAIL_BUCKETS, max_rows=512, min_rows=32), 256
        )

    def test_topk_specs_ready_gate(self):
        dense = types.SimpleNamespace(mlp=types.SimpleNamespace())
        moe_unwarmed = types.SimpleNamespace(mlp=types.SimpleNamespace(experts=object()))
        moe_warmed = types.SimpleNamespace(
            mlp=types.SimpleNamespace(experts=object(), _prefix_topk_spec=((8,), int))
        )
        self.assertFalse(_layers_topk_specs_ready([dense, moe_unwarmed]))
        self.assertTrue(_layers_topk_specs_ready([dense, moe_warmed]))


if __name__ == "__main__":
    unittest.main()
