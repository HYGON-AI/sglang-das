"""Grammar initialization follows the published parallel configuration."""

import unittest
from types import SimpleNamespace

from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class GrammarManagerContextTest(CustomTestCase):
    def test_tp_and_cp_initialization_without_retired_scheduler_fields(self):
        for cp_size, attn_dp_size in ((1, 1), (4, 2)):
            with (
                self.subTest(cp_size=cp_size),
                published_topology(
                    skip_tokenizer_init=True,
                    tp_size=8,
                    attn_cp_size=cp_size,
                    attn_dp_size=attn_dp_size,
                ),
            ):
                cp_group = object()
                scheduler = SimpleNamespace(
                    server_args=SimpleNamespace(),
                    dp_tp_cpu_group=None,
                    dp_tp_group=SimpleNamespace(
                        world_size=8 // (cp_size * attn_dp_size),
                        first_rank=0,
                        is_first_rank=True,
                    ),
                    attn_cp_cpu_group=cp_group,
                    pp_group=None,
                )
                manager = GrammarManager(scheduler)
                self.assertIsNone(manager.grammar_backend)
                self.assertEqual(manager.grammar_cp_sync_size, cp_size)
                self.assertIs(manager.grammar_cp_sync_group, cp_group)


if __name__ == "__main__":
    unittest.main()
