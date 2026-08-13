import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class TestHiSparseCapacity(unittest.TestCase):
    def test_memory_profile_accounts_for_expanded_index_cache(self):
        from sglang.srt.model_executor.pool_configurator import (
            DefaultPoolConfigurator,
        )

        runner = SimpleNamespace(
            enable_hisparse=True,
            kv_cache_dtype=torch.float8_e4m3fn,
            model_config=SimpleNamespace(
                hf_config=object(),
                kv_lora_rank=512,
                qk_rope_head_dim=64,
            ),
            server_args=SimpleNamespace(
                hisparse_config=(
                    '{"top_k":2048,"device_buffer_size":12288,"host_to_device_ratio":20}'
                )
            ),
            use_mla_backend=True,
        )

        with (
            patch(
                "sglang.srt.model_executor.pool_configurator.is_deepseek_nsa",
                return_value=True,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator.get_nsa_index_head_dim",
                return_value=128,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator.is_float4_e2m1fn_x2",
                return_value=False,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator.get_attention_tp_size",
                return_value=1,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator._is_hcu",
                True,
                create=True,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator.is_hcu_native_fp8_supported",
                return_value=False,
                create=True,
            ),
        ):
            configurator = DefaultPoolConfigurator.__new__(DefaultPoolConfigurator)
            cell_size = configurator._compute_cell_size(runner, 2)

        mla_bytes = (512 + 64) * 2 * torch.float8_e4m3fn.itemsize
        expanded_index_bytes = 128 * 2 * torch.bfloat16.itemsize * 20
        self.assertEqual(cell_size, mla_bytes + expanded_index_bytes)

    def test_scheduler_uses_hisparse_logical_capacity(self):
        from sglang.srt.managers.tp_worker import _get_effective_token_capacity

        runner = SimpleNamespace(
            max_total_num_tokens=62016,
            token_to_kv_pool_allocator=SimpleNamespace(size=1240320),
        )

        self.assertEqual(_get_effective_token_capacity(runner, True), 1240320)
        self.assertEqual(_get_effective_token_capacity(runner, False), 62016)


if __name__ == "__main__":
    unittest.main()
