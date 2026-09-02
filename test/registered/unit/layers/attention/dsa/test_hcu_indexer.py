"""Unit coverage for the HCU DSA indexer's LightOp contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, sentinel

import torch

from sglang.srt.layers.attention.dsa import dsa_indexer as indexer_module
from sglang.srt.layers.attention.dsa import paged_mqa_logits_backend as backend_module
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.srt.layers.attention.dsa.paged_mqa_logits_backend import (
    DSAPagedMQALogitsBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHCUDSAPagedMQABackend(CustomTestCase):
    def test_hcu_resolves_before_generic_hip(self):
        with (
            patch.object(backend_module, "is_hcu", return_value=True),
            patch.object(backend_module, "is_hip", return_value=True),
        ):
            self.assertEqual(
                DSAPagedMQALogitsBackend.resolve("auto"),
                DSAPagedMQALogitsBackend.LIGHTOP,
            )
            self.assertEqual(
                DSAPagedMQALogitsBackend.resolve("lightop"),
                DSAPagedMQALogitsBackend.LIGHTOP,
            )
            with self.assertRaisesRegex(ValueError, "only 'lightop'"):
                DSAPagedMQALogitsBackend.resolve("aiter")

    def test_non_hcu_hip_keeps_aiter(self):
        with (
            patch.object(backend_module, "is_hcu", return_value=False),
            patch.object(backend_module, "is_hip", return_value=True),
        ):
            self.assertEqual(
                DSAPagedMQALogitsBackend.resolve("auto"),
                DSAPagedMQALogitsBackend.AITER,
            )


class TestHCUDSAIndexerLightOpContracts(CustomTestCase):
    def test_qk_prepare_uses_lightop_fused_layernorm_rope(self):
        indexer = object.__new__(Indexer)
        indexer.wq_b = MagicMock(return_value=(sentinel.query_projection, None))
        indexer.wk = MagicMock(return_value=(sentinel.key_projection, None))
        indexer.head_dim = 128
        indexer.k_norm = SimpleNamespace(
            weight=sentinel.norm_weight,
            bias=sentinel.norm_bias,
            variance_epsilon=1e-6,
        )
        indexer.rotary_emb = SimpleNamespace(cos_sin_cache=sentinel.cos_sin_cache)
        indexer.dsa_enable_prefill_cp = False
        query = MagicMock()
        key = MagicMock(ndim=3)
        forward_batch = SimpleNamespace(attn_cp_metadata=None)
        lightop_attention = MagicMock()

        with (
            patch.object(indexer_module, "_is_hcu", True),
            patch.object(indexer_module, "rearrange", return_value=query),
            patch.object(
                indexer_module, "rotate_activation", side_effect=lambda x, **_: x
            ),
            patch.object(indexer_module, "is_cp_v2_active", return_value=False),
            patch.object(
                indexer_module,
                "lightop_attention",
                lightop_attention,
                create=True,
            ),
        ):
            got_query, got_key, weights_raw = indexer._get_q_k_bf16(
                sentinel.q_lora,
                sentinel.x,
                sentinel.positions,
                False,
                forward_batch,
            )

        self.assertIs(got_query, query)
        self.assertIs(got_key, key)
        self.assertIsNone(weights_raw)
        lightop_attention.fuse_layernorm_rotary_embedding.assert_called_once_with(
            sentinel.positions,
            query,
            key,
            128,
            sentinel.cos_sin_cache,
            False,
            None,
            None,
            sentinel.norm_weight,
            sentinel.norm_bias,
            None,
            None,
            1e-6,
        )

    def test_paged_and_ragged_mqa_abis(self):
        lightop_attention = MagicMock()
        lightop_attention.paged_mqa_logits.return_value = sentinel.paged_logits
        lightop_attention.mqa_logits.return_value = sentinel.ragged_logits
        q = MagicMock()
        q.unsqueeze.return_value = sentinel.q_next_n

        with patch.object(
            indexer_module,
            "lightop_attention",
            lightop_attention,
            create=True,
        ):
            paged_result = indexer_module._hcu_paged_mqa_logits(
                q,
                sentinel.kv_cache,
                sentinel.weights,
                sentinel.seqlens,
                sentinel.block_tables,
                None,
                4096,
            )
            ragged_result = indexer_module._hcu_mqa_logits(
                sentinel.q,
                sentinel.kv,
                sentinel.weights,
                sentinel.ks,
                sentinel.ke,
                sentinel.kv_scale,
            )

        self.assertIs(paged_result, sentinel.paged_logits)
        lightop_attention.paged_mqa_logits.assert_called_once_with(
            sentinel.q_next_n,
            sentinel.kv_cache,
            sentinel.weights,
            sentinel.seqlens,
            sentinel.block_tables,
            None,
            4096,
            clean_logits=True,
        )
        self.assertIs(ragged_result, sentinel.ragged_logits)
        lightop_attention.mqa_logits.assert_called_once_with(
            sentinel.q,
            sentinel.kv,
            sentinel.weights,
            sentinel.ks,
            sentinel.ke,
            kv_scale=sentinel.kv_scale,
            clean_logit=True,
        )

    def test_paged_cache_layout_selects_bf16_or_fp8(self):
        indexer = object.__new__(Indexer)
        indexer.head_dim = 128
        bf16_cache = torch.empty((2, 64, 1, 128), dtype=torch.bfloat16)
        bf16_pool = SimpleNamespace(
            use_fp8_index_k_cache=False,
            get_index_k_buffer=MagicMock(return_value=bf16_cache),
        )
        fp8_raw = torch.empty((2, 64 * 132), dtype=torch.uint8)
        fp8_pool = SimpleNamespace(
            use_fp8_index_k_cache=True,
            page_size=64,
            get_index_k_with_scale_buffer=MagicMock(return_value=fp8_raw),
        )

        with patch.object(indexer_module, "_is_hcu", True):
            got_bf16, is_bf16 = indexer._get_hcu_paged_index_k_cache(
                bf16_pool, layer_id=3
            )
            got_fp8, is_fp8_bf16 = indexer._get_hcu_paged_index_k_cache(
                fp8_pool, layer_id=3
            )

        self.assertIs(got_bf16, bf16_cache)
        self.assertTrue(is_bf16)
        self.assertEqual(got_fp8.shape, (2, 64, 1, 132))
        self.assertFalse(is_fp8_bf16)
        bf16_pool.get_index_k_buffer.assert_called_once_with(layer_id=3)
        fp8_pool.get_index_k_with_scale_buffer.assert_called_once_with(layer_id=3)

    def test_hcu_cache_store_uses_native_layout(self):
        indexer = object.__new__(Indexer)
        forward_batch = SimpleNamespace(out_cache_loc=MagicMock())
        key = sentinel.key
        bf16_pool = SimpleNamespace(
            use_fp8_index_k_cache=False,
            set_index_k_buffer=MagicMock(),
        )
        fp8_pool = SimpleNamespace(
            use_fp8_index_k_cache=True,
            page_size=64,
            get_index_k_with_scale_buffer=MagicMock(return_value=sentinel.fp8_cache),
        )
        lightop_kvcache = MagicMock()

        with (
            patch.object(indexer_module, "_is_hcu", True),
            patch.object(
                indexer_module,
                "lightop_kvcache",
                lightop_kvcache,
                create=True,
            ),
            patch.object(
                indexer_module, "get_token_to_kv_pool", return_value=bf16_pool
            ),
        ):
            indexer._store_index_k_cache(forward_batch, layer_id=4, key=key)

        bf16_pool.set_index_k_buffer.assert_called_once_with(
            layer_id=4,
            loc=forward_batch.out_cache_loc,
            index_k=key,
        )
        lightop_kvcache.fuse_act_quant_and_store_index_k_cache.assert_not_called()

        contiguous_loc = sentinel.contiguous_loc
        forward_batch.out_cache_loc.contiguous.return_value = contiguous_loc
        with (
            patch.object(indexer_module, "_is_hcu", True),
            patch.object(indexer_module, "_is_fp8_fnuz", False),
            patch.object(
                indexer_module,
                "lightop_kvcache",
                lightop_kvcache,
                create=True,
            ),
            patch.object(indexer_module, "get_token_to_kv_pool", return_value=fp8_pool),
        ):
            indexer._store_index_k_cache(forward_batch, layer_id=4, key=key)

        lightop_kvcache.fuse_act_quant_and_store_index_k_cache.assert_called_once_with(
            key,
            sentinel.fp8_cache,
            contiguous_loc,
            64,
            1e-5,
            False,
            True,
        )

    def test_fused_qk_quant_store_keeps_lightop_scale_contract(self):
        indexer = object.__new__(Indexer)
        indexer.hidden_size = 6144
        indexer.head_dim = 128
        indexer.softmax_scale = 0.125
        pool = SimpleNamespace(
            page_size=64,
            get_index_k_with_scale_buffer=MagicMock(return_value=sentinel.fp8_cache),
        )
        out_cache_loc = MagicMock()
        out_cache_loc.contiguous.return_value = sentinel.contiguous_loc
        expected = (sentinel.q_fp8, sentinel.q_scale, None)
        lightop_kvcache = MagicMock()
        lightop_kvcache.fuse_qk_quant_and_store_index_k_cache.return_value = expected

        with (
            patch.object(indexer_module, "_is_fp8_fnuz", False),
            patch.object(
                indexer_module,
                "lightop_kvcache",
                lightop_kvcache,
                create=True,
            ),
        ):
            result = indexer._hcu_fused_qk_quant_and_store(
                sentinel.query,
                sentinel.key,
                pool,
                layer_id=5,
                out_cache_loc=out_cache_loc,
            )

        self.assertEqual(result, expected)
        hadamard_scale = 128**-0.5
        lightop_kvcache.fuse_qk_quant_and_store_index_k_cache.assert_called_once_with(
            sentinel.query,
            sentinel.key,
            sentinel.fp8_cache,
            sentinel.contiguous_loc,
            64,
            None,
            hadamard_scale * 0.125,
            hadamard_scale,
            1e-5,
            False,
            True,
        )


if __name__ == "__main__":
    unittest.main()
