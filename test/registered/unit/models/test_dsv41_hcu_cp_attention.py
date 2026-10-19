"""Legacy HCU CP must preserve global KV and rank-local query geometry."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.layers.attention import deepseek_v4_backend as backend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import deepseek_v4 as model_module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


def test_hcu_cp_tp_moe_gathers_and_reduce_scatters():
    local_x = torch.zeros((3, 4))
    global_x = torch.ones((24, 4))

    def mlp(x, *args, **kwargs):
        torch.testing.assert_close(x, global_x)
        return x

    obj = SimpleNamespace(mlp=mlp)
    batch = SimpleNamespace(dp_padding_mode=None)
    scoped = Mock(return_value=nullcontext())
    with (
        patch.object(model_module, "_is_hcu", True),
        patch.object(model_module, "is_cp_active", return_value=False),
        patch.object(model_module, "dsa_use_prefill_cp", return_value=True),
        patch.object(model_module, "get_parallel", return_value=SimpleNamespace(attn_dp_size=1, attn_tp_size=1)),
        patch.object(model_module, "get_moe_a2a_backend", return_value=SimpleNamespace(is_none=lambda: True)),
        patch.object(model_module, "get_forward", return_value=SimpleNamespace(scoped=scoped)),
        patch.object(model_module, "_every_row_routed", return_value=nullcontext()),
        patch.object(model_module, "attn_cp_interleave_gather", return_value=global_x) as gather,
        patch.object(model_module, "attn_cp_interleave_reduce_scatter", return_value=local_x) as scatter,
    ):
        result = model_module.DeepseekV4DecoderLayer._run_moe_ffn_dp_sync(
            obj, local_x, batch, input_ids=None, input_ids_global=None,
        )
    gather.assert_called_once()
    scatter.assert_called_once()
    scoped.assert_called_once_with(mlp_reduce_scatter=True)
    torch.testing.assert_close(result, local_x)


@pytest.mark.parametrize("hcu", [True, False])
def test_cp_attention_gathers_global_kv_before_store(hcu):
    from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import env_gate

    x = torch.zeros((3, 4))
    global_kv = torch.ones((24, 4))
    batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND, positions=torch.arange(24))
    store = Mock()
    obj = SimpleNamespace(
        fuse_wqa_wkv=False, wq_a=lambda x: (x, None), wkv=lambda x: (x, None),
        use_fused_qk_norm_rope=False, _normalize_q_lora=lambda x: (x, x),
        _compute_q_b=lambda *args: x, is_dsv41=True, layer_id=0,
        kv_norm=SimpleNamespace(weight=torch.ones(4)), eps=1e-6,
        freqs_cis=None, compress_ratio=1, compressor=None, indexer=None,
        _compute_kv_to_cache=Mock(side_effect=AssertionError("local KV cannot write global slots")),
    )
    attn = SimpleNamespace(forward_metadata=SimpleNamespace(late_layer_tail=None), get_swa_out_cache_loc=lambda fb: torch.arange(24))
    pool = SimpleNamespace(set_swa_key_buffer_radix_fused_norm_rope=store)
    with (
        patch.object(model_module, "_is_hcu", hcu),
        patch.object(model_module, "_is_hip", False),
        patch.object(model_module, "_is_npu", False),
        patch.object(model_module, "is_cp_active", return_value=not hcu),
        patch.object(model_module, "dsa_use_prefill_cp", return_value=hcu),
        patch.object(model_module, "get_token_to_kv_pool", return_value=pool),
        patch.object(model_module, "is_cp_cache_layer_split_deepseek_v4_pool", return_value=True),
        patch.object(model_module, "cp_materialize_global_token_order", return_value=global_kv) as gather,
        patch.object(env_gate, "is_unified_kv_triton", return_value=False),
        patch.object(env_gate, "is_unified_kv_fp8", return_value=False),
        patch.object(torch.cuda, "current_stream", return_value=None),
    ):
        model_module.MQALayer._forward_prepare(obj, x, batch.positions[::8], batch, attn)
    gather.assert_called_once()
    torch.testing.assert_close(store.call_args.kwargs["kv"], global_kv)
    torch.testing.assert_close(store.call_args.kwargs["positions"], batch.positions)


@pytest.mark.parametrize("hcu", [True, False])
def test_low_ratio_cp_compresses_all_real_tokens(hcu):
    lengths = [13, 8]
    total = sum(lengths)
    batch = SimpleNamespace(
        attn_cp_metadata=SimpleNamespace(total_seq_lens=0 if hcu else total),
        extend_seq_lens_cpu=lengths,
        positions=torch.tensor(list(range(13)) + list(range(8)) + [0] * 3),
    )
    global_x = torch.arange(24 * 4).reshape(24, 4)
    compress = Mock()
    obj = SimpleNamespace(
        forward_metadata=SimpleNamespace(late_layer_tail=None),
        _low_ratio_compress_torch=compress,
    )

    def token_indices(fb, *, num_tokens):
        assert num_tokens == total
        return torch.tensor([0] * 13 + [1] * 8)

    with (
        patch.object(backend, "_is_hcu", hcu),
        patch.object(backend, "get_parallel", return_value=SimpleNamespace(attn_cp_rank=0, attn_cp_size=8)),
        patch.object(backend, "token_req_indices", side_effect=token_indices),
        patch.object(backend, "cp_materialize_global_token_order", return_value=global_x),
        patch.object(torch.cuda, "current_stream", return_value=None),
    ):
        backend.DeepseekV4AttnBackend._forward_low_ratio_sources_cp(
            obj, layer=SimpleNamespace(compressor=object()), x=global_x[::8],
            q_lora=None, positions=batch.positions[::8], forward_batch=batch,
            run_compressor=True, run_indexer=False,
        )
    torch.testing.assert_close(compress.call_args.args[1], global_x[:total])
    torch.testing.assert_close(compress.call_args.args[3], batch.positions[:total])


def test_hcu_sparse_prefill_uses_rank_local_query_lengths():
    batch = SimpleNamespace(
        seq_lens_cpu=torch.tensor([13, 8]), seq_lens=torch.tensor([13, 8]),
        extend_seq_lens_cpu=[13, 8], extend_seq_lens=torch.tensor([13, 8]),
        extend_start_loc=torch.tensor([0, 13]), req_pool_indices=torch.tensor([0, 1]),
        forward_mode=ForwardMode.EXTEND,
    )
    core = SimpleNamespace(request_window_layout=None, seq_lens_casual=torch.tensor([1, 9, 4]))
    pool = SimpleNamespace(full_to_swa_index_mapping=None, get_swa_key_page_size=lambda: 64)
    obj = SimpleNamespace(forward_metadata=SimpleNamespace(late_layer_tail=None), token_to_kv_pool=pool, req_to_token=None)
    expected = (torch.tensor([2, 1]), torch.tensor([2, 1]), torch.tensor([0, 8, 3]))
    with (
        patch.object(backend, "_is_hcu", True),
        patch.object(backend, "is_cp_active", return_value=False),
        patch.object(backend, "dsa_use_prefill_cp", return_value=True),
        patch.object(backend, "get_parallel", return_value=SimpleNamespace(attn_cp_rank=0, attn_cp_size=8)),
        patch.object(backend, "build_cp_sparse_query_metadata", return_value=expected) as build_cp,
        patch.object(backend.SparsePrefillChunkCache, "build", return_value=None) as build,
    ):
        backend.DeepseekV4AttnBackend._build_sparse_prefill_chunk_cache(obj, batch, core, num_qo_tokens=3)
    build_cp.assert_called_once()
    torch.testing.assert_close(build.call_args.kwargs["query_lens"], expected[0])
    torch.testing.assert_close(build.call_args.kwargs["real_query_seq_lens"], expected[1])
