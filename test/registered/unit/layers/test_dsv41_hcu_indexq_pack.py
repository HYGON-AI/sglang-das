"""The HCU decode/verify entry point preserves the exact indexer input ABI."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import pytest
import torch
from sglang.srt.layers.attention import deepseek_v4_backend as b
from sglang.srt.layers.attention.dsv4.dsv41_sparse import rope_tail
from sglang.kernels.ops.attention.dsv4.torch_quant import fake_quant_fp4
from sglang.kernels.ops.attention.dsv4.fp4_indexer import quantize_fp4_indexer_tensor
from sglang.kernels.ops.attention.deepseek_v4_rope import precompute_freqs_cis
from sglang.kernels.ops.attention.dsv4.fp4_indexer_rope_hcu import (
    index_q_rope_pack_weights_hcu,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not b._is_hcu, reason="HCU required"
)


def fixture(t, heads=32, layout=None):
    torch.manual_seed(20261028 + t)
    q = torch.randn(t, heads * 128, device="cuda").bfloat16()
    raw = torch.randn(t, heads, device="cuda").bfloat16()
    f = precompute_freqs_cis(64, 4096, 0, 10000, 1, 32, 1).to("cuda")
    pos = torch.arange(t, device="cuda", dtype=torch.int64) * 3
    x = torch.randn(t, 5120, device="cuda").bfloat16()
    ql = torch.randn(t, 128, device="cuda").bfloat16()
    if layout == "x":
        x = torch.empty(t, 10240, device="cuda", dtype=torch.bfloat16)[:, ::2]
    if layout == "q_lora":
        ql = torch.empty(t, 256, device="cuda", dtype=torch.bfloat16)[:, ::2]
    if layout == "positions":
        z = torch.empty(t * 2, device="cuda", dtype=pos.dtype)
        z[::2] = pos
        pos = z[::2]
    if layout == "freqs":
        z = torch.empty(4096, 64, device="cuda", dtype=torch.complex64)
        z[:, ::2] = f
        f = z[:, ::2]
    scale = 128**-0.5 * heads**-0.5
    indexer = SimpleNamespace(
        n_local_heads=heads,
        n_heads=heads,
        index_head_dim=128,
        rope_head_dim=64,
        head_weight_scale=scale,
        wq_b=lambda _: (q, None),
        head_weights_raw=lambda _: raw,
        queries=Mock(
            side_effect=lambda _, freq: fake_quant_fp4(
                rope_tail(q.view(t, heads, 128), freq, 64)
            )
        ),
        head_weights=lambda _: (raw * scale),
        uses_candidates=False,
        is_candidate_source=False,
    )
    layer = SimpleNamespace(compress_ratio=1, layer_id=0, indexer=indexer, freqs_cis=f)
    meta = SimpleNamespace(
        compressed_page_size=64,
        compressed_seq_lens=torch.full((t,), 512, device="cuda", dtype=torch.int32),
        page_table=torch.zeros(t, 8, device="cuda", dtype=torch.int32),
        deep_gemm_metadata=None,
        max_compressed_seq_len=512,
        rows_per_request=1,
    )
    core = SimpleNamespace(
        sparse_page_indices=lambda _: torch.empty(
            t, 512, device="cuda", dtype=torch.int32
        ),
        sparse_raw_indices=lambda _: None,
    )
    self = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(
            get_index_k_with_scale_buffer=lambda _: torch.zeros(
                8, 64 * 68, device="cuda", dtype=torch.uint8
            )
        ),
        forward_metadata=SimpleNamespace(
            core_metadata=core, c1_indexer_metadata=meta, c2_indexer_metadata=meta
        ),
    )
    return self, layer, x, ql, pos


def invoke(data, flag, native=False):
    self, layer, x, ql, pos = data
    captured = []

    def capture(q, cache, weights, *args, **kw):
        captured.append((q[0].clone(), q[1].clone(), weights.clone()))
        return torch.zeros(x.shape[0], 512, device="cuda")

    with patch.object(b, "_use_hcu_index_q_rope_pack", flag), patch.object(
        b, "_has_native_hcu_rope_fake_quant", return_value=native
    ), patch.object(b, "_supports_index_q_rope_pack", return_value=False), patch.object(
        b, "_every_request_fits", return_value=True
    ), patch.object(
        b, "deep_gemm_fp4_paged_mqa_logits", side_effect=capture
    ), patch.object(
        b, "topk_transform_paged_from_metadata"
    ), patch(
        "sglang.srt.model_executor.runner_utils.capture_mode.skip_low_ratio_indexer",
        return_value=False,
    ), patch(
        "sglang.kernels.ops.attention.dsv4.fp4_indexer_rope_hcu.index_q_rope_pack_weights_hcu",
        wraps=index_q_rope_pack_weights_hcu,
    ) as fast:
        b.DeepseekV4AttnBackend._low_ratio_index_topk_decode(self, layer, x, ql, pos)
    assert len(captured) == 1
    return captured[0], fast.call_count


@pytest.mark.parametrize("tokens", [1, 6, 12, 24])
@pytest.mark.parametrize("heads", [32, 64])
def test_supported_route_exact(tokens, heads):
    data = fixture(tokens, heads)
    expected, calls = invoke(data, False)
    assert calls == 0
    data[1].indexer.queries.reset_mock()
    actual, calls = invoke(data, True)
    assert calls == 1 and data[1].indexer.queries.call_count == 0
    for a, z in zip(actual, expected):
        assert a.dtype == z.dtype and torch.equal(a, z)


@pytest.mark.parametrize("layout", ["x", "q_lora", "positions", "freqs"])
def test_strided_inputs_keep_original_route(layout):
    data = fixture(6, layout=layout)
    expected, _ = invoke(data, False)
    data[1].indexer.queries.reset_mock()
    actual, calls = invoke(data, True)
    assert calls == 0 and data[1].indexer.queries.call_count == 1
    for a, z in zip(actual, expected):
        assert torch.equal(a, z)


def test_unsupported_heads_keep_original_route():
    data = fixture(6, heads=16)
    expected, _ = invoke(data, False)
    actual, calls = invoke(data, True)
    assert calls == 0
    for a, z in zip(actual, expected):
        assert torch.equal(a, z)


def test_native_aot_reference_keeps_original_route():
    data = fixture(6)
    expected, _ = invoke(data, False)
    actual, calls = invoke(data, True, native=True)
    assert calls == 0
    for a, z in zip(actual, expected):
        assert torch.equal(a, z)


@pytest.mark.parametrize(
    "case",
    json.loads(Path(__file__).with_name("dsv41_hcu_indexq_rounding.json").read_text()),
)
def test_rope_fp4_rounding_boundary_regression(case):
    tokens = case["tokens"]
    q = torch.zeros(tokens, 32, 128, device="cuda", dtype=torch.bfloat16)
    q[case["token"], case["head"]] = torch.tensor(
        case["q_row"], device="cuda", dtype=torch.bfloat16
    )
    freqs = torch.ones(tokens, 32, device="cuda", dtype=torch.complex64)
    freqs[case["token"]] = torch.view_as_complex(
        torch.tensor(case["freq_row"], device="cuda", dtype=torch.float32)
    )
    positions = torch.arange(tokens, device="cuda")
    raw = torch.zeros(tokens, 32, device="cuda", dtype=torch.bfloat16)
    expected = quantize_fp4_indexer_tensor(
        fake_quant_fp4(rope_tail(q, freqs[positions], 64)).flatten(0, 1), rne=True
    )
    payload, scale, weights = index_q_rope_pack_weights_hcu(
        q, freqs, positions, raw, 1 / 64
    )
    assert torch.equal(payload, expected[0]) and torch.equal(scale, expected[1])
    assert torch.equal(weights, raw.float())


def test_default_enabled_in_fresh_process():
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env.pop("SGLANG_HCU_INDEX_Q_ROPE_PACK", None)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from sglang.srt.layers.attention import deepseek_v4_backend as b; assert b._is_hcu and b._use_hcu_index_q_rope_pack",
        ],
        env=env,
        check=True,
    )
