# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum
from sglang.multimodal_gen.runtime.platforms.rocm import RocmPlatform


def test_rocm_platform_maps_sla_to_hcu_flash_attn_backend():
    backend_cls = RocmPlatform.get_attn_backend_cls_str(
        AttentionBackendEnum.SLA_ATTN,
        head_size=128,
        dtype=torch.bfloat16,
    )

    assert (
        backend_cls
        == "sglang.multimodal_gen.runtime.layers.attention.backends.rocm_sparse_linear_attn.RocmSparseLinearAttentionBackend"
    )


def test_rocm_sparse_linear_attention_calls_flash_attn_sla(monkeypatch):
    calls = {}

    def fake_sparse_attn_with_sla(
        q,
        k,
        v,
        topk=0.2,
        feature_map="softmax",
        use_bf16=True,
        use_fp8=False,
        *,
        return_sparsity=False,
    ):
        calls["q_shape"] = q.shape
        calls["k_shape"] = k.shape
        calls["v_shape"] = v.shape
        calls["topk"] = topk
        calls["feature_map"] = feature_map
        calls["use_bf16"] = use_bf16
        calls["use_fp8"] = use_fp8
        calls["return_sparsity"] = return_sparsity
        return q + k + v

    monkeypatch.setitem(
        sys.modules,
        "flash_attn",
        SimpleNamespace(sparse_attn_with_sla=fake_sparse_attn_with_sla),
    )

    from sglang.multimodal_gen.runtime.layers.attention.backends.rocm_sparse_linear_attn import (
        RocmSparseLinearAttentionImpl,
    )

    impl = RocmSparseLinearAttentionImpl(
        num_heads=2,
        head_size=8,
        topk_ratio=0.25,
        feature_map="relu",
        use_bf16=False,
        use_fp8=True,
    )

    assert impl.proj_l.weight.missing_param_init == "zeros"
    assert impl.proj_l.bias.missing_param_init == "zeros"

    query = torch.randn(1, 4, 2, 8)
    key = torch.randn(1, 4, 2, 8)
    value = torch.randn(1, 4, 2, 8)

    output = impl(query, key, value, None)

    assert calls == {
        "q_shape": torch.Size([1, 4, 2, 8]),
        "k_shape": torch.Size([1, 4, 2, 8]),
        "v_shape": torch.Size([1, 4, 2, 8]),
        "topk": 0.25,
        "feature_map": "relu",
        "use_bf16": False,
        "use_fp8": True,
        "return_sparsity": False,
    }
    torch.testing.assert_close(output, query + key + value)



def test_rocm_sparse_linear_attention_calls_sla_without_compute_flags(monkeypatch):
    calls = {}

    def fake_sparse_attn_with_sla(
        q,
        k,
        v,
        topk=0.2,
        feature_map="softmax",
        *,
        return_sparsity=False,
    ):
        calls["topk"] = topk
        calls["feature_map"] = feature_map
        calls["return_sparsity"] = return_sparsity
        return q + k + v

    monkeypatch.setitem(
        sys.modules,
        "flash_attn",
        SimpleNamespace(sparse_attn_with_sla=fake_sparse_attn_with_sla),
    )

    from sglang.multimodal_gen.runtime.layers.attention.backends.rocm_sparse_linear_attn import (
        RocmSparseLinearAttentionImpl,
    )

    impl = RocmSparseLinearAttentionImpl(
        num_heads=2,
        head_size=8,
        topk_ratio=0.4,
        feature_map="relu",
        use_bf16=False,
        use_fp8=True,
    )

    query = torch.randn(1, 4, 2, 8)
    key = torch.randn(1, 4, 2, 8)
    value = torch.randn(1, 4, 2, 8)

    output = impl(query, key, value, None)

    assert calls == {
        "topk": 0.4,
        "feature_map": "relu",
        "return_sparsity": False,
    }
    torch.testing.assert_close(output, query + key + value)


def test_rocm_sparse_linear_attention_skips_zero_linear_branch_by_default(monkeypatch):
    def fake_sparse_attn_with_sla(q, k, v, **kwargs):
        return q + k + v

    monkeypatch.setitem(
        sys.modules,
        "flash_attn",
        SimpleNamespace(sparse_attn_with_sla=fake_sparse_attn_with_sla),
    )

    from sglang.multimodal_gen.runtime.layers.attention.backends.rocm_sparse_linear_attn import (
        RocmSparseLinearAttentionImpl,
    )

    impl = RocmSparseLinearAttentionImpl(
        num_heads=2,
        head_size=8,
        use_bf16=False,
    )

    def fail_linear_attention(query, key, value):
        raise AssertionError("linear branch should be skipped by default")

    monkeypatch.setattr(impl, "_linear_attention", fail_linear_attention)

    query = torch.randn(1, 4, 2, 8)
    key = torch.randn(1, 4, 2, 8)
    value = torch.randn(1, 4, 2, 8)

    output = impl(query, key, value, None)

    torch.testing.assert_close(output, query + key + value)


def test_rocm_sparse_linear_attention_can_enable_linear_branch(monkeypatch):
    def fake_sparse_attn_with_sla(q, k, v, **kwargs):
        return q + k + v

    monkeypatch.setitem(
        sys.modules,
        "flash_attn",
        SimpleNamespace(sparse_attn_with_sla=fake_sparse_attn_with_sla),
    )

    from sglang.multimodal_gen.runtime.layers.attention.backends.rocm_sparse_linear_attn import (
        RocmSparseLinearAttentionImpl,
    )

    impl = RocmSparseLinearAttentionImpl(
        num_heads=2,
        head_size=8,
        skip_linear_branch=False,
        use_bf16=False,
    )

    query = torch.randn(1, 4, 2, 8)
    key = torch.randn(1, 4, 2, 8)
    value = torch.randn(1, 4, 2, 8)
    linear_output = torch.ones_like(query)

    monkeypatch.setattr(impl, "_linear_attention", lambda q, k, v: linear_output)

    output = impl(query, key, value, None)

    torch.testing.assert_close(output, query + key + value + linear_output)


def test_minimal_a2a_accepts_rocm_sparse_linear_attention_backend(monkeypatch):
    from sglang.multimodal_gen.runtime.layers.attention import turbo_layer
    from sglang.multimodal_gen.runtime.layers.attention.backends.rocm_sparse_linear_attn import (
        RocmSparseLinearAttentionBackend,
        RocmSparseLinearAttentionImpl,
    )

    monkeypatch.setattr(
        turbo_layer,
        "get_attn_backend",
        lambda head_size, dtype, supported_attention_backends=None: RocmSparseLinearAttentionBackend,
    )

    attn_op = turbo_layer.MinimalA2AAttnOp(
        num_heads=2,
        head_size=8,
        attention_type="sla",
        topk=0.25,
    )

    assert isinstance(attn_op.local_attn, RocmSparseLinearAttentionImpl)


def test_minimal_a2a_passes_local_blocks_to_rocm_sparse_linear_attention(monkeypatch):
    from sglang.multimodal_gen.runtime.layers.attention import turbo_layer
    from sglang.multimodal_gen.runtime.layers.attention.backends.rocm_sparse_linear_attn import (
        RocmSparseLinearAttentionBackend,
    )

    monkeypatch.setattr(
        turbo_layer,
        "get_attn_backend",
        lambda head_size, dtype, supported_attention_backends=None: RocmSparseLinearAttentionBackend,
    )

    attn_op = turbo_layer.MinimalA2AAttnOp(
        num_heads=2,
        head_size=8,
        attention_type="sla",
        topk=0.25,
        local_blocks=2,
    )

    assert attn_op.local_attn.local_blocks == 2


def test_minimal_a2a_binds_ulysses_group_when_sequence_sharded(monkeypatch):
    from sglang.multimodal_gen.runtime.layers.attention import turbo_layer
    from sglang.multimodal_gen.runtime.layers.attention.backends.rocm_sparse_linear_attn import (
        RocmSparseLinearAttentionBackend,
    )

    fake_group = object()
    fake_sp_group = SimpleNamespace(ulysses_group=fake_group, ulysses_world_size=4)

    monkeypatch.setattr(
        turbo_layer,
        "get_attn_backend",
        lambda head_size, dtype, supported_attention_backends=None: RocmSparseLinearAttentionBackend,
    )
    monkeypatch.setattr(turbo_layer.dist, "is_available", lambda: True)
    monkeypatch.setattr(turbo_layer.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(turbo_layer, "get_sp_group", lambda: fake_sp_group, raising=False)

    attn_op = turbo_layer.MinimalA2AAttnOp(
        num_heads=4,
        head_size=8,
        attention_type="sla",
        topk=0.25,
    )

    attn_op._maybe_bind_sequence_parallel_group(
        torch.empty(1, 4, 4, 8),
        sequence_shard_enabled=True,
    )

    assert attn_op.pg is fake_group


def test_rocm_sparse_linear_attention_adds_local_blocks_to_sparse_map():
    from sglang.multimodal_gen.runtime.layers.attention.backends.rocm_sparse_linear_attn import (
        RocmSparseLinearAttentionImpl,
    )

    sparse_map = torch.zeros((1, 1, 5, 5), dtype=torch.int8)

    RocmSparseLinearAttentionImpl._add_local_blocks_(sparse_map, local_blocks=1)

    expected = torch.tensor(
        [
            [
                [
                    [1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1],
                ]
            ]
        ],
        dtype=torch.int8,
    )
    torch.testing.assert_close(sparse_map, expected)


def test_rocm_sparse_linear_attention_selects_kernel_block_m():
    from sglang.multimodal_gen.runtime.layers.attention.backends.rocm_sparse_linear_attn import (
        RocmSparseLinearAttentionImpl,
    )

    assert RocmSparseLinearAttentionImpl._select_sparse_block_m(2048) == 64
    assert RocmSparseLinearAttentionImpl._select_sparse_block_m(2049) == 128
    assert RocmSparseLinearAttentionImpl._select_sparse_block_m(18900) == 128


def test_rocm_sparse_linear_attention_aligns_local_blocks_by_token_range():
    from sglang.multimodal_gen.runtime.layers.attention.backends.rocm_sparse_linear_attn import (
        RocmSparseLinearAttentionImpl,
    )

    sparse_map = torch.zeros((1, 1, 3, 8), dtype=torch.int8)

    RocmSparseLinearAttentionImpl._add_local_blocks_(
        sparse_map,
        local_blocks=1,
        block_m=128,
        block_k=64,
    )

    expected = torch.tensor(
        [
            [
                [
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 0],
                ]
            ]
        ],
        dtype=torch.int8,
    )
    torch.testing.assert_close(sparse_map, expected)


def test_wan_sla_backend_switches_default_attention_type():
    from sglang.multimodal_gen.runtime.models.dits import wanvideo

    assert wanvideo.resolve_wan_attention_type("original", "sla_attn") == "sla"
    assert (
        wanvideo.resolve_wan_attention_type("original", "sage_sla_attn")
        == "sagesla"
    )
    assert wanvideo.resolve_wan_attention_type("original", "fa") == "original"
    assert wanvideo.resolve_wan_attention_type("sla", "fa") == "sla"


def test_fsdp_loader_treats_rocm_sla_projection_as_zero_init():
    from sglang.multimodal_gen.runtime.loader.fsdp_load import (
        _resolve_missing_param_init,
    )

    assert (
        _resolve_missing_param_init(
            "blocks.10.attn1.local_attn.proj_l.weight", None
        )
        == "zeros"
    )
    assert (
        _resolve_missing_param_init(
            "blocks.10.attn1.local_attn.proj_l.bias", None
        )
        == "zeros"
    )


def test_wan_sla_topk_uses_attention_backend_config():
    from sglang.multimodal_gen.runtime.models.dits import wanvideo

    assert (
        wanvideo.resolve_wan_sla_topk(
            0.1, "sla", {"sla_topk": 0.3}
        )
        == 0.3
    )
    assert (
        wanvideo.resolve_wan_sla_topk(
            0.1, "sla", {"topk": "0.5"}
        )
        == 0.5
    )
    assert (
        wanvideo.resolve_wan_sla_topk(
            0.1, "original", {"sla_topk": 0.3}
        )
        == 0.1
    )

    with pytest.raises(ValueError, match="sla_topk"):
        wanvideo.resolve_wan_sla_topk(0.1, "sla", {"sla_topk": 0})


def test_wan_sla_local_blocks_uses_attention_backend_config():
    from sglang.multimodal_gen.runtime.models.dits import wanvideo

    assert (
        wanvideo.resolve_wan_sla_local_blocks(0, "sla", {"sla_local_blocks": 3})
        == 3
    )
    assert (
        wanvideo.resolve_wan_sla_local_blocks(0, "sla", {"local_blocks": "2"})
        == 2
    )
    assert (
        wanvideo.resolve_wan_sla_local_blocks(0, "original", {"sla_local_blocks": 3})
        == 0
    )

    with pytest.raises(ValueError, match="sla_local_blocks"):
        wanvideo.resolve_wan_sla_local_blocks(0, "sla", {"sla_local_blocks": -1})


def test_wan_sla_skip_linear_uses_attention_backend_config():
    from sglang.multimodal_gen.runtime.models.dits import wanvideo

    assert (
        wanvideo.resolve_wan_sla_skip_linear(True, "sla", {"sla_skip_linear": False})
        is False
    )
    assert (
        wanvideo.resolve_wan_sla_skip_linear(True, "sla", {"skip_linear_branch": "off"})
        is False
    )
    assert (
        wanvideo.resolve_wan_sla_skip_linear(False, "sla", {"skip_linear": "yes"})
        is True
    )
    assert (
        wanvideo.resolve_wan_sla_skip_linear(True, "original", {"sla_skip_linear": False})
        is True
    )

    with pytest.raises(ValueError, match="sla_skip_linear"):
        wanvideo.resolve_wan_sla_skip_linear(True, "sla", {"sla_skip_linear": "maybe"})
