from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("token_shard,time_shard", [(True, False), (False, True)])
def test_minimal_a2a_forward_uses_distributed_exchange(
    monkeypatch, token_shard, time_shard
):
    from sglang.multimodal_gen.runtime.layers.attention import turbo_layer
    from sglang.multimodal_gen.runtime.layers.attention.backends.rocm_sparse_linear_attn import (
        RocmSparseLinearAttentionBackend,
    )

    group = object()
    monkeypatch.setattr(
        turbo_layer,
        "get_attn_backend",
        lambda *args, **kwargs: RocmSparseLinearAttentionBackend,
    )
    monkeypatch.setattr(turbo_layer.dist, "is_available", lambda: True)
    monkeypatch.setattr(turbo_layer.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        turbo_layer,
        "get_sp_group",
        lambda: SimpleNamespace(
            ulysses_group=group, ulysses_world_size=4, ring_world_size=1
        ),
    )
    monkeypatch.setattr(turbo_layer.dist, "get_world_size", lambda pg: 4)
    batch = SimpleNamespace(
        enable_sequence_shard=token_shard,
        did_sp_shard_latents=time_shard,
    )
    monkeypatch.setattr(
        turbo_layer,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata=None,
            forward_batch=batch,
        ),
    )
    calls = []

    def qkv_exchange(pg, q, k, v, size, stream, direction):
        calls.append(("qkv", pg, size, direction))
        return q, k, v

    def output_exchange(pg, output, direction):
        calls.append(("output", pg, direction))
        return output

    monkeypatch.setattr(turbo_layer._SeqAllToAllQKV, "apply", qkv_exchange)
    monkeypatch.setattr(turbo_layer._SeqAllToAll, "apply", output_exchange)
    op = turbo_layer.MinimalA2AAttnOp(
        num_heads=4,
        head_size=8,
        attention_type="sla",
        topk=0.25,
    )
    monkeypatch.setattr(op.local_attn, "forward", lambda q, k, v, meta: q + k + v)
    q = torch.randn(1, 3, 4, 8)
    result = op(q, q, q)
    assert op.pg is group
    assert calls == [("qkv", group, 4, True), ("output", group, False)]
    torch.testing.assert_close(result, (3 * q).flatten(2))
    assert result.device.type == "cpu"

    # A later replicated request must discard the previously bound SP group.
    batch.enable_sequence_shard = False
    batch.did_sp_shard_latents = False
    torch.testing.assert_close(op(q, q, q), (3 * q).flatten(2))
    assert op.pg is None and op.stream is None
    assert len(calls) == 2

    batch.did_sp_shard_latents = True
    torch.testing.assert_close(op(q, q, q), (3 * q).flatten(2))
    assert op.pg is group and len(calls) == 4

    # Caller overrides take precedence over inferred batch state.
    op(q, q, q, sequence_shard_enabled=False)
    assert op.pg is None and len(calls) == 4
    op(q, q, q, sequence_shard_enabled=True)
    assert op.pg is group and len(calls) == 6


@pytest.mark.parametrize("backend", ["sla_attn", "sage_sla_attn"])
def test_wan_sparse_cross_attention_preserves_automatic_dense_fallback(
    monkeypatch, backend
):
    from sglang.multimodal_gen.runtime.models.dits import wanvideo

    calls = []
    monkeypatch.setattr(wanvideo, "get_tp_world_size", lambda: 1)
    monkeypatch.setattr(
        wanvideo, "ColumnParallelLinear", lambda *a, **k: torch.nn.Identity()
    )
    monkeypatch.setattr(
        wanvideo, "RowParallelLinear", lambda *a, **k: torch.nn.Identity()
    )
    monkeypatch.setattr(
        wanvideo,
        "get_global_server_args",
        lambda: SimpleNamespace(attention_backend=backend),
    )

    def construct_attention(**kwargs):
        calls.append(kwargs)
        return torch.nn.Identity()

    monkeypatch.setattr(wanvideo, "USPAttention", construct_attention)
    wanvideo.WanT2VCrossAttention(dim=256, num_heads=2)
    assert calls[0].get("default_attention_backend") is None
    assert calls[0]["skip_sequence_parallel"] is True
    assert calls[0]["is_cross_attention"] is True


@pytest.mark.parametrize(
    "token_shard,time_shard,expected_local_tokens,expected_exchange",
    [(True, False, 1, True), (False, True, 4, True), (False, False, 4, False)],
)
def test_wan_forward_uses_request_layout_for_sla(
    monkeypatch, token_shard, time_shard, expected_local_tokens, expected_exchange
):
    from sglang.multimodal_gen.runtime.layers.attention import turbo_layer
    from sglang.multimodal_gen.runtime.layers.attention.backends.rocm_sparse_linear_attn import (
        RocmSparseLinearAttentionBackend,
    )
    from sglang.multimodal_gen.runtime.models.dits import wanvideo

    batch = SimpleNamespace(
        enable_sequence_shard=token_shard,
        did_sp_shard_latents=time_shard,
        enable_teacache=False,
        enable_spectrum=False,
    )
    monkeypatch.setattr(
        wanvideo, "get_forward_context", lambda: SimpleNamespace(forward_batch=batch)
    )
    monkeypatch.setattr(
        wanvideo, "get_sp_group", lambda: SimpleNamespace(rank_in_group=0)
    )
    monkeypatch.setattr(
        wanvideo, "sequence_model_parallel_all_gather", lambda x, dim: x.repeat(1, 4, 1)
    )
    group = object()
    monkeypatch.setattr(
        turbo_layer,
        "get_attn_backend",
        lambda *args, **kwargs: RocmSparseLinearAttentionBackend,
    )
    monkeypatch.setattr(turbo_layer.dist, "is_available", lambda: True)
    monkeypatch.setattr(turbo_layer.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(turbo_layer.dist, "get_world_size", lambda pg: 4)
    monkeypatch.setattr(
        turbo_layer,
        "get_sp_group",
        lambda: SimpleNamespace(
            ulysses_group=group, ulysses_world_size=4, ring_world_size=1
        ),
    )
    monkeypatch.setattr(
        turbo_layer,
        "get_forward_context",
        lambda: SimpleNamespace(forward_batch=batch, attn_metadata=None),
    )
    exchanges = []

    def exchange_qkv(pg, q, k, v, *args):
        exchanges.append("qkv")
        return q, k, v

    def exchange_output(pg, output, *args):
        exchanges.append("output")
        return output

    monkeypatch.setattr(turbo_layer._SeqAllToAllQKV, "apply", exchange_qkv)
    monkeypatch.setattr(turbo_layer._SeqAllToAll, "apply", exchange_output)
    attention = turbo_layer.MinimalA2AAttnOp(
        num_heads=4, head_size=2, attention_type="sla", topk=0.25
    )
    monkeypatch.setattr(attention.local_attn, "forward", lambda q, k, v, meta: q)
    captured = []

    class Block(torch.nn.Module):
        def forward(self, x, *args, **kwargs):
            assert "sequence_shard_enabled" not in kwargs
            q = x.unflatten(-1, (4, 2))
            output = attention(q, q, q)
            captured.append((x.shape[1], attention.pg is group))
            return output

    class Condition(torch.nn.Module):
        def forward(self, timestep, text, image, **kwargs):
            return torch.zeros(1, 8), torch.zeros(1, 48), text, image

    class Norm(torch.nn.Module):
        def forward(self, x, *args):
            return x

    class Projection(torch.nn.Module):
        def forward(self, x):
            return x, None

    model = wanvideo.WanTransformer3DModel.__new__(wanvideo.WanTransformer3DModel)
    torch.nn.Module.__init__(model)
    model.sp_size = 4
    model.patch_size = (1, 1, 1)
    model.cnt = 0
    model.patch_embedding = torch.nn.Identity()
    model.condition_embedder = Condition()
    model.blocks = torch.nn.ModuleList([Block()])
    model.norm_out = Norm()
    model.proj_out = Projection()
    model.scale_shift_table = torch.zeros(1, 2, 8)
    model.begin_spectrum_step = lambda: True
    model.should_skip_forward_for_cached_states = lambda **kwargs: False

    def rope_for_tokens(n):
        return torch.ones(n, 4), torch.zeros(n, 4)

    # Position values are deliberately stubbed: this test checks the actual
    # model input slicing and attention contract, not the deferred RoPE issue.
    model.rotary_emb = SimpleNamespace(
        forward_3d_sequence_shard=lambda local_len, rank, stride, width, device: (
            rope_for_tokens(local_len)
        ),
        forward_from_grid=lambda grid, **kwargs: rope_for_tokens(grid[0] // 4),
        forward_uncached=lambda positions: rope_for_tokens(positions.shape[0]),
    )
    x = torch.randn(1, 8, 4, 1, 1)
    out = model(x, torch.zeros(1, 2, 8), torch.tensor([1]))
    assert out.shape == x.shape
    assert captured == [(expected_local_tokens, expected_exchange)]
    assert exchanges == (["qkv", "output"] if expected_exchange else [])
