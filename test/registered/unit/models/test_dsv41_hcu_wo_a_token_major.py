"""HCU G8 WO-A token-major output preserves BF16 bits and route boundaries."""

from unittest.mock import patch

import pytest
import torch

from sglang.srt.models import deepseek_v4 as model

pytestmark = pytest.mark.skipif(not model._is_hcu, reason="HCU required")


def operands(rows, groups=8, strided=False):
    torch.manual_seed(20261028 + rows)
    x = torch.randn(
        rows, groups, 8192 if strided else 4096, device="cuda", dtype=torch.bfloat16
    )
    if strided:
        x = x[:, :, ::2]
    w = torch.randn(groups, 1024, 4096, device="cuda", dtype=torch.bfloat16) / 64
    return x, w


@pytest.mark.parametrize("rows", [2, 6, 12, 15, 18, 20, 24, 32])
def test_bitwise_and_graph(rows):
    x, w = operands(rows)
    with patch.object(model, "_use_hcu_wo_a_token_major", True):
        out = model._apply_wo_a_bf16_matmul(
            x, w, is_decode=False, is_target_verify=True, fast_path=True
        )
        assert out.is_contiguous()
        assert torch.equal(out, torch.einsum("tgd,grd->tgr", x, w))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = model._apply_wo_a_bf16_matmul(x, w, is_decode=True, fast_path=True)
        for _ in range(4):
            x.normal_()
            graph.replay()
            assert torch.equal(out, torch.einsum("tgd,grd->tgr", x, w))


@pytest.mark.parametrize(
    "boundary",
    ["off", "prefill", "dsv4", "one_row", "large", "groups", "strided", "fp8"],
)
def test_fallback(boundary):
    x, w = operands(
        1 if boundary == "one_row" else 33 if boundary == "large" else 6,
        groups=4 if boundary == "groups" else 8,
        strided=boundary == "strided",
    )
    expected = torch.einsum("tgd,grd->tgr", x, w)
    with patch.object(
        model, "_use_hcu_wo_a_token_major", boundary != "off"
    ), patch.object(model, "_wo_a_aiter_batched_gemm_enabled", False), patch(
        "torch.einsum", wraps=torch.einsum
    ) as fallback:
        out = model._apply_wo_a_bf16_matmul(
            x,
            w,
            is_decode=False,
            is_target_verify=boundary != "prefill",
            is_prefill=boundary == "prefill",
            fast_path=boundary != "dsv4",
            emit_fp8=boundary == "fp8",
        )
        fallback.assert_called_once()
        assert torch.equal(out, expected)


def test_default_enabled():
    assert model._use_hcu_wo_a_token_major
