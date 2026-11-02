"""Keep main's CPU padding fallback while HCU rejects unsupported layouts."""

import ast
from pathlib import Path
from typing import Optional

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def padding_functions(hcu):
    path = Path(__file__).resolve().parents[5] / "python/sglang/srt/layers/moe/topk.py"
    names = {"_mask_topk_ids_padded_region", "_zero_topk_weights_padded_region"}
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == 2
    namespace = dict(torch=torch, Optional=Optional, _is_cuda=False, _is_npu=False,
                     _is_hcu=hcu, _can_fuse_padded_region=lambda _: False)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return [namespace[name] for name in sorted(names)]


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("hcu", [False, True])
def test_unsupported_device_kernel_layout_keeps_platform_contract(strided, hcu):
    mask, zero = padding_functions(hcu)
    ids = torch.arange(24).reshape(4, 6)
    weights = torch.ones(4, 6)
    if strided:
        ids, weights = ids[:, ::2], weights[:, ::2]
    original = ids.clone()
    count = torch.tensor(2, dtype=torch.int32)
    if hcu:
        with pytest.raises(RuntimeError, match="Unsupported top-k id layout"):
            mask(ids, count)
        with pytest.raises(RuntimeError, match="Unsupported top-k weight layout"):
            zero(weights, count)
        torch.testing.assert_close(ids, original)
        assert torch.all(weights == 1)
    else:
        mask(ids, count)
        zero(weights, count)
        torch.testing.assert_close(ids[:2], original[:2])
        assert torch.all(ids[2:] == -1)
        assert torch.all(weights[:2] == 1)
        assert torch.all(weights[2:] == 0)
