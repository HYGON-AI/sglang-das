"""Gate rows must be truncated before the normal weight loader validates shape."""

import ast
from pathlib import Path
import pytest
import torch

SOURCE = (
    next(p for p in Path(__file__).resolve().parents if (p / "python/sglang").is_dir())
    / "python/sglang/srt/models/kimi_k3.py"
)
tree = ast.parse(SOURCE.read_text())
node = next(
    n
    for n in tree.body
    if isinstance(n, ast.FunctionDef) and n.name == "_truncate_moe_gate_for_config"
)
scope = {}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), scope)
truncate = scope[node.name]


@pytest.mark.parametrize(
    "suffix,tail", [("weight", (4,)), ("e_score_correction_bias", ())]
)
def test_reduced_and_equal_experts(suffix, tail):
    name = "model.layers.0.mlp.gate." + suffix
    loaded = torch.arange(8 * (4 if tail else 1)).reshape(8, *tail).float()
    param = torch.empty(3, *tail)
    result = truncate(name, param, loaded, 3)
    param.copy_(result)
    torch.testing.assert_close(param, loaded[:3], rtol=0, atol=0)
    assert truncate(name, loaded, loaded, 8) is loaded
    with pytest.raises(ValueError):
        truncate(name, torch.empty(9, *tail), loaded, 9)


def test_other_weights_and_invalid_shape():
    loaded = torch.ones(8, 4)
    assert (
        truncate("model.layers.0.mlp.other.weight", torch.empty(3, 4), loaded, 3)
        is loaded
    )
    assert (
        truncate("model.layers.0.mlp.gate.weight", torch.empty(3, 4), loaded, None)
        is loaded
    )
    with pytest.raises(ValueError):
        truncate("model.layers.0.mlp.gate.weight", torch.empty(3, 5), loaded, 3)


def test_loader_applies_truncation_before_loading():
    text = SOURCE.read_text()
    start = text.index("loaded_weight = _truncate_moe_gate_for_config(")
    assert text.index("weight_loader(param, loaded_weight, **kwargs)", start) > start
