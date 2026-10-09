"""Exercise HCU MLA metadata lifecycle with CPU tensors and a fake native planner.

This checks graph buffer contracts, not native-kernel numerical accuracy.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parents[4]


def _load_backend():
    path = ROOT / "python/sglang/srt/layers/attention/hcu_mla_backend.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "HCUMLABackend"
    )
    names = {
        "init_cuda_graph_state",
        "_capture_cuda_graph_metadata",
        "_replay_cuda_graph_metadata",
    }
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    for method in methods:
        method.decorator_list = []

    def planner(lens, heads, *args):
        # Different row counts exercise the maximum-capacity graph allocation.
        return torch.full((heads, 2), int(lens.sum()), dtype=torch.int32), torch.arange(
            lens.numel() + 1, dtype=torch.int32
        )

    def metadata(meta, splits, table, lens):
        return SimpleNamespace(
            flashmla_metadata=meta,
            num_splits=splits,
            block_kv_indices=table,
            cache_seqlens=lens,
        )

    ns = dict(
        torch=torch,
        triton=SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
        PAGE_SIZE=64,
        get_bool_env_var=lambda *a, **k: True,
        get_mla_metadata=planner,
        hcu_create_flashmla_kv_indices=lambda **k: None,
        VllmMLADecodeMetadata=metadata,
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *methods,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    backend = type("MetadataBackend", (), {name: ns[name] for name in names})()
    backend.num_q_heads = 2
    backend.num_draft_tokens = 3
    backend.metadata_interface_arguments = 4
    backend.req_to_token = torch.zeros((4, 512), dtype=torch.int32)
    backend.init_cuda_graph_state(4, 4, torch.ones((4, 8), dtype=torch.int32))
    return backend


@pytest.mark.parametrize("mode,extra", [("decode", 0), ("verify", 3), ("draft", 0)])
def test_graph_lengths_and_tables_keep_replay_storage(mode, extra):
    backend = _load_backend()
    fm = SimpleNamespace(
        is_decode_or_idle=lambda: mode == "decode",
        is_target_verify=lambda: mode == "verify",
        is_draft_extend_v2=lambda: mode == "draft",
    )
    lengths = torch.tensor([65, 80], dtype=torch.int64)
    backend._capture_cuda_graph_metadata(
        2, torch.tensor([0, 1]), lengths, None, fm, None
    )
    meta = backend.forward_metadata
    lens_ptr = meta.cache_seqlens.data_ptr()
    table_ptr = meta.block_kv_indices.data_ptr()
    assert meta.block_kv_indices.is_contiguous()
    torch.testing.assert_close(meta.cache_seqlens, (lengths + extra).int())
    newer = lengths + 7
    backend._replay_cuda_graph_metadata(
        2, torch.tensor([0, 1]), newer, int(newer.sum()), None, fm, None, newer
    )
    assert meta.cache_seqlens.data_ptr() == lens_ptr
    assert meta.block_kv_indices.data_ptr() == table_ptr
    assert meta.block_kv_indices.is_contiguous()
    torch.testing.assert_close(meta.cache_seqlens, (newer + extra).int())
    assert backend.cuda_graph_mla_metadata.shape[0] == 6


@pytest.mark.parametrize(
    "scale", [torch.tensor(0.25), torch.tensor([0.25]), torch.tensor([[0.25, 0.5]])]
)
def test_fp8_descale_expansion_preserves_calibration(scale):
    path = ROOT / "python/sglang/srt/layers/attention/flashattention_interface.py"
    tree = ast.parse(path.read_text())
    helper = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_as_descale"
    )
    ns = dict(torch=torch, descale_shape=(3, 2), k=torch.empty((1, 2, 4)))
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[])),
            str(path),
            "exec",
        ),
        ns,
    )
    out = ns["_as_descale"](scale)
    torch.testing.assert_close(out, scale.expand(3, 2))
    assert out.is_contiguous()
