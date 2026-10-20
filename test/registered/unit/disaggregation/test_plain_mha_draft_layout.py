"""Pure producer/consumer contracts for plain MHA and mapped hybrid draft KV."""

import ast
import logging
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "python/sglang").is_dir()
)
SRT = ROOT / "python/sglang/srt/disaggregation"
scope = {"logger": logging.getLogger(__name__), "deque": deque}


def load(path, names):
    tree = ast.parse(path.read_text())
    nodes = [
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    for n in nodes:
        n.decorator_list = []
    mod = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(mod), str(path), "exec"), scope)


load(
    SRT / "utils.py",
    {
        "split_kv_infos",
        "normalize_mha_mtp_kv_infos",
        "normalize_plain_mha_draft_args",
        "build_transfer_entry_pairs",
    },
)
load(SRT / "common/conn.py", {"get_mha_kv_ptrs_with_pp"})
load(SRT / "nixl/conn.py", {"_build_transfer_dst_indices"})
normalize = scope["normalize_plain_mha_draft_args"]


def args(main, draft, start=0, mapped=False):
    ptrs = (
        ["K" + str(i) for i in main]
        + ["V" + str(i) for i in main]
        + ["KD" + str(i) for i in draft]
        + ["VD" + str(i) for i in draft]
    )
    return NS(
        kv_data_ptrs=ptrs,
        kv_data_lens=list(ptrs),
        kv_item_lens=list(ptrs),
        kv_data_mem_kinds=list(ptrs),
        num_draft_entries=len(draft) * 2,
        kv_layer_ids=(
            [*main, *main, *[4 + i for i in draft], *[4 + i for i in draft]]
            if mapped
            else []
        ),
        prefill_start_layer=start,
        prefill_end_layer=start + len(main),
    )


def manager(a, total=4):
    m = NS(
        kv_args=a,
        model_config=NS(
            num_hidden_layers=total,
            architectures=["Qwen3ForCausalLM"],
            is_draft_model=False,
        ),
        pp_size=1,
        is_mla_backend=False,
        is_hybrid_mla_backend=False,
    )
    m.get_mha_kv_ptrs_with_pp = lambda src, dst: scope["get_mha_kv_ptrs_with_pp"](
        m, src, dst
    )
    return m


@pytest.mark.parametrize("main,start,total", [([0, 1], 0, 2), ([2, 3], 2, 4)])
def test_plain_mha_draft_and_pp(main, start, total):
    src = args(main, [0], start)
    dst = args(list(range(total)), [0])
    assert normalize(src, is_mla_backend=False, num_hidden_layers=total)
    assert normalize(dst, is_mla_backend=False, num_hidden_layers=total)
    for field in ["kv_data_lens", "kv_item_lens", "kv_data_mem_kinds"]:
        assert getattr(src, field) == src.kv_data_ptrs
    sk, sv, dk, dv, n = manager(src, total).get_mha_kv_ptrs_with_pp(
        src.kv_data_ptrs, dst.kv_data_ptrs
    )
    assert sk == dk == ["K" + str(i) for i in main] + ["KD0"]
    assert sv == dv == ["V" + str(i) for i in main] + ["VD0"]
    assert n == len(main) + 1


def test_equal_pp_partitions():
    src = args([2, 3], [0], 2)
    dst = args([2, 3], [0], 2)
    normalize(src, is_mla_backend=False, num_hidden_layers=4)
    normalize(dst, is_mla_backend=False, num_hidden_layers=4)
    sk, sv, dk, dv, n = manager(src).get_mha_kv_ptrs_with_pp(
        src.kv_data_ptrs, dst.kv_data_ptrs
    )
    assert (sk, sv) == (dk, dv)
    assert n == 3


def test_decode_only_draft_and_nixl_mapping():
    src = args([0, 1], [])
    dst = args([0, 1], [0])
    normalize(dst, is_mla_backend=False, num_hidden_layers=2)
    m = manager(src, 2)
    sk, sv, dk, dv, n = m.get_mha_kv_ptrs_with_pp(src.kv_data_ptrs, dst.kv_data_ptrs)
    assert dk == ["K0", "K1"] and dv == ["V0", "V1"]
    indices = scope["_build_transfer_dst_indices"](
        m, peer_info=NS(dst_kv_layer_ids=[]), n_src=4, n_dst=6
    )
    assert [dst.kv_data_ptrs[i] for i in indices] == src.kv_data_ptrs


def test_mapped_hybrid_and_mla_are_not_reordered():
    hybrid = args([1, 3], [0], mapped=True)
    before = list(hybrid.kv_data_ptrs)
    assert not normalize(hybrid, is_mla_backend=False, num_hidden_layers=4)
    assert hybrid.kv_data_ptrs == before
    dst = args([0, 1, 2, 3], [0], mapped=True)
    pairs = scope["build_transfer_entry_pairs"](
        hybrid.kv_layer_ids, dst.kv_layer_ids, len(before), len(dst.kv_data_ptrs)
    )
    assert [dst.kv_data_ptrs[j] for _, j in pairs] == before
    mla = args([0, 1], [0])
    assert not normalize(mla, is_mla_backend=True, num_hidden_layers=2)


def test_invalid_parallel_metadata_rejected():
    a = args([0, 1], [0])
    a.kv_item_lens = [1]
    with pytest.raises(ValueError):
        normalize(a, is_mla_backend=False, num_hidden_layers=2)
