"""LightOp opt-in dispatch and eager/graph tests for the FP4 decode indexer.

Run from the repository root:
    python -m pytest -q test/manual/dsv4/test_lightop_fp4_indexer.py

GPU cases require a CUDA/HIP device and a LightOp wheel containing
paged_mqa_logits_fp4. CPU dispatch tests mock extension discovery only.
The adapter is loaded directly to avoid importing unrelated attention backends.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("triton")

ROOT = Path(__file__).resolve().parents[3]
FLAG = "SGLANG_USE_LIGHTOP_PAGED_MQA_LOGITS_FP4"


def _load_module(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def adapter(monkeypatch):
    # Use the actual flag registry from this checkout, even when an older
    # SGLang package is installed in the kernel-development environment.
    _load_module(
        "sglang.srt.environ", ROOT / "python/sglang/srt/environ.py", monkeypatch
    )
    module = _load_module(
        "_sglang_lightop_fp4_indexer_test",
        ROOT / "python/sglang/kernels/ops/attention/dsv4/fp4_indexer.py",
        monkeypatch,
    )
    monkeypatch.delenv(FLAG, raising=False)
    return module


@pytest.fixture
def discovery(adapter, monkeypatch):
    native = Mock()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setitem(
        sys.modules,
        "lightop",
        SimpleNamespace(op=SimpleNamespace(paged_mqa_logits_fp4=native)),
    )
    return native


@pytest.mark.parametrize("value", [None, "0", "false"])
def test_disabled_bypasses_cached_native(adapter, discovery, monkeypatch, value):
    native = discovery
    adapter._lightop_fp4_op = native
    adapter._lightop_fp4_op_resolved = True
    if value is not None:
        monkeypatch.setenv(FLAG, value)
    assert adapter._get_lightop_fp4_op() is None
    native.assert_not_called()


def test_enabled_discovers_once_per_process(adapter, discovery, monkeypatch):
    native = discovery
    monkeypatch.setenv(FLAG, "1")
    assert adapter._get_lightop_fp4_op() is native
    assert adapter._get_lightop_fp4_op() is native
    monkeypatch.setenv(FLAG, "0")
    assert adapter._get_lightop_fp4_op() is None
    monkeypatch.setenv(FLAG, "true")
    assert adapter._get_lightop_fp4_op() is native


@pytest.mark.parametrize("reason", ["missing_package", "missing_symbol"])
def test_unavailable_native_falls_back(adapter, discovery, monkeypatch, reason):
    _ = discovery
    monkeypatch.setenv(FLAG, "1")
    if reason == "missing_package":
        monkeypatch.setitem(sys.modules, "lightop", None)
    else:
        monkeypatch.setitem(
            sys.modules, "lightop", SimpleNamespace(op=SimpleNamespace())
        )
    assert adapter._get_lightop_fp4_op() is None
    assert adapter._get_lightop_fp4_op() is None


def test_capture_does_not_load_extension_or_poison_cache(
    adapter, discovery, monkeypatch
):
    native = discovery
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert adapter._get_lightop_fp4_op() is None
    assert not adapter._lightop_fp4_op_resolved
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert adapter._get_lightop_fp4_op() is native
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert adapter._get_lightop_fp4_op() is native


@pytest.mark.parametrize("batch,length", [(0, 32), (1, 0)])
def test_empty_input_never_launches(adapter, monkeypatch, batch, length):
    resolve = Mock(side_effect=AssertionError("empty input must not resolve kernels"))
    monkeypatch.setattr(adapter, "_get_lightop_fp4_op", resolve)
    result = adapter.fp4_index_logits_decode(
        torch.empty(batch, 32, 128, dtype=torch.bfloat16),
        torch.empty(batch, 32),
        torch.empty(batch, length, dtype=torch.int64),
        torch.zeros(batch, dtype=torch.int64),
        torch.zeros(1, 64 * 68, dtype=torch.uint8),
        64,
    )
    assert result.shape == (batch, length)
    assert result.dtype == torch.float32
    resolve.assert_not_called()


@pytest.fixture
def native(adapter, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires a DCU")
    monkeypatch.setenv(FLAG, "1")
    operator = adapter._get_lightop_fp4_op()
    if operator is None:
        pytest.skip("install a LightOp wheel with paged_mqa_logits_fp4")
    return operator


def _case(heads=32, length=65, span=2):
    torch.manual_seed(29)
    device = "cuda:0"
    page_size, pages, batch = 64, 4, 2
    # All Q values are exactly representable as FP4, so large H32 cases also
    # exercise LightOp's FP8 path. The public input remains BF16.
    grid = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=device)
    q = grid[torch.randint(0, 8, (batch, heads, 128), device=device)].bfloat16()
    weights = torch.rand(batch, heads, device=device).bfloat16()
    table = torch.randint(
        0, 256, (pages, page_size * 68), dtype=torch.uint8, device=device
    )
    scales = torch.tensor([127, 127 + span, 127, 127], dtype=torch.uint8, device=device)
    table[:, page_size * 64 :] = scales.repeat(pages, page_size)
    slots = torch.randint(0, pages * page_size, (batch, length), device=device)
    lens = torch.tensor([max(0, length - 5), length], device=device)
    slots[0, lens[0] :] = -123456  # masked slots must never be dereferenced
    return q, weights, slots, lens, table, page_size


def _reference(case, bf16_rounding=False):
    q, weights, slots, lens, table, page_size = case
    grid = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=q.device)
    payload = table[:, : page_size * 64].reshape(-1, 64)
    code = torch.stack((payload & 15, payload >> 4), dim=-1).reshape(-1, 128).long()
    k = grid[code & 7] * torch.where(code < 8, 1, -1)
    scale = torch.exp2(table[:, page_size * 64 :].reshape(-1, 4).float() - 127)
    k = (k.reshape(-1, 4, 32) * scale[:, :, None]).reshape(-1, 128)
    k = k[slots.clamp_min(0)]
    dot = torch.einsum("bhd,bld->bhl", q.float(), k)
    if bf16_rounding:
        dot = dot.bfloat16().float()
    weighted = dot.relu() * weights.float()[:, :, None]
    if bf16_rounding:
        weighted = weighted.bfloat16().float()
    result = weighted.sum(1)
    if bf16_rounding:
        result = result.bfloat16().float()
    return result.masked_fill(
        torch.arange(slots.shape[1], device=q.device)[None] >= lens[:, None], -torch.inf
    )


@pytest.mark.parametrize(
    "heads,length,span",
    [
        (8, 65, 2),
        (16, 65, 2),
        (32, 31, 2),
        (32, 4096, 2),
        (32, 24576, 2),
        (32, 24576, 11),
        (32, 24576, 12),
        (64, 65, 2),
    ],
)
def test_native_dispatch_and_fp32_reference(
    adapter, native, monkeypatch, heads, length, span
):
    case = _case(heads, length, span)
    spy = Mock(wraps=native)
    adapter._lightop_fp4_op = spy
    adapter._lightop_fp4_op_resolved = True
    got = adapter.fp4_index_logits_decode(*case)
    spy.assert_called_once()
    assert spy.call_args.args[0].dtype == torch.bfloat16
    assert spy.call_args.args[3] is case[3]
    torch.testing.assert_close(got, native(*case), rtol=0, atol=0)
    torch.testing.assert_close(got, _reference(case), rtol=2e-5, atol=1e-4)


@pytest.mark.parametrize(
    "fallback",
    ["disabled", "missing_symbol"],
)
def test_triton_fallback(adapter, native, monkeypatch, fallback):
    case = list(_case())
    if fallback == "disabled":
        monkeypatch.setenv(FLAG, "0")
    elif fallback == "missing_symbol":
        adapter._lightop_fp4_op = None
        adapter._lightop_fp4_op_resolved = False
        monkeypatch.setitem(
            sys.modules, "lightop", SimpleNamespace(op=SimpleNamespace())
        )
    spy = Mock(side_effect=AssertionError("must not call LightOp"))
    if fallback != "missing_symbol":
        adapter._lightop_fp4_op = spy
        adapter._lightop_fp4_op_resolved = True
    got = adapter.fp4_index_logits_decode(*case)
    spy.assert_not_called()
    torch.testing.assert_close(
        got, _reference(case, bf16_rounding=True), rtol=0, atol=0
    )


def test_native_error_is_not_silently_retried(adapter, native):
    adapter._lightop_fp4_op = Mock(side_effect=RuntimeError("native launch failed"))
    adapter._lightop_fp4_op_resolved = True
    with pytest.raises(RuntimeError, match="native launch failed"):
        adapter.fp4_index_logits_decode(*_case())


@pytest.mark.parametrize("enabled", [False, True])
def test_graph_replay_reads_updated_inputs(adapter, native, monkeypatch, enabled):
    monkeypatch.setenv(FLAG, "1" if enabled else "0")
    case = list(_case(length=24576))
    spy = Mock(wraps=native)
    adapter._lightop_fp4_op = spy
    adapter._lightop_fp4_op_resolved = True
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            adapter.fp4_index_logits_decode(*case)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        got = adapter.fp4_index_logits_decode(*case)
    address = got.data_ptr()
    for step in range(2):
        # Q conversion, weights, cache data, slots and visibility must be read
        # afresh on replay, not frozen during capture. All storage stays fixed.
        case[0].mul_(0.5)
        case[1].mul_(0.5)
        case[2].fill_(step + 1)
        case[3].fill_(128 + step)
        case[4][:, : 64 * 64].fill_(0x22 + step)
        graph.replay()
        torch.cuda.synchronize()
        assert got.data_ptr() == address
        expected = _reference(case, bf16_rounding=not enabled)
        torch.testing.assert_close(
            got, expected, rtol=2e-5 if enabled else 0, atol=1e-4 if enabled else 0
        )
    assert spy.call_count == (4 if enabled else 0)
