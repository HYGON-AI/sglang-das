"""LightOp paged FP4 decode indexer on HCU (lightop_indexer.py).

Run from the repository root on a DCU with a LightOp build exposing
PAGED_MQA_LOGITS_FP4_ABI >= 2:
    python -m pytest -q test/manual/dsv4/test_lightop_fp4_indexer.py

The pool and the queries are produced by SGLang's own writers
(store_fp4_index_k_cache, quantize_fp4_indexer_tensor), so these tests pin the
byte layout shared by SGLang and LightOp, not just LightOp in isolation.
"""

import pytest
import sgl_kernel  # noqa: F401  registers torch.ops.sgl_kernel, used by topk_transform_paged on HIP
import torch

from sglang.kernels.ops.attention.dsv4 import lightop_indexer
from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    quantize_fp4_indexer_tensor,
    store_fp4_index_k_cache,
)
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_paged

PAGE = 64  # test fixture: the synthetic pool below uses 64 positions per page
BLOCK = lightop_indexer.DEFAULT_CANDIDATE_BLOCK_SIZE
CODES = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


@pytest.fixture(scope="module", autouse=True)
def require_lightop():
    if not torch.cuda.is_available():
        pytest.skip("requires a DCU")
    if not lightop_indexer.lightop_paged_indexer_available():
        pytest.skip("LightOp paged FP4 indexer unavailable")


def dequantize(payload, scale_bytes):
    lut = torch.tensor(CODES, device=payload.device, dtype=torch.float64)
    codes = torch.stack((payload & 15, payload >> 4), dim=-1).flatten(-2).long()
    return lut[codes] * torch.exp2(scale_bytes.double() - 127).repeat_interleave(32, -1)


def make_case(batch, heads, lens, seed=0):
    """A pool written by SGLang, one shuffled page table per row, quantized Q."""
    torch.manual_seed(seed)
    device = "cuda"
    pages_per_row = max((n + PAGE - 1) // PAGE for n in lens)
    pool_pages = batch * pages_per_row + 3
    page_table = (
        torch.randperm(pool_pages, device=device)[: batch * pages_per_row]
        .reshape(batch, pages_per_row)
        .int()
    )
    cache = torch.zeros(pool_pages, PAGE * 68, device=device, dtype=torch.uint8)
    kv = torch.randn(pool_pages * PAGE, 128, device=device, dtype=torch.bfloat16)
    kv *= torch.exp2(torch.randint(-4, 5, (pool_pages * PAGE, 1), device=device))
    store_fp4_index_k_cache(
        kv, cache, torch.arange(pool_pages * PAGE, device=device), page_size=PAGE, rne=True
    )
    q = torch.randn(batch * heads, 128, device=device, dtype=torch.bfloat16)
    q_fp4, q_sf = quantize_fp4_indexer_tensor(q, rne=True)
    return dict(
        q=(q_fp4.view(batch, 1, heads, 64), q_sf.view(batch, 1, heads)),
        k_cache=cache.view(pool_pages, PAGE, 1, 68),
        weights=torch.randn(batch, heads, device=device),
        lens=torch.tensor(lens, device=device, dtype=torch.int32),
        page_table=page_table,
    )


def reference_logits(case, lens, table, table_block_size, width):
    """FP64 logits of column t = slot table[b, t // TB] * TB + t % TB."""
    q_fp4, q_sf = case["q"]
    batch, _, heads, _ = q_fp4.shape
    qd = dequantize(
        q_fp4.view(torch.uint8).reshape(batch, heads, 64),
        q_sf.view(torch.uint8).reshape(batch, heads, 4),
    )
    pool = case["k_cache"].reshape(case["k_cache"].shape[0], PAGE * 68)
    payload = pool[:, : PAGE * 64].unflatten(1, (PAGE, 64))
    scales = pool[:, PAGE * 64 :].unflatten(1, (PAGE, 4))
    out = torch.full((batch, width), float("nan"), device=pool.device, dtype=torch.float64)
    for b in range(batch):
        n = int(lens[b])
        t = torch.arange(n, device=pool.device)
        slot = table[b, t // table_block_size].long() * table_block_size + t % table_block_size
        k = dequantize(payload[slot // PAGE, slot % PAGE], scales[slot // PAGE, slot % PAGE])
        w = case["weights"][b].double()
        out[b, :n] = ((qd[b] @ k.T).relu() * w[:, None]).sum(0)
    return out


def assert_prefix_close(got, expected, lens):
    for b, n in enumerate(lens.tolist()):
        torch.testing.assert_close(
            got[b, :n].double(), expected[b, :n], rtol=2e-5, atol=1e-4
        )


@pytest.mark.parametrize("with_schedule", [False, True])
@pytest.mark.parametrize("heads", [32, 64])
def test_dense_logits_match_reference(with_schedule, heads):
    lens = [0, 1, 777, 8192]
    case = make_case(len(lens), heads, lens, seed=heads)
    schedule = (
        lightop_indexer.get_paged_mqa_logits_schedule(case["lens"]) if with_schedule else None
    )
    width = max(lens)
    got = lightop_indexer.paged_mqa_logits_fp4(
        case["q"], case["k_cache"], case["weights"], case["lens"],
        case["page_table"], schedule, width, table_block_size=PAGE,
    )
    expected = reference_logits(case, case["lens"], case["page_table"], PAGE, width)
    assert_prefix_close(got, expected, case["lens"])


@pytest.mark.parametrize("requests", [4, 8])
@pytest.mark.parametrize("draft", [2, 3, 6])
def test_verify_rows_paired_bit_identical(draft, requests):
    """Non-ragged verify: a request's rows share one page-table row and see
    lengths L..L+draft-1; pairing them (8 requests, past LightOp's gate) or
    falling back below the gate (4) must leave every logit bit unchanged."""
    if not lightop_indexer.lightop_paired_rows_available():
        pytest.skip("LightOp build predates rows_per_request")
    heads = 32  # DSV4.1 index_n_heads; LightOp pairs only H <= 32
    starts = [0, 1, 700, 8192 - draft + 1, 1500, 3000, 4500, 6000][:requests]
    lens = [s + i for s in starts for i in range(draft)]
    case = make_case(len(lens), heads, lens, seed=draft * 10 + requests)
    table = case["page_table"].view(len(starts), draft, -1)[:, :1].expand(-1, draft, -1)
    table = table.reshape(len(lens), -1).contiguous()
    width = table.shape[1] * PAGE

    def logits(rows_per_request):
        schedule = lightop_indexer.get_paged_mqa_logits_schedule(
            case["lens"], rows_per_request=rows_per_request
        )
        return lightop_indexer.paged_mqa_logits_fp4(
            case["q"], case["k_cache"], case["weights"], case["lens"], table,
            schedule, width, table_block_size=PAGE, rows_per_request=rows_per_request,
        )

    want, got = logits(1), logits(draft)
    for b, n in enumerate(lens):
        assert torch.equal(got[b, :n].view(torch.int32), want[b, :n].view(torch.int32)), b
    expected = reference_logits(case, case["lens"], table, PAGE, width)
    assert_prefix_close(got, expected, case["lens"])


def reference_candidate_table(logits, lens, page_table, topk_blocks, block):
    """Level one: top blocks by block max, newest kept, ascending, as slots / 8."""
    per_page = PAGE // block
    phys, valid = [], []
    for b, length in enumerate(lens.tolist()):
        n = (length + block - 1) // block
        if n <= topk_blocks:
            kept = torch.arange(n)
        else:
            keys = logits[b, : (n - 1) * block].cpu().reshape(n - 1, block).amax(1)
            keys = torch.cat((keys, torch.tensor([float("inf")])))
            kept = torch.sort(keys, descending=True, stable=True).indices[:topk_blocks].sort().values
        pt = page_table[b].cpu().long()
        phys.append(pt[kept // per_page] * per_page + kept % per_page)
        valid.append(block * (len(kept) - 1) + (length - 1) % block + 1 if length else 0)
    return phys, valid


def test_two_level_candidate_path():
    """publish (dense logits -> block table) then select (sparse logits -> top-k)."""
    topk_blocks, topk = 256, 512
    lens = [300, 2048, 5000, 30000]
    case = make_case(len(lens), 64, lens, seed=7)
    width = max(lens)
    dense = lightop_indexer.paged_mqa_logits_fp4(
        case["q"], case["k_cache"], case["weights"], case["lens"],
        case["page_table"], lightop_indexer.get_paged_mqa_logits_schedule(case["lens"]), width,
        table_block_size=PAGE,
    )
    phys, valid = lightop_indexer.candidate_block_table(
        dense, case["lens"], case["page_table"], PAGE, topk_blocks, BLOCK
    )
    ref_phys, ref_valid = reference_candidate_table(
        dense, case["lens"], case["page_table"], topk_blocks, BLOCK
    )
    assert valid.tolist() == ref_valid
    for b, blocks in enumerate(ref_phys):
        assert phys[b, : len(blocks)].cpu().tolist() == blocks.tolist()

    sparse_width = topk_blocks * BLOCK
    sparse = lightop_indexer.paged_mqa_logits_fp4(
        case["q"], case["k_cache"], case["weights"], valid, phys,
        lightop_indexer.get_paged_mqa_logits_schedule(valid), sparse_width,
        table_block_size=BLOCK,
    )
    expected = reference_logits(case, valid, phys, BLOCK, sparse_width)
    assert_prefix_close(sparse, expected, valid)

    # The second level returns pool slots; every one must be a kept position.
    page_indices = torch.empty(len(lens), topk, device="cuda", dtype=torch.int32)
    topk_transform_paged(sparse, valid, phys, page_indices, BLOCK, None)
    for b, n in enumerate(valid.tolist()):
        t = torch.arange(n, device="cuda")
        allowed = set((phys[b, t // BLOCK].long() * BLOCK + t % BLOCK).tolist())
        picked = [s for s in page_indices[b].tolist() if s >= 0]
        assert len(picked) == min(n, topk)
        assert set(picked) <= allowed


def test_graph_replay_follows_live_inputs():
    lens = [512, 4096, 9000]
    case = make_case(len(lens), 64, lens, seed=3)
    width = 16384
    schedule = lightop_indexer.get_paged_mqa_logits_schedule(case["lens"])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            lightop_indexer.paged_mqa_logits_fp4(
                case["q"], case["k_cache"], case["weights"], case["lens"],
                case["page_table"], schedule, width, table_block_size=PAGE,
            )
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        got = lightop_indexer.paged_mqa_logits_fp4(
            case["q"], case["k_cache"], case["weights"], case["lens"],
            case["page_table"], schedule, width, table_block_size=PAGE,
        )
    for seed, new_lens in ((11, [9000, 1, 700]), (13, [100, 8192, 0])):
        new = make_case(len(lens), 64, [9000] * 3, seed=seed)
        case["q"][0].copy_(new["q"][0])
        case["q"][1].copy_(new["q"][1])
        case["weights"].copy_(new["weights"])
        case["k_cache"][: new["k_cache"].shape[0]].copy_(new["k_cache"][: case["k_cache"].shape[0]])
        case["lens"].copy_(torch.tensor(new_lens, device="cuda", dtype=torch.int32))
        # The schedule is a function of lens; metadata refreshes it in place.
        schedule.copy_(lightop_indexer.get_paged_mqa_logits_schedule(case["lens"]))
        graph.replay()
        torch.cuda.synchronize()
        expected = reference_logits(case, case["lens"], case["page_table"], PAGE, width)
        assert_prefix_close(got, expected, case["lens"])
