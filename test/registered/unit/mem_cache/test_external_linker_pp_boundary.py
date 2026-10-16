"""Regression coverage for pipeline-parallel external-cache boundaries."""

from array import array
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import LinkerTransferPhase
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinkerWrapper,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_every_pp_stage_probes_l3_and_agrees_on_extend_boundary():
    """A downstream PP stage must not stop at its shorter local-only hit."""

    canonical_hit_len = 5
    prompt_len = 7
    lookups = []

    class _Component:
        @staticmethod
        def build_external_linker_transfer(_phase, _node, keys):
            assert _phase is LinkerTransferPhase.LOOKUP
            return PoolTransfer(name=PoolName.KV, keys=list(keys))

    def run_match(pp_rank, device_hit_len):
        cache = SimpleNamespace(
            page_size=1,
            pp_size=2,
            pp_rank=pp_rank,
            _components_tuple=(_Component(),),
            _all_reduce_attn_groups=lambda _tensor, _op: None,
        )
        backend = SimpleNamespace(
            lookup=lambda rid, transfers: lookups.append(
                (pp_rank, rid, len(transfers[0].keys))
            )
            or [len(transfers[0].keys)]
        )
        wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
        wrapper.cache = cache
        wrapper.cache_linker = backend
        wrapper.hit_markers = {}
        wrapper.host_prefetch_hits = {}
        wrapper._waiting_queue_prefetch_enabled = False
        wrapper._tail_hashes = lambda _key, _result, local_hit: [
            f"page-{index}" for index in range(local_hit, canonical_hit_len)
        ]

        req = SimpleNamespace(rid=f"req-{pp_rank}", external_cache_hit_length=None)
        root = object()
        result = MatchResult(
            device_indices=torch.arange(device_hit_len, dtype=torch.int64),
            last_device_node=root,
            last_host_node=root,
            best_match_node=root,
        )
        matched = wrapper.match(RadixKey(array("q", range(prompt_len))), req, result)
        return req, matched

    pp0_req, pp0_match = run_match(pp_rank=0, device_hit_len=2)
    pp1_req, pp1_match = run_match(pp_rank=1, device_hit_len=3)

    assert [rank for rank, _rid, _pages in lookups] == [0, 1]
    assert pp0_req.external_cache_hit_length == canonical_hit_len
    assert pp1_req.external_cache_hit_length == canonical_hit_len
    assert len(pp0_match.device_indices) + pp0_match.host_hit_length == canonical_hit_len
    assert len(pp1_match.device_indices) + pp1_match.host_hit_length == canonical_hit_len
    assert prompt_len - canonical_hit_len == 2
