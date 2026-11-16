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
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
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


def test_sync_dfs_health_policy_applies_attention_group_minimum():
    applied = []
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.linker = SimpleNamespace(
        get_local_dfs_health_policy=lambda: (True, True),
        apply_dfs_health_policy=lambda read, write: applied.append((read, write)),
    )

    def reduce_to_common_policy(policy, op):
        assert op == torch.distributed.ReduceOp.MIN
        policy.copy_(torch.tensor([0, 1], dtype=torch.int))

    cache._all_reduce_attn_groups = reduce_to_common_policy
    cache._sync_dfs_health_policy()

    assert applied == [(False, True)]


def test_sync_dfs_health_policy_applies_pp_group_minimum(monkeypatch):
    applied = []
    pp_group = object()
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.linker = SimpleNamespace(
        get_local_dfs_health_policy=lambda: (True, True),
        apply_dfs_health_policy=lambda read, write: applied.append((read, write)),
    )
    cache._all_reduce_attn_groups = lambda policy, op: None
    cache.pp_size = 2
    cache.pp_group = pp_group

    def reduce_pp_policy(policy, op, group):
        assert op == torch.distributed.ReduceOp.MIN
        assert group is pp_group
        policy.copy_(torch.tensor([1, 0], dtype=torch.int))

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce_pp_policy)
    cache._sync_dfs_health_policy()

    assert applied == [(False, False)]


def test_sync_dfs_health_policy_failure_is_nonfatal_and_throttled(monkeypatch):
    now = [100.0]
    calls = []
    reductions = []
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)

    def fail_policy(read, write):
        calls.append((read, write))
        raise RuntimeError("standalone RPC unavailable")

    cache.linker = SimpleNamespace(
        get_local_dfs_health_policy=lambda: (True, True),
        apply_dfs_health_policy=fail_policy,
    )
    cache._all_reduce_attn_groups = lambda policy, op: reductions.append(op)
    monkeypatch.setattr(
        "sglang.srt.mem_cache.unified_radix_cache.time.monotonic", lambda: now[0]
    )

    cache._sync_dfs_health_policy()
    now[0] += 0.5
    cache._sync_dfs_health_policy()

    assert calls == [(True, True)]
    assert reductions == [
        torch.distributed.ReduceOp.MIN,
        torch.distributed.ReduceOp.MIN,
    ]
    assert cache._dfs_policy_failed_target == (True, True)
    assert cache._dfs_policy_consecutive_failures == 1

    now[0] += 0.5
    cache._sync_dfs_health_policy()
    assert calls == [(True, True), (True, True)]
    assert cache._dfs_policy_consecutive_failures == 2


def test_sync_dfs_health_policy_recovers_and_applies_latest_target(monkeypatch):
    now = [200.0]
    target = [(True, True)]
    calls = []
    fail = [True]
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)

    def apply_policy(read, write):
        calls.append((read, write))
        if fail[0]:
            raise RuntimeError("standalone RPC unavailable")

    cache.linker = SimpleNamespace(
        get_local_dfs_health_policy=lambda: target[0],
        apply_dfs_health_policy=apply_policy,
    )
    cache._all_reduce_attn_groups = lambda policy, op: None
    monkeypatch.setattr(
        "sglang.srt.mem_cache.unified_radix_cache.time.monotonic", lambda: now[0]
    )

    cache._sync_dfs_health_policy()
    target[0] = (False, True)
    fail[0] = False
    # A changed (and more restrictive) target bypasses the stale target's
    # retry delay, so failure recovery does not postpone DFS disablement.
    cache._sync_dfs_health_policy()

    assert calls == [(True, True), (False, True)]
    assert cache._dfs_policy_failed_target is None
    assert cache._dfs_policy_consecutive_failures == 0
    assert cache._dfs_policy_next_retry_at == 0.0
