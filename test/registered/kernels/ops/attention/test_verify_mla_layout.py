from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.attention import verify_mla
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-c-test-cpu")


def _inputs(k_paged=False, v_paged=False):
    # page_size == head_dim reproduces the shape checks that previously
    # admitted paged HND buffers to the flat-token prefix kernel.
    q = torch.empty((6, 4, 64), device="cpu")
    k = torch.empty((6, 1, 64), device="cpu")
    v = torch.empty_like(k)
    flat_shape = (128, 1, 64)
    paged_shape = (2, 1, 64, 64)
    kb = torch.empty(paged_shape if k_paged else flat_shape, device="cpu")
    vb = torch.empty(paged_shape if v_paged else flat_shape, device="cpu")
    qo = torch.tensor([0, 6], dtype=torch.int32)
    kvp = torch.tensor([0, 2], dtype=torch.int32)
    kvi = torch.tensor([0, 65], dtype=torch.int32)
    return q, k, v, kb, vb, qo, kvp, kvi


def test_flat_kv_layout_remains_supported():
    assert verify_mla.can_handle(*_inputs(), None, True, None, 6)


@pytest.mark.parametrize(
    "k_paged,v_paged", [(True, False), (False, True), (True, True)]
)
def test_paged_kv_layout_falls_back_without_launch(k_paged, v_paged):
    q, k, v, kb, vb, qo, kvp, kvi = _inputs(k_paged, v_paged)
    assert not verify_mla.can_handle(q, k, v, kb, vb, qo, kvp, kvi, None, True, None, 6)
    out = torch.full_like(q, 123.0)
    with patch.object(
        verify_mla, "_get_vmla", side_effect=AssertionError("kernel must not launch")
    ):
        assert not verify_mla.verify_shared_kv_fwd(
            q, k, v, out, kb, vb, qo, kvp, kvi, None, True, None, 6, 1.0, 1.0
        )
    assert torch.all(out == 123.0)


@pytest.mark.parametrize("legacy", [True, False])
def test_paged_kv_layout_accepts_explicit_physical_page_size(legacy):
    q, k, v, kb, vb, qo, kvp, kvi = _inputs(True, True)
    if not legacy:
        kb = kb.permute(0, 2, 1, 3).contiguous()
        vb = vb.permute(0, 3, 1, 2).contiguous()
    with patch.object(verify_mla, "_kv_layout_hcu_fa", legacy):
        assert verify_mla.can_handle(
            q, k, v, kb, vb, qo, kvp, kvi, None, True, None, 6, page_size=64
        )
        assert not verify_mla.can_handle(
            q, k, v, kb, vb, qo, kvp, kvi, None, True, None, 6, page_size=32
        )


@pytest.mark.parametrize(
    "enabled,fp8,expected",
    [(True, True, True), (False, True, False), (True, False, False)],
)
@pytest.mark.parametrize("bs", [1, 32, 128])
def test_v2_launch_is_not_restricted_to_tuned_batch_sizes(enabled, fp8, expected, bs):
    # Exercise the real Python dispatch without allocating a GPU context.
    policy = verify_mla.VerifyMLA.__new__(verify_mla.VerifyMLA)
    policy.h_q, policy.kv_group_num = 8, 4
    policy.head_dim = policy.v_head_dim = policy.nope_dim = 64
    policy.pe_dim = 0
    policy.l_ext, policy.l_pad = 7, 8
    policy.block_h, policy.block_n, policy.num_warps = 4, 256, 4
    policy.n_head_blocks, policy.has_kv_heads = 2, True
    policy.att_out = torch.empty((bs, 8, 32, 8, 64), device="meta")
    policy.att_lse = torch.empty((bs, 8, 32, 8), device="meta")
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    q = torch.empty((bs * 7, 8, 64), dtype=torch.bfloat16, device="meta")
    kb = torch.empty((4, 2, 64, 64), dtype=dtype, device="meta")
    vb = torch.empty_like(kb)
    indptr = torch.empty((bs + 1,), dtype=torch.int32, device="meta")
    indices = torch.empty((bs * 64,), dtype=torch.int32, device="meta")
    with (
        patch.object(verify_mla, "_is_hcu", True),
        patch.object(verify_mla, "_DSPARK_SPLITKV_V2", enabled),
        patch.object(verify_mla, "_kv_layout_hcu_fa", True),
        patch.object(verify_mla, "_AMD_LAUNCH_KWARGS", {"waves_per_eu": 1}),
        patch.object(verify_mla, "_verify_mla_prefix_stage1") as kernel,
    ):
        policy._run_prefix_kernel(
            bs, 1, q, kb, vb, indptr, indptr, indices, 0.125, 1.0, 1.0, page_size=64
        )
        assert kernel.__getitem__.return_value.call_args.kwargs["PAGED_V2"] is expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
