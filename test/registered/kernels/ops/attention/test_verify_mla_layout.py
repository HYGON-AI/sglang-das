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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
