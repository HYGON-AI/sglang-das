"""Aligned attention/MoE groups retain local rows but still owe an expert sum."""

from unittest.mock import patch

import pytest

from sglang.srt.layers.layer_boundary.contracts import BatchVariant
from sglang.srt.layers.layer_boundary.factories import _resolve_ffn
from sglang.srt.layers.layer_boundary.layout import SumGroup, TokenAxis
from sglang.srt.layers.moe import utils as moe_utils
from sglang.test.boundary_fixtures import parallel_of, planning


@pytest.mark.parametrize(
    "variant", [BatchVariant.ORDINARY, BatchVariant.INPUT_SCATTERED]
)
@pytest.mark.parametrize(
    "is_hcu,moe_dp,moe_tp,moe_ep,aligned",
    [
        (True, 4, 2, 1, True),
        (False, 4, 2, 1, False),
        (True, 1, 8, 1, False),
        (True, 2, 2, 2, False),
    ],
)
def test_aligned_rows_keep_the_moe_sum(
    variant, is_hcu, moe_dp, moe_tp, moe_ep, aligned
):
    parallel = parallel_of(
        attn_dp=4, attn_tp=2, moe_dp_size=moe_dp, moe_tp_size=moe_tp, moe_ep_size=moe_ep
    )
    with planning(parallel), patch.object(moe_utils, "_is_hcu", is_hcu):
        contract, _, _ = _resolve_ffn(variant, sparse=True)
    assert (TokenAxis.ATTN_DP in contract.input.layout.sharded) is aligned
    assert contract.output.group is SumGroup.MOE_OUTPUT
    if aligned:
        assert not contract.output.may_reduce_scatter
        assert not contract.output.may_reduce_scatterv
