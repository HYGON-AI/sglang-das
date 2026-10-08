from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.cp.cp_decode_attn_tp import CpDecodeAttnTpContext

from sglang.test.ci.ci_register import register_hcu_ci
register_hcu_ci(est_time=5, suite="stage-b-test-1-hcu-small")


def _context() -> CpDecodeAttnTpContext:
    context = CpDecodeAttnTpContext.__new__(CpDecodeAttnTpContext)
    context.decode_tp_rank = 3
    context.decode_tp_size = 8
    context._slice_cache = {}
    return context


def test_replicated_singleton_scale_is_not_sliced():
    context = _context()
    scale = torch.nn.Parameter(torch.ones((128, 1)), requires_grad=False)
    linear = SimpleNamespace(weight_scale=scale)

    activated = context._activate(
        linear,
        "weight_scale",
        dim=1,
        allow_replicated_singleton=True,
    )

    assert not activated
    assert linear.weight_scale is scale
    assert tuple(linear.weight_scale.shape) == (128, 1)
    assert context._slice_cache == {}


def test_non_singleton_scale_still_requires_even_partition():
    context = _context()
    linear = SimpleNamespace(weight_scale=torch.ones((128, 10)))

    with pytest.raises(AssertionError, match="not divisible by decode_tp_size=8"):
        context._activate(
            linear,
            "weight_scale",
            dim=1,
            allow_replicated_singleton=True,
        )


def test_partitioned_scale_is_sliced_and_restored():
    context = _context()
    original = torch.arange(128 * 16, dtype=torch.float32).reshape(128, 16)
    linear = SimpleNamespace(weight_scale=original)

    assert context._activate(
        linear,
        "weight_scale",
        dim=1,
        allow_replicated_singleton=True,
    )
    torch.testing.assert_close(linear.weight_scale, original[:, 6:8])

    context._restore(linear, "weight_scale")
    assert linear.weight_scale is original


@pytest.mark.parametrize(
    ("shape", "input_size", "output_size", "default_dim", "expected_dim"),
    [
        ((8192, 512), 512, 8192, 0, 0),
        ((512, 8192), 512, 8192, 0, 1),
        ((6144, 8192), 8192, 6144, 1, 1),
        ((8192, 6144), 8192, 6144, 1, 0),
    ],
)
def test_weight_partition_dim_tracks_post_load_layout(
    shape, input_size, output_size, default_dim, expected_dim
):
    linear = SimpleNamespace(
        weight=torch.empty(shape),
        input_size_per_partition=input_size,
        output_size_per_partition=output_size,
    )

    assert (
        CpDecodeAttnTpContext._weight_partition_dim(linear, default_dim)
        == expected_dim
    )


def test_weight_partition_dim_uses_column_linear_size_fields():
    linear = SimpleNamespace(
        weight=torch.empty((512, 8192)),
        input_size=512,
        output_size_per_partition=8192,
    )

    assert CpDecodeAttnTpContext._weight_partition_dim(linear, 0) == 1


def test_weight_partition_dim_uses_row_linear_size_fields():
    linear = SimpleNamespace(
        weight=torch.empty((8192, 6144)),
        input_size_per_partition=8192,
        output_size=6144,
    )

    assert CpDecodeAttnTpContext._weight_partition_dim(linear, 1) == 0


def test_transposed_column_weight_keeps_output_slice_view():
    context = _context()
    original = torch.arange(8 * 32, dtype=torch.float32).reshape(8, 32)
    linear = SimpleNamespace(weight=original)

    assert context._activate(
        linear,
        "weight",
        dim=1,
        transposed_weight=True,
    )
    torch.testing.assert_close(linear.weight, original[:, 12:16])
    assert linear.weight.stride() == original.stride()
    assert not linear.weight.is_contiguous()


def test_transposed_row_weight_repacks_reduced_input_stride():
    context = _context()
    original = torch.arange(32 * 7, dtype=torch.float32).reshape(32, 7)
    linear = SimpleNamespace(weight=original)

    assert context._activate(
        linear,
        "weight",
        dim=0,
        transposed_weight=True,
    )
    torch.testing.assert_close(linear.weight, original[12:16, :])
    assert linear.weight.stride() == (1, 4)
    assert linear.weight.t().is_contiguous()

    context._restore(linear, "weight")
    assert linear.weight is original


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
