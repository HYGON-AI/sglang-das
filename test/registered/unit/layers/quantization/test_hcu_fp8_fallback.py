"""CPU dispatch contracts; GPU numerical equivalence is verified separately."""

from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.quantization import fp8_utils


@pytest.mark.parametrize("is_hcu", [False, True])
def test_hipblaslt_no_solution_fallback_is_hcu_only_and_cached(is_hcu):
    qinput = torch.zeros((2, 16), dtype=torch.float8_e4m3fn)
    weight = torch.zeros((16, 8), dtype=torch.float8_e4m3fn)
    expected = torch.full((2, 8), 2.0, dtype=torch.bfloat16)
    kwargs = dict(
        qinput=qinput,
        weight=weight,
        x_scale=torch.ones((2, 1)),
        weight_scale=torch.ones((8, 1)),
        input_2d_shape=(2, 16),
        output_shape=(2, 8),
        bias=None,
        input_dtype=torch.bfloat16,
    )
    with (
        patch.object(fp8_utils, "_is_hcu", is_hcu),
        patch.object(fp8_utils, "TORCH_DEVICE_IDENTITY", None),
        patch.object(fp8_utils, "_HIPBLASLT_UNSUPPORTED_FP8_SHAPES", set()),
        patch.object(
            torch,
            "_scaled_mm",
            side_effect=RuntimeError("could not find valid hipblaslt solution"),
        ) as scaled,
        patch.object(fp8_utils, "triton_scaled_mm", return_value=expected) as fallback,
    ):
        if is_hcu:
            for _ in range(2):
                torch.testing.assert_close(
                    fp8_utils._apply_fallback_scaled_mm(**kwargs), expected
                )
            assert scaled.call_count == 1
            assert fallback.call_count == 2
        else:
            with pytest.raises(RuntimeError, match="hipblaslt solution"):
                fp8_utils._apply_fallback_scaled_mm(**kwargs)
            fallback.assert_not_called()
