from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module(input_dtype: torch.dtype, state_dtype: torch.dtype) -> Module:
    args = make_cpp_args(input_dtype, state_dtype)
    return load_jit(
        "fused_sigmoid_gating_delta_rule_update_hcu",
        *args,
        cuda_files=["attention/fused_sigmoid_gating_delta_rule_update_hcu.cuh"],
        cuda_wrappers=[
            (
                "run",
                f"FusedSigmoidGatingDeltaRuleUpdateHcu<{args}>::run",
            )
        ],
    )


def fused_sigmoid_gating_delta_rule_update_cuda_hcu(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    lower_bound: Optional[float] = None,
    beta_scale: float = 1.0,
) -> torch.Tensor:
    """Experimental HIP/CUDA decode kernel for the fixed GLM5 KDA shape."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    output = torch.empty_like(v)
    module = _jit_module(q.dtype, initial_state_source.dtype)
    module.run(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        initial_state_source,
        initial_state_indices,
        output,
        float(scale),
        use_qk_l2norm_in_kernel,
        lower_bound is not None,
        float(lower_bound or 0.0),
        float(beta_scale),
    )
    return output
