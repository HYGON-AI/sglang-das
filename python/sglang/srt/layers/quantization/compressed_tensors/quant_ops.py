# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch

try:
    from lightop import gemm_ops as quant_ops
except Exception:
    print(
        "INFO: Please install lightop if you want to infer gptq or awq or w8a8 model.\n"
    )


def triton_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
    best_config: Optional[list] = None,
) -> torch.Tensor:
    # w8a8 decode with small M is not covered by the smooth GEMM tuning table;
    # explicit slide values avoid excessive padding in the default heuristic.
    if (
        best_config is None
        and bias is None
        and a.dtype == torch.int8
        and b.dtype == torch.int8
        and b.dim() == 2
        and not b.is_contiguous()
        and a.shape[-1] == b.shape[0]
        and b.shape[0] >= 128
    ):
        m_i = a.shape[0]
        k_i = b.shape[0]
        n_i = b.shape[1]
        if n_i <= 1024 and m_i <= 256:
            tune = 1
        elif n_i <= 2048 and m_i <= 128:
            tune = 1
        elif m_i <= 64:
            tune = 9
        else:
            tune = None
        # The tuned slide values are valid for complete K=256 tiles.
        if tune is not None and k_i % 256 == 0:
            from lightop import op as _lop

            try:
                return _lop.gemm_w8a8_smooth_asm(
                    a,
                    b,
                    scale_a,
                    scale_b,
                    None,
                    out_dtype,
                    m_i,
                    n_i,
                    k_i,
                    True,
                    tune,
                )
            except Exception:
                pass

    return quant_ops.triton_scaled_mm(
        a, b, scale_a, scale_b, out_dtype, bias, best_config
    )


def cutlass_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    `cutlass_scaled_mm` implements a fused version of
        `output = torch.mm((scale_a * a), (scale_b * b)).to(out_dtype)`
    where scale_a * a and scale_b * b are implemented using numpy-style
    broadcasting.

    In order to support blockwise scaling like found in DeepSeek V3 we also
    support extended "group" broadcast rules. We extend the numpy-style
    broadcasting rules with the following rule:
        "if the extent of a dimension in the source shape is between 1 and
        corresponding extent in the target shape we repeat each element along
        that dimension  src_shape[dim] // target_shape[dim] times consecutively"
    example if we have:
          a = [[1, 2], and target_shape = (2, 4)
               [3, 4]]
    then we would expand a to:
          a = [[1, 1, 2, 2],
               [3, 3, 4, 4]]
    currently we only support the case:
        scale_a.shape * [1, 128] == a.shape
        scale_b.shape * [128, 128] == b.shape
    """
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert bias is None or bias.shape[0] == b.shape[1] and bias.dtype == out_dtype

    # m = a.shape[0]
    # n = b.shape[1]

    # cutlass_compatible_b = (b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0)
    # if current_platform.is_rocm() or not cutlass_compatible_b:
    #     from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa
    #         triton_scaled_mm)
    #     return triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    # out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    # torch.ops._C.cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias)

    # return out
    # return quant_ops.cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    return quant_ops.rocblas_scaled_mm_nn(a, b, scale_a, scale_b, out_dtype, bias)


def rocblas_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    return quant_ops.rocblas_scaled_mm_nn(a, b, scale_a, scale_b, out_dtype, bias)


def blaslt_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """W8A8 scaled GEMM via hipBLASLt.

    Preferred path on KME/gfx928: lmslim ``hipblaslt_gemm_channelwise_kme``
    (swapped TN, bf16 C/D, fp32 compute) with weight layout ``[K, N]``
    col-major stride ``(1, K)``.

    Fallback: legacy ``hipblaslt_w8a8_gemm`` expecting contiguous weight
    ``[N, K]`` (NT logical / TN API).
    """
    m, k = a.shape[-2], a.shape[-1]
    # KME packed weight: [K, N] with stride (1, K)
    if b.dim() == 2 and b.shape[0] == k and b.stride(0) == 1 and b.stride(1) == k:
        n = b.shape[1]
        return _blaslt_scaled_mm_kme(a, b, scale_a, scale_b, m, n, k, out_dtype, bias)

    # Legacy contiguous [N, K]
    n = b.shape[0]
    _, out = quant_ops.hipblaslt_w8a8_gemm(
        a, b, scale_a, scale_b, m, n, k, "NT", out_dtype
    )
    return out


def _blaslt_scaled_mm_kme(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    m: int,
    n: int,
    k: int,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Logical O[M,N]=X[M,K]@W[K,N] via hipblaslt_gemm_channelwise_kme TN."""
    import lmslimquant

    if out_dtype is not torch.bfloat16:
        raise ValueError(
            "hipblaslt_gemm_channelwise_kme currently requires bfloat16 out"
        )
    if not a.is_contiguous():
        a = a.contiguous()
    scale_a = scale_a.reshape(-1).contiguous()
    scale_b = scale_b.reshape(-1).contiguous()
    bias_arg = None
    if bias is not None:
        if bias.dtype != out_dtype:
            bias = bias.to(out_dtype)
        bias_arg = bias.reshape(-1).contiguous()
    alpha = torch.tensor(1, dtype=torch.int32)
    beta = torch.tensor(0, dtype=torch.int32)
    # API (m,n,k)=(N,M,K), A=W, B=X, trans=TN
    out = lmslimquant.hipblaslt_gemm_channelwise_kme(
        b,
        a,
        scale_b,
        scale_a,
        out_dtype,
        n,
        m,
        k,
        1,
        "TN",
        alpha,
        beta,
        bias_arg,
    )
    if out.dim() == 3:
        out = out.squeeze(0)
    return out


def triton_int8_gemm_helper(
    m: int,
    n: int,
    k: int,
    per_token_act_quant: bool,
    per_out_channel_weight_quant: bool,
    use_bias: bool,
    out_dtype: type[torch.dtype] = torch.float16,
    device: str = "cuda:0",
    best_config: Optional[list] = None,
    repeat: Optional[int] = 2,
):
    return quant_ops.triton_int8_gemm_helper(
        m,
        n,
        k,
        per_token_act_quant,
        per_out_channel_weight_quant,
        use_bias,
        out_dtype,
        device,
        best_config,
        repeat,
    )
