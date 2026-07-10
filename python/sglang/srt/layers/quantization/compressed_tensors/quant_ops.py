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

import torch
from typing import List, Optional, Tuple
try:
    from lmslim import quant_ops 
    from lmslim import quant_tools 
except Exception:
    print("INFO: Please install lmslim if you want to infer gptq or awq  or w8a8 model.\n") 

def triton_scaled_mm(a: torch.Tensor,
                      b: torch.Tensor,
                      scale_a: torch.Tensor,
                      scale_b: torch.Tensor,
                      out_dtype: torch.dtype,
                      bias: Optional[torch.Tensor] = None,
                      best_config:Optional[list] = None) -> torch.Tensor:

    return quant_ops.triton_scaled_mm(a, b,scale_a,scale_b,out_dtype,bias,best_config)

def cutlass_scaled_mm(a: torch.Tensor,
                      b: torch.Tensor,
                      scale_a: torch.Tensor,
                      scale_b: torch.Tensor,
                      out_dtype: torch.dtype,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
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
    assert (out_dtype is torch.bfloat16 or out_dtype is torch.float16)
    assert bias is None or bias.shape[0] == b.shape[
        1] and bias.dtype == out_dtype

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
    #return quant_ops.cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    return quant_ops.rocblas_scaled_mm_nn(a, b, scale_a, scale_b, out_dtype, bias)

def rocblas_scaled_mm(a: torch.Tensor,
                      b: torch.Tensor,
                      scale_a: torch.Tensor,
                      scale_b: torch.Tensor,
                      out_dtype: torch.dtype,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:

    return quant_ops.rocblas_scaled_mm_nn(a, b, scale_a, scale_b, out_dtype, bias)

def blaslt_scaled_mm(a: torch.Tensor,
                      b: torch.Tensor,
                      scale_a: torch.Tensor,
                      scale_b: torch.Tensor,
                      out_dtype: torch.dtype,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    m = a.shape[0]
    n = b.shape[0]
    k = a.shape[1]
    _, out = quant_ops.hipblaslt_w8a8_gemm(a, b, scale_a, scale_b, m, n, k, 'NT', out_dtype)
    return out

def triton_int8_gemm_helper(m: int,
                             n: int,
                             k: int,
                             per_token_act_quant: bool,
                             per_out_channel_weight_quant: bool,
                             use_bias: bool,
                             out_dtype: type[torch.dtype] = torch.float16,
                             device: str = "cuda:0",
                             best_config:Optional[list] = None,
                             repeat:Optional[int] = 2):
    return quant_tools.triton_int8_gemm_helper(m,n,k,per_token_act_quant,per_out_channel_weight_quant,use_bias,out_dtype,device,best_config,repeat)

