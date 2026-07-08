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

from __future__ import annotations

import warnings

import torch

from sglang.srt.utils import get_bool_env_var, direct_register_custom_op

_USE_OPT_CAT = get_bool_env_var("SGLANG_USE_OPT_CAT")

if _USE_OPT_CAT:
    try:
        from lightop import ds_cat  # type: ignore
    except ImportError:  # pragma: no cover
        ds_cat = None
        warnings.warn(
            "SGLANG_USE_OPT_CAT 已开启但无法导入 lightop.ds_cat，退回 torch.cat"
        )
else:
    ds_cat = None
    




# TODO: 单独注册有些问题
def ds_cat_wrapper(A: torch.Tensor,
                   B: torch.Tensor,
                   dim: int,
                   mode: int) -> torch.Tensor:
    output_shape = list(A.shape)
    output_shape[dim] = A.shape[dim] + B.shape[dim]
    C = torch.empty(output_shape, device=A.device, dtype=A.dtype)
    ds_cat(A, B, C, mode)
    return C
    
def ds_cat_fake(A: torch.Tensor,
                B: torch.Tensor,
                dim: int,
                mode: int) -> torch.Tensor:
    # 使用标准cat作为fake实现
    return torch.cat([A, B], dim=dim)

direct_register_custom_op(
    op_name="ds_cat",
    op_func=ds_cat_wrapper,
    mutates_args=[],  # 没有修改参数，只有返回值
    fake_impl=ds_cat_fake
)

def concat_decode_opt(A: torch.Tensor, B: torch.Tensor, dim: int):
    assert dim == 2, "tensor dim must be 3 and concat dim must be 2"
    mode = 0
    if dim != 0:
        return torch.ops.sglang.ds_cat(A, B, dim, mode)
    assert False, "not support"

# def concat_decode_opt(A:torch.Tensor, B:torch.Tensor, dim:int):
#     assert dim==2 , "tensor dim must be 3 and concat dim must be 2"
#     output_shape = list(A.shape)
#     output_shape[dim] = A.shape[dim] + B.shape[dim]  
#     C = torch.empty(output_shape, device=A.device, dtype=A.dtype)
#     mode=0
#     if dim!=0 :
#         ds_cat(A, B, C, mode)
#         return C
#     assert False, "not support"
