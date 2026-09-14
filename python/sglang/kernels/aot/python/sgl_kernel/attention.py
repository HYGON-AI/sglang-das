# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Hygon modifications to this file are licensed under the Apache License,
# Version 2.0 (the "License"); you may not use these modifications except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple

import torch


def merge_state_v2(
    v_a: torch.Tensor,
    s_a: torch.Tensor,
    v_b: torch.Tensor,
    s_b: torch.Tensor,
    v_merged: Optional[torch.Tensor] = None,
    s_merged: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    s_a = s_a.to(torch.float32)
    s_b = s_b.to(torch.float32)
    # TODO(DefTruth): Currently, the custom merge_attn_states kernel
    # does not support the FP8 data type and non - CUDA devices.
    # It may be necessary to fall back to using the Triton kernel.

    # Avoid creating new tensors if they are already provided
    if v_merged is None:
        v_merged = torch.empty_like(v_a)
    if s_merged is None:
        s_merged = torch.empty_like(s_a)
    torch.ops.sgl_kernel.merge_state_v2.default(v_a, s_a, v_b, s_b, v_merged, s_merged)
    return v_merged, s_merged


def normal_decode_metadata_general(
    seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: torch.Tensor,
    swa_page_table: Optional[torch.Tensor],
    full_to_swa_mapping: Optional[torch.Tensor],
    max_seq_pages: int,
    page_size: int,
    seq_len_delta: int,
    use_swa: bool,
) -> None:
    torch.ops.sgl_kernel.normal_decode_metadata_general.default(
        seq_lens,
        req_to_token,
        req_pool_indices,
        cache_seqlens_int32,
        cu_seqlens_k,
        page_table,
        swa_page_table,
        full_to_swa_mapping,
        max_seq_pages,
        page_size,
        seq_len_delta,
        use_swa,
    )
