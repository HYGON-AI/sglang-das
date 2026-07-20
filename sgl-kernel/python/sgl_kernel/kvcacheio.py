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

from typing import List

import torch


def is_hip() -> bool:
    return torch.version.hip is not None


_is_hip = is_hip()

def hcu_create_extend_after_decode_spec_info(
    verified_id: torch.Tensor,
    seq_lens: torch.Tensor,
    accept_lens: torch.Tensor,
    positions: torch.Tensor,
    new_verified_id: torch.Tensor,
    bs: int,
):
    torch.ops.sgl_kernel.hcu_create_extend_after_decode_spec_info(
        verified_id,
        seq_lens,
        accept_lens,
        positions,
        new_verified_id,
        bs,
    )

def hcu_alloc_extend_kernel(
    pre_lens_ptr: torch.Tensor,
    seq_lens_ptr: torch.Tensor,
    last_loc_ptr: torch.Tensor,
    free_page_ptr: torch.Tensor,
    out_indices: torch.Tensor,
    bs: int,
    page_size: int,
):
    torch.ops.sgl_kernel.hcu_alloc_extend_kernel(
        pre_lens_ptr,
        seq_lens_ptr,
        last_loc_ptr,
        free_page_ptr,
        out_indices,
        bs,
        page_size,
    )

def hcu_alloc_decode_kernel(
    seq_lens_ptr: torch.Tensor,   
    last_loc_ptr: torch.Tensor,    
    free_page_ptr: torch.Tensor ,   
    out_indices: torch.Tensor , 
    bs: int,          
    page_size: int,              
):
    torch.ops.sgl_kernel.hcu_alloc_decode_kernel(
        seq_lens_ptr,
        last_loc_ptr,
        free_page_ptr,
        out_indices,
        bs,
        page_size,
    )

def transfer_kv_all_kernel_lf_pf_D2H_hcu(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    srt_layout_dim: int,
    dst_layout_dim: int,
    page_size: int,
    layer_num: int,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_all_kernel_lf_pf_D2H_hcu.default(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        item_size,
        srt_layout_dim,
        dst_layout_dim,
        page_size,
        layer_num,
        num_warps_per_block,
    )

def transfer_kv_per_layer_kernel_pf_lf_H2D_hcu(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    src_layout_dim: int,
    page_size: int,
    layer_id: int,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer_kernel_pf_lf_H2D_hcu.default(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        item_size,
        src_layout_dim,
        page_size,
        layer_id,
        num_warps_per_block,
    )

def transfer_kv_all_direct_pf_lf_H2D_hcu(
    src_ptrs_k: torch.Tensor,
    src_ptrs_v: torch.Tensor,
    dst_ptrs_k: List[torch.Tensor],
    dst_ptrs_v: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    start_layer_id: int,
    page_size: int,
):
    torch.ops.sgl_kernel.transfer_kv_all_direct_pf_lf_H2D_hcu.default(
        src_ptrs_k,
        src_ptrs_v,
        dst_ptrs_k,
        dst_ptrs_v,
        src_indices,
        dst_indices,
        start_layer_id,
        page_size,
    )

def transfer_kv_all_direct_lf_pf_D2H_hcu(
    src_ptrs_k: List[torch.Tensor],
    src_ptrs_v: List[torch.Tensor],
    dst_ptrs_k: torch.Tensor,
    dst_ptrs_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    start_layer_id: int,
    page_size: int,
):
    torch.ops.sgl_kernel.transfer_kv_all_direct_lf_pf_D2H_hcu.default(
        src_ptrs_k,
        src_ptrs_v,
        dst_ptrs_k,
        dst_ptrs_v,
        src_indices,
        dst_indices,
        start_layer_id,
        page_size,
    )

def transfer_kv_per_layer(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer.default(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        item_size,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_per_layer_pf_lf(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer_pf_lf.default(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        layer_id,
        item_size,
        src_layout_dim,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_per_layer_ph_lf(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    page_size: int,
    head_num: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer_ph_lf.default(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        layer_id,
        item_size,
        src_layout_dim,
        page_size,
        head_num,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer(
    src_k_layers: torch.Tensor,
    dst_k_layers: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v_layers: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer.default(
        src_k_layers,
        dst_k_layers,
        src_v_layers,
        dst_v_layers,
        src_indices,
        dst_indices,
        item_size,
        num_layers,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer_lf_pf(
    src_k_layers: torch.Tensor,
    dst_k: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer_lf_pf.default(
        src_k_layers,
        dst_k,
        src_v_layers,
        dst_v,
        src_indices,
        dst_indices,
        item_size,
        dst_layout_dim,
        num_layers,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer_lf_ph(
    src_k_layers: torch.Tensor,
    dst_k: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    page_size: int,
    head_num: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer_lf_ph.default(
        src_k_layers,
        dst_k,
        src_v_layers,
        dst_v,
        src_indices,
        dst_indices,
        item_size,
        dst_layout_dim,
        num_layers,
        page_size,
        head_num,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_direct(
    src_layers: List[torch.Tensor],
    dst_layers: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
):
    torch.ops.sgl_kernel.transfer_kv_direct.default(
        src_layers, dst_layers, src_indices, dst_indices, page_size
    )


def transfer_kv_per_layer_direct_pf_lf(
    src_ptrs: List[torch.Tensor],
    dst_ptrs: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    page_size: int,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer_direct_pf_lf.default(
        src_ptrs, dst_ptrs, src_indices, dst_indices, layer_id, page_size
    )


def transfer_kv_all_layer_direct_lf_pf(
    src_ptrs: List[torch.Tensor],
    dst_ptrs: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer_direct_lf_pf.default(
        src_ptrs, dst_ptrs, src_indices, dst_indices, page_size
    )


def transfer_kv_per_layer_mla(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer_mla.default(
        src,
        dst,
        src_indices,
        dst_indices,
        item_size,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_per_layer_mla_pf_lf(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer_mla_pf_lf.default(
        src,
        dst,
        src_indices,
        dst_indices,
        layer_id,
        item_size,
        src_layout_dim,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer_mla(
    src_layers: torch.Tensor,
    dst_layers: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer_mla.default(
        src_layers,
        dst_layers,
        src_indices,
        dst_indices,
        item_size,
        num_layers,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer_mla_lf_pf(
    src_layers: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer_mla_lf_pf.default(
        src_layers,
        dst,
        src_indices,
        dst_indices,
        item_size,
        dst_layout_dim,
        num_layers,
        block_quota,
        num_warps_per_block,
    )

def hcu_assign_req_to_token_pool(
    req_pool_indices:torch.Tensor,
    req_to_token:torch.Tensor,
    allocate_lens:torch.Tensor,
    new_allocate_lens:torch.Tensor,
    out_cache_loc:torch.Tensor,
    shape:int,
    bs:int,
):
    torch.ops.sgl_kernel.hcu_assign_req_to_token_pool(
        req_pool_indices,
        req_to_token,
        allocate_lens,
        new_allocate_lens,
        out_cache_loc,
        shape,
        bs,
    )

def hcu_get_last_loc(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
):
    result = torch.ops.sgl_kernel.hcu_get_last_loc(
        req_to_token,
        req_pool_indices,
        prefix_lens,
    )
    return result


def hcu_assign_extend_cache_locs(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    pool_len: int,
    bs: int,
):
    torch.ops.sgl_kernel.hcu_assign_extend_cache_locs(
        req_pool_indices,
        req_to_token, 
        start_offset, 
        end_offset,
        out_cache_loc, 
        pool_len, 
        bs,
    )

def hcu_create_chunked_prefix_cache_kv_indices(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    chunk_starts: torch.Tensor,
    chunk_seq_lens: torch.Tensor,
    chunk_cu_seq_lens: torch.Tensor,
    chunk_kv_indices: torch.Tensor,
    col_num: int,
    bs: int,
):
    torch.ops.sgl_kernel.hcu_create_chunked_prefix_cache_kv_indices(
        req_to_token,
        req_pool_indices, 
        chunk_starts, 
        chunk_seq_lens,
        chunk_cu_seq_lens, 
        chunk_kv_indices, 
        col_num, 
        bs,
    )

def hcu_align_evict_mask_to_page_size(
    seq_lens: torch.Tensor,
    evict_mask: torch.Tensor,
    page_size: int,
    num_draft_tokens: int,
    bs:int,
):
    torch.ops.sgl_kernel.hcu_align_evict_mask_to_page_size(
        seq_lens,
        evict_mask, 
        page_size, 
        num_draft_tokens,
        bs,
    )


def copy_all_layer_kv_cache_cpu(
    data_ptrs: torch.Tensor,
    strides: torch.Tensor,
    tgt_loc: torch.Tensor,
    src_loc: torch.Tensor,
):
    torch.ops.sgl_kernel.copy_all_layer_kv_cache_cpu(
        data_ptrs,
        strides,
        tgt_loc,
        src_loc,
    )
