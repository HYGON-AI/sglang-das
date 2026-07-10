/*
 * Copyright 2026 Hygon Information Technology Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <optional>

namespace {

template <typename SeqT>
__global__ void normal_decode_metadata_prefix_kernel(
    const SeqT* __restrict__ seq_lens,
    int32_t* __restrict__ cache_seqlens_int32,
    int32_t* __restrict__ cu_seqlens_k,
    int64_t seq_lens_stride_0,
    int64_t cache_seqlens_int32_stride_0,
    int64_t cu_seqlens_k_stride_0,
    int64_t batch_size,
    int64_t seq_len_delta) {
  int32_t acc = 0;
  for (int64_t i = 0; i < batch_size; ++i) {
    const auto val = static_cast<int32_t>(seq_lens[i * seq_lens_stride_0] + seq_len_delta);
    cache_seqlens_int32[i * cache_seqlens_int32_stride_0] = val;
    cu_seqlens_k[i * cu_seqlens_k_stride_0] = acc;
    acc += val;
  }
  cu_seqlens_k[batch_size * cu_seqlens_k_stride_0] = acc;
}

template <typename PoolT>
__global__ void normal_decode_metadata_gather_kernel(
    const int32_t* __restrict__ req_to_token,
    const PoolT* __restrict__ req_pool_indices,
    int32_t* __restrict__ page_table,
    int32_t* __restrict__ swa_page_table,
    const int32_t* __restrict__ full_to_swa_mapping,
    int64_t req_to_token_stride_0,
    int64_t req_to_token_stride_1,
    int64_t req_pool_indices_stride_0,
    int64_t page_table_stride_0,
    int64_t page_table_stride_1,
    int64_t swa_page_table_stride_0,
    int64_t swa_page_table_stride_1,
    int64_t full_to_swa_mapping_stride_0,
    int64_t batch_size,
    int64_t max_seq_pages,
    int64_t shift,
    bool use_swa) {
  const int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch_size * max_seq_pages;
  if (linear_idx >= total) {
    return;
  }

  const int64_t batch_idx = linear_idx / max_seq_pages;
  const int64_t col = linear_idx - batch_idx * max_seq_pages;
  const auto row_idx = static_cast<int64_t>(req_pool_indices[batch_idx * req_pool_indices_stride_0]);
  const int64_t col_idx = col << shift;
  const int64_t rt_offset =
      static_cast<int64_t>(row_idx) * req_to_token_stride_0 + col_idx * req_to_token_stride_1;
  const int32_t page_index = req_to_token[rt_offset];
  const int32_t page_table_val = page_index >> shift;

  page_table[batch_idx * page_table_stride_0 + col * page_table_stride_1] = page_table_val;

  if (use_swa) {
    const int32_t swa_slot = full_to_swa_mapping[static_cast<int64_t>(page_index) * full_to_swa_mapping_stride_0];
    const int32_t swa_val = swa_slot >> shift;
    swa_page_table[batch_idx * swa_page_table_stride_0 + col * swa_page_table_stride_1] = swa_val;
  }
}

template <typename SeqT, typename PoolT>
void launch_normal_decode_metadata_general(
    const torch::Tensor& seq_lens,
    const torch::Tensor& req_to_token,
    const torch::Tensor& req_pool_indices,
    torch::Tensor& cache_seqlens_int32,
    torch::Tensor& cu_seqlens_k,
    torch::Tensor& page_table,
    std::optional<torch::Tensor> swa_page_table,
    std::optional<torch::Tensor> full_to_swa_mapping,
    int64_t max_seq_pages,
    int64_t page_size,
    int64_t seq_len_delta,
    bool use_swa) {
  const int64_t batch_size = cache_seqlens_int32.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();

  normal_decode_metadata_prefix_kernel<SeqT><<<1, 1, 0, stream>>>(
      static_cast<const SeqT*>(seq_lens.data_ptr()),
      static_cast<int32_t*>(cache_seqlens_int32.data_ptr()),
      static_cast<int32_t*>(cu_seqlens_k.data_ptr()),
      seq_lens.stride(0),
      cache_seqlens_int32.stride(0),
      cu_seqlens_k.stride(0),
      batch_size,
      seq_len_delta);

  if (max_seq_pages <= 0) {
    return;
  }

  constexpr int threads = 256;
  const int64_t total = batch_size * max_seq_pages;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  int64_t shift = 0;
  for (int64_t v = page_size; v > 1; v >>= 1) {
    ++shift;
  }

  int32_t* swa_page_table_ptr = nullptr;
  const int32_t* full_to_swa_mapping_ptr = nullptr;
  int64_t swa_page_table_stride_0 = 0;
  int64_t swa_page_table_stride_1 = 0;
  int64_t full_to_swa_mapping_stride_0 = 0;
  if (use_swa) {
    swa_page_table_ptr = static_cast<int32_t*>(swa_page_table->data_ptr());
    full_to_swa_mapping_ptr = static_cast<const int32_t*>(full_to_swa_mapping->data_ptr());
    swa_page_table_stride_0 = swa_page_table->stride(0);
    swa_page_table_stride_1 = swa_page_table->stride(1);
    full_to_swa_mapping_stride_0 = full_to_swa_mapping->stride(0);
  }

  normal_decode_metadata_gather_kernel<PoolT><<<blocks, threads, 0, stream>>>(
      static_cast<const int32_t*>(req_to_token.data_ptr()),
      static_cast<const PoolT*>(req_pool_indices.data_ptr()),
      static_cast<int32_t*>(page_table.data_ptr()),
      swa_page_table_ptr,
      full_to_swa_mapping_ptr,
      req_to_token.stride(0),
      req_to_token.stride(1),
      req_pool_indices.stride(0),
      page_table.stride(0),
      page_table.stride(1),
      swa_page_table_stride_0,
      swa_page_table_stride_1,
      full_to_swa_mapping_stride_0,
      batch_size,
      max_seq_pages,
      shift,
      use_swa);
}

}  // namespace

void normal_decode_metadata_general(
    const torch::Tensor& seq_lens,
    const torch::Tensor& req_to_token,
    const torch::Tensor& req_pool_indices,
    torch::Tensor& cache_seqlens_int32,
    torch::Tensor& cu_seqlens_k,
    torch::Tensor& page_table,
    std::optional<torch::Tensor> swa_page_table,
    std::optional<torch::Tensor> full_to_swa_mapping,
    int64_t max_seq_pages,
    int64_t page_size,
    int64_t seq_len_delta,
    bool use_swa) {
  TORCH_CHECK(seq_lens.is_cuda(), "seq_lens must be a CUDA tensor");
  TORCH_CHECK(req_to_token.is_cuda(), "req_to_token must be a CUDA tensor");
  TORCH_CHECK(req_pool_indices.is_cuda(), "req_pool_indices must be a CUDA tensor");
  TORCH_CHECK(cache_seqlens_int32.is_cuda(), "cache_seqlens_int32 must be a CUDA tensor");
  TORCH_CHECK(cu_seqlens_k.is_cuda(), "cu_seqlens_k must be a CUDA tensor");
  TORCH_CHECK(page_table.is_cuda(), "page_table must be a CUDA tensor");
  TORCH_CHECK(req_to_token.scalar_type() == at::ScalarType::Int, "req_to_token must be int32");
  TORCH_CHECK(
      req_pool_indices.scalar_type() == at::ScalarType::Int ||
          req_pool_indices.scalar_type() == at::ScalarType::Long,
      "req_pool_indices must be int32 or int64");
  TORCH_CHECK(cache_seqlens_int32.scalar_type() == at::ScalarType::Int, "cache_seqlens_int32 must be int32");
  TORCH_CHECK(cu_seqlens_k.scalar_type() == at::ScalarType::Int, "cu_seqlens_k must be int32");
  TORCH_CHECK(page_table.scalar_type() == at::ScalarType::Int, "page_table must be int32");
  TORCH_CHECK(page_size > 0 && (page_size & (page_size - 1)) == 0, "page_size must be a power of two");
  TORCH_CHECK(cache_seqlens_int32.dim() == 1, "cache_seqlens_int32 must be 1D");
  TORCH_CHECK(cu_seqlens_k.numel() >= cache_seqlens_int32.numel() + 1, "cu_seqlens_k is too small");
  TORCH_CHECK(page_table.dim() == 2, "page_table must be 2D");
  TORCH_CHECK(page_table.size(0) >= cache_seqlens_int32.size(0), "page_table batch dimension is too small");
  TORCH_CHECK(!use_swa || (swa_page_table.has_value() && full_to_swa_mapping.has_value()),
              "SWA metadata tensors are required when use_swa is true");
  if (use_swa) {
    TORCH_CHECK(swa_page_table->is_cuda(), "swa_page_table must be a CUDA tensor");
    TORCH_CHECK(full_to_swa_mapping->is_cuda(), "full_to_swa_mapping must be a CUDA tensor");
    TORCH_CHECK(swa_page_table->scalar_type() == at::ScalarType::Int, "swa_page_table must be int32");
    TORCH_CHECK(full_to_swa_mapping->scalar_type() == at::ScalarType::Int, "full_to_swa_mapping must be int32");
  }

  const c10::cuda::OptionalCUDAGuard device_guard(seq_lens.device());
  if (seq_lens.scalar_type() == at::ScalarType::Int && req_pool_indices.scalar_type() == at::ScalarType::Int) {
    launch_normal_decode_metadata_general<int32_t, int32_t>(
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
        use_swa);
  } else if (seq_lens.scalar_type() == at::ScalarType::Int && req_pool_indices.scalar_type() == at::ScalarType::Long) {
    launch_normal_decode_metadata_general<int32_t, int64_t>(
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
        use_swa);
  } else if (seq_lens.scalar_type() == at::ScalarType::Long && req_pool_indices.scalar_type() == at::ScalarType::Int) {
    launch_normal_decode_metadata_general<int64_t, int32_t>(
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
        use_swa);
  } else if (seq_lens.scalar_type() == at::ScalarType::Long && req_pool_indices.scalar_type() == at::ScalarType::Long) {
    launch_normal_decode_metadata_general<int64_t, int64_t>(
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
        use_swa);
  } else {
    TORCH_CHECK(false, "seq_lens must be int32 or int64");
  }

  cudaError_t status = cudaGetLastError();
  TORCH_CHECK(status == cudaSuccess, "normal_decode_metadata_general launch failed: ", cudaGetErrorString(status));
}
