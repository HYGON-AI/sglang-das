// SPDX-License-Identifier: Apache-2.0
// Adapted from the SGLang rotary embedding kernel for MiniMax-H3 on HCU.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/all.h>

#include "utils.h"

namespace {

template <typename scalar_t>
__device__ void rotate_pair(scalar_t* row, const scalar_t* cache, int offset, int half) {
  const scalar_t x = row[offset];
  const scalar_t y = row[half + offset];
  const scalar_t cos = SGLANG_LDG(cache + offset);
  const scalar_t sin = SGLANG_LDG(cache + half + offset);
  row[offset] = x * cos - y * sin;
  row[half + offset] = y * cos + x * sin;
}

template <typename scalar_t>
__global__ __launch_bounds__(256) void minimax_h3_rope_kernel(
    const int64_t* positions,
    scalar_t* query,
    scalar_t* key,
    const scalar_t* cache,
    int heads,
    int head_size,
    int rot_dim,
    int64_t query_stride,
    int64_t key_stride) {
  const int64_t token = blockIdx.x;
  const int half = rot_dim / 2;
  const scalar_t* row_cache = cache + positions[token] * rot_dim;
  for (int i = threadIdx.x; i < heads * half; i += blockDim.x) {
    const int64_t head = (i / half) * head_size;
    const int offset = i % half;
    rotate_pair(query + token * query_stride + head, row_cache, offset, half);
    if (key != nullptr) {
      rotate_pair(key + token * key_stride + head, row_cache, offset, half);
    }
  }
}

}  // namespace

void minimax_h3_rotary_embedding(
    torch::Tensor& positions,
    torch::Tensor& query,
    std::optional<torch::Tensor> key,
    int64_t head_size,
    torch::Tensor& cos_sin_cache,
    bool is_neox) {
  TORCH_CHECK(is_neox, "MiniMax-H3 RoPE requires the NeoX layout");
  TORCH_CHECK(query.is_cuda() && query.dim() == 2 && query.stride(1) == 1,
              "MiniMax-H3 RoPE requires packed GPU Q/K with contiguous heads");
  TORCH_CHECK(positions.device() == query.device() && positions.scalar_type() == torch::kInt64 &&
                  positions.dim() == 1 && positions.is_contiguous() && positions.numel() == query.size(0),
              "MiniMax-H3 RoPE requires one GPU int64 position per token");
  TORCH_CHECK(head_size > 0 && query.size(1) > 0 && query.size(1) % head_size == 0,
              "Invalid MiniMax-H3 attention head size");
  TORCH_CHECK(cos_sin_cache.device() == query.device() && cos_sin_cache.scalar_type() == query.scalar_type() &&
                  cos_sin_cache.dim() == 2 && cos_sin_cache.is_contiguous(),
              "MiniMax-H3 RoPE cache must have the Q/K device and dtype");
  const int64_t rot_dim = cos_sin_cache.size(1);
  TORCH_CHECK(rot_dim > 0 && rot_dim % 2 == 0 && rot_dim <= head_size,
              "Invalid MiniMax-H3 rotary dimension");
  if (key.has_value()) {
    TORCH_CHECK(key->device() == query.device() && key->scalar_type() == query.scalar_type() &&
                    key->sizes() == query.sizes() && key->stride(1) == 1,
                "MiniMax-H3 RoPE requires matching packed Q/K");
  }
  if (query.size(0) == 0) {
    return;
  }
  const int64_t heads = query.size(1) / head_size;
  const dim3 grid(query.size(0));
  const dim3 block(std::min<int64_t>(heads * rot_dim / 2, 256));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_FLOAT_TYPES(query.scalar_type(), "minimax_h3_rotary_embedding", [&] {
    minimax_h3_rope_kernel<scalar_t><<<grid, block, 0, stream>>>(
        positions.data_ptr<int64_t>(),
        query.data_ptr<scalar_t>(),
        key.has_value() ? key->data_ptr<scalar_t>() : nullptr,
        cos_sin_cache.data_ptr<scalar_t>(),
        heads,
        head_size,
        rot_dim,
        query.stride(0),
        key.has_value() ? key->stride(0) : 0);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
