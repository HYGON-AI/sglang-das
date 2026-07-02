/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cmath>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_bf16.h>
#endif

#ifdef USE_ROCM
#include "pytorch_extension_utils_rocm.h"
#endif
#include "utils.h"

namespace {

constexpr int kMaxThreadsPerBlock = 256;
constexpr int kWarpsPerBlock = kMaxThreadsPerBlock / 32;
constexpr int kOptRowsThreshold = 256;

#ifdef USE_ROCM
__device__ __forceinline__ float bf16_to_float(at::BFloat16 v) {
  return __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(&v));
}
__device__ __forceinline__ at::BFloat16 float_to_bf16(float v) {
  at::BFloat16 out;
  *reinterpret_cast<__hip_bfloat16*>(&out) = __float2bfloat16(v);
  return out;
}
#endif

inline __device__ float to_float(float u) {
  return u;
}
inline __device__ float to_float(at::Half u) {
  return static_cast<float>(u);
}
inline __device__ float to_float(at::BFloat16 u) {
#ifdef USE_ROCM
  return bf16_to_float(u);
#else
  return static_cast<float>(u);
#endif
}
inline __device__ void from_float(float& d, float s) {
  d = s;
}
inline __device__ void from_float(at::Half& d, float s) {
  d = at::Half(s);
}
inline __device__ void from_float(at::BFloat16& d, float s) {
#ifdef USE_ROCM
  d = float_to_bf16(s);
#else
  d = at::BFloat16(s);
#endif
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

template <typename scalar_t>
__global__ void l2norm_fwd_kernel_orig(
    scalar_t* __restrict__ y,
    const scalar_t* __restrict__ x,
    int64_t D,
    float eps,
    int64_t num_rows) {
  const int64_t row = blockIdx.x;
  if (row >= num_rows) {
    return;
  }

  const scalar_t* x_row = x + row * D;
  scalar_t* y_row = y + row * D;

  constexpr int kMaxLocalElems = 8;
  float vals[kMaxLocalElems];
  int local_count = 0;
  for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
    if (local_count < kMaxLocalElems) {
      vals[local_count++] = to_float(x_row[d]);
    }
  }

  float sum_sq = 0.f;
  for (int i = 0; i < local_count; ++i) {
    sum_sq += vals[i] * vals[i];
  }

  __shared__ float shared[kWarpsPerBlock];
  const int lane = threadIdx.x % 32;
  const int warp_id = threadIdx.x / 32;
  sum_sq = warp_reduce_sum(sum_sq);
  if (lane == 0) {
    shared[warp_id] = sum_sq;
  }
  __syncthreads();

  if (warp_id == 0) {
    float block_sum = (lane < blockDim.x / 32) ? shared[lane] : 0.f;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0) {
      shared[0] = block_sum;
    }
  }
  __syncthreads();

  const float inv_norm = rsqrtf(shared[0] + eps);
  for (int i = 0; i < local_count; ++i) {
    const int64_t d = threadIdx.x + static_cast<int64_t>(i) * blockDim.x;
    scalar_t out_val;
    from_float(out_val, vals[i] * inv_norm);
    y_row[d] = out_val;
  }
}

template <typename scalar_t, int ROWS_PER_BLOCK, int MAX_ELEMS_PER_LANE>
__global__ void l2norm_fwd_kernel_tiled(
    scalar_t* __restrict__ y,
    const scalar_t* __restrict__ x,
    int64_t D,
    float eps,
    int64_t num_rows) {
  const int warp_id_in_block = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t row = blockIdx.x * ROWS_PER_BLOCK + warp_id_in_block;
  const bool valid = (row < num_rows);

  const int n_iters = static_cast<int>((D + 31) / 32);
  float vals[MAX_ELEMS_PER_LANE];
#pragma unroll
  for (int i = 0; i < MAX_ELEMS_PER_LANE; ++i) {
    vals[i] = 0.f;
  }

  if (valid) {
    const scalar_t* x_row = x + row * D;
#pragma unroll
    for (int i = 0; i < MAX_ELEMS_PER_LANE; ++i) {
      if (i < n_iters) {
        const int64_t d = lane + static_cast<int64_t>(i) * 32;
        if (d < D) {
          vals[i] = to_float(x_row[d]);
        }
      }
    }
  }

  float sum_sq = 0.f;
#pragma unroll
  for (int i = 0; i < MAX_ELEMS_PER_LANE; ++i) {
    if (i < n_iters) {
      const int64_t d = lane + static_cast<int64_t>(i) * 32;
      if (d < D) {
        sum_sq += vals[i] * vals[i];
      }
    }
  }

  sum_sq = warp_reduce_sum(sum_sq);
  const float inv_norm = rsqrtf(sum_sq + eps);

  if (valid) {
    scalar_t* y_row = y + row * D;
#pragma unroll
    for (int i = 0; i < MAX_ELEMS_PER_LANE; ++i) {
      if (i < n_iters) {
        const int64_t d = lane + static_cast<int64_t>(i) * 32;
        if (d < D) {
          scalar_t out_val;
          from_float(out_val, vals[i] * inv_norm);
          y_row[d] = out_val;
        }
      }
    }
  }
}

template <typename scalar_t, int ROWS_PER_BLOCK>
__global__ void l2norm_fwd_kernel_tiled_d128(
    scalar_t* __restrict__ y,
    const scalar_t* __restrict__ x,
    float eps,
    int64_t num_rows) {
  constexpr int D = 128;
  constexpr int ELEMS_PER_LANE = 4;

  const int warp_id_in_block = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * ROWS_PER_BLOCK + warp_id_in_block;
  const bool valid = row < num_rows;

  float vals[ELEMS_PER_LANE] = {0.f, 0.f, 0.f, 0.f};

  if (valid) {
    const scalar_t* x_row = x + row * D;
#pragma unroll
    for (int i = 0; i < ELEMS_PER_LANE; ++i) {
      vals[i] = to_float(x_row[lane + i * 32]);
    }
  }

  float sum_sq = 0.f;
#pragma unroll
  for (int i = 0; i < ELEMS_PER_LANE; ++i) {
    sum_sq += vals[i] * vals[i];
  }
  sum_sq = warp_reduce_sum(sum_sq);
  const float inv_norm = rsqrtf(sum_sq + eps);

  if (valid) {
    scalar_t* y_row = y + row * D;
#pragma unroll
    for (int i = 0; i < ELEMS_PER_LANE; ++i) {
      scalar_t out_val;
      from_float(out_val, vals[i] * inv_norm);
      y_row[lane + i * 32] = out_val;
    }
  }
}

#ifdef USE_ROCM
template <int ROWS_PER_BLOCK>
__launch_bounds__(ROWS_PER_BLOCK * 32, 4)
__global__ void l2norm_fwd_kernel_d128_bf16(
    at::BFloat16* __restrict__ y,
    const at::BFloat16* __restrict__ x,
    float eps,
    int64_t num_rows) {
  constexpr int D = 128;
  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * ROWS_PER_BLOCK + warp_id;
  const bool valid = row < num_rows;

  float vals[4] = {0.f, 0.f, 0.f, 0.f};

  if (valid) {
    const __hip_bfloat16* x_row = reinterpret_cast<const __hip_bfloat16*>(x + row * D);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      vals[i] = __bfloat162float(x_row[lane + i * 32]);
    }
  }

  float sum_sq = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    sum_sq += vals[i] * vals[i];
  }
  sum_sq = warp_reduce_sum(sum_sq);
  const float inv_norm = rsqrtf(sum_sq + eps);

  if (valid) {
    __hip_bfloat16* y_row = reinterpret_cast<__hip_bfloat16*>(y + row * D);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      y_row[lane + i * 32] = __float2bfloat16(vals[i] * inv_norm);
    }
  }
}
#endif

template <typename scalar_t>
void l2norm_fwd_launcher_orig(
    scalar_t* y,
    const scalar_t* x,
    int64_t num_rows,
    int64_t D,
    float eps,
    cudaStream_t stream) {
  const int threads = static_cast<int>(std::min<int64_t>(kMaxThreadsPerBlock, (D + 31) / 32 * 32));
  const dim3 grid(static_cast<unsigned int>(num_rows));
  const dim3 block(threads);
  l2norm_fwd_kernel_orig<scalar_t><<<grid, block, 0, stream>>>(y, x, D, eps, num_rows);
}

template <typename scalar_t>
void l2norm_fwd_launcher_opt(
    scalar_t* y,
    const scalar_t* x,
    int64_t num_rows,
    int64_t D,
    float eps,
    cudaStream_t stream) {
  if (D == 128 && num_rows >= kOptRowsThreshold) {
    constexpr int ROWS_PER_BLOCK = 16;
    const int threads = ROWS_PER_BLOCK * 32;
    const int64_t grid_x = (num_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    const dim3 grid(static_cast<unsigned int>(grid_x));
    const dim3 block(threads);
    l2norm_fwd_kernel_tiled_d128<scalar_t, ROWS_PER_BLOCK>
        <<<grid, block, 0, stream>>>(y, x, eps, num_rows);
  } else if (D <= 256 && num_rows >= kOptRowsThreshold) {
    constexpr int ROWS_PER_BLOCK = 16;
    constexpr int MAX_ELEMS = 8;
    const int threads = ROWS_PER_BLOCK * 32;
    const int64_t grid_x = (num_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    const dim3 grid(static_cast<unsigned int>(grid_x));
    const dim3 block(threads);
    l2norm_fwd_kernel_tiled<scalar_t, ROWS_PER_BLOCK, MAX_ELEMS>
        <<<grid, block, 0, stream>>>(y, x, D, eps, num_rows);
  } else {
    l2norm_fwd_launcher_orig<scalar_t>(y, x, num_rows, D, eps, stream);
  }
}

#ifdef USE_ROCM
template <>
void l2norm_fwd_launcher_opt<at::BFloat16>(
    at::BFloat16* y,
    const at::BFloat16* x,
    int64_t num_rows,
    int64_t D,
    float eps,
    cudaStream_t stream) {
  if (D == 128 && num_rows >= kOptRowsThreshold) {
    constexpr int ROWS_PER_BLOCK = 16;
    const int threads = ROWS_PER_BLOCK * 32;
    const int64_t grid_x = (num_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    const dim3 grid(static_cast<unsigned int>(grid_x));
    const dim3 block(threads);
    l2norm_fwd_kernel_d128_bf16<ROWS_PER_BLOCK><<<grid, block, 0, stream>>>(y, x, eps, num_rows);
  } else if (D <= 256 && num_rows >= kOptRowsThreshold) {
    constexpr int ROWS_PER_BLOCK = 16;
    constexpr int MAX_ELEMS = 8;
    const int threads = ROWS_PER_BLOCK * 32;
    const int64_t grid_x = (num_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    const dim3 grid(static_cast<unsigned int>(grid_x));
    const dim3 block(threads);
    l2norm_fwd_kernel_tiled<at::BFloat16, ROWS_PER_BLOCK, MAX_ELEMS>
        <<<grid, block, 0, stream>>>(y, x, D, eps, num_rows);
  } else {
    l2norm_fwd_launcher_orig<at::BFloat16>(y, x, num_rows, D, eps, stream);
  }
}
#endif

}  // namespace

at::Tensor l2norm(at::Tensor& input, double eps) {
  CHECK_CUDA(input);
  TORCH_CHECK(input.stride(-1) == 1, "l2norm: input last dimension must be contiguous");

  const auto orig_sizes = input.sizes();
  const int64_t D = orig_sizes.back();
  const int64_t num_rows = input.numel() / D;
  auto x = input.view({num_rows, D});
  at::Tensor y = at::empty_like(x);

  const c10::cuda::OptionalCUDAGuard device_guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (input.scalar_type()) {
    case at::ScalarType::Float:
      l2norm_fwd_launcher_opt<float>(
          y.data_ptr<float>(), x.data_ptr<float>(), num_rows, D, static_cast<float>(eps), stream);
      break;
    case at::ScalarType::Half:
      l2norm_fwd_launcher_opt<at::Half>(
          y.data_ptr<at::Half>(), x.data_ptr<at::Half>(), num_rows, D, static_cast<float>(eps), stream);
      break;
    case at::ScalarType::BFloat16:
      l2norm_fwd_launcher_opt<at::BFloat16>(
          y.data_ptr<at::BFloat16>(),
          x.data_ptr<at::BFloat16>(),
          num_rows,
          D,
          static_cast<float>(eps),
          stream);
      break;
    default:
      TORCH_CHECK(false, "l2norm: unsupported dtype ", input.scalar_type());
  }

  return y.view(orig_sizes);
}
