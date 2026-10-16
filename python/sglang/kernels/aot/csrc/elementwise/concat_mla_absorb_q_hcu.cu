// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int64_t kALastDim = 512;
constexpr int64_t kBLastDim = 64;
constexpr int64_t kOutLastDim = kALastDim + kBLastDim;
constexpr int64_t kBf16PerVec = 8;
constexpr int64_t kAVecsPerRow = kALastDim / kBf16PerVec;
constexpr int64_t kBVecsPerRow = kBLastDim / kBf16PerVec;
constexpr int kWaveSize = 64;
constexpr int kThreads = 256;
constexpr int kWavesPerBlock = kThreads / kWaveSize;
constexpr int kRowsPerWave = 8;
constexpr int kRowsPerBlock = kWavesPerBlock * kRowsPerWave;

static_assert(sizeof(uint4) == 16, "uint4 must be a 16-byte vector");
static_assert(kAVecsPerRow == kWaveSize, "one wave must cover one a row");
static_assert(kWavesPerBlock == 4, "one block must contain four waves");

__global__ void concat_mla_absorb_q_hcu_kernel(
    const at::BFloat16* __restrict__ a,
    const at::BFloat16* __restrict__ b,
    at::BFloat16* __restrict__ out,
    int64_t num_rows,
    int64_t dim_1,
    int64_t a_stride_0,
    int64_t a_stride_1,
    int64_t b_stride_0,
    int64_t b_stride_1,
    int64_t out_stride_0,
    int64_t out_stride_1) {
  const int lane = threadIdx.x & (kWaveSize - 1);
  const int wave = threadIdx.x / kWaveSize;
  const int64_t first_row = static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + wave * kRowsPerWave;

#pragma unroll
  for (int row_in_wave = 0; row_in_wave < kRowsPerWave; ++row_in_wave) {
    const int64_t row = first_row + row_in_wave;
    if (row >= num_rows) return;

    const int64_t idx_0 = row / dim_1;
    const int64_t idx_1 = row - idx_0 * dim_1;
    const auto* a_row = a + idx_0 * a_stride_0 + idx_1 * a_stride_1;
    auto* out_row = out + idx_0 * out_stride_0 + idx_1 * out_stride_1;
    reinterpret_cast<uint4*>(out_row)[lane] = reinterpret_cast<const uint4*>(a_row)[lane];

    if (lane < kBVecsPerRow) {
      const auto* b_row = b + idx_0 * b_stride_0 + idx_1 * b_stride_1;
      reinterpret_cast<uint4*>(out_row)[kAVecsPerRow + lane] = reinterpret_cast<const uint4*>(b_row)[lane];
    }
  }
}

void check_concat_tensor(const at::Tensor& tensor, const char* name, int64_t expected_last_dim) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be on an HCU device");
  TORCH_CHECK(tensor.dim() == 3, name, " must be a 3D tensor");
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::BFloat16, name, " must have dtype torch.bfloat16");
  TORCH_CHECK(tensor.size(2) == expected_last_dim, name, ".size(2) must be ", expected_last_dim);
  TORCH_CHECK(tensor.stride(2) == 1, name, " must be contiguous in its last dimension");
  TORCH_CHECK(
      tensor.stride(0) % kBf16PerVec == 0 && tensor.stride(1) % kBf16PerVec == 0,
      name,
      " row addresses must be 16-byte aligned");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignof(uint4) == 0,
      name,
      " base address must be 16-byte aligned");
}

}  // namespace

void concat_mla_absorb_q(at::Tensor a, at::Tensor b, at::Tensor out) {
  check_concat_tensor(a, "a", kALastDim);
  check_concat_tensor(b, "b", kBLastDim);
  check_concat_tensor(out, "out", kOutLastDim);
  TORCH_CHECK(a.device() == b.device() && a.device() == out.device(), "all tensors must be on the same device");
  TORCH_CHECK(
      a.size(0) == b.size(0) && a.size(0) == out.size(0) && a.size(1) == b.size(1) && a.size(1) == out.size(1),
      "a, b, and out leading dimensions must match");

  const int64_t num_rows = a.size(0) * a.size(1);
  if (num_rows == 0) return;

  const int64_t num_blocks = (num_rows + kRowsPerBlock - 1) / kRowsPerBlock;
  TORCH_CHECK(num_blocks <= std::numeric_limits<uint32_t>::max(), "concat_mla_absorb_q input is too large");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  concat_mla_absorb_q_hcu_kernel<<<static_cast<uint32_t>(num_blocks), kThreads, 0, stream>>>(
      static_cast<const at::BFloat16*>(a.data_ptr()),
      static_cast<const at::BFloat16*>(b.data_ptr()),
      static_cast<at::BFloat16*>(out.data_ptr()),
      num_rows,
      a.size(1),
      a.stride(0),
      a.stride(1),
      b.stride(0),
      b.stride(1),
      out.stride(0),
      out.stride(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
