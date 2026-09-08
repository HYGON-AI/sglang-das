// Copyright (c) 2026 gencheng liu
// SPDX-License-Identifier: MIT
//
// The MMAC main loop is instantiated from the MIT-licensed AITER headers that
// ship in the Hygon SGLang image.  This wrapper removes AITER's TP-only
// stride_bse == 256 dispatch restriction and presents already-dispatched
// DeepEP rows as a top-k=1 sorted-token problem.

#include <ATen/ATen.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <cstdint>

// AITER's monolithic moe_c.hip includes a WNA16 header before the W4A8
// implementation.  Besides WNA16 helpers, that header supplies the generic
// vector aliases (vec<T, N>, vec_element_8, ... ) consumed by W4A8.  Include
// it explicitly because this translation unit intentionally instantiates only
// the W4A8 path.
// clang-format off: moe_wna16_utils.h defines vec<T, N> for the W4A8 header.
#include "moe_wna16_utils.h"
#include "moe_w4a8_opt_hip.h"
// clang-format on

namespace {

constexpr int64_t kMaskedBlockM = 16;

class AiterHipDeviceGuard {
 public:
  explicit AiterHipDeviceGuard(int device) : previous_device_(-1), changed_(false) {
    TORCH_CHECK(hipGetDevice(&previous_device_) == hipSuccess, "hipGetDevice failed");
    if (previous_device_ != device) {
      TORCH_CHECK(hipSetDevice(device) == hipSuccess, "hipSetDevice failed for device ", device);
      changed_ = true;
    }
  }

  ~AiterHipDeviceGuard() {
    if (changed_) {
      (void)hipSetDevice(previous_device_);
    }
  }

 private:
  int previous_device_;
  bool changed_;
};

__global__ void prepare_deepep_metadata_kernel(
    const int32_t* __restrict__ m_indices,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ block_experts,
    int32_t* __restrict__ num_tokens_post_pad,
    uint32_t m,
    uint32_t block_m) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < m) {
    sorted_token_ids[index] = static_cast<int32_t>(index);
  }
  if (index < m / block_m) {
    // DeepEP normal dispatch pads every expert to 256 rows, so a 32-row
    // or 64-row MMAC tile cannot cross an expert boundary.
    block_experts[index] = m_indices[index * block_m];
  }
  if (index == 0) {
    *num_tokens_post_pad = static_cast<int32_t>(m);
  }
}

// DeepEP low-latency dispatch stores a fixed-capacity [E, T, K] tensor and a
// device-side valid-row count for every expert.  Compact those already sorted
// rows into AITER's metadata format without a host synchronization.  A single
// workgroup is intentional: E <= 256 and it gives us a cheap device prefix
// sum while remaining CUDA/HIP graph capturable.
__global__ void prepare_deepep_masked_metadata_kernel(
    const int32_t* __restrict__ masked_m,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ block_experts,
    int32_t* __restrict__ num_tokens_post_pad,
    uint32_t num_experts,
    uint32_t rows_per_expert,
    uint32_t total_rows,
    uint32_t metadata_rows) {
  __shared__ uint32_t block_offsets[257];
  const uint32_t tid = threadIdx.x;

  for (uint32_t i = tid; i < metadata_rows; i += blockDim.x) {
    sorted_token_ids[i] = static_cast<int32_t>(total_rows);
  }
  for (uint32_t i = tid; i < metadata_rows / kMaskedBlockM; i += blockDim.x) {
    block_experts[i] = -1;
  }

  if (tid < num_experts) {
    const int32_t raw_count = masked_m[tid];
    const uint32_t count = raw_count <= 0 ? 0U : min(static_cast<uint32_t>(raw_count), rows_per_expert);
    block_offsets[tid] = (count + kMaskedBlockM - 1) / kMaskedBlockM;
  }
  __syncthreads();

  if (tid == 0) {
    uint32_t prefix = 0;
    for (uint32_t expert = 0; expert < num_experts; ++expert) {
      const uint32_t blocks = block_offsets[expert];
      block_offsets[expert] = prefix;
      prefix += blocks;
    }
    block_offsets[num_experts] = prefix;
    const uint32_t padded_rows = prefix * kMaskedBlockM;
    *num_tokens_post_pad = padded_rows <= metadata_rows ? static_cast<int32_t>(padded_rows) : 0;
  }
  __syncthreads();

  if (tid < num_experts) {
    const int32_t raw_count = masked_m[tid];
    const uint32_t count = raw_count <= 0 ? 0U : min(static_cast<uint32_t>(raw_count), rows_per_expert);
    const uint32_t first_block = block_offsets[tid];
    const uint32_t last_block = block_offsets[tid + 1];
    const uint32_t first_row = first_block * kMaskedBlockM;
    const uint32_t last_row = last_block * kMaskedBlockM;
    if (last_row <= metadata_rows) {
      for (uint32_t block = first_block; block < last_block; ++block) {
        block_experts[block] = static_cast<int32_t>(tid);
      }
      for (uint32_t row = 0; row < count; ++row) {
        sorted_token_ids[first_row + row] = static_cast<int32_t>(tid * rows_per_expert + row);
      }
    }
  }
}

void launch_w4a8_mmac(
    const at::Tensor& a,
    const at::Tensor& a_scale,
    const at::Tensor& weight,
    const at::Tensor& weight_scale,
    const int32_t* sorted_token_ids,
    const int32_t* block_experts,
    const int32_t* num_tokens_post_pad,
    at::Tensor& out,
    int64_t block_m,
    uint32_t sorted_token_lens = 0,
    int64_t kernel_variant = 0) {
  const auto m = static_cast<uint32_t>(a.size(0));
  const auto k = static_cast<uint32_t>(a.size(1));
  const auto n = static_cast<uint32_t>(weight.size(1));
  constexpr uint32_t kTopK = 1;
  if (sorted_token_lens == 0) {
    sorted_token_lens = m;
  }

  GemmParams_w4a8<char, bhalf_t> params(
      reinterpret_cast<const char*>(a.data_ptr<int8_t>()),
      reinterpret_cast<const char*>(weight.data_ptr<int8_t>()),
      reinterpret_cast<bhalf_t*>(out.data_ptr<at::BFloat16>()),
      const_cast<float*>(a_scale.data_ptr<float>()),
      const_cast<float*>(weight_scale.data_ptr<float>()),
      nullptr,
      sorted_token_ids,
      block_experts,
      0,
      num_tokens_post_pad,
      m,
      n,
      k,
      static_cast<uint32_t>(a_scale.stride(0)),
      a_scale.dim() == 1 ? 0U : static_cast<uint32_t>(a_scale.stride(1)),
      static_cast<uint32_t>(weight_scale.stride(0)),
      static_cast<uint32_t>(weight_scale.stride(1)),
      weight_scale.dim() == 2 ? 0U : static_cast<uint32_t>(weight_scale.stride(2)),
      sorted_token_lens,
      kTopK,
      kTopK,
      true);

  // Only the three schedules selected by the gfx936 production tuner are
  // instantiated. This keeps startup compilation and the code object small.
  switch (kernel_variant) {
    case 6:
      TORCH_CHECK(block_m == 16, "decode variant 6 requires BLOCK_M=16");
      launch_moe_w4a8_first_stage_prefill_GEMM1Nloop<1, 16, 128, 64, 16, 32, 64, 2>(params);
      return;
    case 11:
      TORCH_CHECK(block_m == 32, "prefill variant 11 requires BLOCK_M=32");
      launch_moe_w4a8_first_stage_prefill_GEMM1Nloop<2, 32, 128, 64, 32, 32, 64, 2>(params);
      return;
    case 15:
      TORCH_CHECK(block_m == 64, "prefill variant 15 requires BLOCK_M=64");
      launch_moe_w4a8_first_stage_prefill_GEMM1Nloop<4, 64, 128, 64, 64, 32, 64, 2>(params);
      return;
    default:
      TORCH_CHECK(false, "unsupported W4A8 MMAC kernel_variant=", kernel_variant);
  }
}

}  // namespace

void w4a8_mmac_contiguous_out_hip(
    const at::Tensor& a,
    const at::Tensor& a_scale,
    const at::Tensor& weight,
    const at::Tensor& weight_scale,
    const at::Tensor& m_indices,
    at::Tensor& workspace,
    at::Tensor& out,
    int64_t kernel_variant) {
  const AiterHipDeviceGuard device_guard(a.get_device());
  const auto m = static_cast<uint32_t>(a.size(0));
  if (m == 0) {
    return;
  }

  const uint32_t block_m = kernel_variant >= 13 && kernel_variant <= 15 ? 64U : 32U;
  int32_t* const workspace_ptr = workspace.data_ptr<int32_t>();
  int32_t* const sorted_token_ids = workspace_ptr;
  int32_t* const block_experts = sorted_token_ids + m;
  int32_t* const num_tokens_post_pad = block_experts + m / block_m;

  constexpr uint32_t kThreads = 256;
  const dim3 grid((m + kThreads - 1) / kThreads);
  const hipStream_t stream = at::cuda::getCurrentHIPStream();
  hipLaunchKernelGGL(
      prepare_deepep_metadata_kernel,
      grid,
      dim3(kThreads),
      0,
      stream,
      m_indices.data_ptr<int32_t>(),
      sorted_token_ids,
      block_experts,
      num_tokens_post_pad,
      m,
      block_m);
  auto error = hipGetLastError();
  TORCH_CHECK(error == hipSuccess, "DeepEP W4A8 metadata setup launch failed: ", hipGetErrorString(error));

  launch_w4a8_mmac(
      a,
      a_scale,
      weight,
      weight_scale,
      sorted_token_ids,
      block_experts,
      num_tokens_post_pad,
      out,
      block_m,
      0,
      kernel_variant);
  error = hipGetLastError();
  TORCH_CHECK(
      error == hipSuccess, "AITER-derived W4A8 DeepEP contiguous MMAC launch failed: ", hipGetErrorString(error));
}

void w4a8_mmac_masked_out_hip(
    const at::Tensor& a,
    const at::Tensor& a_scale,
    const at::Tensor& weight,
    const at::Tensor& weight_scale,
    const at::Tensor& masked_m,
    at::Tensor& workspace,
    at::Tensor& out,
    int64_t metadata_rows,
    int64_t kernel_variant) {
  const AiterHipDeviceGuard device_guard(a.get_device());
  const auto num_experts = static_cast<uint32_t>(a.size(0));
  const auto rows_per_expert = static_cast<uint32_t>(a.size(1));
  const auto k = a.size(2);
  const auto total_rows = num_experts * rows_per_expert;

  int32_t* const workspace_ptr = workspace.data_ptr<int32_t>();
  int32_t* const sorted_token_ids = workspace_ptr;
  int32_t* const block_experts = sorted_token_ids + metadata_rows;
  int32_t* const num_tokens_post_pad = block_experts + metadata_rows / kMaskedBlockM;

  const hipStream_t stream = at::cuda::getCurrentHIPStream();
  hipLaunchKernelGGL(
      prepare_deepep_masked_metadata_kernel,
      dim3(1),
      dim3(256),
      0,
      stream,
      masked_m.data_ptr<int32_t>(),
      sorted_token_ids,
      block_experts,
      num_tokens_post_pad,
      num_experts,
      rows_per_expert,
      total_rows,
      static_cast<uint32_t>(metadata_rows));
  auto error = hipGetLastError();
  TORCH_CHECK(error == hipSuccess, "DeepEP masked W4A8 metadata setup launch failed: ", hipGetErrorString(error));

  auto a_flat = a.view({static_cast<int64_t>(total_rows), k});
  auto a_scale_flat = a_scale.view({static_cast<int64_t>(total_rows), 1});
  auto out_flat = out.view({static_cast<int64_t>(total_rows), weight.size(1)});
  launch_w4a8_mmac(
      a_flat,
      a_scale_flat,
      weight,
      weight_scale,
      sorted_token_ids,
      block_experts,
      num_tokens_post_pad,
      out_flat,
      kMaskedBlockM,
      static_cast<uint32_t>(metadata_rows),
      kernel_variant);
  error = hipGetLastError();
  TORCH_CHECK(error == hipSuccess, "AITER-derived W4A8 DeepEP masked MMAC launch failed: ", hipGetErrorString(error));
}
