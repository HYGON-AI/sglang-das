// Fused C4 indexer Q: trailing RoPE, 128-point Hadamard, and INT8 quantization.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

constexpr uint32_t kWarpThreads = 32;
constexpr uint32_t kBlockSize = 128;
constexpr uint32_t kWarpsPerBlock = kBlockSize / kWarpThreads;

__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
  for (int offset = kWarpThreads / 2; offset > 0; offset >>= 1) {
    value = fmaxf(__shfl_xor(value, offset, kWarpThreads), value);
  }
  return value;
}

__device__ __forceinline__ int8_t round_clamp_int8(float value) {
  value = fminf(fmaxf(value, -127.0f), 127.0f);
  const float rounded = value >= 0.0f ? floorf(value + 0.5f) : ceilf(value - 0.5f);
  return static_cast<int8_t>(rounded);
}

__global__ __launch_bounds__(kBlockSize, 16) void fused_q_indexer_rope_hadamard_quant_int8_kernel(
    const __hip_bfloat16* __restrict__ q_input,
    int8_t* __restrict__ q_output,
    const __hip_bfloat16* __restrict__ weight,
    float* __restrict__ weights_out,
    float weight_scale,
    const float* __restrict__ freqs_cis,
    const int32_t* __restrict__ positions,
    uint32_t batch_size,
    uint32_t num_heads) {
  constexpr int kHeadDim = 128;
  constexpr int kRopeDim = 64;
  constexpr int kVecSize = 4;
  constexpr uint32_t kRopeSize = kRopeDim / kVecSize;
  static_assert(kHeadDim == kWarpThreads * kVecSize);

  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t lane_id = threadIdx.x % kWarpThreads;
  const uint32_t work_id = blockIdx.x * kWarpsPerBlock + warp_id;
  const bool is_rope_lane = lane_id >= kWarpThreads - kRopeSize;
  const uint32_t total_works = batch_size * num_heads;
  if (work_id >= total_works) return;

  const uint32_t batch_id = work_id / num_heads;
  const auto* input_row = q_input + work_id * kHeadDim;
  const int32_t position = positions[batch_id];
  const float* freq_row = freqs_cis + static_cast<int64_t>(position) * kRopeDim;

  float data[kVecSize];
  float freq[kVecSize];
  const __hip_bfloat162* input_vec = reinterpret_cast<const __hip_bfloat162*>(
      input_row + lane_id * kVecSize);
  const __hip_bfloat162 input_01 = input_vec[0];
  const __hip_bfloat162 input_23 = input_vec[1];
  data[0] = __bfloat162float(__low2bfloat16(input_01));
  data[1] = __bfloat162float(__high2bfloat16(input_01));
  data[2] = __bfloat162float(__low2bfloat16(input_23));
  data[3] = __bfloat162float(__high2bfloat16(input_23));

  if (is_rope_lane) {
    const float* freq_row_lane = freq_row + (lane_id - (kWarpThreads - kRopeSize)) * kVecSize;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) freq[i] = freq_row_lane[i];

    const float x_real = data[0];
    const float x_imag = data[1];
    const float y_real = data[2];
    const float y_imag = data[3];
    const float fxr = freq[0];
    const float fxi = freq[1];
    const float fyr = freq[2];
    const float fyi = freq[3];
    data[0] = x_real * fxr - x_imag * fxi;
    data[1] = x_real * fxi + x_imag * fxr;
    data[2] = y_real * fyr - y_imag * fyi;
    data[3] = y_real * fyi + y_imag * fyr;
  }

  // Two local butterfly stages followed by five logical-warp stages.
  {
    const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
    data[0] = a0 + a1;
    data[1] = a0 - a1;
    data[2] = a2 + a3;
    data[3] = a2 - a3;
  }
  {
    const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
    data[0] = a0 + a2;
    data[1] = a1 + a3;
    data[2] = a0 - a2;
    data[3] = a1 - a3;
  }
#pragma unroll
  for (uint32_t mask = 1; mask < kWarpThreads; mask <<= 1) {
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const float other = __shfl_xor(data[i], mask, kWarpThreads);
      data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
    }
  }
  const float hadamard_scale = rsqrtf(static_cast<float>(kHeadDim));
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) data[i] *= hadamard_scale;

  float local_max = fabsf(data[0]);
#pragma unroll
  for (int i = 1; i < kVecSize; ++i) local_max = fmaxf(local_max, fabsf(data[i]));
  const float abs_max = warp_reduce_max(local_max);
  const float scale = fmaxf(1e-4f, abs_max) / 127.0f;
  const float inv_scale = 1.0f / scale;
  int8_t* output_row = q_output + work_id * kHeadDim;
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    output_row[lane_id * kVecSize + i] = round_clamp_int8(data[i] * inv_scale);
  }
  if (lane_id == 0) {
    const float weight_value = __bfloat162float(weight[work_id]);
    weights_out[work_id] = weight_value * weight_scale * scale;
  }
}

}  // namespace

static void fused_q_int8_host(
    torch::Tensor q_input,
    torch::Tensor q_output,
    torch::Tensor weight,
    torch::Tensor weights_out,
    double weight_scale,
    torch::Tensor freqs_real,
    torch::Tensor positions) {
  TORCH_CHECK(q_input.is_cuda(), "q_input must be a CUDA/HIP tensor");
  TORCH_CHECK(q_input.scalar_type() == torch::kBFloat16, "q_input must be BF16");
  TORCH_CHECK(q_input.dim() == 3 && q_input.size(2) == 128,
              "q_input must have shape (B, H, 128)");
  TORCH_CHECK(q_input.is_contiguous(), "q_input must be contiguous");
  TORCH_CHECK(q_output.device() == q_input.device(), "q_output device mismatch");
  TORCH_CHECK(q_output.scalar_type() == torch::kChar, "q_output must be int8");
  TORCH_CHECK(q_output.sizes() == q_input.sizes(), "q_output shape mismatch");
  TORCH_CHECK(q_output.is_contiguous(), "q_output must be contiguous");
  TORCH_CHECK(weight.device() == q_input.device(), "weight device mismatch");
  TORCH_CHECK(weight.scalar_type() == torch::kBFloat16, "weight must be BF16");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(weight.numel() == q_input.numel() / q_input.size(2),
              "weight must contain one value per query head");
  TORCH_CHECK(weights_out.device() == q_input.device(), "weights_out device mismatch");
  TORCH_CHECK(weights_out.scalar_type() == torch::kFloat, "weights_out must be float32");
  TORCH_CHECK(weights_out.numel() == weight.numel(), "weights_out shape mismatch");
  TORCH_CHECK(weights_out.is_contiguous(), "weights_out must be contiguous");
  TORCH_CHECK(freqs_real.device() == q_input.device(), "freqs device mismatch");
  TORCH_CHECK(freqs_real.scalar_type() == torch::kFloat && freqs_real.dim() == 2 &&
                  freqs_real.size(1) == 64,
              "freqs_real must have shape (max_position, 64) and be float32");
  TORCH_CHECK(freqs_real.is_contiguous(), "freqs_real must be contiguous");
  TORCH_CHECK(positions.device() == q_input.device(), "positions device mismatch");
  TORCH_CHECK(positions.scalar_type() == torch::kInt, "positions must be int32");
  TORCH_CHECK(positions.dim() == 1 && positions.size(0) == q_input.size(0),
              "positions must have shape (B,)");
  TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");

  const uint32_t batch_size = static_cast<uint32_t>(q_input.size(0));
  const uint32_t num_heads = static_cast<uint32_t>(q_input.size(1));
  if (batch_size == 0 || num_heads == 0) return;

  const uint32_t total_works = batch_size * num_heads;
  const uint32_t blocks = (total_works + kWarpsPerBlock - 1) / kWarpsPerBlock;
  auto stream = at::cuda::getCurrentCUDAStream();
  hipLaunchKernelGGL(
      fused_q_indexer_rope_hadamard_quant_int8_kernel,
      dim3(blocks),
      dim3(kBlockSize),
      0,
      stream,
      reinterpret_cast<const __hip_bfloat16*>(q_input.data_ptr()),
      q_output.data_ptr<int8_t>(),
      reinterpret_cast<const __hip_bfloat16*>(weight.data_ptr()),
      weights_out.data_ptr<float>(),
      static_cast<float>(weight_scale),
      freqs_real.data_ptr<float>(),
      positions.data_ptr<int32_t>(),
      batch_size,
      num_heads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("fused_q_int8", &fused_q_int8_host, "Fused C4 indexer Q INT8");
}
