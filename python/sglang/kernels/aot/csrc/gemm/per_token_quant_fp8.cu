#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Float8_e4m3fn.h>

#include <cmath>
#include <limits>

#include "utils.h"

#ifdef USE_ROCM
namespace fp8_vec = sgl_hip;
#else
#include <flashinfer/vec_dtypes.cuh>
namespace fp8_vec = flashinfer;
#endif

using FP8_TYPE = c10::Float8_e4m3fn;
C10_HOST_DEVICE constexpr auto FP8_E4M3_MAX = std::numeric_limits<FP8_TYPE>::max();

static constexpr int kWarpSize = 32;
static constexpr int DEFAULT_SHARED_MEM_THRESHOLD_KB = 48;  // Default shared memory quota in KB

// ---------------------------------------------------------------------------
// 1. Warp‑local with configurable shared memory
//    • One warp handles one token.
//    • Eight tokens per 256‑thread CTA.
//    • Shared memory usage is configurable via template parameter.
// ---------------------------------------------------------------------------
template <typename T, typename DST_DTYPE, int kTokensPerCTA = 8, int kVecSize = 16, bool USE_SMEM = true>
__global__ void per_token_quant_fp8_kernel(
    const T* __restrict__ input,
    DST_DTYPE* __restrict__ output_q,
    float* __restrict__ output_s,
    const int64_t hidden_dim,
    const int64_t num_tokens) {
  const int warp_id = threadIdx.x / kWarpSize;        // 0‑7  (8 warps)
  const int lane_id = threadIdx.x & (kWarpSize - 1);  // 0‑31
  const int token_id = blockIdx.x * kTokensPerCTA + warp_id;
  if (token_id >= num_tokens) return;

  // Global tensors for this token
  const T* token_input = input + token_id * hidden_dim;
  DST_DTYPE* token_output = output_q + token_id * hidden_dim;
  float* token_scale = output_s + token_id;

  extern __shared__ char smem_buffer[];
  const int smem_padding = 32;  // Pad to bank boundary (32 banks * 4 bytes = 128 bytes)
  const int warp_smem_stride = (hidden_dim * sizeof(T) + smem_padding - 1) / smem_padding * smem_padding;
  const int warp_smem_offset = warp_id * warp_smem_stride;
  T* shared_input = reinterpret_cast<T*>(smem_buffer + warp_smem_offset);

  //
  // Pass-1: Load data and compute max_value
  //
  float max_value = 0.f;
  using vec_t = fp8_vec::vec_t<T, kVecSize>;
  const int32_t num_vec_elems = hidden_dim / kVecSize;

  for (int32_t i = lane_id; i < num_vec_elems; i += kWarpSize) {
    vec_t input_vec;
    input_vec.cast_load(token_input + i * kVecSize);

    // Store to shared memory if USE_SMEM=true
    if constexpr (USE_SMEM) {
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        shared_input[i * kVecSize + j] = input_vec[j];
      }
    }

    // Compute max value in parallel
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      max_value = fmaxf(max_value, fabsf(castToFloat(input_vec[j])));
    }
  }

  // Ensure all threads in the warp have finished writing to shared memory
  if constexpr (USE_SMEM) {
    __syncwarp();
  }

  float warp_max = warpReduceMax(max_value);

  // NOTE: one CTA has multiple warps (each warp handles one token), so `scale`
  // must be per-warp/per-thread (register) instead of a single shared variable.
  const float scale = warp_max / FP8_E4M3_MAX;
  // Broadcast scale
  if (lane_id == 0) {
    token_scale[0] = scale;
  }
  const float scale_inv = (scale == 0.f) ? 0.f : 1.0f / scale;

  //
  // Pass-2: Quantize and write back
  //
  for (int i = lane_id; i < num_vec_elems; i += kWarpSize) {
    vec_t input_vec;

    if constexpr (USE_SMEM) {
      // Load from shared memory
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        input_vec[j] = shared_input[i * kVecSize + j];
      }
    } else {
      // Reload from global memory
      input_vec.cast_load(token_input + i * kVecSize);
    }

    DST_DTYPE output_arr[kVecSize];
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      float val = castToFloat(input_vec[j]) * scale_inv;
      val = fmaxf(fminf(val, FP8_E4M3_MAX), -FP8_E4M3_MAX);
      output_arr[j] = static_cast<DST_DTYPE>(val);
    }
    if constexpr (kVecSize == 16) {
      *(uint4*)(token_output + i * kVecSize) = *(uint4*)output_arr;
    } else {
      // Use element-wise copy for vector size 8 to ensure correctness
      for (int k = 0; k < kVecSize; ++k) {
        token_output[i * kVecSize + k] = output_arr[k];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 2.  Baseline kernel (1 token / CTA, CUB block reduce)
// ---------------------------------------------------------------------------
template <typename T, typename DST_DTYPE, int kVecSize = 16>
__global__ void per_token_quant_fp8_small_batch_kernel(
    const T* __restrict__ input,
    DST_DTYPE* __restrict__ output_q,
    float* __restrict__ output_s,
    const int64_t hidden_dim,
    const int64_t num_tokens) {
  const int token_idx = blockIdx.x;
  if (token_idx >= num_tokens) return;

  const int tid = threadIdx.x;
  const int block_dim = blockDim.x;

  const T* token_input = input + token_idx * hidden_dim;
  DST_DTYPE* token_output = output_q + token_idx * hidden_dim;

  float max_value = 0.0f;

  // Use template parameter for vector size
  using vec_t = fp8_vec::vec_t<T, kVecSize>;
  const int32_t num_vec_elems = hidden_dim / kVecSize;

  // Find max using vectorized loads
  for (int32_t i = tid; i < num_vec_elems; i += block_dim) {
    vec_t input_vec;
    input_vec.cast_load(token_input + i * kVecSize);

#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      float val = castToFloat(input_vec[j]);
      max_value = fmaxf(max_value, fabsf(val));
    }
  }

  // Do not use utils.h:blockReduceMax here. On gfx9 it uses a physical
  // wavefront size of 64 while warpReduceMax intentionally reduces logical
  // 32-lane groups, which can drop half of a wavefront for this kernel.
  // A CTA-wide shared-memory reduction is portable across CUDA warp32 and
  // ROCm wave32/wave64 execution.
  __shared__ float block_max[256];
  block_max[tid] = max_value;
  __syncthreads();
#pragma unroll
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      block_max[tid] = fmaxf(block_max[tid], block_max[tid + stride]);
    }
    __syncthreads();
  }
  max_value = block_max[0];

  __shared__ float scale;
  if (tid == 0) {
    scale = max_value / FP8_E4M3_MAX;
    output_s[token_idx] = scale;
  }
  __syncthreads();

  const float scale_inv = (scale == 0.f) ? 0.f : 1.0f / scale;

  // Quantize using vectorized loads
  for (int32_t i = tid; i < num_vec_elems; i += block_dim) {
    vec_t input_vec;
    input_vec.cast_load(token_input + i * kVecSize);

    DST_DTYPE output_arr[kVecSize];
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      float val = fmaxf(fminf(castToFloat(input_vec[j]) * scale_inv, FP8_E4M3_MAX), -FP8_E4M3_MAX);
      output_arr[j] = static_cast<DST_DTYPE>(val);
    }

    if constexpr (kVecSize == 16) {
      *(uint4*)(token_output + i * kVecSize) = *(uint4*)output_arr;
    } else {
      // Use element-wise copy for vector size 8 to ensure correctness
      for (int k = 0; k < kVecSize; ++k) {
        token_output[i * kVecSize + k] = output_arr[k];
      }
    }
  }
}

template <bool USE_SMEM, typename scalar_t, int TOKENS_PER_CTA>
static inline void launch_per_token_quant_fp8_warp_kernel(
    const dim3& grid,
    const dim3& block,
    size_t dynamicSmemSz,
    cudaStream_t stream,
    bool use_vec16,
    bool use_vec8,
    torch::Tensor input,
    torch::Tensor output_q,
    torch::Tensor output_s,
    const int64_t hidden_dim,
    const int64_t num_tokens) {
  const size_t smem_size = USE_SMEM ? dynamicSmemSz : 0;

  if (use_vec16) {
    per_token_quant_fp8_kernel<scalar_t, FP8_TYPE, TOKENS_PER_CTA, 16, USE_SMEM><<<grid, block, smem_size, stream>>>(
        static_cast<const scalar_t*>(input.data_ptr()),
        static_cast<FP8_TYPE*>(output_q.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        hidden_dim,
        num_tokens);
  } else if (use_vec8) {
    per_token_quant_fp8_kernel<scalar_t, FP8_TYPE, TOKENS_PER_CTA, 8, USE_SMEM><<<grid, block, smem_size, stream>>>(
        static_cast<const scalar_t*>(input.data_ptr()),
        static_cast<FP8_TYPE*>(output_q.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        hidden_dim,
        num_tokens);
  } else {
    per_token_quant_fp8_kernel<scalar_t, FP8_TYPE, TOKENS_PER_CTA, 4, USE_SMEM><<<grid, block, smem_size, stream>>>(
        static_cast<const scalar_t*>(input.data_ptr()),
        static_cast<FP8_TYPE*>(output_q.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        hidden_dim,
        num_tokens);
  }
}

void sgl_per_token_quant_fp8(torch::Tensor input, torch::Tensor output_q, torch::Tensor output_s) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "input must be a contiguous CUDA tensor");
  TORCH_CHECK(output_q.is_cuda() && output_q.is_contiguous(), "output_q must be a contiguous CUDA tensor");
  TORCH_CHECK(output_s.is_cuda() && output_s.is_contiguous(), "output_s must be a contiguous CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "input must be a 2D tensor, but got ", input.dim(), " dimensions");
  TORCH_CHECK(output_q.sizes() == input.sizes(), "output_q shape must match input shape");
  TORCH_CHECK(
      output_s.numel() == input.size(0),
      "output_s must contain one scale per token, expected ",
      input.size(0),
      ", got ",
      output_s.numel());
  TORCH_CHECK(output_q.scalar_type() == at::ScalarType::Float8_e4m3fn, "output_q must be float8_e4m3fn on this device");
  TORCH_CHECK(output_s.scalar_type() == torch::kFloat32, "output_s must be float32");
  const auto input_sizes = input.sizes();
  const int64_t num_tokens = input_sizes[0];
  const int64_t hidden_dim = input_sizes[1];
  TORCH_CHECK(hidden_dim % 4 == 0, "Hidden dimension must be divisible by 4, but got ", hidden_dim);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int TOKENS_PER_CTA = 8;
  const bool use_warp_kernel = (num_tokens >= sm_count * 2 * TOKENS_PER_CTA);
  const bool use_vec16 = (hidden_dim % 16 == 0);
  const bool use_vec8 = (hidden_dim % 8 == 0);

  const int sizeof_T = input.scalar_type() == torch::kFloat16 ? 2 : (input.scalar_type() == torch::kBFloat16 ? 2 : 4);
  const int smem_padding = 32;  // Pad to bank boundary to avoid conflicts
  const int warp_smem_stride = (hidden_dim * sizeof_T + smem_padding - 1) / smem_padding * smem_padding;
  const size_t dynamicSmemSz = warp_smem_stride * TOKENS_PER_CTA;

  bool use_smem = (hidden_dim < 2048);

  if (dynamicSmemSz >= DEFAULT_SHARED_MEM_THRESHOLD_KB * 1024) {
    use_smem = false;  // Disable shared memory if >= 48KB to avoid allocation failures
  }

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), scalar_t, [&] {
    if (use_warp_kernel) {
      // -------- warp‑local ---------------------------------------------------
      constexpr int THREADS = TOKENS_PER_CTA * kWarpSize;
      dim3 grid((num_tokens + TOKENS_PER_CTA - 1) / TOKENS_PER_CTA);
      dim3 block(THREADS);

      if (use_smem) {
        launch_per_token_quant_fp8_warp_kernel</*USE_SMEM=*/true, scalar_t, TOKENS_PER_CTA>(
            grid, block, dynamicSmemSz, stream, use_vec16, use_vec8, input, output_q, output_s, hidden_dim, num_tokens);
      } else {
        launch_per_token_quant_fp8_warp_kernel</*USE_SMEM=*/false, scalar_t, TOKENS_PER_CTA>(
            grid, block, dynamicSmemSz, stream, use_vec16, use_vec8, input, output_q, output_s, hidden_dim, num_tokens);
      }
    } else {
      // -------- baseline -----------------------------------------------------
      constexpr int THREADS = 256;
      dim3 grid(num_tokens);
      dim3 block(THREADS);

      if (use_vec16) {
        per_token_quant_fp8_small_batch_kernel<scalar_t, FP8_TYPE, 16><<<grid, block, 0, stream>>>(
            static_cast<const scalar_t*>(input.data_ptr()),
            static_cast<FP8_TYPE*>(output_q.data_ptr()),
            static_cast<float*>(output_s.data_ptr()),
            hidden_dim,
            num_tokens);
      } else if (use_vec8) {
        per_token_quant_fp8_small_batch_kernel<scalar_t, FP8_TYPE, 8><<<grid, block, 0, stream>>>(
            static_cast<const scalar_t*>(input.data_ptr()),
            static_cast<FP8_TYPE*>(output_q.data_ptr()),
            static_cast<float*>(output_s.data_ptr()),
            hidden_dim,
            num_tokens);
      } else {
        per_token_quant_fp8_small_batch_kernel<scalar_t, FP8_TYPE, 4><<<grid, block, 0, stream>>>(
            static_cast<const scalar_t*>(input.data_ptr()),
            static_cast<FP8_TYPE*>(output_q.data_ptr()),
            static_cast<float*>(output_s.data_ptr()),
            hidden_dim,
            num_tokens);
      }
    }
    return true;
  });
}
