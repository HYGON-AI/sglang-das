// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <limits>

namespace sglang {

#if defined(USE_ROCM)

using hcu_half4_t =
    __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
using hcu_bf16_mfma_t =
    __attribute__((__vector_size__(4 * sizeof(short)))) short;
using hcu_float4_t =
    __attribute__((__vector_size__(4 * sizeof(float)))) float;
using hcu_uint32x4_t =
    std::uint32_t __attribute__((ext_vector_type(4)));

struct HcuBf16x8 {
  hcu_half4_t data[2];
};

SGL_DEVICE void hcu_mmac_bf16(
    const hcu_half4_t& reg_a,
    const hcu_half4_t& reg_b,
    hcu_float4_t& reg_c) {
#if defined(__gfx936__) || defined(__gfx928__)
  reg_c = __builtin_amdgcn_mmac_f32_16x16x16bf16(
      *reinterpret_cast<const hcu_bf16_mfma_t*>(&reg_a),
      *reinterpret_cast<const hcu_bf16_mfma_t*>(&reg_b),
      reg_c);
#elif defined(__gfx938__)
  reg_c = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(
      *reinterpret_cast<const hcu_bf16_mfma_t*>(&reg_a),
      *reinterpret_cast<const hcu_bf16_mfma_t*>(&reg_b),
      reg_c,
      false,
      false);
#endif
}

SGL_DEVICE hcu_uint32x4_t make_buffer_resource(
    const std::uint32_t* ptr) {
  hcu_uint32x4_t resource = {};
  const std::uint64_t address = reinterpret_cast<std::uint64_t>(ptr);
  resource[0] =
      __builtin_amdgcn_readfirstlane(static_cast<std::uint32_t>(address));
  resource[1] = __builtin_amdgcn_readfirstlane(
      static_cast<std::uint32_t>(address >> 32));
  resource[2] = 0x80000000u;
  resource[3] = 0x00020000u;
  return resource;
}

template <typename T>
SGL_DEVICE void async_load8(
    T* lds_base,
    std::int32_t lds_offset,
    hcu_uint32x4_t resource,
    std::int32_t gmem_offset) {
  auto* destination =
      (__attribute__((address_space(3))) int*)(lds_base + lds_offset);
  __builtin_hcu_raw_buffer_load_lds(
      resource, destination, 16, gmem_offset * sizeof(T), 0, 0, 0);
}

template <int kWarps = 4, int kNPerBlock = 16>
__global__ void gemm_nt_bf16_fp32(
    const bf16_t* __restrict__ a,
    const bf16_t* __restrict__ b,
    float* __restrict__ out,
    std::int32_t m,
    std::int32_t n,
    std::int32_t k) {
  constexpr std::int32_t kWaveSize = 64;
  const std::int32_t block_x = blockIdx.x;
  const std::int32_t block_y = blockIdx.y;
  const std::int32_t thread = threadIdx.x;
  const std::int32_t wave =
      __builtin_amdgcn_readfirstlane(thread / kWaveSize);
  const std::int32_t lane = thread % kWaveSize;
  const std::int32_t row_in_tile = lane % 16;
  const std::int32_t row_group = lane / 16;
  const std::int32_t row = block_y * 16 + row_in_tile;
  const std::int32_t col = block_x * kNPerBlock + row_in_tile;
  const std::int32_t k_offset = wave * 16 * 2 + row_group * 4 * 2;

  HcuBf16x8 a_vec = {};
  HcuBf16x8 b_vec;
  hcu_float4_t accumulator = {0, 0, 0, 0};
  const bf16_t* a_row =
      row < m ? a + row * k + k_offset : a;
  const bf16_t* b_row = b + col * k + k_offset;

  for (std::int32_t i = 0; i + 16 * 2 * wave < k;
       i += 16 * 2 * kWarps) {
    if (row < m) {
      a_vec = *reinterpret_cast<const HcuBf16x8*>(a_row + i);
    }
    b_vec = *reinterpret_cast<const HcuBf16x8*>(b_row + i);
    hcu_mmac_bf16(a_vec.data[0], b_vec.data[0], accumulator);
    hcu_mmac_bf16(a_vec.data[1], b_vec.data[1], accumulator);
  }

  extern __shared__ hcu_float4_t reduction[];
#pragma unroll
  for (std::int32_t groups = kWarps; groups > 1; groups /= 2) {
    const std::int32_t midpoint = groups / 2;
    if (row < m && wave >= midpoint && wave < groups) {
      reduction[
          (wave - midpoint) * 4 * 16 + row_in_tile * 4 + row_group] =
          accumulator;
    }
    __syncthreads();
    if (row < m && wave < midpoint) {
      const hcu_float4_t partial =
          reduction[wave * 4 * 16 + row_in_tile * 4 + row_group];
#pragma unroll
      for (std::int32_t i = 0; i < 4; ++i) {
        accumulator[i] += partial[i];
      }
    }
    __syncthreads();
  }

  if (row < m && wave == 0) {
    float* output =
        out + block_x * kNPerBlock + row * n + row_group;
#pragma unroll
    for (std::int32_t i = 0; i < 4; ++i) {
      output[i * 4] = accumulator[i];
    }
  }
}

template <
    int kWarps = 4,
    int kNPerBlock = 64,
    int kStage = 64,
    int kSplit = 4>
__global__ void gemm_nt_bf16_fp32_splitk(
    const bf16_t* __restrict__ a,
    const bf16_t* __restrict__ b,
    float* __restrict__ out,
    std::int32_t m,
    std::int32_t n,
    std::int32_t k) {
  constexpr std::int32_t kWaveSize = 64;
  constexpr std::int32_t kNPerWave = kNPerBlock / kWarps;
  const std::int32_t block_x = blockIdx.x;
  const std::int32_t block_y = blockIdx.y;
  const std::int32_t split = blockIdx.z;
  const std::int32_t thread = threadIdx.x;
  const std::int32_t wave =
      __builtin_amdgcn_readfirstlane(thread / kWaveSize);
  const std::int32_t lane = thread % kWaveSize;
  const std::int32_t row_in_tile = lane % 16;
  const std::int32_t row_group = lane / 16;
  const std::int32_t row = block_y * 16 + row_in_tile;
  const std::int32_t col_wave =
      block_x * kNPerBlock + wave * kNPerWave;
  const std::int32_t k_offset = row_group * 8;
  const std::int32_t k_per_split = k / kSplit;
  const std::int32_t k_start = split * k_per_split;
  const std::int32_t stages = k_per_split / kStage;

  __shared__ bf16_t a_shared[2][16][kStage];
  __shared__ bf16_t b_shared[2][kNPerBlock][kStage];
  const auto a_resource = make_buffer_resource(
      reinterpret_cast<const std::uint32_t*>(a));
  const auto b_resource = make_buffer_resource(
      reinterpret_cast<const std::uint32_t*>(b));
  const std::int32_t b_row_start = block_x * kNPerBlock;
  hcu_float4_t accumulator = {0, 0, 0, 0};

  auto load_stage = [&](std::int32_t k_block, std::int32_t buffer) {
    if (thread < 128) {
      const std::int32_t row_index = thread >> 3;
      const std::int32_t vector = thread & 7;
      const std::int32_t a_row = block_y * 16 + row_index;
      auto* destination =
          &a_shared[buffer][row_index][vector * 8];
      if (a_row < m) {
        async_load8(
            &a_shared[buffer][0][0],
            row_index * kStage + vector * 8,
            a_resource,
            a_row * k + k_block + vector * 8);
      } else {
        *reinterpret_cast<HcuBf16x8*>(destination) = {};
      }
    }
#pragma unroll
    for (std::int32_t j = 0; j < 2; ++j) {
      const std::int32_t index = thread + j * 256;
      const std::int32_t row_index = index >> 3;
      const std::int32_t vector = index & 7;
      async_load8(
          &b_shared[buffer][0][0],
          row_index * kStage + vector * 8,
          b_resource,
          (b_row_start + row_index) * k + k_block + vector * 8);
    }
  };

  load_stage(k_start, 0);
  __builtin_amdgcn_s_waitcnt(0xF70);
  __builtin_amdgcn_s_barrier();

  for (std::int32_t stage = 1; stage < stages; ++stage) {
    const std::int32_t next = stage & 1;
    const std::int32_t current = next ^ 1;
    const std::int32_t k_block = k_start + stage * kStage;
    load_stage(k_block, next);
#pragma unroll
    for (std::int32_t i = 0; i < kStage; i += 32) {
      const auto a_vec = *reinterpret_cast<const HcuBf16x8*>(
          &a_shared[current][row_in_tile][k_offset + i]);
      const auto b_vec = *reinterpret_cast<const HcuBf16x8*>(
          &b_shared[current][wave * kNPerWave + row_in_tile][k_offset + i]);
      hcu_mmac_bf16(a_vec.data[0], b_vec.data[0], accumulator);
      hcu_mmac_bf16(a_vec.data[1], b_vec.data[1], accumulator);
    }
    __builtin_amdgcn_s_waitcnt(0xF70);
    __builtin_amdgcn_s_barrier();
  }

  const std::int32_t current = (stages - 1) & 1;
#pragma unroll
  for (std::int32_t i = 0; i < kStage; i += 32) {
    const auto a_vec = *reinterpret_cast<const HcuBf16x8*>(
        &a_shared[current][row_in_tile][k_offset + i]);
    const auto b_vec = *reinterpret_cast<const HcuBf16x8*>(
        &b_shared[current][wave * kNPerWave + row_in_tile][k_offset + i]);
    hcu_mmac_bf16(a_vec.data[0], b_vec.data[0], accumulator);
    hcu_mmac_bf16(a_vec.data[1], b_vec.data[1], accumulator);
  }

  if (row < m) {
    float* output = out + row * n + col_wave + row_group;
#pragma unroll
    for (std::int32_t i = 0; i < 4; ++i) {
      atomicAdd(&output[i * 4], accumulator[i]);
    }
  }
}

/**
 * \brief Validate and launch the HCU BF16 x BF16 -> FP32 NT GEMM.
 *
 * \param out Contiguous FP32 output [M, N]; must be zero-initialized when
 *            the selected implementation uses inter-block split-K.
 * \param x Contiguous BF16 activation [M, K].
 * \param weight Contiguous BF16 weight [N, K].
 */
struct LinearBf16Fp32Kernel {
  static void run(
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView weight) {
    using namespace host;
    auto M = SymbolicSize{"M"};
    auto N = SymbolicSize{"N"};
    auto K = SymbolicSize{"K"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();

    TensorMatcher({M, K})
        .with_dtype<bf16_t>()
        .with_device<kDLGPU>(device)
        .verify(x);
    TensorMatcher({N, K})
        .with_dtype<bf16_t>()
        .with_device<kDLGPU>(device)
        .verify(weight);
    TensorMatcher({M, N})
        .with_dtype<float>()
        .with_device<kDLGPU>(device)
        .verify(out);

    const auto m64 = M.unwrap();
    const auto n64 = N.unwrap();
    const auto k64 = K.unwrap();
    CHECK_HOST(m64 > 0 && m64 <= 64)
        << "linear_bf16_fp32: M must be in [1, 64], got " << m64;
    CHECK_HOST(
        n64 == 256 || n64 == 512 || n64 == 1024 || n64 == 2048)
        << "linear_bf16_fp32: unsupported N " << n64;
    CHECK_HOST(k64 > 0 && k64 % 128 == 0)
        << "linear_bf16_fp32: K must be a positive multiple of 128, got "
        << k64;
    CHECK_HOST(
        m64 * k64 <= std::numeric_limits<std::int32_t>::max() &&
        n64 * k64 <= std::numeric_limits<std::int32_t>::max())
        << "linear_bf16_fp32: matrix offsets exceed int32 range";
    CHECK_HOST(
        reinterpret_cast<std::uintptr_t>(x.data_ptr()) % 16 == 0 &&
        reinterpret_cast<std::uintptr_t>(weight.data_ptr()) % 16 == 0 &&
        reinterpret_cast<std::uintptr_t>(out.data_ptr()) % 16 == 0)
        << "linear_bf16_fp32: tensor addresses must be 16-byte aligned";

    const auto m = static_cast<std::int32_t>(m64);
    const auto n = static_cast<std::int32_t>(n64);
    const auto k = static_cast<std::int32_t>(k64);
    const auto device_value = device.unwrap();
    const auto* a = static_cast<const bf16_t*>(x.data_ptr());
    const auto* b = static_cast<const bf16_t*>(weight.data_ptr());
    auto* output = static_cast<float*>(out.data_ptr());
    constexpr std::int32_t kBlockSize = 4 * 64;
    const auto blocks_y = static_cast<std::uint32_t>(host::div_ceil(m, 16));

    const bool use_splitk =
        k % 512 == 0 && (n == 1024 || (n == 2048 && m >= 8));
    if (use_splitk) {
      const auto split = n == 1024 ? 8u : 4u;
      const auto blocks_x =
          static_cast<std::uint32_t>(host::div_ceil(n, 64));
      if (n == 1024) {
        LaunchKernel({blocks_x, blocks_y, split}, kBlockSize, device_value)(
            gemm_nt_bf16_fp32_splitk<4, 64, 64, 8>,
            a,
            b,
            output,
            m,
            n,
            k);
      } else {
        LaunchKernel({blocks_x, blocks_y, split}, kBlockSize, device_value)(
            gemm_nt_bf16_fp32_splitk<4, 64, 64, 4>,
            a,
            b,
            output,
            m,
            n,
            k);
      }
    } else {
      const auto blocks_x =
          static_cast<std::uint32_t>(host::div_ceil(n, 16));
      constexpr std::uint32_t kSharedBytes =
          4 / 2 * 16 * 16 * sizeof(float);
      LaunchKernel(
          {blocks_x, blocks_y},
          kBlockSize,
          device_value,
          kSharedBytes)(gemm_nt_bf16_fp32<4, 16>, a, b, output, m, n, k);
    }
  }
};

#else

struct LinearBf16Fp32Kernel {
  static void run(
      const tvm::ffi::TensorView,
      const tvm::ffi::TensorView,
      const tvm::ffi::TensorView) {
    host::Panic(
        "linear_bf16_fp32: HCU MFMA requires gfx936, gfx938, or gfx928");
  }
};

#endif

}  // namespace sglang