#pragma once

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstdint>
#ifdef USE_ROCM
#include <hip/hip_fp8.h>
#else
#include <cuda_fp8.h>
#endif

// Small helpers shared by the DeepSeek-V4 FP8/UE8M0 quantization kernels
// (silu_and_mul_masked_post_quant, store, mega_moe_pre_dispatch, ...).
// All functions are `SGL_DEVICE` (= `__forceinline__ __device__`) so
// including this header in multiple translation units is ODR-safe.

namespace deepseek_v4::fp8 {

// Round `x` to the nearest representable UE8M0 value. Returns the raw
// 8-bit biased exponent; the actual fp32 scale is `2^(exp - 127)`
// (i.e. `__uint_as_float(exp << 23)`).
SGL_DEVICE int32_t cast_to_ue8m0(float x) {
  uint32_t u = __float_as_uint(x);
  int32_t exp = int32_t((u >> 23) & 0xFF);
  uint32_t mant = u & 0x7FFFFF;
  return exp + (mant != 0);
}

// 1 / 2^(exp - 127) as fp32. Equivalent to `1.0f / __uint_as_float(exp << 23)`.
SGL_DEVICE float inv_scale_ue8m0(int32_t exp) {
  return __uint_as_float((127 + 127 - exp) << 23);
}

// Clamp to [-FP8_E4M3_MAX, FP8_E4M3_MAX].
// Uses platform-specific max from type.cuh (448 for E4M3FN, 224 for E4M3FNUZ).
SGL_DEVICE float fp8_e4m3_clip(float val) {
  return fmaxf(fminf(val, kFP8E4M3Max), -kFP8E4M3Max);
}

// Pack two fp32 values into a single fp8x2_e4m3 with clamping.
SGL_DEVICE fp8x2_e4m3_t pack_fp8(float x, float y) {
  return fp8x2_e4m3_t{fp32x2_t{fp8_e4m3_clip(x), fp8_e4m3_clip(y)}};
}

}  // namespace deepseek_v4::fp8
