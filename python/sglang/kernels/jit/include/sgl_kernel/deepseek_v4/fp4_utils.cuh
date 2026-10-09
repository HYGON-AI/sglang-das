#pragma once

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#ifndef USE_ROCM
#include <cuda_fp4.h>
#endif

// FP4 (e2m1) helpers: per-32 UE8M0 for the indexer, per-16 E4M3 for compressed KV.

namespace sglang {

namespace deepseek_v4::fp4 {

/// Largest finite e2m1 value.
constexpr float kMax = 6.0f;
/// `6 * 2^-126`, the amax floor `torch_quant.fake_quant_fp4` clamps to.
constexpr float kAmaxFloor = 6.0f * 1.1754943508222875e-38f;
/// Elements sharing one ue8m0 scale.
constexpr uint32_t kBlockSize = 32;
/// Compressed-KV elements sharing one E4M3 scale.
constexpr uint32_t kCompressedKVBlockSize = 16;

#ifdef USE_ROCM

/// HIP does not provide CUDA's ``__nv_cvt_*fp4*`` intrinsics on the HCU
/// targets used by DSV4.  Keep the fused c1/c2 compressor numerically aligned
/// with ``torch_quant.fake_quant_compressed_kv`` with the same scalar E2M1
/// round-to-nearest-even operation used by the standalone HIP RoPE kernel.
SGL_DEVICE float round_fp4_e2m1(float x) {
  const float ax = fabsf(x);
  const float step = ax < 2.0f ? 0.5f : (ax < 4.0f ? 1.0f : 2.0f);
  return copysignf(rintf(ax / step) * step, x);
}

/// Round a positive fp32 value to E4M3FN.  The caller clamps to the finite
/// range [2^-9, 448], so exponent overflow and NaN handling are unnecessary
/// here and the bit implementation is deterministic across CUDA/HIP.
SGL_DEVICE float round_e4m3fn(float x) {
  uint32_t u = __float_as_uint(x);
  uint32_t mant = u & 0x7FFFFFu;
  constexpr uint32_t kShift = 20u;
  const uint32_t guard = (mant >> (kShift - 1)) & 1u;
  const uint32_t sticky = (mant & ((1u << (kShift - 1)) - 1u)) != 0u;
  const uint32_t lsb = (mant >> kShift) & 1u;
  const uint32_t add = guard & (sticky | lsb);
  const uint32_t rounded = (mant >> kShift) + add;
  if (rounded >= 8u) {
    u = (((u >> 23) & 0xFFu) + 1u) << 23;
  } else {
    u = (u & 0xFF800000u) | (rounded << kShift);
  }
  return __uint_as_float(u);
}

SGL_DEVICE float compressed_kv_scale_hip(float amax) {
  const float raw = fminf(fmaxf(amax * (1.0f / kMax), 0x1p-9f), 448.0f);
  return round_e4m3fn(raw);
}

SGL_DEVICE fp32x2_t fake_quant_compressed_kv_x2_hip(fp32x2_t x, float scale) {
  const float x0 = fminf(fmaxf(x.x / scale, -kMax), kMax);
  const float x1 = fminf(fmaxf(x.y / scale, -kMax), kMax);
  return {round_fp4_e2m1(x0) * scale, round_fp4_e2m1(x1) * scale};
}

SGL_DEVICE fp32x2_t fake_quant_x2_hip(fp32x2_t x, float scale, float inv_scale) {
  const float x0 = fminf(fmaxf(x.x * inv_scale, -kMax), kMax);
  const float x1 = fminf(fmaxf(x.y * inv_scale, -kMax), kMax);
  return {round_fp4_e2m1(x0) * scale, round_fp4_e2m1(x1) * scale};
}

/// Encode one normalized value with the FP4 E2M1 nibble layout used by the
/// index-K cache.  The comparisons implement round-to-nearest-even at the
/// representable midpoints and drop the sign on zero (the CUDA intrinsic can
/// otherwise emit the distinct -0 nibble 0x8).
SGL_DEVICE uint8_t fp4_e2m1_code_hip(float x) {
  const float ax = fminf(fabsf(x), kMax);
  uint8_t idx = static_cast<uint8_t>(ax >= 0.25f) +
                static_cast<uint8_t>(ax >= 0.75f) +
                static_cast<uint8_t>(ax >= 1.25f) +
                static_cast<uint8_t>(ax >= 1.75f) +
                static_cast<uint8_t>(ax >= 2.5f) +
                static_cast<uint8_t>(ax >= 3.5f) +
                static_cast<uint8_t>(ax >= 5.0f);
  const bool midpoint = ax == 0.25f || ax == 0.75f || ax == 1.25f ||
                        ax == 1.75f || ax == 2.5f || ax == 3.5f || ax == 5.0f;
  if (midpoint && (idx & 1)) --idx;
  return idx | static_cast<uint8_t>((x < 0.0f && idx != 0) ? 0x8 : 0);
}

SGL_DEVICE uint32_t fp4_e2m1_code_x2_hip(fp32x2_t x, float inv_scale) {
  const auto c0 = fp4_e2m1_code_hip(x.x * inv_scale);
  const auto c1 = fp4_e2m1_code_hip(x.y * inv_scale);
  return static_cast<uint32_t>(c0) | (static_cast<uint32_t>(c1) << 4);
}

#endif  // USE_ROCM

/// \brief Round amax / 6 to a positive finite E4M3 scale, ties to even.
SGL_DEVICE float compressed_kv_scale(float amax) {
#ifdef USE_ROCM
  return compressed_kv_scale_hip(amax);
#else
  const auto raw = fminf(fmaxf(amax * (1.0f / kMax), 0x1p-9f), 448.0f);
  return static_cast<float>(__nv_fp8_e4m3(raw));
#endif
}

/// \brief Quantize compressed KV with its E4M3 scale and return dequantized values.
SGL_DEVICE fp32x2_t fake_quant_compressed_kv_x2(fp32x2_t x, float scale) {
#ifdef USE_ROCM
  return fake_quant_compressed_kv_x2_hip(x, scale);
#else
  const fp32x2_t scaled{__fdiv_rn(x.x, scale) + 0.0f, __fdiv_rn(x.y, scale) + 0.0f};
  const auto code = __nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest);
  const auto grid = device::cast<fp32x2_t>(fp16x2_t{__nv_cvt_fp4x2_to_halfraw2(code, __NV_E2M1)});
  return {grid.x * scale, grid.y * scale};
#endif
}

/// \brief Per-block ue8m0 scale and its reciprocal, from the block's absmax.
SGL_DEVICE fp32x2_t block_scale(float amax) {
  const auto exponent = fp8::cast_to_ue8m0(fmaxf(amax, kAmaxFloor) * (1.0f / kMax));
  return {__uint_as_float(static_cast<uint32_t>(exponent) << 23), fp8::inv_scale_ue8m0(exponent)};
}

/// \brief Round a pair onto the e2m1 grid and back, through `scale`.
///
/// Every e2m1 value is exact in fp16, so the roundtrip is lossless. Adding `0.0f` during
/// scaling clears negative zero to match `torch.sign(0) == 0` in `torch_quant.round_fp4`.
SGL_DEVICE fp32x2_t fake_quant_x2(fp32x2_t x, float scale, float inv_scale) {
#ifdef USE_ROCM
  return fake_quant_x2_hip(x, scale, inv_scale);
#else
  const fp32x2_t scaled{__fmaf_rn(x.x, inv_scale, 0.0f), __fmaf_rn(x.y, inv_scale, 0.0f)};
  const auto code = __nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest);
  const auto grid = device::cast<fp32x2_t>(fp16x2_t{__nv_cvt_fp4x2_to_halfraw2(code, __NV_E2M1)});
  return {grid.x * scale, grid.y * scale};
#endif
}

}  // namespace deepseek_v4::fp4

}  // namespace sglang
