// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

/*
 * Fused RoPE-tail + FP4 fake-quant for DeepSeek V4.1 KV cache write (HIP/DCU).
 *
 * Replaces the Python 4-pass fallback of _rope_fq4 (dsv41_sparse.py) with a
 * single kernel that does all four passes (load, rope, reduce-amax, quant) in
 * registers, cutting memory round-trips from 4x to 1x.
 *
 * Supported configs (D, rope_dim, BLK):
 *   D=512, rope_dim=64, BLK=32 (UE8M0 per-32 scale, normal KV)
 *   D=512, rope_dim=64, BLK=16 (E4M3FN per-16 scale, compressed_kv)
 *   D=128, rope_dim=64, BLK=32 (UE8M0 per-32 scale, normal KV)
 *   D=128, rope_dim=64, BLK=16 (E4M3FN per-16 scale, compressed_kv)
 */

#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

// AMD wavefront size for gfx9xx / gfx938 / gfx942 / gfx950.
static constexpr int AMD_WARP = 64;

#define HIP_CHECK(expr)                                              \
  TORCH_CHECK(                                                       \
      (expr) == hipSuccess, "HIP error: ", hipGetErrorString((expr)))

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// Round to the FP4 E2M1 grid: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}.
__device__ __forceinline__ float round_fp4_e2m1(float x) {
  float ax   = fabsf(x);
  float step = (ax < 2.0f) ? 0.5f : ((ax < 4.0f) ? 1.0f : 2.0f);
  return copysignf(rintf(ax / step) * step, x);
}

// UE8M0 scale = smallest power-of-2 >= max(amax, FLOOR) / 6.
// FLOOR = 6 * 2^-126 matches torch_quant.py's FP4_AMAX_FLOOR.
__device__ __forceinline__ float ceil_pow2_ue8m0(float amax) {
  const float FLOOR = __uint_as_float(0x00800000u) * 6.0f;  // 6 * 2^-126
  float a          = fmaxf(amax, FLOOR) * (1.0f / 6.0f);
  uint32_t bits    = __float_as_uint(a);
  uint32_t expo    = (bits >> 23) & 0xFFu;
  expo            += (bits & 0x7FFFFFu) != 0u;  // ceil: bump exponent if any mantissa bit set
  return __uint_as_float(expo << 23);
}

// Round fp32 > 0 to the nearest E4M3FN representable value.
// E4M3FN has 3 mantissa bits; we keep bits [22:20] of the fp32 mantissa and
// apply round-half-to-even on bit 19.  Precondition: x in [2^-9, 448].
__device__ __forceinline__ float round_to_e4m3fn(float x) {
  uint32_t u       = __float_as_uint(x);
  uint32_t mant    = u & 0x7FFFFFu;
  const uint32_t SHIFT = 20u;
  uint32_t guard   = (mant >> (SHIFT - 1)) & 1u;
  // sticky must be a boolean (0 or 1) — bitwise OR with lsb would give wrong
  // results if any sticky bit other than bit-0 is set while bit-0 is clear.
  uint32_t sticky  = (mant & ((1u << (SHIFT - 1)) - 1u)) != 0u ? 1u : 0u;
  uint32_t lsb     = (mant >> SHIFT) & 1u;
  uint32_t add     = guard & (sticky | lsb);
  uint32_t rounded = (mant >> SHIFT) + add;
  if (rounded >= 8u) {
    // Mantissa overflow -> increment exponent, zero mantissa.
    u = (((u >> 23) & 0xFFu) + 1u) << 23;
  } else {
    u = (u & 0xFF800000u) | (rounded << SHIFT);
  }
  return __uint_as_float(u);
}

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------

// Template parameters:
//   D            - total feature dim per row (512 or 128)
//   ROPE_DIM     - features that receive RoPE rotation (64)
//   BLK          - FP4 quantization block width (32 normal, 16 compressed_kv)
//   COMPRESSED_KV- true -> E4M3FN per-BLK scale; false -> UE8M0 per-BLK scale
template <int D, int ROPE_DIM, int BLK, bool COMPRESSED_KV>
__global__ void __launch_bounds__(D) rope_fp4_fq_kernel(
    const __hip_bfloat16* __restrict__ x,      // [N, D] bf16
    const float*           __restrict__ freqs,  // [T, ROPE_DIM] fp32 re/im interleaved
    __hip_bfloat16*        __restrict__ out,    // [N, D] bf16
    int rows_per_token,
    int N) {
  constexpr int HEAD_LEN = D - ROPE_DIM;
  const int row = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (row >= N) return;

  const int token = row / rows_per_token;

  // ------------------------------------------------------------------
  // 1. Load x[row, tid] and apply RoPE on the tail ROPE_DIM features.
  // ------------------------------------------------------------------
  float val = __bfloat162float(x[row * D + tid]);

  if (tid >= HEAD_LEN) {
    const int pos     = tid - HEAD_LEN;
    const int pair    = pos >> 1;
    const bool is_imag = pos & 1;

    const float* fptr = freqs + static_cast<int64_t>(token) * ROPE_DIM;
    float re = __bfloat162float(x[row * D + HEAD_LEN + 2 * pair]);
    float im = __bfloat162float(x[row * D + HEAD_LEN + 2 * pair + 1]);
    float fr = fptr[2 * pair];
    float fi = fptr[2 * pair + 1];

    float rot = is_imag ? (re * fi + im * fr) : (re * fr - im * fi);
    // bf16 round-trip to match Python rope_tail output exactly.
    val = __bfloat162float(__float2bfloat16(rot));
  }

  // ------------------------------------------------------------------
  // 2. Intra-block amax reduction.
  //
  // AMD wavefront = 64; we partition it into (AMD_WARP/BLK) independent
  // sub-groups of width BLK using __shfl with width=BLK.
  // ------------------------------------------------------------------
  float amax = fabsf(val);
#pragma unroll
  for (int off = BLK >> 1; off >= 1; off >>= 1)
    amax = fmaxf(amax, __shfl_down(amax, off, BLK));
  // Broadcast from the first lane of this BLK-group back to all lanes.
  amax = __shfl(amax, (tid / BLK) * BLK, AMD_WARP);

  // ------------------------------------------------------------------
  // 3. Scale computation and FP4 fake-quant.
  // ------------------------------------------------------------------
  float scale;
  if constexpr (COMPRESSED_KV) {
    // Per-16 E4M3FN scale: clamp amax/6 to [2^-9, 448] then round.
    float s = fminf(fmaxf(amax * (1.0f / 6.0f), 1.953125e-3f /*2^-9*/), 448.0f);
    scale   = round_to_e4m3fn(s);
  } else {
    // Per-32 UE8M0 scale: next power-of-two >= max(amax, FLOOR) / 6.
    scale = ceil_pow2_ue8m0(amax);
  }

  float q = fminf(fmaxf(val / scale, -6.0f), 6.0f);
  q       = round_fp4_e2m1(q);
  out[row * D + tid] = __float2bfloat16(q * scale);
}

// ---------------------------------------------------------------------------
// Per-D launchers
// ---------------------------------------------------------------------------

static void launch_d512(
    const __hip_bfloat16* x,
    const float*           freqs,
    __hip_bfloat16*        out,
    int                    N,
    int                    rows_per_token,
    bool                   compressed_kv,
    hipStream_t            stream) {
  if (compressed_kv)
    hipLaunchKernelGGL(
        (rope_fp4_fq_kernel<512, 64, 16, true>),
        dim3(N), dim3(512), 0, stream, x, freqs, out, rows_per_token, N);
  else
    hipLaunchKernelGGL(
        (rope_fp4_fq_kernel<512, 64, 32, false>),
        dim3(N), dim3(512), 0, stream, x, freqs, out, rows_per_token, N);
  HIP_CHECK(hipGetLastError());
}

static void launch_d128(
    const __hip_bfloat16* x,
    const float*           freqs,
    __hip_bfloat16*        out,
    int                    N,
    int                    rows_per_token,
    bool                   compressed_kv,
    hipStream_t            stream) {
  if (compressed_kv)
    hipLaunchKernelGGL(
        (rope_fp4_fq_kernel<128, 64, 16, true>),
        dim3(N), dim3(128), 0, stream, x, freqs, out, rows_per_token, N);
  else
    hipLaunchKernelGGL(
        (rope_fp4_fq_kernel<128, 64, 32, false>),
        dim3(N), dim3(128), 0, stream, x, freqs, out, rows_per_token, N);
  HIP_CHECK(hipGetLastError());
}

// ---------------------------------------------------------------------------
// Public entry point (registered via TORCH_LIBRARY)
// ---------------------------------------------------------------------------

// \brief Fused RoPE-tail + FP4 fake-quant for DSV4.1 KV cache write.
//
// \param x           [N, D] or [B, H, D] bfloat16 on HIP device.
// \param freqs       [T, rope_dim/2] complex64; position-gathered before call.
// \param rope_dim    Number of features rotated by RoPE (must equal freqs dim * 2).
// \param compressed_kv true -> per-16 E4M3FN scale; false -> per-32 UE8M0 scale.
// \return            Tensor of same shape/dtype as x.
at::Tensor rope_fp4_fake_quant_hip(
    const at::Tensor& x,
    const at::Tensor& freqs,
    int64_t           rope_dim,
    bool              compressed_kv) {
  TORCH_CHECK(x.is_cuda(), "x must be on a CUDA/HIP device");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(freqs.scalar_type() == at::kComplexFloat, "freqs must be complex64");
  TORCH_CHECK(freqs.is_contiguous(), "freqs must be contiguous");
  TORCH_CHECK(
      rope_dim == 64,
      "rope_fp4_fake_quant_hip: only rope_dim=64 is supported; got ", rope_dim);

  auto out = at::empty_like(x);

  int64_t D            = x.size(-1);
  int rows_per_token   = (x.dim() == 3) ? static_cast<int>(x.size(1)) : 1;
  at::Tensor x2   = (x.dim() == 2) ? x   : x.reshape({-1, D});
  at::Tensor out2  = (x.dim() == 2) ? out : out.reshape({-1, D});
  int N = static_cast<int>(x2.size(0));

  // View complex64 [T, rope_dim/2] as float32 [T, rope_dim] (re/im interleaved).
  at::Tensor freqs_f32 =
      at::view_as_real(freqs).contiguous().view({-1, rope_dim});

  hipStream_t stream =
      at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();

  const auto* xp = reinterpret_cast<const __hip_bfloat16*>(
      x2.const_data_ptr<at::BFloat16>());
  auto* op = reinterpret_cast<__hip_bfloat16*>(out2.data_ptr<at::BFloat16>());
  const float* fp = freqs_f32.const_data_ptr<float>();

  TORCH_CHECK(
      D == 512 || D == 128,
      "rope_fp4_fake_quant_hip: unsupported D=", D, "; expected 128 or 512");

  if (D == 512)
    launch_d512(xp, fp, op, N, rows_per_token, compressed_kv, stream);
  else
    launch_d128(xp, fp, op, N, rows_per_token, compressed_kv, stream);

  return out;
}
