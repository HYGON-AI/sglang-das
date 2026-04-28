/// \file warp.cuh
/// \brief Warp-level reduction primitives using `__shfl_xor_sync`.

#pragma once
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/utils.cuh>

namespace device::warp {

/// \brief Full 32-thread active mask.
static constexpr uint32_t kFullMask = 0xffffffffu;

template <typename T>
SGL_DEVICE T shfl_down(uint32_t active_mask, T value, int delta, int width = kWarpThreads) {
#ifdef USE_ROCM
  (void)active_mask;
  return __shfl_down(value, delta, width);
#else
  return __shfl_down_sync(active_mask, value, delta, width);
#endif
}

template <typename T>
SGL_DEVICE T shfl_xor(uint32_t active_mask, T value, int lane_mask, int width = kWarpThreads) {
#ifdef USE_ROCM
  (void)active_mask;
  return __shfl_xor(value, lane_mask, width);
#else
  return __shfl_xor_sync(active_mask, value, lane_mask, width);
#endif
}

template <uint32_t kThreads = kWarpThreads, typename T>
SGL_DEVICE T reduce_sum(T value, uint32_t active_mask = kFullMask) {
  static_assert(kNumThreads >= 1 && kNumThreads <= kWarpThreads);
  static_assert(std::has_single_bit(kNumThreads), "must be pow of 2");
#pragma unroll
  for (auto mask = kThreads >> 1; mask > 0; mask >>= 1)
#ifdef USE_ROCM
    value = value + __shfl_xor(value, mask, 32);
#else
    value = value + __shfl_xor_sync(active_mask, value, mask, 32);
#endif
  return value;
}

/**
 * \brief Warp-level max reduction.
 *
 * Computes the maximum of `value` across all active lanes using
 * butterfly shuffles. The result is broadcast to all participating
 * lanes.
 *
 * \tparam kNumThreads Group size for the reduction (defaults to a full warp).
 * \tparam T Numeric type (must be supported by `math::max`).
 * \param value Per-lane input value.
 * \param active_mask Bitmask of participating lanes (default: all 32).
 * \return The maximum across all active lanes.
 */
template <uint32_t kNumThreads = kWarpThreads, typename T>
SGL_DEVICE T reduce_max(T value, uint32_t active_mask = kFullMask) {
  static_assert(kNumThreads >= 1 && kNumThreads <= kWarpThreads);
  static_assert(std::has_single_bit(kNumThreads), "must be pow of 2");
#pragma unroll
  for (auto mask = kThreads >> 1; mask > 0; mask >>= 1)
#ifdef USE_ROCM
    value = math::max(value, __shfl_xor(value, mask, 32));
#else
    value = math::max(value, shfl_xor(active_mask, value, mask, kWarpThreads));
#endif
  return value;
}

}  // namespace device::warp
