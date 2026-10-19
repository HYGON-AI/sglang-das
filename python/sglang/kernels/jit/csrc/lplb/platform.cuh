#pragma once

#include <sgl_kernel/utils.cuh>  // For SGL_DEVICE and CUDA/HIP runtime declarations

#include <dlpack/dlpack.h>

namespace sglang::lplb {

#if defined(__HIP_PLATFORM_AMD__)
inline constexpr DLDeviceType kDeviceType = kDLROCM;
#else
inline constexpr DLDeviceType kDeviceType = kDLCUDA;
#endif

// The LP reductions use logical groups of 32 lanes. Specify the width on
// HIP so the two halves of a 64-lane wavefront reduce independently.
SGL_DEVICE float shuffle_xor_32(float value, int offset) {
#if defined(__HIP_PLATFORM_AMD__)
  return __shfl_xor(value, offset, 32);
#else
  return __shfl_xor_sync(0xffffffff, value, offset, 32);
#endif
}

SGL_DEVICE float shuffle_down_32(float value, int offset) {
#if defined(__HIP_PLATFORM_AMD__)
  return __shfl_down(value, offset, 32);
#else
  return __shfl_down_sync(0xffffffff, value, offset, 32);
#endif
}

}  // namespace sglang::lplb
