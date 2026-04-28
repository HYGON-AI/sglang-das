/// \file runtime.cuh
/// \brief Host-side CUDA runtime query helpers.
///
/// Thin wrappers around CUDA occupancy and device-property APIs with
/// automatic error checking via `RuntimeDeviceCheck`.

#pragma once

#include <sgl_kernel/utils.cuh>

#include <cstddef>
#include <cstdint>
#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace host::runtime {

#ifdef USE_ROCM
// Return the maximum number of active blocks per SM for the given kernel
template <typename T>
inline auto get_blocks_per_sm(T&& kernel, int32_t block_dim, std::size_t dynamic_smem = 0) -> uint32_t {
  int num_blocks_per_sm = 0;
  RuntimeDeviceCheck(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks_per_sm,
      kernel,
      block_dim,
      dynamic_smem));
  return static_cast<uint32_t>(num_blocks_per_sm);
}

// Return the number of CUs for the given device
inline auto get_sm_count(int device_id) -> uint32_t {
  int sm_count;
  RuntimeDeviceCheck(hipDeviceGetAttribute(&sm_count, hipDeviceAttributeMultiprocessorCount, device_id));
  return static_cast<uint32_t>(sm_count);
}

// Return the major compute capability for the given device
inline auto get_cc_major(int device_id) -> int {
  int cc_major;
  RuntimeDeviceCheck(hipDeviceGetAttribute(&cc_major, hipDeviceAttributeComputeCapabilityMajor, device_id));
  return cc_major;
}

// Return the runtime version
inline auto get_runtime_version() -> int {
  int runtime_version;
  RuntimeDeviceCheck(hipRuntimeGetVersion(&runtime_version));
  return runtime_version;
}

// Return the maximum dynamic shared memory per block for the given kernel
template <typename T>
inline auto get_available_dynamic_smem_per_block(T&& kernel, int num_blocks, int block_size) -> std::size_t {
  std::size_t smem_size;
  RuntimeDeviceCheck(hipOccupancyAvailableDynamicSMemPerBlock(
      &smem_size,
      kernel,
      num_blocks,
      block_size));
  return smem_size;
}
#else
// Return the maximum number of active blocks per SM for the given kernel
template <typename T>
inline auto get_blocks_per_sm(T&& kernel, int32_t block_dim, std::size_t dynamic_smem = 0) -> uint32_t {
  int num_blocks_per_sm = 0;
  RuntimeDeviceCheck(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel, block_dim, dynamic_smem));
  return static_cast<uint32_t>(num_blocks_per_sm);
}

// Return the number of SMs for the given device
inline auto get_sm_count(int device_id) -> uint32_t {
  int sm_count;
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));
  return static_cast<uint32_t>(sm_count);
}

// Return the Major compute capability for the given device
inline auto get_cc_major(int device_id) -> int {
  int cc_major;
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device_id));
  return cc_major;
}

// Return the runtime version
inline auto get_runtime_version() -> int {
  int runtime_version;
  RuntimeDeviceCheck(cudaRuntimeGetVersion(&runtime_version));
  return runtime_version;
}

// Return the maximum dynamic shared memory per block for the given kernel
template <typename T>
inline auto get_available_dynamic_smem_per_block(T&& kernel, int num_blocks, int block_size) -> std::size_t {
  std::size_t smem_size;
  RuntimeDeviceCheck(cudaOccupancyAvailableDynamicSMemPerBlock(&smem_size, kernel, num_blocks, block_size));
  return smem_size;
}
#endif

}  // namespace host::runtime
