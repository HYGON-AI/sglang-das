#include <hip/hip_runtime.h>
#include <cstdint>

// One generic kernel for every batch size: no per-shape compilation.
__global__ void fused_replay_state_indices_hip_kernel(
    int64_t *__restrict__ req, const int32_t *__restrict__ mapping,
    int32_t *__restrict__ out, int valid_bs, int total_bs) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= total_bs)
    return;
  if (row < valid_bs) {
    out[row] = mapping[req[row]];
  } else {
    out[row] = -1;
    req[row] = 0;
  }
}

extern "C" int sgl_fused_replay_state_indices(int64_t *req,
                                              const int32_t *mapping,
                                              int32_t *out, int valid_bs,
                                              int total_bs,
                                              hipStream_t stream) {
  if (valid_bs < 0 || valid_bs > total_bs)
    return hipErrorInvalidValue;
  if (total_bs == 0)
    return hipSuccess;
  constexpr int threads = 128;
  hipLaunchKernelGGL(fused_replay_state_indices_hip_kernel,
                     dim3((total_bs - 1) / threads + 1), dim3(threads), 0,
                     stream, req, mapping, out, valid_bs, total_bs);
  return hipGetLastError();
}

extern "C" const char *sgl_replay_state_indices_error(int error) {
  return hipGetErrorString(static_cast<hipError_t>(error));
}
