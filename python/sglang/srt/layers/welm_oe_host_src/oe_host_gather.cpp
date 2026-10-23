// Gather rows of a host-resident (hipHostRegister'ed, mapped) bf16 table into device memory.
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>

__global__ void gather_rows(const uint4* __restrict__ table, const int64_t* __restrict__ idx,
                            uint4* __restrict__ out, int64_t n, int row_vec, int64_t vocab) {
  int64_t r = blockIdx.x * (int64_t)blockDim.y + threadIdx.y;
  if (r >= n) return;
  int64_t s = idx[r];
  if (s < 0 || s >= vocab) s = 0;
  const uint4* src = table + s * row_vec;
  uint4* dst = out + r * row_vec;
  for (int c = threadIdx.x; c < row_vec; c += blockDim.x) dst[c] = src[c];
}

// table_dev_ptr: device pointer of the mapped host table [vocab, dim] bf16; idx int64 [n] on device.
torch::Tensor oe_host_gather(int64_t table_dev_ptr, int64_t vocab, int64_t dim, torch::Tensor idx, torch::Tensor out) {
  TORCH_CHECK(idx.is_cuda() && idx.scalar_type() == torch::kInt64 && idx.is_contiguous());
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16 && out.is_contiguous());
  TORCH_CHECK((dim * 2) % 16 == 0);
  int64_t n = idx.numel();
  if (n == 0) return out;
  int row_vec = dim * 2 / 16;
  dim3 block(64, 4);
  dim3 grid((n + 3) / 4);
  hipLaunchKernelGGL(gather_rows, grid, block, 0, at::hip::getCurrentHIPStream(),
                     reinterpret_cast<const uint4*>(table_dev_ptr), idx.data_ptr<int64_t>(),
                     reinterpret_cast<uint4*>(out.data_ptr()), n, row_vec, vocab);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("oe_host_gather", &oe_host_gather); }
