/*
 * JIT kpool write-plan update kernels for NSA CUDA graph replay.
 *
 * The decode replay path refreshes identical kpool write-plan inputs for
 * multiple speculative draft backends, but each backend owns separate captured
 * output buffers. Updating the four backend-local plans in one small JIT
 * kernel avoids repeated Triton launcher scheduling on the CPU hot path.
 */

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>

#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace sglang {

struct KPoolWritePlanMultiDecodeParams {
  const int32_t* __restrict__ write_start;
  const int64_t* __restrict__ req_pool_indices;
  const int32_t* __restrict__ real_page_table;

  int64_t* __restrict__ req_out0;
  int32_t* __restrict__ write_start_out0;
  int32_t* __restrict__ tail_logical_start_out0;
  int64_t* __restrict__ write_loc_out0;

  int64_t* __restrict__ req_out1;
  int32_t* __restrict__ write_start_out1;
  int32_t* __restrict__ tail_logical_start_out1;
  int64_t* __restrict__ write_loc_out1;

  int64_t* __restrict__ req_out2;
  int32_t* __restrict__ write_start_out2;
  int32_t* __restrict__ tail_logical_start_out2;
  int64_t* __restrict__ write_loc_out2;

  int64_t* __restrict__ req_out3;
  int32_t* __restrict__ write_start_out3;
  int32_t* __restrict__ tail_logical_start_out3;
  int64_t* __restrict__ write_loc_out3;

  int bs;
  int real_page_table_stride;
  int pool_size;
  int slots_per_page;
};

struct KPoolWritePlanParams {
  const int32_t* __restrict__ write_start;
  const int64_t* __restrict__ req_pool_indices;
  const int32_t* __restrict__ real_page_table;
  int64_t* __restrict__ req_out;
  int32_t* __restrict__ write_start_out;
  int32_t* __restrict__ tail_logical_start_out;
  int64_t* __restrict__ write_loc_out;
  int32_t* __restrict__ pool_seqlens_per_q_out;
  int32_t* __restrict__ seqlens_per_q_out;
  int bs;
  int real_page_table_stride;
  int pool_size;
  int slots_per_page;
};

template <int N, bool HAS_PER_Q>
__global__ void kpool_write_plan_kernel(const KPoolWritePlanParams params) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = gridDim.x * blockDim.x;

  for (int b = tid; b < params.bs; b += total_threads) {
    const int32_t ws = params.write_start[b];
    const int64_t req = params.req_pool_indices[b];
    const int32_t base_pool = ws / params.pool_size;

    if constexpr (HAS_PER_Q) {
      for (int k = 0; k < N; ++k) {
        const int row = b * N + k;
        const int32_t seqlen_per_q = ws + k + 1;
        params.seqlens_per_q_out[row] = seqlen_per_q;
        params.pool_seqlens_per_q_out[row] = seqlen_per_q / params.pool_size;
      }
    }

    const int32_t tail_logical_start = base_pool * params.pool_size;
    const int32_t pool_page_group = base_pool / params.slots_per_page;
    const int32_t packed_page = params.real_page_table[(b * N) * params.real_page_table_stride + pool_page_group];
    const int64_t write_loc =
        static_cast<int64_t>(packed_page) * params.slots_per_page + (base_pool % params.slots_per_page);

    params.req_out[b] = req;
    params.write_start_out[b] = ws;
    params.tail_logical_start_out[b] = tail_logical_start;
    params.write_loc_out[b] = write_loc;
  }
}

__global__ void kpool_write_plan_multi_decode_kernel(const KPoolWritePlanMultiDecodeParams params) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = gridDim.x * blockDim.x;

  for (int b = tid; b < params.bs; b += total_threads) {
    const int32_t ws = params.write_start[b];
    const int64_t req = params.req_pool_indices[b];
    const int32_t base_pool = ws / params.pool_size;
    const int32_t tail_logical_start = base_pool * params.pool_size;
    const int32_t pool_page_group = base_pool / params.slots_per_page;
    const int32_t packed_page = params.real_page_table[b * params.real_page_table_stride + pool_page_group];
    const int64_t write_loc =
        static_cast<int64_t>(packed_page) * params.slots_per_page + (base_pool % params.slots_per_page);

    params.req_out0[b] = req;
    params.write_start_out0[b] = ws;
    params.tail_logical_start_out0[b] = tail_logical_start;
    params.write_loc_out0[b] = write_loc;

    params.req_out1[b] = req;
    params.write_start_out1[b] = ws;
    params.tail_logical_start_out1[b] = tail_logical_start;
    params.write_loc_out1[b] = write_loc;

    params.req_out2[b] = req;
    params.write_start_out2[b] = ws;
    params.tail_logical_start_out2[b] = tail_logical_start;
    params.write_loc_out2[b] = write_loc;

    params.req_out3[b] = req;
    params.write_start_out3[b] = ws;
    params.tail_logical_start_out3[b] = tail_logical_start;
    params.write_loc_out3[b] = write_loc;
  }
}

namespace {

constexpr int THREADS_PER_BLOCK = 256;
constexpr int MAX_GRID_SIZE = 1024;

template <typename T>
inline const T* unwrap_data_ptr(const tvm::ffi::TensorView& tensor, const char* name) {
  using namespace host;
  RuntimeCheck(tensor.data_ptr() != nullptr, "Tensor ", name, " must not be null");
  RuntimeCheck(is_type<T>(tensor.dtype()), "Tensor ", name, " has unexpected dtype");
  return static_cast<const T*>(tensor.data_ptr());
}

template <typename T>
inline T* unwrap_data_ptr_mut(const tvm::ffi::TensorView& tensor, const char* name) {
  using namespace host;
  RuntimeCheck(tensor.data_ptr() != nullptr, "Tensor ", name, " must not be null");
  RuntimeCheck(is_type<T>(tensor.dtype()), "Tensor ", name, " has unexpected dtype");
  return static_cast<T*>(tensor.data_ptr());
}

inline dim3 get_launch_config(int total_work, int threads_per_block = THREADS_PER_BLOCK) {
  int num_blocks = (total_work + threads_per_block - 1) / threads_per_block;
  num_blocks = std::min(num_blocks, MAX_GRID_SIZE);
  return dim3(num_blocks);
}

template <int N, bool HAS_PER_Q>
struct KPoolWritePlanKernel {
  static void
  run(const tvm::ffi::TensorView write_start,
      const tvm::ffi::TensorView req_pool_indices,
      const tvm::ffi::TensorView real_page_table,
      const tvm::ffi::TensorView req_out,
      const tvm::ffi::TensorView write_start_out,
      const tvm::ffi::TensorView tail_logical_start_out,
      const tvm::ffi::TensorView write_loc_out,
      const tvm::ffi::TensorView pool_seqlens_per_q_out,
      const tvm::ffi::TensorView seqlens_per_q_out,
      int pool_size,
      int slots_per_page) {
    using namespace host;

    const int bs = static_cast<int>(write_start.shape()[0]);
    if (bs == 0) {
      return;
    }
    RuntimeCheck(N > 0, "num_draft_tokens must be positive");
    RuntimeCheck(pool_size > 0, "pool_size must be positive");
    RuntimeCheck(slots_per_page > 0, "slots_per_page must be positive");

    const auto params = KPoolWritePlanParams{
        .write_start = unwrap_data_ptr<int32_t>(write_start, "write_start"),
        .req_pool_indices = unwrap_data_ptr<int64_t>(req_pool_indices, "req_pool_indices"),
        .real_page_table = unwrap_data_ptr<int32_t>(real_page_table, "real_page_table"),
        .req_out = unwrap_data_ptr_mut<int64_t>(req_out, "req_out"),
        .write_start_out = unwrap_data_ptr_mut<int32_t>(write_start_out, "write_start_out"),
        .tail_logical_start_out = unwrap_data_ptr_mut<int32_t>(tail_logical_start_out, "tail_logical_start_out"),
        .write_loc_out = unwrap_data_ptr_mut<int64_t>(write_loc_out, "write_loc_out"),
        .pool_seqlens_per_q_out = unwrap_data_ptr_mut<int32_t>(pool_seqlens_per_q_out, "pool_seqlens_per_q_out"),
        .seqlens_per_q_out = unwrap_data_ptr_mut<int32_t>(seqlens_per_q_out, "seqlens_per_q_out"),
        .bs = bs,
        .real_page_table_stride = static_cast<int>(real_page_table.stride(0)),
        .pool_size = pool_size,
        .slots_per_page = slots_per_page,
    };

    dim3 grid = get_launch_config(bs);
    dim3 block(THREADS_PER_BLOCK);
    DLDevice device = write_start.device();
    host::LaunchKernel(grid, block, device)(kpool_write_plan_kernel<N, HAS_PER_Q>, params);
  }
};

struct KPoolWritePlanMultiDecodeKernel {
  static void
  run(const tvm::ffi::TensorView write_start,
      const tvm::ffi::TensorView req_pool_indices,
      const tvm::ffi::TensorView real_page_table,
      const tvm::ffi::TensorView req_out0,
      const tvm::ffi::TensorView write_start_out0,
      const tvm::ffi::TensorView tail_logical_start_out0,
      const tvm::ffi::TensorView write_loc_out0,
      const tvm::ffi::TensorView req_out1,
      const tvm::ffi::TensorView write_start_out1,
      const tvm::ffi::TensorView tail_logical_start_out1,
      const tvm::ffi::TensorView write_loc_out1,
      const tvm::ffi::TensorView req_out2,
      const tvm::ffi::TensorView write_start_out2,
      const tvm::ffi::TensorView tail_logical_start_out2,
      const tvm::ffi::TensorView write_loc_out2,
      const tvm::ffi::TensorView req_out3,
      const tvm::ffi::TensorView write_start_out3,
      const tvm::ffi::TensorView tail_logical_start_out3,
      const tvm::ffi::TensorView write_loc_out3,
      int pool_size,
      int slots_per_page) {
    using namespace host;

    const int bs = static_cast<int>(write_start.shape()[0]);
    if (bs == 0) {
      return;
    }
    RuntimeCheck(pool_size > 0, "pool_size must be positive");
    RuntimeCheck(slots_per_page > 0, "slots_per_page must be positive");

    const auto params = KPoolWritePlanMultiDecodeParams{
        .write_start = unwrap_data_ptr<int32_t>(write_start, "write_start"),
        .req_pool_indices = unwrap_data_ptr<int64_t>(req_pool_indices, "req_pool_indices"),
        .real_page_table = unwrap_data_ptr<int32_t>(real_page_table, "real_page_table"),
        .req_out0 = unwrap_data_ptr_mut<int64_t>(req_out0, "req_out0"),
        .write_start_out0 = unwrap_data_ptr_mut<int32_t>(write_start_out0, "write_start_out0"),
        .tail_logical_start_out0 = unwrap_data_ptr_mut<int32_t>(tail_logical_start_out0, "tail_logical_start_out0"),
        .write_loc_out0 = unwrap_data_ptr_mut<int64_t>(write_loc_out0, "write_loc_out0"),
        .req_out1 = unwrap_data_ptr_mut<int64_t>(req_out1, "req_out1"),
        .write_start_out1 = unwrap_data_ptr_mut<int32_t>(write_start_out1, "write_start_out1"),
        .tail_logical_start_out1 = unwrap_data_ptr_mut<int32_t>(tail_logical_start_out1, "tail_logical_start_out1"),
        .write_loc_out1 = unwrap_data_ptr_mut<int64_t>(write_loc_out1, "write_loc_out1"),
        .req_out2 = unwrap_data_ptr_mut<int64_t>(req_out2, "req_out2"),
        .write_start_out2 = unwrap_data_ptr_mut<int32_t>(write_start_out2, "write_start_out2"),
        .tail_logical_start_out2 = unwrap_data_ptr_mut<int32_t>(tail_logical_start_out2, "tail_logical_start_out2"),
        .write_loc_out2 = unwrap_data_ptr_mut<int64_t>(write_loc_out2, "write_loc_out2"),
        .req_out3 = unwrap_data_ptr_mut<int64_t>(req_out3, "req_out3"),
        .write_start_out3 = unwrap_data_ptr_mut<int32_t>(write_start_out3, "write_start_out3"),
        .tail_logical_start_out3 = unwrap_data_ptr_mut<int32_t>(tail_logical_start_out3, "tail_logical_start_out3"),
        .write_loc_out3 = unwrap_data_ptr_mut<int64_t>(write_loc_out3, "write_loc_out3"),
        .bs = bs,
        .real_page_table_stride = static_cast<int>(real_page_table.stride(0)),
        .pool_size = pool_size,
        .slots_per_page = slots_per_page,
    };

    dim3 grid = get_launch_config(bs);
    dim3 block(THREADS_PER_BLOCK);
    DLDevice device = write_start.device();
    host::LaunchKernel(grid, block, device)(kpool_write_plan_multi_decode_kernel, params);
  }
};

}  // namespace

}  // namespace sglang
