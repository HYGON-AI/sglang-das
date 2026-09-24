#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <algorithm>
#include <optional>

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaxGridSize = 1024;

__global__ void kpool_write_plan_kernel(
    const int32_t* __restrict__ write_start,
    const int64_t* __restrict__ req_pool_indices,
    const int32_t* __restrict__ real_page_table,
    int64_t* __restrict__ req_out,
    int32_t* __restrict__ write_start_out,
    int32_t* __restrict__ tail_logical_start_out,
    int64_t* __restrict__ write_loc_out,
    int32_t* __restrict__ pool_seqlens_per_q_out,
    int32_t* __restrict__ seqlens_per_q_out,
    int bs,
    int real_page_table_stride,
    int pool_size,
    int num_draft_tokens,
    int slots_per_page,
    bool has_per_q) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_threads = gridDim.x * blockDim.x;

  for (int b = tid; b < bs; b += total_threads) {
    const int32_t ws = write_start[b];
    const int64_t req = req_pool_indices[b];
    const int32_t base_pool = ws / pool_size;

    if (has_per_q) {
      for (int k = 0; k < num_draft_tokens; ++k) {
        const int row = b * num_draft_tokens + k;
        const int32_t seqlen_per_q = ws + k + 1;
        seqlens_per_q_out[row] = seqlen_per_q;
        pool_seqlens_per_q_out[row] = seqlen_per_q / pool_size;
      }
    }

    const int32_t tail_logical_start = base_pool * pool_size;
    const int32_t pool_page_group = base_pool / slots_per_page;
    const int32_t packed_page = real_page_table[(b * num_draft_tokens) * real_page_table_stride + pool_page_group];
    const int64_t write_loc = static_cast<int64_t>(packed_page) * slots_per_page + (base_pool % slots_per_page);

    req_out[b] = req;
    write_start_out[b] = ws;
    tail_logical_start_out[b] = tail_logical_start;
    write_loc_out[b] = write_loc;
  }
}

void check_tensor(const torch::Tensor& tensor, const char* name, c10::ScalarType dtype, int64_t min_dim) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has unexpected dtype");
  TORCH_CHECK(tensor.dim() >= min_dim, name, " has unexpected rank");
}

}  // namespace

void kpool_write_plan(
    const torch::Tensor& write_start,
    const torch::Tensor& req_pool_indices,
    const torch::Tensor& real_page_table,
    torch::Tensor& req_out,
    torch::Tensor& write_start_out,
    torch::Tensor& tail_logical_start_out,
    torch::Tensor& write_loc_out,
    std::optional<torch::Tensor> pool_seqlens_per_q_out,
    std::optional<torch::Tensor> seqlens_per_q_out,
    int64_t pool_size,
    int64_t num_draft_tokens,
    int64_t slots_per_page) {
  check_tensor(write_start, "write_start", torch::kInt32, 1);
  check_tensor(req_pool_indices, "req_pool_indices", torch::kInt64, 1);
  check_tensor(real_page_table, "real_page_table", torch::kInt32, 2);
  check_tensor(req_out, "req_out", torch::kInt64, 1);
  check_tensor(write_start_out, "write_start_out", torch::kInt32, 1);
  check_tensor(tail_logical_start_out, "tail_logical_start_out", torch::kInt32, 1);
  check_tensor(write_loc_out, "write_loc_out", torch::kInt64, 1);

  const bool has_per_q = pool_seqlens_per_q_out.has_value();
  TORCH_CHECK(
      has_per_q == seqlens_per_q_out.has_value(),
      "pool_seqlens_per_q_out and seqlens_per_q_out must both be set or both be None");

  const int bs = static_cast<int>(write_start.size(0));
  if (bs == 0 || num_draft_tokens == 0) {
    return;
  }
  TORCH_CHECK(num_draft_tokens <= pool_size, "num_draft_tokens must be <= pool_size");
  TORCH_CHECK(req_pool_indices.size(0) >= bs, "req_pool_indices is too small");
  TORCH_CHECK(req_out.size(0) >= bs, "req_out is too small");
  TORCH_CHECK(write_start_out.size(0) >= bs, "write_start_out is too small");
  TORCH_CHECK(tail_logical_start_out.size(0) >= bs, "tail_logical_start_out is too small");
  TORCH_CHECK(write_loc_out.size(0) >= bs, "write_loc_out is too small");
  TORCH_CHECK(real_page_table.size(0) >= bs * num_draft_tokens, "real_page_table has too few rows");

  int32_t* pool_seqlens_ptr = nullptr;
  int32_t* seqlens_ptr = nullptr;
  if (has_per_q) {
    auto& pool_seqlens = *pool_seqlens_per_q_out;
    auto& seqlens = *seqlens_per_q_out;
    check_tensor(pool_seqlens, "pool_seqlens_per_q_out", torch::kInt32, 1);
    check_tensor(seqlens, "seqlens_per_q_out", torch::kInt32, 1);
    TORCH_CHECK(pool_seqlens.size(0) >= bs * num_draft_tokens, "pool_seqlens_per_q_out is too small");
    TORCH_CHECK(seqlens.size(0) >= bs * num_draft_tokens, "seqlens_per_q_out is too small");
    pool_seqlens_ptr = pool_seqlens.data_ptr<int32_t>();
    seqlens_ptr = seqlens.data_ptr<int32_t>();
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(write_start));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int blocks = std::min((bs + kThreadsPerBlock - 1) / kThreadsPerBlock, kMaxGridSize);
  kpool_write_plan_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      write_start.data_ptr<int32_t>(),
      req_pool_indices.data_ptr<int64_t>(),
      real_page_table.data_ptr<int32_t>(),
      req_out.data_ptr<int64_t>(),
      write_start_out.data_ptr<int32_t>(),
      tail_logical_start_out.data_ptr<int32_t>(),
      write_loc_out.data_ptr<int64_t>(),
      pool_seqlens_ptr,
      seqlens_ptr,
      bs,
      static_cast<int>(real_page_table.stride(0)),
      static_cast<int>(pool_size),
      static_cast<int>(num_draft_tokens),
      static_cast<int>(slots_per_page),
      has_per_q);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
