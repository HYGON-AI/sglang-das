// Partial-output variant of the vendor paged-attention kernel for WeLM
// prefix-deduplicated decode: every (virtual sequence, query, partition)
// writes its normalized partial output plus (max, exp_sum) in log2 units,
// never the final combined output. The caller merges partials (and sinks).
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include "welm_pa_part_kernel.h"

namespace {

template <int REUSE>
void launch(torch::Tensor& out_tmp, torch::Tensor& q, torch::Tensor& kc, torch::Tensor& vc,
            torch::Tensor& bt, torch::Tensor& seq, int mtp, int partition_size, int max_parts,
            const int* q_rows, int num_seqs, int q_stride, hipStream_t stream) {
  using scalar_t = uint16_t;
  using cache_t = scalar_t;
  constexpr int HEAD_SIZE = 256, BLOCK_SIZE = 64, NUM_THREADS = 256;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int num_heads = (q_rows ? q.size(1) : q.size(2)) * mtp;
  const int num_kv_heads = kc.size(1);
  const int real_reuse = num_heads / num_kv_heads;
  int other_use = (real_reuse * NUM_WARPS + NUM_WARPS + real_reuse * 2) * sizeof(float);
  int shared_mem_size = partition_size * sizeof(scalar_t) * real_reuse + other_use;
  dim3 grid(num_kv_heads, num_seqs, max_parts);
  dim3 block(NUM_THREADS);
  hipLaunchKernelGGL((welm_pa_partial_kernel<scalar_t, cache_t, false, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, REUSE>),
      grid, block, shared_mem_size, stream,
      (scalar_t*)out_tmp.data_ptr(), (scalar_t*)out_tmp.data_ptr(), (scalar_t*)q.data_ptr(),
      (cache_t*)kc.data_ptr(), (cache_t*)vc.data_ptr(), num_heads, num_kv_heads,
      bt.data_ptr<int>(), seq.data_ptr<int>(), (int)bt.size(1), (const float*)nullptr,
      q_stride, (int)kc.stride(0), max_parts, partition_size,
      (const scalar_t*)nullptr, mtp, false, -1, -1, -1, q_rows);
}

}  // namespace

// q [NV, mtp, H, D] bf16; kc [blocks, KVH, 64, 256]; vc [blocks, KVH, 256, 64];
// bt [NV, C] int32; seq [NV] int32 (<= C*64); out_tmp: vendor tmp layout.
// q_rows: optional int32 [NV, mtp]; with it q is [B, H, D] (rows gathered in-kernel).
void pa_partial(torch::Tensor out_tmp, torch::Tensor q, torch::Tensor kc, torch::Tensor vc,
                torch::Tensor bt, torch::Tensor seq, int64_t mtp, int64_t partition_size,
                int64_t max_parts, c10::optional<torch::Tensor> q_rows_) {
  TORCH_CHECK(q.dtype() == torch::kBFloat16 && kc.dtype() == torch::kBFloat16);
  TORCH_CHECK(q.size(-1) == 256 && kc.size(2) == 64 && kc.size(3) == 256);
  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  hipStream_t stream = at::cuda::getCurrentCUDAStream();
  const int* q_rows = q_rows_.has_value() ? q_rows_.value().data_ptr<int>() : nullptr;
  const int num_seqs = bt.size(0);
  const int q_stride = q.stride(0);
  int num_heads = (q_rows ? q.size(1) : q.size(2)) * mtp;
  int reuse = get_reusekv_local(num_heads, kc.size(1));
  switch (reuse) {
    case 8: launch<8>(out_tmp, q, kc, vc, bt, seq, mtp, partition_size, max_parts, q_rows, num_seqs, q_stride, stream); break;
    case 16: launch<16>(out_tmp, q, kc, vc, bt, seq, mtp, partition_size, max_parts, q_rows, num_seqs, q_stride, stream); break;
    case 24: launch<24>(out_tmp, q, kc, vc, bt, seq, mtp, partition_size, max_parts, q_rows, num_seqs, q_stride, stream); break;
    case 32: launch<32>(out_tmp, q, kc, vc, bt, seq, mtp, partition_size, max_parts, q_rows, num_seqs, q_stride, stream); break;
    case 48: launch<48>(out_tmp, q, kc, vc, bt, seq, mtp, partition_size, max_parts, q_rows, num_seqs, q_stride, stream); break;
    default: launch<64>(out_tmp, q, kc, vc, bt, seq, mtp, partition_size, max_parts, q_rows, num_seqs, q_stride, stream); break;
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pa_partial", &pa_partial, "WeLM partial paged attention",
        py::arg("out_tmp"), py::arg("q"), py::arg("kc"), py::arg("vc"), py::arg("bt"), py::arg("seq"),
        py::arg("mtp"), py::arg("partition_size"), py::arg("max_parts"), py::arg("q_rows") = py::none());
}
