// Copyright (c) 2026 gencheng liu
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

void w4a8_mmac_contiguous_out_hip(
    const torch::Tensor& a,
    const torch::Tensor& a_scale,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const torch::Tensor& m_indices,
    torch::Tensor& workspace,
    torch::Tensor& out,
    int64_t kernel_variant);

void w4a8_mmac_masked_out_hip(
    const torch::Tensor& a,
    const torch::Tensor& a_scale,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const torch::Tensor& masked_m,
    torch::Tensor& workspace,
    torch::Tensor& out,
    int64_t metadata_rows,
    int64_t kernel_variant);

namespace {
void w4a8_mmac_contiguous_out(
    const torch::Tensor& a,
    const torch::Tensor& a_scale,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const torch::Tensor& m_indices,
    torch::Tensor workspace,
    torch::Tensor out,
    int64_t kernel_variant) {
  TORCH_CHECK(
      a.is_cuda() && a.scalar_type() == torch::kInt8 && a.dim() == 2 && a.is_contiguous(),
      "a must be contiguous GPU int8 [M,K]");
  TORCH_CHECK(
      a_scale.is_cuda() && a_scale.scalar_type() == torch::kFloat32 && a_scale.is_contiguous() &&
          a_scale.numel() == a.size(0),
      "a_scale must contain one contiguous float32 value per row");
  TORCH_CHECK(
      weight.is_cuda() && weight.scalar_type() == torch::kInt8 && weight.dim() == 3 && weight.is_contiguous(),
      "weight must be AITER-shuffled contiguous GPU int8 [E,N,K/2]");
  TORCH_CHECK(weight.size(2) * 2 == a.size(1), "weight K dimension does not match a");
  TORCH_CHECK(
      weight_scale.is_cuda() && weight_scale.scalar_type() == torch::kFloat32 && weight_scale.is_contiguous() &&
          (weight_scale.dim() == 2 || (weight_scale.dim() == 3 && weight_scale.size(2) == 1)) &&
          weight_scale.size(0) == weight.size(0) && weight_scale.size(1) == weight.size(1) &&
          weight_scale.numel() == weight.size(0) * weight.size(1),
      "weight_scale must be contiguous float32 [E,N] or [E,N,1]");
  TORCH_CHECK(
      m_indices.is_cuda() && m_indices.scalar_type() == torch::kInt32 && m_indices.dim() == 1 &&
          m_indices.is_contiguous() && m_indices.numel() == a.size(0),
      "m_indices must be contiguous GPU int32 [M]");
  TORCH_CHECK(a.size(0) % 32 == 0, "M must be divisible by BLOCK_M=32 (DeepEP aligns to 256)");
  TORCH_CHECK(a.size(1) == 2048 || a.size(1) == 4096, "optimized MMAC kernel supports K=2048 or K=4096");
  TORCH_CHECK(weight.size(1) % 512 == 0, "optimized MMAC kernel requires N divisible by 512");
  const auto required_workspace = a.size(0) + a.size(0) / 32 + 1;
  TORCH_CHECK(
      workspace.is_cuda() && workspace.scalar_type() == torch::kInt32 && workspace.is_contiguous() &&
          workspace.numel() >= required_workspace,
      "workspace must be contiguous GPU int32 with at least ",
      required_workspace,
      " elements");
  TORCH_CHECK(
      out.is_cuda() && out.scalar_type() == torch::kBFloat16 && out.is_contiguous() && out.dim() == 2 &&
          out.size(0) == a.size(0) && out.size(1) == weight.size(1),
      "out must be contiguous GPU bfloat16 [M,N]");
  TORCH_CHECK(
      a.device() == a_scale.device() && a.device() == weight.device() && a.device() == weight_scale.device() &&
          a.device() == m_indices.device() && a.device() == workspace.device() && a.device() == out.device(),
      "all tensors must be on the same GPU");
  // Auto-tuned on gfx936/BW1000.  DeepEP-normal pads each active expert to
  // 256 rows, so M is a stable dispatch key and remains known on the host.
  const auto resolved_variant =
      kernel_variant < 0 ? ((a.size(1) == 4096 ? a.size(0) <= 1024 : a.size(0) <= 2048) ? 11 : 15) : kernel_variant;
  w4a8_mmac_contiguous_out_hip(a, a_scale, weight, weight_scale, m_indices, workspace, out, resolved_variant);
}

void w4a8_mmac_masked_out(
    const torch::Tensor& a,
    const torch::Tensor& a_scale,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const torch::Tensor& masked_m,
    torch::Tensor workspace,
    torch::Tensor out,
    int64_t metadata_rows,
    int64_t kernel_variant) {
  TORCH_CHECK(
      a.is_cuda() && a.scalar_type() == torch::kInt8 && a.dim() == 3 && a.is_contiguous(),
      "a must be contiguous GPU int8 [E,T,K]");
  const auto experts = a.size(0);
  const auto rows_per_expert = a.size(1);
  const auto k = a.size(2);
  const auto total_rows = experts * rows_per_expert;
  TORCH_CHECK(experts > 0 && experts <= 256, "masked MMAC supports 1..256 local experts");
  TORCH_CHECK(
      rows_per_expert > 0 && rows_per_expert % 16 == 0,
      "masked MMAC requires a positive DeepEP row capacity divisible by 16");
  TORCH_CHECK(
      a_scale.is_cuda() && a_scale.scalar_type() == torch::kFloat32 && a_scale.is_contiguous() &&
          a_scale.numel() == total_rows,
      "a_scale must contain one contiguous float32 value per row");
  TORCH_CHECK(
      weight.is_cuda() && weight.scalar_type() == torch::kInt8 && weight.dim() == 3 && weight.is_contiguous() &&
          weight.size(0) == experts && weight.size(2) * 2 == k,
      "weight must be AITER-shuffled int8 [E,N,K/2]");
  TORCH_CHECK(
      weight_scale.is_cuda() && weight_scale.scalar_type() == torch::kFloat32 && weight_scale.is_contiguous() &&
          (weight_scale.dim() == 2 || (weight_scale.dim() == 3 && weight_scale.size(2) == 1)) &&
          weight_scale.size(0) == experts && weight_scale.size(1) == weight.size(1) &&
          weight_scale.numel() == experts * weight.size(1),
      "weight_scale must be contiguous float32 [E,N] or [E,N,1]");
  TORCH_CHECK(
      masked_m.is_cuda() && masked_m.scalar_type() == torch::kInt32 && masked_m.is_contiguous() &&
          masked_m.numel() == experts,
      "masked_m must be contiguous GPU int32 [E]");
  TORCH_CHECK(k == 2048 || k == 4096, "optimized MMAC kernel supports K=2048 or K=4096");
  TORCH_CHECK(weight.size(1) % 512 == 0, "optimized MMAC kernel requires N divisible by 512");
  TORCH_CHECK(
      metadata_rows > 0 && metadata_rows <= total_rows && metadata_rows % 16 == 0,
      "metadata_rows must be a positive multiple of 16 <= E*T");
  const auto required_workspace = metadata_rows + metadata_rows / 16 + 1;
  TORCH_CHECK(
      workspace.is_cuda() && workspace.scalar_type() == torch::kInt32 && workspace.is_contiguous() &&
          workspace.numel() >= required_workspace,
      "workspace must contain at least ",
      required_workspace,
      " contiguous GPU int32 elements");
  TORCH_CHECK(
      out.is_cuda() && out.scalar_type() == torch::kBFloat16 && out.dim() == 3 && out.is_contiguous() &&
          out.size(0) == experts && out.size(1) == rows_per_expert && out.size(2) == weight.size(1),
      "out must be contiguous GPU bfloat16 [E,T,N]");
  TORCH_CHECK(
      a.device() == a_scale.device() && a.device() == weight.device() && a.device() == weight_scale.device() &&
          a.device() == masked_m.device() && a.device() == workspace.device() && a.device() == out.device(),
      "all tensors must be on the same GPU");
  // Variant 6 is the verified sparse-decode schedule across 1..48 local
  // assignments.  Keep an explicit override for the standalone tuner.
  const auto resolved_variant = kernel_variant < 0 ? 6 : kernel_variant;
  w4a8_mmac_masked_out_hip(a, a_scale, weight, weight_scale, masked_m, workspace, out, metadata_rows, resolved_variant);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "w4a8_mmac_contiguous_out",
      &w4a8_mmac_contiguous_out,
      "W4A8 contiguous grouped GEMM for DeepEP m_indices (gfx936 MMAC)",
      pybind11::arg("a"),
      pybind11::arg("a_scale"),
      pybind11::arg("weight"),
      pybind11::arg("weight_scale"),
      pybind11::arg("m_indices"),
      pybind11::arg("workspace"),
      pybind11::arg("out"),
      pybind11::arg("kernel_variant") = -1);
  module.def(
      "w4a8_mmac_masked_out",
      &w4a8_mmac_masked_out,
      "W4A8 masked grouped GEMM for DeepEP low latency (gfx936 MMAC)",
      pybind11::arg("a"),
      pybind11::arg("a_scale"),
      pybind11::arg("weight"),
      pybind11::arg("weight_scale"),
      pybind11::arg("masked_m"),
      pybind11::arg("workspace"),
      pybind11::arg("out"),
      pybind11::arg("metadata_rows"),
      pybind11::arg("kernel_variant") = -1);
}
