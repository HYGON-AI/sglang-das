#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>

namespace sglang {

namespace hcu_delta_rule {

constexpr int kK = 128;
constexpr int kV = 128;
constexpr int kThreads = 256;
constexpr int kLogicalWarp = 32;
constexpr int kWarps = kThreads / kLogicalWarp;
constexpr int kValuesPerWarp = 4;
constexpr int kValuesPerBlock = kWarps * kValuesPerWarp;

template <typename T>
SGL_DEVICE float to_float(T value) {
  return device::cast<fp32_t>(value);
}

template <typename T>
SGL_DEVICE T from_float(float value) {
  return device::cast<T>(value);
}

template <typename TInput, typename TState, bool kL2Norm, bool kLowerBound>
__global__ __launch_bounds__(kThreads) void fused_sigmoid_gating_delta_rule_update_hcu_kernel(
    const TInput* __restrict__ q,
    const TInput* __restrict__ k,
    const TInput* __restrict__ v,
    const TInput* __restrict__ a,
    const TInput* __restrict__ beta_input,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    TState* __restrict__ state,
    const int32_t* __restrict__ state_indices,
    TInput* __restrict__ output,
    int num_q_heads,
    int num_v_heads,
    float scale,
    float lower_bound,
    float beta_scale) {
  const int value_tile = blockIdx.x;
  const int hv = blockIdx.y;
  const int batch = blockIdx.z;
  const int warp = threadIdx.x / kLogicalWarp;
  const int lane = threadIdx.x % kLogicalWarp;
  const int value_base = value_tile * kValuesPerBlock + warp * kValuesPerWarp;
  const int q_head = hv;
  const int state_slot = state_indices[batch];

  float q_reg[4];
  float k_reg[4];
  float gate_reg[4];
  float h[kValuesPerWarp][4];

  float q_ss = 0.0f;
  float k_ss = 0.0f;
#pragma unroll
  for (int ki = 0; ki < 4; ++ki) {
    const int kk = lane + ki * kLogicalWarp;
    q_reg[ki] = to_float(q[(batch * num_q_heads + q_head) * kK + kk]);
    k_reg[ki] = to_float(k[(batch * num_q_heads + q_head) * kK + kk]);
    q_ss += q_reg[ki] * q_reg[ki];
    k_ss += k_reg[ki] * k_reg[ki];

    const float x = to_float(a[(batch * num_v_heads + hv) * kK + kk]) + dt_bias[hv * kK + kk];
    if constexpr (kLowerBound) {
      gate_reg[ki] = lower_bound / (1.0f + expf(-expf(A_log[hv]) * x));
    } else {
      const float softplus = x <= 20.0f ? log1pf(expf(x)) : x;
      gate_reg[ki] = -expf(A_log[hv]) * softplus;
    }
  }

  if constexpr (kL2Norm) {
    q_ss = device::warp::reduce_sum<kLogicalWarp>(q_ss);
    k_ss = device::warp::reduce_sum<kLogicalWarp>(k_ss);
    const float q_inv_norm = rsqrtf(q_ss + 1.0e-6f) * scale;
    const float k_inv_norm = rsqrtf(k_ss + 1.0e-6f);
#pragma unroll
    for (int ki = 0; ki < 4; ++ki) {
      q_reg[ki] *= q_inv_norm;
      k_reg[ki] *= k_inv_norm;
    }
  } else {
#pragma unroll
    for (int ki = 0; ki < 4; ++ki) q_reg[ki] *= scale;
  }

#pragma unroll
  for (int vi = 0; vi < kValuesPerWarp; ++vi) {
    const int vv = value_base + vi;
#pragma unroll
    for (int ki = 0; ki < 4; ++ki) {
      const int kk = lane + ki * kLogicalWarp;
      const int64_t offset = ((static_cast<int64_t>(state_slot) * num_v_heads + hv) * kV + vv) * kK + kk;
      h[vi][ki] = state_slot >= 0 ? to_float(state[offset]) : 0.0f;
    }
  }

  const float beta = beta_scale /
      (1.0f + expf(-to_float(beta_input[batch * num_v_heads + hv])));

#pragma unroll
  for (int vi = 0; vi < kValuesPerWarp; ++vi) {
    const int vv = value_base + vi;
    float hk = 0.0f;
#pragma unroll
    for (int ki = 0; ki < 4; ++ki) {
      h[vi][ki] *= expf(gate_reg[ki]);
      hk += h[vi][ki] * k_reg[ki];
    }
    hk = device::warp::reduce_sum<kLogicalWarp>(hk);
    const float residual =
        (to_float(v[(batch * num_v_heads + hv) * kV + vv]) - hk) * beta;

    float out = 0.0f;
#pragma unroll
    for (int ki = 0; ki < 4; ++ki) {
      h[vi][ki] += k_reg[ki] * residual;
      out += h[vi][ki] * q_reg[ki];
    }
    out = device::warp::reduce_sum<kLogicalWarp>(out);
    if (lane == 0) {
      output[(batch * num_v_heads + hv) * kV + vv] = from_float<TInput>(out);
    }
  }

  if (state_slot >= 0) {
#pragma unroll
    for (int vi = 0; vi < kValuesPerWarp; ++vi) {
      const int vv = value_base + vi;
#pragma unroll
      for (int ki = 0; ki < 4; ++ki) {
        const int kk = lane + ki * kLogicalWarp;
        const int64_t offset = ((static_cast<int64_t>(state_slot) * num_v_heads + hv) * kV + vv) * kK + kk;
        state[offset] = from_float<TState>(h[vi][ki]);
      }
    }
  }
}

}  // namespace hcu_delta_rule

template <typename TInput, typename TState>
struct FusedSigmoidGatingDeltaRuleUpdateHcu {
  static void run(
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView a,
      const tvm::ffi::TensorView beta_input,
      const tvm::ffi::TensorView A_log,
      const tvm::ffi::TensorView dt_bias,
      const tvm::ffi::TensorView state,
      const tvm::ffi::TensorView state_indices,
      const tvm::ffi::TensorView output,
      const double scale,
      const bool use_l2norm,
      const bool use_lower_bound,
      const double lower_bound,
      const double beta_scale) {
    using namespace host;
    using namespace hcu_delta_rule;

    RuntimeCheck(q.ndim() == 4 && k.ndim() == 4 && v.ndim() == 4, "q/k/v must be rank-4");
    RuntimeCheck(q.shape()[0] == 1 && k.shape()[0] == 1 && v.shape()[0] == 1, "decode requires T=1");
    RuntimeCheck(q.shape()[3] == kK && k.shape()[3] == kK && v.shape()[3] == kV, "only K=V=128 is supported");
    RuntimeCheck(q.shape()[1] == k.shape()[1] && q.shape()[1] == v.shape()[1], "batch mismatch");
    RuntimeCheck(q.shape()[2] == k.shape()[2], "q/k head mismatch");
    RuntimeCheck(q.shape()[2] == v.shape()[2], "HCU prototype requires equal q/k/v head counts");
    RuntimeCheck(a.numel() == q.shape()[1] * v.shape()[2] * kK, "invalid a shape");
    RuntimeCheck(beta_input.numel() == q.shape()[1] * v.shape()[2], "invalid beta shape");
    RuntimeCheck(A_log.numel() == v.shape()[2], "invalid A_log shape");
    RuntimeCheck(dt_bias.numel() == v.shape()[2] * kK, "invalid dt_bias shape");
    RuntimeCheck(state.ndim() == 4 && state.shape()[1] == v.shape()[2] && state.shape()[2] == kV && state.shape()[3] == kK, "invalid state shape");
    RuntimeCheck(state_indices.numel() == q.shape()[1], "invalid state_indices shape");
    RuntimeCheck(output.numel() == v.numel(), "invalid output shape");
    RuntimeCheck(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && a.is_contiguous() && beta_input.is_contiguous(), "inputs must be contiguous");
    RuntimeCheck(state.is_contiguous() && output.is_contiguous(), "state/output must be contiguous");

    const int batch = static_cast<int>(q.shape()[1]);
    const int q_heads = static_cast<int>(q.shape()[2]);
    const int v_heads = static_cast<int>(v.shape()[2]);
    if (batch == 0) return;

    const dim3 grid(kV / kValuesPerBlock, v_heads, batch);
    const dim3 block(kThreads);
    const auto device = q.device();

#define LAUNCH(L2, LOWER)                                                                                         \
  LaunchKernel(grid, block, device)(                                                                              \
      fused_sigmoid_gating_delta_rule_update_hcu_kernel<TInput, TState, L2, LOWER>,                               \
      static_cast<const TInput*>(q.data_ptr()), static_cast<const TInput*>(k.data_ptr()),                         \
      static_cast<const TInput*>(v.data_ptr()), static_cast<const TInput*>(a.data_ptr()),                         \
      static_cast<const TInput*>(beta_input.data_ptr()), static_cast<const float*>(A_log.data_ptr()),             \
      static_cast<const float*>(dt_bias.data_ptr()), static_cast<TState*>(state.data_ptr()),                       \
      static_cast<const int32_t*>(state_indices.data_ptr()), static_cast<TInput*>(output.data_ptr()),             \
      q_heads, v_heads, static_cast<float>(scale), static_cast<float>(lower_bound), static_cast<float>(beta_scale))

    if (use_l2norm) {
      if (use_lower_bound) LAUNCH(true, true);
      else LAUNCH(true, false);
    } else {
      if (use_lower_bound) LAUNCH(false, true);
      else LAUNCH(false, false);
    }
#undef LAUNCH
  }
};

}  // namespace sglang
