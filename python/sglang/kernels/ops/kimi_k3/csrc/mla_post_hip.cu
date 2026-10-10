#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

// K3 skip-RoPE post boundary. Inputs retain their BF16 GEMM/norm rounding.
// Separate Q/KV blocks share one launch and write final destinations.
template <class T> struct alignas(sizeof(T) * 4) K3Vec4 {
  T values[4];
};

template <class Dst>
__global__ void k3_mla_concat_cast_store(
    const c10::BFloat16 *qn, const c10::BFloat16 *qp, const c10::BFloat16 *kn,
    const c10::BFloat16 *kp, Dst *qout, Dst *cache, const int64_t *loc, int T,
    int H, int64_t qn_t, int64_t qn_h, int64_t qp_t, int64_t qp_h, int64_t kn_t,
    int64_t kp_t, int64_t cache_t, int64_t cache_rows, int loc_stride) {
  int row = blockIdx.x;
  const c10::BFloat16 *a;
  const c10::BFloat16 *b;
  Dst *dst;
  if (row < T * H) {
    int t = row / H, h = row % H;
    a = qn + t * qn_t + h * qn_h;
    b = qp + t * qp_t + h * qp_h;
    dst = qout + int64_t(row) * 576;
  } else {
    int t = row - T * H;
    int64_t slot = loc[int64_t(t) * loc_stride];
    // Slot zero is the allocator's dummy slot. Real slots remain untouched on
    // invalid/padded locations; FA does not consume dummy outputs.
    if (slot <= 0 || slot >= cache_rows)
      return;
    a = kn + int64_t(t) * kn_t;
    b = kp + int64_t(t) * kp_t;
    dst = cache + slot * cache_t;
  }
  constexpr int VEC = 4;
  for (int base = threadIdx.x * VEC; base < 576; base += blockDim.x * VEC) {
    const c10::BFloat16 *src = base < 512 ? a + base : b + (base - 512);
    K3Vec4<Dst> converted;
#pragma unroll
    for (int j = 0; j < VEC; ++j)
      converted.values[j] = Dst(float(src[j]));
    *reinterpret_cast<K3Vec4<Dst> *>(dst + base) = converted;
  }
}
template <class Dst>
int run(const void *qn, const void *qp, const void *kn, const void *kp,
        void *out, void *cache, const int64_t *loc, int T, int H, int64_t qnt,
        int64_t qnh, int64_t qpt, int64_t qph, int64_t knt, int64_t kpt,
        int64_t ct, int64_t rows, int ls, hipStream_t stream, int threads) {
  hipLaunchKernelGGL(k3_mla_concat_cast_store<Dst>, dim3(T * (H + 1)),
                     dim3(threads), 0, stream, (const c10::BFloat16 *)qn,
                     (const c10::BFloat16 *)qp, (const c10::BFloat16 *)kn,
                     (const c10::BFloat16 *)kp, (Dst *)out, (Dst *)cache, loc,
                     T, H, qnt, qnh, qpt, qph, knt, kpt, ct, rows, ls);
  return int(hipGetLastError());
}
extern "C" int k3_mla_post(const void *qn, const void *qp, const void *kn,
                           const void *kp, void *out, void *cache,
                           const int64_t *loc, int T, int H, int64_t qnt,
                           int64_t qnh, int64_t qpt, int64_t qph, int64_t knt,
                           int64_t kpt, int64_t ct, int64_t rows, int ls,
                           int dtype, void *stream, int threads) {
  if (T == 0)
    return 0;
#define CALL(TYPE)                                                             \
  run<TYPE>(qn, qp, kn, kp, out, cache, loc, T, H, qnt, qnh, qpt, qph, knt,    \
            kpt, ct, rows, ls, (hipStream_t)stream, threads)
  if (dtype == 0)
    return CALL(c10::BFloat16);
  if (dtype == 1)
    return CALL(c10::Float8_e4m3fn);
  if (dtype == 2)
    return CALL(c10::Float8_e4m3fnuz);
  return int(hipErrorInvalidValue);
}
extern "C" const char *k3_mla_error(int error) {
  return hipGetErrorString((hipError_t)error);
}
