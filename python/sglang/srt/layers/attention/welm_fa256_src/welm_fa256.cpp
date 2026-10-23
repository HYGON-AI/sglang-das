// Standalone hdim256 bf16 varlen causal forward with a faster tile config
// (128x64, 2 stages) for WeLM full-attention prefill. Built separately from the
// vendor libflash_attention.so (hidden visibility, -Bsymbolic).
#include <torch/extension.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/HIPContext.h>
#include "src/flash_fwd_launch_template.h"
#define CHECK_DEVICE(x)     TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == at::IntArrayRef({__VA_ARGS__}), #x " bad shape")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
namespace {
void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      const int d_v,
                      const int d_v_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      float softcap=0.0,
                      bool seqlenq_ngroups_swapped=false,
                      const bool unpadded_lse=false,
                      const bool is_kvcache=false,
                      const bool is_seqlens_k_cumulative=false,
                      const int layout=0,
                      const bool is_prefix=false
                    ) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));
    params.is_bf16 = q.dtype() == at::ScalarType::BFloat16;

    params.is_e4m3 = q.dtype() == at::ScalarType::Float8_e4m3fn;
    params.is_e5m2 = q.dtype() == at::ScalarType::Float8_e5m2;

    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    // All stride are in elements, not bytes.
    params.o_ptr = out.data_ptr();
    params.layout = layout;
    if (cu_seqlens_k_d == nullptr and !is_kvcache) {
        params.q_batch_stride = q.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
        params.o_batch_stride = out.stride(0);

        params.q_row_stride = params.layout ? q.stride(1): q.stride(2);
        params.k_row_stride = params.layout ? k.stride(1): k.stride(2);
        params.v_row_stride = params.layout ? v.stride(1): v.stride(2);
        params.o_row_stride = params.layout ? out.stride(1): out.stride(2);

        params.q_head_stride = params.layout ? q.stride(2): q.stride(1);
        params.k_head_stride = params.layout ? k.stride(2): k.stride(1);
        params.v_head_stride = params.layout ? v.stride(2): v.stride(1);
        params.o_head_stride = params.layout ? out.stride(2): out.stride(1);
        params.is_seqlens_k_cumulative = false;
        // params.varlen_proj_qkv_head = h; // uniform computation to reduce vgpr/sgpr
    }
    else {
        params.is_seqlens_k_cumulative = is_seqlens_k_cumulative;
        if (is_kvcache) {
            // when kvcache, q/o shape is different from training/prefill
            params.q_batch_stride = q.stride(0);
            params.o_batch_stride = out.stride(0);
            params.q_head_stride  = (layout == 1) ? q.stride(2): q.stride(1);
            params.k_head_stride  = (layout == 1) ? k.stride(2): k.stride(1);
            params.v_head_stride  = (layout == 1) ? v.stride(2): v.stride(1);
            params.o_head_stride  = (layout == 1) ? out.stride(2): out.stride(1);
            params.q_row_stride   = (layout == 1) ? q.stride(1): q.stride(2);
            params.k_row_stride   = (layout == 1) ? k.stride(1): k.stride(2);
            params.v_row_stride   = (layout == 1) ? v.stride(1): v.stride(2);
            params.o_row_stride   = (layout == 1) ? out.stride(1): out.stride(2);
        } else if (is_prefix) {
            if (params.layout == 1) {
                params.q_head_stride = q.stride(-2);
                params.k_head_stride = k.stride(-2);
                params.v_head_stride = v.stride(-2);
                params.o_head_stride = out.stride(1);
                params.q_row_stride  = q.stride(0);
                params.k_row_stride  = k.stride(1);
                params.v_row_stride  = v.stride(1);
                params.o_row_stride  = out.stride(0);
            } else {
                params.q_head_stride = q.stride(1);
                params.k_head_stride = k.stride(1);
                params.v_head_stride = v.stride(1);
                params.o_head_stride = out.stride(1);
                params.q_row_stride  = q.stride(0);
                params.k_row_stride  = k.stride(2);
                params.v_row_stride  = v.stride(2);
                params.o_row_stride  = out.stride(0);
            }
        } else {
            params.q_head_stride = params.layout ? q.stride(-2): q.stride(0);
            params.k_head_stride = params.layout ? k.stride(-2): k.stride(0);
            params.v_head_stride = params.layout ? v.stride(-2): v.stride(0);
            params.o_head_stride = params.layout ? out.stride(1): out.stride(0);
            params.q_row_stride = params.layout ? q.stride(0): params.q_head_stride/*also .stride(0)*/;
            params.k_row_stride = params.layout ? k.stride(0): params.k_head_stride;
            params.v_row_stride = params.layout ? v.stride(0): params.v_head_stride;
            params.o_row_stride = params.layout ? out.stride(0): params.o_head_stride;
            // params.varlen_proj_qkv_head = params.layout ? k.stride(-3) / k.stride(-2): 0;
            // in vllm, K and V is not contiguous due to rope, but Q is contiguous. However, in some sceniros, K is contiguous but V is not contiguous()
        }
    }
    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_k = static_cast<int *>(seqused_k);
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;
    params.d_value = d_v;
    params.d_value_rounded = d_v_rounded;
    params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;

    // Set the different scale values.
    #ifdef FLASHATTENTION_DISABLE_SOFTCAP
        TORCH_CHECK(softcap <= 0.0, "This flash attention build does not support softcap.");
    #endif
    if (softcap > 0.0) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else{
        // Remove potential NaN
        params.softcap = 0.0;
        // Set the different scale values.
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }
    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    TORCH_CHECK(p_dropout < 1.f);

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0;

    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;
}


template <int CFG>
void launch(Flash_fwd_params &params, hipStream_t stream) {
    using T = BFloat16;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        if constexpr (CFG == 4) {
            run_flash_fwd<Flash_fwd_kernel_traits<256, 256, 128, 64, 32, 32, 32, 2, false, false, T>, false, Is_causal>(params, stream);
        } else if constexpr (CFG == 20) {
            run_flash_fwd<Flash_fwd_kernel_traits<256, 256, 128, 32, 32, 32, 32, 2, false, false, T>, false, Is_causal>(params, stream);
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<256, 256, 64, 32, 32, 32, 32, 1, false, false, T>, false, Is_causal>(params, stream);
        }
    });
}

std::vector<at::Tensor> varlen_fwd256(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                         c10::optional<at::Tensor> out_,
                         const at::Tensor &cu_seqlens_q, const at::Tensor &cu_seqlens_k,
                         int max_seqlen_q, int max_seqlen_k, double softmax_scale,
                         bool is_causal, int window_size_left, int window_size_right,
                         c10::optional<at::Tensor> s_aux_, int cfg) {
    const at::cuda::HIPGuardMasqueradingAsCUDA device_guard(q.device().index());
    if (is_causal) { window_size_right = 0; }
    if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
    TORCH_CHECK(q.dtype() == at::ScalarType::BFloat16 && k.dtype() == q.dtype() && v.dtype() == q.dtype(), "bf16 only");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "bshd 3-dim");
    TORCH_CHECK(q.size(2) == 256 && v.size(2) == 256, "hdim 256 only");
    TORCH_CHECK(cu_seqlens_q.dtype() == at::ScalarType::Int && cu_seqlens_k.dtype() == at::ScalarType::Int);
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1);
    CHECK_CONTIGUOUS(cu_seqlens_q); CHECK_CONTIGUOUS(cu_seqlens_k);
    const int num_heads = q.size(1), num_heads_k = k.size(1);
    TORCH_CHECK(num_heads % num_heads_k == 0);
    const int batch_size = cu_seqlens_q.numel() - 1;
    const int total_q = q.size(0), total_k = k.size(0);
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    at::Tensor out = out_.has_value() ? out_.value() : at::empty_like(q);
    TORCH_CHECK(out.stride(-1) == 1);
    auto opts = q.options();
    auto softmax_lse = at::empty({num_heads, total_q}, opts.dtype(at::kFloat));
    Flash_fwd_params params;
    set_params_fprop(params, batch_size, max_seqlen_q, max_seqlen_k,
                     round_multiple(max_seqlen_q, 32), round_multiple(max_seqlen_k, 32),
                     num_heads, num_heads_k, 256, 256, 256, 256,
                     q, k, v, out,
                     cu_seqlens_q.data_ptr(), cu_seqlens_k.data_ptr(), nullptr, nullptr,
                     softmax_lse.data_ptr(), 0.f, (float)softmax_scale,
                     window_size_left, window_size_right, 0.f, false,
                     /*unpadded_lse*/false, /*is_kvcache*/false,
                     /*is_seqlens_k_cumulative*/true, /*layout bshd*/1);
    params.total_q = total_q;
    params.total_k = total_k;
    auto rng_state = at::empty({2}, opts.dtype(at::ScalarType::Long));
    params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());
    params.alibi_slopes_ptr = nullptr;
    if (s_aux_.has_value()) {
        auto s_aux = s_aux_.value();
        CHECK_DEVICE(s_aux); CHECK_CONTIGUOUS(s_aux); CHECK_SHAPE(s_aux, num_heads);
        params.s_aux_ptr = s_aux.data_ptr();
        auto dt = s_aux.scalar_type();
        params.s_aux_type = dt == at::kFloat ? 1 : dt == at::kHalf ? 2 : dt == at::kBFloat16 ? 3 : 0;
        TORCH_CHECK(params.s_aux_type != 0, "sink dtype");
    }
    TORCH_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
                "only causal or full attention (no local window) supported");
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    if (cfg == 4) launch<4>(params, stream); else if (cfg == 20) launch<20>(params, stream); else launch<1>(params, stream);
    return {out, softmax_lse};
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("varlen_fwd256", &varlen_fwd256);
}
