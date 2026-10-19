// paged_mqa_pers_jit.cu -- persistent INT8 paged-MQA (JIT build, ref: topk512_jit).
//
// Persistent rewrite of lightop's paged_mqa_logits_int8: a prologue kernel
// compacts the real chunks into a device task list and a fixed grid of CTAs
// iterates it, instead of launching (max_c4_seq_len/256) x B CTAs whose
// invalid majority only writes -INFINITY. The mmac compute body is unchanged.
//
// Invariants (do not simplify away):
//  * all warps must traverse the same barrier sequence: tail warps beyond the
//    sequence keep computing (kv_block_idx=-1 => scale 0 => sums 0) instead of
//    returning early;
//  * dynamic smem must stay exactly 16KB (the Q region is reused as KV staging
//    and again as the writeback buffer; growing it drops occupancy);
//  * the padding region of the logits row is left unwritten:
//    topk_transform_512 scans only [0, seq_len) per row.

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>

constexpr int kNextN = 1;
constexpr int kNumHeads = 64;
constexpr int kHeadDim = 128;
constexpr int BLOCK_KV = 64;
constexpr int kNumWarps = 4;
constexpr int kTokensPerWarp = 64;
constexpr int kBatchSplit = BLOCK_KV * kNumWarps;

using int8x8_t = __attribute__((__vector_size__(8 * sizeof(int8_t)))) int8_t;
using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int32_t)))) int32_t;
using v4i = int32x4_t;
union i8x16_t { int8x8_t data; struct { int8x8_t front; int8x8_t rear; }; };
#define WARP_SIZE_GPU 64
__host__ __device__ __forceinline__ constexpr int constexpr_ceil_div(int a, int b) { return (a + b - 1) / b; }
__host__ __device__ __forceinline__ constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__ void builtin_i8_mmac(const int8x8_t& a, const int8x8_t& b, int32x4_t& c) {
#if defined(__gfx936__) || defined(__gfx928__)
    const auto* pa = reinterpret_cast<const long*>(&a);
    const auto* pb = reinterpret_cast<const long*>(&b);
    c = __builtin_amdgcn_mmac_i32_16x16x32i8(*pa, *pb, c);
#endif
}

__device__ __forceinline__ void buffer_load_lds_x4(v4i* desc, uint8_t* smem_ptr, int offset) {
#if defined(__gfx936__) || defined(__gfx938__)
    __builtin_amdgcn_raw_buffer_load_lds(*desc,
        *(__attribute__((address_space(3))) int**)&smem_ptr, 16, offset, 0, 0, 0);
#endif
}

// Wait for the per-wave buffer load; a CTA barrier would desynchronize tail warps.
__device__ __forceinline__ void WAIT_VMCNT_LDS(int X) {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(%0)\n\t" :: "I"(X) :);
    __builtin_amdgcn_sched_barrier(0);
}

// Dead branch stub: kNumHeads=64 is not smaller than Q_TILE=16.
template <int NH, int QT, typename T>
__device__ __forceinline__ T load_padded_q_operand(uint8_t* smem, int offset, int row) { return T{}; }
template <int NH, int QT>
__device__ __forceinline__ float load_padded_weight(float* smem, int idx) { return 0.f; }
__attribute__((amdgpu_flat_work_group_size(1, 512))) __global__ void
mqa_pers_kernel(const signed char* q, const signed char* kv_block, const float* kv_block_scales, const float* weights,
                      const int batch_size, const int total_chunks_pers, const int64_t kv_cache_stride_bytes, const int64_t logits_stride,
                      const int64_t block_table_stride, const int* context_lens, float* logits, const int* block_table,
                      const int* task_list, const int* num_real_ptr) {
    const int& warp_idx = threadIdx.x / 64;
    const int& lane_idx = threadIdx.x % 64;
    const int& t_id = threadIdx.x;

    static constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;

    static constexpr int K_TILE = 64;
    static constexpr int Q_TILE = 16;
    static constexpr int KV_TILE = 16;
    static constexpr int Q_ITER = constexpr_ceil_div(kNumHeads, Q_TILE);
    static constexpr int KV_ITER = kTokensPerWarp / KV_TILE;
    static constexpr int PAGES_PER_WARP = kTokensPerWarp / BLOCK_KV;
    static constexpr int Stages = kHeadDim / K_TILE;
    static_assert(true, "Invalid INT8 element type");
    static_assert(kBatchSplit == BLOCK_KV * kNumWarps, "Invalid INT8 batch split");
    static_assert(BLOCK_KV == 16 || BLOCK_KV == 32 || BLOCK_KV == 64, "Invalid BLOCK_KV");
    static_assert(kTokensPerWarp % BLOCK_KV == 0 && kTokensPerWarp <= WARP_SIZE_GPU, "Invalid tokens per warp");

    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];

    auto* smem_kv_block = reinterpret_cast<uint8_t*>(smem_buffer);

    const int next_n = 0;
    const int num_real = *num_real_ptr;  // One global read per CTA
    for (int ti = blockIdx.x; ti < num_real; ti += gridDim.x) {
    const int task_id = task_list[ti];

    int q_idx = task_id / total_chunks_pers;
    int start_kv_idx = (task_id - q_idx * total_chunks_pers) * kNumWarps * PAGES_PER_WARP;
    int kv_idx = start_kv_idx + warp_idx * PAGES_PER_WARP;
    int context_len = context_lens[q_idx];

    // Calculate logits KV block offset in advance
    auto base_logits_offset = q_idx * kNextN * logits_stride + next_n * logits_stride;
    auto base_seq_kv_offset = kv_idx * BLOCK_KV;
    int num_kv = ceil_div(context_len, BLOCK_KV);

    if (context_len <= 0) {
        for (int offset = t_id; offset < logits_stride; offset += blockDim.x) {
            logits[base_logits_offset + offset] = -INFINITY;
        }
        __syncthreads();
        continue;
    }

    // Non-empty rows have start_kv_idx < num_kv; empty rows use one task
    // only to initialize their output.

    // fetch Q && Q weights
    auto gQ = q + q_idx * kNextN * kNumHeads * kHeadDim + next_n * kNumHeads * kHeadDim;
    constexpr int kAlignmentQ = 16 / sizeof(uint8_t);
    constexpr int fetch_q_cnt = constexpr_ceil_div(kNumHeads * kHeadDim, kNumWarps * WARP_SIZE_GPU * kAlignmentQ);
    constexpr int fetch_q_stride = kAlignmentQ * kNumWarps * WARP_SIZE_GPU;

#pragma unroll
    for (int i = 0; i < fetch_q_cnt; ++i) {
        auto pos = t_id * kAlignmentQ + i * fetch_q_stride;
        if (pos + kAlignmentQ <= kNumHeads * kHeadDim) {
            *reinterpret_cast<i8x16_t*>(&smem_kv_block[pos]) = *reinterpret_cast<const i8x16_t*>(&gQ[pos]);
        }
    }

    auto gW = weights + q_idx * kNextN * kNumHeads + next_n * kNumHeads;
    auto smem_weight = reinterpret_cast<float*>(smem_kv_block + kNumHeads * kHeadDim);
    if (warp_idx == 0 && lane_idx < kNumHeads) {
        smem_weight[lane_idx] = gW[lane_idx];
    }

    // fetch kv block idx
    int kv_block_idx[PAGES_PER_WARP];
    uint64_t block_table_offset = q_idx * block_table_stride + kv_idx;
#pragma unroll
    for (int i = 0; i < PAGES_PER_WARP; ++i) {
        kv_block_idx[i] = kv_idx + i < num_kv ? block_table[block_table_offset + i] : -1;
    }

    // fetch kv block scales
    const int kv_scale_stride = kv_cache_stride_bytes / sizeof(float);
    auto* smem_kv_scales = reinterpret_cast<float*>(smem_kv_block + kNumHeads * kHeadDim + kNumHeads * sizeof(float));
    if (lane_idx < kTokensPerWarp) {
        const int lane_page_delta = lane_idx / BLOCK_KV;
        const int lane_row_in_page = lane_idx % BLOCK_KV;
        const int lane_block_idx = kv_block_idx[lane_page_delta];
        if (lane_block_idx != -1) {
            auto gKv_scales = kv_block_scales + lane_block_idx * kv_scale_stride;
            smem_kv_scales[warp_idx * kTokensPerWarp + lane_idx] = gKv_scales[lane_row_in_page];
        } else {
            smem_kv_scales[warp_idx * kTokensPerWarp + lane_idx] = 0.0f;
        }
    }

    constexpr int heads_cnt = Q_ITER;
    constexpr int K_BLOCK_MAX = kHeadDim / K_TILE;
    i8x16_t reg_q_operand[heads_cnt][K_BLOCK_MAX];

    __syncthreads();
    // pre-fetch Q from smem
    auto base_fetch_lds_Q = lane_idx % 16 * kHeadDim + lane_idx / 16 * 16;
#pragma unroll
    for (int i = 0; i < heads_cnt; ++i) {
#pragma unroll
        for (int j = 0; j < K_BLOCK_MAX; ++j) {
            auto offset = base_fetch_lds_Q + i * Q_TILE * kHeadDim + j * K_TILE;
            if constexpr (kNumHeads < Q_TILE) {
                reg_q_operand[i][j] = load_padded_q_operand<kNumHeads, Q_TILE, i8x16_t>(
                    smem_kv_block, offset, i * Q_TILE + lane_idx % Q_TILE);
            } else {
                reg_q_operand[i][j] = *reinterpret_cast<i8x16_t*>(&smem_kv_block[offset]);
            }
        }
    }

    // pre-fetch weights
    float reg_weights[heads_cnt * 4];
#pragma unroll
    for (int j = 0; j < Q_ITER; ++j) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            if constexpr (kNumHeads < Q_TILE) {
                reg_weights[j * 4 + k] = load_padded_weight<kNumHeads, Q_TILE>(
                    smem_weight, j * Q_TILE + k * 4 + lane_idx / Q_TILE);
            } else {
                reg_weights[j * 4 + k] = smem_weight[j * 16 + k * 4 + lane_idx / 16];
            }
        }
    }

    // pre-fetch kv_scale
    float kv_scale[KV_ITER];
    const int logits_t_offset = warp_idx * kTokensPerWarp + lane_idx % 16;
#pragma unroll
    for (int i = 0; i < KV_ITER; ++i) {
        kv_scale[i] = smem_kv_scales[logits_t_offset + i * 16];
    }

    __syncthreads();


    // fetch kv block
    constexpr int kAlignmentKV = 16 / sizeof(uint8_t);
    constexpr int prefetch_kv_stage = KV_ITER >= 2 ? 2 : 1;
    constexpr int smem_kv_block_per_warp = prefetch_kv_stage * KV_TILE * kHeadDim;

    auto smem_warp_kv_start = smem_kv_block + warp_idx * smem_kv_block_per_warp;

#pragma unroll
    for (int i = 0; i < prefetch_kv_stage; ++i) {
        const int page_delta = i * KV_TILE / BLOCK_KV;
        const int row_in_page = i * KV_TILE % BLOCK_KV;
        const int current_block_idx = kv_block_idx[page_delta];
        auto gKv =
            current_block_idx != -1 ? kv_block + (int64_t)current_block_idx * BLOCK_KV * (kHeadDim + 4) : kv_block;
        long glob_kv_desc[2];
        glob_kv_desc[0] = *reinterpret_cast<const long*>(&gKv);
        glob_kv_desc[1] = (long)0x20000 << 32 | 0xFFFFFFFE;
        auto* src = reinterpret_cast<v4i*>(glob_kv_desc);
        auto glob_stage_offset = row_in_page * kHeadDim;
        auto smem_per_stage = smem_warp_kv_start + i * KV_TILE * kHeadDim;
#pragma unroll
        for (int j = 0; j < Stages; ++j) {
            auto smem_ptr = smem_per_stage + j * KV_TILE * K_TILE;
            auto inner_warp_offset = current_block_idx != -1 ? glob_stage_offset + lane_idx % 4 * kAlignmentKV +
                                                                   lane_idx / 4 * kHeadDim + j * K_TILE
                                                             : -1;
            buffer_load_lds_x4(src, smem_ptr, inner_warp_offset);
        }
    }

    // pre-fetch KV from smem
    i8x16_t reg_kv_operand;

    auto base_fetch_lds_KV = lane_idx % 16 * K_TILE + lane_idx / 16 * 16;

    int32x4_t acc[Q_ITER][KV_ITER];
#pragma unroll
    for (int q_iter = 0; q_iter < Q_ITER; ++q_iter) {
#pragma unroll
        for (int kv_iter = 0; kv_iter < KV_ITER; ++kv_iter) {
            acc[q_iter][kv_iter] = {0, 0, 0, 0};
        }
    }
    float sum_result[KV_ITER] = {0.f};
    constexpr int num_issue_kv_prefetch = Stages * prefetch_kv_stage;

#pragma unroll
    for (uint32_t kv_iter = 0; kv_iter < KV_ITER - prefetch_kv_stage; ++kv_iter) {
        const uint32_t load_kv_iter = kv_iter + prefetch_kv_stage;
        const int page_delta = load_kv_iter * KV_TILE / BLOCK_KV;
        const int row_in_page = load_kv_iter * KV_TILE % BLOCK_KV;
        const int current_block_idx = kv_block_idx[page_delta];
        auto gKv =
            current_block_idx != -1 ? kv_block + (int64_t)current_block_idx * BLOCK_KV * (kHeadDim + 4) : kv_block;
        long glob_kv_desc[2];
        glob_kv_desc[0] = *reinterpret_cast<const long*>(&gKv);
        glob_kv_desc[1] = (long)0x20000 << 32 | 0xFFFFFFFE;
        auto* src = reinterpret_cast<v4i*>(glob_kv_desc);
        auto glob_stage_offset = row_in_page * kHeadDim;
        auto smem_per_stage = smem_warp_kv_start + kv_iter % prefetch_kv_stage * KV_TILE * kHeadDim;
#pragma unroll
        for (uint32_t k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            auto fetch_lds_kv = base_fetch_lds_KV + k_block * K_TILE * KV_TILE;
            WAIT_VMCNT_LDS(num_issue_kv_prefetch - 1);

            int lds_addr = reinterpret_cast<size_t>(smem_per_stage + fetch_lds_kv);
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("\n ds_read_b128 %0 ,%1\n\t"
                         "s_waitcnt lgkmcnt(0) \n\t"
                         : "=v"(reg_kv_operand)
                         : "v"(lds_addr)
                         :);
            __builtin_amdgcn_sched_barrier(0);

            auto smem_ptr = smem_per_stage + k_block * KV_TILE * K_TILE;
            int inner_warp_offset = current_block_idx != -1 ? glob_stage_offset + lane_idx % 4 * kAlignmentKV +
                                                                  lane_idx / 4 * kHeadDim + k_block * K_TILE
                                                            : -1;
            buffer_load_lds_x4(src, smem_ptr, inner_warp_offset);

#pragma unroll
            for (int i = 0; i < Q_ITER; ++i) {
                builtin_i8_mmac(reg_kv_operand.front, reg_q_operand[i][k_block].front, acc[i][kv_iter]);
                builtin_i8_mmac(reg_kv_operand.rear, reg_q_operand[i][k_block].rear, acc[i][kv_iter]);
            }

            if (k_block == K_BLOCK_MAX - 1) {
                const auto& transform = [&](const uint32_t& j, const uint32_t& k, const int32_t& value) {
                    return fmaxf(static_cast<float>(value), 0.0f) * reg_weights[j * 4 + k];
                };

                auto& sum = sum_result[kv_iter];
                // Intra-thread reduction
                for (int j = 0; j < Q_ITER; ++j) {
                    for (int k = 0; k < 4; ++k) {
                        sum += transform(j, k, acc[j][kv_iter][k]);
                    }
                }

                sum *= kv_scale[kv_iter];

// Inter-thread reduction
#pragma unroll
                for (uint32_t shfl_idx = 16, j = 0; j < 2; ++j, shfl_idx = shfl_idx << 1) {
                    sum += __shfl_down(sum, shfl_idx);
                }
            }
        }
    }

#pragma unroll
    for (uint32_t kv_iter = KV_ITER - prefetch_kv_stage; kv_iter < KV_ITER; ++kv_iter) {
        auto smem_per_stage = smem_warp_kv_start + kv_iter % prefetch_kv_stage * KV_TILE * kHeadDim;
#pragma unroll
        for (uint32_t k_block = 0; k_block < Stages; ++k_block) {
            auto fetch_lds_kv = base_fetch_lds_KV + k_block * K_TILE * KV_TILE;
            WAIT_VMCNT_LDS(num_issue_kv_prefetch - (kv_iter - (KV_ITER - prefetch_kv_stage)) * Stages - k_block - 1);

            int lds_addr = reinterpret_cast<size_t>(smem_per_stage + fetch_lds_kv);
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("\n ds_read_b128 %0 ,%1\n\t"
                         "s_waitcnt lgkmcnt(0) \n\t"
                         : "=v"(reg_kv_operand)
                         : "v"(lds_addr)
                         :);
            __builtin_amdgcn_sched_barrier(0);

#pragma unroll
            for (int i = 0; i < Q_ITER; ++i) {
                builtin_i8_mmac(reg_kv_operand.front, reg_q_operand[i][k_block].front, acc[i][kv_iter]);
                builtin_i8_mmac(reg_kv_operand.rear, reg_q_operand[i][k_block].rear, acc[i][kv_iter]);
            }

            if (k_block == K_BLOCK_MAX - 1) {
                const auto& transform = [&](const uint32_t& j, const uint32_t& k, const int32_t& value) {
                    return fmaxf(static_cast<float>(value), 0.0f) * reg_weights[j * 4 + k];
                };

                auto& sum = sum_result[kv_iter];
                // Intra-thread reduction
                for (int j = 0; j < Q_ITER; ++j) {
                    for (int k = 0; k < 4; ++k) {
                        sum += transform(j, k, acc[j][kv_iter][k]);
                    }
                }

                sum *= kv_scale[kv_iter];

// Inter-thread reduction
#pragma unroll
                for (uint32_t shfl_idx = 16, j = 0; j < 2; ++j, shfl_idx = shfl_idx << 1) {
                    sum += __shfl_down(sum, shfl_idx);
                }
            }
        }
    }

    auto* smem_write_back = reinterpret_cast<float*>(smem_kv_block);
    if (lane_idx < 16) {
#pragma unroll
        for (int i = 0; i < KV_ITER; ++i) {
            smem_write_back[warp_idx * kTokensPerWarp + i * KV_TILE + lane_idx % 16] = sum_result[i];
        }
    }

    // LDS ordering: wait for ds_write before ds_read.
    asm volatile("s_waitcnt lgkmcnt(0)\n\t" ::: "memory");
    float permuted_result = lane_idx < kTokensPerWarp ? smem_write_back[warp_idx * kTokensPerWarp + lane_idx] : 0.0f;

    // Store into the global memory
    auto seq_kv_offset = base_seq_kv_offset + lane_idx;
    if (lane_idx < kTokensPerWarp && kv_idx < num_kv && seq_kv_offset < context_len - (kNextN - next_n) + 1) {
        logits[base_logits_offset + seq_kv_offset] = permuted_result;
    } else if (lane_idx < kTokensPerWarp && seq_kv_offset < logits_stride) {
        logits[base_logits_offset + seq_kv_offset] = -INFINITY;
    }

    __syncthreads();  // Reuse the writeback region for the next task.
    } // persistent task loop
}

// ===== host wrapper =====
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>

extern "C" int mqa_pers_check(void) { return 42; }

// Prologue: one CTA compacts real chunks into a task list and writes num_real.
__global__ void mqa_compact_kernel(const int* context_lens, int B, int chunks_per_q,
                                   int* task_list, int* num_real_out) {
    extern __shared__ int prefix[];  // B+1 ints
    const int t = threadIdx.x;
    if (t == 0) prefix[0] = 0;
    for (int qq = t; qq < B; qq += blockDim.x) {
        const int len = context_lens[qq];
        const int pages = (len + 63) >> 6;
        // Keep one task for an empty row so its output is initialized to -INF.
        prefix[qq + 1] = pages > 0 ? (pages + 3) >> 2 : 1;
    }
    __syncthreads();
    if (t == 0) {
        int acc = 0;
        for (int qq = 1; qq <= B; ++qq) { acc += prefix[qq]; prefix[qq] = acc; }
        *num_real_out = acc;
    }
    __syncthreads();
    for (int qq = t; qq < B; qq += blockDim.x) {
        const int base = prefix[qq], cnt = prefix[qq + 1] - base;
        for (int i = 0; i < cnt; ++i) task_list[base + i] = qq * chunks_per_q + i;
    }
}

torch::Tensor mqa_pers_host(
    torch::Tensor q, torch::Tensor kv_cache, torch::Tensor weights,
    torch::Tensor context_lens, torch::Tensor block_table,
    int64_t max_context_len, int64_t num_sms, int64_t fill_upto) {

    const int B = q.size(0);
    const int bkv = kv_cache.size(1);
    const int nblk = kv_cache.size(0);
    const int64_t kvs = (int64_t)kv_cache.stride(0);
    const int aligned = ((int)max_context_len + bkv - 1) / bkv * bkv;

    auto kv_vals = torch::from_blob(kv_cache.data_ptr(),
        {(int64_t)nblk, (int64_t)bkv, (int64_t)128}, {(int64_t)kvs, 128, 1},
        torch::TensorOptions().dtype(torch::kInt8));
    auto kv_sc = torch::from_blob(
        reinterpret_cast<uint8_t*>(kv_cache.data_ptr()) + (int64_t)bkv * 128,
        {(int64_t)nblk, (int64_t)bkv}, {(int64_t)(kvs / 4), 1},
        torch::TensorOptions().dtype(torch::kFloat32));

    // Optional prefix fill; zero means no fill because TopK bounds its reads.
    auto logits = torch::empty({B, (int64_t)aligned},
        torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));
    if (fill_upto > 0) {
        logits.narrow(1, 0, (int64_t)fill_upto).fill_(-INFINITY);
    }
    if (B == 0 || max_context_len <= 0) {
        return logits;
    }

    const int chunks = ((int)max_context_len + 255) / 256;
    const int total = B * chunks;
    dim3 grid((unsigned)std::min(num_sms, (int64_t)total), 1, 1);
    dim3 block(256, 1, 1);
    const int smem = 4 * 2 * 16 * 128;  // Reuse staging and writeback to keep 4 CTAs/CU.
    auto stream = at::hip::getCurrentHIPStream();

    // Launch the prologue and main kernel in order on the same stream.
    auto task_list = torch::empty({(int64_t)total},
        torch::TensorOptions().dtype(torch::kInt32).device(q.device()));
    auto num_real = torch::empty({1},
        torch::TensorOptions().dtype(torch::kInt32).device(q.device()));
    mqa_compact_kernel<<<dim3(1,1,1), dim3(256,1,1), (B + 1) * sizeof(int), stream>>>(
        context_lens.data_ptr<int>(), B, chunks,
        task_list.data_ptr<int>(), num_real.data_ptr<int>());

    mqa_pers_kernel<<<grid, block, smem, stream>>>(
        (const signed char*)q.data_ptr(),
        (const signed char*)kv_vals.data_ptr(),
        kv_sc.data_ptr<float>(), weights.data_ptr<float>(),
        B, chunks, kvs, (int64_t)aligned,
        (int64_t)block_table.stride(0),
        context_lens.data_ptr<int>(),
        logits.data_ptr<float>(), block_table.data_ptr<int>(),
        task_list.data_ptr<int>(), num_real.data_ptr<int>());

    return logits;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("persistent", &mqa_pers_host, "persistent int8 paged mqa v6");
    m.def("check", &mqa_pers_check, "sanity check");
}
