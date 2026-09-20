# 官方同步总结 · 20260920

| 项 | 值 |
|---|---|
| 上次同步点 | `0d95a9c1ff14773b722009a6c6fd6ad66c7ce395`（2026-09-14） |
| 本次官方 tip | `df0dc44931433b8a3488fbd4c6489e08cdee8703`（2026-09-20） |
| 区间 commit 数 | 312 |
| 变更规模 | 1543 files, +126334 / −30981 |
| 分支 | `sync/official-main-daily-20260920`（基于 main `07f94109f3`） |
| 冲突 | 50 文件 / 50 处 |

## 0. 同步点确认

20260914 那次同步落 main 依旧是 **squash merge** `ca25bc4275 (#362)`，单 parent，
`0d95a9c1ff` 因此不是 main 的祖先，直接合并要重放 **1814** 个 commit。

`#362` 的 tree 与本地 main tip `07f94109f3` **逐字节相同**，说明内容确实已在 main。
于是 `git merge -s ours 0d95a9c1ff` 锚定（commit `357c2b148a`，tree 不变），重放量降到 **312**。
**这是连续第五次 squash 陷阱**，范式见 `fp-base-is-second-parent`。

---

## 1. 组件优化点

### 1.1 runtime context 收口：进程组只剩一条读路径（本轮对我方冲突面影响最大）

一组 5 笔连号 PR 把并行世界的读取统一到 `get_parallel()`：

- `Read process groups through the runtime context` (#40068)
- `One read path for every parallel name` (#40069)
- `Give the attention-DP width and rank one home` (#40067)
- `Name the two widths of the WORLD group` (#40070)
- `Record a process's placement at publish, not at group build` (#40071)

后果：`get_pp_group()` / `get_tp_group()` / `get_moe_ep_group()` 这类自由函数在官方侧被大面积替换为
`get_parallel().pp_group` / `.tp_group` / `.moe_ep_group`。我方 fork 里仍在用旧访问器的文件
（`minimax_m2.py`、`hunyuan_v3.py`、`deepep.py`、`dynamic_chunk_sizer.py`）因此全部冲突，
**取 theirs 时必须回补 import**，否则 F821（本轮踩到一次，见 §3.4）。

### 1.2 MegaMoE：SM 预算 + per-experts mma_type

- `Fix MegaMoE buffer allocation and caching for effective SM budgets` (#39223)：
  symm buffer 的分配和 cache key 都要进 `deep_gemm.get_num_sms()`，并用
  `_configure_mega_moe_deep_gemm_num_sms()` 在分配期临时压低 SM 预算（SM100 的整网格 clustered launch 需要驻留余量）。
- `[MegaMoE] Wire Qwen MoE blocks to DeepGEMM MegaMoE (MXFP4 and NVFP4 experts)` (#38080)：
  `_mega_moe_mma_type()` 改为接收 `experts`，按 `experts._mega_moe_nvfp4` 选 `nvfp4xnvfp4`。
- `_run_mega_routed` 拆成 moe 级 wrapper + 可复用的 `run_mega_routed_experts(experts, ...)`（Kimi K3 复用同一入口）。

### 1.3 MoE / DeepEP

- `Support MXFP8 and deferred route weighting in DeepEP v2` (#40030)、
  `[Feature] Support BF16 and batch-invariant inference with DeepEP v2` (#38160)。
  NPU 的量化 kwargs 收进 `_get_quantization_kwargs()` / `_get_npu_mxfp_quantization_kwargs()` 两个 helper，
  原先按变体展开的多份 `buffer.dispatch()` / `low_latency_dispatch()` 合并成一份。
- `[MoE] Disable FlashInfer fused finalize by default for numerical accuracy` (#40105)。
- `[Bugfix] Fix top-1 MoE routing with non-unit scaling` (#40187)。
- `[Moe] Fix flashinfer_trtllm silently dropping swiglu_limit clamped SwiGLU activation` (#39920)。
- `[NVIDIA][comm] Merge EP+MoE-TP post-experts all-reduces into one _TP reduction` (#32963)：
  EP/TP 两段 post-expert all-reduce 合成 `post_experts_all_reduce()` 单一 helper。

### 1.4 topk / 路由门

- DSV4.1 的 fused gate 新增两个 kwargs：`packed_out`（packed ids）与 `sqrtsoftplus_log1p`（数值稳定形式），
  只有 `biased_topk_jit_kernel_impl` 支持。`TopKConfig` 增加 `fused_gate_packed_ids` / `sqrtsoftplus_log1p`，
  在 `deepseek_v2.py` 里按 `model_type == "deepseek_v41"` 开启。
- `keeping router GEMM in fp32 for deterministic inference (DeepSeek V3/V4)` (#38176)。

### 1.5 HiCache（15 笔）

- `Auto-size the host pool to fit available host memory` (#40135)、`Size MHA host pools from device row width` (#40304)。
- `Back up MXFP8 KV scales in the host pool` (#39089)：新增 `MHATokenToKVPoolMXFP8Host`，
  `get_mha_host_pool_class()` 增加 MXFP8 分支（与我方 `layout_hcu` 分支并列）。
- `Rework the buffer-mode storage prefetch pipeline and retry bookkeeping` (#39283)、
  `Optimize buffer-mode storage existence bookkeeping` (#39480)。
- `[PP + HiCache] Add PP Prefetch Tickets for eager cross-stage storage prefetch` (#36700)。
- `[ROCm] Widen the HiCache JIT copy rounds and enable the K-only host pool` (#37152)。

### 1.6 内存 / Cache

- `[kv-shard 2/4] Sharded pools` (#37615)。
- `Use a shared byte budget for unified hybrid-SWA memory` (#36729)。
- `Add Agentic-Aware Tail-Optimized LRU eviction to the unified radix cache` (#34012)。
- `[Fix] Merge adjacent KV-row frees so a mid-page split under DCP cannot double-free` (#38941)。
- `[mem_cache] Release up to owned_kv_len on radix cache insert` (#40075)。
- DSV4 KV pool 引入 **KVLayout** 抽象（V4 / V4.1-FP8 / V4.1-FP4），`get_extra_key_layout(layer_id)`
  取代按 compress_ratio 硬分支；`set_key_buffer_fused()` 新增 `freqs_cis`（V4.1 在 kernel 内做 RoPE）。

### 1.7 Scheduler / PD

- `[Scheduler] Add shortest-prefill-first scheduling` (#40024)、`Count complete prefill bursts and their tokens` (#40006)。
- `Enable optimistic prefill for Mamba radix-cache models` (#40184)。
- `[PD] Introduce runtime role switching between prefill and decode` (#28403)。
- `Support unified memory page-envelope transfers in PD` (#39477)、`[PD] Add optional KV transfer checksums` (#39500)。
- `[PD] Allow decode radix cache and HiCache L1/L2 with DCP` (#40263)、
  `[PD] Enable optimistic prefill with buffer-only L3 write-through HiCache` (#40043)。
- PP 发送 proxy tensor 抽成 `_pp_send_proxy_to_next_stage()`，并把 launch event 作为
  `ready_event=` 交给 `_pp_send_dict_to_next_stage()` —— 语义上正是我方原先手写的
  `current_stream().wait_event(self.launch_event)`。

### 1.8 投机解码 / PP

- `Pipeline parallelism x speculative decoding (EAGLE/MTP) compatibility` (#30775)：
  draft 上下文外再包一层 `draft_pp_context()`。
- `[Kernel] Move CUDA and ROCm speculative kernels to JIT` (#40033)：`eagle.cuh` 迁到 `kernels/jit/include/`。
- `Use runtime token widths for Triton speculative verification` (#39859)。
- `[ROCm] Fix EAGLE spec-decode verify silently sampling greedy on HIP` (#37134)。

### 1.9 Kernel / 量化

- `[Kernel] Add OOT dispatch for clamp position` (#38687)：`clamp_position` 改为 `BaseFusedOp`
  （JIT / TORCH 两档 + `CapabilityRequirement.{CUDA,HIP}`），调度从 `forward_batch_info.py` 搬进 kernels 层。
- `[Quant] Serve 32-wide-K ue8m0 block-FP8 linears through the FlashInfer MXFP8 GEMMs` (#40039)。
- `[Kernel] Fuse hc_combine_norm for mid-size verify batches (9-96 rows)` (#40208)。
- `[Kernel] Coalesce the KDA CuTe DSL decode state transpose: ~3x faster, bit-identical` (#39680)。
- `[kernel] Share the warp vectorized copy and enforce its alignment` (#36176)：
  `warp.cuh` 删掉 `sync()` / `shfl_down()` / `shfl_xor()`，改为 `get_lane_id()` / `elect_one_lane()`。
- `feat(kv-cache): support SM100 NVFP4 GenMHA and speculative decoding` (#36340)。
- `bumping sgl-deep-gemm to 0.2.0` (#39371)。

### 1.10 Logprob / sampling / loader

- `[Logprob] Borrow graph-pool memory for input logprob logits construction` (#40007, #40038)。
- `perf(sampling): avoid GPU syncs when applying custom logit processors` (#39234)：
  `merged_custom_logit_processor` 的值从 `(processor, mask)` 二元组换成 `ProcessorEntry`。
- `[model-loader] Split weight loading from postprocessing` (#34981)：
  `load_weights_and_postprocess` 拆成 `load_weights_only()` + 静态 `postprocess_weights()`。

### 1.11 Router（22 笔）/ CI

- 错误状态派生 3 连（#39463/#39464/#39465）、fleet-wide sampling contract 3 连（#39000/#39001/#39002）、
  KV storage-tier 感知 4 连（#39108–#39111）、`--worker-queue-limit`、`--min-load-choices`、
  `--saturation-queue-floor`、shutdown drain、h2c、dynamo-render chat formatter。
- `[AMD][CI] Consolidate AMD workflows and retire ROCm 7.0 CI` (#38632)：删除
  `nightly-test-amd-rocm720.yml` / `pr-test-amd-rocm720.yml`。
- `[Test] Drop dead and strictly-subsumed CI test registrations` (#40264)。

---

## 2. 模型优化点

### 2.1 DeepSeek-V4.1 全量落地（本轮最大的一块，16 笔连号 PR）

| PR | 内容 |
|---|---|
| #39646 | standalone kernels 与 Python wrappers |
| #39648 | Top-k kernels 与 candidate selection |
| #39652 | compression / KV I/O / metadata kernels |
| #39653 | communication kernels 与 wrappers |
| #39656 | RoPE 与 FP4 packing kernels |
| #39657 | Hopper FP8 matmul kernels 与 tuning |
| #39664 | mHC 计算与 compensated projections |
| #39665 | chat encoding 与 tool parsing |
| #39666 | Engram 模块与 request history |
| #39668 | vision tower 与图像预处理 |
| #39671 | candidate indexer library |
| #39677 | Rust 扩展：图像预处理 / KV pool 名字 / PD bootstrap |
| #38798 | 余下的模型与 runtime 集成 |
| #39704 | 降低 mHC、metadata 与小 batch router 开销 |
| #39957 | big fused wo_a quant |

配套：`KVLayout` 三态、`_maybe_precompute_flashmla_sched_meta()`、SM100 `swapab_attention` decode 路径。

### 2.2 DeepSeek-V4

- `[DSV4] Generalize attention metadata, sparse prefill, and KV pool over compress ratios` (#39921)：
  `compress_ratio == 4 / == 128` 的硬分支换成 `!= 0` 的通用路径。
- `[DSV4] Chunk the indexer MQA logits by query rows under a free-memory budget` (#39095)。
- `[PP][DeepSeek V4] Overlap communication and optimize SM120 prefill` (#38792)。
- `[DSV4][BCG] Optimize the heavy memory use of C4 Indexer when BCG is enabled` (#36534)。
- `[Intra-node PD][DSV4] Pack all layers into one batch for INTRA_NODE_NVLINK path` (#38984)。
- `[Fix] Wait for PDL before reading DeepSeek V4 K cache locations` (#38409)。
- `forward()` 拆成薄 `forward()` + `_forward_attention()`，新增 `late_layer_tail` /
  `token_to_kv_pool.request_window` 两个 sparse-prefill 门。

### 2.3 其它模型

- **Kimi K3**：复用 `run_mega_routed_experts()`；`[NPU] support kimi k3 on A5 and improve performance` (#39589)。
- **Qwen MoE**：#38080 接 DeepGEMM MegaMoE；`[AMD] Load fused shared experts for Qwen4-Exp and Qwen3.5 MTP` (#38878)。
- **GLM-5.x**：`[perf] Optimize w4a8 MoE for glm5.2 on H200` (#38220)、`[Fix] Fix GLM5 mHC PP forward` (#39720)、
  `Fix disagg PP MTP for GLM-5.2` (#39378)、`[Kernel] Add H20 block-FP8 MoE configs for GLM-5.3-Flash EP4/EP8` (#38913)、
  `[AMD] Prefer HIP Top-K for GLM-5.x on ROCm` (#39631)。
- **Bailing / Ling**：新增 multi_gate + `MultiRouter`（`_forward_gate()` 抽取，`forward_batch` 贯穿 MoE forward 链）。
- **MiniMax-M3**：`allow shared-experts fusion on ROCm gfx942 and newer` (#36576)。
- **新模型**：`Add Ling-3.0-flash-VL model support` (#38526)。
- **修复**：Mistral3 只读一层却保留整棵 vision tower (#39185)、Gemma4 `lm_head_is_tied` (#35809)、
  GLM-OCR MTP 多模态 embedding/position (#39088)。

---

## 3. 冲突处理要点

50 处冲突，除下列几处外均为 import/常量并集。

### 3.1 `mega_moe.py`（7 处）— 双 runtime × SM 预算

`_get_mega_moe_symm_buffer()` 整函数重建：签名同时带官方的 `mma_type` 和我方的
`runtime` / `cuda_graph_max_tokens_per_rank`（后两者改成带默认值的 keyword-only，
因为官方新入口 `kimi_k3.py` 不传）。HCU standalone megamoe 不走 deep_gemm，
因此该分支用 `nullcontext()` 且 key 里的 `get_num_sms()` 记 `None`。

`_run_mega_routed` 按官方拆分重建；HCU 分支下沉到 `run_mega_routed_experts()`，
两个 HCU helper 的入参由 `moe: DeepseekV2MoE` 改成 `experts` + `activation_clamp`。
`activation_clamp` 取官方的 `moe.experts.moe_runner_config.swiglu_limit`，
**HCU 下为 None 时回落到我方原来的 `getattr(moe.config, "swiglu_limit", None)`**，避免静默丢 clamp。

`is_mega_moe_experts_ready()` 是官方新函数，按 `_device_sm` 判 90 / 10x —— HCU 两者都不是，
直接返回 False 会让 HCU 永远用不上 MegaMoE，故加 `if _IS_HCU: return True` 开口。

### 3.2 `deepseek_v4_backend.py`（4 处）— `_forward_attention` 整函数重建

官方把 `forward()` 拆成 `forward()` + `_forward_attention()`，并**删掉了
`SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA` 那条尾巴**（连同 `_forward_flash_mla_decode/prefill`）。
我方 HCU 正是靠 `if not _is_hcu:` 跳过整个 CUDA 块后落到这条尾巴。
处理：以官方函数为底，把「sparse-prefill 门 → SM100 swapab → SM120 → flash_mla → `return o`」
整块缩进进 `if not _is_hcu:`，其后接我方尾巴；ndim reshape 那段保持我方的 `_is_hcu` 包裹。

### 3.3 `deepep.py`（3 处）— 保住 `SGLANG_GROUPGEMM`

官方把按变体展开的 dispatch 合并成一份 + NPU 量化 kwargs helper。
我方 `use_groupgemm` 路径（lightop per-token 量化、marlin/fp8 需要的 256 expert alignment、
`quant_type` 0/1/2）上游没有对应物，因此保留为 `if use_groupgemm:` 分支，`else:` 换成官方那份。
`_deepep_precompile_tp_barrier()` 已改读 `get_parallel().tp_group`，`get_tp_group` import 随之删除。

### 3.4 `minimax_m2.py` / `hunyuan_v3.py`（各 4 处）— 取 theirs 要回补 import

两个文件的冲突都是 §1.1 的 runtime context 迁移引起的，取 theirs 后 ruff 报 12 个 F821：
我方 fork 在这两个文件里还有自己的代码在用被官方删掉的 import。回补了
`get_tp_group`（minimax int8 a2a）、`get_pp_group` / `get_attn_tensor_model_parallel_*` /
`ExpertLocationDispatchInfo` / `get_global_expert_distribution_recorder` / `get_moe_impl_class`（hunyuan）。
`minimax_m2` 里我方的 `self.pp_group = get_pp_group()` 与官方新加的
`self.pp_group = get_parallel().pp_group` 重复，删我方那句。

### 3.5 `topk.py`（1 处）— lightop gate 与新 kwargs 的互斥

我方 sqrtsoftplus 走 `biased_topk_lightop_impl`，它不认识 `packed_out` / `sqrtsoftplus_log1p`。
解法：先算 `_packed_kwargs`，**`_packed_kwargs` 非空时不选 lightop、强制 jit kernel**。
DSV4-Flash（`model_type == deepseek_v4`）两个开关都是 False，lightop 路径不受影响；
DSV4.1 会自动落到 jit gate，拿到正确数值形式。

### 3.6 `clamp_position` —— 名字遮蔽陷阱

官方把 dispatch 搬进 `kernels/ops/attention/clamp_position.py` 的 fused-op registry，
`forward_batch_info.py` 顶部已经 `from ... import clamp_position`。
我方原先在该文件**模块级重新绑定** `clamp_position`，会在 import 期把官方的 op 覆盖掉。
改为采纳官方 import，HCU 的绕开改在 op 本身：`ClampPositionOp.capabilities`
在 `_is_hcu` 时只声明 `CUDA`，于是 HCU 落到 `forward_native`（等价于我方原来的 `_clamp_position_native`）。

### 3.7 `warp.cuh` —— 官方删掉的 helper 我方还在用

官方 #36176 删了 `sync()` / `shfl_down()` / `shfl_xor()`；我方 `csrc/moe/moe_fused_gate.cuh`
是它们唯一的消费者（6 处 `shfl_down` + 2 处 `sync()`），且 wave64 DCU 需要 width-aware 版本
（官方的 `__shfl_down_sync(SGL_WARP_SYNC_MASK, ...)` 会把 32 位 mask 交给 wave64 设备通道）。
按并集保留：官方新 helper + 我方三个 ROCm 保护的 wrapper，并注明消费者。

**并集第一版是坏的，只有真跑才暴露**：冲突两侧都停在函数中间——官方侧停在
`elect_one_lane()` 的 `return pred != 0;`，我方侧停在第三个 wrapper 的末尾——
标记之后那一个 `}` 是**共享**的，只能闭合其中一边。并集之后 `elect_one_lane()` 少一个右括号，
整个文件的命名空间嵌套塌掉，表现为 `compress_v2.cuh` 一堆
`unknown type name 'SymbolicSize'; did you mean '::sglang::host::SymbolicSize'?`
（与 20260810 的 `::host::panic` 同一族症状）。
补回 `}` 后还有第二层问题：`elect_one_lane()` 的函数体是裸 NVIDIA PTX（`elect.sync`），
而这个头文件会被**每一个 ROCm JIT 模块**包含，hipcc 编不过。
已补 `#ifdef USE_ROCM` 分支，用"选最低 active lane"（`__ballot` + `__ffsll`）作等价实现。

### 3.8 AOT 构建的两处断裂（只有真编才暴露）

1. `eagle.cuh` 被 #40033 挪到 `kernels/jit/include/`，而 `setup_hip.py` 的 `include_dirs`
   只有 aot 自己的三个目录 → `fatal error: 'sgl_kernel/speculative/eagle.cuh' file not found`。
   补 `root.parent / "jit" / "include"`（`setup_musa.py` 早就有这一条）。
2. `transfer.cu`：官方把我方的 ROCm host-pointer 修复上游化成
   `resolve_device_accessible_ptr()`（用 `hipHostGetDevicePointer`），但我方在
   `transfer_kernel_impl_hcu` 那段还有**第二处**调用点仍叫 `get_rocm_kernel_accessible_ptr`，
   auto-merge 保留了调用、删掉了定义。改名的同时补上官方 helper 缺的 `.defined()` 判空。

### 3.9 `utils/common.py` —— 定义被删、调用点还在

`get_physical_device_id()` 是我方 helper，唯一调用点在我方的 NUMA 查询里。
auto-merge 把定义删了、调用留了。**这一处没有冲突标记，是 ruff 3-way 门抓到的**（§4 门 3）。

---

## 4. 静态门（7 道，全部与两个 parent 做 3-way）

| # | 门 | 结果 |
|---|---|---|
| 1 | `compileall python/sglang test/` | 通过 |
| 2 | `environ.py` 全文件符号 diff + `envs.*` 引用对拍 | 通过（ours-only 声明丢失 0，新增未声明引用 0） |
| 3 | ruff `F821,F811,F401,F722` 3-way | **抓到 1 个**：`get_physical_device_id`（见 §3.9），修复后通过 |
| 4 | 跨模块 import（AST）3-way | 通过（merged 304 / ours 288 / theirs 307，新引入 0） |
| 5 | C 系预处理器结构门 | 通过（432 个 C 系文件，0 新引入） |
| 6 | 我方独有行的悬空属性门 | 通过（696 个属性访问，41 个"未绑定"全是 unittest 方法 / 模块路径 / 字符串注册的假阳性） |
| 7 | 上游 ratchet 单测 | 与 parent 一致：`test_global_config_read_ratchet` 仍报 31 处直读，**数量未增长**（既有债，见 §6） |
| 8 | **C 系大括号/圆括号平衡门**（新增） | 修复后通过（427 个文件，0 新引入） |

> **门 8 是本轮新加的**，原因见 §3.7：预处理器门只看 `#if/#else/#endif` 栈，
> 看不见少一个 `}`。新门剥掉注释与字符串字面量后统计 `{}` / `()` 差值，
> 与两个 parent 对比，只报合并新引入的失衡。脚本：`/tmp/gate_braces.py`。

> 本轮教训：解冲突途中误跑了一次 `git add -u`，把还带冲突标记的文件 stage 成已解决，
> `:1:/:2:/:3:` 三方 stage 全丢、`git checkout -m` 也失效。工作区内容完好，
> 改用 `git show HEAD:<path>` / `git show officials/main:<path>` 取两边继续。
> **合并期间不要用 `git add -u`。**

---

## 5. 我方为适配官方改动而新增的 `_is_hcu` 保护

| 位置 | 保护内容 |
|---|---|
| `mega_moe.is_mega_moe_experts_ready` | HCU 不按 CUDA compute capability 判定 |
| `mega_moe._get_mega_moe_symm_buffer` | standalone megamoe 不进 deep_gemm SM 预算与 key |
| `mega_moe.run_mega_routed_experts` | HCU 双 runtime 分发 + capture 期 buffer 定尺 |
| `deepseek_v4_backend._forward_attention` | 整个 CUDA sparse/SM100/SM120/flash_mla 块 |
| `clamp_position.ClampPositionOp` | HCU 只声明 CUDA capability，落 `forward_native` |
| `dsv4/metadata.PagedIndexerMetadata.copy_` | `(is_hip() and not _is_hcu) or chunked` |
| `fp8.process_weights_after_loading_block_quant` | HCU 不走 aiter MXFP4 专家路径 |
| `attention/dsv4/attn.py` | V4.1 layout 断言对所有 HIP 生效，triton fallback 仍排除 HCU |
| `pool_host/mha.get_mha_host_pool_class` | `layout_hcu` 分支排在 MXFP8 分支之前 |
| `hybrid_pool_assembler` | `layout_hcu → page_first`，与官方的 `page_first_kv_split → page_first_direct` 并列 |

---

## 6. 既有技术债（parent 上一模一样，本次不处理）

- `test_global_config_read_ratchet`：我方 31 处 `get_server_args().field` 直读
  （communicator 6、bailing_moe 7、minimax_m2 8、qwen2/3/3_moe 7、loader 1、triton fused_moe 1、
  compressed_tensors 1），与 20260914 记录一致，数量未增长。
- 跨模块 import 门的 288 条既有条目（多为 diffusion 的惰性再导出与 NPU/XPU 专属模块）。

---

## 7. 未对齐 / 主动不跟进

- `nightly-test-amd*.yml` / `pr-test-amd*.yml` / `pr-test-npu.yml` / `pr-states.yml`：
  我方在 `3ca61517d0` 里已刻意清空 AMD 状态行，继续取 ours。
- `setup_rocm.py` 与 `setup_hip.py` 一样缺 `jit/include`（官方自己的 ROCm AOT 构建也会挂），
  本轮只修我方实际使用的 `setup_hip.py`。

---

## 8. 验证（zz-nmz110 / rye_sglang_latest，2026-09-20）

装法：容器内 `bash /home/scripts/install_sglang.sh /home/proj_sglang_open/sglang-das`
（sgl-kernel 0.4.7 按该容器的 torch 2.11 重编）。
纯 TP8，`/models/DeepSeek-V4-Flash-0731-FP8-Channel`，`max_total_num_tokens=7917056`：

| 项 | 结果 |
|---|---|
| 贪心 sanity | `The capital of France is **Paris**.` |
| **GSM8K 100 题** | **1.000** |
| 峰值 decode 吞吐 | 326.13 tok/s |
| Scheduler 异常 / Traceback / VMFault | 0 / 0 / 0 |

比 20260914 的 0.99 回到满分（上轮唯一错题是柠檬树回本那道边界题）。

**环境变动**：zz-nmz26 的 `rye_sglang_latest` 现在把 editable sglang 指向
`/home/proj_sglang_fork/sglang-das`（另一条 feat 分支），因此本轮验证改在 zz-nmz110。
nmz110 两个容器都没有 evalscope，GSM8K 客户端从 zz-nmz26 的 `rye_sglang_open`
（evalscope 1.10.0）打 `http://12.12.12.110:10015`。

**共享启动脚本的两处过时**（都没有改共享文件，复制到
`/home/proj_sglang_open/scripts_local/` 后改）：

1. `--cuda-graph-max-bs` 已被上游拆成 `--cuda-graph-max-bs-decode` / `-prefill`，
   argparse 报 `ambiguous option`。注意这**不是本次合并引入**——两个 parent 上都已经拆好了。
2. 脚本按 hostname 选网卡，没有 nmz110 的条目，落到默认的 `ens19f0`，
   gloo 直接 `Unable to find address for: ens19f0`。nmz110 是 `ens65f0np0`。
