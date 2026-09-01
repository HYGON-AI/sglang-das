# 20260826 同步 · 未对齐清单

同步范围 `c8e1ddc707` → `a9d5ca723`（372 commits）。下面是**刻意没有跟随官方**、或**官方改了而我们没跟**的地方，供后续单独处理。

## 1. 整文件推迟跟随（需要单独一轮）

| 文件 | 推迟的上游改动 | 原因 |
|---|---|---|
| `multimodal_gen/runtime/loader/fsdp_load.py` | −450 / +101 | 按 hunk 合并会产出语法错误：官方大幅简化了 loader，我们的 `_get_param_for_weight_loading`（本地独有，3 处引用）与官方重写的区域结构交织，任何单侧都不自洽。不在 DSV4 路径上，保留我方版本 |
| `multimodal_gen/runtime/models/dits/wanvideo.py` | −156 / +48 | 同上。我方 `resolve_wan_*` 系列在本文件内定义并被调用，官方版本没有这些调用点 |

## 2. 刻意分歧（保留我方实现）

- **`mem_cache/allocator/swa.py` 的 `free_swa`**：官方 #35773 只清 `mapping_indices`；我方额外做去重、跳过已在 free list 的页、清掉所有指向被释放 SWA 页的 full index。`page_size > 1` 时官方写法可能重复释放共享页。沿用 20260824 的判断，收敛回官方前需先给 C10 `bs>1` 乱码问题定性
- **`mem_cache/allocator/paged.py` 的 release 路径**：同上，保留更严格的版本，但守卫已跟随官方改名（`is_not_in_free_group` → `free_group is None`）
- **`layers/quantization/compressed_tensors/compressed_tensors.py`**：官方把 W8A8-INT8 Fused MoE 限制为「仅 NPU」并抛 `NotImplementedError`；我方保留该 scheme，HCU 的 SlimQuant INT8 需要它

## 3. 官方删掉、我们没有替代的

- **`utils/common.py` 的 `minimax_opt` 对齐规则**：官方把 `require_gathered_buffer()` 改成无参，`get_cuda_graph_batch_size_alignment()` 里也不再有 `server_args`，`runtime_context` 没有对应 accessor。原来的 `require_gathered_buffer(server_args) or server_args.minimax_opt` 无法保留。若 MiniMax 需要这条对齐，要另提 accessor

## 4. HCU 侧新增的门控（跟随官方但加了保护）

- **`quick_all_reduce.cuh`**：官方用裸 ISA `v_cvt_pk_f16_f32` 防 LLVM 重结合，gfx906/926/928/936/938 都没有该指令。按 arch 门控，HCU 走等价的分量式转换，其他平台不变
- **`kernels/aot/setup_rocm.py`**：保留 gfx938 的 48KB LDS 预算，采纳官方 gfx95x 的 40KB 默认
- **`arg_groups/overrides.py` / `model_overrides/deepseek_v2.py`**：官方把声明拆进 `model_overrides/`，我方 6 处 HCU 补丁随之迁移，其中 aiter preshuffle 守卫落到 `deepseek_v2.py`

## 5. 流程债

上一次（20260824）经个人 fork 压平 cherry-pick 进 main，原 merge 未成为 main 祖先，本次必须先 `git merge -s ours c8e1ddc707` 锚定，否则重放 769 笔而非 372 笔。**只要还走 fork 这条路，每次合进 main 后都应立刻补一笔锚定。**

## 6. 顺带修掉的既有缺陷

`kernels/aot/CMakeLists.txt` 中 14 处 CUTLASS 宏被历史上的 `DCU→HCU` 全局替换改坏（`-DCUTLASS_` → `-HCUTLASS_`）。位于 FA3 sm90 分支，HCU 不编译所以从未暴露，但会破坏 NVIDIA 上的 FA3 构建。
