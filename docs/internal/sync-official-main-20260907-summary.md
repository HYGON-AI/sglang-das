# 官方同步总结 · 20260907

| 项 | 值 |
|---|---|
| 上次同步点 | `a9d5ca723a30b80655954688eef6e0e66cacb025` |
| 本次官方 tip | `97dcbf9410c1df1b7a4d341c4ed021f06f8d5087` |
| 区间 commit 数 | 401 |
| 变更规模 | 3592 files, +210994 / −53009 |
| 分支 | `sync/official-main-daily-20260907` |
| 冲突 | 79 文件 / 113 处 |

## 0. 关于"上次同步点"的确认

`a9d5ca723` **内容正确，但不是 main 的 git 祖先**——20260831 那次同步是以
squash merge `50bacb0280 Sync/official main daily 20260831 (#273)` 落地的，squash 丢掉了
第二 parent，所以 `git merge-base` 会一路回退到很早的位置（1170 个 commit 要重放）。

抽样核对：随机 30 个 srt 文件里 29 个与官方 `a9d5ca723` 逐字节相同，唯一不同的
`arg_groups/model_overrides/deepseek_v2.py` 是我们自己的 HCU 门控。据此用
`git merge -s ours a9d5ca723` 做锚定（tree 不变，`3c38e25aaed…`），重放量降到
**401 commits**。这是本仓库第三次踩 squash-merge 陷阱，处理范式见 `fp-base-is-second-parent`。

---

## 1. 组件优化点

### 1.1 配置系统重构（影响最大，本次冲突主要来源）

官方把 `server_args.py` 拆成 `arg_groups/`：**−3585 行**。

- `[Config] Round 6.2` (#38047)：字段声明从 `ServerArgs` 迁到各自命名空间模块
  `arg_groups/fields/{model,parallel,memory,exec_,schedule,serving,spec,lora,mm,disagg,device,observability}.py`，
  每个子命名空间一个 dataclass（`ExecKernel._NS_PATH = "exec.kernel"`、`ExecMoe`、`ExecComm` …），
  `ServerArgs` 由它们组合而成。
- `[Config] Round 6.5` (#38113)：派生量（`attn_tp_size`/`moe_ep_size`/`dcp_enabled` 等）
  用 `Derived(...)` 声明在它们所依赖的叶子字段旁边。
- 选项清单集中到 `arg_groups/choices.py`；字段顺序冻结在 `field_order.py`
  （**新字段不在表里 → 自动排最后**，所以我们的新字段不需要动这张表）。

### 1.2 内存 / KV cache

- `[mem_cache]` #38072 / #38103：unified-memory allocator 迁到 `allocator/`，复合分配器拆分。
- `free_segment` 要求 page 对齐起点并去掉边界裁剪（#37729）；hybrid SWA full 侧
  kv-row 释放改走 `free_segment`（#37876）；`page_size == 1` 的 `free_swa` 做到 sync-free（#36723）。
- `[Perf]` #37324：radix tree 按 offset 遍历，不再重切 token 存储。
- `[Perf]` #37938：`alloc_extend_naive` 向量化，去掉 per-request Python 循环。
- HiCache：buffer mode 支持 sidecar pool（#37424）、anchor-lock 预取默认开启（#37464）、
  L3 storage 预取生命周期指标与跨层归因修复（#37503）。

### 1.3 PD 分离

- `SGLANG_DISAGGREGATION_ENGINE_INIT_TIMEOUT` 给 transfer engine 初始化加超时（#37874）。
- decode KV 延迟释放按 backend 能力门控（#37454）；paged allocator free-list 释放优化（#37146）。
- decode 预分配期间轮询 receiver，修死锁（#37483）。

### 1.4 投机解码

- 支持在 unified SWA 内存上做投机解码（#36403）。
- EAGLE：draft-extend logits 只保留被选中的行（#35546）；多层 EAGLE shared-read event 发布修复（#36752）。
- 自适应投机解码支持自定义策略（#37274）；draft 容量与 runtime state 解耦（#36897）。
- `[2/N][Mixed]` #36933：mixed chunk prefill 与 spec 共存。

### 1.5 量化 / MoE / kernel

- `[1/N] Quantization Refactor` #37552：删死代码，FP4 marlin helper 去重。
- `[MoE Refactor]` #32405：SM100 trtllm-gen mxfp4 MoE 迁到 MoeRunner。
- 新增 fused silu-mul-quant fp8 kernel（#37376）、KDA FP8 skinny GEMM for SM120（#38082）。
- `ue8m0` group requant 降低 GPU 预留显存（#31755）。

### 1.6 Context Parallel

- `[CP V1 Deprecation 3/5]` #36228：通用 prefill CP v1 runtime 移除，统一走 cp_v2
  （`is_cp_v2_active(forward_batch)` 取代 `is_prefill_context_parallel_enabled()`）。

### 1.7 Router

- bucket-aware policy domain + native cache indexing（#38108）、cache-aware 负载状态发布（#38139）、
  load-aware prefill 准入与有界策略提议（#37843）。

---

## 2. 模型优化点

### 2.1 DeepSeek-V4（与我们最相关）

- **DSA indexer**：`[DSA]` #36831 把 top-k transform 入口名里的 `512` 去掉
  （`topk_transform_512*` → `topk_transform_paged*`），并把整段改写成
  行分块的 `run_topk_transform(rows, logits)` 闭包 —— prefill 时按 logits 块大小切行，
  逐块打分再归约，显存峰值随上下文长度不再线性膨胀。
- `[ROCm]` #37591：DSA indexer top-k 用 cooperative selection 做到精确。
- `[ROCm]` #37124：融合 DSA metadata kernel，absorb 路径去掉冗余计算。
- `[AMD][DSv4]` #37423 + #37658：输出投影 `oproj_a` 切 fp8；inverse-RoPE 融进 fp8 wo_a quant。
- `[AMD][DSv4]` #37764：FP4 indexer prefill 调度前导融成单 kernel；#37353 打开 AMD FP4 indexer。
- `[SM120]` #29927：DeepGEMM paged-MQA indexer + FP4 MoE + page-split。
- `[DSV4]` #35118：hc-prenorm 的 combine 步骤融进 Triton kernel。
- `[Deepseek V4]` #33608：mxfp4 trtllm MoE 保持 fp32 routing weight。
- `--dsa-topk-backend flashinfer` 融合 top-k（#33237）。

### 2.2 其它模型

- **GLM**：GLM-5.3-Flash 支持（#36507）；GLM5.1 DSA on XPU（#24959）；GLM-5.2 NVFP4 B200/B300。
- **Kimi-K3**：ROCm 上 KDA 输入投影融成单 GEMM（#35176）、gfx950 Triton MLA prefill 优化（#35770）、
  MoE 优化（#33838）、aiter gluon 路径支持 qlen>1（#37601）。
- **Qwen3.5**：GDN prefill 投影布局优化（#36267）；MTP embedding 在 PP 下的加载修复（#37471）。
- **MiniMax**：H3 tiered AdaLN plan cache（#37266）、SM120 SubBlock sparse attention（#37332）。
- **LFM2**：SM90 上 gating 与 short convolution 融合（#37622）；LFM2/LFM2-MoE DSpark 支持（#31041）。
- **Mamba/GDN**：radix cache ssm state 索引修复（#37836）；`causal_conv1d` col* dtype 统一（#38039）。
- **K2 Horizon**：FP8 checkpoint 支持（#38033）、无 MoVA 的 MoE（#37825）。

---

## 3. 本次冲突处理要点

### 3.1 官方修得比我们好，采纳官方（3 处）

| 位置 | 说明 |
|---|---|
| `kernels/aot/csrc/allreduce/quick_all_reduce.cuh` | 上次为绕开 `v_cvt_pk_f16_f32`（DCU 无此指令）加了 arch 门控。官方改用**空 asm 屏障** `asm volatile("" : "+v"(scaled.x), "+v"(scaled.y))` 挡住 LLVM 重结合，再走可移植 intrinsic —— 保住了防溢出语义且不依赖指令集。我方门控整块删除。 |
| `layers/moe/ep_moe/layer.py` | 低时延 DeepEP 的 DeepGEMM 断言，我们当初整段注释掉；官方重新启用并加了 `and quant_config is not None`，正是我们要绕的条件。注释块删除，断言恢复。 |
| `layers/attention/dsv4/indexer.py` | 官方的行分块 `run_topk_transform` 结构整体采纳，只补回我方的 aiter arch 门控与 `SGLANG_TOPK_TRANSFORM_512_TORCH` 覆盖。我方原有的 `raw_indices` shape 断言**主动删除**：分块后 `logits` 是块内的，断言不再成立（官方用 `raw_indices[rows]` 从结构上保证了对齐）。 |

### 3.2 保留我方实现（官方版本在 DCU 上不正确）

| 位置 | 理由 |
|---|---|
| `kernels/jit/csrc/moe/moe_fused_gate.cuh` | 官方新的 `SGL_WARP_SYNC_MASK` 只对 gfx1250 和 AMD host pass 加宽到 64 位，DCU device pass 仍发 32 位 mask —— 正是我们当初修掉的 bug。继续走 `device::warp::shfl_down`（ROCm 下丢掉 mask，wave64 正确）。官方宏保留并加注释说明本 fork 不使用。 |
| `mem_cache/allocator/swa.py` | 官方简化成一句 `clear_full_to_swa_mapping(mapping_indices)`；我方保留去重 + 跳过已在 free list 的页 + 清理所有指向已释放 SWA 页的 full 索引。`page_size > 1` 时官方版本会重复释放共享页并留下陈旧映射。 |
| `models/glm4_moe.py` | 官方一行 `self.shared_experts.gate_up_proj.quant_method.quant_config.weight_block_size`，在 `quant_method` 没有 `quant_config` 时会 AttributeError；保留我方的 `getattr` 防御版本。 |
| `mem_cache/kv_cache_dtype.py` | 官方把 `fp8_e5m2` 在 `_is_hip` 下改映射到 `fp8_dtype`（e4m3）。HCU 原生支持 e5m2（见 `HCU_GENERIC_KV_CACHE_DTYPE_CHOICES`），故门控为 `_is_hip and not _is_hcu`；官方新增的 CPU 拒绝逻辑照收。 |

### 3.3 我方改动需要"搬家"（官方挪走了它的宿主）

| 我方内容 | 新位置 |
|---|---|
| `_pre_warm_nccl_help()` + `pre_warm_nccl` 的 help | `arg_groups/fields/exec_.py`（`ExecComm` 之前） |
| `minimax_opt` | `arg_groups/fields/parallel.py` |
| `record_nolora_graph` | `arg_groups/fields/exec_.py` → `ExecMoe` |
| `pack_paged_kv_to_varlen{,_min_kv_tokens,_min_q_tokens}` | `arg_groups/fields/exec_.py` → `ExecKernel` |
| `"lightop"` MoE runner + 8 个 `HCU_*_CHOICES` | `arg_groups/choices.py` |
| `"layout_hcu"` hicache 布局 | `arg_groups/fields/memory.py` |
| `SGLANG_OPT_USE_TOPK_V2 and not _is_hcu` 门控 | `layers/attention/dsa/dsa_topk_backend.py::should_use_topk_v2()`（单点，覆盖 metadata / backend / indexer 四处调用方） |
| mHC 三维 PP proxy (`get_pp_proxy_hidden_states_shape`) | `managers/scheduler_components/dynamic_chunk_sizer.py`（官方把 `profile_and_init_predictor` 抽了出去，但抽出去的版本用的是二维 `(tokens, hidden)`；DSV4 的 hidden_states 打包了 mHC，必须是 `(tokens, hc_mult, hidden)`） |

### 3.4 我方遗留的重复代码（本次清理）

自动合并把上游后来的重构与我们几次同步的旧拷贝叠在一起，产生了三处
"嵌套重复"，本次一并拍平：

- `layers/attention/dsv4/compressor_v2.py`：整段 if/elif/else 级联被嵌套复制了一份，
  内层的 `is_in_indexer` / `is_unified_kv_triton()` 分支永远不可达。取官方扁平结构，
  只把我方 `bf16_store = token_to_kv_pool.is_bf16_attention_kv_cache` 保留在 compress_kv_pool 分支。
- `models/deepseek_v2.py`：router GEMM 的分支链同样嵌套重复。取官方链，
  把我方 `elif _is_hcu and self.is_deepseek_v4:`（Hash-MoE 需要 fp32 router logits）接回去。
- `managers/scheduler_components/dp_attn.py`：我方带着重复计算的 `breakable_prefill`，
  且冲突下方的共享代码需要官方的 `full_prefill`，取官方。
- `layers/moe/ep_moe/layer.py`：aiter `expert_mask` 已被上游搬进
  `DeepEPDispatcher.expert_mask_gpu`，我方层内的那份无人读取，删除。

---

## 4. 静态门结果

| 门 | 结果 |
|---|---|
| 全量 `compileall`（python/sglang + test） | **通过**，0 报错 |
| `environ.py` 全文件符号 diff | **通过**。base 588 / ours 647 / theirs 600 / merged 659。ours-only 少了 3 个（`SGLANG_ENABLE_CP_V2`、`SGLANG_ENABLE_HICACHE_BUFFER_ANCHOR_LOCK`、`SGLANG_SORT_FREE_LIST_AFTER_MERGE`）——查证三者在 base 与 ours 中完全一致、由**官方主动删除**，且合并后全树零引用，属正确继承而非丢失 |
| `envs.SGLANG_*` 引用对拍 | **通过**。merged 82 / ours 97 / theirs 94 未声明引用，合并**未引入任何新的**未声明引用 |
| 跨模块 import（AST，三方对比） | **通过**。唯一疑似项 `FLYDSL_NORM_MIN_ALIGNED_DIM` 经运行时验证为误报：`kernels/ops/diffusion/__init__.py` 用 `__getattr__` 惰性再导出，路径正确，只是容器未装 `flydsl`（gfx1250 依赖） |
| ruff F821/F811/F401（三方对比） | **通过**。merged 255 / ours 248 / theirs 237，归一化行号后**合并新引入 0 条** |

### 静态门抓到的真实缺陷（2 处，均在从未冲突的文件里）

1. **`model_loader/checkpoint_quantization.py` — `NameError`（真 bug）**
   官方把 `_get_field()` helper 换成了 `_as_config_mapping(...).get(...)`，并顺手删掉了
   `text_config.compression_config` 这条查找。自动合并采纳了官方的改写，却保留了我方那一行
   `_get_field(text_config, "compression_config")` 调用 —— helper 已不存在，该路径运行必炸
   （kimi_k26 这类 multimodal compressed-tensors checkpoint 会走到）。
   已改写为官方惯用法并保留我方这条查找。
2. **`multimodal_gen/runtime/loader/fsdp_load.py` — 死 import**
   官方用 `initialize_model()` 取代了 `set_default_torch_dtype(...)` 上下文，
   合并后只剩我方的 import。已删除。

> 两处都不在 79 个冲突文件里 —— 这正是"只看冲突区不够"的又一次印证，
> 参见 `environ-symbol-diff-whole-file`。

---

## 5. 未对齐 / 主动不跟进的地方

| 项 | 状态 |
|---|---|
| `.github/workflows/{nightly-test-amd,pr-test-amd,pr-test-amd-rocm720,nightly-test-npu}.yml` | 保留我方空触发器。官方要加 `schedule`/`push`，但这些 workflow 在本 fork 跑不起来，加了只会制造红叉 |
| `moe_fused_gate.cuh` 的 `SGL_WARP_SYNC_MASK` | 宏保留但本 fork 不使用（见 3.2），已加注释说明，避免下次同步再冲突 |
| `dsa_backend.py` 的 6 条 F811 | 官方自身的重复 import（第 17/51 行、第 18/97 行），合并结果与官方逐字节一致，作为上游债务保留 |
| `quant_dequant_mxfp4` | 全树零调用方，跟随官方删除 |
| XPU / NPU / gfx1250 / SM120 相关路径 | 全部照收，未在 DCU 上验证 |
| `multimodal_gen`（diffusion）220 个文件 | 照收，本 fork 不使用，未验证 |

---

## 6. 静态门看不见、只有跑起来才暴露的缺陷

首个 prefill 崩溃：

```
allocator/swa.py:379  if len(self.swa_attn_allocator.release_pages) > 0:
TypeError: object of type 'NoneType' has no len()
```

**根因**：官方 #37146（`[PD] Optimize paged allocator free-list release`）把
`PagedTokenToKVPoolAllocator.release_pages` 缓冲整个换成了 `staged_pages` 暂存表 ——
`clear()` 不再初始化 `release_pages`，它就一直停在 `base.__init__` 的 `None`；
`get_all_free_pages()` 被 paged 重写为 `cat(free_pages, *staged_pages)`，成为横跨两个
容器的唯一正确访问器。

我方有两处直接摸这个私有字段（官方该文件里 `release_pages` 出现 **0** 次，确认两处都是我方独有）：

| 位置 | 我方逻辑 |
|---|---|
| `allocator/paged.py` `free()` | 重复释放保护 |
| `allocator/swa.py` `free_swa()` | 跳过已在空闲表上的页 |

两处均改走 `get_all_free_pages()`；`swa.py` 那处顺带去掉了自己的 `merge_and_sort_free()`，
因为访问器已经把 staged 的算进去了。

**这不是合并引入的回归** —— 我方 parent `d93a583ba2` 里是同样两行裸 `len()`。
之前不炸，是因为那时 paged allocator 的 `clear()` 还会把 `release_pages` 初始化成空张量；
官方把实现换掉之后我们的代码就悬空了。冲突标记和所有静态门都看不见它：
`release_pages` 是属性访问不是未定义名字（ruff F821 无效），两个文件也都不在 79 个冲突文件里。
**这类跨文件隐式耦合只有真跑才能抓到。**

附带收益：改用 `get_all_free_pages()` 后过滤范围反而更全（原来只比 `free_pages`，
漏掉 staged 的），顺手消掉了我方 pre-pass free 与 `_release_swa` 尾部之间一个潜在的
重复释放 —— 官方新加的 `assert available_size() <= size` 正好会抓它。

---

## 7. 验证结果（zz-nmz26 / rye_sglang_latest，2026-09-08）

sgl-kernel 已按本次合并重编（19 个 hipcc 单元，gfx906/926/928/936/938）——
`quick_all_reduce.cuh`、`quick_all_reduce_base.h`、`topk.hip`、
`include/hip/dsa_topk_coop.cuh` 与 wheel 的 Python 侧都动了，必须重编。

`bash run_dpsk-v4.sh 10015 /module/DeepSeek-V4-Flash-0731-FP8-Channel`
（纯 TP8，`mem_fraction_static=0.8`，`max_total_num_tokens=2870016`）：

| 检查项 | 结果 |
|---|---|
| 服务启动 | 正常 |
| Greedy sanity | `"The capital of France is **Paris**."` |
| **GSM8K 100 题** | **1.000**（平均 7.678 s / 15.62 tok/s 每请求） |
| 聚合 decode 吞吐峰值 | 803.26 tok/s |
| CUDA graph | decode 已启用 |
| Traceback / VMFault / 非法访存 / scheduler 异常 | 0 / 0 / 0 / 0 |
| `multimem all-gather disabled` | 8 条，与 20260817 那次一致，RCCL 良性回退 |

关服后八张卡均回到 0%。

**踩坑记录**：本次之前失败三轮，均非代码问题 ——
(1) 第一轮即上面的 `release_pages` 缺陷；
(2) 第二轮 `[Errno 98] Address already in use`，因为 `pkill -f "sglang.launch_server"`
匹配不到任何进程（这套脚本起的是 `sglang serve`），第一轮的 15 个进程一直活着占着端口；
(3) 第三轮被宿主机重启打断在 CUDA graph 捕获阶段。
注意第二轮时 `hy-smi` 的 `HCU%` 显示 0%，但陈旧进程仍持有显存和端口 ——
**判断机器是否真空闲要看 `VRAM%` 那列，不是 `HCU%`**。
