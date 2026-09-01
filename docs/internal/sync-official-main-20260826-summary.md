# 官方 main 同步总结 · 20260826

**范围** `c8e1ddc707` → `a9d5ca723`（`[CPU] Fix weight missing issue in fused_input_proj_cpu for GPTQ INT4 for Qwen 3.5 (#35805)`）
**规模** 372 commits · 1429 files · +123 090 / −29 422 · 41 个冲突文件 / 67 处

## 0. 基点的坑：必须先锚定

上次（20260824）同步因为 `HYGON-AI` 写权限问题，是**经个人 fork 压平 cherry-pick** 再通过 PR #222 进 main 的，原来的双 parent merge `6d77e62f8` 从未成为 main 的祖先。后果：

```
c8e1ddc707 是 main 的祖先吗？      NO
merge-base(main, officials/main) = 92b1d382c7   ← 上上次的基点
不锚定直接 merge 要重放 769 笔（其中 397 笔已经合过）
```

内容层面 main 确实已携带 `c8e1ddc707`（抽样 40 个 srt 文件，29 个与官方逐字节相同，10 个是我们做过 HCU 改造的）。用 `git merge -s ours c8e1ddc707` 锚定后 tree 完全不变（`5c0cf919fa…` 前后一致），merge-base 回到 `c8e1ddc707`，重放量降到 **372 笔**。

> **给下次的建议**：走 fork 压平这条路的话，合进 main 后立刻补一笔 `-s ours` 锚定，否则每次同步都要重踩。

## 1. 模型支持

范围内 94 笔与具体模型相关。

| 模型 | 主要改动 |
|---|---|
| **GLM-5.3 / 5.3-Flash** | 本轮最密集的一族。cookbook 成体系（#36440/#36519/#36544/#36660/#36740），Blackwell 默认 FP8 KV + TRT-LLM DSA，HiCache for LL、EAGLE、DCP4 overlay；AMD 三档 recipe（MI300X/325X/355X，#36608）；NPU GLM4.7-Flash FIA 非连续输入修复（#36170）；运行时权重更新后 MoE 路由不刷新的 bug（#35883） |
| **Qwen 3.5 / 3.6 / 3.8** | 3.8 rebase（#35758）与 Flash-Next cookbook（#36496）；H200 MoE 配置（#35374）；fused RoPE 丢 mrope 高宽的修复（#34446）；AMD 侧 ASM FMHA chunked-prefill（#36758）、gfx950 MTP unified attention 优化（#36330）、MTP 丢 fused shared-expert 权重（#35719）；CPU GPTQ INT4 权重缺失（#35805） |
| **DeepSeek-V4** | `topk_transform` v2 kernel 打开（#36684）；MI355X decode split-K 重新调参（#36094）；MXFP8 MoRI dispatch 对齐 w4a8 MoE 输入格式（#36119）、decode 期 MoRI 收包缓冲收敛（#36130）；prefill CP 支持 `moe_a2a_backend=mori`（#35611）；multistream QKV buffer 生命周期修复（#36547）；fp4 kv-cache aiter 修复（#37083）；MI30x 上新增 FP8 精度看护（#36396） |
| **Kimi K3** | Mooncake MoE A2A 走 EP-A2A / SP-MoE 快路径（#36862）；aiter MLA ASM 通过 padding attn heads 打通（#36356）；dense ModelSlim MLA 权重保留（#36603）；`packed_modules_mapping` 声明（#36211） |
| **MiniMax / 新模型** | Ling-3.0-flash（BailingMoeV3）支持（#33561）+ `enable_dp_lm_head` 读错来源的修复（#36584）；Nemotron 3.5 Lightning 投机解码（#36186）；ERNIE 上 CPU（#35222）；XPU minimax_m2 all_reduce 懒加载（#35290） |
| **Diffusion（185 个文件，仍是最大单块）** | Qwen-Image 输出投影 bias 吸收（#37116）、CFG 串行分支间 modulation 缓存（#37090）；Wan2.2 Blackwell NVFP4 bias+GELU 融合（#37075）、Wan FFN GELU epilogue 融合（#36592）；Cosmos3 Nano Hopper T2I attention 融合（#36571）、96GB 常驻（#36641）、transfer capability（#34747）、action generation 批处理（#36301）；GLM-Image NPU 分布式推理（#31320） |

## 2. 关键技术与组件

### 内存 / KV cache（54 个文件）

本轮最有结构性意义的一块：**`ReqKvInfo` 聚合重构**——把散落在 Req 上的 KV 记账字段逐个收进一个对象：`cache_protected_len`/`swa_evict_floor`（#36982）、`kv_committed_len`（#37078）、`req_pool_idx`（#37094），并让 streaming session slot 与其 request 共享同一个 `ReqKvInfo`（#37108）。`alloc_for_extend` 内部结算 extend 的 `kv_committed_len`（#37085）。

其余：unified-memory 为 uniform-row MHA/SWA 模型提供 dense KV views（#34602）、lazy-compaction 映射批量查询（#34066）；MLA KV cache 跳过保留区写入（#36003）；**权重加载内存尚被引用时 KV pool 被算得过小**的修复（#36583，值得关注，直接影响可用 KV 容量）；`SWARadixCache.cache_unfinished_req` 带上 `swa_evicted_seqlen`（#36909）。

### HiCache

L2 明确为实例私有、只有 L3 共享（#37050）；decode offload 状态按请求隔离（#37026）；load-back 用 forward stream 围栏（#36738）；按请求 namespace 做 storage prefetch（#36382）；拒绝声明了正在 load-back 的节点的 load-back spec（#35931）；L3 prefetch 失败重试（#36227）；hybrid prefetch 丢弃时修复 existence cache（#36386）；`check_prefetch_progress` 去掉不必要的 all-reduce（#36425）；chunked CUDA host 注册对齐（#36798）。

### 投机解码

**DSpark / DFlash 跨 TP rank 状态发散修复（#33614）** 是本轮最重要的正确性修复之一；DSpark draft `sample_block` 重复调用（#36934）、DSV4 DSpark sample-from-anchor 初始化（#36419）；hybrid SWA MTP draft pool 路由泛化（#35379）；投机分配路径避免 tensor 标量读取（#35377）；新增 `--speculative-dsa-topk-backend`（#36313）；speculative adaptive 的启动崩溃与 CUDA graph 显存占用（#35275）；AMD 侧 PD DSA fused-TopK seed remap（#36714）、AITER EAGLE draft extend 的 eager metadata（#36915）。

### Kernel / 通信

custom all-reduce 拆成 push/pull 两个平面（#35735）；SM90 FP8 decode 回归修复，按实测 M/K/N 路由（#37018）；大尺寸 SM90 row/column-scaled FP8 GEMM 改走 Torch（#34318）；W4AFP8 DeepEP 低延迟 requant 的 launch geometry 调参（#35760）；sglang-kernel wheel 声明 PyTorch ABI 依赖（#36465）。

### 调度 / 并行

DP1 下跳过冗余的 scheduler metadata gather（#36568）；breakable prefill CUDA graph 减少 DP 空转（#33871）；FlashInfer EXTEND 针对 DP prefill 调参（#36219）。

## 3. 冲突处理立场（41 文件 / 67 处）

原则不变：官方重构方向优先，HCU 特有行为用 `_is_hcu` 保护，两边各加各的就取并集。

### 顺手修掉的既有缺陷

`kernels/aot/CMakeLists.txt` 我们这侧的 CUTLASS 宏被写成 `-HCUTLASS_ENABLE_TENSOR_CORE_MMA=1` / `-HCUTE_USE_PACKED_TUPLE=1` —— 是某次把 `DCU` 全局替换成 `HCU` 时误伤了 `-DCUTLASS_`。这些都在 `SGL_KERNEL_ENABLE_FA3` 的 `compute_90a` 分支里，HCU 上不编所以一直没暴露，但在 NVIDIA 上开 FA3 就会踩到。本次取官方写法，并把冲突区外同样被改坏的 6 处一并修复。

### 典型取舍

- `setup_rocm.py`：保留我们给 **gfx938** 的 48KB LDS 预算（BW1100 与 gfx942 同为 64KB/workgroup），采纳官方把 else 分支从 128KB 收到 40KB 的 occupancy 优化
- `arg_groups/speculative_hook.py`：采纳官方的 `resolving_view(server_args)` 重构，**保留 HCU 放行**（官方那版只认 CUDA/NPU）
- `batch_overlap/two_batch_overlap.py`：取官方。查过文件历史，我们那侧的 `pin_memory()` 本就是同步来的旧官方代码，不是 HCU 改动
- `dsa/dsa_topk_backend.py`、`add_constant.cuh`：import / 版权头并集，官方的新文件路径

## 4. 验证

纯 TP：`bash run_dpsk-v4.sh 10015 /module/DeepSeek-V4-Flash-0731-FP8-Channel`，环境 `zz-nmz26 | rye_sglang_latest`。
