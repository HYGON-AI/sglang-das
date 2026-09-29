# 官方同步总结 · 20260928

| 项 | 值 |
|---|---|
| 上次同步点 | `df0dc44931433b8a3488fbd4c6489e08cdee8703`（2026-09-20，已确认） |
| 本次官方 tip | `8b2ca8ecc2410f3ab170adff23fcd2db4cd47d2b`（2026-09-28 10:00） |
| 区间 commit 数 | 477 |
| 变更规模 | 2602 files, +168992 / −59721（287 个重命名、96 个删除） |
| 分支 | `sync/official-main-daily-20260928`（基于 main `665b27d1fa`） |
| 冲突 | 73 文件：57 content、12 modify/delete、4 file location |

## 0. 同步点确认

上次同步 PR #410 落 main 仍是 **squash merge** `16229e94d0`（单 parent），`df0dc44931` 不是 main 的祖先，
直接合并要重放 **2291** 个 commit。

`16229e94d0` 的 tree（`332531a44e`）与同步分支最终 tip `aa33884a91`（合并 commit `fdf02bbe46` + 门禁修复）
**逐字节相同**，确认 `df0dc44931` 的内容已在 main。于是 `git merge -s ours df0dc44931` 锚定
（commit `c91fc476a9`，tree 不变），重放量降到 **477**。**连续第六次 squash 陷阱**。

自 #410 以后 main 上有 11 笔我方提交（主要是 CI；另有 #435 DSV4 chat 的 assistant 前缀与 markdown 伪工具调用解析）。

---

## 1. 组件优化点

### 1.1 Layer communicator 重建（本轮对我方冲突面影响最大，约 40 笔）

一组连号重构把 `layers/communicator.py` 从"按 ScatterMode 选路径"改成"按声明构造边界"：

- `Split the layer communicator into a package (move only)` (#41439)：拆成 `layers/communicator/` 包
  （`layer.py` / `ops.py` / `boundary.py` / `layout.py` / `output.py` / `residual/` / `adapters/`）。
- `Replace LayerScatterModes with LayerFacts and remove ScatterMode` (#41443)。
- `Split prepare_attn into a reduction step and per-quant-format residual steps` (#41084)，
  `Choose prepare_attn / prepare_mlp steps and fused kernels at construction` (#41252)，
  `Give the fused prepare_mlp kernels an explicit contract` (#41418)。
- FFN 出口：`Carry a deferred FFN all-reduce as UnreducedOutput …` (#41196)、
  `Leave the FFN reduction to the next layer under attention DP` (#41197)、
  `Drive an FFN exit's flags and its completion from one selection` (#41253)；
  模型侧统一改用 `with layer_communicator.ffn_exit(fb) as ffn_exit: … ffn_exit.finish(…)`
  （#40870、#40871、#41198）。`should_use_reduce_scatter()` 等旧接口已删除。
- `CommunicateWithAllReduceAndLayerNormFn` 整个消失，所有边界步骤统一通过参数拿 stage 的 norm 并调用。

### 1.2 并行运行时收口（#40339–#40345、#40638、#40707）

- `Take the parallel getters off the package's public surface` (#40344)：
  `get_tensor_model_parallel_world_size` / `get_attn_tensor_model_parallel_rank` /
  `get_moe_expert_parallel_*` / `get_pp_group` 等**不再从 `sglang.srt.distributed` 导出**
  （函数本身仍在 `parallel_state.py`），推荐改读 `get_parallel()`。
- `Bringing the parallel runtime up becomes a phase, not a side effect` (#40345)、
  `A runner and the objects it builds freeze the placement they describe` (#40341)。

### 1.3 内存 / Cache

- 删除：`HiRadixCache`（#40787）、SWA 与 Mamba radix cache（#40313）、实验性 C++ radix tree（#40775）。
- `Replace cache_finished_req with insert_req; release_kv_cache frees and unpins` (#41281)、
  `Never free the protected prefix on request release` (#41312)。
- kv-shard 3/4（#38468、#39964）；unified memory：每种 pool 形状的分级缓存（#37507）、decode host pool（#39478）。
- HiCache：page-unified KV load-back JIT kernel（#39726）、sm90+ TMA 分段 host↔device 传输（#40278）、
  write_back 淘汰时把 SWA KV 降到 host 而非丢弃（#40712）、host 回收与传输顺序解耦（#40512）、
  flush 内批量备份（#40960）；新后端 TensorCast（#27265）、LMCache unified radix（#38652）。
- DSV4：`DSV4PoolConfigurator` 按 ratio 统一定尺（#41049、#41048）、FlashMLA 物理页填充计入预算（#41091）。

### 1.4 PD / DP

- `Enable deferred decode-side KV release by default` (#41023)、自定义传输后端的 decode host receive（#40238）、
  abort ACK 在 KV 传输排空前保留并广播到所有 decode peer（#40645、#41402）。
- `Publish DP buffer sizes from a ForwardBatch` (#40858)、DCP 用逻辑 token 容量做 PD 准入（#39731）。

### 1.5 投机解码

- LiLiCorr：DFlash 草稿的候选格重排（#37462）；XQA 后端做 verify（#32269）；
  内置 EAGLE/MTP 草稿的窗口化 draft-decode attention（#32673）；DP 间 prefill 期间保留投机（#40118）。
- `[DSpark] Fix draft CUDA graph stream explosion` (#40658)。

### 1.6 Kernel / 性能 / 其它

- JIT：保留 occupancy 的 L1 carveout 偏好（#40767）；Mamba2 8x1 launch（#41223，B200 最多 2x）；
  小 batch MXFP8 量化 MoE 排序路径（#36559）；MSCCL++ 8 节点 AllReduce/AllGather（#37442）。
- 启动：fork-safe import，import 期不建 CUDA context（#40201）；内置模型定义懒加载（#41061）。
- 采样 / API：采样 mask 按请求流式下发（#40986）、selected/support logprob 模式（#40932）、
  Score API setwise（#38965、#41188）、chat_parsing core（#40477）、默认拒绝请求自带 chat_template（#28135）。
- RL：权重更新 session（#40777）、TMS pause/resume 保持 DSA graph 状态（#40804）。
- `Remove deprecated endpoints, env vars and aliases past two releases` (#40795)。
- 测试：kernel 测试重组（#41243）、serving perf 按主题拆到 `basic_perf/`（#40505）、删除死 eval 模块（#41215）。
- AMD：53 笔，重点是 DSV4 的 MegaMoEv2（#35619）、fp8 decode `.co`（#41120）、gfx950 各类 kernel。

---

## 2. 模型优化点

### 2.1 DeepSeek-V4.1

- DeepSelect JIT kernel（#40556）；低 ratio index top-k 挪到 `dsv4/low_ratio_indexer`（#41125、#41291）；
  prefill consumer index 层在候选块上用 DeepGEMM 打分（#40352）；FP4 indexer 跳过不可见 tile（#40431）；
  dense prefill indexer 内存上界（#40217）；eager 前向处理分块的 paged MQA metadata（#40637）。
- reasoning effort 预算对齐（#39929）、XGrammar V4.1 DSML 参数约束（#39026）、
  HiCache 读低 ratio index-K 前等层传输完成（#41345）。
- AMD：KV 布局 / FP4 indexer / compressor / router kernel（#41019）、gfx950 MXFP8 matmul（#41018）。

### 2.2 DeepSeek-V4

- 池子定尺：`DSV4PoolConfigurator`（#41049、#41048）、TRTLLM uniform FP8 KV 预算（#41090）、
  C4 state ring 按页寻址定尺（#40337）。
- AMD：MegaMoEv2（#35619）、fp8 decode kernel 与分组 decode（#41120）、Triton KV store int32 偏移溢出（#41159）。

### 2.3 DSA

- kpool indexer 的 MQA logits 按空闲显存预算分块（#40854）；DP attention 下 pooled-indexer 可断 prefill 桥（#41311）。

### 2.4 其它模型

- **Kimi K3**：模型专属 kernel 命名空间退役（#40922）、DFlash（#40794）、load_weights O(1) 专家查找（#38805）、
  PP prefill + DCP decode + DSpark（#40045）。
- **GLM-5.3-Flash**：KDA ReplaySSM（#40517）、KPool 同步削减（#39695）、KDA 投影融合（#39688）、
  混合注意力 CPU offload + PD（#40310）、DeepEP v2（#40466）。
- **Qwen 3.8 Next / Qwen4-Exp**：小 graph 输入拷贝融合（#41166）、PLE gate 融合（#40041）、PP + PD-prefill MTP（#40501）。
- **MiniMax-M3**：gfx950 MXFP8（#36574）、fp8 lightning-indexer K cache（#36549）、wave64 top-k（#36560）、HiSparse MHA（#31446）。
- **新模型**：Gigachat 3.5（#29189）、DiffusionGemma serving（#34061）；DeepSeek-OCR-2 按官方 768px 局部裁剪（#38996）；
  LFM2-VL DSpark（#40651）；MiMo V2 bf16 router + mxfp4 MoE（#40448）。
- communicator 迁移顺带修了 Step-3.5 / Falcon-H1 / Nemotron-H / LongCat 在 DP attention 下重复求和等问题（#41082、#41433、#40800、#40799）。

---

## 3. 冲突处理要点

### 3.1 `layers/communicator.py` 被拆包 + 重建 —— 我方 HCU 钩子整体重接

旧文件 modify/delete（官方删除）。我方在旧文件上有约 180 行 HCU 改动，逐项处理：

| 我方改动 | 处理 |
|---|---|
| bailing 的 fused RMSNorm + per-token FP8 量化（`SGLANG_USE_FUSED_BAILING_RMS_QUANT`，prepare_attn 1 处 + 旧 `CommunicateWithAllReduceAndLayerNormFn` 里 6 处 layernorm 替换） | **移植**。新架构里所有边界步骤都从参数拿 stage 的 norm 调用，于是在 `StageCommunicator.prepare` 一处给出替身 norm（`_HcuBailingRmsQuantNorm`，`_is_hcu` 保护），调用约定与 RMSNorm 相同，返回量化对 `(fp8, scale)`。attention 输入沿用旧条件（EP=1、DP=1，首层走普通 norm——旧代码首层分支条件 `enable_dp_attention is None or moe_a2a_backend is None` 恒假）；FFN 输入由 `forward_batch.rms_quant_flag` 触发 |
| prepare_attn 开头 tuple 解包 | 保留，加 `_is_hcu` 保护 |
| int8 fused rms quant（`SGLANG_USE_FUSED_RMS_QUANT`：跳过 norm、把 residual 放进 `forward_batch.residual_rms_per_quant_int8`） | **不移植**：该字段在 communicator 之外**没有任何消费者**；唯一使用方 minimax_m2 调的是 `prepare_attn(..., skip_layernorm=True)`，而我方 main 上的 `prepare_attn` 根本没有这个参数（打开该开关本来就会 TypeError）。属既有死代码，见 §6 |
| aiter gfx95 分支 `quant_format == "fp8"` → `"fp8" in quant_format` | 不移植：官方已单列 `fp8_per_token` 分支，`in` 会让 per-token 格式误入 group-quant 路径；且与 HCU 无关 |

### 3.2 `ForwardBatch.rms_quant_flag` —— 自 20260728 起就丢了

bailing 每个 decoder 层都**无条件**读 `forward_batch.rms_quant_flag`，而该字段在 20260728 同步（#136）时从
`ForwardBatch` 里丢了、至今未声明（dataclass 无 `__getattr__` 兜底）——意味着 main 上**任何 bailing 模型第一层就 AttributeError**。
本轮恢复为 `rms_quant_flag: bool = False`。

### 3.3 `bailing_moe.py`

- 官方把 `self.post_attention_layernorm = RMSNorm(...)` 挪到了 `LayerCommunicator(...)` 构造之前，
  **3-way 合并把挪动后的那一行丢了**，而冲突区里 theirs 一侧为空——若取 theirs，构造 communicator 时直接 AttributeError。已保留。
- FFN 出口改用官方 `ffn_exit`，稀疏层继续接收 HCU fused 量化给出的 `moe_i_q` / `moe_i_s`。
- MoE 末尾的 TP all-reduce 已按官方新设计交给 `ffn_exit`（auto-merge 删除正确）。

### 3.4 `fp8.py` `Fp8MoEMethod.process_weights_after_loading_block_quant` —— 整函数重建

官方把 aiter FP4 的三个分支（MegaMoEv2 / native MXFP4 / DSV4 dequant）收进一个
`if _use_aiter and self.is_fp4_expert:`，冲突两侧都停在表达式中间、共享后面的 `)`，按 hunk 拼必坏。
以官方函数为底重放我方 4 处改动：

1. 外层条件加 `(not _is_hcu or self.dequant_fp4_to_fp8)`——HCU 没有 aiter MXFP4 MoE kernel，
   以前是"跳过 native、但 dequant 照走"，照此保持（**不能**把 `not _is_hcu` 直接加在外层，否则 dequant 也被跳过）。
2. fnuz 分支里 (16,16) shuffle 排除 HCU。
3. HCU aiter ASM MoE 的 b8 shuffle 链。
4. FP4 专家按 int8 字节 view：我方原来是**无条件**改，现收窄为 `_use_aiter and not _is_hcu`，非 HCU 保持官方行为。

### 3.5 `hiradix_cache.py` 被删除

我方在其中的唯一改动是给 `get_mha_host_pool_cls` 传 `hicache_mem_layout`（让 `layout_hcu` 选到 HCU host pool）。
主路径 `hybrid_pool_assembler` 早已传了；官方 8/6 的 MTP/EAGLE draft host pool 路径没传，本轮顺手补上
（选择器里 `layout_hcu` 分支本身有 `_is_hcu` 保护）。

### 3.6 `hunyuan_v3.py` —— 迁到 `LayerFacts` + `FfnExit`

官方的 hunyuan_v3 不走 communicator，我方版本走（DP attention）。`LayerScatterModes` → `LayerFacts`（参数一致），
decoder forward 的 `should_fuse_mlp_allreduce_with_next_layer` / `should_use_reduce_scatter` / `postprocess_layer`
换成 `ffn_exit`，我方块的显式参数改从 `ffn_exit.fuse_mlp_allreduce` / `ffn_exit.mlp_reduce_scatter` 取。

### 3.7 退役的并行 getter —— 4 个 fork 文件 import 断裂

`ep_moe/layer.py`、`slimquant_w4a8.py`、`slimquant_w4a8_marlin.py`、`hunyuan_v3.py` 从 `sglang.srt.distributed`
导入已撤出公开面的 getter，**import 期即 ImportError**（`ep_moe/layer.py` 在启动路径上）。
改为从 `sglang.srt.distributed.parallel_state` 导入（函数仍在、行为不变）。

### 3.8 其它

- `mega_moe.py`：官方新的 AMD FlyDSL MegaMoEv2 入口只看 `_is_hip`，排在所有 HCU 分支之前；加 `not _IS_HCU`。
- `dsv4/attn.py`：官方放开了 V4.1 布局在 HIP 上的限制，triton 回退改为只对 V4；我方继续把 HCU 排除在 triton 回退之外，断言照常生效。
- `deepseek_v4_backend.py`：CUDA 块在我方 `if not _is_hcu:` 下多一层缩进，官方只改了注释，按我方缩进取官方注释；`_forward_prefill_sparse` 保留 HCU 的 flash_mla 导入分支。
- `deepseek_v2.py`：HCU Hash-MoE 的 fp32 router GEMM 仍走我方分支，`linear_bf16_fp32` 改从新位置 `kernels/ops/gemm/bf16_fp32` 导入（启动脚本不设 `SGLANG_USE_AITER`，行为与之前一致）。
- `eagle_worker_v2.py`：官方把 target prefill 挪进 `_forward_prefill_batch`，我方 mHC 的 `return_hidden_states_before_norm` 跟过去。
- `disaggregation/decode.py`：官方的 host-receive-threshold 逻辑放进我方 PD 预算计时的 try/finally 里。
- `memory_pool.py`：HCU lightop fused 量化写 MLA KV 分支保留，通用分支用官方新注释。
- `fp8_kernel.py` 的 `is_weak_contiguous`、`test_deepseekv4_detector.py` 的 `CustomTestCase`：定义/导入被上游挪走，我方代码还在用，补回。
- 测试：6 个 serving perf 测试随官方拆到 `basic_perf/`，我方在其上的 HCU 注册**全是 `disabled=` 占位**，随文件删除；
  `quant/test_fp8_kernel.py` 被官方改成共享 helper `test/kernels/fp8.py`，取官方；3 个只因上游 CRLF→我方 LF 冲突的文件取官方内容转 LF；
  采样测试阈值 0.50（HCU CI）保留。
- workflow：上游 AMD/NPU workflow 保持只 `workflow_dispatch`（不加 schedule）。

---

## 4. 静态门

| # | 门 | 结果 |
|---|---|---|
| 1 | compileall | 通过 |
| 2 | environ 全文件符号 diff | 通过（门脚本改为 base-aware：官方自己删掉的 9 个上游开关不再误报为"我方丢失"） |
| 3 | ruff F821/F811/F401/F722 3-way | **抓到 2 个**：`is_weak_contiguous`、`CustomTestCase`（§3.8），修复后 0 |
| 4 | 跨模块 import 3-way | **抓到 2 个**：`linear_bf16_fp32`、`LayerScatterModes`（§3.6、§3.8），修复后 0 |
| 5 | C 预处理器结构 | 通过（449 个文件） |
| 6 | 悬空属性 | 通过（42 个"未绑定"均为 unittest 方法 / 字符串注册 / 类属性的假阳性） |
| 7 | 上游 ratchet 单测 | **收集阶段 ImportError**，由此发现 §3.7 的 4 个文件；修复后与 parent 一致（见 §6） |
| 8 | C 括号平衡 | 通过（444 个文件） |

**本轮新增两项检查**（都来自门的盲区）：

- **star 再导出包的运行时 import 检查**：`sglang.srt.distributed/__init__.py` 用 `from .parallel_state import *`，
  门 4 的静态分析对 star 再导出直接跳过，于是 4 个文件的 ImportError 漏网，是 ratchet 测试收集时才炸出来的。
  新检查真正 import 这些包再逐名核对（`/tmp/gate_star_imports.py`）。
- **communicator 方法可达性检查**：`layer_communicator.<name>` 的调用必须在新 communicator 类上可解析
  （门 6 的接收者清单不含 `layer_communicator`，hunyuan_v3 的 `should_use_reduce_scatter` 因此漏网；`/tmp/gate_comm_api.py`）。

---

## 5. 本轮新增 / 调整的 `_is_hcu` 保护

| 位置 | 内容 |
|---|---|
| `communicator/layer.py` `StageCommunicator.prepare` | HCU bailing 的 fused RMSNorm+FP8 量化替身 norm |
| `communicator/layer.py` `prepare_attn` | tuple 输入解包 |
| `fp8.py` aiter FP4 外层条件 | HCU 只为 DSV4 dequant 进入 |
| `fp8.py` FP4 int8 view | 由无条件改为 HCU 专属 |
| `mega_moe._use_amd_flydsl_mega_moe` | HCU 不进 AMD FlyDSL MegaMoEv2 |
| `dsv4/attn.py` | V4 布局的 triton 回退排除 HCU |
| `memory_pool.py` DSA FP8 写 KV | `_is_hip and not _is_hcu` + HCU lightop 分支 |
| `speculative/cache_locs.py` | 官方 OOT 平台钩子在前，HCU 专用 kernel 次之 |

---

## 6. 既有技术债（本轮未处理）

- ratchet：我方 25 处 `get_server_args().field` 直读（communicator 的 6 处随旧文件删除而消失），
  外加上游新版测试开始识别的"绑定变量"写法 2 处（`w8a8_fp8.py` 的 `server_args = get_server_args()`，代码未变）。
- `SGLANG_USE_FUSED_RMS_QUANT`：minimax_m2 调 `prepare_attn(..., skip_layernorm=True)`，main 上即 TypeError（§3.1）。
- 上游自身：`forward_batch_info.py` 里 `ForwardMode.is_dllm_extend` 重复定义；2 个 manual 测试导入已不存在的 utils 名字。
- `speculative/cache_locs.py` 里 `hcu_assign_req_to_token_pool` 未使用的 import（main 上即如此）。

## 7. 未对齐 / 主动不跟进

- 上游 AMD/NPU workflow：继续只保留 `workflow_dispatch`。
- 被删除的 6 个 serving perf 测试上的 HCU 注册（均为 disabled 占位）：若要在新的 `basic_perf/` 上做 HCU 性能覆盖，需重新定基线再注册。
- bailing 的 fused 量化（§3.1）与 hunyuan_v3 迁移（§3.6）本轮纯 TP DSV4 验证不覆盖，属静态迁移，建议后续用对应模型实测。

---

## 8. 验证（zz-nmz28 / rye_sglang_latest，2026-09-28）

装法：容器内 `bash /home/scripts/install_sglang.sh /home/proj_sglang_open/sglang-das`（sgl-kernel 0.4.7，torch 2.11）。
纯 TP8，`/models/DeepSeek-V4-Flash-0731-FP8-Channel`，`max_total_num_tokens=9654784`：

| 项 | 结果 |
|---|---|
| 贪心 sanity | `The capital of France is **Paris**.` |
| **GSM8K 100 题** | **0.98**（复跑 **0.99**） |
| 峰值 decode 吞吐 | 673.18 tok/s |
| Scheduler 异常 / VMFault | 0 / 0 |

错题分析（第一遍）：

- index 12：老熟人"柠檬树"——算对了 90/7.5=12，但把回本当成"开始赚钱"（目标 13），与 20260914 同一题。
- index 71（Kelian 的两份菜谱，目标 60）：32 并发下模型首 token 翻成 `# Your task:`，自编了一道"机器人社团"题并作答（45）。
  **batch=1 单独重放 3 次均稳定答对 60**；9/20 版本同题答对。100 题的输入 token 数与 9/20 **逐题一致**（排除模板渲染变化），
  答案 98/100 一致、完整生成 16/100 一致——与 20260914 那次 #34459 数值漂移（19/100）同量级，是本轮上游
  kernel / communicator 改动带来的贪心轨迹漂移，输出连贯，不是乱码类正确性问题。复跑 0.99 也印证了这一点。

**环境**：用户指定 zz-nmz28（zz-nmz26 的 GPU0/1 被他人服务占用，且其 `rye_sglang_latest` 已停）。
容器内无 evalscope，GSM8K 客户端从 zz-nmz26 的 `rye_sglang_open`（evalscope 1.10.0）打 `http://12.12.12.28:10015`。
共享启动脚本仍有两处过时（`--cuda-graph-max-bs` 已拆分；没有 nmz28 的网卡映射，nmz28 用 `ens14f0`），
沿用私有副本 `/home/proj_sglang_open/scripts_local/run_dpsk-v4.sh`（本轮加了 nmz28 映射）。
